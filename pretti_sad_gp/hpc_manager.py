"""File-based job queue for running the experiment grid on a SLURM cluster.

Sub-commands (run from `pretti_sad_gp/`):

- `generate`: write `slurm_jobs.json` with one job per (benchmark, method, seed), for
  `BENCHMARKS` x `METHODS` x `N_SEEDS` seeds, skipping results that already exist.
- `init_queue --job_file slurm_jobs.json`: create one lock file `job_<i>.lock` per job in
  `<queue_dir>/pending/`. It re-queues failed jobs and done jobs whose result is missing,
  and with `--requeue_running` also jobs left in `running/` by killed workers.
- `worker --job_file slurm_jobs.json --hardware {H100,A100,CPU}`: claim pending locks
  by atomic rename and run them with `run_experiments.py` until none are left. CPU
  workers only take StandardGP jobs, GPU workers only CustomDeepGP jobs.
- `audit --job_file slurm_jobs.json`: count jobs whose result file is missing.

Module constants:
    BENCHMARKS: Benchmarks in the grid.
    METHODS: Methods in the grid.
    N_SEEDS: Seeds 0 .. N_SEEDS - 1 per (benchmark, method).
    N_INIT: Initial design size (10, as on the poster).
    N_ITER: BO iterations per run (20, as on the poster).
"""

import os
import json
import argparse
import sys
import random
import time
import subprocess
import glob

# --- EXPERIMENT CONFIGURATION ---
BENCHMARKS = ['Polymerization', 'PolymerizationV2', 'PharmaV1', 'PharmaV2',
              # 'HydrogenReforming', 'Bioethanol'
              ]
METHODS = ['StandardGP', 'CustomDeepGP']
N_SEEDS = 5  # Increased to 5
N_INIT = 10  # Target Start
N_ITER = 20  # Matches the stored results and the ESCAPE-36 poster (10 init + 20 BO steps)


def generate_job_list(save_dir, continue_mode=False):
    """Build the list of jobs to run.

    Args:
        save_dir: Directory with existing `Result_*.pt` files.
        continue_mode: If False, list only jobs without a result file. If True, list only
            jobs that have one, flagged with "continue" so they are extended to `N_ITER`.

    Returns:
        list[dict]: Jobs with keys "benchmark", "method", "seed", "n_init", "n_iter" and
        "continue".
    """
    jobs = []
    print(f"Generating jobs: {len(BENCHMARKS)} Bench x {len(METHODS)} Methods x {N_SEEDS} Seeds...")
    print(f"Settings: Init={N_INIT}, Iter={N_ITER}")

    skipped = 0
    for bench in BENCHMARKS:
        for method in METHODS:
            for seed in range(N_SEEDS):
                expected_filename = f"Result_{bench}_{method}_Seed{seed}.pt"
                path = os.path.join(save_dir, expected_filename)

                if continue_mode:
                    if os.path.exists(path):
                        jobs.append({'benchmark': bench, 'method': method, 'seed': seed,
                                     'n_init': N_INIT, 'n_iter': N_ITER, 'continue': True})
                    else:
                        skipped += 1
                else:
                    if not os.path.exists(path):
                        jobs.append({'benchmark': bench, 'method': method, 'seed': seed,
                                     'n_init': N_INIT, 'n_iter': N_ITER, 'continue': False})
                    else:
                        skipped += 1

    print(f"Generated {len(jobs)} jobs (Skipped {skipped} existing).")
    return jobs


def run_job_by_index(jobs, index, save_dir, mock=False):
    """Run one job in a subprocess via `run_experiments.py` (cwd must be `pretti_sad_gp/`).

    Args:
        jobs: Job list as written by `generate`.
        index: Index into `jobs`. Out-of-range indices are ignored.
        save_dir: Result directory passed to `run_experiments.py`.
        mock: If True, only print the command and sleep for 1 s.

    Raises:
        subprocess.CalledProcessError: If the experiment exits with a non-zero status.
    """
    if index < 0 or index >= len(jobs): return
    job = jobs[index]

    cmd = [
        sys.executable, "run_experiments.py",
        "--benchmark", job['benchmark'],
        "--method", job['method'],
        "--seed", str(job['seed']),
        "--save_dir", save_dir,
        "--n_init", str(job['n_init']),
        "--n_iter", str(job['n_iter'])
    ]

    if job.get('continue', False):
        cmd.append("--continue_run")

    if mock:
        print(f"[MOCK] Would run: {' '.join(cmd)}")
        time.sleep(1)
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERR: Subprocess failed for job {index}")
            raise e


def get_job_type(jobs, lock_filename):
    """Look up the method of the job a lock file belongs to.

    Args:
        jobs: Job list.
        lock_filename: Lock file name of the form `job_<index>.lock`.

    Returns:
        tuple: (method, index). It is (None, -1) if the name cannot be parsed or the index
        is out of range.
    """
    try:
        idx = int(lock_filename.split('_')[1].split('.')[0])
        if 0 <= idx < len(jobs):
            return jobs[idx]['method'], idx
    except (ValueError, IndexError, KeyError):
        pass
    return None, -1


def worker_loop(job_file, queue_dir, save_dir, hardware_type, mock=False):
    """Claim and run pending jobs until none that suit this hardware are left.

    A job is claimed by renaming its lock from `pending/` to `running/`, which is atomic on
    a shared file system. On success the lock moves to `done/`. On any error it moves to
    `failed/`, so a job that always fails is not picked up forever. Returns immediately if
    `job_file` does not exist.

    Args:
        job_file: Path to the job list (JSON).
        queue_dir: Queue directory with `pending/`, `running/`, `done/` and `failed/`
            (created if missing).
        save_dir: Result directory.
        hardware_type: "CPU" takes only StandardGP jobs; any other value (e.g. "H100",
            "A100") takes only CustomDeepGP jobs.
        mock: Passed to `run_job_by_index`.
    """
    if not os.path.exists(job_file): return

    with open(job_file, 'r') as f:
        jobs = json.load(f)

    pending_dir = os.path.join(queue_dir, 'pending')
    running_dir = os.path.join(queue_dir, 'running')
    done_dir = os.path.join(queue_dir, 'done')
    failed_dir = os.path.join(queue_dir, 'failed')

    for d in [pending_dir, running_dir, done_dir, failed_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Worker started on {hardware_type}. Polling {pending_dir}...")

    while True:
        try:
            candidates = [f for f in os.listdir(pending_dir) if f.endswith('.lock')]
        except FileNotFoundError:
            break

        if not candidates:
            print("No pending jobs found. Exiting.")
            break

        # --- STRICT SEPARATION LOGIC ---
        deep_candidates = []
        std_candidates = []

        for c in candidates:
            method, _ = get_job_type(jobs, c)
            if method == 'CustomDeepGP':
                deep_candidates.append(c)
            elif method == 'StandardGP':
                std_candidates.append(c)

        final_list = []

        if hardware_type == 'CPU':
            # CPU Workers ONLY take StandardGP
            random.shuffle(std_candidates)
            final_list = std_candidates
        else:
            # GPU Workers (H100/A100) ONLY take CustomDeepGP
            # (StandardGP is too fast to waste a GPU slot on)
            random.shuffle(deep_candidates)
            final_list = deep_candidates

        if not final_list:
            print(f"No suitable jobs for {hardware_type} available. Exiting.")
            break

        task_file = None
        for candidate in final_list:
            src = os.path.join(pending_dir, candidate)
            dst = os.path.join(running_dir, candidate)
            try:
                os.rename(src, dst)
                task_file = candidate
                break
            except OSError:
                continue

        if task_file:
            try:
                job_idx = int(task_file.split('_')[1].split('.')[0])
                print(f"Processing Job {job_idx} ({jobs[job_idx]['method']} on {hardware_type})...")
                run_job_by_index(jobs, job_idx, save_dir, mock)

                final_dst = os.path.join(done_dir, task_file)
                os.rename(dst, final_dst)
            except Exception as e:
                # Park it in failed/ instead of pending/, otherwise a job that always
                # fails is picked up again forever. `init_queue` re-queues failed jobs.
                print(f"ERR: Critical error executing {task_file}: {e}")
                print(f"Moving {task_file} to failed.")
                try:
                    os.rename(dst, os.path.join(failed_dir, task_file))
                except OSError:
                    pass
        else:
            time.sleep(0.5)


def result_path(job, save_dir):
    """Path of the final result file of a job.

    Args:
        job: Job dict with "benchmark", "method" and "seed".
        save_dir: Result directory.

    Returns:
        str: `<save_dir>/Result_<benchmark>_<method>_Seed<seed>.pt`.
    """
    return os.path.join(save_dir, f"Result_{job['benchmark']}_{job['method']}_Seed{job['seed']}.pt")


def initialize_queue(job_file, queue_dir, save_dir='data/results', requeue_running=False):
    """Create or repair the lock files of the queue for a job list.

    Removes locks whose index is not in the job file. Moves `failed/` locks back to
    `pending/`, and `done/` locks back too if the job's result file does not exist (lock
    indices refer to the current job file, which `generate` may have rewritten). With
    `requeue_running`, it also moves `running/` locks back. Finally it creates pending
    locks for jobs that have none.

    Args:
        job_file: Path to the job list (JSON).
        queue_dir: Queue directory.
        save_dir: Result directory, used to check whether done jobs really finished.
        requeue_running: Also re-queue `running/` locks. Only safe when no worker is active.

    Raises:
        FileNotFoundError: If `job_file` does not exist.
    """
    with open(job_file, 'r') as f:
        jobs = json.load(f)
    os.makedirs(os.path.join(queue_dir, 'pending'), exist_ok=True)
    os.makedirs(os.path.join(queue_dir, 'running'), exist_ok=True)
    os.makedirs(os.path.join(queue_dir, 'done'), exist_ok=True)
    os.makedirs(os.path.join(queue_dir, 'failed'), exist_ok=True)

    # Locks left over from a longer, older job file: no worker can run them and
    # `submit_cascade --monitor` would wait on them forever.
    for d in ['pending', 'running', 'done', 'failed']:
        for lock in glob.glob(os.path.join(queue_dir, d, '*.lock')):
            if get_job_type(jobs, os.path.basename(lock))[1] < 0:
                os.remove(lock)

    count = 0
    for i in range(len(jobs)):
        fname = f"job_{i}.lock"
        pending_path = os.path.join(queue_dir, 'pending', fname)
        failed_path = os.path.join(queue_dir, 'failed', fname)
        done_path = os.path.join(queue_dir, 'done', fname)
        running_path = os.path.join(queue_dir, 'running', fname)
        if os.path.exists(failed_path):
            os.rename(failed_path, pending_path)
            count += 1
            continue
        # Locks are named by index into the job file. After `generate` rewrites it, an old
        # done lock can belong to a different job, so trust it only if that job's result exists.
        if os.path.exists(done_path) and not os.path.exists(result_path(jobs[i], save_dir)):
            os.rename(done_path, pending_path)
            count += 1
            continue
        # A worker killed by the SLURM time limit leaves its lock in running/ forever.
        # Only safe when no workers are active.
        if requeue_running and os.path.exists(running_path):
            os.rename(running_path, pending_path)
            count += 1
            continue
        # Check if exists in any state
        if not any(os.path.exists(os.path.join(queue_dir, d, fname)) for d in ['done', 'running', 'pending']):
            with open(os.path.join(queue_dir, 'pending', fname), 'w') as f: pass
            count += 1
    print(f"Initialized {count} pending locks.")


def main():
    """Command-line entry point; see the module docstring for the sub-commands."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    gen = subparsers.add_parser('generate')
    gen.add_argument('--save_dir', default='data/results')
    gen.add_argument('--continue_mode', action='store_true')

    init = subparsers.add_parser('init_queue')
    init.add_argument('--job_file', required=True)
    init.add_argument('--queue_dir', default='data/queue_state')
    init.add_argument('--save_dir', default='data/results')
    init.add_argument('--requeue_running', action='store_true',
                      help="Move running/ locks back to pending (only when no workers are active)")

    work = subparsers.add_parser('worker')
    work.add_argument('--job_file', required=True)
    work.add_argument('--queue_dir', default='data/queue_state')
    work.add_argument('--save_dir', default='data/results')
    work.add_argument('--hardware', default='H100')
    work.add_argument('--mock', action='store_true')

    audit = subparsers.add_parser('audit')
    audit.add_argument('--job_file', required=True)
    audit.add_argument('--save_dir', default='data/results')

    args = parser.parse_args()
    if args.command == 'generate':
        jobs = generate_job_list(args.save_dir, continue_mode=args.continue_mode)
        with open('slurm_jobs.json', 'w') as f:
            json.dump(jobs, f, indent=2)
    elif args.command == 'init_queue':
        initialize_queue(args.job_file, args.queue_dir, args.save_dir, args.requeue_running)
    elif args.command == 'worker':
        worker_loop(args.job_file, args.queue_dir, args.save_dir, args.hardware, args.mock)
    elif args.command == 'audit':
        with open(args.job_file) as f:
            jobs = json.load(f)
        missing = 0
        for job in jobs:
            if not os.path.exists(result_path(job, args.save_dir)): missing += 1
        print(f"Audit: {missing} jobs missing out of {len(jobs)}")


if __name__ == "__main__":
    main()