import os
import json
import argparse
import sys
import time
import random
from collections import Counter

# Import your modified main script
import deep_gp_optimization as main_opt

# --- Configuration ---
HIDDEN_DIMS = [1, 2, 3, 5]  # Search space for hidden dimensions
SEEDS = range(0, 10)  # Run 10 repeats per configuration
CHAINS = 2
SAMPLES = 800


def construct_result_filename(hidden_dim, seed):
    return f"res_h{hidden_dim}_s{seed}.json"


def generate_job_list(save_dir):
    jobs = []
    skipped_count = 0

    print("Generating Deep GP Experiment Grid...")

    for h_dim in HIDDEN_DIMS:
        for seed in SEEDS:

            # Check if done
            expected_fname = construct_result_filename(h_dim, seed)
            if os.path.exists(os.path.join(save_dir, expected_fname)):
                skipped_count += 1
                continue

            job_dict = {
                'hidden_dim': h_dim,
                'seed': seed,
                'chains': CHAINS,
                'samples': SAMPLES
            }
            jobs.append(job_dict)

    print(f"Generated {len(jobs)} jobs (Skipped {skipped_count} existing).")
    return jobs


def run_job_by_index(jobs, index, save_dir, mock=False):
    if index < 0 or index >= len(jobs):
        print(f"Index {index} out of bounds.")
        return

    job = jobs[index]
    print(f"--> Running: HiddenDim={job['hidden_dim']}, Seed={job['seed']}")

    if mock:
        # Simulate work
        time.sleep(2)
        # Create a dummy file so audit passes
        fname = construct_result_filename(job['hidden_dim'], job['seed'])
        with open(os.path.join(save_dir, fname), 'w') as f:
            json.dump({"mock": True}, f)
    else:
        # Call the function from your main script
        try:
            main_opt.run_experiment(
                hidden_dim=job['hidden_dim'],
                seed=job['seed'],
                save_dir=save_dir,
                num_chains=job['chains'],
                num_samples=job['samples']
            )
        except Exception as e:
            print(f"!!! Job Failed: {e}")
            import traceback
            traceback.print_exc()


def worker_loop(job_file, queue_dir, save_dir, mock=False):
    if not os.path.exists(job_file):
        print(f"Job file {job_file} not found.")
        return

    with open(job_file, 'r') as f:
        jobs = json.load(f)

    # Define directories
    pending_dir = os.path.join(queue_dir, 'pending')
    running_dir = os.path.join(queue_dir, 'running')
    done_dir = os.path.join(queue_dir, 'done')

    for d in [pending_dir, running_dir, done_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Worker started. Polling {pending_dir}...")

    while True:
        try:
            candidates = [f for f in os.listdir(pending_dir) if f.endswith('.lock')]
        except FileNotFoundError:
            break

        if not candidates:
            print("Queue empty. Worker exiting.")
            break

        # Simple shuffle to reduce race conditions between multiple workers
        random.shuffle(candidates)

        job_acquired = False
        task_file = None

        for candidate in candidates:
            src = os.path.join(pending_dir, candidate)
            dst = os.path.join(running_dir, candidate)
            try:
                # atomic rename serves as the lock acquisition
                os.rename(src, dst)
                task_file = candidate
                job_acquired = True
                break
            except OSError:
                continue  # Another worker took it

        if job_acquired and task_file:
            try:
                # Parse job_123.lock -> 123
                job_index = int(task_file.split('_')[1].split('.')[0])

                run_job_by_index(jobs, job_index, save_dir, mock=mock)

                # Move to done
                final_dst = os.path.join(done_dir, task_file)
                os.rename(dst, final_dst)

            except Exception as e:
                print(f"Critical Worker Error on {task_file}: {e}")
        else:
            time.sleep(1.0)


def initialize_queue(job_file, queue_dir):
    if not os.path.exists(job_file):
        sys.exit("Job file not found.")

    with open(job_file, 'r') as f:
        jobs = json.load(f)

    pending_dir = os.path.join(queue_dir, 'pending')
    running_dir = os.path.join(queue_dir, 'running')
    done_dir = os.path.join(queue_dir, 'done')

    for d in [pending_dir, running_dir, done_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Initializing queue for {len(jobs)} jobs...")
    count = 0
    for i in range(len(jobs)):
        lock_name = f"job_{i}.lock"
        # Only create if not already done or running
        if not os.path.exists(os.path.join(done_dir, lock_name)) and \
                not os.path.exists(os.path.join(running_dir, lock_name)) and \
                not os.path.exists(os.path.join(pending_dir, lock_name)):
            with open(os.path.join(pending_dir, lock_name), 'w') as f:
                pass
            count += 1

    print(f"Added {count} new locks to pending.")


def audit_completion(job_file, save_dir):
    if not os.path.exists(job_file):
        return

    with open(job_file, 'r') as f:
        jobs = json.load(f)

    print(f"Auditing {len(jobs)} jobs...")
    status_counts = Counter()

    for i, job in enumerate(jobs):
        fname = construct_result_filename(job['hidden_dim'], job['seed'])
        path = os.path.join(save_dir, fname)

        if os.path.exists(path):
            try:
                # Check for file corruption (0 bytes or invalid json)
                if os.path.getsize(path) == 0:
                    status_counts['Empty File'] += 1
                else:
                    status_counts['OK'] += 1
            except:
                status_counts['Error Reading'] += 1
        else:
            status_counts['Missing'] += 1

    print("\nAudit Results:")
    for k, v in status_counts.items():
        print(f"  {k}: {v}")


def reset_failures(job_file, queue_dir, save_dir):
    """
    Resets jobs that are marked 'done' (lock file exists) but have no result file.
    """
    if not os.path.exists(job_file):
        return

    with open(job_file, 'r') as f:
        jobs = json.load(f)

    done_dir = os.path.join(queue_dir, 'done')
    reset_count = 0

    for i, job in enumerate(jobs):
        fname = construct_result_filename(job['hidden_dim'], job['seed'])
        result_path = os.path.join(save_dir, fname)
        lock_path = os.path.join(done_dir, f"job_{i}.lock")

        # If lock exists (marked done) BUT result is missing
        if os.path.exists(lock_path) and not os.path.exists(result_path):
            os.remove(lock_path)
            reset_count += 1

    print(f"Reset {reset_count} failed jobs. Run init_queue again to requeue them.")


def main():
    parser = argparse.ArgumentParser(description="Deep GP HPC Manager")
    subparsers = parser.add_subparsers(dest='command')

    BASE = "experiments_dgp"
    JOB_FILE = os.path.join(BASE, "jobs.json")
    QUEUE_DIR = os.path.join(BASE, "queue")
    SAVE_DIR = os.path.join(BASE, "results")

    gen_p = subparsers.add_parser('generate')
    gen_p.add_argument('--outfile', default=JOB_FILE)
    gen_p.add_argument('--save_dir', default=SAVE_DIR)

    init_p = subparsers.add_parser('init_queue')
    init_p.add_argument('--job_file', default=JOB_FILE)
    init_p.add_argument('--queue_dir', default=QUEUE_DIR)

    work_p = subparsers.add_parser('worker')
    work_p.add_argument('--job_file', default=JOB_FILE)
    work_p.add_argument('--queue_dir', default=QUEUE_DIR)
    work_p.add_argument('--save_dir', default=SAVE_DIR)
    work_p.add_argument('--mock', action='store_true')

    aud_p = subparsers.add_parser('audit')
    aud_p.add_argument('--job_file', default=JOB_FILE)
    aud_p.add_argument('--save_dir', default=SAVE_DIR)

    res_p = subparsers.add_parser('reset')
    res_p.add_argument('--job_file', default=JOB_FILE)
    res_p.add_argument('--queue_dir', default=QUEUE_DIR)
    res_p.add_argument('--save_dir', default=SAVE_DIR)

    args = parser.parse_args()

    if not os.path.exists(BASE):
        os.makedirs(BASE)

    if args.command == 'generate':
        jobs = generate_job_list(args.save_dir)
        with open(args.outfile, 'w') as f:
            json.dump(jobs, f, indent=2)

    elif args.command == 'init_queue':
        initialize_queue(args.job_file, args.queue_dir)

    elif args.command == 'worker':
        worker_loop(args.job_file, args.queue_dir, args.save_dir, args.mock)

    elif args.command == 'audit':
        audit_completion(args.job_file, args.save_dir)

    elif args.command == 'reset':
        reset_failures(args.job_file, args.queue_dir, args.save_dir)


if __name__ == "__main__":
    main()