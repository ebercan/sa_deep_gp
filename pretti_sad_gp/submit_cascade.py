"""Submit and top up SLURM worker arrays for the `hpc_manager` queue.

Each pass counts pending and running jobs per method and the workers already queued
or running (via `squeue`). It then submits just enough array tasks: CustomDeepGP jobs
go to H100 workers first, then A100, and StandardGP jobs go to CPU workers, up to the
limits in `CAPACITY`. With `--monitor` the pass repeats every 30 s until the queue is
empty.

Run from `pretti_sad_gp/` after `hpc_manager.py generate` and `init_queue`. The
generated batch scripts activate `venv/bin/activate` in that directory.

Module constants:
    JOB_FILE, QUEUE_DIR, RESULTS_DIR, LOG_DIR: Paths, relative to `pretti_sad_gp/`.
    EMAIL: Address for SLURM failure mails.
    PARTITION_GPU, PARTITION_CPU: SLURM partitions. Adapt them to your cluster.
    CAPACITY: Maximum number of concurrent workers per hardware type.
    RESOURCE_PROFILES: CPUs, memory and thread count per hardware type.
    TIME_LIMIT: SLURM wall-time limit per worker.
"""

import os
import subprocess
import glob
import time
import datetime
import argparse
import json

# --- CONFIGURATION ---
JOB_FILE = "slurm_jobs.json"
QUEUE_DIR = "data/queue_state"
RESULTS_DIR = "data/results"
LOG_DIR = "logs"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"

# PARTITIONS (Check your cluster documentation!)
PARTITION_GPU = "gpu"
PARTITION_CPU = "long"  # Usually 'cpu', 'batch', 'normal', or 'standard'

# Hardware Limits
CAPACITY = {
    "H100": 28,
    "A100": 12,
    "CPU": 50  # Arbitrary high limit for CPU jobs since they are cheap
}

# Resource Profiles
RESOURCE_PROFILES = {
    "H100": {"cpus": 16, "mem": "64G", "threads": 4},
    "A100": {"cpus": 8, "mem": "32G", "threads": 2},
    "CPU": {"cpus": 4, "mem": "16G", "threads": 4}
}

TIME_LIMIT = "12:00:00"


class Colors:
    """ANSI escape codes for coloured terminal output."""
    HEADER = '\033[95m';
    BLUE = '\033[94m';
    GREEN = '\033[92m'
    WARNING = '\033[93m';
    FAIL = '\033[91m';
    ENDC = '\033[0m';
    BOLD = '\033[1m'


def log(msg, type="INFO"):
    """Print a timestamped, coloured log line.

    Args:
        msg: Message text.
        type: "INFO" (blue), "WARN" (yellow) or anything else (green).
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    color = Colors.BLUE if type == "INFO" else (Colors.WARNING if type == "WARN" else Colors.GREEN)
    print(f"{Colors.BOLD}[{timestamp}] {color}{type}:{Colors.ENDC} {msg}")


def get_queue_counts():
    """Count lock files per queue state.

    Returns:
        dict or None: "pending", "running", "done" and "total" counts, or None if
        `QUEUE_DIR` does not exist.
    """
    if not os.path.exists(QUEUE_DIR): return None
    pending = len(glob.glob(os.path.join(QUEUE_DIR, 'pending', '*.lock')))
    running = len(glob.glob(os.path.join(QUEUE_DIR, 'running', '*.lock')))
    done = len(glob.glob(os.path.join(QUEUE_DIR, 'done', '*.lock')))
    return {"pending": pending, "running": running, "done": done, "total": pending + running + done}


def get_pending_types(state='pending'):
    """Count the locks in one queue state per method.

    Args:
        state: Queue sub-directory, e.g. "pending" or "running".

    Returns:
        dict: {"CustomDeepGP": int, "StandardGP": int}. Both are 0 if `JOB_FILE` is
        missing.
    """
    counts = {'CustomDeepGP': 0, 'StandardGP': 0}
    if not os.path.exists(JOB_FILE): return counts
    with open(JOB_FILE, 'r') as f:
        jobs = json.load(f)
    pending = glob.glob(os.path.join(QUEUE_DIR, state, '*.lock'))
    for p in pending:
        try:
            idx = int(os.path.basename(p).split('_')[1].split('.')[0])
            method = jobs[idx]['method']
            if method in counts: counts[method] += 1
        except (ValueError, IndexError, KeyError):
            pass
    return counts


def get_active_workers(prefix):
    """Count the user's queued or running workers with a given job name.

    Args:
        prefix: SLURM job name, e.g. "Exp_H100".

    Returns:
        int or None: The number of array tasks with that name, or None if `squeue` could
        not be queried.
    """
    user = os.environ.get('USER')
    try:
        # -r lists every array task on its own line; otherwise a queued array counts as one worker
        res = subprocess.check_output(["squeue", "-r", "-u", user, "--noheader", "--format=%j"], text=True)
        return len([l for l in res.split('\n') if l.strip() == prefix])
    except (OSError, subprocess.CalledProcessError, TypeError):
        # Counting this as 0 active workers would submit a full batch on every pass
        return None


def submit_array(hardware_type, num_workers, job_prefix):
    """Write a batch script for a worker array and submit it with `sbatch`.

    Does nothing if `num_workers` <= 0. Submission failures are logged, not raised. The
    temporary script `.submit_<hardware>.sh` is deleted afterwards.

    Args:
        hardware_type: "H100", "A100" or "CPU" (key into `RESOURCE_PROFILES`).
        num_workers: Number of array tasks.
        job_prefix: Prefix of the SLURM job name (`<prefix>_<hardware>`).
    """
    if num_workers <= 0: return
    profile = RESOURCE_PROFILES[hardware_type]
    partition = PARTITION_GPU if hardware_type in ["H100", "A100"] else PARTITION_CPU
    job_name = f"{job_prefix}_{hardware_type}"

    # GPU Arg Logic
    gpu_arg = f"#SBATCH --gpus={hardware_type}:1" if hardware_type in ["H100", "A100"] else ""

    script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --array=1-{num_workers}
#SBATCH --partition={partition}
{gpu_arg}
#SBATCH --cpus-per-task={profile['cpus']}
#SBATCH --mem={profile['mem']}
#SBATCH --time={TIME_LIMIT}
#SBATCH -o {LOG_DIR}/{job_name}_%A_%a.out
#SBATCH -e {LOG_DIR}/{job_name}_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={EMAIL}

source venv/bin/activate
export OMP_NUM_THREADS={profile['threads']}
export MKL_NUM_THREADS={profile['threads']}
export TORCH_NUM_THREADS={profile['threads']}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

sleep $((RANDOM % 10))

python hpc_manager.py worker \\
    --job_file {JOB_FILE} \\
    --queue_dir {QUEUE_DIR} \\
    --save_dir {RESULTS_DIR} \\
    --hardware {hardware_type}
"""
    fname = f".submit_{hardware_type}.sh"
    with open(fname, "w") as f:
        f.write(script)

    log(f"Submitting {num_workers} {hardware_type} workers...", "INFO")
    try:
        subprocess.run(["sbatch", fname], check=True)
        log(f"Launched {hardware_type} array.", "SUCCESS")
    except (OSError, subprocess.CalledProcessError):
        log(f"Failed to submit {hardware_type}.", "FAIL")
    if os.path.exists(fname): os.remove(fname)


def main():
    """Command-line entry point.

    Command-line arguments:
        --monitor: Keep topping up workers every 30 s until the queue is empty.
        --job_prefix: Prefix of the worker job names (default "Exp").
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--job_prefix", default="Exp")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(QUEUE_DIR): return

    while True:
        counts = get_queue_counts()
        pending_types = get_pending_types()
        running_types = get_pending_types('running')

        # Every live worker (queued or running) will take one more job, so only the
        # remaining jobs not yet covered by a worker need new submissions.
        # 1. GPU Logic (DeepGP)
        active_h100 = get_active_workers(f"{args.job_prefix}_H100")
        active_a100 = get_active_workers(f"{args.job_prefix}_A100")
        active_cpu = get_active_workers(f"{args.job_prefix}_CPU")
        if None in (active_h100, active_a100, active_cpu):
            log("squeue query failed; not submitting anything this pass.", "WARN")
            if not args.monitor:
                break
            time.sleep(30)
            continue
        n_deep = max(0, pending_types['CustomDeepGP'] + running_types['CustomDeepGP'] - active_h100 - active_a100)
        n_deep = min(n_deep, pending_types['CustomDeepGP'])
        if n_deep > 0:
            free_h100 = max(0, CAPACITY['H100'] - active_h100)
            free_a100 = max(0, CAPACITY['A100'] - active_a100)

            take_h100 = min(n_deep, free_h100)
            n_deep -= take_h100
            take_a100 = min(n_deep, free_a100)

            submit_array("H100", take_h100, args.job_prefix)
            submit_array("A100", take_a100, args.job_prefix)

        # 2. CPU Logic (StandardGP)
        n_std = max(0, pending_types['StandardGP'] + running_types['StandardGP'] - active_cpu)
        n_std = min(n_std, pending_types['StandardGP'])
        if n_std > 0:
            free_cpu = max(0, CAPACITY['CPU'] - active_cpu)
            take_cpu = min(n_std, free_cpu)
            submit_array("CPU", take_cpu, args.job_prefix)

        # Monitor UI
        print("\033[H\033[J", end="")
        print(f"Queue: {counts['pending']} Pending | {counts['running']} Running | {counts['done']} Done")
        print(f"Pending Breakdown: DeepGP={pending_types['CustomDeepGP']}, StdGP={pending_types['StandardGP']}")

        if not args.monitor or (counts['pending'] == 0 and counts['running'] == 0):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()