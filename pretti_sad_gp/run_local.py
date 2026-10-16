"""Run one job from `slurm_jobs.json` locally, through the same code path as the cluster.

Usage, from `pretti_sad_gp/`::

    python run_local.py --find Polymerization StandardGP
    python run_local.py --job_index 3 --force_cpu

If the job file is missing, it is generated first (see `hpc_manager.generate_job_list`).
"""

import argparse
import json
import os
import sys
import torch

# Import your existing manager to ensure identical execution logic
import hpc_manager


def main():
    """Parse arguments, pick a job and run it with `hpc_manager.run_job_by_index`.

    Command-line arguments:
        --job_index: Index of the job to run.
        --find: Search terms. The first job whose values contain all of them is run.
        --save_dir: Result directory (default "data/results").
        --job_file: Job list (default "slurm_jobs.json").
        --force_cpu: Hide CUDA devices from the child process.

    With neither `--job_index` nor `--find`, job 0 is run. Errors in the job are printed
    with a traceback, not raised.

    Raises:
        SystemExit: If the job file cannot be generated or no job matches `--find`.
    """
    parser = argparse.ArgumentParser(description="Run a single HPC job locally for debugging.")
    parser.add_argument("--job_index", type=int, help="Index of the job to run (0 to N)")
    parser.add_argument("--find", nargs="+", help="Search for a job (e.g. --find Polymerization CustomDeepGP)")
    parser.add_argument("--save_dir", default="data/results", help="Where to save outputs")
    parser.add_argument("--job_file", default="slurm_jobs.json", help="Path to job list")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage (useful for Mac M1/M2/M3 with Float64)")
    args = parser.parse_args()

    # 1. Environment Setup (Mimic HPC Node)
    # -------------------------------------
    # Force single thread to match SLURM constraints for debugging
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"--- Local Debug Environment ---")

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Some PyTorch versions on Mac ignore CUDA_VISIBLE_DEVICES, so we might need more hints
        # but typically your script checks cuda.is_available() which returns False if this is empty.
        print(f"Device:  Forced CPU (via --force_cpu)")
    else:
        if torch.cuda.is_available():
            print(f"Device:  GPU ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Device:  CPU")

    print(f"Threads: 1 (Simulating HPC Worker)")
    print(f"-------------------------------")

    # 2. Load Jobs
    # -------------------------------------
    if not os.path.exists(args.job_file):
        print(f"Error: {args.job_file} not found. Running 'hpc_manager.py generate' for you...")
        try:
            with open(args.job_file, 'w') as f:
                json.dump(hpc_manager.generate_job_list(args.save_dir), f, indent=2)
        except Exception as e:
            print(f"Could not generate jobs: {e}")
            sys.exit(1)

    with open(args.job_file, 'r') as f:
        jobs = json.load(f)

    target_index = -1

    # 3. Find the Job
    # -------------------------------------
    if args.job_index is not None:
        target_index = args.job_index
    elif args.find:
        # Search for a job matching all search terms
        search_terms = [str(s).lower() for s in args.find]
        print(f"Searching for job matching: {search_terms}...")

        found_jobs = []
        for i, job in enumerate(jobs):
            # Convert job values to string for easy searching
            job_str = " ".join([str(v).lower() for v in job.values()])
            if all(term in job_str for term in search_terms):
                found_jobs.append((i, job))

        if len(found_jobs) == 0:
            print("No matching job found.")
            sys.exit(1)
        elif len(found_jobs) == 1:
            target_index = found_jobs[0][0]
            print(f"Found match at index {target_index}: {found_jobs[0][1]}")
        else:
            print(f"Found {len(found_jobs)} matches. Running the first one:")
            target_index = found_jobs[0][0]
            print(f"Index {target_index}: {found_jobs[0][1]}")

    else:
        # Default: Run the first job
        print("No index specified. Defaulting to Job 0.")
        target_index = 0

    # 4. Run Execution
    # -------------------------------------
    if 0 <= target_index < len(jobs):
        job = jobs[target_index]
        print(f"\n🚀 Launching Job {target_index}")
        print(json.dumps(job, indent=2))
        print("..." * 10)

        # Create save dir if it doesn't exist (hpc_manager usually expects it)
        os.makedirs(args.save_dir, exist_ok=True)

        try:
            # We call the function directly from hpc_manager to ensure
            # we test the EXACT code path used on the cluster.
            hpc_manager.run_job_by_index(jobs, target_index, args.save_dir, mock=False)
            print("\n✅ Job completed successfully.")
        except KeyboardInterrupt:
            print("\n⚠️ Job interrupted by user.")
        except Exception as e:
            print(f"\n❌ Job failed locally.")
            # print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: Job index {target_index} out of bounds (0-{len(jobs) - 1}).")


if __name__ == "__main__":
    main()