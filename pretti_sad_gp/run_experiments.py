r"""Run one BO experiment (benchmark x method x seed) and save its result.

Usage, from `pretti_sad_gp/`::

    python run_experiments.py --benchmark Polymerization --method CustomDeepGP \
        --seed 0 --n_init 10 --n_iter 20 --save_dir data/results

The result is written to `<save_dir>/Result_<benchmark>_<method>_Seed<seed>.pt`, a
`torch.save`d dict with "history" and "prior_data" (see `optimization.run_standard_bo`).
After every BO iteration the current state is checkpointed to `<result>.pt.partial`
(written atomically via a `.tmp` file). A rerun with the same arguments resumes from that
checkpoint automatically, so a worker killed at the SLURM time limit loses at most one
iteration. The checkpoint is deleted once the final result is saved.
"""

import argparse
import os
import torch
from benchmarks import get_benchmark
from optimization import run_standard_bo, run_custom_deep_gp_bo


def main():
    """Parse the command line, resume from a checkpoint if one exists, run and save.

    Command-line arguments:
        --benchmark: "Polymerization", "PolymerizationV2", "PharmaV1" or "PharmaV2".
        --method: "StandardGP" or "CustomDeepGP".
        --seed: Run seed (default 0).
        --save_dir: Output directory (created if missing).
        --n_init: Initial design size (default 10).
        --n_iter: Total number of BO iterations (default 40; the poster used 20).
        --continue_run: Extend an existing final result to `--n_iter` iterations instead of
            skipping it.

    Raises:
        SystemExit: If the benchmark name is unknown. The non-zero exit status keeps
            `hpc_manager` from marking the job as done.
        Exception: Any error raised during the run is re-raised after printing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['StandardGP', 'CustomDeepGP'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--n_init', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=40)
    parser.add_argument('--continue_run', action='store_true', help="Resume from existing .pt file")
    args = parser.parse_args()

    save_path = os.path.join(args.save_dir, f"Result_{args.benchmark}_{args.method}_Seed{args.seed}.pt")
    # Written after every BO iteration, so a worker killed at the SLURM time limit loses at most one step
    partial_path = save_path + ".partial"
    existing_data = None

    if os.path.exists(save_path) and not args.continue_run:
        print(f"Result {save_path} already exists and --continue_run is False. Skipping.")
        return

    if os.path.exists(partial_path):
        print(f"Loading checkpoint {partial_path} to resume an interrupted run.")
        existing_data = torch.load(partial_path, map_location='cpu')
    elif args.continue_run:
        if os.path.exists(save_path):
            print(f"Loading {save_path} to continue run.")
            existing_data = torch.load(save_path, map_location='cpu')
        else:
            print(f"Warning: --continue_run set but {save_path} not found. Starting fresh.")

    def save_atomic(obj, path):
        """Save `obj` with `torch.save` atomically (write to `<path>.tmp`, then rename).

        Args:
            obj: Object to save.
            path: Destination path.
        """
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    os.makedirs(args.save_dir, exist_ok=True)

    if existing_data is not None:
        current_iters = max(h['iter'] for h in existing_data['history'])
        if current_iters >= args.n_iter:
            print(f"Run already completed ({current_iters} >= {args.n_iter}).")
            if os.path.exists(partial_path):  # killed after the last step, before the final save
                save_atomic(existing_data, save_path)
                os.remove(partial_path)
            return

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    try:
        benchmark_obj = get_benchmark(args.benchmark)
    except ValueError as e:
        # Non-zero exit so hpc_manager does not mark the job as done
        raise SystemExit(f"Error: {e}")

    checkpoint = lambda res: save_atomic(res, partial_path)

    try:
        if args.method == 'StandardGP':
            results = run_standard_bo(benchmark_obj, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
                                      existing_data=existing_data, checkpoint=checkpoint)
        elif args.method == 'CustomDeepGP':
            # Note: We pass bench_name so optimization.py knows which config to load
            results = run_custom_deep_gp_bo(
                benchmark_obj,
                n_init=args.n_init,
                n_iter=args.n_iter,
                seed=args.seed,
                bench_name=args.benchmark,
                existing_data=existing_data,
                checkpoint=checkpoint
            )

        save_atomic(results, save_path)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"Success. Saved to {save_path}")

    except Exception as e:
        print(f"ERR: Job Failed: {e}")
        raise e


if __name__ == "__main__":
    main()