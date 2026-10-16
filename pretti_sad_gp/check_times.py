"""Summarise the single-iteration timing runs in `data/results/trial_walltime_fit/`.

The runs are named `Result_<bench>_CustomDeepGP_Seed<seed>_NInit<n>_NIter1.pt`. For
each one the NUTS fit time of its single BO iteration is printed, together with an
estimate of a full run's time. The estimate is a lower bound: fit time grows with the
data, and acquisition time is not included. Run from `pretti_sad_gp/`.
"""

import torch
import os
import glob
import pandas as pd

from hpc_manager import N_ITER

# Directory where results are saved
RESULTS_DIR = "data/results"


def get_run_stats():
    """Load the timing runs, print a table and the time estimates.

    Files that fail to load are reported and skipped.
    """
    # Find all profiling result files
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*_NInit*_NIter1.pt"), recursive=True)

    data = []

    for f in files:
        try:
            # Load on CPU
            res = torch.load(f, map_location=torch.device('cpu'))

            # Parse filename
            fname = os.path.basename(f)
            parts = fname.replace('.pt', '').split('_')

            bench = parts[1]
            seed = parts[3].replace('Seed', '')
            n_init_str = parts[4].replace('NInit', '')

            # --- FIX ---
            # Look at the LAST entry in history to find the optimization step
            last_entry = res['history'][-1]

            if last_entry['iter'] > 0:
                train_time = last_entry['train_time']
                best_y = last_entry['best_y']
            else:
                # Should not happen if N_ITER=1 ran successfully
                train_time = 0.0
                best_y = last_entry['best_y']

            data.append({
                "Benchmark": bench,
                "Seed": seed,
                "N_Init": int(n_init_str),
                "Time (s)": round(train_time, 2),
                "Best Y": round(best_y, 4)
            })

        except Exception as e:
            print(f"Error loading {f}: {e}")

    df = pd.DataFrame(data)
    if not df.empty:
        # Sort for readability
        df = df.sort_values(by=["Benchmark", "N_Init", "Seed"])

        print("\n=== Profiling Run Stats (Corrected) ===")
        print(df.to_string(index=False))
        print("\n")

        # Stats
        avg_time = df['Time (s)'].mean()
        max_time = df['Time (s)'].max()
        print(f"Avg Time per Iter: {avg_time:.2f} s")
        print(f"Max Time per Iter: {max_time:.2f} s")
        # Lower bound: fit time grows with the data, and acquisition time is not included
        print(f"Est. Total Time ({N_ITER} Iters): {(avg_time * N_ITER / 60):.2f} min (Avg) / "
              f"{(max_time * N_ITER / 60):.2f} min (Max)")
    else:
        print("No matching result files found.")


if __name__ == "__main__":
    get_run_stats()