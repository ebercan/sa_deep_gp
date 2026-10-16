"""Summarise wall times of the sampling study written by `sample_and_visualize.py`.

It reads `Meta_*.json` from `plots/sampling_study` (relative to the working directory)
and prints the mean, std and count of the training wall time per (benchmark, method,
n_samples, hidden_dim).
"""

import os
import json
import glob
import pandas as pd


def run_audit(results_dir):
    """Aggregate and print the wall times found in `results_dir`.

    Args:
        results_dir: Directory with `Meta_*.json` files.
    """
    meta_files = glob.glob(os.path.join(results_dir, "Meta_*.json"))
    all_data = []

    for f in meta_files:
        with open(f, 'r') as j:
            all_data.append(json.load(j))

    if not all_data:
        print("No metadata found.")
        return

    df = pd.DataFrame(all_data)

    # Group by config to get average wall time
    summary = df.groupby(['benchmark', 'method', 'n_samples', 'hidden_dim']).agg({
        'wall_time': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten columns
    summary.columns = ['Bench', 'Method', 'N', 'H', 'Avg Time (s)', 'Std Dev', 'Reps']

    print("## Sampling Study Audit Results\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_audit('plots/sampling_study')