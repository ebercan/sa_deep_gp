"""Plot normalised improvement of stored BO runs (as in poster Fig. 2).

Usage, from `pretti_sad_gp/`::

    python visualize_results.py --dir data/results
    python visualize_results.py --dir data/results --plot_individual

For every run, the normalised improvement at iteration i is
(best_y_i - y_init) / (y_best - y_init). Here y_init is the best value of the initial
design, and y_best is the best value any run in `--dir` reached on that benchmark
(across both methods). Iteration 0 is 0 by construction. The aggregated plot shows
mean +- 95 % CI (1.96 standard errors), whereas the poster shows mean +- 1 std. Figures
open in the browser via Plotly.
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import glob
import pandas as pd


# --- UTILS ---

def load_and_process_data(results_dir):
    """Load all `Result_*.pt` files in a directory and compute normalised improvement.

    File names must follow `Result_<benchmark>_<method>_Seed<seed>.pt`. Files that fail to
    load are reported and skipped. Sub-directories are not searched.

    Args:
        results_dir: Directory with result files.

    Returns:
        pandas.DataFrame or None: One row per (run, iteration) with columns "Benchmark",
        "Method", "Iteration", "Normalized Yield" and "Seed", or None if no `.pt` files
        were found.
    """
    print(f"Loading results from {results_dir}...")
    files = glob.glob(os.path.join(results_dir, "*.pt"))

    if not files:
        print("No result files found.")
        return None

    benchmark_global_best = {}
    raw_data = []

    print("Scanning for Global Best values...")
    for f in files:
        try:
            res = torch.load(f, map_location='cpu')
            fname = os.path.basename(f)
            parts = fname.replace('.pt', '').split('_')

            if len(parts) >= 4:
                bench = parts[1]
                method = parts[2]
                seed = parts[3].replace('Seed', '')
            else:
                continue

            if 'history' not in res or not res['history']:
                continue

            history = res['history']
            run_max = history[-1]['best_y']

            if bench not in benchmark_global_best:
                benchmark_global_best[bench] = -float('inf')
            if run_max > benchmark_global_best[bench]:
                benchmark_global_best[bench] = run_max

            raw_data.append({
                'benchmark': bench,
                'method': method,
                'seed': seed,
                'history': history
            })

        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Global Bests found: {benchmark_global_best}")

    plot_rows = []
    for entry in raw_data:
        bench = entry['benchmark']
        method = entry['method']
        seed = entry['seed']
        history = entry['history']
        global_best = benchmark_global_best[bench]

        init_entries = [h for h in history if h['iter'] == 0]
        opt_entries = [h for h in history if h['iter'] > 0]

        if not init_entries: continue

        best_init = init_entries[-1]['best_y']
        denominator = global_best - best_init
        if abs(denominator) < 1e-9: denominator = 1.0

        plot_rows.append({
            "Benchmark": bench,
            "Method": method,
            "Iteration": 0,
            "Normalized Yield": 0.0,
            "Seed": str(seed)
        })

        for step in opt_entries:
            iter_idx = step['iter']
            current_best = step['best_y']
            normalized_score = (current_best - best_init) / denominator

            plot_rows.append({
                "Benchmark": bench,
                "Method": method,
                "Iteration": iter_idx,
                "Normalized Yield": normalized_score,
                "Seed": str(seed)
            })

    return pd.DataFrame(plot_rows)


def plot_convergence_interactive(df):
    """Plot mean +- 95 % CI of normalised improvement per benchmark and method.

    Args:
        df: Output of `load_and_process_data`. None or empty prints a message.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    benchmarks = df['Benchmark'].unique()
    methods = sorted(df['Method'].unique())
    n_bench = len(benchmarks)

    # Dynamic Layout
    cols = min(n_bench, 3)
    rows = (n_bench + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=benchmarks,
        shared_yaxes=True,
        x_title="Iteration",
        y_title="Normalized Improvement"
    )

    # Consistent Colors
    colors = px.colors.qualitative.Plotly
    method_color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    print("Generating interactive aggregated plots...")

    for idx, bench in enumerate(benchmarks):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        bench_data = df[df['Benchmark'] == bench]

        for method in methods:
            m_data = bench_data[bench_data['Method'] == method]
            if m_data.empty: continue

            # Aggregation: Mean & Std/CI
            stats = m_data.groupby("Iteration")["Normalized Yield"].agg(['mean', 'std', 'count'])
            stats['sem'] = stats['std'] / np.sqrt(stats['count'])
            stats['ci_95'] = 1.96 * stats['sem']
            stats = stats.fillna(0)  # Handle single runs (std=NaN)

            x = stats.index
            y = stats['mean']
            y_upper = y + stats['ci_95']
            y_lower = y - stats['ci_95']

            c = method_color_map[method]

            # Show legend only for the first subplot to avoid duplicates
            show_legend = (idx == 0)

            # 1. CI Band (Transparent filled area)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill='toself',
                fillcolor=c,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=method
            ), row=row, col=col)

            # 2. Mean Line
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                line=dict(color=c, width=3),
                marker=dict(size=6),
                name=method,
                legendgroup=method,
                showlegend=show_legend
            ), row=row, col=col)

    fig.update_layout(
        title="Optimization Convergence (Mean ± 95% CI)",
        template="plotly_white",
        height=400 * rows,
        hovermode="x unified"
    )
    fig.show()


def plot_individual_reps_interactive(df):
    """Plot every seed's trajectory, coloured by method.

    Args:
        df: Output of `load_and_process_data`. None or empty prints a message.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    benchmarks = df['Benchmark'].unique()
    methods = sorted(df['Method'].unique())
    n_bench = len(benchmarks)

    # Dynamic Layout
    cols = min(n_bench, 2)  # Fewer cols to give space for dense lines
    rows = (n_bench + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=benchmarks,
        shared_yaxes=True,
        x_title="Iteration",
        y_title="Normalized Improvement"
    )

    # Consistent Colors
    colors = px.colors.qualitative.Plotly
    method_color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    print("Generating interactive individual rep plots...")

    for idx, bench in enumerate(benchmarks):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        bench_data = df[df['Benchmark'] == bench]

        for method in methods:
            m_data = bench_data[bench_data['Method'] == method]
            if m_data.empty: continue

            c = method_color_map[method]
            unique_seeds = sorted(m_data['Seed'].unique())

            for seed in unique_seeds:
                s_data = m_data[m_data['Seed'] == seed]

                # Legend grouping: Clicking one method in legend toggles all its seeds
                # We append seed to name for hover info, but group by method

                # Trick: Only show in legend if it's the first seed of the first plot
                # actually, user wants "labeled with their according seed".
                # To make this usable, we label them individually but Group them.

                fig.add_trace(go.Scatter(
                    x=s_data["Iteration"],
                    y=s_data["Normalized Yield"],
                    mode='lines+markers',
                    line=dict(color=c, width=1.5),
                    marker=dict(size=4),
                    name=f"{method} (S{seed})",
                    legendgroup=method,
                    hovertext=f"Seed: {seed}",
                    showlegend=True
                ), row=row, col=col)

    fig.update_layout(
        title="Individual Trajectories (All Seeds)",
        template="plotly_white",
        height=500 * rows,
        hovermode="closest"  # 'x unified' gets messy with distinct seeds
    )
    fig.show()


def main():
    """Command-line entry point.

    Command-line arguments:
        --dir: Result directory (default "data/results").
        --plot_individual: Plot individual seeds instead of mean +- CI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/results")
    parser.add_argument("--plot_individual", action="store_true",
                        help="Plot individual seed trajectories instead of mean/std")
    args = parser.parse_args()

    df = load_and_process_data(args.dir)
    if args.plot_individual:
        plot_individual_reps_interactive(df)
    else:
        plot_convergence_interactive(df)


if __name__ == "__main__":
    main()
