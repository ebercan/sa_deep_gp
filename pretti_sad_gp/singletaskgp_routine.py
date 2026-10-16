"""Baseline sweep: SingleTaskGP BO on all four synthetic benchmarks.

A helper study, not part of the poster pipeline. For each benchmark it first
estimates the global optimum (random search over `GRID_SIZE_GLOBAL` points, then
L-BFGS-B), then runs `N_REPS` repetitions of `N_INIT` + `N_ITER` evaluations of
LogEI BO. It plots the mean +- 95 % CI of the best value so far divided by the
optimum. On the noisy pharma benchmarks this ratio can exceed 1.
"""

import torch
import numpy as np
import warnings
import plotly.graph_objects as go
from scipy.optimize import minimize
from math import sqrt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
N_REPS = 10
N_INIT = 10
N_ITER = 90
GRID_SIZE_GLOBAL = 50000

warnings.filterwarnings("ignore", category=UserWarning, module="botorch")


# ==========================================
# 1. BENCHMARKS (shared definitions from benchmarks/)
# ==========================================

from benchmarks import get_benchmark

BENCHMARK_NAMES = {
    "Polymerization": "Disc. Polymerization V1",
    "PolymerizationV2": "Disc. Polymerization V2",
    "PharmaV1": "Pharma V1 (Deep)",
    "PharmaV2": "Pharma V2 (Seeding)",
}


def make_env(key):
    """Build a benchmark on `DEVICE` and attach a display name.

    Args:
        key: Key of `BENCHMARK_NAMES`.

    Returns:
        BenchmarkBase: The benchmark, with a `name` attribute.
    """
    env = get_benchmark(key, device=DEVICE, dtype=DTYPE)
    env.name = BENCHMARK_NAMES[key]
    return env


# ==========================================
# 2. UTILITIES
# ==========================================

def get_hybrid_global_optimum(env, n_grid=GRID_SIZE_GLOBAL):
    """Estimate the global maximum by random search followed by L-BFGS-B refinement.

    The target is evaluated in HF mode, so on the pharma benchmarks the estimate includes
    noise.

    Args:
        env: Benchmark with `bounds`, `observe` and `name`.
        n_grid: Number of uniform random points.

    Returns:
        float: The larger of the random-search maximum and the L-BFGS-B result.
    """
    print(f"   > Calculating Global Optimum for {env.name}...")

    # 1. Dense Grid Search
    bounds_np = env.bounds.cpu().numpy()
    dim = bounds_np.shape[1]

    X_grid = np.random.uniform(bounds_np[0], bounds_np[1], size=(n_grid, dim))

    chunk_size = 5000
    best_y = -np.inf
    best_x = None

    for i in range(0, n_grid, chunk_size):
        X_chunk = torch.tensor(X_grid[i:i + chunk_size], device=DEVICE, dtype=DTYPE)
        Y_chunk = env.observe(X_chunk)

        curr_max = Y_chunk.max().item()
        if curr_max > best_y:
            best_y = curr_max
            best_idx = Y_chunk.argmax().item()
            best_x = X_chunk[best_idx].cpu().numpy()

    # 2. L-BFGS-B Refinement
    def objective_func(x):
        """Negative HF yield at a single point, for `scipy.optimize.minimize`.

        Args:
            x: Point, NumPy array of shape (D,).

        Returns:
            float: -yield.
        """
        # Env expects [N, D]
        x_in = torch.tensor(x, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        y = env.observe(x_in).item()
        return -y  # Minimize negative

    res = minimize(
        objective_func,
        best_x,
        method='L-BFGS-B',
        bounds=list(zip(bounds_np[0], bounds_np[1])),
        options={'maxiter': 50, 'ftol': 1e-9}
    )

    final_max = max(best_y, -res.fun)  # L-BFGS-B can end below its start on this non-smooth, noisy target
    print(f"   > Grid Max: {best_y:.5f} | L-BFGS Refined: {final_max:.5f}")
    return final_max


def generate_initial_data(env, n):
    """Uniform random points inside the benchmark bounds.

    Args:
        env: Benchmark with `bounds` (2, D).
        n: Number of points.

    Returns:
        torch.Tensor: Shape (n, D) on `DEVICE`.
    """
    mins = env.bounds[0, :]
    maxs = env.bounds[1, :]
    rand = torch.rand(n, env.bounds.shape[1], device=DEVICE, dtype=DTYPE)
    X = rand * (maxs - mins) + mins
    return X


# ==========================================
# 3. MAIN LOOP (With Reps & CI)
# ==========================================

def run_suite():
    """Run all benchmarks and repetitions, then show the plot."""
    benchmarks = [make_env(k) for k in BENCHMARK_NAMES]

    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'purple']
    fill_colors = ['rgba(0,0,255,0.2)', 'rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(128,0,128,0.2)']

    for idx, env in enumerate(benchmarks):
        print(f"\n--- Running: {env.name} ---")

        # A. Find Global Optimum (Once per benchmark)
        true_max = get_hybrid_global_optimum(env)

        # Storage for all repetitions [N_REPS, N_TOTAL]
        all_traces = []

        for rep in range(N_REPS):
            print(f"   > Repetition {rep + 1}/{N_REPS}...", end='\r')

            # B. Initialization
            train_x = generate_initial_data(env, N_INIT)
            train_obj = env.observe(train_x)

            # Trace best found
            best_trace = np.maximum.accumulate(train_obj.cpu().numpy()).flatten().tolist()

            # C. BO Loop
            for i in range(N_ITER):
                # Fit GP
                gp = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                # Optimize Acq Function
                acq = LogExpectedImprovement(gp, best_f=train_obj.max().item())
                candidate, _ = optimize_acqf(
                    acq,
                    bounds=env.bounds,
                    q=1,
                    num_restarts=3,
                    raw_samples=64
                )

                # Observe
                new_val = env.observe(candidate)
                train_x = torch.cat([train_x, candidate])
                train_obj = torch.cat([train_obj, new_val])

                # Update Trace
                current_best = max(best_trace[-1], new_val.item())
                best_trace.append(current_best)

            # Normalize this repetition
            norm_trace = np.array(best_trace) / true_max
            all_traces.append(norm_trace)

        print(f"   > Completed {N_REPS} reps.           ")

        # D. Statistics
        all_traces = np.array(all_traces)  # [N_REPS, N_EVALS]
        mean_trace = np.mean(all_traces, axis=0)
        std_trace = np.std(all_traces, axis=0)

        # 95% Confidence Interval
        ci = 1.96 * std_trace / sqrt(N_REPS)

        eval_nums = list(range(1, N_INIT + N_ITER + 1))

        # E. Plot with Shaded Area
        # Upper Bound
        fig.add_trace(go.Scatter(
            x=eval_nums, y=mean_trace + ci,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))

        # Lower Bound (with fill)
        fig.add_trace(go.Scatter(
            x=eval_nums, y=mean_trace - ci,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=fill_colors[idx],
            showlegend=False, hoverinfo='skip'
        ))

        # Mean Line
        fig.add_trace(go.Scatter(
            x=eval_nums, y=mean_trace,
            mode='lines+markers', name=f"{env.name}",
            line=dict(color=colors[idx], width=2)
        ))

    fig.update_layout(
        title=f"Benchmark Suite: Standard GP Performance (N={N_REPS} Reps, 95% CI)",
        xaxis_title="Evaluations",
        yaxis_title="Normalized Yield (Best Found / True Max)",
        template="plotly_white",
        yaxis=dict(range=[0, 1.1]),
        height=700
    )

    fig.show()


if __name__ == "__main__":
    run_suite()
