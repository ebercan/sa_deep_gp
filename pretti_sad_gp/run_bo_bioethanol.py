"""Standalone SingleTaskGP BO on the bioethanol flowsheet (helper study, not on the poster).

It runs 5 random initial points and 10 BO iterations with UCB (beta = 0.7) and writes
the inputs and IRRs to `pretti_sad_gp/bo_results_bioethanol.json`. Run it from the
repository root (it imports `pretti_sad_gp.benchmarks...`). It needs the optional
dependencies biosteam and biorefineries. Each evaluation rebuilds and simulates the
whole flowsheet.
"""

import json
import time

import torch
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from pretti_sad_gp.benchmarks.benchmark_bioethanol_fs import BioethanolBenchmark


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

benchmark = BioethanolBenchmark(mode="HF")
bounds = benchmark.bounds


n_initial = 5

n_iterations = 10
batch_size = 1

print(f"Device: {device}")
print(f"Bounds:\n{bounds}")
print(f"Starting with {n_initial} initial points, then {n_iterations} BO iterations")

lower_bounds = bounds[0]
upper_bounds = bounds[1]

train_X = torch.rand(n_initial, bounds.shape[1], device=device, dtype=dtype)
train_X = lower_bounds + (upper_bounds - lower_bounds) * train_X

print(f"\nGenerating {n_initial} initial observations...")
t0_init = time.time()
train_Y = benchmark.observe(train_X)
t_init = time.time() - t0_init
print(f"Initial observations complete. Best value: {train_Y.max().item():.6f}")
print(
    f"Time for {n_initial} initial observations: {t_init:.2f}s ({t_init/n_initial:.2f}s per evaluation)"
)

t0_bo = time.time()
for iteration in range(n_iterations):
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(d=bounds.shape[1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    best_f = train_Y.max()
    acq_function = UpperConfidenceBound(model=model, beta=0.7)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=256,
    )

    new_Y = benchmark.observe(candidate)

    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_Y], dim=0)

    print(
        f"Iteration {iteration + 1}/{n_iterations}: \n"
        f"New Y = {new_Y.item():.6f}, Best Y = {train_Y.max().item():.6f}\n"
        f"Last X= {train_X[-1,:]}"
    )
t_bo = time.time() - t0_bo
print(f"\nOptimization complete!")
print(
    f"Time for {n_iterations} BO iterations: {t_bo:.2f}s ({t_bo/n_iterations:.2f}s per iteration)"
)
print(f"Final best value: {train_Y.max().item():.6f}")
print(f"Total evaluations: {train_X.shape[0]}")

results = {
    "inputs": train_X.detach().cpu().tolist(),
    "outputs": train_Y.detach().cpu().tolist(),
}

output_file = "pretti_sad_gp/bo_results_bioethanol.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")
print(f"Inputs shape: {len(results['inputs'])} x {len(results['inputs'][0])}")
print(f"Outputs shape: {len(results['outputs'])} x {len(results['outputs'][0])}")
