"""Fit one surrogate to random benchmark data and write an interactive slice plot.

A helper study, not part of the poster pipeline. The DGP variant here has no LF priors
and no input injection: `LightweightDeepGP` with a single hidden layer of width
`--hidden_dim` and linear means. Inputs are min-max scaled to [0, 1] with the benchmark
bounds, and targets are standardised. The plot sweeps x0 and has one slider per other
input. The other inputs sit at 0.5, and training points near the slice are overlaid.
The plot is saved to `<save_dir>/SlicePlot_<bench>_<method>_N<n>.html`, the training
wall time to `<save_dir>/Meta_...json` (read by `audit_study.py`). It runs on the CPU.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import os
import argparse
import time
import json

# Import homogenized benchmarks and models
from benchmarks import get_benchmark
from optimization import LightweightDeepGP, DeepGPBoTorchModel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def main():
    """Parse arguments, fit the model, build the plot and write the outputs.

    Command-line arguments:
        --benchmark: Benchmark name (see `benchmarks.get_benchmark`).
        --method: "StandardGP" or "CustomDeepGP".
        --n_samples: Number of random training points (default 30).
        --hidden_dim: DGP hidden-layer width (default 1).
        --seed: torch seed (default 42).
        --save_dir: Output directory (default "plots/sampling_study").
        --mcmc_samples: NUTS draws (one chain, default 300).
        --mcmc_warmup: NUTS warm-up steps (default 300).

    An unknown benchmark name prints an error and returns.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['StandardGP', 'CustomDeepGP'])
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='plots/sampling_study')
    parser.add_argument('--mcmc_samples', type=int, default=300)
    parser.add_argument('--mcmc_warmup', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Device Setup
    # Use CPU to ensure float64 compatibility and stability for visualization loops
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    print(f"--- Running {args.benchmark} on {device} ---")

    # 2. Setup Benchmark
    try:
        bench = get_benchmark(args.benchmark, device=device)
    except ValueError as e:
        print(f"Error: {e}")
        return

    bounds = bench.bounds.to(device)
    dim = bench.input_dim

    # 3. Generate Training Data
    train_x_raw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(args.n_samples, dim, device=device,
                                                                   dtype=torch.double)
    train_y_raw = bench.observe(train_x_raw)

    # Scaling
    x_min, x_max = bounds[0], bounds[1]
    train_x_scaled = (train_x_raw - x_min) / (x_max - x_min)
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std()
    if y_std < 1e-6: y_std = 1.0
    train_y_scaled = (train_y_raw - y_mean) / y_std

    # 4. Model Training
    start_time = time.time()

    if args.method == 'CustomDeepGP':
        print(f"Fitting Deep GP (H={args.hidden_dim})...")
        model_obj = LightweightDeepGP(train_x_scaled, train_y_scaled.squeeze(-1), hidden_dims=[args.hidden_dim])
        model_obj.fit_numpyro(num_chains=1, num_samples=args.mcmc_samples, warmup_steps=args.mcmc_warmup,
                              rng_seed=args.seed)
        bo_model = DeepGPBoTorchModel(model_obj)
    else:
        print("Fitting Standard GP...")
        model_obj = SingleTaskGP(train_x_scaled, train_y_scaled)
        mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        fit_gpytorch_mll(mll)

    wall_time = time.time() - start_time
    print(f"Training Time: {wall_time:.2f}s")

    # =========================================================================
    # 5. VISUALIZATION LOGIC: Slice Explorer
    # =========================================================================

    # Configuration
    N_GRID_X0 = 100  # Resolution of the main X-axis
    N_SLIDER = 20  # Resolution of the sliders

    # Base Grid for X0 (Axis)
    grid_x0 = torch.linspace(0, 1, N_GRID_X0, device=device, dtype=torch.double)

    # Default fixed value for other dimensions (Midpoint 0.5)
    default_fixed_val = 0.5

    # Helper to Predict
    def get_prediction(x_test_scaled):
        """Predictive mean and std of the fitted model, in raw target units.

        Args:
            x_test_scaled: Inputs scaled to [0, 1], shape (N, D).

        Returns:
            tuple[np.ndarray, np.ndarray]: The mean and std, each of shape (N,). For the DGP
            the std includes no observation noise. For the SingleTaskGP it is the latent
            posterior std.
        """
        with torch.no_grad():
            if args.method == 'CustomDeepGP':
                post = bo_model.posterior(x_test_scaled.unsqueeze(0))
                mu = post.mean.squeeze()
                sigma = post.variance.sqrt().squeeze()
            else:
                post = model_obj.posterior(x_test_scaled)
                mu = post.mean.squeeze()
                sigma = post.variance.sqrt().squeeze()

        # Unscale
        mu_raw = mu * y_std + y_mean
        sigma_raw = sigma * y_std
        return mu_raw.numpy(), sigma_raw.numpy()

    # --- Pre-compute Frames for each Slider ---
    # We create a set of frames for varying X1, then a set for varying X2, etc.

    frames = []
    sliders = []

    # Initial Base Trace (X0 varies, all others fixed at 0.5)
    base_x = torch.full((N_GRID_X0, dim), default_fixed_val, device=device, dtype=torch.double)
    base_x[:, 0] = grid_x0

    # Evaluate Base
    mu_base, sigma_base = get_prediction(base_x)
    true_y_base = bench.observe(base_x * (x_max - x_min) + x_min).squeeze().numpy()

    # Create Base Figure
    x_plot = base_x[:, 0].numpy() * (x_max[0] - x_min[0]).item() + x_min[0].item()

    fig = go.Figure()

    # Trace 0: Ground Truth
    fig.add_trace(
        go.Scatter(x=x_plot, y=true_y_base, mode='lines', line=dict(color='black', dash='dash'), name='Ground Truth'))
    # Trace 1: Model Mean
    fig.add_trace(go.Scatter(x=x_plot, y=mu_base, mode='lines', line=dict(color='blue'), name='Model Mean'))
    # Trace 2: Upper Confidence
    fig.add_trace(go.Scatter(x=x_plot, y=mu_base + 2 * sigma_base, mode='lines', line=dict(width=0), showlegend=False,
                             hoverinfo='skip'))
    # Trace 3: Lower Confidence (Fill)
    fig.add_trace(go.Scatter(x=x_plot, y=mu_base - 2 * sigma_base, mode='lines', line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(0,0,255,0.2)', name='Confidence'))
    # Trace 4: Training Points (Placeholder, updated by frames)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red', size=8), name='Nearby Train Data'))

    # --- GENERATE SLIDERS ---

    # We iterate over every additional dimension (1 to D-1)
    for dim_idx in range(1, dim):

        steps = []
        slider_vals = np.linspace(0, 1, N_SLIDER)
        slider_step_size = 1.0 / (N_SLIDER - 1)

        for val in slider_vals:
            frame_name = f"dim{dim_idx}_val{val:.2f}"

            # 1. Construct Test Grid
            # Start with all 0.5
            x_test = torch.full((N_GRID_X0, dim), default_fixed_val, device=device, dtype=torch.double)
            # Vary X0
            x_test[:, 0] = grid_x0
            # Set specific slider value for current dimension
            x_test[:, dim_idx] = float(val)

            # 2. Predict & Observe
            mu, sigma = get_prediction(x_test)
            true_y = bench.observe(x_test * (x_max - x_min) + x_min).squeeze().numpy()

            # 3. Filter Training Points
            # Logic: Show points where x[dim_idx] is close to val, AND others are close to default
            # We defined "tolerance" as +- one slider step
            tol = slider_step_size

            mask = torch.abs(train_x_scaled[:, dim_idx] - val) < tol

            # Check other dimensions are near 0.5 (to ensure we are looking at the relevant slice)
            # If we don't check this, we show points projected from far away, which is confusing.
            for other_d in range(1, dim):
                if other_d != dim_idx:
                    mask &= torch.abs(
                        train_x_scaled[:, other_d] - default_fixed_val) < 0.25  # Looser tolerance for off-axis

            visible_train_x = train_x_raw[mask, 0].cpu().numpy()
            visible_train_y = train_y_raw[mask].cpu().numpy().flatten()

            # 4. Create Frame
            frames.append(go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(y=true_y),  # Update Trace 0
                    go.Scatter(y=mu),  # Update Trace 1
                    go.Scatter(y=mu + 2 * sigma),  # Update Trace 2
                    go.Scatter(y=mu - 2 * sigma),  # Update Trace 3
                    go.Scatter(x=visible_train_x, y=visible_train_y)  # Update Trace 4
                ]
            ))

            # 5. Add Slider Step
            real_val = val * (x_max[dim_idx] - x_min[dim_idx]).item() + x_min[dim_idx].item()
            steps.append(dict(
                method="animate",
                args=[[frame_name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=f"{real_val:.2f}"
            ))

        # Add the Slider control for this dimension
        sliders.append(dict(
            active=int(N_SLIDER / 2),
            currentvalue={"prefix": f"X{dim_idx} Value: "},
            pad={"t": 50},
            steps=steps,
            y=-0.1 * (dim_idx)  # Stack sliders vertically
        ))

    # Add Frames to Figure
    fig.frames = frames

    # Update Layout
    fig.update_layout(
        title=f"{args.benchmark} Slice Explorer | {args.method}",
        xaxis_title=f"X0 (Input Dimension 0)",
        yaxis_title="Target Y",
        sliders=sliders,
        height=600 + (dim * 50),  # Adjust height for multiple sliders
        margin=dict(b=100 + (dim * 30))
    )

    # Save
    fname = f"SlicePlot_{args.benchmark}_{args.method}_N{args.n_samples}.html"
    fig.write_html(os.path.join(args.save_dir, fname))
    print(f"Saved Interactive Slice Plot to {os.path.join(args.save_dir, fname)}")

    # Meta
    meta_fname = f"Meta_{args.benchmark}_{args.method}_N{args.n_samples}_H{args.hidden_dim}_Seed{args.seed}.json"
    metadata = {
        "wall_time": wall_time,
        "benchmark": args.benchmark,
        "method": args.method,
        "n_samples": args.n_samples,
        "hidden_dim": args.hidden_dim
    }
    with open(os.path.join(args.save_dir, meta_fname), 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()