from pyro.infer import SVGD, RBFSteinKernel
from pyro.optim import Adam, ClippedAdam
import torch
import numpy as np
import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import gpytorch

from pyro.infer.autoguide import init_to_uniform

import os
import multiprocessing
import time


def get_auto_compute_settings(target_total_samples=1000):
    """
    Auto-detects CPU cores and calculates optimal Chains & Threads.
    Returns a config dictionary to pass into MCMC/PyTorch.
    """
    # 1. Detect Cores (Physical preferred, logical as fallback)
    # On M3 Max, cpu_count is typically 12, 14, or 16.
    try:
        # Try to get physical cores only (avoids hyperthreading issues on Intel, safe on Mac)
        # Note: On macOS, os.cpu_count() is robust.
        total_cores = os.cpu_count()
    except Exception:
        total_cores = 4  # Fallback safe default

    # 2. Determine Number of Chains
    # Standard Bayesian practice is 4 chains to check convergence (R-hat).
    # We cap chains at total_cores to ensure we don't oversubscribe.
    num_chains = min(4, total_cores)

    # 3. Determine Threads per Chain
    # We divide the remaining cores among the chains.
    # Floor division ensures we don't exceed capacity.
    threads_per_chain = max(1, total_cores // num_chains)

    # 4. Determine Samples per Chain
    # To maintain precision, we ensure each chain is at least 500 steps long
    # (Deep GPs need time to burn in).
    # If user wants 1000 total samples, with 4 chains, that's 250 each.
    # BUT 250 is too short for Deep GPs. We force a floor of 500.
    samples_per_chain = max(500, target_total_samples // num_chains)

    print(f"--- Auto-Configured Compute ---")
    print(f"Detected Cores:      {total_cores}")
    print(f"Setting Chains:      {num_chains}")
    print(f"Threads per Chain:   {threads_per_chain}")
    print(f"Samples per Chain:   {samples_per_chain} (Total: {samples_per_chain * num_chains})")
    print(f"-------------------------------")

    return {
        "num_chains": num_chains,
        "threads_per_chain": threads_per_chain,
        "num_samples": samples_per_chain,
        "warmup_steps": samples_per_chain  # Usually equal to sampling steps is safe
    }

import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

def ackley_1d(x):
    a, b, c = 20, 0.2, 2 * np.pi
    term1 = -a * torch.exp(-b * torch.abs(x))
    term2 = -torch.exp(torch.cos(c * x))
    return term1 + term2 + a + np.e


def matern52_kernel(x, z, lengthscale, amplitude):
    if x.dim() == 1: x = x.unsqueeze(-1)
    if z.dim() == 1: z = z.unsqueeze(-1)

    dist = torch.cdist(x, z, p=2)  # Shape (N, N)

    # FIX: If params are a batch (SVGD particles), reshape to (B, 1, 1)
    if lengthscale.dim() == 1:
        lengthscale = lengthscale.view(-1, 1, 1)
        amplitude = amplitude.view(-1, 1, 1)

    sqrt5 = np.sqrt(5.0)
    d_scaled = sqrt5 * dist / lengthscale
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k


def deep_gp_model(x, y=None):
    n = x.size(0)
    # Ensure x is 2D for kernel: (N, 1)
    x_in = x.unsqueeze(-1) if x.dim() == 1 else x

    ls1 = pyro.sample("ls1", dist.LogNormal(-1.0, 1.0))
    amp1 = pyro.sample("amp1", dist.LogNormal(-2.0, 0.5))

    # Kernel handles parameter broadcasting internally
    # Returns (20, N, N) if using SVGD, or (N, N) if standard
    cov1 = matern52_kernel(x_in, x_in, ls1, amp1) + torch.eye(n) * 1e-4

    layer1_slope = pyro.sample("slope1", dist.Normal(1.0, 0.5))
    layer1_bias = pyro.sample("bias1", dist.Normal(0.0, 0.5))

    # --- FIX 1: Mean Broadcasting (You applied this previously) ---
    if layer1_slope.dim() == 1 and layer1_slope.size(0) > 1:
        h_mean = layer1_slope.view(-1, 1) * x.view(1, -1) + layer1_bias.view(-1, 1)
    else:
        h_mean = layer1_slope * x + layer1_bias

    h = pyro.sample("h", dist.MultivariateNormal(h_mean, cov1))

    ls2 = pyro.sample("ls2", dist.LogNormal(-2.0, 0.5))
    amp2 = pyro.sample("amp2", dist.LogNormal(0, 1.0))
    noise = pyro.sample("noise", dist.HalfNormal(0.3))

    # Ensure h is properly shaped for the kernel
    # If SVGD, h is (20, 50). We need (20, 50, 1).
    if h.dim() == 2:
        h_in = h.unsqueeze(-1)
    else:
        h_in = h.unsqueeze(-1)

    cov2_base = matern52_kernel(h_in, h_in, ls2, amp2)

    # --- FIX 2: Noise Broadcasting (The source of your current error) ---
    # We need to add noise to the diagonal.
    # If noise is a batch (20,), we must view it as (20, 1, 1) to add to (50, 50)
    if noise.dim() == 1:
        noise_term = noise.view(-1, 1, 1)
    else:
        noise_term = noise

    # Now (20, 50, 50) + (1, 50, 50) * (20, 1, 1) works perfectly
    cov2 = cov2_base + torch.eye(n) * (noise_term + 1e-4)

    pyro.sample("obs", dist.MultivariateNormal(torch.zeros(n), cov2), obs=y)
def predict_vectorized(samples, x_train, x_test, y_train):
    S = samples['ls1'].shape[0]
    N = x_train.shape[0]
    M = x_test.shape[0]
    device = samples['ls1'].device

    x_train = x_train.to(device).view(1, N, 1).expand(S, N, 1)
    x_test = x_test.to(device).view(1, M, 1).expand(S, M, 1)
    y_train = y_train.to(device).view(1, N).expand(S, N)

    # Reshape params
    ls1 = samples['ls1'].view(S, 1, 1)
    amp1 = samples['amp1'].view(S, 1, 1)
    ls2 = samples['ls2'].view(S, 1, 1)
    amp2 = samples['amp2'].view(S, 1, 1)
    noise = samples['noise'].view(S, 1, 1)
    slope1 = samples['slope1'].view(S, 1, 1)
    bias1 = samples['bias1'].view(S, 1, 1)

    # 1. LAYER 1 PREDICTION (Standard GP)
    # We use samples['h'] directly again (Centered approach)
    h_train_samples = samples['h'].unsqueeze(-1)  # (S, N, 1)

    K_rr = matern52_kernel(x_train, x_train, ls1, amp1) + torch.eye(N, device=device) * 1e-4
    L_rr = torch.linalg.cholesky(K_rr)

    # Residuals = h_sample - h_prior_mean
    h_prior_mean_train = slope1 * x_train + bias1
    residuals = h_train_samples - h_prior_mean_train

    weights_1 = torch.cholesky_solve(residuals, L_rr)

    K_tr = matern52_kernel(x_test, x_train, ls1, amp1)
    K_tt = matern52_kernel(x_test, x_test, ls1, amp1) + torch.eye(M, device=device) * 1e-4

    h_prior_mean_test = slope1 * x_test + bias1
    h_test_mean = h_prior_mean_test + K_tr @ weights_1

    v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
    h_test_cov = K_tt - v.transpose(-1, -2) @ v

    # Sample h_test
    h_test_sample = dist.MultivariateNormal(h_test_mean.squeeze(-1), h_test_cov).sample()
    h_test_input = h_test_sample.unsqueeze(-1)

    # 2. LAYER 2 PREDICTION
    K2_tr = matern52_kernel(h_test_input, h_train_samples, ls2, amp2)
    K2_rr = matern52_kernel(h_train_samples, h_train_samples, ls2, amp2) + \
            torch.eye(N, device=device) * (noise + 1e-4)

    L2_rr = torch.linalg.cholesky(K2_rr)
    weights_2 = torch.cholesky_solve(y_train.unsqueeze(-1), L2_rr)

    y_test_mean = K2_tr @ weights_2

    return y_test_mean.squeeze(-1).detach().cpu().numpy()

def train_and_predict_deep_gp_model(train_x, train_y, test_x):
    nuts_kernel = NUTS(deep_gp_model,
                       jit_compile=True,
                       max_tree_depth=7,
                       # full_mass=True,
                       )
    mcmc = MCMC(nuts_kernel,
                num_samples=1000,
                warmup_steps=500,
                num_chains=2,
                mp_context="spawn"
                )

    # config = get_auto_compute_settings(target_total_samples=1000)
    #
    # torch.set_num_threads(config["threads_per_chain"])
    #
    # nuts_kernel = NUTS(deep_gp_model, max_tree_depth=7, jit_compile=True, full_mass=True)
    #
    # # 3. Run MCMC with Auto-Settings
    # mcmc = MCMC(nuts_kernel,
    #             num_samples=config["num_samples"],
    #             warmup_steps=config["warmup_steps"],
    #             num_chains=config["num_chains"],
    #             mp_context="spawn")  # Required for Mac

    mcmc.run(train_x, train_y)
    samples = mcmc.get_samples()

    preds = predict_vectorized(samples, train_x, test_x, train_y)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return preds, preds.mean(axis=0), preds.std(axis=0)


def train_and_predict_svgd(train_x, train_y, test_x):
    # 1. Setup SVGD
    # Using 20 particles is a good balance for speed/diversity
    num_particles = 100
    # optimizer = Adam({"lr": 0.05})
    optimizer = ClippedAdam({"lr": 0.01, "betas": (0.9, 0.999), "clip_norm": 10.0})
    kernel = RBFSteinKernel()

    # FIX: Add max_plate_nesting=0
    svgd = SVGD(deep_gp_model,
                kernel,
                optimizer,
                num_particles=num_particles,
                max_plate_nesting=0)  # <--- REQUIRED ARGUMENT

    # 2. Run Optimization Loop (Fast!)
    print("Running SVGD...")
    for step in range(3000):
        svgd.step(train_x, train_y)
        if step % 300 == 0:
            print(f"Step {step}/3000")

    # 3. Extract Results
    # get_named_particles() returns a dict of shape (num_particles, ...)
    # which is exactly format-compatible with your existing predict_vectorized function.
    # We detach() to stop gradient tracking on the samples.
    samples = {k: v.detach().clone() for k, v in svgd.get_named_particles().items()}

    # 4. Predict
    # Your predict_vectorized function will treat 'num_particles' as 'num_mcmc_samples'
    preds = predict_vectorized(samples, train_x, test_x, train_y)

    return preds, preds.mean(axis=0), preds.std(axis=0)


def run_single_chain(seed, train_x, train_y, queue):
    """
    Runs a single MCMC chain and puts the result in a queue.
    """
    # 1. Set seed for diversity (CRITICAL!)
    torch.manual_seed(seed)

    # 2. Setup Kernel (Standard settings that work for you)
    nuts_kernel = NUTS(deep_gp_model,
                       jit_compile=True,
                       max_tree_depth=7,
                       )

    # 3. Run just 1 chain
    mcmc = MCMC(nuts_kernel,
                num_samples=1000,
                warmup_steps=500,
                num_chains=1,  # Always 1 here
                # disable_progbar=True
                )  # Reduce noise

    start_time = time.time()
    mcmc.run(train_x, train_y)
    end_time = time.time()

    # 4. Send back results
    samples = mcmc.get_samples()
    # Detach to allow pickling across processes
    samples = {k: v.detach().cpu() for k, v in samples.items()}

    queue.put({"samples": samples, "time": end_time - start_time})

def train_with_speed_limit(train_x, train_y, num_chains=4, min_it_s=2.0):
    """
    Runs chains in parallel. Kills any chain that averages slower than 'min_it_s'.
    """
    # 1. Calculate Timeout
    # We run 1000 samples + 500 warmup = 1500 total steps (approx)

    torch.set_num_threads(num_chains*2)

    total_steps = 1500
    timeout_seconds = total_steps / min_it_s

    print(f"--- Launching {num_chains} Chains ---")
    print(f"--- Speed Limit: {min_it_s} it/s (Max Duration: {timeout_seconds:.1f}s) ---")

    queue = multiprocessing.Queue()
    processes = []

    # 2. Launch Processes
    for i in range(num_chains):
        seed = torch.initial_seed() + i
        p = multiprocessing.Process(
            target=run_single_chain,  # Uses the function from the previous answer
            args=(seed, train_x, train_y, queue)
        )
        p.start()
        processes.append(p)

    # 3. Monitor
    results = []
    start_wait = time.time()

    while len(results) < num_chains:
        elapsed = time.time() - start_wait

        # KILL SWITCH
        if elapsed > timeout_seconds:
            print(f"\n[Watchdog] Time limit reached! Killing slow chains.")
            break

        while not queue.empty():
            res = queue.get()
            # Calculate actual speed of finished chain
            actual_speed = total_steps / res['time']
            print(f"[Finished] Chain completed at {actual_speed:.2f} it/s")
            results.append(res)

        time.sleep(1)

    # 4. Cleanup
    killed_count = 0
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()
            killed_count += 1

    print(f"--- Summary: {len(results)} Survived, {killed_count} Killed (Too Slow) ---")

    if len(results) == 0:
        raise RuntimeError("All chains were too slow! Lower your speed limit.")

    # 5. Merge
    final_samples = {}
    keys = results[0]['samples'].keys()
    for k in keys:
        final_samples[k] = torch.cat([r['samples'][k] for r in results], dim=0)

    return final_samples

def train_and_predict_smart(train_x, train_y, test_x):
    # Set timeout based on your "fast" run expectation (e.g. 3 mins = 180s)
    # Use 4 chains so if 1 dies, you still have 3 good ones (3000 samples).
    samples = train_with_speed_limit(train_x, train_y, num_chains=4, min_it_s=2.0)

    preds = predict_vectorized(samples, train_x, test_x, train_y)
    return preds, preds.mean(axis=0), preds.std(axis=0)

def train_hybrid_svgd_nuts(train_x, train_y, test_x):
    # ==========================================
    # PHASE 1: SVGD Pathfinder (The "Sprint")
    # ==========================================
    # We match particles to chains so every chain gets a unique good starting spot.
    num_chains = 4
    print(f"--- Phase 1: Running SVGD to find start points ({num_chains} particles) ---")

    optimizer = Adam({"lr": 0.05})
    kernel = RBFSteinKernel()

    # Run SVGD briefly (500 steps is usually plenty to find the mode)
    svgd = SVGD(deep_gp_model, kernel, optimizer, num_particles=num_chains, max_plate_nesting=0)

    for _ in range(750):
        svgd.step(train_x, train_y)

    # Extract the final positions of the particles
    # Detach them so they become just standard Tensors (not part of a comp graph)
    init_params = {k: v.detach().clone() for k, v in svgd.get_named_particles().items()}

    # ==========================================
    # PHASE 2: NUTS Refinement (The "Climb")
    # ==========================================
    print(f"--- Phase 2: Handing off to NUTS (Warmup: 75 steps) ---")

    # Optimize threads for the NUTS phase
    torch.set_num_threads(2)

    nuts_kernel = NUTS(deep_gp_model,
                       jit_compile=True,
                       max_tree_depth=7,  # Efficient cap
                       )  # Diagonal mass is safer for stability

    mcmc = MCMC(nuts_kernel,
                num_samples=1000,
                warmup_steps=75,  # <--- MASSIVE SAVINGS (Usually 500+)
                num_chains=num_chains,
                initial_params=init_params,  # Inject SVGD positions here
                mp_context="spawn")

    mcmc.run(train_x, train_y)
    samples = mcmc.get_samples()

    preds = predict_vectorized(samples, train_x, test_x, train_y)
    return preds, preds.mean(axis=0), preds.std(axis=0)

def train_and_predict_standard_gp_model(train_x, train_y, test_x):
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_x_bo = train_x.unsqueeze(-1)
    train_y_bo = train_y.unsqueeze(-1)
    test_x_bo = test_x.unsqueeze(-1)

    model = SingleTaskGP(train_x_bo, train_y_bo)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll(mll)

    model.eval()
    preds = model(test_x_bo)

    mean_pred = preds.mean.detach().squeeze().numpy()
    std_pred = preds.variance.sqrt().detach().squeeze().numpy()

    return preds, mean_pred, std_pred


def main():
    torch.manual_seed(42)

    # 1. Create Raw Data (Keep these for plotting!)
    train_x_raw = torch.linspace(-5, 5, 50, dtype=torch.float64)
    train_y_raw = ackley_1d(train_x_raw) + torch.randn(50) * 0.1
    test_x_raw = torch.linspace(-6, 6, 200, dtype=torch.float64)

    x_min = train_x_raw.min()
    x_max = train_x_raw.max()
    x_range = x_max - x_min

    train_x = (train_x_raw - x_min) / x_range
    test_x = (test_x_raw - x_min) / x_range  # Use Train min/range to scale Test!

    y_mean = train_y_raw.mean()
    y_std = train_y_raw.std()
    train_y = (train_y_raw - y_mean) / y_std

    # ==========================================
    # MODEL 1: DEEP GP
    # ==========================================
    # Note: Increase samples/warmup for real results (e.g., 500/500)
    preds_norm, mean_pred_norm, std_pred_norm = train_hybrid_svgd_nuts(train_x, train_y, test_x)

    # De-normalize predictions
    # preds_norm is a Numpy Array, so this math works fine
    preds = preds_norm * y_std.numpy() + y_mean.numpy()
    mean_pred = mean_pred_norm * y_std.numpy() + y_mean.numpy()
    std_pred = std_pred_norm * y_std.numpy()

    fig = go.Figure()

    # Plot 2-Sigma Uncertainty
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_x_raw.numpy(), test_x_raw.numpy()[::-1]]),  # Use RAW X
        y=np.concatenate([mean_pred + 2 * std_pred, (mean_pred - 2 * std_pred)[::-1]]),
        fill='toself', fillcolor='rgba(0,100,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="2σ Uncertainty (Deep GP)"
    ))

    # Plot True Function (computed on RAW X)
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=ackley_1d(test_x_raw), mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Ackley'
    ))

    # Plot Deep GP Mean
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=mean_pred, mode='lines',
        line=dict(color='blue', width=3),
        name='Deep GP Mean'
    ))

    # Plot Training Data
    fig.add_trace(go.Scatter(
        x=train_x_raw, y=train_y_raw, mode='markers',
        marker=dict(color='red', size=8, line=dict(color='black', width=1)),
        name='Train Data'
    ))

    # ==========================================
    # MODEL 2: STANDARD GP
    # ==========================================
    # preds_dist is a MultivariateNormal object (we ignore it for plotting)
    preds_dist, mean_pred_norm, std_pred_norm = train_and_predict_standard_gp_model(train_x, train_y, test_x)

    # De-normalize predictions
    # FIX: We do NOT scale 'preds_dist'. We only scale the mean/std arrays.
    mean_pred_bo = mean_pred_norm * y_std.numpy() + y_mean.numpy()
    std_pred_bo = std_pred_norm * y_std.numpy()

    # Plot Standard GP Uncertainty
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_x_raw.numpy(), test_x_raw.numpy()[::-1]]),
        y=np.concatenate([mean_pred_bo + 2 * std_pred_bo, (mean_pred_bo - 2 * std_pred_bo)[::-1]]),
        fill='toself', fillcolor='rgba(0,255,100,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="2σ Uncertainty (Standard GP)"
    ))

    # Plot Standard GP Mean
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=mean_pred_bo, mode='lines',
        line=dict(color='green', width=3),
        name='Standard GP Mean'
    ))

    fig.update_layout(
        title="Deep GP vs Standard GP (Ackley 1D)",
        xaxis_title="Input x", yaxis_title="Output f(x)",
        template="plotly_white", hovermode="x unified",
        legend=dict(groupclick="toggleitem")
    )

    fig.show()


if __name__ == "__main__":
    main()