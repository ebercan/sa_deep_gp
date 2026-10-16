import torch
import numpy as np
import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_median
import multiprocessing
import time
import warnings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Global Configuration ---
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
torch.set_default_dtype(torch.float64)


# --- Module-Level Helpers (Must be pickleable for Multiprocessing) ---

def matern52_kernel(x, z, lengthscale, amplitude):
    """Manual Matern 5/2 kernel implementation for raw speed."""
    if x.dim() == 1: x = x.unsqueeze(-1)
    if z.dim() == 1: z = z.unsqueeze(-1)

    dist_mat = torch.cdist(x, z, p=2)

    # Broadcast batch parameters (SVGD/Parallel samples)
    if lengthscale.dim() == 1:
        lengthscale = lengthscale.view(-1, 1, 1)
        amplitude = amplitude.view(-1, 1, 1)

    sqrt5 = np.sqrt(5.0)
    d_scaled = sqrt5 * dist_mat / lengthscale
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k


def _deep_gp_probabilistic_model(x, y=None):
    """The Pyro probabilistic model definition."""
    n = x.size(0)
    x_in = x.unsqueeze(-1) if x.dim() == 1 else x

    # --- Layer 1 ---
    ls1 = pyro.sample("ls1", dist.LogNormal(-1.0, 1.0))
    amp1 = pyro.sample("amp1", dist.LogNormal(-2.0, 0.5))
    cov1 = matern52_kernel(x_in, x_in, ls1, amp1) + torch.eye(n) * 1e-4

    slope1 = pyro.sample("slope1", dist.Normal(1.0, 0.5))
    bias1 = pyro.sample("bias1", dist.Normal(0.0, 0.5))

    # Handle broadcasting if slope is batched
    if slope1.dim() == 1 and slope1.size(0) > 1:
        h_mean = slope1.view(-1, 1) * x.view(1, -1) + bias1.view(-1, 1)
    else:
        h_mean = slope1 * x + bias1

    h = pyro.sample("h", dist.MultivariateNormal(h_mean, cov1))

    # --- Layer 2 ---
    ls2 = pyro.sample("ls2", dist.LogNormal(-2.0, 0.5))
    amp2 = pyro.sample("amp2", dist.LogNormal(0, 1.0))
    noise = pyro.sample("noise", dist.HalfNormal(0.3))

    h_in = h.unsqueeze(-1)
    cov2_base = matern52_kernel(h_in, h_in, ls2, amp2)

    # Handle broadcasting for noise
    noise_term = noise.view(-1, 1, 1) if noise.dim() == 1 else noise
    cov2 = cov2_base + torch.eye(n) * (noise_term + 1e-4)

    pyro.sample("obs", dist.MultivariateNormal(torch.zeros(n), cov2), obs=y)


def _worker_chain_standard(seed, train_x, train_y, queue, num_samples, warmup_steps):
    """Worker for multiprocessing chains."""
    torch.manual_seed(seed)

    nuts_kernel = NUTS(_deep_gp_probabilistic_model,
                       jit_compile=True,
                       max_tree_depth=9,
                       )

    mcmc = MCMC(nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                num_chains=1,
                )

    start = time.time()
    mcmc.run(train_x, train_y)
    duration = time.time() - start

    samples = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}
    queue.put({"samples": samples, "time": duration})


# --- The Main Class ---

class LightweightDeepGP:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.samples = None

    def fit_nuts_standard(self, num_chains=2, num_samples=1000, warmup_steps=500):
        """Runs standard parallel NUTS (waits for all chains)."""
        print(f"--- Fitting Standard NUTS ({num_chains} chains) ---")
        torch.set_num_threads(4)  # Optimize for M3 Max

        nuts_kernel = NUTS(_deep_gp_probabilistic_model,
                           jit_compile=True,
                           max_tree_depth=9,
                           )

        mcmc = MCMC(nuts_kernel,
                    num_samples=num_samples,
                    warmup_steps=warmup_steps,
                    num_chains=num_chains,
                    mp_context="spawn")

        mcmc.run(self.train_x, self.train_y)
        self.samples = mcmc.get_samples()
        print("Standard NUTS finished.")

    def fit_nuts_watchdog(self, num_chains=4, min_it_s=2.0):
        """Runs NUTS with early killing of slow chains."""
        # Calculate timeout
        # 1000 samples + 500 warmup = 1500 steps
        total_steps = 1500
        timeout = total_steps / min_it_s

        print(f"--- Fitting Watchdog NUTS ({num_chains} chains) ---")
        print(f"--- Speed Limit: {min_it_s} it/s (Timeout: {timeout:.1f}s) ---")

        torch.set_num_threads(2)  # Lower threads per chain since we run more chains

        queue = multiprocessing.Queue()
        processes = []

        for i in range(num_chains):
            seed = torch.initial_seed() + i
            p = multiprocessing.Process(
                target=_worker_chain_standard,
                args=(seed, self.train_x, self.train_y, queue, 1000, 500)
            )
            p.start()
            processes.append(p)

        results = []
        start_wait = time.time()

        while len(results) < num_chains:
            if time.time() - start_wait > timeout:
                print("\n[Watchdog] Timeout reached! Killing stragglers.")
                break

            while not queue.empty():
                res = queue.get()
                speed = total_steps / res['time']
                print(f"  > Chain finished: {speed:.2f} it/s")
                results.append(res)
            time.sleep(0.5)

        # Kill remaining
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

        if not results:
            raise RuntimeError("All chains died. Lower speed limit.")

        # Merge samples
        print(f"--- Merging {len(results)}/{num_chains} successful chains ---")
        final_samples = {}
        for k in results[0]['samples'].keys():
            final_samples[k] = torch.cat([r['samples'][k] for r in results], dim=0)

        self.samples = final_samples

    def predict(self, test_x):
        """Vectorized prediction using stored samples."""
        if self.samples is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        S = self.samples['ls1'].shape[0]
        N = self.train_x.shape[0]
        M = test_x.shape[0]
        device = self.samples['ls1'].device

        # Prepare Data
        x_tr = self.train_x.to(device).view(1, N, 1).expand(S, N, 1)
        x_ts = test_x.to(device).view(1, M, 1).expand(S, M, 1)
        y_tr = self.train_y.to(device).view(1, N).expand(S, N)

        # Prepare Params
        ls1 = self.samples['ls1'].view(S, 1, 1)
        amp1 = self.samples['amp1'].view(S, 1, 1)
        ls2 = self.samples['ls2'].view(S, 1, 1)
        amp2 = self.samples['amp2'].view(S, 1, 1)
        noise = self.samples['noise'].view(S, 1, 1)
        slope1 = self.samples['slope1'].view(S, 1, 1)
        bias1 = self.samples['bias1'].view(S, 1, 1)
        h_samples = self.samples['h'].unsqueeze(-1)  # (S, N, 1)

        # Layer 1 (Reconstruction)
        K_rr = matern52_kernel(x_tr, x_tr, ls1, amp1) + torch.eye(N, device=device) * 1e-4
        L_rr = torch.linalg.cholesky(K_rr)

        h_prior_mean = slope1 * x_tr + bias1
        residuals = h_samples - h_prior_mean
        weights_1 = torch.cholesky_solve(residuals, L_rr)

        K_tr = matern52_kernel(x_ts, x_tr, ls1, amp1)
        K_tt = matern52_kernel(x_ts, x_ts, ls1, amp1) + torch.eye(M, device=device) * 1e-4

        h_test_mean = (slope1 * x_ts + bias1) + K_tr @ weights_1
        v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
        h_test_cov = K_tt - v.transpose(-1, -2) @ v

        h_test_sample = dist.MultivariateNormal(h_test_mean.squeeze(-1), h_test_cov).sample().unsqueeze(-1)

        # Layer 2 (Prediction)
        K2_tr = matern52_kernel(h_test_sample, h_samples, ls2, amp2)
        K2_rr = matern52_kernel(h_samples, h_samples, ls2, amp2) + \
                torch.eye(N, device=device) * (noise + 1e-4)

        L2_rr = torch.linalg.cholesky(K2_rr)
        weights_2 = torch.cholesky_solve(y_tr.unsqueeze(-1), L2_rr)

        y_pred = K2_tr @ weights_2

        preds = y_pred.squeeze(-1).detach().cpu().numpy()
        return preds.mean(axis=0), preds.std(axis=0)


# --- Standard GP Helper ---

def train_standard_gp(train_x, train_y, test_x):
    """Wrapper for BoTorch SingleTaskGP."""
    model = SingleTaskGP(train_x.unsqueeze(-1), train_y.unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    with torch.no_grad():
        preds = model(test_x.unsqueeze(-1))

    return preds.mean.squeeze().numpy(), preds.variance.sqrt().squeeze().numpy()


# --- Main Execution ---

def main():
    torch.manual_seed(42)

    # 1. Data Generation
    def ackley_1d(x):
        a, b, c = 20, 0.2, 2 * np.pi
        return -a * torch.exp(-b * torch.abs(x)) - torch.exp(torch.cos(c * x)) + a + np.e

    train_x_raw = torch.linspace(-5, 5, 50)
    train_y_raw = ackley_1d(train_x_raw) + torch.randn(50) * 0.1
    test_x_raw = torch.linspace(-6, 6, 200)

    # Scaling
    x_min, x_range = train_x_raw.min(), train_x_raw.max() - train_x_raw.min()
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std()

    train_x = (train_x_raw - x_min) / x_range
    test_x = (test_x_raw - x_min) / x_range
    train_y = (train_y_raw - y_mean) / y_std

    # 2. Model 1: Deep GP (Standard NUTS)
    dgp_std = LightweightDeepGP(train_x, train_y)
    start = time.time()
    dgp_std.fit_nuts_standard(num_chains=2, num_samples=1000, warmup_steps=500)
    print(f"Standard NUTS Time: {time.time() - start:.2f}s")
    mu_dgp1, sigma_dgp1 = dgp_std.predict(test_x)

    # 3. Model 2: Deep GP (Watchdog NUTS)
    dgp_wd = LightweightDeepGP(train_x, train_y)
    start = time.time()
    dgp_wd.fit_nuts_watchdog(num_chains=4, min_it_s=2.0)
    print(f"Watchdog NUTS Time: {time.time() - start:.2f}s")
    mu_dgp2, sigma_dgp2 = dgp_wd.predict(test_x)

    # 4. Model 3: Standard GP
    start = time.time()
    mu_sgp, sigma_sgp = train_standard_gp(train_x, train_y, test_x)
    print(f"Standard GP Time: {time.time() - start:.2f}s")

    # 5. Denormalize
    def denorm(mu, sigma):
        return mu * y_std.numpy() + y_mean.numpy(), sigma * y_std.numpy()

    mu_dgp1, sigma_dgp1 = denorm(mu_dgp1, sigma_dgp1)
    mu_dgp2, sigma_dgp2 = denorm(mu_dgp2, sigma_dgp2)
    mu_sgp, sigma_sgp = denorm(mu_sgp, sigma_sgp)

    # 6. Visualization
    fig = go.Figure()

    # Truth & Data
    fig.add_trace(
        go.Scatter(x=test_x_raw, y=ackley_1d(test_x_raw), name="True Ackley", line=dict(color='black', dash='dash')))
    fig.add_trace(
        go.Scatter(x=train_x_raw, y=train_y_raw, mode='markers', name="Train Data", marker=dict(color='red', size=8)))

    # Models
    colors = [('blue', 'Standard NUTS'), ('purple', 'Watchdog NUTS'), ('green', 'Standard GP')]
    results = [(mu_dgp1, sigma_dgp1), (mu_dgp2, sigma_dgp2), (mu_sgp, sigma_sgp)]

    # CONVERT TO NUMPY ONCE HERE
    x_plot = test_x_raw.numpy()

    for (mu, sigma), (color, name) in zip(results, colors):
        fig.add_trace(go.Scatter(
            # FIX: Use the numpy array (x_plot) instead of the tensor
            x=np.concatenate([x_plot, x_plot[::-1]]),
            y=np.concatenate([mu + 2 * sigma, (mu - 2 * sigma)[::-1]]),
            fill='toself',
            fillcolor=f"rgba{str(tuple(int(c * 255) for c in (0, 0, 1)) if color == 'blue' else tuple(int(c * 255) for c in (0.5, 0, 0.5)) if color == 'purple' else tuple(int(c * 255) for c in (0, 1, 0))).replace(')', ', 0.15)')}",
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(x=x_plot, y=mu, mode='lines', name=name, line=dict(color=color, width=2)))

    fig.update_layout(title="Deep GP (Std vs Watchdog) vs Standard GP", template="plotly_white")
    fig.show()


if __name__ == "__main__":
    main()