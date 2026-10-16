import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys

# --- JAX / NumPyro Imports ---
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# --- BoTorch Imports ---
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

# --- Benchmark Import ---
# Assuming benchmarks.py is in the same directory
from benchmarks import get_benchmark

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

JITTER = 1e-6


# ==========================================
# 1. JAX/NumPyro Components (Unchanged)
# ==========================================

def squared_dist_jax(x, z):
    x2 = jnp.sum(x ** 2, axis=-1, keepdims=True)
    z2 = jnp.sum(z ** 2, axis=-1, keepdims=True)
    dist_sq = x2 + z2.T - 2 * jnp.dot(x, z.T)
    return jnp.clip(dist_sq, a_min=1e-12)


def matern52_kernel_jax(x, z, lengthscale, amplitude):
    x_scaled = x / lengthscale
    z_scaled = z / lengthscale
    d2 = squared_dist_jax(x_scaled, z_scaled)
    d = jnp.sqrt(d2)
    sqrt5 = jnp.sqrt(5.0)
    val = amplitude * (1 + sqrt5 * d + 5.0 / 3.0 * d2) * jnp.exp(-sqrt5 * d)
    return val


def _deep_gp_model_jax(x, y=None, hidden_dims=[2], standardized_map=None):
    n = x.shape[0]
    if standardized_map is None:
        standardized_map = {0: list(range(x.shape[1]))}

    current_h = None

    for layer_idx, width in enumerate(hidden_dims):
        inj_dims = standardized_map.get(layer_idx, [])

        inj_x = x[:, inj_dims] if inj_dims else jnp.empty((n, 0))
        current_input = inj_x if current_h is None else jnp.concatenate([current_h, inj_x], axis=1)

        input_dim = current_input.shape[1]

        h_list = []
        for dim_idx in range(width):
            ls_d = numpyro.sample(f"L{layer_idx}_ls_{dim_idx}", dist.LogNormal(0.0, 1.0).expand([input_dim]))
            amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(-1.0, 0.5))
            slope_d = numpyro.sample(f"L{layer_idx}_slope_{dim_idx}", dist.Normal(0.0, 0.5).expand([input_dim]))
            bias_d = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))

            k_mat = matern52_kernel_jax(current_input, current_input, ls_d, amp_d)
            cov = k_mat + jnp.eye(n) * JITTER
            mean = jnp.dot(current_input, slope_d) + bias_d
            h_d = numpyro.sample(f"h_{layer_idx}_{dim_idx}", dist.MultivariateNormal(mean, covariance_matrix=cov))
            h_list.append(h_d)

        current_h = jnp.stack(h_list, axis=1)

    # 3. Final Output Layer
    out_inj_dims = standardized_map.get("out", [])
    final_input = current_h if not out_inj_dims else jnp.concatenate([current_h, x[:, out_inj_dims]], axis=1)

    input_dim = final_input.shape[1]
    ls_out = numpyro.sample("out_ls", dist.LogNormal(-1.0, 1.0).expand([input_dim]))
    amp_out = numpyro.sample("out_amp", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("obs_noise", dist.HalfNormal(0.3))

    cov_out = matern52_kernel_jax(final_input, final_input, ls_out, amp_out)
    cov_final = cov_out + jnp.eye(n) * (noise + JITTER)
    numpyro.sample("obs", dist.MultivariateNormal(jnp.zeros(n), covariance_matrix=cov_final), obs=y)


# ==========================================
# 2. PyTorch Helper & Class (Unchanged Logic)
# ==========================================

def clamped_cdist_torch(x, z):
    x2 = x.square().sum(-1, keepdim=True)
    z2 = z.square().sum(-1, keepdim=True)
    dist_sq = (x2 + z2.transpose(-1, -2) - 2 * x @ z.transpose(-1, -2)).clamp(min=1e-12)
    return dist_sq.sqrt()


def matern52_kernel_torch(x, z, lengthscale, amplitude):
    x_scaled = x / lengthscale
    z_scaled = z / lengthscale
    dist_mat = clamped_cdist_torch(x_scaled, z_scaled)
    if amplitude.dim() == 1: amplitude = amplitude.view(-1, 1, 1)
    sqrt5 = np.sqrt(5.0)
    d_scaled = sqrt5 * dist_mat
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k


class LightweightDeepGP:
    def __init__(self, train_x, train_y, hidden_dims=[2], injection_map=None):
        self.train_x = train_x.clone()
        if self.train_x.dim() == 1: self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = train_y
        self.hidden_dims = hidden_dims
        # Default: all inputs to Layer 0
        self.injection_map = injection_map if injection_map else {0: list(range(self.train_x.shape[1]))}
        self.samples = None

    def _validate_and_standardize_map(self):
        num_inputs = self.train_x.shape[-1]
        num_hidden = len(self.hidden_dims)
        # The output layer index is equal to the number of hidden layers
        out_idx = num_hidden

        standardized = {}
        mapped_dims = set()

        for key, dims in self.injection_map.items():
            # Convert "out" or valid integer index to the internal "out" key
            if key == "out" or key == out_idx:
                target_key = "out"
            elif isinstance(key, int) and key < num_hidden:
                target_key = key
            else:
                raise ValueError(f"Invalid injection key '{key}'. For {num_hidden} hidden layers, "
                                 f"valid keys are 0-{num_hidden - 1} or '{out_idx}'/ 'out'.")

            if target_key not in standardized: standardized[target_key] = []
            standardized[target_key].extend(dims)
            mapped_dims.update(dims)

        # Check for unmapped dimensions
        all_dims = set(range(num_inputs))
        unmapped = all_dims - mapped_dims
        if unmapped:
            print(f"Warning: Dimensions {unmapped} are not mapped. Adding them to Layer 0.")
            if 0 not in standardized: standardized[0] = []
            standardized[0].extend(list(unmapped))

        return standardized

    def fit_numpyro(self, num_chains=1, num_samples=1000, warmup_steps=500):
        print(f"--- Fitting Deep GP on {jax.lib.xla_bridge.get_backend().platform} ---")
        self.std_map = self._validate_and_standardize_map()

        X_jax = jnp.array(self.train_x.cpu().numpy())
        y_jax = jnp.array(self.train_y.cpu().numpy())

        nuts_kernel = NUTS(_deep_gp_model_jax)
        mcmc = MCMC(nuts_kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True)

        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, X_jax, y_jax, hidden_dims=self.hidden_dims, standardized_map=self.injection_map)

        jax_samples = mcmc.get_samples()
        # Transfer to same device as train_x (likely CUDA if available)
        device = self.train_x.device
        self.samples = {k: torch.tensor(np.array(v), device=device) for k, v in jax_samples.items()}

        # Restructure params
        for layer_idx, width in enumerate(self.hidden_dims):
            ls_list = [self.samples[f"L{layer_idx}_ls_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_ls"] = torch.stack(ls_list, dim=1)
            amp_list = [self.samples[f"L{layer_idx}_amp_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_amp"] = torch.stack(amp_list, dim=1)
            slope_list = [self.samples[f"L{layer_idx}_slope_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_slope"] = torch.stack(slope_list, dim=1)
            bias_list = [self.samples[f"L{layer_idx}_bias_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_bias"] = torch.stack(bias_list, dim=1)

    def predict_torch(self, X):
        if self.samples is None: raise RuntimeError("Model not trained.")
        if X.dim() == 1: X = X.unsqueeze(0).unsqueeze(0)
        if X.dim() == 2: X = X.unsqueeze(0)

        B, N_test, D = X.shape
        S = self.samples['out_ls'].shape[0]
        device = self.samples['out_ls'].device
        X = X.to(device)
        train_x = self.train_x.to(device)
        N_train = train_x.shape[0]

        curr_h_train = None
        curr_h_test = None

        for layer_idx, width in enumerate(self.hidden_dims):
            inj = self.std_map.get(layer_idx, [])
            x_inj_train = train_x[:, inj].unsqueeze(0).expand(S, -1, -1)
            x_inj_test = X[:, :, inj].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)

            curr_in_train = x_inj_train if curr_h_train is None else torch.cat([curr_h_train, x_inj_train], dim=-1)
            curr_in_test = x_inj_test if curr_h_test is None else torch.cat([curr_h_test, x_inj_test], dim=-1)

            next_h_train_list = []
            next_h_test_list = []

            for d_idx in range(width):
                ls = self.samples[f'L{layer_idx}_ls'][:, d_idx, :].unsqueeze(1)
                amp = self.samples[f'L{layer_idx}_amp'][:, d_idx].view(S, 1, 1)
                slope = self.samples[f'L{layer_idx}_slope'][:, d_idx, :]
                bias = self.samples[f'L{layer_idx}_bias'][:, d_idx].view(S, 1, 1)
                h_train_samples = self.samples[f'h_{layer_idx}_{d_idx}'].unsqueeze(-1)
                next_h_train_list.append(h_train_samples)

                K_rr = matern52_kernel_torch(curr_in_train, curr_in_train, ls, amp) + torch.eye(N_train,
                                                                                                device=device) * JITTER
                L_rr = torch.linalg.cholesky(K_rr)
                train_prior_mean = (curr_in_train * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias
                weights = torch.cholesky_solve(h_train_samples - train_prior_mean, L_rr)

                # Expansion for Batch dimension B
                ls_b = ls.repeat_interleave(B, dim=0);
                amp_b = amp.repeat_interleave(B, dim=0)
                slope_b = slope.repeat_interleave(B, dim=0);
                bias_b = bias.repeat_interleave(B, dim=0)
                L_rr_b = L_rr.repeat_interleave(B, dim=0);
                weights_b = weights.repeat_interleave(B, dim=0)
                curr_in_train_b = curr_in_train.repeat_interleave(B, dim=0)

                K_tr = matern52_kernel_torch(curr_in_test, curr_in_train_b, ls_b, amp_b)
                K_tt = matern52_kernel_torch(curr_in_test, curr_in_test, ls_b, amp_b) + torch.eye(N_test,
                                                                                                  device=device) * JITTER
                test_prior_mean = (curr_in_test * slope_b.unsqueeze(1)).sum(-1, keepdim=True) + bias_b

                mu_test = test_prior_mean + K_tr @ weights_b
                v = torch.linalg.solve(L_rr_b, K_tr.transpose(-1, -2))
                cov_test = K_tt - v.transpose(-1, -2) @ v
                next_h_test_list.append(
                    torch.distributions.MultivariateNormal(mu_test.squeeze(-1), cov_test).rsample().unsqueeze(-1))

            curr_h_train = torch.cat(next_h_train_list, dim=2)
            curr_h_test = torch.cat(next_h_test_list, dim=2)

        # Final Output Layer Calculation
        inj_out = self.std_map.get("out", [])
        x_inj_train = train_x[:, inj_out].unsqueeze(0).expand(S, -1, -1)
        x_inj_test = X[:, :, inj_out].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)

        final_in_train = torch.cat([curr_h_train, x_inj_train], dim=-1) if inj_out else curr_h_train
        final_in_test = torch.cat([curr_h_test, x_inj_test], dim=-1) if inj_out else curr_h_test

        ls_out = self.samples['out_ls'].unsqueeze(1)
        amp_out = self.samples['out_amp'].view(S, 1, 1)
        noise = self.samples['obs_noise'].view(S, 1, 1)

        K_rr = matern52_kernel_torch(final_in_train, final_in_train, ls_out, amp_out) + torch.eye(N_train,
                                                                                                  device=device) * (
                           noise + JITTER)
        y_tr = self.train_y.to(device).view(1, N_train, 1).expand(S, -1, -1)
        L_rr = torch.linalg.cholesky(K_rr)
        weights = torch.cholesky_solve(y_tr, L_rr)

        K_tr = matern52_kernel_torch(final_in_test, final_in_train.repeat_interleave(B, dim=0),
                                     ls_out.repeat_interleave(B, dim=0), amp_out.repeat_interleave(B, dim=0))
        y_pred = K_tr @ weights.repeat_interleave(B, dim=0)

        return y_pred.view(S, B, N_test, 1)


    def predict(self, test_x):
        with torch.no_grad():
            x_in = torch.as_tensor(test_x).to(self.samples['out_ls'].device)
            if x_in.dim() == 1: x_in = x_in.unsqueeze(-1)
            preds = self.predict_torch(x_in)
            preds = preds.squeeze(-1).cpu().numpy()
            return preds.mean(axis=0), preds.std(axis=0)


class DeepGPBoTorchModel(Model):
    def __init__(self, dgp_model):
        super().__init__()
        self.dgp = dgp_model
        self._num_outputs = 1

    def posterior(self, X, observation_noise=False, **kwargs):
        preds_samples = self.dgp.predict_torch(X)
        mu = preds_samples.mean(dim=0)
        var = preds_samples.var(dim=0)
        cov = torch.diag_embed(var.squeeze(-1) + 1e-6)
        mvn = MultivariateNormal(mu.squeeze(-1), cov)
        return GPyTorchPosterior(mvn)

    @property
    def num_outputs(self): return self._num_outputs


# ==========================================
# 3. Main with Benchmark Logic
# ==========================================

def main():
    print("Initializing PolymerizationV2 Benchmark...")
    bench = get_benchmark("PolymerizationV2")
    input_dim = bench.input_dim  # Should be 3

    # 1. Generate Initial Data
    n_init = 50
    # Generate random inputs within [0,1] bounds (Benchmark bounds)
    train_x_raw = torch.rand(n_init, input_dim).double()
    train_x_raw = train_x_raw * (bench.bounds[1] - bench.bounds[0]).cpu() + bench.bounds[0].cpu()

    # Query Benchmark
    # Note: bench.observe expects raw inputs and returns tensor on 'device' (likely CUDA)
    # We move to CPU for the standardizing logic for simplicity, then back.
    print(f"Observing {n_init} initial points...")
    train_y_raw = bench.observe(train_x_raw).squeeze(-1).cpu()

    # 2. Standardization (Crucial for Deep GPs)
    x_mean, x_std = train_x_raw.mean(0), train_x_raw.std(0) + 1e-6
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std() + 1e-6

    train_x = (train_x_raw - x_mean) / x_std
    train_y = (train_y_raw - y_mean) / y_std

    # Move to GPU if available for Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    print(f"Data Shapes: X={train_x.shape}, Y={train_y.shape}")

    injection_map = {
        0: [0, 1, 2],
        # 1: [2]
    }
    # 3. Fit Deep GP
    # Polymerization is complex, so we use slightly wider layers [4, 4]
    dgp = LightweightDeepGP(train_x, train_y, hidden_dims=[3], injection_map=injection_map)
    dgp.fit_numpyro(num_chains=1, num_samples=1000, warmup_steps=500)

    # 4. BoTorch Optimization
    # We optimize in the standardized space
    # Bounds for standardized data roughly [-2, 2] usually covers the [0,1] raw space
    # But strictly: (0 - mean)/std to (1 - mean)/std
    std_bounds_lower = (bench.bounds[0].cpu() - x_mean) / x_std
    std_bounds_upper = (bench.bounds[1].cpu() - x_mean) / x_std
    bounds = torch.stack([std_bounds_lower, std_bounds_upper]).to(device)

    bo_model = DeepGPBoTorchModel(dgp)
    log_ei = LogExpectedImprovement(model=bo_model, best_f=train_y.min(), maximize=False)

    print("Optimizing Acquisition Function...")
    candidate_norm, _ = optimize_acqf(log_ei, bounds=bounds, q=1, num_restarts=5, raw_samples=100)

    # 5. Denormalize & Validate
    candidate_raw = candidate_norm.cpu() * x_std + x_mean
    print(f"Next Candidate (Raw Inputs): {candidate_raw.numpy()}")

    # Query benchmark with new candidate
    y_new_raw = bench.observe(candidate_raw).cpu()
    print(f"Observed Result: {y_new_raw.item()}")

    # 6. Visualization (Slicing 3D -> 2D)
    # Variable 1: Catalyst (Input 0)
    # Variable 2: Monomer (Input 1)
    # Fixed: Temperature (Input 2) = 0.5 (Raw scale)

    print("Generating Slice Plot...")
    n_grid = 30
    x_range = torch.linspace(0, 1, n_grid)
    y_range = torch.linspace(0, 1, n_grid)
    X_grid, Y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

    # Create test points (N*N, 3)
    # Dim 0: Grid, Dim 1: Grid, Dim 2: Fixed at 0.5
    flat_grid = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
    fixed_dim = torch.full((flat_grid.shape[0], 1), 0.5)
    test_x_raw = torch.cat([flat_grid, fixed_dim], dim=1)  # (900, 3)

    # Normalize Test Inputs
    test_x_norm = (test_x_raw - x_mean) / x_std

    # GP Prediction
    mu_norm, _ = dgp.predict(test_x_norm)
    mu_raw = mu_norm * y_std.item() + y_mean.item()

    # True Function (Benchmark)
    true_y = bench.observe(test_x_raw).cpu().numpy().flatten()

    # Plot
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Benchmark Truth (Slice @ Temp=0.5)", "Deep GP Mean (Slice @ Temp=0.5)"))

    # Plot Truth
    fig.add_trace(go.Contour(z=true_y.reshape(n_grid, n_grid).T,
                             x=x_range.numpy(), y=y_range.numpy(),
                             colorscale='Viridis', zmin=0, zmax=1), row=1, col=1)

    # Plot GP
    fig.add_trace(go.Contour(z=mu_raw.reshape(n_grid, n_grid).T,
                             x=x_range.numpy(), y=y_range.numpy(),
                             colorscale='Viridis', zmin=0, zmax=1), row=1, col=2)

    # Overlay Training Points that are "close" to the slice (Temp approx 0.5)
    # Filter points where Input 2 is within 0.4 and 0.6
    mask = (train_x_raw[:, 2] > 0.4) & (train_x_raw[:, 2] < 0.6)
    subset_x = train_x_raw[mask]
    if len(subset_x) > 0:
        fig.add_trace(go.Scatter(x=subset_x[:, 0], y=subset_x[:, 1],
                                 mode='markers', marker=dict(color='red', size=8, line=dict(width=1, color='white')),
                                 name="Nearby Train Data"), row=1, col=2)

    fig.update_layout(title="PolymerizationV2: Catalyst vs Monomer (Temp Fixed @ 0.5)", height=500)
    fig.show()


if __name__ == "__main__":
    main()