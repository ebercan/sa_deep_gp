import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- JAX / NumPyro Imports ---
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# --- BoTorch Imports ---
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Benchmark Import ---
from benchmarks import get_benchmark

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

JITTER = 1e-6


# ==========================================
# 1. GP Mean Function Wrapper (Fixed active_dims)
# ==========================================

class GPLayerMean:
    def __init__(self, gpytorch_model, train_x, train_y, active_dims=None):
        self.train_x_torch = train_x.clone()
        self.train_y_torch = train_y.clone()
        self.model = gpytorch_model
        self.model.eval()
        self.active_dims = active_dims

        with torch.no_grad():
            noise = self.model.likelihood.noise
            covar_module = self.model.covar_module
            K_xx = covar_module(self.train_x_torch).to_dense()
            K_inv = torch.inverse(K_xx + torch.eye(K_xx.shape[0], device=K_xx.device) * noise)

            self.constant_mean = self.model.mean_module.constant.item()
            y_centered = (self.train_y_torch.view(-1) - self.constant_mean)

            self.alpha_torch = K_inv @ y_centered
            self.alpha_jax = jnp.array(self.alpha_torch.cpu().numpy()).reshape(-1, 1)

            self.ls_jax = jnp.array(covar_module.base_kernel.lengthscale.cpu().numpy()).reshape(1, -1)
            self.amp_jax = jnp.array(covar_module.outputscale.cpu().numpy()).item()
            self.train_x_jax = jnp.array(self.train_x_torch.cpu().numpy())

    def forward_jax(self, x):
        if self.active_dims is not None:
            x = x[..., self.active_dims]

        # Consistent Scaled Distance
        x_scaled = x / self.ls_jax
        z_scaled = self.train_x_jax / self.ls_jax

        # d2 calculation
        x2 = jnp.sum(x_scaled ** 2, axis=-1, keepdims=True)
        z2 = jnp.sum(z_scaled ** 2, axis=-1, keepdims=True)
        d2 = jnp.clip(x2 + z2.T - 2 * jnp.dot(x_scaled, z_scaled.T), a_min=1e-12)
        d = jnp.sqrt(d2)

        sqrt5 = jnp.sqrt(5.0)
        k_val = self.amp_jax * (1 + sqrt5 * d + 5.0 / 3.0 * d2) * jnp.exp(-sqrt5 * d)

        # Result must be (N, 1) for the model logic
        return (jnp.dot(k_val, self.alpha_jax) + self.constant_mean)

# ==========================================
# 2. JAX/NumPyro Deep GP (Unchanged)
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
    sqrt5 = np.sqrt(5.0)
    val = amplitude * (1 + sqrt5 * d + 5.0 / 3.0 * d2) * jnp.exp(-sqrt5 * d)
    return val


def _deep_gp_model_jax(x, y=None, hidden_dims=[2], mean_preds_jax=None):
    n = x.shape[0]
    current_x = x

    # --- Hidden Layers ---
    for layer_idx, width in enumerate(hidden_dims):
        input_dim = current_x.shape[1]
        custom_mean_fn = None
        if mean_preds_jax is not None and layer_idx < len(mean_preds_jax):
            custom_mean_fn = mean_preds_jax[layer_idx]

        h_list = []
        base_mean = None
        if custom_mean_fn is not None:
            base_mean = custom_mean_fn(current_x)  # (N, 1) or (N, width)

        for dim_idx in range(width):
            ls_d = numpyro.sample(f"L{layer_idx}_ls_{dim_idx}", dist.LogNormal(0.0, 1.0).expand([input_dim]))
            amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(-1.0, 0.5))

            k_mat = matern52_kernel_jax(current_x, current_x, ls_d, amp_d)
            cov = k_mat + jnp.eye(n) * JITTER

            if custom_mean_fn is not None:
                # FIX: Ensure base_mean is flattened to (N,) for MultivariateNormal
                if base_mean.shape[1] == width:
                    mu = base_mean[:, dim_idx]
                else:
                    mu = base_mean[:, 0]

                # Loosen bias prior from 0.1 to 0.5 to allow flexibility
                bias_correction = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))
                mean = mu + bias_correction
            else:
                slope_d = numpyro.sample(f"L{layer_idx}_slope_{dim_idx}", dist.Normal(0.0, 0.5).expand([input_dim]))
                bias_d = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))
                mean = jnp.dot(current_x, slope_d) + bias_d

            h_d = numpyro.sample(f"h_{layer_idx}_{dim_idx}", dist.MultivariateNormal(mean, covariance_matrix=cov))
            h_list.append(h_d)

        current_x = jnp.stack(h_list, axis=1)

    # --- Output Layer ---
    input_dim = current_x.shape[1]
    out_layer_idx = len(hidden_dims)
    custom_out_mean = None
    if mean_preds_jax is not None and out_layer_idx < len(mean_preds_jax):
        custom_out_mean = mean_preds_jax[out_layer_idx]

    ls_out = numpyro.sample("out_ls", dist.LogNormal(-1.0, 1.0).expand([input_dim]))
    amp_out = numpyro.sample("out_amp", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("obs_noise", dist.HalfNormal(0.3))

    cov_out = matern52_kernel_jax(current_x, current_x, ls_out, amp_out)
    cov_final = cov_out + jnp.eye(n) * (noise + JITTER)

    if custom_out_mean is not None:
        # FIX: Ensure (N,) shape for observation
        mean_final = custom_out_mean(current_x).reshape(-1)
        bias_c = numpyro.sample("out_bias", dist.Normal(0.0, 0.5))
        mean_final = mean_final + bias_c
    else:
        # FIX: Don't hardcode zero. Use a learnable constant mean at minimum.
        out_bias = numpyro.sample("out_bias", dist.Normal(0.0, 0.5))
        mean_final = jnp.full((n,), out_bias)

    numpyro.sample("obs", dist.MultivariateNormal(mean_final, covariance_matrix=cov_final), obs=y)

# ==========================================
# 3. PyTorch Helper & Class (Unchanged Logic)
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
    sqrt5 = 2.2360679775
    d_scaled = sqrt5 * dist_mat
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k


class LightweightDeepGP:
    def __init__(self, train_x, train_y, hidden_dims=[2], mean_modules=None):
        self.train_x = train_x.clone()
        if self.train_x.dim() == 1: self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = train_y
        self.hidden_dims = hidden_dims
        self.samples = None
        self.mean_modules = mean_modules if mean_modules else []

    def fit_numpyro(self, num_chains=1, num_samples=1000, warmup_steps=500):
        print(f"--- Fitting Deep GP on {jax.lib.xla_bridge.get_backend().platform} ---")
        X_jax = jnp.array(self.train_x.cpu().numpy())
        if X_jax.ndim == 1: X_jax = X_jax.reshape(-1, 1)
        y_jax = jnp.array(self.train_y.cpu().numpy())

        mean_preds_jax = []
        for m in self.mean_modules:
            if m is not None:
                mean_preds_jax.append(m.forward_jax)
            else:
                mean_preds_jax.append(None)

        nuts_kernel = NUTS(_deep_gp_model_jax)
        mcmc = MCMC(nuts_kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True)

        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, X_jax, y_jax, hidden_dims=self.hidden_dims, mean_preds_jax=mean_preds_jax)

        jax_samples = mcmc.get_samples()
        device = self.train_x.device
        self.samples = {k: torch.tensor(np.array(v), device=device) for k, v in jax_samples.items()}

        for layer_idx, width in enumerate(self.hidden_dims):
            has_custom = (layer_idx < len(self.mean_modules) and self.mean_modules[layer_idx] is not None)

            ls_list = [self.samples[f"L{layer_idx}_ls_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_ls"] = torch.stack(ls_list, dim=1)

            amp_list = [self.samples[f"L{layer_idx}_amp_{d}"] for d in range(width)]
            self.samples[f"L{layer_idx}_amp"] = torch.stack(amp_list, dim=1)

            if not has_custom:
                slope_list = [self.samples[f"L{layer_idx}_slope_{d}"] for d in range(width)]
                self.samples[f"L{layer_idx}_slope"] = torch.stack(slope_list, dim=1)
                bias_list = [self.samples[f"L{layer_idx}_bias_{d}"] for d in range(width)]
                self.samples[f"L{layer_idx}_bias"] = torch.stack(bias_list, dim=1)
            else:
                bias_list = [self.samples[f"L{layer_idx}_bias_{d}"] for d in range(width)]
                self.samples[f"L{layer_idx}_bias"] = torch.stack(bias_list, dim=1)

    def predict_torch(self, X):
        if self.samples is None: raise RuntimeError("Model not trained.")
        if X.dim() == 1: X = X.unsqueeze(-1)
        if X.dim() == 2:
            M = X.shape[0];
            X_flat = X
        else:
            M = X.shape[0] * X.shape[1];
            X_flat = X.view(M, -1)

        device = self.samples['out_ls'].device
        X_flat = X_flat.to(device)
        train_x = self.train_x.to(device)
        S = self.samples['out_ls'].shape[0]
        N = train_x.shape[0]

        curr_x_train = train_x.unsqueeze(0).expand(S, -1, -1)
        curr_x_test = X_flat.unsqueeze(0).expand(S, -1, -1)

        for layer_idx, width in enumerate(self.hidden_dims):
            has_custom = (layer_idx < len(self.mean_modules) and self.mean_modules[layer_idx] is not None)

            next_x_test_list = []
            next_x_train_list = []

            for dim_idx in range(width):
                ls = self.samples[f'L{layer_idx}_ls'][:, dim_idx, :].unsqueeze(1)
                amp = self.samples[f'L{layer_idx}_amp'][:, dim_idx].view(S, 1, 1)
                h_train_samples = self.samples[f'h_{layer_idx}_{dim_idx}'].unsqueeze(-1)
                next_x_train_list.append(h_train_samples)

                K_rr = matern52_kernel_torch(curr_x_train, curr_x_train, ls, amp) + torch.eye(N, device=device) * JITTER
                L_rr = torch.linalg.cholesky(K_rr)

                if has_custom:
                    flat_train = curr_x_train.reshape(-1, curr_x_train.shape[-1])
                    mu_train_flat = self.mean_modules[layer_idx].forward_torch(flat_train)
                    mu_train = mu_train_flat.reshape(S, N, -1)

                    flat_test = curr_x_test.reshape(-1, curr_x_test.shape[-1])
                    mu_test_flat = self.mean_modules[layer_idx].forward_torch(flat_test)
                    mu_test = mu_test_flat.reshape(S, M, -1)

                    if mu_train.shape[-1] == width:
                        mu_train = mu_train[:, :, dim_idx:dim_idx + 1]
                        mu_test = mu_test[:, :, dim_idx:dim_idx + 1]

                    bias = self.samples[f'L{layer_idx}_bias'][:, dim_idx].view(S, 1, 1)
                    train_prior_mean = mu_train + bias
                    test_prior_mean = mu_test + bias

                else:
                    slope = self.samples[f'L{layer_idx}_slope'][:, dim_idx, :]
                    bias = self.samples[f'L{layer_idx}_bias'][:, dim_idx].view(S, 1, 1)
                    train_prior_mean = (curr_x_train * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias
                    test_prior_mean = (curr_x_test * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias

                residuals = h_train_samples - train_prior_mean
                weights = torch.cholesky_solve(residuals, L_rr)

                K_tr = matern52_kernel_torch(curr_x_test, curr_x_train, ls, amp)
                K_tt = matern52_kernel_torch(curr_x_test, curr_x_test, ls, amp) + torch.eye(M, device=device) * JITTER

                mu_test_post = test_prior_mean + K_tr @ weights
                v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
                cov_test = K_tt - v.transpose(-1, -2) @ v

                rng_state = torch.get_rng_state()
                h_dist = torch.distributions.MultivariateNormal(mu_test_post.squeeze(-1), cov_test)
                h_test_sample = h_dist.rsample().unsqueeze(-1)
                torch.set_rng_state(rng_state)
                next_x_test_list.append(h_test_sample)

            curr_x_test = torch.cat(next_x_test_list, dim=2)
            curr_x_train = torch.cat(next_x_train_list, dim=2)

        out_idx = len(self.hidden_dims)
        has_out_custom = (out_idx < len(self.mean_modules) and self.mean_modules[out_idx] is not None)

        ls_out = self.samples['out_ls'].unsqueeze(1)
        amp_out = self.samples['out_amp'].view(S, 1, 1)
        noise = self.samples['obs_noise'].view(S, 1, 1)
        K_rr = matern52_kernel_torch(curr_x_train, curr_x_train, ls_out, amp_out) + torch.eye(N, device=device) * (
                    noise + JITTER)
        K_tr = matern52_kernel_torch(curr_x_test, curr_x_train, ls_out, amp_out)

        y_tr = self.train_y.to(device).view(1, N).expand(S, N).unsqueeze(-1)

        if has_out_custom:
            flat_train = curr_x_train.reshape(-1, curr_x_train.shape[-1])
            mu_train = self.mean_modules[out_idx].forward_torch(flat_train).reshape(S, N, 1)
            flat_test = curr_x_test.reshape(-1, curr_x_test.shape[-1])
            mu_test = self.mean_modules[out_idx].forward_torch(flat_test).reshape(S, M, 1)

            bias = self.samples['out_bias'].view(S, 1, 1)
            y_mean_train = mu_train + bias
            y_mean_test = mu_test + bias
            y_centered = y_tr - y_mean_train
        else:
            y_centered = y_tr
            y_mean_test = 0.0

        L_rr = torch.linalg.cholesky(K_rr)
        weights = torch.cholesky_solve(y_centered, L_rr)
        y_pred = y_mean_test + K_tr @ weights

        if X.dim() == 3: return y_pred.view(S, X.shape[0], X.shape[1], 1)
        return y_pred.view(S, X.shape[0], 1)

    def predict(self, test_x):
        with torch.no_grad():
            x_in = torch.as_tensor(test_x).to(self.samples['out_ls'].device)
            if x_in.dim() == 1: x_in = x_in.unsqueeze(-1)
            preds = self.predict_torch(x_in)
            preds = preds.squeeze(-1).cpu().numpy()
            return preds.mean(axis=0), preds.std(axis=0)


# ==========================================
# 4. Helper with Active Dims Support
# ==========================================

def train_individual_step_gp(X, y, active_dims=None):
    """
    Trains a standard BoTorch SingleTaskGP on a process step.
    Wraps it to handle active_dims (slicing) automatically.
    """
    train_x = X if torch.is_tensor(X) else torch.tensor(X)
    train_y = y if torch.is_tensor(y) else torch.tensor(y)

    if train_y.ndim == 1: train_y = train_y.unsqueeze(-1)

    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Pass active_dims to wrapper so it knows how to slice input
    return GPLayerMean(gp, train_x, train_y, active_dims=active_dims)


# ... (Previous imports and class definitions remain the same) ...

def visualize_interactive_slice(dgp_model, benchmark, train_x_raw, train_y_raw, x_mean, x_std, y_mean, y_std):
    """
    Interactive visualization with:
    - 2 Sliders (Monomer, Temperature) nicely spaced below the chart.
    - Flattened data arrays to ensure points render correctly.
    """
    print("Generating Interactive Visualization...")

    # 1. Setup Resolution
    n_grid = 100  # Resolution for X-axis (Catalyst)
    n_slider = 10  # Slider steps

    # Ranges (Raw Scale)
    x0_range = torch.linspace(0, 1, n_grid).double()
    x1_steps = torch.linspace(0, 1, n_slider).double()  # Monomer
    x2_steps = torch.linspace(0, 1, n_slider).double()  # Temperature

    device = dgp_model.samples['out_ls'].device

    # 2. Pre-Compute Predictions
    preds_mean = np.zeros((n_slider, n_slider, n_grid))
    preds_std = np.zeros((n_slider, n_slider, n_grid))
    truth_vals = np.zeros((n_slider, n_slider, n_grid))

    print(f"Pre-computing {n_slider * n_slider} slices...")

    for t_idx, val_temp in enumerate(x2_steps):
        for m_idx, val_mono in enumerate(x1_steps):
            grid = torch.zeros(n_grid, 3, dtype=torch.float64)
            grid[:, 0] = x0_range
            grid[:, 1] = val_mono
            grid[:, 2] = val_temp

            # Normalize & Predict
            grid_norm = (grid - x_mean.cpu()) / x_std.cpu()
            grid_norm = grid_norm.to(device)
            mu, sigma = dgp_model.predict(grid_norm)

            preds_mean[t_idx, m_idx, :] = mu * y_std.item() + y_mean.item()
            preds_std[t_idx, m_idx, :] = sigma * y_std.item()
            truth_vals[t_idx, m_idx, :] = benchmark.observe(grid).cpu().numpy().flatten()

    # 3. Initial State
    t_init = n_slider // 2
    m_init = n_slider // 2
    val_temp_init = x2_steps[t_init].item()
    val_mono_init = x1_steps[m_init].item()

    # --- FIX 1: FLATTEN ARRAYS ---
    # Plotly fails if y is (N, 1). We ensure (N,).
    tolerance = 0.08  # 25% tolerance

    def get_nearby_points(v_mono, v_temp):
        # Calculate distance in the 2D plane (Dim1, Dim2)
        dist = torch.sqrt((train_x_raw[:, 1] - v_mono) ** 2 + (train_x_raw[:, 2] - v_temp) ** 2)
        mask = dist < tolerance
        # Flattening is critical here
        sx = train_x_raw[mask, 0].cpu().numpy().flatten()
        sy = train_y_raw[mask].cpu().numpy().flatten()
        return sx, sy

    sx, sy = get_nearby_points(val_mono_init, val_temp_init)
    print(f"Initial slice has {len(sx)} nearby training points.")

    # 4. Create Traces
    fig = go.Figure()

    # [0] True Function
    fig.add_trace(go.Scatter(
        x=x0_range.numpy(), y=truth_vals[t_init, m_init],
        mode='lines', line=dict(color='green', dash='dash'), name='True Function'
    ))

    # [1] GP Mean
    fig.add_trace(go.Scatter(
        x=x0_range.numpy(), y=preds_mean[t_init, m_init],
        mode='lines', line=dict(color='blue'), name='Deep GP Mean'
    ))

    # [2] Upper Bound
    upper = preds_mean[t_init, m_init] + 2 * preds_std[t_init, m_init]
    fig.add_trace(go.Scatter(
        x=x0_range.numpy(), y=upper,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # [3] Lower Bound
    lower = preds_mean[t_init, m_init] - 2 * preds_std[t_init, m_init]
    fig.add_trace(go.Scatter(
        x=x0_range.numpy(), y=lower,
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)',
        name='Uncertainty (2σ)'
    ))

    # [4] Training Points
    fig.add_trace(go.Scatter(
        x=sx, y=sy,
        mode='markers', marker=dict(color='red', size=9, line=dict(width=1, color='black')),
        name=f'Train Data (±{tolerance})'
    ))

    # 5. Create Sliders
    # Slider 1: Monomer
    steps_mono = []
    for m_idx, v_mono in enumerate(x1_steps):
        y_mu = preds_mean[t_init, m_idx]
        y_std_ = preds_std[t_init, m_idx]
        y_true = truth_vals[t_init, m_idx]
        sx_new, sy_new = get_nearby_points(v_mono.item(), val_temp_init)

        step = dict(
            method="update",
            args=[{
                "y": [y_true, y_mu, y_mu + 2 * y_std_, y_mu - 2 * y_std_, sy_new],
                "x": [x0_range.numpy(), x0_range.numpy(), x0_range.numpy(), x0_range.numpy(), sx_new]
            }],
            label=f"{v_mono:.2f}"
        )
        steps_mono.append(step)

    # Slider 2: Temperature
    steps_temp = []
    for t_idx, v_temp in enumerate(x2_steps):
        y_mu = preds_mean[t_idx, m_init]
        y_std_ = preds_std[t_idx, m_init]
        y_true = truth_vals[t_idx, m_init]
        sx_new, sy_new = get_nearby_points(val_mono_init, v_temp.item())

        step = dict(
            method="update",
            args=[{
                "y": [y_true, y_mu, y_mu + 2 * y_std_, y_mu - 2 * y_std_, sy_new],
                "x": [x0_range.numpy(), x0_range.numpy(), x0_range.numpy(), x0_range.numpy(), sx_new]
            }],
            label=f"{v_temp:.2f}"
        )
        steps_temp.append(step)

    # --- FIX 2: LAYOUT SPACING ---
    # Use negative 'y' to push sliders below the chart, and add bottom margin
    fig.update_layout(
        title="Deep GP Slice Analysis",
        xaxis_title="Catalyst (Dim 0)",
        yaxis_title="Target",
        height=700,
        margin=dict(b=180),  # Extra space at bottom for sliders
        sliders=[
            dict(
                active=m_init,
                currentvalue={"prefix": "Monomer (Dim 1): "},
                pad={"t": 50},
                steps=steps_mono,
                y=-0.15  # Push below X-axis
            ),
            dict(
                active=t_init,
                currentvalue={"prefix": "Temperature (Dim 2): "},
                pad={"t": 50},
                steps=steps_temp,
                y=-0.35  # Push further down
            )
        ]
    )

    fig.show()
# --- Main Execution Block ---

def main():
    bench = get_benchmark("PolymerizationV2")

    # 1. Generate & Observe Data
    n_init = 30
    train_x_raw = torch.rand(n_init, 3).double()
    train_x_raw = train_x_raw * (bench.bounds[1] - bench.bounds[0]).cpu() + bench.bounds[0].cpu()

    # Get all intermediate steps
    outputs = bench.observe_all_steps(train_x_raw)
    y1, y2, y3 = outputs[0].cpu(), outputs[1].cpu(), outputs[2].cpu()

    # Normalize Inputs
    x_mean, x_std = train_x_raw.mean(0), train_x_raw.std(0) + 1e-6
    train_x = (train_x_raw - x_mean) / x_std

    # 2. Train Individual GPs
    print("Training Individual Step GPs...")
    X_step1 = train_x[:, [0, 1]]
    print(train_x.shape)
    mean_layer1 = train_individual_step_gp(X_step1, y1, active_dims=[0, 1])

    means = [mean_layer1, None]

    # 3. Fit Deep GP
    y_final = y3
    y_mean, y_std = y_final.mean(), y_final.std() + 1e-6
    train_y_norm = (y_final - y_mean) / y_std

    dgp = LightweightDeepGP(train_x, train_y_norm, hidden_dims=[3],
                            # mean_modules=means
                            )

    # Fast training for demo
    dgp.fit_numpyro(num_chains=1, num_samples=1000, warmup_steps=500)

    # 4. Predict & Visualize
    print("Running Visualization...")
    # Pass 'outputs[2]' or 'train_y_raw' (which is y3)
    visualize_interactive_slice(dgp, bench, train_x_raw, y3, x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":
    main()