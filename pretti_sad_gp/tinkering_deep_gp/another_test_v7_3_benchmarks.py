import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import time

# --- JAX / NumPyro Imports ---
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# --- BoTorch / GPyTorch Imports ---
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# --- Benchmark Import ---
from benchmarks import get_benchmark

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

JITTER = 1e-6


# ==========================================
# 1. JAX/NumPyro Components (Generic)
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


def jax_gp_mean_fn(x, train_x, weights, lengthscale, amplitude, input_mean=None, input_std=None):
    if input_mean is not None and input_std is not None:
        x = (x - input_mean) / input_std
    k_star = matern52_kernel_jax(x, train_x, lengthscale, amplitude)
    return jnp.dot(k_star, weights)


def _deep_gp_model_jax(x, y=None, hidden_dims=[2], standardized_map=None, mean_params_list=None):
    n = x.shape[0]
    # Default map if none provided
    if standardized_map is None: standardized_map = {0: list(range(x.shape[1]))}
    # Ensure params list matches layers (hidden + output)
    if mean_params_list is None: mean_params_list = [None] * (len(hidden_dims) + 1)

    current_h = None

    # --- Hidden Layers ---
    for layer_idx, width in enumerate(hidden_dims):
        # 1. Prepare Input for Layer
        inj_dims = standardized_map.get(layer_idx, [])
        inj_x = x[:, inj_dims] if inj_dims else jnp.empty((n, 0))

        # Concatenate previous hidden state with injected inputs
        current_input = inj_x if current_h is None else jnp.concatenate([current_h, inj_x], axis=1)
        input_dim = current_input.shape[1]

        layer_priors = mean_params_list[layer_idx]
        h_list = []

        # 2. Iterate over width of hidden layer (independent GPs)
        for dim_idx in range(width):
            ls_d = numpyro.sample(f"L{layer_idx}_ls_{dim_idx}", dist.LogNormal(0.0, 1.0).expand([input_dim]))
            amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(0.0, 1.0))

            k_mat = matern52_kernel_jax(current_input, current_input, ls_d, amp_d)
            cov = k_mat + jnp.eye(n) * JITTER

            # Mean Function (Prior or Linear)
            mean = jnp.zeros(n)
            if layer_priors is not None and len(layer_priors) > dim_idx:
                p = layer_priors[dim_idx]
                if p is not None:
                    calc_mean = jax_gp_mean_fn(
                        current_input, p['train_x'], p['weights'], p['ls'], p['amp'],
                        input_mean=p.get('input_mean'), input_std=p.get('input_std')
                    )
                    mean = calc_mean.squeeze()
                else:
                    # Fallback to linear if prior missing for this node
                    slope_d = numpyro.sample(f"L{layer_idx}_slope_{dim_idx}", dist.Normal(0.0, 0.5).expand([input_dim]))
                    bias_d = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))
                    mean = jnp.dot(current_input, slope_d) + bias_d
            else:
                slope_d = numpyro.sample(f"L{layer_idx}_slope_{dim_idx}", dist.Normal(0.0, 0.5).expand([input_dim]))
                bias_d = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))
                mean = jnp.dot(current_input, slope_d) + bias_d

            h_d = numpyro.sample(f"h_{layer_idx}_{dim_idx}", dist.MultivariateNormal(mean, covariance_matrix=cov))
            h_list.append(h_d)

        current_h = jnp.stack(h_list, axis=1)

    # --- Output Layer (Key -1) ---
    out_inj_dims = standardized_map.get(-1, [])
    final_input = current_h if not out_inj_dims else jnp.concatenate([current_h, x[:, out_inj_dims]], axis=1)
    input_dim = final_input.shape[1]

    ls_out = numpyro.sample("out_ls", dist.LogNormal(-1.0, 1.0).expand([input_dim]))
    amp_out = numpyro.sample("out_amp", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("obs_noise", dist.HalfNormal(0.3))

    mean_out = jnp.zeros(n)
    out_priors = mean_params_list[-1]

    if out_priors is not None and len(out_priors) > 0:
        p = out_priors[0]
        calc_mean = jax_gp_mean_fn(final_input, p['train_x'], p['weights'], p['ls'], p['amp'],
                                   input_mean=p.get('input_mean'), input_std=p.get('input_std'))
        mean_out = calc_mean.squeeze()

    cov_out = matern52_kernel_jax(final_input, final_input, ls_out, amp_out)
    cov_final = cov_out + jnp.eye(n) * (noise + JITTER)
    numpyro.sample("obs", dist.MultivariateNormal(mean_out, covariance_matrix=cov_final), obs=y)


# ==========================================
# 2. PyTorch Helper & Class
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


def extract_gp_params_for_jax(gp_model):
    gp_model.eval()
    with torch.no_grad():
        train_x = gp_model.train_inputs[0]
        train_y = gp_model.train_targets.unsqueeze(-1)
        ls = gp_model.covar_module.base_kernel.lengthscale.squeeze().detach()
        amp = gp_model.covar_module.outputscale.detach()
        noise = gp_model.likelihood.noise.detach()

        K = matern52_kernel_torch(train_x.unsqueeze(0), train_x.unsqueeze(0), ls, amp).squeeze(0)
        K_plus_noise = K + torch.eye(K.shape[0], device=train_x.device) * noise
        L = torch.linalg.cholesky(K_plus_noise)
        weights = torch.cholesky_solve(train_y, L)

    return {
        'train_x': jnp.array(train_x.cpu().numpy()),
        'weights': jnp.array(weights.cpu().numpy()),
        'ls': jnp.array(ls.cpu().numpy()),
        'amp': jnp.array(amp.cpu().numpy())
    }


class LightweightDeepGP:
    def __init__(self, train_x, train_y, hidden_dims=[2], injection_map=None, mean_params_list=None):
        self.train_x = train_x.clone()
        if self.train_x.dim() == 1: self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = train_y
        self.hidden_dims = hidden_dims
        # Default map: Layer 0 takes all inputs, Output takes only hidden
        self.injection_map = injection_map if injection_map else {0: list(range(self.train_x.shape[1]))}
        self.mean_params_list = mean_params_list
        self.samples = None
        self.std_map = None

    def _validate_and_standardize_map(self):
        # Maps user friendly keys (like 'out') to internal indices (like -1)
        num_inputs = self.train_x.shape[-1]
        num_hidden = len(self.hidden_dims)
        out_idx = num_hidden
        standardized = {}

        for key, dims in self.injection_map.items():
            if key == "out" or key == out_idx or key == -1:
                target_key = -1
            elif isinstance(key, int) and key < num_hidden:
                target_key = key
            else:
                raise ValueError(f"Invalid injection key '{key}'.")
            standardized[target_key] = dims
        return standardized

    def fit_numpyro(self, num_chains=1, num_samples=1000, warmup_steps=500, verbose=True):
        if verbose:
            print(f"--- Fitting Deep GP on {jax.lib.xla_bridge.get_backend().platform} ---")
        self.std_map = self._validate_and_standardize_map()
        X_jax = jnp.array(self.train_x.cpu().numpy())
        y_jax = jnp.array(self.train_y.cpu().numpy())

        nuts_kernel = NUTS(_deep_gp_model_jax)
        mcmc = MCMC(nuts_kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=verbose)
        rng_key = jax.random.PRNGKey(int(time.time()))

        mcmc.run(rng_key, X_jax, y_jax, hidden_dims=self.hidden_dims,
                 standardized_map=self.std_map, mean_params_list=self.mean_params_list)

        jax_samples = mcmc.get_samples()
        device = self.train_x.device
        self.samples = {k: torch.tensor(np.array(v), device=device) for k, v in jax_samples.items()}

        for layer_idx, width in enumerate(self.hidden_dims):
            self.samples[f"L{layer_idx}_ls"] = torch.stack([self.samples[f"L{layer_idx}_ls_{d}"] for d in range(width)],
                                                           dim=1)
            self.samples[f"L{layer_idx}_amp"] = torch.stack(
                [self.samples[f"L{layer_idx}_amp_{d}"] for d in range(width)], dim=1)
            # Handle potentially missing linear params if using priors
            if f"L{layer_idx}_slope_0" in self.samples:
                self.samples[f"L{layer_idx}_slope"] = torch.stack(
                    [self.samples[f"L{layer_idx}_slope_{d}"] for d in range(width)], dim=1)
                self.samples[f"L{layer_idx}_bias"] = torch.stack(
                    [self.samples[f"L{layer_idx}_bias_{d}"] for d in range(width)], dim=1)

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

        curr_h_train, curr_h_test = None, None

        # --- Propagate through Hidden Layers ---
        for layer_idx, width in enumerate(self.hidden_dims):
            # Construct inputs (Hidden + Injected)
            inj = self.std_map.get(layer_idx, [])
            x_inj_train = train_x[:, inj].unsqueeze(0).expand(S, -1, -1)
            x_inj_test = X[:, :, inj].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)

            curr_in_train = x_inj_train if curr_h_train is None else torch.cat([curr_h_train, x_inj_train], dim=-1)
            curr_in_test = x_inj_test if curr_h_test is None else torch.cat([curr_h_test, x_inj_test], dim=-1)

            next_h_train_list, next_h_test_list = [], []
            layer_priors = self.mean_params_list[layer_idx] if self.mean_params_list else None

            for d_idx in range(width):
                ls = self.samples[f'L{layer_idx}_ls'][:, d_idx, :].unsqueeze(1)
                amp = self.samples[f'L{layer_idx}_amp'][:, d_idx].view(S, 1, 1)
                h_train_samples = self.samples[f'h_{layer_idx}_{d_idx}'].unsqueeze(-1)
                next_h_train_list.append(h_train_samples)

                K_rr = matern52_kernel_torch(curr_in_train, curr_in_train, ls, amp) + torch.eye(N_train,
                                                                                                device=device) * JITTER
                L_rr = torch.linalg.cholesky(K_rr)

                # Prior Mean
                has_prior = layer_priors is not None and len(layer_priors) > d_idx and layer_priors[d_idx] is not None
                if has_prior:
                    p = layer_priors[d_idx]
                    p_tx = torch.as_tensor(np.array(p['train_x']), device=device)
                    p_w = torch.as_tensor(np.array(p['weights']), device=device)
                    p_ls = torch.as_tensor(np.array(p['ls']), device=device)
                    p_amp = torch.as_tensor(np.array(p['amp']), device=device)
                    # Note: p_tx is already normalized in prior training, but we might need input_mean/std shift if stored
                    # Simplified: We assume prior training and DGP usage share same normalization

                    # Handle separate normalization if present in prior dict
                    p_in_mean = torch.as_tensor(np.array(p.get('input_mean', 0.0)), device=device)
                    p_in_std = torch.as_tensor(np.array(p.get('input_std', 1.0)), device=device)

                    c_tr = (curr_in_train - p_in_mean) / p_in_std
                    c_te = (curr_in_test - p_in_mean) / p_in_std

                    k_tr_p = matern52_kernel_torch(c_tr, p_tx.expand(S, -1, -1), p_ls, p_amp)
                    train_p_m = k_tr_p @ p_w.expand(S, -1, -1)
                    k_te_p = matern52_kernel_torch(c_te, p_tx.expand(S * B, -1, -1), p_ls, p_amp)
                    test_p_m = k_te_p @ p_w.expand(S * B, -1, -1)
                else:
                    slope = self.samples[f'L{layer_idx}_slope'][:, d_idx, :]
                    bias = self.samples[f'L{layer_idx}_bias'][:, d_idx].view(S, 1, 1)
                    train_p_m = (curr_in_train * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias
                    slope_rep = slope.repeat_interleave(B, dim=0)
                    bias_rep = bias.repeat_interleave(B, dim=0)
                    test_p_m = (curr_in_test * slope_rep.unsqueeze(1)).sum(-1, keepdim=True) + bias_rep

                weights = torch.cholesky_solve(h_train_samples - train_p_m, L_rr)

                # Expand for Batch B
                ls_rep = ls.repeat_interleave(B, dim=0)
                amp_rep = amp.repeat_interleave(B, dim=0)
                train_in_rep = curr_in_train.repeat_interleave(B, dim=0)
                weights_rep = weights.repeat_interleave(B, dim=0)
                L_rr_rep = L_rr.repeat_interleave(B, dim=0)

                K_tr = matern52_kernel_torch(curr_in_test, train_in_rep, ls_rep, amp_rep)
                K_tt = matern52_kernel_torch(curr_in_test, curr_in_test, ls_rep, amp_rep) + torch.eye(N_test,
                                                                                                      device=device) * JITTER

                mu_test = test_p_m + K_tr @ weights_rep
                v = torch.linalg.solve(L_rr_rep, K_tr.transpose(-1, -2))
                cov_test = K_tt - v.transpose(-1, -2) @ v

                dist = torch.distributions.MultivariateNormal(mu_test.squeeze(-1), cov_test)
                next_h_test_list.append(dist.rsample().unsqueeze(-1))

            curr_h_train = torch.cat(next_h_train_list, dim=2)
            curr_h_test = torch.cat(next_h_test_list, dim=2)

        # --- Output Layer ---
        inj_out = self.std_map.get(-1, [])
        final_in_train = torch.cat([curr_h_train, train_x[:, inj_out].unsqueeze(0).expand(S, -1, -1)],
                                   dim=-1) if inj_out else curr_h_train
        final_in_test = torch.cat(
            [curr_h_test, X[:, :, inj_out].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)],
            dim=-1) if inj_out else curr_h_test

        ls_out = self.samples['out_ls'].unsqueeze(1)
        amp_out = self.samples['out_amp'].view(S, 1, 1)
        noise = self.samples['obs_noise'].view(S, 1, 1)

        K_rr = matern52_kernel_torch(final_in_train, final_in_train, ls_out, amp_out) + torch.eye(N_train,
                                                                                                  device=device) * (
                           noise + JITTER)
        L_rr = torch.linalg.cholesky(K_rr)

        out_priors = self.mean_params_list[-1]
        has_out_prior = out_priors is not None and len(out_priors) > 0 and out_priors[0] is not None

        if has_out_prior:
            p = out_priors[0]
            p_tx = torch.as_tensor(np.array(p['train_x']), device=device)
            p_w = torch.as_tensor(np.array(p['weights']), device=device)
            p_ls = torch.as_tensor(np.array(p['ls']), device=device)
            p_amp = torch.as_tensor(np.array(p['amp']), device=device)

            p_in_mean = torch.as_tensor(np.array(p.get('input_mean', 0.0)), device=device)
            p_in_std = torch.as_tensor(np.array(p.get('input_std', 1.0)), device=device)

            c_tr = (final_in_train - p_in_mean) / p_in_std
            c_te = (final_in_test - p_in_mean) / p_in_std

            k_tr_p = matern52_kernel_torch(c_tr, p_tx.expand(S, -1, -1), p_ls, p_amp)
            train_p_m = k_tr_p @ p_w.expand(S, -1, -1)
            k_te_p = matern52_kernel_torch(c_te, p_tx.expand(S * B, -1, -1), p_ls, p_amp)
            test_p_m = k_te_p @ p_w.expand(S * B, -1, -1)
        else:
            train_p_m = torch.zeros(S, N_train, 1, device=device)
            test_p_m = torch.zeros(S * B, N_test, 1, device=device)

        weights = torch.cholesky_solve(self.train_y.to(device).view(1, N_train, 1).expand(S, -1, -1) - train_p_m, L_rr)

        train_in_rep = final_in_train.repeat_interleave(B, dim=0)
        ls_out_rep = ls_out.repeat_interleave(B, dim=0)
        amp_out_rep = amp_out.repeat_interleave(B, dim=0)
        weights_rep = weights.repeat_interleave(B, dim=0)

        K_tr = matern52_kernel_torch(final_in_test, train_in_rep, ls_out_rep, amp_out_rep)
        y_pred = test_p_m + K_tr @ weights_rep

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
        mu, var = preds_samples.mean(dim=0), preds_samples.var(dim=0)

        mu = mu.squeeze(-1)
        diff = preds_samples.squeeze(-1).permute(1, 2, 0) - mu.unsqueeze(-1)
        num_samples = diff.shape[-1]
        cov = (diff @ diff.transpose(-1, -2)) / (num_samples - 1)
        jitter = JITTER * torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0).expand(cov.shape[0], -1, -1)
        cov = cov + jitter
        return GPyTorchPosterior(MultivariateNormal(mu, cov))

    @property
    def num_outputs(self): return self._num_outputs


# ==========================================
# 3. Prior Training Logic (Per Benchmark)
# ==========================================

def train_poly_priors(bench, train_x_raw, x_mean, x_std):
    print("   [Polymerization Logic] Training Priors on LF Physics...")
    # 1. Generate dense synthetic inputs (LF)
    n_synth_grid = 1000
    bounds = bench.bounds.cpu()
    synth_x_raw = torch.rand(n_synth_grid, train_x_raw.shape[1]).double() * (bounds[1] - bounds[0]) + bounds[0]

    # Observe all steps (Low Fidelity)
    all_steps_synth = bench.observe_all_steps(synth_x_raw, mode='LF')
    synth_x_std = (synth_x_raw - x_mean) / x_std

    p1_params = []
    # Prior 1: Stage 1 (Mixing output)
    y1_raw = all_steps_synth[0]
    y1_std_norm = (y1_raw - y1_raw.mean()) / (y1_raw.std() + 1e-6)
    gp_y1 = SingleTaskGP(synth_x_std, y1_std_norm)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y1.likelihood, gp_y1))
    p1_params.append(extract_gp_params_for_jax(gp_y1))

    # Prior 2: Stage 2 (Thermal output)
    y2_raw = all_steps_synth[1]
    y2_std_norm = (y2_raw - y2_raw.mean()) / (y2_raw.std() + 1e-6)
    gp_y2 = SingleTaskGP(synth_x_std, y2_std_norm)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y2.likelihood, gp_y2))
    p1_params.append(extract_gp_params_for_jax(gp_y2))

    # Prior 3: Final Stage (Latent Space -> Yield)
    n_synth_lat = 400
    synth_latents = torch.rand(n_synth_lat, 2).double()
    # Explicitly use mode='LF' for the final reaction
    y_synth = torch.tensor(
        bench.step_3_reaction(synth_latents[:, 0].numpy(), synth_latents[:, 1].numpy(), mode='LF')).unsqueeze(-1)

    synth_lat_std = (synth_latents - 0.5) / 0.288  # Approx norm
    y_synth_std = (y_synth - y_synth.mean()) / (y_synth.std() + 1e-6)

    gp_y3 = SingleTaskGP(synth_lat_std, y_synth_std)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y3.likelihood, gp_y3))

    p2_params = [{
        **extract_gp_params_for_jax(gp_y3),
        'input_mean': jnp.array(synth_latents.mean(0).numpy()),
        'input_std': jnp.array(synth_latents.std(0).numpy())
    }]
    return [p1_params, p2_params]


def train_pharma_priors(bench, train_x_raw, x_mean, x_std):
    print("   [Pharma Logic] Training Priors on LF Physics...")
    n_synth = 1000
    bounds = bench.bounds.cpu()
    synth_x_raw = torch.rand(n_synth, train_x_raw.shape[1]).double() * (bounds[1] - bounds[0]) + bounds[0]

    # Run LF Physics
    steps = bench.observe_all_steps(synth_x_raw, mode='LF')

    # -- Layer 1 Priors: Reactor Outputs (Step 1) --
    # Inputs: x0, x1, x2, x4 (indices [0, 1, 2, 4])
    # Outputs: steps[1] (3 dims: Prod, Deg, Side)

    input_indices = [0, 1, 2, 4]
    X_L1 = synth_x_raw[:, input_indices]

    # Normalize inputs relative to global stats
    X_L1_std = (X_L1 - x_mean[input_indices]) / x_std[input_indices]

    Y_Reactor = steps[1]  # (N, 3)
    p1_params = []

    for i in range(3):
        y_dim = Y_Reactor[:, i:i + 1]
        y_dim_norm = (y_dim - y_dim.mean()) / (y_dim.std() + 1e-6)

        gp = SingleTaskGP(X_L1_std, y_dim_norm)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        # Important: Store normalization so DGP can match it
        params = extract_gp_params_for_jax(gp)
        # Note: Inputs are already standardized by Global mean/std, so we don't need extra shift in the GP mean function
        p1_params.append(params)

    # -- Layer 2 Prior: Output Yield (Step 3) --
    # Inputs: Hidden (Reactor Outputs) + x3
    # Targets: Final Yield

    # We use the Reactor outputs generated by the physics to ensure manifold validity
    # Hidden inputs (3 dims) + x3 (1 dim)
    X_L2_Hidden = Y_Reactor  # Raw physics output
    X_L2_Inj = synth_x_raw[:, 3:4]
    X_L2 = torch.cat([X_L2_Hidden, X_L2_Inj], dim=1)

    Y_Final = steps[3]
    Y_Final_norm = (Y_Final - Y_Final.mean()) / (Y_Final.std() + 1e-6)

    # Normalize inputs for this specific GP
    X_L2_mean = X_L2.mean(0)
    X_L2_std = X_L2.std(0) + 1e-6
    X_L2_norm = (X_L2 - X_L2_mean) / X_L2_std

    gp_out = SingleTaskGP(X_L2_norm, Y_Final_norm)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_out.likelihood, gp_out))

    p2_params = [{
        **extract_gp_params_for_jax(gp_out),
        'input_mean': jnp.array(X_L2_mean.numpy()),
        'input_std': jnp.array(X_L2_std.numpy())
    }]

    return [p1_params, p2_params]


# ==========================================
# 4. Configuration
# ==========================================

BENCH_CONFIGS = {
    "Polymerization": {
        "hidden_dims": [2],
        "injection_map": {0: [0, 1, 2], -1: []},
        "prior_fn": train_poly_priors
    },
    "PolymerizationV2": {
        "hidden_dims": [2],
        "injection_map": {0: [0, 1, 2], -1: []},
        "prior_fn": train_poly_priors
    },
    "PharmaV1": {
        "hidden_dims": [3],
        "injection_map": {0: [0, 1, 2, 4], -1: [3]},
        "prior_fn": train_pharma_priors
    },
    "PharmaV2": {
        "hidden_dims": [3],
        "injection_map": {0: [0, 1, 2, 4], -1: [3]},
        "prior_fn": train_pharma_priors
    }
}


# ==========================================
# 5. Visualization
# ==========================================

def visualize_poly_slices(dgp, log_ei, bench, train_x_raw, candidate_raw, x_mean, x_std, y_mean, y_std, device):
    print("Generating 2D Slices for Polymerization...")
    # Downsample for speed
    full_samples = dgp.samples
    dgp.samples = {k: v[-50:] for k, v in full_samples.items()}

    temps = [0.0, 0.5, 1.0]
    n_grid = 30
    x_range = torch.linspace(0, 1, n_grid)
    y_range = torch.linspace(0, 1, n_grid)
    X_grid, Y_grid = torch.meshgrid(x_range, y_range, indexing='ij')
    flat_grid = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

    fig = make_subplots(rows=3, cols=3, subplot_titles=[f"HF Truth (T={t})" for t in temps] +
                                                       [f"Post Mean (T={t})" for t in temps] +
                                                       [f"Acq Func (T={t})" for t in temps])

    for i, t in enumerate(temps):
        fixed_dim = torch.full((flat_grid.shape[0], 1), t)
        test_x_raw = torch.cat([flat_grid, fixed_dim], dim=1)
        test_x_norm = (test_x_raw - x_mean.cpu()) / x_std.cpu()

        # Truth
        true_y = bench.observe(test_x_raw, mode='HF').cpu().numpy().flatten()
        # Mean
        mu, _ = dgp.predict(test_x_norm)
        mu_raw = mu * y_std.item() + y_mean.item()
        # Acq
        with torch.no_grad():
            acq = log_ei(test_x_norm.to(device).unsqueeze(1)).cpu().numpy()

        hm_args = dict(x=x_range.numpy(), y=y_range.numpy(), showscale=False)

        # Plot
        fig.add_trace(go.Contour(z=true_y.reshape(n_grid, n_grid).T, colorscale='Viridis', **hm_args), row=1, col=i + 1)
        fig.add_trace(go.Contour(z=mu_raw.reshape(n_grid, n_grid).T, colorscale='Viridis', **hm_args), row=2, col=i + 1)
        fig.add_trace(go.Contour(z=acq.reshape(n_grid, n_grid).T, colorscale='Cividis', **hm_args), row=3, col=i + 1)

        # Overlay candidate if in slice
        cand_t = candidate_raw[0, 2].item()
        if abs(cand_t - t) < 0.15:
            fig.add_trace(go.Scatter(x=[candidate_raw[0, 0].item()], y=[candidate_raw[0, 1].item()],
                                     mode='markers', marker=dict(color='red', size=10, symbol='star'),
                                     showlegend=False),
                          row=3, col=i + 1)

    dgp.samples = full_samples
    fig.update_layout(height=900, width=1200, title_text="Polymerization Optimization Slices")
    fig.show()


def visualize_convergence(y_history, bench_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_history.numpy(), mode='lines+markers', name='Observed Y'))
    fig.update_layout(title=f"Optimization Convergence: {bench_name}", xaxis_title="Iteration",
                      yaxis_title="Objective (Yield)")
    fig.show()


# ==========================================
# 6. Main Loop
# ==========================================

def run_benchmark_loop(bench_name):
    print(f"\n{'=' * 60}\nRunning Benchmark: {bench_name}\n{'=' * 60}")
    bench = get_benchmark(bench_name)
    config = BENCH_CONFIGS[bench_name]

    # Init Data
    N_INIT = 10
    N_ITER = 1

    # Observe HF
    train_x_raw = torch.rand(N_INIT, bench.input_dim).double()
    train_x_raw = train_x_raw * (bench.bounds[1] - bench.bounds[0]).cpu() + bench.bounds[0].cpu()
    train_y_raw = bench.observe(train_x_raw, mode='HF').squeeze(-1).cpu()

    x_mean, x_std = train_x_raw.mean(0), train_x_raw.std(0) + 1e-6
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std() + 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train Priors (LF)
    prior_params = config['prior_fn'](bench, train_x_raw, x_mean, x_std)

    # Optimization Loop
    final_dgp, final_acq = None, None

    std_bounds_lower = (bench.bounds[0].cpu() - x_mean) / x_std
    std_bounds_upper = (bench.bounds[1].cpu() - x_mean) / x_std
    bounds = torch.stack([std_bounds_lower, std_bounds_upper]).to(device)

    for i in range(N_ITER):
        print(f"   [Iter {i + 1}] Best Y so far: {train_y_raw.min().item():.4f}")

        train_x = (train_x_raw - x_mean) / x_std
        train_y = (train_y_raw - y_mean) / y_std
        train_x, train_y = train_x.to(device), train_y.to(device)

        dgp = LightweightDeepGP(train_x, train_y,
                                hidden_dims=config['hidden_dims'],
                                injection_map=config['injection_map'],
                                mean_params_list=prior_params)

        dgp.fit_numpyro(num_chains=1, num_samples=3, warmup_steps=3, verbose=True)

        bo_model = DeepGPBoTorchModel(dgp)
        log_ei = LogExpectedImprovement(model=bo_model, best_f=train_y.max(), maximize=True)

        cand_norm, _ = optimize_acqf(log_ei, bounds=bounds, q=1, num_restarts=3, raw_samples=50)
        cand_raw = cand_norm.cpu() * x_std + x_mean
        new_y = bench.observe(cand_raw, mode='HF').squeeze(-1).cpu()

        train_x_raw = torch.cat([train_x_raw, cand_raw])
        train_y_raw = torch.cat([train_y_raw, new_y])
        final_dgp, final_acq = dgp, log_ei

    # Visualizations
    visualize_convergence(train_y_raw, bench_name)

    # Detailed viz only for 3D benchmarks
    if bench.input_dim == 3:
        visualize_poly_slices(final_dgp, final_acq, bench, train_x_raw, cand_raw, x_mean, x_std, y_mean, y_std, device)
    else:
        print(f"Skipping 2D slice viz for {bench_name} (Input Dim {bench.input_dim} > 3).")


if __name__ == "__main__":
    benchmarks_to_run = ["Polymerization", "PolymerizationV2", "PharmaV1", "PharmaV2"]

    for b in benchmarks_to_run:
        try:
            run_benchmark_loop(b)
        except Exception as e:
            print(f"Error running {b}: {e}")
            import traceback

            traceback.print_exc()