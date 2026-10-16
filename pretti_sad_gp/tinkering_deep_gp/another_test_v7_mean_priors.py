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

# --- BoTorch / GPyTorch Imports ---
from botorch.models import SingleTaskGP, ModelListGP
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
# 1. JAX/NumPyro Components (FIXED)
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
    """
    Fixed: Uses integer key -1 to retrieve output injection dims instead of "out".
    """
    n = x.shape[0]
    if standardized_map is None: standardized_map = {0: list(range(x.shape[1]))}
    if mean_params_list is None: mean_params_list = [None] * (len(hidden_dims) + 1)

    current_h = None

    # --- Hidden Layers ---
    for layer_idx, width in enumerate(hidden_dims):
        inj_dims = standardized_map.get(layer_idx, [])
        inj_x = x[:, inj_dims] if inj_dims else jnp.empty((n, 0))
        current_input = inj_x if current_h is None else jnp.concatenate([current_h, inj_x], axis=1)
        input_dim = current_input.shape[1]

        layer_priors = mean_params_list[layer_idx]

        h_list = []
        for dim_idx in range(width):
            ls_d = numpyro.sample(f"L{layer_idx}_ls_{dim_idx}", dist.LogNormal(0.0, 1.0).expand([input_dim]))
            # amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(-1.0, 0.5))
            amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(0.0, 1.0))

            k_mat = matern52_kernel_jax(current_input, current_input, ls_d, amp_d)
            cov = k_mat + jnp.eye(n) * JITTER

            mean = jnp.zeros(n)
            if layer_priors is not None and len(layer_priors) > dim_idx:
                p = layer_priors[dim_idx]
                if p is not None:
                    calc_mean = jax_gp_mean_fn(
                        current_input,
                        p['train_x'],
                        p['weights'],
                        p['ls'],
                        p['amp'],
                        input_mean=p.get('input_mean'),  # Pass the stats
                        input_std=p.get('input_std')
                    )
                    mean = calc_mean.squeeze()
                else:
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

    # --- Output Layer (FIXED: Uses key -1) ---
    # We use -1 to represent the output layer to keep all dictionary keys as integers
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
        calc_mean = jax_gp_mean_fn(final_input, p['train_x'], p['weights'], p['ls'], p['amp'])
        mean_out = calc_mean.squeeze()

    cov_out = matern52_kernel_jax(final_input, final_input, ls_out, amp_out)
    cov_final = cov_out + jnp.eye(n) * (noise + JITTER)
    numpyro.sample("obs", dist.MultivariateNormal(mean_out, covariance_matrix=cov_final), obs=y)


# ==========================================
# 2. PyTorch Helper & Class (FIXED)
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
        self.injection_map = injection_map if injection_map else {0: list(range(self.train_x.shape[1]))}
        self.mean_params_list = mean_params_list
        self.samples = None
        self.std_map = None

    def _validate_and_standardize_map(self):
        """
        Fixed: Maps 'out' to integer -1 so the dictionary keys are all integers.
        """
        num_inputs = self.train_x.shape[-1]
        num_hidden = len(self.hidden_dims)
        out_idx = num_hidden
        standardized = {}
        mapped_dims = set()

        for key, dims in self.injection_map.items():
            # CHANGE: We map 'out' to -1 to ensure all dict keys are integers.
            if key == "out" or key == out_idx:
                target_key = -1
            elif isinstance(key, int) and key < num_hidden:
                target_key = key
            else:
                raise ValueError(f"Invalid injection key '{key}'.")

            if target_key not in standardized: standardized[target_key] = []
            standardized[target_key].extend(dims)
            mapped_dims.update(dims)

        all_dims = set(range(num_inputs))
        unmapped = all_dims - mapped_dims
        if unmapped:
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
            if f"L{layer_idx}_slope_0" in self.samples:
                self.samples[f"L{layer_idx}_slope"] = torch.stack(
                    [self.samples[f"L{layer_idx}_slope_{d}"] for d in range(width)], dim=1)
                self.samples[f"L{layer_idx}_bias"] = torch.stack(
                    [self.samples[f"L{layer_idx}_bias_{d}"] for d in range(width)], dim=1)

    def predict_torch(self, X):
        if self.samples is None: raise RuntimeError("Model not trained.")
        if X.dim() == 1: X = X.unsqueeze(0).unsqueeze(0)
        if X.dim() == 2: X = X.unsqueeze(0)

        # B: Batch size of candidates (from BoTorch optimizer)
        # N_test: Number of points per candidate
        # S: Number of MCMC samples
        B, N_test, D = X.shape
        S = self.samples['out_ls'].shape[0]
        device = self.samples['out_ls'].device

        X = X.to(device)
        train_x = self.train_x.to(device)
        N_train = train_x.shape[0]

        curr_h_train, curr_h_test = None, None

        # --- Propagate through Hidden Layers ---
        for layer_idx, width in enumerate(self.hidden_dims):
            # 1. Construct Inputs for this layer
            inj = self.std_map.get(layer_idx, [])

            # Train input: (S, N_train, D_in)
            x_inj_train = train_x[:, inj].unsqueeze(0).expand(S, -1, -1)
            # Test input: (S*B, N_test, D_in) - reshaped to handle BoTorch batches
            x_inj_test = X[:, :, inj].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)

            curr_in_train = x_inj_train if curr_h_train is None else torch.cat([curr_h_train, x_inj_train], dim=-1)
            curr_in_test = x_inj_test if curr_h_test is None else torch.cat([curr_h_test, x_inj_test], dim=-1)

            next_h_train_list, next_h_test_list = [], []
            layer_priors = self.mean_params_list[layer_idx] if self.mean_params_list else None

            # 2. Iterate over hidden units (width)
            for d_idx in range(width):
                # Retrieve Params
                ls = self.samples[f'L{layer_idx}_ls'][:, d_idx, :].unsqueeze(1)
                amp = self.samples[f'L{layer_idx}_amp'][:, d_idx].view(S, 1, 1)
                h_train_samples = self.samples[f'h_{layer_idx}_{d_idx}'].unsqueeze(-1)

                # --- CRITICAL FIX: Save training samples for next layer ---
                next_h_train_list.append(h_train_samples)

                # Compute Kernel Matrices
                # K_rr: (S, N_train, N_train)
                K_rr = matern52_kernel_torch(curr_in_train, curr_in_train, ls, amp) + torch.eye(N_train,
                                                                                                device=device) * JITTER
                L_rr = torch.linalg.cholesky(K_rr)

                # --- Mean Function Logic ---
                has_prior = layer_priors is not None and len(layer_priors) > d_idx and layer_priors[d_idx] is not None

                if has_prior:
                    p = layer_priors[d_idx]
                    p_tx = torch.as_tensor(np.array(p['train_x']), device=device)
                    p_w = torch.as_tensor(np.array(p['weights']), device=device)
                    p_ls = torch.as_tensor(np.array(p['ls']), device=device)
                    p_amp = torch.as_tensor(np.array(p['amp']), device=device)

                    # Compute Mean manually
                    k_tr_p = matern52_kernel_torch(curr_in_train, p_tx.expand(S, -1, -1), p_ls, p_amp)
                    train_p_m = k_tr_p @ p_w.expand(S, -1, -1)

                    k_te_p = matern52_kernel_torch(curr_in_test, p_tx.expand(S * B, -1, -1), p_ls, p_amp)
                    test_p_m = k_te_p @ p_w.expand(S * B, -1, -1)
                else:
                    slope = self.samples[f'L{layer_idx}_slope'][:, d_idx, :]
                    bias = self.samples[f'L{layer_idx}_bias'][:, d_idx].view(S, 1, 1)

                    train_p_m = (curr_in_train * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias

                    # Repeat slope/bias for batch dimension B
                    slope_rep = slope.repeat_interleave(B, dim=0)
                    bias_rep = bias.repeat_interleave(B, dim=0)
                    test_p_m = (curr_in_test * slope_rep.unsqueeze(1)).sum(-1, keepdim=True) + bias_rep

                # --- Prediction (Standard GP Equations) ---
                # Solve: K_rr^-1 (h_train - mean_train)
                weights = torch.cholesky_solve(h_train_samples - train_p_m, L_rr)

                # Expand params for Batch B
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

                # Sample from posterior for this layer
                # We need one sample per chain/candidate to propagate uncertainty
                dist = torch.distributions.MultivariateNormal(mu_test.squeeze(-1), cov_test)
                next_h_test_list.append(dist.rsample().unsqueeze(-1))

            # Stack for next layer
            curr_h_train = torch.cat(next_h_train_list, dim=2)
            curr_h_test = torch.cat(next_h_test_list, dim=2)

        # --- Output Layer (Standard GP) ---
        # Fixed: Uses key -1
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

        # Output Mean Prior
        out_priors = self.mean_params_list[-1]
        has_out_prior = out_priors is not None and len(out_priors) > 0 and out_priors[0] is not None

        if has_out_prior:
            p = out_priors[0]
            p_tx = torch.as_tensor(np.array(p['train_x']), device=device)
            p_w = torch.as_tensor(np.array(p['weights']), device=device)
            p_ls = torch.as_tensor(np.array(p['ls']), device=device)
            p_amp = torch.as_tensor(np.array(p['amp']), device=device)

            k_tr_p = matern52_kernel_torch(final_in_train, p_tx.expand(S, -1, -1), p_ls, p_amp)
            train_p_m = k_tr_p @ p_w.expand(S, -1, -1)

            k_te_p = matern52_kernel_torch(final_in_test, p_tx.expand(S * B, -1, -1), p_ls, p_amp)
            test_p_m = k_te_p @ p_w.expand(S * B, -1, -1)
        else:
            train_p_m = torch.zeros(S, N_train, 1, device=device)
            test_p_m = torch.zeros(S * B, N_test, 1, device=device)

        weights = torch.cholesky_solve(self.train_y.to(device).view(1, N_train, 1).expand(S, -1, -1) - train_p_m, L_rr)

        # Prepare for Batched Prediction
        train_in_rep = final_in_train.repeat_interleave(B, dim=0)
        ls_out_rep = ls_out.repeat_interleave(B, dim=0)
        amp_out_rep = amp_out.repeat_interleave(B, dim=0)
        weights_rep = weights.repeat_interleave(B, dim=0)

        K_tr = matern52_kernel_torch(final_in_test, train_in_rep, ls_out_rep, amp_out_rep)
        y_pred = test_p_m + K_tr @ weights_rep

        # Return shape: (Samples, Batch, N_test, 1)
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

        mu = mu.squeeze(-1)  # (B, N)
        # var = var.squeeze(-1)  # (B, N)
        # cov = torch.diag_embed(var + 1e-6)

        diff = preds_samples.squeeze(-1).permute(1, 2, 0) - mu.unsqueeze(-1)  # (Batch, N, Samples)
        num_samples = diff.shape[-1]
        cov = (diff @ diff.transpose(-1, -2)) / (num_samples - 1)

        jitter = JITTER * torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0).expand(cov.shape[0], -1, -1)
        cov = cov + jitter

        return GPyTorchPosterior(MultivariateNormal(mu, cov))

    @property
    def num_outputs(self): return self._num_outputs


# ==========================================
# 3. Main with BoTorch & Benchmark Logic
# ==========================================

def train_botorch_priors(bench, train_x_raw, x_mean, x_std):
    print("--- Training BoTorch Priors ---")
    all_steps = bench.observe_all_steps(train_x_raw)
    x_train_std = (train_x_raw - x_mean) / x_std

    p1_params = []

    # Prior 1: Stage 1
    gp_y1 = SingleTaskGP(x_train_std, (all_steps[0] - all_steps[0].mean()) / all_steps[0].std())
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y1.likelihood, gp_y1))
    p1_params.append(extract_gp_params_for_jax(gp_y1))

    gp_y2 = SingleTaskGP(x_train_std, (all_steps[1] - all_steps[1].mean()) / all_steps[1].std())
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y2.likelihood, gp_y2))
    p1_params.append(extract_gp_params_for_jax(gp_y2))

    # Prior 2: Stage 2
    n_synth = 400
    synth_latents = torch.rand(n_synth, 2).double()
    y_synth = torch.tensor(
        bench.step_3_reaction(synth_latents[:, 0].numpy(), synth_latents[:, 1].numpy(), mode='HF')).unsqueeze(-1)

    synth_lat_std = (synth_latents - 0.5) / 0.288
    y_synth_std = (y_synth - y_synth.mean()) / (y_synth.std() + 1e-6)

    gp_y3 = SingleTaskGP(synth_lat_std, y_synth_std)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_y3.likelihood, gp_y3))

    synth_mean = synth_latents.mean(0)
    synth_std = synth_latents.std(0)

    p2_params = [{
        **extract_gp_params_for_jax(gp_y3),
        'input_mean': jnp.array(synth_mean.numpy()),
        'input_std': jnp.array(synth_std.numpy())
    }]

    return [p1_params, p2_params]


def visualize_priors(bench, p1_params, p2_params, x_mean, x_std):
    import torch
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Helper: Compute GP Std Dev
    # Re-computes K_ff to get the inverse term needed for variance: V = K_ss - K_sf * K_ff^-1 * K_fs
    def get_gp_std(test_x, train_x, ls, amp, noise=1e-5):
        # 1. Compute Kernel(Train, Train)
        # Note: We assume matern52_kernel_torch is available in scope
        K_ff = matern52_kernel_torch(train_x.unsqueeze(0),
                                     train_x.unsqueeze(0),
                                     ls, amp).squeeze(0)

        # 2. Add jitter/noise for numerical stability
        L = torch.linalg.cholesky(K_ff + torch.eye(K_ff.shape[0]) * noise)

        # 3. Compute Kernel(Train, Test)
        K_fs = matern52_kernel_torch(train_x.unsqueeze(0),
                                     test_x.unsqueeze(0),
                                     ls, amp).squeeze(0)

        # 4. Solve for v = L^-1 * K_fs
        # Variance = Amp - v.T * v
        v = torch.linalg.solve_triangular(L, K_fs, upper=False)
        predictive_var = amp - (v ** 2).sum(dim=0)

        # Clip negative variance due to numerical precision
        predictive_var = torch.clamp(predictive_var, min=1e-9)
        return torch.sqrt(predictive_var)

    # 1. Setup Grid
    n_grid = 30
    grid_pts = torch.linspace(0, 1, n_grid)
    X, Y = torch.meshgrid(grid_pts, grid_pts, indexing='ij')
    flat_2d = torch.stack([X.flatten(), Y.flatten()], dim=1)

    z_data = {}

    # --- PRE-CALCULATION PHASE ---

    # A. Rows 1 & 2 (Intermediate Steps 1 & 2)
    test_x_raw = torch.cat([flat_2d, torch.full((n_grid ** 2, 1), 0.5)], dim=1)
    test_x_norm = (test_x_raw - x_mean) / x_std
    all_steps_truth = bench.observe_all_steps(test_x_raw)

    for i in range(2):
        row_key = f'row{i + 1}'
        # Truth
        z_data[f'{row_key}_truth'] = all_steps_truth[i].reshape(n_grid, n_grid).numpy()

        # GP Parameters
        p = p1_params[i]
        train_x_t = torch.tensor(p['train_x'])
        ls_t = torch.tensor(p['ls'])
        amp_t = torch.tensor(p['amp'])

        # GP Mean
        k_star = matern52_kernel_torch(test_x_norm.unsqueeze(0),
                                       train_x_t.unsqueeze(0),
                                       ls_t, amp_t)
        gp_pred = (k_star.squeeze(0) @ torch.tensor(p['weights'])).reshape(n_grid, n_grid).numpy()
        z_data[f'{row_key}_gp'] = gp_pred

        # GP Uncertainty
        gp_std = get_gp_std(test_x_norm, train_x_t, ls_t, amp_t)
        z_data[f'{row_key}_std'] = gp_std.reshape(n_grid, n_grid).numpy()

    # B. Row 3 (Final Step, Latent Space)
    z_row3_truth = np.array([bench.step_3_reaction(pt[0], pt[1], mode='HF') for pt in flat_2d.numpy()])
    z_data['row3_truth'] = z_row3_truth.reshape(n_grid, n_grid)

    p_last = p2_params[0]
    test_lat_norm = (flat_2d - 0.5) / 0.288
    train_x_last = torch.tensor(p_last['train_x'])
    ls_last = torch.tensor(p_last['ls'])
    amp_last = torch.tensor(p_last['amp'])

    # GP Mean
    k_star_out = matern52_kernel_torch(test_lat_norm.unsqueeze(0),
                                       train_x_last.unsqueeze(0),
                                       ls_last, amp_last)
    gp_pred_out = (k_star_out.squeeze(0) @ torch.tensor(p_last['weights'])).reshape(n_grid, n_grid).numpy()
    z_data['row3_gp'] = gp_pred_out

    # GP Uncertainty
    gp_std_out = get_gp_std(test_lat_norm, train_x_last, ls_last, amp_last)
    z_data['row3_std'] = gp_std_out.reshape(n_grid, n_grid).numpy()

    # --- PLOTTING PHASE ---
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Step 1: Truth", "Step 1: GP Mean", "Step 1: Uncertainty (Std)",
            "Step 2: Truth", "Step 2: GP Mean", "Step 2: Uncertainty (Std)",
            "Step 3: Truth", "Step 3: GP Mean", "Step 3: Uncertainty (Std)"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # 1. Value Settings (Truth & Mean) - Fixed [0, 1]
    heatmap_val_args = dict(
        x=grid_pts.numpy(), y=grid_pts.numpy(),
        zmin=0.0, zmax=1.0,
        colorscale='Viridis',
        contours=dict(coloring='heatmap', showlines=False),
        showscale=False
    )

    # 2. Uncertainty Settings - Auto-scaled, distinct color
    heatmap_std_args = dict(
        x=grid_pts.numpy(), y=grid_pts.numpy(),
        colorscale='Plasma',  # Distinct from Viridis
        contours=dict(coloring='heatmap', showlines=False),
        showscale=False
    )

    # Loop through rows
    for i in range(3):
        row_idx = i + 1
        row_key = f'row{row_idx}'

        # Column 1: Truth
        fig.add_trace(go.Contour(z=z_data[f'{row_key}_truth'].T, **heatmap_val_args), row=row_idx, col=1)

        # Column 2: GP Mean
        # Show value scale only on top-middle plot
        args_gp = heatmap_val_args.copy()
        if i == 0: args_gp['showscale'] = True
        fig.add_trace(go.Contour(z=z_data[f'{row_key}_gp'].T, **args_gp), row=row_idx, col=2)

        # Column 3: GP Uncertainty
        # Show uncertainty scale on top-right plot
        args_std = heatmap_std_args.copy()
        if i == 0: args_std['showscale'] = True
        fig.add_trace(go.Contour(z=z_data[f'{row_key}_std'].T, **args_std), row=row_idx, col=3)

        # Plot Training Points (Overlay on Mean and Uncertainty)
        # Note: We calculate coordinates differently for Row 3 vs Rows 1-2
        if i < 2:
            p = p1_params[i]
            train_pts = torch.tensor(p['train_x']) * x_std + x_mean
            targets = bench.observe_all_steps(train_pts)[i].numpy().flatten()
        else:
            p = p2_params[0]
            train_pts = torch.tensor(p['train_x']) * 0.288 + 0.5
            targets = np.array([bench.step_3_reaction(pt[0], pt[1], mode='HF') for pt in train_pts.numpy()]).flatten()

        # Add Scatter to Col 2 (Mean)
        fig.add_trace(go.Scatter(
            x=train_pts[:, 0], y=train_pts[:, 1],
            mode='markers',
            marker=dict(color='white', size=5, line=dict(color='black', width=1)),
            showlegend=False
        ), row=row_idx, col=2)

        # Add Scatter to Col 3 (Uncertainty) - Red points for contrast on Plasma
        fig.add_trace(go.Scatter(
            x=train_pts[:, 0], y=train_pts[:, 1],
            mode='markers',
            marker=dict(color='white', size=5, line=dict(color='black', width=1)),
            showlegend=False
        ), row=row_idx, col=3)

    fig.update_layout(
        height=1200, width=1400,
        title_text="Priors: Truth vs Mean [0, 1] vs Uncertainty (Auto-scale)"
    )
    fig.show()


def visualize_optimization_state(dgp, log_ei, bench, train_x_raw, candidate_raw, x_mean, x_std, y_mean, y_std, device):
    print("Generating 4-Column Slice Plots (Truth | Mean | Std | AF)...")

    temps = [0.0, 0.25, 0.5, 0.75, 1.0]
    step = 0.25
    n_grid = 30

    # Setup Grid
    x_range = torch.linspace(0, 1, n_grid)
    y_range = torch.linspace(0, 1, n_grid)
    X_grid, Y_grid = torch.meshgrid(x_range, y_range, indexing='ij')
    flat_grid = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)

    titles = []
    for t in temps:
        titles.extend([
            f"Truth (T={t})",
            f"Post. Mean (T={t})",
            f"Post. Std (T={t})",
            f"Acq. Func (T={t})"
        ])

    fig = make_subplots(
        rows=5, cols=4,
        subplot_titles=titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.03
    )

    # Base heatmap settings
    heatmap_base = dict(
        x=x_range.numpy(), y=y_range.numpy(),
        contours=dict(coloring='heatmap', showlines=False),
        showscale=False
    )

    for i, t_val in enumerate(temps):
        row_idx = i + 1

        # 1. Prepare Data
        fixed_dim = torch.full((flat_grid.shape[0], 1), t_val)
        test_x_raw = torch.cat([flat_grid, fixed_dim], dim=1)
        test_x_norm = (test_x_raw - x_mean.cpu()) / x_std.cpu()

        # 2. Predictions
        mu_norm, var_norm = dgp.predict(test_x_norm)

        if isinstance(mu_norm, torch.Tensor): mu_norm = mu_norm.detach().cpu().numpy()
        if isinstance(var_norm, torch.Tensor): var_norm = var_norm.detach().cpu().numpy()

        mu_raw = mu_norm * y_std.item() + y_mean.item()

        if var_norm.ndim > 1:
            if var_norm.shape[0] == var_norm.shape[1] and var_norm.shape[0] > 1:
                var_norm = np.diag(var_norm)
            else:
                var_norm = var_norm.flatten()

        std_raw = np.sqrt(var_norm) * y_std.item()

        # 3. Truth
        true_y = bench.observe(test_x_raw).cpu().numpy().flatten()

        # 4. Acquisition
        with torch.no_grad():
            acq_vals = log_ei(test_x_norm.to(device).unsqueeze(1)).cpu()
            acq_min, acq_max = acq_vals.min(), acq_vals.max()
            if acq_max > acq_min:
                acq_viz = (acq_vals - acq_min) / (acq_max - acq_min)
            else:
                acq_viz = torch.zeros_like(acq_vals)
            acq_viz = acq_viz.numpy()

        # --- PLOTTING ---

        # Col 1: Truth [0, 1] Fixed
        fig.add_trace(go.Contour(
            z=true_y.reshape(n_grid, n_grid).T,
            zmin=0.0, zmax=1.0, colorscale='Viridis',
            **heatmap_base
        ), row=row_idx, col=1)

        # Col 2: Mean [0, 1] Fixed
        args_mean = heatmap_base.copy()
        args_mean.update(dict(zmin=0.0, zmax=1.0, colorscale='Viridis'))
        if i == 0: args_mean['showscale'] = True
        fig.add_trace(go.Contour(z=mu_raw.reshape(n_grid, n_grid).T, **args_mean), row=row_idx, col=2)

        # Col 3: Std (Auto-Scale)
        args_std = heatmap_base.copy()
        args_std['colorscale'] = 'Plasma'  # distinct for uncertainty
        if i == 0: args_std['showscale'] = True
        fig.add_trace(go.Contour(z=std_raw.reshape(n_grid, n_grid).T, **args_std), row=row_idx, col=3)

        # Col 4: AF (Auto-Scale)
        args_af = heatmap_base.copy()
        args_af['colorscale'] = 'Cividis'  # distinct for utility
        if i == 0: args_af['showscale'] = True
        fig.add_trace(go.Contour(z=acq_viz.reshape(n_grid, n_grid).T, **args_af), row=row_idx, col=4)

        # Overlays
        t_min = t_val - step / 2 if t_val > 0 else 0
        t_max = t_val + step / 2 if t_val < 1.0 else 1.0
        mask = (train_x_raw[:, 2] >= t_min) & (train_x_raw[:, 2] <= t_max)
        pts = train_x_raw[mask]

        if len(pts) > 0:
            scatter = dict(
                x=pts[:, 0], y=pts[:, 1], mode='markers',
                marker=dict(color='white', size=7, line=dict(color='black', width=1)),
                showlegend=False
            )
            for c in [2, 3, 4]: fig.add_trace(go.Scatter(**scatter), row=row_idx, col=c)

        cand_t = candidate_raw[0, 2].item()
        if t_min <= cand_t <= t_max:
            cand_config = dict(
                x=[candidate_raw[0, 0].item()], y=[candidate_raw[0, 1].item()], mode='markers',
                marker=dict(color='red', size=14, symbol='star', line=dict(color='white', width=1)),
                name='Next Query', showlegend=(i == 0)
            )
            for c in [2, 3, 4]: fig.add_trace(go.Scatter(**cand_config), row=row_idx, col=c)

    fig.update_layout(height=1800, width=1600, title_text="Optimization State")
    fig.show()


def main():
    print("Initializing PolymerizationV2 Benchmark...")
    bench = get_benchmark("PolymerizationV2")
    input_dim = 3

    n_init = 40
    train_x_raw = torch.rand(n_init, input_dim).double()
    train_x_raw = train_x_raw * (bench.bounds[1] - bench.bounds[0]).cpu() + bench.bounds[0].cpu()
    print(f"Observing {n_init} initial points...")
    train_y_raw = bench.observe(train_x_raw).squeeze(-1).cpu()

    x_mean, x_std = train_x_raw.mean(0), train_x_raw.std(0) + 1e-6
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std() + 1e-6
    train_x = (train_x_raw - x_mean) / x_std
    train_y = (train_y_raw - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)

    prior_params = train_botorch_priors(bench, train_x_raw.cpu(), x_mean.cpu(), x_std.cpu())

    visualize_priors(bench, prior_params[0], prior_params[1], x_mean, x_std)

    dgp = LightweightDeepGP(train_x, train_y,
                            hidden_dims=[2],
                            injection_map={0: [0, 1, 2], "out": []},
                            mean_params_list=prior_params)

    dgp.fit_numpyro(num_chains=1, num_samples=1000, warmup_steps=500)

    # --- Optimization (Standard) ---
    std_bounds_lower = (bench.bounds[0].cpu() - x_mean.cpu()) / x_std.cpu()
    std_bounds_upper = (bench.bounds[1].cpu() - x_mean.cpu()) / x_std.cpu()
    bounds = torch.stack([std_bounds_lower, std_bounds_upper]).to(device)

    bo_model = DeepGPBoTorchModel(dgp)
    log_ei = LogExpectedImprovement(model=bo_model, best_f=train_y.min(), maximize=False)

    print("Optimizing Acquisition Function...")
    candidate_norm, _ = optimize_acqf(log_ei, bounds=bounds, q=1, num_restarts=5, raw_samples=100)
    candidate_raw = candidate_norm.cpu() * x_std.cpu() + x_mean.cpu()
    print(f"Next Candidate: {candidate_raw.numpy()}")

    visualize_optimization_state(dgp, log_ei, bench, train_x_raw, candidate_raw, x_mean, x_std, y_mean, y_std, device)


if __name__ == "__main__":
    main()