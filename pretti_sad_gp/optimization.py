"""Structure-aligned, prior-informed deep GP and the Bayesian optimisation runners.

This module holds the method of the ESCAPE-36 contribution (poster Fig. 1):

1. A deep Gaussian process (DGP) written as a NumPyro model (`_deep_gp_model_jax`). Each
   hidden layer has one GP node per latent dimension, and process inputs are injected at
   the layer where they act. Kernel hyperparameters and the latent function values of all
   hidden nodes are sampled jointly with NUTS (`LightweightDeepGP.fit_numpyro`).
2. Low-fidelity (LF) prior means. For each node, an exact GP is fitted to cheap LF data of
   the matching process step, and its posterior mean is used as that node's prior mean
   (`train_poly_priors`, `train_pharma_priors`, `align_output_prior`). The DGP then models
   the residual between the LF model and the high-fidelity (HF) observations.
3. A PyTorch twin of the predictive equations (`LightweightDeepGP.predict_torch`) and a
   BoTorch wrapper (`DeepGPBoTorchModel`), so the NUTS draws can be used inside
   `LogExpectedImprovement`.
4. The BO loops `run_custom_deep_gp_bo` (the DGP) and `run_standard_bo` (the
   `SingleTaskGP` baseline).

`BENCH_CONFIGS` maps a benchmark name to the hidden-layer widths (`"h"`), the
input-injection map (`"inj"`, layer index -> input columns; -1 is the output layer) and
the function that trains the LF priors (`"p_fn"`).

Module-level settings:
    JITTER: Diagonal jitter added to every kernel matrix (standardised units).
    PREDICT_SEED: Seed of the fixed noise used for hidden-layer draws in
        `LightweightDeepGP.predict_torch` (common random numbers, so the acquisition
        surface is deterministic while L-BFGS optimises it).
    JAX_GPU: True if JAX has a GPU backend. NUTS then runs all chains vectorised on one
        GPU; otherwise it runs on the CPU.

Importing the module enables 64-bit floats in JAX (`jax_enable_x64`) and silences
BoTorch's `InputDataWarning`.
"""

import torch
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import warnings
import os

# BoTorch / GPyTorch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from botorch.exceptions import InputDataWarning
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

# --- CONFIG ---
warnings.filterwarnings("ignore", category=InputDataWarning)
jax.config.update("jax_enable_x64", True)
JITTER = 1e-6
# Fixed seed for the hidden-layer draws in predict_torch, so the acquisition surface is
# deterministic (common random numbers) while optimize_acqf runs L-BFGS on it.
PREDICT_SEED = 1234



def _jax_has_gpu():
    """Check whether JAX itself has a GPU backend (torch seeing a GPU is not enough).

    Returns:
        bool: True if `jax.devices("gpu")` returns at least one device.
    """
    # Ask JAX itself: torch can see a GPU while JAX is a CPU-only build (plain `pip install jax`),
    # and forcing numpyro onto "cuda" then makes every JAX op fail.
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


JAX_GPU = _jax_has_gpu()
if torch.cuda.is_available() and not JAX_GPU:
    # print, not warnings.warn: benchmarks.py silences all warnings on import
    print("WARNING: torch sees a GPU but JAX has no GPU backend; NUTS runs on CPU. "
          "Install a CUDA build of JAX (e.g. `pip install -U \"jax[cuda12]\"`).")


# ==========================================
# 1. JAX/NumPyro GP Kernels & Model
# ==========================================

def squared_dist_jax(x, z):
    """Pairwise squared Euclidean distances in JAX, clamped away from zero.

    The clamp at 1e-12 keeps the gradient of the later square root finite at zero distance.

    Args:
        x: Array of shape (N, D).
        z: Array of shape (M, D).

    Returns:
        jnp.ndarray: Shape (N, M), entry (i, j) = max(||x_i - z_j||^2, 1e-12).
    """
    x2 = jnp.sum(x ** 2, axis=-1, keepdims=True)
    z2 = jnp.sum(z ** 2, axis=-1, keepdims=True)
    dist_sq = x2 + z2.T - 2 * jnp.dot(x, z.T)
    return jnp.maximum(dist_sq, 1e-12)


def matern52_kernel_jax(x, z, lengthscale, amplitude):
    """Matern-5/2 kernel with ARD lengthscales in JAX.

    k(x, z) = amplitude * (1 + sqrt(5) d + 5/3 d^2) * exp(-sqrt(5) d), with d the Euclidean
    distance between x / lengthscale and z / lengthscale. Because the squared distance is
    clamped at 1e-12, k(x, x) equals `amplitude` only up to about 1e-6 relative error. The
    torch twin `matern52_kernel_torch` uses the same clamp.

    Args:
        x: Array of shape (N, D).
        z: Array of shape (M, D).
        lengthscale: Scalar or array of shape (D,).
        amplitude: Scalar kernel variance (k(x, x) is about `amplitude`, not its square).

    Returns:
        jnp.ndarray: Kernel matrix of shape (N, M).
    """
    x_scaled = x / lengthscale
    z_scaled = z / lengthscale
    d2 = squared_dist_jax(x_scaled, z_scaled)
    d = jnp.sqrt(d2)
    sqrt5 = jnp.sqrt(5.0)
    return amplitude * (1 + sqrt5 * d + 5.0 / 3.0 * d2) * jnp.exp(-sqrt5 * d)


def jax_gp_mean_fn(x, train_x, weights, lengthscale, amplitude, input_mean=None, input_std=None,
                   out_scale=1.0, out_offset=0.0):
    """Evaluate the posterior mean of a fitted exact GP in JAX.

    Computes out_scale * k(x', train_x) @ weights + out_offset, with
    x' = (x - input_mean) / input_std when both normalisation arguments are given and
    x' = x otherwise.

    Args:
        x: Query points, shape (N, D), in the space of the calling DGP layer.
        train_x: Training inputs of the fitted GP, shape (M, D), in its own normalised space.
        weights: Shape (M, 1), (K + noise I)^-1 (y - c), see `extract_gp_params_for_jax`.
        lengthscale: Scalar or shape (D,).
        amplitude: Scalar kernel variance.
        input_mean: Optional shape (D,), maps `x` into the GP's input space.
        input_std: Optional shape (D,), maps `x` into the GP's input space.
        out_scale: Scalar factor applied to the GP output.
        out_offset: Scalar added to the GP output.

    Returns:
        jnp.ndarray: Posterior mean, shape (N, 1).
    """
    if input_mean is not None and input_std is not None:
        x = (x - input_mean) / input_std
    k_star = matern52_kernel_jax(x, train_x, lengthscale, amplitude)
    return out_scale * jnp.dot(k_star, weights) + out_offset


def _prior_mean_jax(p, x):
    """Evaluate an LF prior-mean function given as a parameter dict, in JAX.

    Args:
        p: Prior parameters as returned by `extract_gp_params_for_jax`, possibly extended by
            `_input_affine` and `align_output_prior`. Required keys: "train_x", "weights",
            "ls", "amp". Optional keys: "input_mean", "input_std", "out_scale" (default 1),
            "out_offset" (default 0).
        x: Layer inputs, shape (N, D).

    Returns:
        jnp.ndarray: Prior mean, shape (N,).
    """
    return jax_gp_mean_fn(x, p['train_x'], p['weights'], p['ls'], p['amp'],
                          input_mean=p.get('input_mean'), input_std=p.get('input_std'),
                          out_scale=p.get('out_scale', 1.0), out_offset=p.get('out_offset', 0.0)).reshape(-1)


def _deep_gp_model_jax(x, y=None, hidden_dims=[2], standardized_map=None, mean_params_list=None):
    """NumPyro model of the structure-aligned deep GP.

    Hidden layer l has `hidden_dims[l]` independent GP nodes. The input of layer 0 is the
    injected input columns `standardized_map[0]`. Every later layer, and the output layer,
    gets the previous layer's latents with its own injected columns appended. For each
    hidden node d the model samples:

    - `L{l}_ls_{d}` ~ LogNormal(0, 1), shape (input_dim,): ARD lengthscales.
    - `L{l}_amp_{d}` ~ LogNormal(0, 1): kernel variance.
    - `L{l}_slope_{d}` ~ Normal(0, 0.5), shape (input_dim,), and `L{l}_bias_{d}` ~
      Normal(0, 0.5): a linear mean. These are sampled only when the node has no LF prior.
    - `h_{l}_{d}` ~ MVN(mean, K + JITTER I), shape (n,): noise-free latent values at the
      training inputs.

    The output layer samples `out_ls` ~ LogNormal(-1, 1) (shape (input_dim,)), `out_amp`
    ~ LogNormal(0, 1) and `obs_noise` ~ HalfNormal(0.3). `obs_noise` is a variance added
    to the diagonal. The likelihood is `obs` ~ MVN(mean_out, K + (obs_noise + JITTER) I),
    where mean_out is the output LF prior or zero.

    Args:
        x: Standardised training inputs, shape (n, D).
        y: Standardised training targets, shape (n,), or None to sample from the prior.
        hidden_dims: Width of each hidden layer, e.g. [3]. The default list is never mutated.
        standardized_map: Dict mapping a layer index (0 .. len(hidden_dims) - 1, or -1 for
            the output layer) to the list of input columns injected there. None injects all
            columns at layer 0. Use `LightweightDeepGP._process_map` to build it.
        mean_params_list: List of length len(hidden_dims) + 1. Entry l is None or a list
            with one prior-parameter dict (or None) per node of layer l. The last entry
            belongs to the output layer, and only its first element is used. None means no
            LF priors anywhere.

    Returns:
        None. The model only registers NumPyro sample sites.
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
            amp_d = numpyro.sample(f"L{layer_idx}_amp_{dim_idx}", dist.LogNormal(0.0, 1.0))

            k_mat = matern52_kernel_jax(current_input, current_input, ls_d, amp_d)
            cov = k_mat + jnp.eye(n) * JITTER

            # Mean Function Logic
            if layer_priors is not None and len(layer_priors) > dim_idx and layer_priors[dim_idx] is not None:
                mean = _prior_mean_jax(layer_priors[dim_idx], current_input)
            else:
                slope = numpyro.sample(f"L{layer_idx}_slope_{dim_idx}", dist.Normal(0.0, 0.5).expand([input_dim]))
                bias = numpyro.sample(f"L{layer_idx}_bias_{dim_idx}", dist.Normal(0.0, 0.5))
                mean = jnp.dot(current_input, slope) + bias

            h_list.append(
                numpyro.sample(f"h_{layer_idx}_{dim_idx}", dist.MultivariateNormal(mean, covariance_matrix=cov)))

        current_h = jnp.stack(h_list, axis=1)

    # --- Output Layer ---
    out_inj_dims = standardized_map.get(-1, [])
    final_input = current_h if not out_inj_dims else jnp.concatenate([current_h, x[:, out_inj_dims]], axis=1)
    input_dim = final_input.shape[1]

    ls_out = numpyro.sample("out_ls", dist.LogNormal(-1.0, 1.0).expand([input_dim]))
    amp_out = numpyro.sample("out_amp", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("obs_noise", dist.HalfNormal(0.3))

    mean_out = jnp.zeros(n)
    out_priors = mean_params_list[-1]
    if out_priors is not None and len(out_priors) > 0 and out_priors[0] is not None:
        mean_out = _prior_mean_jax(out_priors[0], final_input)

    cov_out = matern52_kernel_jax(final_input, final_input, ls_out, amp_out)
    numpyro.sample("obs", dist.MultivariateNormal(mean_out, covariance_matrix=cov_out + jnp.eye(n) * (noise + JITTER)),
                   obs=y)


# ==========================================
# 2. PyTorch / BoTorch Integration
# ==========================================

def clamped_cdist_torch(x, z):
    """Batched pairwise Euclidean distances in torch, clamped like the JAX version.

    Args:
        x: Tensor of shape (..., N, D).
        z: Tensor of shape (..., M, D). Batch dimensions broadcast against `x`.

    Returns:
        torch.Tensor: Shape (..., N, M), sqrt(max(||x_i - z_j||^2, 1e-12)).
    """
    # Support broadcasting for Batched Inputs (S, N, D)
    x_ex = x.unsqueeze(-2)  # (..., N, 1, D)
    z_ex = z.unsqueeze(-3)  # (..., 1, M, D)
    return (x_ex - z_ex).pow(2).sum(-1).clamp(min=1e-12).sqrt()


def matern52_kernel_torch(x, z, lengthscale, amplitude):
    """Batched Matern-5/2 kernel in torch, numerically matching `matern52_kernel_jax`.

    Args:
        x: Tensor of shape (..., N, D).
        z: Tensor of shape (..., M, D).
        lengthscale: Tensor that broadcasts against the last dimension of `x` and `z`, e.g.
            shape (D,) or (S, 1, D).
        amplitude: Kernel variance. A 0-d tensor, a tensor of shape (S,) (reshaped to
            (S, 1, 1)) or a tensor already of shape (S, 1, 1).

    Returns:
        torch.Tensor: Kernel matrix of shape (..., N, M).
    """
    d = clamped_cdist_torch(x / lengthscale, z / lengthscale)
    if amplitude.dim() == 1: amplitude = amplitude.view(-1, 1, 1)
    sqrt5 = np.sqrt(5.0)
    return amplitude * (1 + sqrt5 * d + 5.0 / 3.0 * d.pow(2)) * torch.exp(-sqrt5 * d)


def _solve_lower_shared(L, K_tr, S, B):
    """Solve L^-1 K_tr^T with one Cholesky factor per posterior sample, shared by B test batches.

    This avoids materialising `L.repeat_interleave(B, 0)`, which has shape (S*B, n, n).

    Args:
        L: Lower Cholesky factors of the training kernel matrices, shape (S, n, n).
        K_tr: Test-train cross-covariances, shape (S*B, N, n), ordered sample-major (all B
            batches of sample 0 first).
        S: Number of posterior samples.
        B: Number of test batches.

    Returns:
        torch.Tensor: L^-1 K_tr^T for each (sample, batch), shape (S*B, n, N).
    """
    N, n = K_tr.shape[-2:]
    rhs = K_tr.reshape(S, B * N, n).transpose(-1, -2)
    v = torch.linalg.solve_triangular(L, rhs, upper=False)
    return v.transpose(-1, -2).reshape(S * B, N, n).transpose(-1, -2)


def _prior_mean_torch(p, inp):
    """Torch twin of `_prior_mean_jax`, evaluated with batch dimensions.

    The parameter arrays in `p` are converted to tensors on the device and dtype of `inp`.

    Args:
        p: Prior parameter dict (see `_prior_mean_jax`).
        inp: Layer inputs, shape (..., N, D).

    Returns:
        torch.Tensor: Prior mean, shape (..., N, 1).
    """
    dev, dt = inp.device, inp.dtype
    t = lambda v: torch.as_tensor(np.array(v), device=dev, dtype=dt)
    x = (inp - t(p.get('input_mean', 0.))) / t(p.get('input_std', 1.))
    k = matern52_kernel_torch(x, t(p['train_x']), t(p['ls']), t(p['amp']))
    return t(p.get('out_scale', 1.)) * (k @ t(p['weights'])) + t(p.get('out_offset', 0.))


def _fit_prior_gp(x, y):
    """Fit an exact GP with the same Matern-5/2 kernel the JAX and torch mean functions use.

    BoTorch's SingleTaskGP default kernel (RBF since 0.12) and its default Standardize
    outcome transform would not match `extract_gp_params_for_jax`, so both are set here:
    a ScaleKernel(Matern-5/2, ARD) covariance and no outcome transform. Hyperparameters
    are fitted by maximising the exact marginal log-likelihood.

    Args:
        x: Normalised training inputs, shape (n, d).
        y: Standardised training targets, shape (n, 1).

    Returns:
        SingleTaskGP: The fitted model.
    """
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[-1]))
    gp = SingleTaskGP(x, y, covar_module=covar, outcome_transform=None)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def extract_gp_params_for_jax(gp_model):
    """Turn a fitted GP into a closed-form posterior-mean function usable in JAX and torch.

    The returned weights are (K + noise I)^-1 (y - c), where c is the GP's constant mean.
    Adding c back as "out_offset" makes `jax_gp_mean_fn` reproduce the GP's posterior mean.
    The posterior variance is not kept: LF uncertainty is not propagated (poster,
    Limitations).

    Args:
        gp_model: A fitted exact GP with a Matern-5/2 kernel, optionally wrapped in a
            ScaleKernel, and no input or outcome transform. Fit it with `_fit_prior_gp`.
            It is switched to eval mode.

    Returns:
        dict: JAX arrays "train_x" (n, d), "weights" (n, 1), "ls" (d,), or a scalar if
        d == 1, "amp" (scalar) and "out_offset" (scalar).

    Raises:
        ValueError: If the base kernel is not a Matern kernel with nu = 2.5.
    """
    gp_model.eval()
    with torch.no_grad():
        train_x = gp_model.train_inputs[0]
        train_y = gp_model.train_targets.unsqueeze(-1)

        # Robust Kernel Extraction
        covar = gp_model.covar_module
        if hasattr(covar, 'base_kernel'):
            base_kernel = covar.base_kernel
            amp = covar.outputscale.detach()
        else:
            base_kernel = covar
            amp = torch.tensor(1.0, device=train_x.device, dtype=train_x.dtype)
        if not (isinstance(base_kernel, MaternKernel) and base_kernel.nu == 2.5):
            raise ValueError(f"Expected a Matern-5/2 kernel, got {type(base_kernel).__name__}; fit with _fit_prior_gp")

        ls = base_kernel.lengthscale.squeeze().detach()
        noise = gp_model.likelihood.noise.detach()

        # Subtract the GP's own constant mean so the extracted function equals its posterior mean
        const = gp_model.mean_module.constant.detach() if hasattr(gp_model.mean_module, 'constant') else 0.
        K = matern52_kernel_torch(train_x, train_x, ls, amp) + torch.eye(train_x.shape[0],
                                                                         device=train_x.device) * noise
        weights = torch.linalg.solve(K, train_y - const)

    return {'train_x': jnp.array(train_x.cpu().numpy()), 'weights': jnp.array(weights.cpu().numpy()),
            'ls': jnp.array(ls.cpu().numpy()), 'amp': jnp.array(amp.cpu().numpy()),
            'out_offset': jnp.array(float(const))}


class LightweightDeepGP:
    """Structure-aligned deep GP whose parameters and hidden latents are sampled with NUTS.

    Sampling runs in NumPyro/JAX (`fit_numpyro`). Prediction runs in torch
    (`predict_torch`), so `DeepGPBoTorchModel` can feed it to BoTorch acquisition functions.
    All data are expected in standardised units: inputs scaled with the initial design's
    mean and std, targets likewise.

    Attributes:
        train_x: Standardised training inputs, torch tensor of shape (n, D).
        train_y: Standardised training targets, torch tensor of shape (n,).
        hidden_dims: Width of each hidden layer.
        injection_map: Input columns injected per layer, as passed in.
        mean_params_list: LF prior parameters per layer (see `_deep_gp_model_jax`).
        samples: Dict of posterior draws (torch tensors on the device of `train_x`, leading
            dimension S = num_chains * num_samples), or None before `fit_numpyro`.
        std_map: `injection_map` with the output-layer key normalised to -1.
    """
    def __init__(self, train_x, train_y, hidden_dims=[2], injection_map=None, mean_params_list=None):
        """Store the training data and the architecture. No sampling happens here.

        Args:
            train_x: Standardised inputs, tensor of shape (n, D).
            train_y: Standardised targets, tensor of shape (n,).
            hidden_dims: Hidden-layer widths. Only one hidden layer is used in `BENCH_CONFIGS`.
            injection_map: Dict from layer key to the list of input columns injected there. A
                layer key is a hidden-layer index, or -1, "out" or len(hidden_dims) for the
                output layer. None or empty injects every column at layer 0.
            mean_params_list: LF prior parameters, one entry per layer plus the output layer
                (see `_deep_gp_model_jax`). None or empty means no priors, so every node gets
                a sampled linear mean.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.hidden_dims = hidden_dims
        self.injection_map = injection_map if injection_map else {0: list(range(train_x.shape[1]))}
        self.mean_params_list = mean_params_list if mean_params_list else [None] * (len(hidden_dims) + 1)
        self.samples = None
        self.std_map = self._process_map()

    def _process_map(self):
        """Normalise the output-layer key of `injection_map` to -1.

        Returns:
            dict: Layer index (hidden layers) or -1 (output layer) -> list of input columns.
        """
        std = {}
        for k, v in self.injection_map.items():
            key = -1 if (k == "out" or k == -1 or k == len(self.hidden_dims)) else k
            std[key] = v
        return std

    def fit_numpyro(self, num_chains=1, num_samples=500, warmup_steps=300, rng_seed=None):
        """Draw posterior samples of all DGP parameters and hidden latents with NUTS.

        With a JAX GPU backend the chains run vectorised in one kernel on one GPU. Otherwise
        NumPyro's default chain method is used, which runs chains one after another when there
        are fewer CPU devices than chains. Draws are copied to torch tensors on the device of
        `train_x` and stored in `self.samples`. Each per-node site gets a stacked version:
        "L{l}_ls" (S, width, input_dim) and "L{l}_amp" (S, width), plus "L{l}_slope"
        (S, width, input_dim) and "L{l}_bias" (S, width) when node 0 of layer l uses a linear
        mean.

        Args:
            num_chains: Number of MCMC chains.
            num_samples: Post-warm-up draws per chain.
            warmup_steps: Warm-up (adaptation) steps per chain.
            rng_seed: Seed for `jax.random.PRNGKey`. None uses int(time.time()), which is not
                reproducible.
        """
        nuts_kernel = NUTS(_deep_gp_model_jax)
        # Force vectorized chain method for high throughput on single GPU
        if JAX_GPU:
            mcmc = MCMC(nuts_kernel, num_warmup=warmup_steps, num_samples=num_samples,
                        num_chains=num_chains, chain_method="vectorized", progress_bar=False)
        else:
            mcmc = MCMC(nuts_kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=False)

        if rng_seed is None:
            rng_seed = int(time.time())
        mcmc.run(jax.random.PRNGKey(rng_seed),
                 jnp.array(self.train_x.cpu().numpy()),
                 jnp.array(self.train_y.cpu().numpy()),
                 hidden_dims=self.hidden_dims,
                 standardized_map=self.std_map,
                 mean_params_list=self.mean_params_list)

        dev = self.train_x.device
        self.samples = {k: torch.tensor(np.array(v), device=dev) for k, v in mcmc.get_samples().items()}

        for l_idx, w in enumerate(self.hidden_dims):
            self.samples[f"L{l_idx}_ls"] = torch.stack([self.samples[f"L{l_idx}_ls_{d}"] for d in range(w)], dim=1)
            self.samples[f"L{l_idx}_amp"] = torch.stack([self.samples[f"L{l_idx}_amp_{d}"] for d in range(w)], dim=1)
            if f"L{l_idx}_slope_0" in self.samples:
                self.samples[f"L{l_idx}_slope"] = torch.stack([self.samples[f"L{l_idx}_slope_{d}"] for d in range(w)],
                                                              dim=1)
                self.samples[f"L{l_idx}_bias"] = torch.stack([self.samples[f"L{l_idx}_bias_{d}"] for d in range(w)],
                                                             dim=1)

    def predict_torch(self, X, return_cov=False):
        """Per-sample predictive distribution of the latent output f at test inputs.

        For each posterior draw s and each hidden node, the node's GP is conditioned on the
        drawn noise-free latents at the training inputs. One reparameterised draw is then taken
        at the test inputs, with noise from a generator seeded with `PREDICT_SEED`. Hidden-layer
        uncertainty is therefore represented by the S posterior draws, not marginalised
        analytically. The output GP is conditioned on the training targets with the drawn
        observation noise. Its predictive mean and, optionally, covariance are returned without
        observation noise.

        Args:
            X: Standardised test inputs, shape (B, N, D). A 2-D input is not supported.
            return_cov: If True, also return the output layer's predictive covariance.

        Returns:
            torch.Tensor or tuple: The mean of shape (S, B, N, 1). With `return_cov`, the
            tuple (mean, cov), where cov has shape (S, B, N, N). Units are standardised targets.

        Raises:
            TypeError: If called before `fit_numpyro` (`samples` is None).
            torch.linalg.LinAlgError: If a kernel matrix is not positive definite.
        """
        B, N_test, _ = X.shape
        S = self.samples['out_ls'].shape[0]
        dev = X.device
        gen = torch.Generator(device=dev).manual_seed(PREDICT_SEED)

        curr_h = None

        for l_idx, width in enumerate(self.hidden_dims):
            inj = self.std_map.get(l_idx, [])
            x_inj_tr = self.train_x[:, inj].unsqueeze(0).expand(S, -1, -1)
            x_inj_te = X[:, :, inj].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)

            if l_idx == 0:
                curr_in_tr = x_inj_tr
                curr_in_te = x_inj_te
            else:
                curr_in_tr = torch.cat([curr_h, x_inj_tr], -1)
                curr_in_te = torch.cat([curr_in_te_prev_h, x_inj_te], -1)

            h_tr_next, h_te_next = [], []
            priors = self.mean_params_list[l_idx]

            for d in range(width):
                ls = self.samples[f'L{l_idx}_ls'][:, d, :].unsqueeze(1)
                amp = self.samples[f'L{l_idx}_amp'][:, d].view(S, 1, 1)
                h_tr = self.samples[f'h_{l_idx}_{d}'].unsqueeze(-1)
                h_tr_next.append(h_tr)

                K_rr = matern52_kernel_torch(curr_in_tr, curr_in_tr, ls, amp) + torch.eye(curr_in_tr.shape[1],
                                                                                          device=dev) * JITTER

                # Mean
                if priors and len(priors) > d and priors[d]:
                    tr_mean = _prior_mean_torch(priors[d], curr_in_tr)
                    te_mean = _prior_mean_torch(priors[d], curr_in_te)
                else:
                    sl = self.samples[f'L{l_idx}_slope'][:, d, :]
                    bi = self.samples[f'L{l_idx}_bias'][:, d].view(S, 1, 1)
                    tr_mean = (curr_in_tr * sl.unsqueeze(1)).sum(-1, keepdim=True) + bi
                    te_mean = (curr_in_te * sl.repeat_interleave(B, 0).unsqueeze(1)).sum(-1,
                                                                                         keepdim=True) + bi.repeat_interleave(
                        B, 0)

                weights = torch.linalg.solve(K_rr, h_tr - tr_mean)

                # Prep Test Kernels
                ls_rep = ls.repeat_interleave(B, 0)
                amp_rep = amp.repeat_interleave(B, 0)
                tr_in_rep = curr_in_tr.repeat_interleave(B, 0)
                w_rep = weights.repeat_interleave(B, 0)

                K_tr = matern52_kernel_torch(curr_in_te, tr_in_rep, ls_rep, amp_rep)
                K_tt = matern52_kernel_torch(curr_in_te, curr_in_te, ls_rep, amp_rep) + torch.eye(N_test,
                                                                                                  device=dev) * JITTER

                mu = te_mean + K_tr @ w_rep
                v = _solve_lower_shared(torch.linalg.cholesky(K_rr), K_tr, S, B)
                var = K_tt - v.transpose(-1, -2) @ v

                # Reparameterised draw with fixed noise (see PREDICT_SEED)
                L_te = torch.linalg.cholesky(var + torch.eye(N_test, device=dev) * JITTER)
                eps = torch.randn(mu.shape, generator=gen, device=dev, dtype=mu.dtype)
                h_te_next.append(mu + L_te @ eps)

            curr_h = torch.cat(h_tr_next, 2)
            curr_in_te_prev_h = torch.cat(h_te_next, 2)

        # Final Layer
        inj_out = self.std_map.get(-1, [])
        fin_tr = torch.cat([curr_h, self.train_x[:, inj_out].unsqueeze(0).expand(S, -1, -1)], -1) if inj_out else curr_h
        fin_te = torch.cat(
            [curr_in_te_prev_h, X[:, :, inj_out].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B, N_test, -1)],
            -1) if inj_out else curr_in_te_prev_h

        ls_out = self.samples['out_ls'].unsqueeze(1)
        amp_out = self.samples['out_amp'].view(S, 1, 1)
        noise = self.samples['obs_noise'].view(S, 1, 1)

        K_rr = matern52_kernel_torch(fin_tr, fin_tr, ls_out, amp_out) + torch.eye(fin_tr.shape[1], device=dev) * (
                    noise + JITTER)

        tr_mean = torch.zeros(S, fin_tr.shape[1], 1, device=dev)
        te_mean = torch.zeros(S * B, N_test, 1, device=dev)

        p = self.mean_params_list[-1]
        if p and p[0]:
            tr_mean = _prior_mean_torch(p[0], fin_tr)
            te_mean = _prior_mean_torch(p[0], fin_te)

        weights = torch.linalg.solve(K_rr, self.train_y.view(1, -1, 1).expand(S, -1, -1) - tr_mean)

        ls_rep = ls_out.repeat_interleave(B, 0)
        amp_rep = amp_out.repeat_interleave(B, 0)
        fin_tr_rep = fin_tr.repeat_interleave(B, 0)
        w_rep = weights.repeat_interleave(B, 0)

        K_tr = matern52_kernel_torch(fin_te, fin_tr_rep, ls_rep, amp_rep)
        mean = (te_mean + K_tr @ w_rep).view(S, B, N_test, 1)
        if not return_cov:
            return mean

        v = _solve_lower_shared(torch.linalg.cholesky(K_rr), K_tr, S, B)
        K_tt = matern52_kernel_torch(fin_te, fin_te, ls_rep, amp_rep)
        cov = (K_tt - v.transpose(-1, -2) @ v).view(S, B, N_test, N_test)
        return mean, cov


class DeepGPBoTorchModel(Model):
    """BoTorch `Model` wrapper around a fitted `LightweightDeepGP`.

    The posterior is the Gaussian that matches the first two moments of the mixture over
    the S posterior draws, so the analytic `LogExpectedImprovement` can be used.

    Attributes:
        dgp: The wrapped, already fitted `LightweightDeepGP`.
    """
    def __init__(self, dgp):
        """Wrap a fitted deep GP.

        Args:
            dgp: A `LightweightDeepGP` whose `fit_numpyro` has already been called.
        """
        super().__init__()
        self.dgp = dgp
        self._num_outputs = 1

    def posterior(self, X, **kwargs):
        """Moment-matched Gaussian posterior over the S posterior draws.

        By the law of total covariance, Cov = E_s[Cov_s] + Cov_s[E_s], so the output GP's own
        uncertainty is kept alongside the spread between draws. The between-draw term uses
        the unbiased (S - 1) normalisation. Observation noise is not added.

        Args:
            X: Standardised inputs, shape (B, N, D). With BoTorch's analytic acquisition
                functions and q = 1, N is 1.
            **kwargs: Ignored, including `observation_noise` and `posterior_transform`.

        Returns:
            GPyTorchPosterior: A MultivariateNormal with mean of shape (B, N) and covariance of
            shape (B, N, N), plus JITTER on the diagonal, in standardised target units.
        """
        # Moment-matched mixture over posterior samples (law of total covariance):
        # E_s[Cov_s] + Cov_s[E_s], so the output GP's own uncertainty is kept.
        m, c = self.dgp.predict_torch(X, return_cov=True)  # (S, B, N, 1), (S, B, N, N)
        s = m.squeeze(-1)
        mu = s.mean(0)
        diff = s.permute(1, 2, 0) - mu.unsqueeze(-1)
        cov = c.mean(0) + (diff @ diff.transpose(-1, -2)) / max(s.shape[0] - 1, 1)
        return GPyTorchPosterior(MultivariateNormal(mu, cov + torch.eye(cov.shape[-1], device=cov.device) * JITTER))

    @property
    def num_outputs(self):
        """Number of outputs of the model.

        Returns:
            int: Always 1.
        """
        return 1


# ==========================================
# 3. Prior Training & Helpers
# ==========================================
#
# Each prior GP is fitted in its own normalised space. The `input_mean`/`input_std` and
# `out_scale`/`out_offset` entries map the space the DGP layer actually works in onto it:
#  - hidden latents h live in the prior's standardised target space of each step,
#  - injected inputs are standardised with the HF design's x_mean/x_std,
#  - the output layer models the HF target standardised with y_mean/y_std
#    (set later by `align_output_prior`).

def _std_stats(t):
    """Column-wise mean and standard deviation.

    Args:
        t: Tensor of shape (n, k).

    Returns:
        tuple: The mean of shape (k,) and the unbiased std of shape (k,), plus 1e-6.
    """
    return t.mean(0), t.std(0) + 1e-6


def _input_affine(fit_mean, fit_std, src_scale, src_shift):
    """Map inputs from a DGP layer's space into the normalised space a prior was fitted in.

    The prior was fitted on (u - fit_mean) / fit_std, but the layer feeds it z with
    u = z * src_scale + src_shift. The returned `input_mean` and `input_std` satisfy
    (z - input_mean) / input_std == (u - fit_mean) / fit_std.

    Args:
        fit_mean: Mean used to normalise the prior's training inputs, shape (k,).
        fit_std: Std used to normalise the prior's training inputs, shape (k,).
        src_scale: Scale from layer space z to raw space u, shape (k,).
        src_shift: Shift from layer space z to raw space u, shape (k,).

    Returns:
        dict: "input_mean" and "input_std", JAX arrays of shape (k,).
    """
    return {'input_mean': jnp.array(((fit_mean - src_shift) / src_scale).cpu().numpy()),
            'input_std': jnp.array((fit_std / src_scale).cpu().numpy())}


def _fit_target_prior(x_n, y):
    """Fit an LF prior GP on standardised targets.

    Args:
        x_n: Normalised inputs, shape (n, d).
        y: Raw LF targets, shape (n, 1).

    Returns:
        tuple: The prior parameter dict (see `extract_gp_params_for_jax`) and the 0-d
        tensors (y_mean, y_std) used to standardise `y`.
    """
    y_m, y_s = _std_stats(y)
    gp = _fit_prior_gp(x_n, (y - y_m) / y_s)
    return extract_gp_params_for_jax(gp), (y_m.squeeze(), y_s.squeeze())


def align_output_prior(priors, y_mean, y_std):
    """Express the output prior, fitted on LF-standardised yield, in HF-standardised units.

    The output prior predicts f_lf = (yield - lf_m) / lf_s, but the output layer models
    (yield - y_mean) / y_std. This sets out_scale = lf_s / y_std and
    out_offset = out_offset * out_scale + (lf_m - y_mean) / y_std.

    Args:
        priors: Prior list from a `p_fn` in `BENCH_CONFIGS`. The dict `priors[-1][0]` is
            modified in place, and its "_target_stats" entry is removed.
        y_mean: Mean of the HF initial-design targets (0-d tensor).
        y_std: Std of the HF initial-design targets (0-d tensor).

    Raises:
        KeyError: If the output prior has no "_target_stats", e.g. when called twice.
    """
    p = priors[-1][0]
    lf_m, lf_s = p.pop('_target_stats')
    scale = float(lf_s / y_std)
    p['out_offset'] = jnp.array(float(p.get('out_offset', 0.)) * scale + float((lf_m - y_mean) / y_std))
    p['out_scale'] = jnp.array(scale)


def train_poly_priors(bench, train_x_raw, x_mean, x_std, seed=0):
    """Fit the LF priors of the 3-D polymerisation benchmarks (both versions).

    Hidden layer (two nodes): 200 inputs drawn uniformly in the bounds (seeded CPU
    generator) and evaluated with the benchmark's LF mode. One prior is fitted per
    intermediate: effective catalyst concentration C_eff (step 1) and system temperature
    T_sys (step 2). Inputs are standardised with the HF design's statistics. Each target is
    standardised on its own, and that standardised space is the space the hidden latents
    live in.

    Output layer: 400 latent pairs (C_eff, T_sys) drawn uniformly in [0, 1]^2 (NumPy
    generator) and passed through the LF reaction step. The prior is fitted on
    self-standardised latents, and `_input_affine` maps the hidden latents onto that space.

    Args:
        bench: A `DiscontinuousPolymerization` or `DiscontinuousPolymerizationV2` instance.
        train_x_raw: HF initial design. Only its device is used.
        x_mean: Mean of the HF initial design, shape (3,).
        x_std: Std of the HF initial design, shape (3,).
        seed: Seed of the LF sample draws.

    Returns:
        tuple: (priors, prior_data). priors is [p1, p2], where p1 holds two hidden-node
        dicts and p2 one output dict that still carries "_target_stats" (see
        `align_output_prior`). prior_data holds CPU tensors for later inspection: "type",
        "sx" (200, 3), "steps" (three (200, 1) tensors), "lat" (400, 2) and "y_l" (400, 1).
    """
    print("Training Poly Priors (LF)...")
    device = train_x_raw.device
    bounds = bench.bounds.to(device)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    rng = np.random.default_rng(seed)

    # 200 points for memory safety with high sample count
    sx = (torch.rand(200, 3, generator=gen).double().to(device) * (bounds[1] - bounds[0]) + bounds[0])
    steps = bench.observe_all_steps(sx, mode='LF')
    sx_std = (sx - x_mean) / x_std

    p1, h_stats = [], []
    for i in range(2):
        p, st = _fit_target_prior(sx_std, steps[i])
        p1.append(p)
        h_stats.append(st)
    h_m = torch.stack([m for m, _ in h_stats])
    h_s = torch.stack([s for _, s in h_stats])

    lat_np = rng.random((400, 2))
    y_l_np = bench.step_3_reaction(lat_np[:, 0], lat_np[:, 1], mode='LF')

    lat = torch.tensor(lat_np, device=device)
    y_l = torch.tensor(y_l_np, device=device).unsqueeze(-1)

    lat_m, lat_s = _std_stats(lat)
    p_out, y_stats = _fit_target_prior((lat - lat_m) / lat_s, y_l)
    p2 = [{**p_out, **_input_affine(lat_m, lat_s, h_s, h_m), '_target_stats': y_stats}]

    prior_data = {'type': 'Polymerization', 'sx': sx.cpu(), 'steps': [s.cpu() for s in steps], 'lat': lat.cpu(),
                  'y_l': y_l.cpu()}
    return [p1, p2], prior_data


def train_pharma_priors(bench, train_x_raw, x_mean, x_std, seed=0):
    """Fit the LF priors of the 5-D pharma benchmarks (both versions).

    Hidden layer (three nodes): it covers activation and reactor, with input columns
    [metal, ligand, temperature, residence time] = x[:, [0, 1, 2, 4]]. 200 LF samples are
    drawn uniformly in the bounds (seeded CPU generator). One prior is fitted per reactor
    outlet concentration: product, degradant and side product.

    Output layer: it covers nanofiltration and crystallisation, with inputs [the three
    reactor outlets, nanofiltration pressure x[:, 3]] and the final LF yield as target.
    `_input_affine` maps [hidden latents, standardised pressure] onto the prior's normalised
    input space.

    Args:
        bench: A `ComplexPharmaFlowsheet` or `ComplexPharmaFlowsheetV2` instance.
        train_x_raw: HF initial design. Only its device is used.
        x_mean: Mean of the HF initial design, shape (5,).
        x_std: Std of the HF initial design, shape (5,).
        seed: Seed of the LF sample draws.

    Returns:
        tuple: (priors, prior_data). priors is [p1, p2], where p1 holds three hidden-node
        dicts and p2 one output dict with "_target_stats". prior_data holds CPU tensors:
        "type", "sx" (200, 5) and "steps" (shapes (200, 3), (200, 3), (200, 3), (200, 1)).
    """
    print("Training Pharma Priors (LF)...")
    device = train_x_raw.device
    bounds = bench.bounds.to(device)
    gen = torch.Generator(device='cpu').manual_seed(seed)

    # 200 points for memory safety
    sx = (torch.rand(200, 5, generator=gen).double().to(device) * (bounds[1] - bounds[0]) + bounds[0])
    steps = bench.observe_all_steps(sx, mode='LF')

    idx = [0, 1, 2, 4]
    x_l1 = (sx[:, idx] - x_mean[idx]) / x_std[idx]
    y_r = steps[1]

    p1, h_stats = [], []
    for i in range(3):
        p, st = _fit_target_prior(x_l1, y_r[:, i:i + 1])
        p1.append(p)
        h_stats.append(st)
    h_m = torch.stack([m for m, _ in h_stats])
    h_s = torch.stack([s for _, s in h_stats])

    # Output layer input = [h (3 latents), standardised pressure x[:, 3]]
    x_l2 = torch.cat([y_r, sx[:, 3:4]], 1)
    x_l2_m, x_l2_s = _std_stats(x_l2)
    p_out, y_stats = _fit_target_prior((x_l2 - x_l2_m) / x_l2_s, steps[3])
    src_scale = torch.cat([h_s, x_std[3:4]])
    src_shift = torch.cat([h_m, x_mean[3:4]])
    p2 = [{**p_out, **_input_affine(x_l2_m, x_l2_s, src_scale, src_shift), '_target_stats': y_stats}]

    prior_data = {'type': 'Pharma', 'sx': sx.cpu(), 'steps': [s.cpu() for s in steps]}
    return [p1, p2], prior_data


BENCH_CONFIGS = {
    "Polymerization": {"h": [2], "inj": {0: [0, 1, 2], -1: []}, "p_fn": train_poly_priors},
    "PolymerizationV2": {"h": [2], "inj": {0: [0, 1, 2], -1: []}, "p_fn": train_poly_priors},
    "PharmaV1": {"h": [3], "inj": {0: [0, 1, 2, 4], -1: [3]}, "p_fn": train_pharma_priors},
    "PharmaV2": {"h": [3], "inj": {0: [0, 1, 2, 4], -1: [3]}, "p_fn": train_pharma_priors},
}


# ==========================================
# 4. Runners
# ==========================================

def _seed_everything(seed):
    """Seed torch (all devices) and NumPy's global RNG.

    The global NumPy RNG drives the HF noise of the pharma benchmarks. JAX is seeded
    separately through explicit PRNG keys.

    Args:
        seed: Integer seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # benchmark HF noise uses the global numpy RNG


def _iter_seed(seed, i):
    """Derive the seed of BO iteration `i` of run `seed`.

    Resumed runs reseed each iteration with this value, so they continue exactly like an
    uninterrupted run. Prior fitting uses i = -1.

    Args:
        seed: Run seed.
        i: Iteration index (0-based), or -1 for prior fitting.

    Returns:
        int: seed * 100003 + i + 1.
    """
    return seed * 100003 + i + 1


def _initial_design(bench, n_init, seed):
    """Uniform random initial design inside the benchmark bounds.

    The design depends only on `seed`, so both methods start from the same points. The
    stored CustomDeepGP runs predate this and used a different design.

    Args:
        bench: Benchmark with `bounds` (2, D) and `input_dim`.
        n_init: Number of points.
        seed: Seed of the design.

    Returns:
        torch.Tensor: Shape (n_init, D), on the device of `bench.bounds`, in raw input units.
    """
    # Drawn on the CPU generator so it does not depend on the device: StandardGP runs on CPU
    # workers and CustomDeepGP on GPU workers, and the CUDA RNG gives a different design.
    # float32 draw cast to double: the same numbers the stored StandardGP runs started from
    u = torch.rand(n_init, bench.input_dim, generator=torch.Generator().manual_seed(seed)).double()
    lo, hi = bench.bounds[0], bench.bounds[1]
    return lo + (hi - lo) * u.to(bench.bounds.device)


def _history_xy(history, device):
    """Rebuild training tensors from a stored run history.

    Args:
        history: List of history dicts with keys "x" (D,) and "y" (one element).
        device: Target device.

    Returns:
        tuple: x of shape (n, D) and y of shape (n, 1), float64 on `device`.
    """
    x = torch.stack([torch.as_tensor(h['x']) for h in history]).to(device).double()
    y = torch.stack([torch.as_tensor(h['y']).reshape(()) for h in history]).to(device).double().unsqueeze(-1)
    return x, y


def run_standard_bo(bench, n_init=10, n_iter=40, seed=0, existing_data=None, checkpoint=None):
    """Black-box BO baseline with a BoTorch SingleTaskGP.

    Each iteration fits a SingleTaskGP (BoTorch default kernel, Normalize input and
    Standardize outcome transforms) to all HF data and maximises LogExpectedImprovement with
    q = 1, 5 restarts and 64 raw samples. best_f is the best noisy observation so far.

    Args:
        bench: Benchmark with `bounds`, `input_dim` and `observe` (HF mode is used).
        n_init: Size of the initial design.
        n_iter: Total number of BO iterations, counting those already in `existing_data`.
        seed: Run seed (initial design and per-iteration seeding).
        existing_data: A result dict from an earlier, interrupted run to resume from, or None.
        checkpoint: Optional callable that receives the result dict after every iteration.

    Returns:
        dict: "history", a list with one dict per evaluation: "iter" (0 for the initial
        design), "x" (D,) and "y" (1,) as CPU tensors, "best_y" (best observation so far) and
        "train_time" (GP fit time in s). "prior_data" is None.
    """
    _seed_everything(seed)
    device = bench.bounds.device

    if existing_data:
        history = existing_data['history']
        train_x, train_y = _history_xy(history, device)
        start_iter = max([h['iter'] for h in history])
        print(f"Resuming Standard GP from iter {start_iter}")
    else:
        train_x = _initial_design(bench, n_init, seed)
        train_y = bench.observe(train_x)

        history = []
        for i in range(n_init):
            history.append({'iter': 0, 'x': train_x[i].cpu(), 'y': train_y[i].cpu(), 'best_y': train_y.max().item(),
                            'train_time': 0.0})
        start_iter = 0

    print(f"Standard GP (Seed {seed}) on {device}")
    for i in range(start_iter, n_iter):
        _seed_everything(_iter_seed(seed, i))  # a resumed run then continues exactly like an uninterrupted one
        t0 = time.time()
        gp = SingleTaskGP(train_x, train_y, input_transform=Normalize(bench.input_dim),
                          outcome_transform=Standardize(1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        train_time = time.time() - t0

        acq = LogExpectedImprovement(gp, best_f=train_y.max())
        cand, _ = optimize_acqf(acq, bench.bounds, q=1, num_restarts=5, raw_samples=64)

        y_new = bench.observe(cand)
        train_x = torch.cat([train_x, cand])
        train_y = torch.cat([train_y, y_new])

        history.append({'iter': i + 1, 'x': cand[0].cpu(), 'y': y_new[0].cpu(), 'best_y': train_y.max().item(),
                        'train_time': train_time})
        if i % 10 == 0: print(f"Iter {i}: Best {history[-1]['best_y']:.4f}")
        if checkpoint: checkpoint({"history": history, "prior_data": None})

    return {"history": history, "prior_data": None}


def run_custom_deep_gp_bo(bench, n_init=10, n_iter=40, seed=0, bench_name="Polymerization", existing_data=None,
                          n_samples=1000, n_warmup=500, n_chains=None, checkpoint=None):
    """BO with the structure-aligned, prior-informed deep GP.

    Inputs and targets are standardised with the statistics of the first `n_init` points;
    this also holds for resumed runs. The LF priors are refitted with the run seed, which
    is deterministic. If the initial HF yields are nearly constant (std < 0.1 * LF yield
    std), the LF yield std is used for standardisation. Each iteration samples the DGP
    with NUTS, maximises LogExpectedImprovement (q = 1, 5 restarts, 64 raw samples) over the
    standardised box, maps the candidate back to raw units, clamps it to the bounds and
    evaluates it in HF mode.

    Args:
        bench: Benchmark instance; must match `bench_name`.
        n_init: Size of the initial design.
        n_iter: Total number of BO iterations, counting those already in `existing_data`.
        seed: Run seed.
        bench_name: Key into `BENCH_CONFIGS`.
        existing_data: A result dict from an interrupted run to resume from, or None.
        n_samples: NUTS draws per chain (poster: 1000).
        n_warmup: NUTS warm-up steps per chain (poster: 500).
        n_chains: Number of chains. None gives 32 with a JAX GPU and 1 on the CPU.
        checkpoint: Optional callable that receives the result dict after every iteration.

    Returns:
        dict: "history" in the format of `run_standard_bo`, where "train_time" is the NUTS
        time in s; and "prior_data", the LF samples (see `train_poly_priors` and
        `train_pharma_priors`).

    Raises:
        KeyError: If `bench_name` is not in `BENCH_CONFIGS`.
    """
    _seed_everything(seed)
    cfg = BENCH_CONFIGS[bench_name]
    device = bench.bounds.device

    if n_chains is None:
        n_chains = 32 if JAX_GPU else 1
    print(f"CustomDeepGP (NUTS on {'GPU' if JAX_GPU else 'CPU'}, {n_chains}x{n_samples}; torch on {device})")

    if existing_data:
        history = existing_data['history']
        prior_data = existing_data['prior_data']
        train_x_raw, train_y_raw = _history_xy(history, device)
        start_iter = max(h['iter'] for h in history)
        print(f"Resuming {bench_name} from iter {start_iter}...")
    else:
        train_x_raw = _initial_design(bench, n_init, seed)
        train_y_raw = bench.observe(train_x_raw, mode='HF')
        prior_data = None

        history = []
        for i in range(n_init):
            history.append(
                {'iter': 0, 'x': train_x_raw[i].cpu(), 'y': train_y_raw[i].cpu(), 'best_y': train_y_raw.max().item(),
                 'train_time': 0.0})
        start_iter = 0

    # Normalisation is fixed by the initial design, both for fresh and resumed runs
    x_mean, x_std = train_x_raw[:n_init].mean(0), train_x_raw[:n_init].std(0) + 1e-6
    y_mean, y_std = train_y_raw[:n_init].mean(), train_y_raw[:n_init].std() + 1e-6
    _seed_everything(_iter_seed(seed, -1))  # fit_gpytorch_mll may draw random restarts
    priors, fresh_prior_data = cfg['p_fn'](bench, train_x_raw[:n_init], x_mean, x_std, seed=seed)
    # A near-constant initial design (e.g. PharmaV2 seed 57: ten yields of ~0, std 8e-7) would put
    # every later observation at ~1e5 standardised units. Fall back to the LF yield spread then.
    lf_std = float(priors[-1][0]['_target_stats'][1])
    if y_std < 0.1 * lf_std:
        print(f"Initial HF yields are nearly constant (std {y_std.item():.1e}); standardising with the "
              f"LF yield std {lf_std:.1e} instead.")
        y_std = torch.as_tensor(lf_std, device=device, dtype=train_y_raw.dtype)
    align_output_prior(priors, y_mean, y_std)
    if prior_data is None:
        prior_data = fresh_prior_data

    for i in range(start_iter, n_iter):
        _seed_everything(_iter_seed(seed, i))  # a resumed run then continues exactly like an uninterrupted one
        tx = (train_x_raw - x_mean) / x_std
        ty = (train_y_raw - y_mean) / y_std

        t0 = time.time()
        dgp = LightweightDeepGP(tx, ty.squeeze(-1), hidden_dims=cfg['h'], injection_map=cfg['inj'],
                                mean_params_list=priors)
        dgp.fit_numpyro(num_chains=n_chains, num_samples=n_samples, warmup_steps=n_warmup,
                        rng_seed=_iter_seed(seed, i))
        train_time = time.time() - t0

        bo_model = DeepGPBoTorchModel(dgp)
        b_low = (bench.bounds[0] - x_mean) / x_std
        b_high = (bench.bounds[1] - x_mean) / x_std
        acq_bounds = torch.stack([b_low, b_high])

        acq = LogExpectedImprovement(bo_model, best_f=ty.max())
        cand_norm, _ = optimize_acqf(acq, acq_bounds, q=1, num_restarts=5, raw_samples=64)

        cand_raw = cand_norm * x_std + x_mean
        cand_raw = cand_raw.clamp(bench.bounds[0], bench.bounds[1])
        y_new = bench.observe(cand_raw, mode='HF')

        train_x_raw = torch.cat([train_x_raw, cand_raw])
        train_y_raw = torch.cat([train_y_raw, y_new])

        history.append({'iter': i + 1, 'x': cand_raw[0].cpu(), 'y': y_new[0].cpu(), 'best_y': train_y_raw.max().item(),
                        'train_time': train_time})
        if i % 5 == 0: print(f"Iter {i}: Best {history[-1]['best_y']:.4f}")
        if checkpoint: checkpoint({"history": history, "prior_data": prior_data})

    return {"history": history, "prior_data": prior_data}
