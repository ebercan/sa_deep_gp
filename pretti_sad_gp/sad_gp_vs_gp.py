"""Early prototype: a two-GP "composite" surrogate versus a SingleTaskGP on Polymerization.

A helper study, not the method of the poster. The composite model chains two exact GPs
fitted by marginal likelihood, not NUTS. GP 1 maps (x0, x1) to the measured
intermediate C_eff; GP 2 maps (C_eff, x2) to the yield. Each GP uses an LF GP's
posterior mean as its mean function (`GPMean`). Unlike the main method, it observes
the intermediate C_eff in HF. Both loops run `N_ITER` iterations from the same random
initial design and are plotted as the best value so far divided by a brute-force
estimate of the global maximum.
"""

import torch
import gpytorch
import numpy as np
import warnings
import plotly.graph_objects as go

from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from botorch.sampling import IIDNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import Mean

# Import your benchmark
from benchmarks import DiscontinuousPolymerization

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
N_INIT = 10
N_ITER = 10  # Reduced slightly for speed in demo, increase as needed
N_LF_SAMPLES = 200

# Suppress noisy BoTorch data warnings
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")


# --- 1. SENSOR WRAPPER ---
class SensorBenchmark(DiscontinuousPolymerization):
    """Polymerization benchmark that also reports the intermediate C_eff ("sensor").

    Attributes:
        mode: Fidelity used by `observe_full`, "HF" or "LF".
    """
    def __init__(self, mode='HF'):
        """Create the benchmark on `DEVICE` in `DTYPE`.

        Args:
            mode: "HF" or "LF".
        """
        super().__init__(device=DEVICE, dtype=DTYPE)
        self.mode = mode

    def observe_full(self, x):
        """Evaluate the process and return the intermediate and the final yield.

        Args:
            x: Inputs, shape (N, 3).

        Returns:
            torch.Tensor: [C_eff, yield], shape (N, 2).
        """
        y1_C, _, y_final = self.observe_all_steps(x, mode=self.mode)
        return torch.cat([y1_C, y_final], dim=1)


# --- 2. GLOBAL OPTIMUM ESTIMATOR ---
def calculate_global_max(env, n_samples=10000):
    """Estimate the best achievable yield by uniform random search.

    Args:
        env: `SensorBenchmark` instance.
        n_samples: Number of random points in [0, 1]^3.

    Returns:
        float: The largest yield found.
    """
    # 1. Dense Grid Search
    X = torch.rand(n_samples, 3, device=DEVICE, dtype=DTYPE)
    Y = env.observe_full(X)[:, 1]
    current_max = Y.max().item()
    return current_max


# --- 3. PHYSICS-INFORMED COMPONENTS ---
class GPMean(Mean):
    """GPyTorch mean module that uses a frozen prior GP's posterior mean.

    The host GP models its targets after its own Standardize transform, so the prior's
    prediction (raw units) is mapped into that space.

    Attributes:
        model: The frozen prior GP (eval mode, no gradients).
        t_mean: Mean of the host GP's outcome transform (buffer).
        t_std: Std of the host GP's outcome transform (buffer).
    """
    def __init__(self, model, target_transform):
        """Freeze `model` and store the host GP's outcome-transform statistics.

        Args:
            model: A fitted BoTorch GP whose posterior mean serves as the prior mean.
            target_transform: The host GP's fitted `Standardize` transform.
        """
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.register_buffer("t_mean", target_transform.means.detach().reshape(()))
        self.register_buffer("t_std", target_transform.stdvs.detach().reshape(()))

    def forward(self, x):
        """Evaluate the prior mean in the host GP's standardised target space.

        Args:
            x: Inputs, shape (..., N, d).

        Returns:
            torch.Tensor: Mean, shape (..., N).
        """
        self.model.eval()
        raw = self.model.posterior(x).mean.squeeze(-1)
        return (raw - self.t_mean) / self.t_std


class DeepPosterior(Posterior):
    """BoTorch posterior of the two-GP composite model.

    `mean` and `variance` plug GP 1's mean into GP 2, so they ignore layer-1 uncertainty.
    `rsample` propagates it by sampling layer 1 first.

    Attributes:
        gp1: Layer-1 GP, inputs X[..., :2].
        gp2: Layer-2 GP, inputs [layer-1 output, X[..., 2:]].
        X: Query points, shape (batch..., q, 3).
    """
    def __init__(self, gp1, gp2, X):
        """Store both layers and the query points.

        Args:
            gp1: Layer-1 GP.
            gp2: Layer-2 GP.
            X: Query points, shape (batch..., q, 3).
        """
        self.gp1 = gp1
        self.gp2 = gp2
        self.X = X

    @property
    def device(self):
        """Device of the query points.

        Returns:
            torch.device: The device of `X`.
        """
        return self.X.device

    @property
    def dtype(self):
        """Dtype of the query points.

        Returns:
            torch.dtype: The dtype of `X`.
        """
        return self.X.dtype

    @property
    def event_shape(self):
        """Event shape as defined by this class.

        Returns:
            torch.Size: torch.Size([1]).
        """
        return torch.Size([1])

    @property
    def batch_shape(self):
        """Batch shape of the query points.

        Returns:
            torch.Size: X.shape[:-2].
        """
        return self.X.shape[:-2]

    @property
    def batch_range(self):
        """Batch range used by BoTorch samplers.

        Returns:
            tuple: (0, -2).
        """
        return (0, -2)

    @property
    def base_sample_shape(self):
        """Shape of the base normal samples: two noise values (one per layer) per query point.

        Returns:
            torch.Size: X.shape[:-1] + (2,).
        """
        return self.X.shape[:-1] + torch.Size([2])

    @property
    def mean(self):
        """Plug-in mean: GP 2's mean at [GP 1's mean, x2].

        Returns:
            torch.Tensor: Shape (batch..., q, 1).
        """
        m1 = self.gp1.posterior(self.X[..., :2]).mean
        x2 = self.X[..., 2:]
        inp2 = torch.cat([m1, x2], dim=-1)
        return self.gp2.posterior(inp2).mean

    @property
    def variance(self):
        """Plug-in variance: GP 2's variance at [GP 1's mean, x2]. Ignores layer-1 uncertainty.

        Returns:
            torch.Tensor: Shape (batch..., q, 1).
        """
        m1 = self.gp1.posterior(self.X[..., :2]).mean
        x2 = self.X[..., 2:]
        inp2 = torch.cat([m1, x2], dim=-1)
        return self.gp2.posterior(inp2).variance

    def rsample_from_base_samples(self, sample_shape, base_samples):
        """Draw reparameterised samples by passing layer-1 draws through GP 2.

        Each layer is sampled from its marginal (diagonal) distribution, so correlations
        across the q points are ignored.

        Args:
            sample_shape: Leading sample shape, e.g. torch.Size([64]).
            base_samples: Standard normal noise, shape sample_shape + X.shape[:-1] + (2,).

        Returns:
            torch.Tensor: Samples, shape sample_shape + (batch..., q, 1).
        """
        noise_layer1 = base_samples[..., 0:1]
        noise_layer2 = base_samples[..., 1:2]

        # Layer 1
        post1 = self.gp1.posterior(self.X[..., :2])
        mean1 = post1.mean.expand(sample_shape + post1.mean.shape)
        std1 = post1.variance.sqrt().expand(sample_shape + post1.variance.shape)
        z_samples = mean1 + std1 * noise_layer1

        # Layer 2 (Inputs = [Z, x2])
        x2_expanded = self.X[..., 2:].unsqueeze(0).expand(sample_shape + self.X.shape[:-1] + (1,))
        layer2_inputs = torch.cat([z_samples, x2_expanded], dim=-1)

        post2 = self.gp2.posterior(layer2_inputs)
        mean2 = post2.mean
        std2 = post2.variance.sqrt()

        return mean2 + std2 * noise_layer2

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        """Draw reparameterised samples, generating base samples if none are given.

        Args:
            sample_shape: Leading sample shape.
            base_samples: Optional standard normal noise (see `rsample_from_base_samples`).

        Returns:
            torch.Tensor: Samples, shape sample_shape + (batch..., q, 1).
        """
        if base_samples is not None:
            return self.rsample_from_base_samples(sample_shape, base_samples)
        noise_shape = sample_shape + self.base_sample_shape
        base_samples = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        return self.rsample_from_base_samples(sample_shape, base_samples)


class CompositeDeepGP(Model):
    """Two-layer composite surrogate: (x0, x1) -> C_eff -> yield (with x2).

    Both layers are SingleTaskGPs with a Standardize outcome transform and an LF GP as
    their mean function. They are fitted separately on observed intermediates.

    Attributes:
        gp1: Layer-1 GP, (x0, x1) -> C_eff.
        gp2: Layer-2 GP, (C_eff, x2) -> yield.
    """
    def __init__(self, train_x, train_y_inter, train_y_final, lf_gp1, lf_gp2):
        """Build both layers.

        Args:
            train_x: Inputs, shape (n, 3).
            train_y_inter: Observed C_eff, shape (n, 1).
            train_y_final: Observed yield, shape (n, 1).
            lf_gp1: LF GP for C_eff, used as the layer-1 mean.
            lf_gp2: LF GP for the yield, used as the layer-2 mean.
        """
        super().__init__()
        # IMPROVEMENT: Added outcome_transform=Standardize
        # This helps the GP fit the residual scale much better.
        self.gp1 = SingleTaskGP(train_x[..., :2], train_y_inter, outcome_transform=Standardize(m=1))
        self.gp1.mean_module = GPMean(lf_gp1, self.gp1.outcome_transform)

        layer2_inputs = torch.cat([train_y_inter, train_x[..., 2:]], dim=-1)
        self.gp2 = SingleTaskGP(layer2_inputs, train_y_final, outcome_transform=Standardize(m=1))
        self.gp2.mean_module = GPMean(lf_gp2, self.gp2.outcome_transform)

    def posterior(self, X, observation_noise=False, **kwargs):
        """Return the composite posterior at X.

        Args:
            X: Query points, shape (batch..., q, 3).
            observation_noise: Ignored.
            **kwargs: Ignored.

        Returns:
            DeepPosterior: The posterior object.
        """
        return DeepPosterior(self.gp1, self.gp2, X)

    @property
    def num_outputs(self):
        """Number of outputs.

        Returns:
            int: 1.
        """
        return 1

    def state_dict(self):
        """Merged state dicts of both layers. Keys of the two layers can collide.

        Returns:
            dict: Parameters and buffers of `gp1` and `gp2`.
        """
        return {**self.gp1.state_dict(), **self.gp2.state_dict()}

    def load_state_dict(self, state_dict):
        """No-op. Loading a state dict is not supported.

        Args:
            state_dict: Ignored.
        """
        pass


# --- 4. EXECUTION UTILS ---
def train_prior_gp(X, Y):
    """Fit a SingleTaskGP (default kernel, Standardize outcome) by marginal likelihood.

    Args:
        X: Inputs, shape (n, d).
        Y: Targets, shape (n, 1).

    Returns:
        SingleTaskGP: The fitted GP.
    """
    gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def run_benchmark():
    """Run both BO loops side by side and show the comparison plot.

    LF priors are fitted on `N_LF_SAMPLES` random LF points. The standard loop uses
    analytic LogEI, and the composite loop uses qLogEI with 64 IID normal MC samples.
    """
    # Setup Environments
    lf_env = SensorBenchmark(mode='LF')
    hf_env = SensorBenchmark(mode='HF')

    # 1. Calc Global Max for Normalization
    print("--- Calculating Best Possible Target (Approximation) ---")
    global_max = calculate_global_max(hf_env)
    print(f"Global Max Est: {global_max:.4f}")

    # 2. Train Priors
    print("--- Training Low-Fidelity Priors ---")
    X_lf = torch.rand(N_LF_SAMPLES, 3, device=DEVICE, dtype=DTYPE)
    Y_lf_full = lf_env.observe_full(X_lf)

    prior_gp1 = train_prior_gp(X_lf[..., :2], Y_lf_full[:, 0:1])

    X_lf_step2 = torch.cat([Y_lf_full[:, 0:1], X_lf[:, 2:]], dim=-1)
    prior_gp2 = train_prior_gp(X_lf_step2, Y_lf_full[:, 1:2])

    # 3. Initialization
    print(f"--- Starting BO ({N_INIT} Init + {N_ITER} Iter) ---")
    train_x = torch.rand(N_INIT, 3, device=DEVICE, dtype=DTYPE)
    train_obj = hf_env.observe_full(train_x)

    # Separate Histories
    train_x_std, train_obj_std = train_x.clone(), train_obj.clone()
    train_x_deep, train_obj_deep = train_x.clone(), train_obj.clone()

    # Evaluation Axis (Includes Initial Points)
    eval_nums = list(range(1, N_INIT + N_ITER + 1))

    # Track "Best Found so far" for every single evaluation index
    # (Initialize with the running max of the init batch)
    std_trace = np.maximum.accumulate(train_obj_std[:, 1].cpu().numpy()).tolist()
    deep_trace = np.maximum.accumulate(train_obj_deep[:, 1].cpu().numpy()).tolist()

    # 4. Optimization Loop
    for i in range(N_ITER):
        # --- STANDARD BO ---
        best_f_std = train_obj_std[:, 1].max().item()
        std_gp = SingleTaskGP(train_x_std, train_obj_std[:, 1:2], outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(std_gp.likelihood, std_gp))

        acq_std = LogExpectedImprovement(std_gp, best_f=best_f_std)
        cand_std, _ = optimize_acqf(acq_std, bounds=hf_env.bounds, q=1, num_restarts=5, raw_samples=64)

        # --- DEEP GP ---
        best_f_deep = train_obj_deep[:, 1].max().item()
        deep_gp = CompositeDeepGP(
            train_x_deep, train_obj_deep[:, 0:1], train_obj_deep[:, 1:2],
            prior_gp1, prior_gp2
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(deep_gp.gp1.likelihood, deep_gp.gp1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(deep_gp.gp2.likelihood, deep_gp.gp2))

        mc_sampler = IIDNormalSampler(sample_shape=torch.Size([64]))
        acq_deep = qLogExpectedImprovement(deep_gp, best_f=best_f_deep, sampler=mc_sampler)
        cand_deep, _ = optimize_acqf(acq_deep, bounds=hf_env.bounds, q=1, num_restarts=5, raw_samples=64)

        # --- EVALUATE & UPDATE ---
        # Evaluate
        val_std = hf_env.observe_full(cand_std)
        val_deep = hf_env.observe_full(cand_deep)

        # Update Data
        train_x_std = torch.cat([train_x_std, cand_std])
        train_obj_std = torch.cat([train_obj_std, val_std])

        train_x_deep = torch.cat([train_x_deep, cand_deep])
        train_obj_deep = torch.cat([train_obj_deep, val_deep])

        # Update Traces (Max so far)
        std_trace.append(max(std_trace[-1], val_std[0, 1].item()))
        deep_trace.append(max(deep_trace[-1], val_deep[0, 1].item()))

        print(f"Iter {i + 1}/{N_ITER}: Std={std_trace[-1]:.4f} | Deep={deep_trace[-1]:.4f}")

    # 5. PLOT WITH PLOTLY
    # Normalize by Global Max
    y_std_norm = np.array(std_trace) / global_max
    y_deep_norm = np.array(deep_trace) / global_max

    fig = go.Figure()

    # Standard GP Line
    fig.add_trace(go.Scatter(
        x=eval_nums, y=y_std_norm,
        mode='lines+markers', name='Standard GP (Black Box)',
        line=dict(color='gray', dash='dash')
    ))

    # Deep GP Line
    fig.add_trace(go.Scatter(
        x=eval_nums, y=y_deep_norm,
        mode='lines+markers', name='Structure-Aligned DGP (Ours)',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title="Optimization Efficiency: Deep vs Standard GP",
        xaxis_title="Number of High-Fidelity Evaluations",
        yaxis_title="Normalized Best Yield (Found / Possible)",
        template="plotly_white",
        yaxis=dict(range=[0, 1.05]),  # Fix range 0-1
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    fig.show()


if __name__ == "__main__":
    run_benchmark()