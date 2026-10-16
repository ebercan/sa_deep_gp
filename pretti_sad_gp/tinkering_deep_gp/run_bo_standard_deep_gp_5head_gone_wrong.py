# run_bo_standard_deep_gp.py
# -------------------------------------------
# DeepGP + LogEI on Ackley with warm-start BO
# - Float64 everywhere for GP stability
# - Stable training loop (early stopping, MC ramp, clip)
# - Robust posterior (no NaN std warnings) — uses .reshape, not .view
# - Optional mid-training inducing-point reinit
# - Cleaned imports & less spammy logging
# -------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior, LogNormalPrior

from botorch.test_functions import Ackley
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.transforms import unnormalize

import matplotlib.pyplot as plt


# --------------------------
# Global defaults & helpers
# --------------------------
torch.manual_seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 4))
torch.set_default_dtype(torch.float64)  # keep everything in float64 for stability
DEVICE = "cpu"

DEBUG_POSTERIOR = False  # set True for per-call diagnostics


def ensure_dtype_device(x: Tensor, dtype=torch.get_default_dtype(), device=DEVICE) -> Tensor:
    return x.to(dtype=dtype, device=device)


from dataclasses import dataclass, field, asdict
from typing import List, Dict

@dataclass
class EpochLog:
    epoch: int
    loss: float
    beta: float
    num_samples: int
    kl_total: float
    ell_est: float
    grad_norm: float
    noise: float
    ls_comp_mean: float
    ls_out_mean: float
    os_comp_mean: float
    os_out_mean: float
    rmse_scaled: float
    nlpdb_scaled: float
    rmse: float
    nlpdb: float

@dataclass
class TrainReport:
    epoch_logs: List[EpochLog] = field(default_factory=list)
    early_stopped_epoch: int | None = None
    plateau_reinit_done: bool = False
    notes: List[str] = field(default_factory=list)

def _to_scalar(x: torch.Tensor) -> float:
    return float(x.detach().mean().cpu())


# --------------------------
# DeepGP definition
# --------------------------
class GPLayer(DeepGPLayer):
    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int],
        num_inducing: int = 32,
        mean_type: str = "constant",
    ):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims, dtype=torch.get_default_dtype(), device=DEVICE)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(
                output_dims, num_inducing, input_dims, dtype=torch.get_default_dtype(), device=DEVICE
            )
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = ConstantMean(batch_shape=batch_shape) if mean_type == "constant" else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims), batch_shape=batch_shape
        )

        # gentle constraints & a prior for outputscale
        self.covar_module.register_constraint("raw_outputscale", GreaterThan(0.05))
        try:
            self.covar_module.register_prior("outputscale_prior", LogNormalPrior(0.3, 0.5), "outputscale")
        except Exception:
            pass

        # sensible initialization
        with torch.no_grad():
            ls = self.covar_module.base_kernel.lengthscale
            ls.copy_(torch.full_like(ls, 0.3))
            self.covar_module.outputscale.fill_(1.5)
            if isinstance(self.mean_module, ConstantMean):
                self.mean_module.constant.fill_(0.0)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RWGSDeepGP(DeepGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(
        self,
        train_x_shape: torch.Size,
        X1_dims: int,
        num_hidden_dims: int,
        num_inducing: int,
    ):
        super().__init__()
        _d = train_x_shape[-1]
        self.X1_dims = X1_dims

        self.compressor_layer = GPLayer(
            input_dims=_d,
            output_dims=num_hidden_dims,
            mean_type="constant",
            num_inducing=int(num_inducing),
        )
        self.output_layer = GPLayer(
            input_dims=num_hidden_dims,
            output_dims=None,
            mean_type="constant",
            num_inducing=int(num_inducing),
        )

        self.likelihood = GaussianLikelihood(noise_prior=GammaPrior(2.0, 40.0))
        self.likelihood.noise_covar.initialize(noise=1e-3)

        with torch.no_grad():
            # slightly different inits for two layers
            self.compressor_layer.covar_module.base_kernel.lengthscale.fill_(0.3)
            self.compressor_layer.covar_module.outputscale.fill_(1.5)
            self.output_layer.covar_module.base_kernel.lengthscale.fill_(0.6)
            self.output_layer.covar_module.outputscale.fill_(1.2)

    def forward(self, inputs: Tensor) -> MultivariateNormal:
        latent = self.compressor_layer(inputs).rsample()
        return self.output_layer(latent)

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        posterior_transform=None,
        enable_grad: Optional[bool] = None,
        *,
        num_samples: int = 16,
        variance_inflation_tau2: float = 2.0,  # NEW: knob
        **kwargs,
    ) -> GPyTorchPosterior:
        if enable_grad is None:
            enable_grad = torch.is_grad_enabled()
        grad_mode = torch.enable_grad() if enable_grad else torch.no_grad()

        with grad_mode, gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(int(num_samples)):
            if X.dim() == 2:
                X = X.unsqueeze(0)
            b, q, d = X.shape
            X = X.reshape(-1, d)

            preds = self.likelihood(self(X)) if observation_noise else self(X)
            mu, var = preds.mean, preds.variance

            flat = b * q
            if mu.numel() > flat:
                S = mu.numel() // flat
                mu = mu.reshape(S, flat)
                var = var.reshape(S, flat)
                mu_mean = mu.mean(dim=0)
                second = (var + mu**2).mean(dim=0)
                var_mean = (second - mu_mean**2).clamp_min(1e-12)
                mu, var = mu_mean, var_mean
            else:
                mu = mu.reshape(-1)
                var = var.reshape(-1)

            var = float(variance_inflation_tau2) * var + 1e-4
            mu = mu.reshape(b, q)
            var = var.reshape(b, q)

            post = GPyTorchPosterior(MultivariateNormal(mu, torch.diag_embed(var)))
            if posterior_transform is not None:
                post = posterior_transform(post)
            return post


# --------------------------
# Utilities
# --------------------------
def _train_fit_metrics(
    model: RWGSDeepGP,
    X_scaled: Tensor,
    y_scaled: Tensor,
    y_mean: Tensor,
    y_std: Tensor,
    *,
    num_samples: int = 32,
) -> Dict[str, float]:
    with torch.no_grad():
        post = model.posterior(
            X_scaled.unsqueeze(1),
            observation_noise=True,
            enable_grad=False,
            num_samples=num_samples,
            variance_inflation_tau2=1.0,  # honest diagnostics
        )
        mu_s = post.mean.squeeze(0).reshape(-1)
        var_s = post.variance.squeeze(0).reshape(-1).clamp_min(1e-12)
        y_s = y_scaled.reshape(-1)

        rmse_scaled = torch.sqrt(torch.mean((mu_s - y_s)**2))
        nlpdb_scaled = 0.5 * (torch.log(2 * torch.pi * var_s) + (y_s - mu_s)**2 / var_s)
        nlpdb_scaled = nlpdb_scaled.mean()

        mu = mu_s * y_std + y_mean
        var = var_s * (y_std**2)
        y = y_s * y_std + y_mean

        rmse = torch.sqrt(torch.mean((mu - y)**2))
        nlpdb = 0.5 * (torch.log(2 * torch.pi * var) + (y - mu)**2 / var)
        nlpdb = nlpdb.mean()

        return {
            "rmse_scaled": _to_scalar(rmse_scaled),
            "nlpdb_scaled": _to_scalar(nlpdb_scaled),
            "rmse": _to_scalar(rmse),
            "nlpdb": _to_scalar(nlpdb),
        }


def reinitialize_inducing_points(model: RWGSDeepGP, train_x: Tensor, sigma: float = 1.5, verbose: bool = True) -> None:
    """Reinitialize inducing points around data mean with scaled noise."""
    with torch.no_grad():
        if verbose:
            print("\n🔁 Reinitializing inducing points...")
        for layer_name in ["compressor_layer", "output_layer"]:
            layer = getattr(model, layer_name, None)
            if layer is None or not hasattr(layer, "variational_strategy"):
                continue
            vs = layer.variational_strategy
            Z = vs.inducing_points
            if Z is None:
                continue

            shape = Z.shape
            dtype, device = Z.dtype, Z.device

            mean_data = train_x.mean(dim=0, keepdim=True)
            std_data = train_x.std(dim=0, keepdim=True).clamp_min(1e-12)
            new_Z = mean_data + sigma * torch.randn(shape, dtype=dtype, device=device) * std_data
            Z.copy_(new_Z)

            if verbose:
                zmin = Z.reshape(-1, Z.shape[-1]).min(dim=0).values.min().item()
                zmax = Z.reshape(-1, Z.shape[-1]).max(dim=0).values.max().item()
                ls = layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                print(
                    f"  • {layer_name:16s} shape={tuple(Z.shape)} "
                    f"x-range=[{zmin:+.2f},{zmax:+.2f}] "
                    f"ls.mean={ls.mean():.3f}"
                )
        if verbose:
            print("✅ Inducing points reinitialized.\n")


def init_data(ackley: Ackley, n_init: int = 10) -> Tuple[Tensor, Tensor]:
    bounds = ackley.bounds
    X_unit = torch.rand(n_init, ackley.dim, dtype=torch.get_default_dtype(), device=DEVICE)
    X = unnormalize(X_unit, bounds)
    y = ackley(X).unsqueeze(-1)
    return X, y


def standardize_xy(X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    X_mean, X_std = X.mean(0), X.std(0).clamp_min(1e-12)
    y_mean, y_std = y.mean(), y.std().clamp_min(1e-12)
    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean) / y_std
    return X_scaled, y_scaled, X_mean, X_std, y_mean, y_std


# --------------------------
# Training
# --------------------------
def train_deepgp_stable(
    model: RWGSDeepGP,
    train_x: Tensor,
    train_y: Tensor,
    *,
    lr: float = 7e-4,
    epochs: int = 300,
    initial_num_samples: int = 8,
    final_num_samples: int = 40,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    patience: int = 30,
    min_delta: float = 1e-3,
    reinit_on_plateau: bool = True,
    verbose: bool = False,          # DEFAULT: no prints
    eval_every: int = 10,
    y_mean: Tensor | None = None,
    y_std: Tensor | None = None,
) -> tuple[RWGSDeepGP, TrainReport]:
    model.train()
    report = TrainReport()

    params = [
        {"params": model.compressor_layer.parameters(), "lr": lr},
        {"params": model.output_layer.parameters(), "lr": lr * 2},
        {"params": model.likelihood.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)

    best = float("inf")
    noimp = 0
    plateau_reinit_done = False
    N = train_x.shape[0]
    _y_mean = torch.zeros((), dtype=train_y.dtype, device=train_y.device) if y_mean is None else y_mean
    _y_std  = torch.ones((),  dtype=train_y.dtype, device=train_y.device) if y_std  is None else y_std

    for epoch in range(epochs):
        optimizer.zero_grad()
        frac = epoch / max(1, epochs)
        beta = max(1e-3, min(1.0, (frac**1.5) * 1.5))
        num_s = int(initial_num_samples + (final_num_samples - initial_num_samples) * frac)
        num_s = max(initial_num_samples, min(final_num_samples, num_s))

        base_elbo = VariationalELBO(model.likelihood, model, num_data=N)
        base_elbo.beta = beta
        mll = DeepApproximateMLL(base_elbo)

        with gpytorch.settings.num_likelihood_samples(num_s), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
            out = model(train_x)
            loss = -mll(out, train_y.squeeze(-1))
            loss = loss.mean() if loss.ndim > 0 else loss

            # KL across layers
            kl_total = 0.0
            for lyr in (model.compressor_layer, model.output_layer):
                try:
                    kl = lyr.variational_strategy.kl_divergence()
                    kl_total += float(kl.sum().detach().cpu())
                except Exception:
                    pass

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))
        optimizer.step()

        # clamp hypers
        for m in [model.compressor_layer, model.output_layer]:
            if hasattr(m, "covar_module"):
                m.covar_module.outputscale.data.clamp_(min=0.2, max=5.0)
                m.covar_module.base_kernel.lengthscale.data.clamp_(min=0.05)
        if hasattr(model.likelihood, "noise_covar"):
            model.likelihood.noise_covar.noise.clamp_(1e-6, 5.0)

        # early stop bookkeeping
        loss_val = float(loss.detach())
        if loss_val + min_delta < best:
            best = loss_val
            noimp = 0
        else:
            noimp += 1

        if reinit_on_plateau and (not plateau_reinit_done) and noimp >= max(3, patience // 2):
            reinitialize_inducing_points(model, train_x, sigma=1.5, verbose=False)  # no prints
            plateau_reinit_done = True
            report.plateau_reinit_done = True
            report.notes.append(f"Reinitialized inducing points at epoch {epoch} due to plateau.")

        # collect metrics (silent)
        if (epoch % eval_every == 0) or (epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                noise = _to_scalar(model.likelihood.noise_covar.noise)
                ls_comp = _to_scalar(model.compressor_layer.covar_module.base_kernel.lengthscale)
                os_comp = _to_scalar(model.compressor_layer.covar_module.outputscale)
                ls_out  = _to_scalar(model.output_layer.covar_module.base_kernel.lengthscale)
                os_out  = _to_scalar(model.output_layer.covar_module.outputscale)

                ell_est = float(beta * kl_total - loss_val)
                fit = _train_fit_metrics(model, train_x, train_y, _y_mean, _y_std, num_samples=max(16, num_s//2))

                report.epoch_logs.append(
                    EpochLog(
                        epoch=epoch,
                        loss=loss_val,
                        beta=float(beta),
                        num_samples=int(num_s),
                        kl_total=float(kl_total),
                        ell_est=ell_est,
                        grad_norm=grad_norm,
                        noise=noise,
                        ls_comp_mean=ls_comp,
                        ls_out_mean=ls_out,
                        os_comp_mean=os_comp,
                        os_out_mean=os_out,
                        rmse_scaled=fit["rmse_scaled"],
                        nlpdb_scaled=fit["nlpdb_scaled"],
                        rmse=fit["rmse"],
                        nlpdb=fit["nlpdb"],
                    )
                )
            model.train()

        if noimp >= patience:
            report.early_stopped_epoch = epoch + 1
            break

    model.eval()
    if report.epoch_logs:
        last = report.epoch_logs[-1]
        ratio = (last.ell_est / max(1e-8, last.kl_total)) if last.kl_total > 0 else float("inf")
        report.notes.append(f"Final ELL/KL ratio ≈ {ratio:.2f} (≫1: data-driven, ≪1: prior-dominated).")
    return model, report


# --------------------------
# Plotting (1D)
# --------------------------
def plot_bo_state_1d(
    model: RWGSDeepGP,
    ackley: Ackley,
    X_scaled: Tensor,
    y_scaled: Tensor,
    X_mean: Tensor,
    X_std: Tensor,
    y_mean: Tensor,
    y_std: Tensor,
    bounds: Tensor,
    best_f: float,
    iteration: int,
    show_af: bool = True,
):
    """Visualize state for 1D problem: true f, posterior (mean ±2σ), and LogEI."""
    model.eval()

    Xtest = torch.linspace(bounds[0].item(), bounds[1].item(), 400, dtype=torch.get_default_dtype(), device=DEVICE).unsqueeze(-1)
    Xtest_scaled = (Xtest - X_mean) / X_std

    with torch.no_grad():
        post = model.posterior(Xtest_scaled.unsqueeze(1), enable_grad=False, num_samples=64)
        mean = post.mean.squeeze(0).reshape(-1)
        var = post.variance.squeeze(0).reshape(-1)

    mean_plot = mean * y_std + y_mean
    std_plot = var.clamp_min(1e-12).sqrt() * y_std
    y_true = ackley(Xtest).squeeze()

    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    with torch.no_grad():
        af_vals = acqf(Xtest_scaled.unsqueeze(1)).reshape(-1)

    # numpy for plotting
    Xtest_np = Xtest.detach().cpu().numpy().squeeze()
    mean_np = mean_plot.detach().cpu().numpy()
    std_np = std_plot.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    X_samples_np = (X_mean + X_scaled * X_std).detach().cpu().numpy()
    y_samples_np = (y_scaled * y_std + y_mean).detach().cpu().numpy()
    af_np = af_vals.detach().cpu().numpy()

    fig, axs = plt.subplots(3 if show_af else 2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(f"DeepGP BO Progress (Iter {iteration})", fontsize=14, y=0.95)

    # 1) True function + samples
    axs[0].plot(Xtest_np, y_true_np, "k--", label="True Ackley")
    axs[0].scatter(X_samples_np, y_samples_np, color="red", label="Samples", zorder=5)
    axs[0].set_ylabel("f(x)")
    axs[0].set_title("Objective and Observations")
    axs[0].legend()

    # 2) Posterior
    axs[1].plot(Xtest_np, mean_np, "b", label="Posterior mean")
    axs[1].fill_between(Xtest_np, mean_np - 2 * std_np, mean_np + 2 * std_np, alpha=0.2, color="blue", label="±2σ")
    axs[1].scatter(X_samples_np, y_samples_np, color="red", s=25, label="Samples")
    axs[1].set_ylabel("Predicted mean")
    axs[1].set_title("DeepGP Posterior")
    axs[1].legend()

    # 3) Acquisition
    if show_af:
        axs[2].plot(Xtest_np, af_np, "g", label="LogEI")
        axs[2].set_ylabel("Acquisition")
        axs[2].set_xlabel("x")
        axs[2].set_title("Acquisition Function")
        axs[2].legend()

    plt.tight_layout()
    plt.show()


# --------------------------
# BO drivers
# --------------------------
def choose_next_x(
    model: RWGSDeepGP,
    bounds_scaled: Tensor,
    *,
    num_restarts: int = 10,
    raw_samples: int = 128,
) -> Tensor:
    acqf = LogExpectedImprovement(model=model, best_f=model_best_f(model))
    new_x_scaled, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_scaled,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 10, "maxiter": 100},
    )
    return new_x_scaled


def model_best_f_from_data(y: Tensor, y_mean: Tensor, y_std: Tensor) -> float:
    return ((y.max() - y_mean) / y_std).item()


def model_best_f(model: RWGSDeepGP) -> float:
    # placeholder if ever needed; in this script we pass explicit best_f
    return 0.0


def run_bo_ackley_warm_1d(n_plot_interval: int = 5):
    torch.manual_seed(0)

    # Problem setup
    ackley = Ackley(dim=1, negate=True).to(DEVICE)
    bounds = ackley.bounds.to(DEVICE)
    n_init, n_iter = 3, 25

    # Initial data
    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    # Standardize
    X_scaled, y_scaled, X_mean, X_std, y_mean, y_std = standardize_xy(train_x, train_y)

    # Initialize model once
    model = RWGSDeepGP(
        train_x_shape=X_scaled.shape,
        X1_dims=1,
        num_hidden_dims=8,
        num_inducing=max(8, min(64, train_x.shape[0])),
    )

    # inside run_bo_ackley_warm_1d(), after training:
    with gpytorch.settings.cholesky_jitter(1e-3):
        model, report = train_deepgp_stable(
            model,
            X_scaled,
            y_scaled,
            lr=7e-4,
            epochs=300,
            patience=30,
            reinit_on_plateau=True,
            verbose=False,  # <- no logs
            eval_every=10,
            y_mean=y_mean,
            y_std=y_std,
        )

    # if you want a single-line terminal summary per iteration:
    if report.epoch_logs:
        e = report.epoch_logs[-1]
        ell_kl = (e.ell_est / max(1e-8, e.kl_total)) if e.kl_total > 0 else float("inf")
        print(f"[fit] loss={e.loss:.3f} | ELL/KL≈{ell_kl:.2f} | noise={e.noise:.3g} "
              f"| ls(comp,out)=({e.ls_comp_mean:.3g},{e.ls_out_mean:.3g}) | RMSE={e.rmse:.3g} NLPD={e.nlpdb:.3g}")

    # or dump everything (JSON-like) to terminal when you want:
    import json
    print(json.dumps(asdict(report), indent=2))

    # Plot initial state
    print("\n📊 Plotting initial model state...")
    best_f = model_best_f_from_data(train_y, y_mean, y_std)
    plot_bo_state_1d(
        model, ackley, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds, best_f, iteration=0
    )

    # BO loop
    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # Re-standardize with all data so far
        X_scaled, y_scaled, X_mean, X_std, y_mean, y_std = standardize_xy(train_x, train_y)

        # Warm-start training on updated data
        with gpytorch.settings.cholesky_jitter(1e-3):
            model, report = train_deepgp_stable(
                model,
                X_scaled,
                y_scaled,
                lr=7e-4,
                epochs=300,
                patience=30,
                reinit_on_plateau=True,
                verbose=False,  # <- no logs
                eval_every=10,
                y_mean=y_mean,
                y_std=y_std,
            )

            if report.epoch_logs:
                e = report.epoch_logs[-1]
                ell_kl = (e.ell_est / max(1e-8, e.kl_total)) if e.kl_total > 0 else float("inf")
                print(f"[fit] loss={e.loss:.3f} | ELL/KL≈{ell_kl:.2f} | noise={e.noise:.3g} "
                      f"| ls(comp,out)=({e.ls_comp_mean:.3g},{e.ls_out_mean:.3g}) | RMSE={e.rmse:.3g} NLPD={e.nlpdb:.3g}")

            # or dump everything (JSON-like) to terminal when you want:
            import json
            print(json.dumps(asdict(report), indent=2))

            # periodic inducing point refresh can help exploration
            if (iteration + 1) % 5 == 0:
                reinitialize_inducing_points(model, X_scaled, sigma=1.5, verbose=True)

        # Acquisition optimization in scaled space
        best_f = model_best_f_from_data(train_y, y_mean, y_std)
        scaled_bounds = torch.stack([(bounds[0] - X_mean) / X_std, (bounds[1] - X_mean) / X_std])

        new_x_scaled, _ = optimize_acqf(
            acq_function=LogExpectedImprovement(model=model, best_f=best_f),
            bounds=scaled_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"batch_limit": 10, "maxiter": 100},
        )

        # Unscale & evaluate
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        # Update dataset
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

        # Plot every n_plot_interval iterations
        if (iteration + 1) % n_plot_interval == 0:
            print(f"\n📈 Plotting BO progress after {iteration + 1} iterations...")
            plot_bo_state_1d(
                model, ackley, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds, best_f, iteration + 1
            )

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


# Baseline (exact GP) 1D BO for reference
def run_gp_bo_ackley():
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    torch.manual_seed(0)
    ackley = Ackley(dim=1, negate=True).to(DEVICE)
    bounds = ackley.bounds.to(DEVICE)
    n_init, n_iter = 3, 40

    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    for i in range(n_iter):
        print(f"\n=== GP BO Iteration {i + 1}/{n_iter} ===")
        X_scaled, y_scaled, X_mean, X_std, y_mean, y_std = standardize_xy(train_x, train_y)
        model = SingleTaskGP(train_X=X_scaled, train_Y=y_scaled)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = model_best_f_from_data(train_y, y_mean, y_std)
        acqf = LogExpectedImprovement(model=model, best_f=best_f)
        scaled_bounds = torch.stack([(bounds[0] - X_mean) / X_std, (bounds[1] - X_mean) / X_std])

        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

    print("\n=== GP Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


def main():
    # Run the DeepGP warm-start 1D Ackley demo
    run_bo_ackley_warm_1d()
    # For a baseline with an exact GP, uncomment:
    # run_gp_bo_ackley()


if __name__ == "__main__":
    main()