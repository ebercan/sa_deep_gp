from sklearn.neighbors import KNeighborsRegressor
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import umap
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union, Any
from torch import Tensor
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch.models.utils import gpt_posterior_settings
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils import standardize
from botorch.acquisition.objective import IdentityMCObjective
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
from botorch.test_functions import Ackley
import tqdm
from gpytorch.priors import GammaPrior, LogNormalPrior
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from torch import Size
from botorch.posteriors import GPyTorchPosterior
from botorch.models.gpytorch import GPyTorchModel

torch.manual_seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 4))
torch.set_default_dtype(torch.float32)
device = "cpu"

ackley = Ackley(dim=4, negate=True).to(device)
N = 20
bounds = ackley.bounds.clone().detach()
X = torch.rand(N, ackley.dim, dtype=torch.float64, device=device)
X = bounds[0] + (bounds[1] - bounds[0]) * X
y = ackley(X).unsqueeze(-1)

X_mean, X_std = X.mean(0), X.std(0).clamp_min(1e-12)
y_mean, y_std = y.mean(), y.std().clamp_min(1e-12)
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

train_n = int(0.8 * N)
train_x, test_x = X[:train_n].contiguous(), X[train_n:].contiguous()
train_y, test_y = y[:train_n].contiguous(), y[train_n:].contiguous()

train_loader = DataLoader(TensorDataset(train_x, train_y.squeeze(-1)), batch_size=len(train_y))
test_loader = DataLoader(TensorDataset(test_x, test_y.squeeze(-1)), batch_size=len(test_y))

# # def _make_inducing_points(
# #     M: int, input_dims: int, ref_X: Optional[torch.Tensor] = None, device=None, dtype=None
# # ):
# #     """
# #     Build M x D inducing points that cover the input space better than just sub-sampling data.
# #     - For 1D: even linspace over a conservative standardized range [-3, 3]
# #     - For D>1: Gaussian around ref_X (if provided) + a few random wide points
# #     """
# #     device = device or (ref_X.device if ref_X is not None else "cpu")
# #     dtype = dtype or (ref_X.dtype if ref_X is not None else torch.float64)
# #
# #     if input_dims == 1:
# #         # Even grid in standardized space (works because you scale X before training)
# #         z = torch.linspace(-3.0, 3.0, M, device=device, dtype=dtype).unsqueeze(-1)
# #         print(z.shape)
# #         return z.contiguous()
# #
# #     # D > 1
# #     if ref_X is not None:
# #         N = ref_X.size(0)
# #         # mix of sampled-from-ref and some wide randoms
# #         k = min(M // 2, N)
# #         idx = torch.randperm(N, device=device)[:k]
# #         A = ref_X[idx]
# #         # B = torch.randn(M - k, input_dims, device=device, dtype=dtype) * 1.5
# #         Z = torch.cat([A, B], dim=0)
# #     else:
# #         Z = torch.randn(M, input_dims, device=device, dtype=dtype) * 1.5
# #
# #     return Z[:M].contiguous()

from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior


def ensure_dtype_consistency(*tensors, dtype=torch.get_default_dtype()):
    """Casts all given tensors to the same dtype/device."""
    out = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            out.append(t.to(dtype=dtype))
        else:
            out.append(t)
    return out


class GPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32, mean_type='constant', inducing_source=None):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        # Mean/covar modules
        self.mean_module = ConstantMean(batch_shape=batch_shape) if mean_type == 'constant' else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

        # Constraint & prior
        self.covar_module.register_constraint("raw_outputscale", GreaterThan(0.1))
        try:
            self.covar_module.register_prior(
                "outputscale_prior", LogNormalPrior(0.3, 0.5), "outputscale"
            )
        except Exception:
            pass

        # Better init
        with torch.no_grad():
            ls = self.covar_module.base_kernel.lengthscale
            ls.copy_(torch.full_like(ls, 0.2))
            self.covar_module.outputscale.fill_(2.0)
            if isinstance(self.mean_module, ConstantMean):
                self.mean_module.constant.normal_(0.0, 0.05)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RWGSDeepGP(DeepGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x_shape, X1_dims, num_hidden_dims, num_inducing, inducing_source):
        super().__init__()
        self.X1_dims = X1_dims

        self.compressor_layer = GPLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type='constant',
            num_inducing=num_inducing if isinstance(num_inducing, int) else num_inducing[0],
        )
        self.output_layer = GPLayer(
            input_dims=num_hidden_dims,
            output_dims=None,
            mean_type='constant',
            num_inducing=num_inducing if isinstance(num_inducing, int) else num_inducing[-1],
        )
        self.likelihood = GaussianLikelihood(noise_prior=GammaPrior(2.0, 40.0))
        self.likelihood.noise_covar.initialize(noise=1e-3)

        with torch.no_grad():
            self.compressor_layer.covar_module.base_kernel.lengthscale.fill_(0.2)
            self.compressor_layer.covar_module.outputscale.fill_(2.0)
            self.output_layer.covar_module.base_kernel.lengthscale.fill_(0.5)
            self.output_layer.covar_module.outputscale.fill_(1.5)

    def forward(self, inputs):
        latent = self.compressor_layer(inputs).rsample()
        return self.output_layer(latent)

    # def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
    #     self.eval()
    #     if X.dim() == 2:
    #         X = X.unsqueeze(0)
    #     q, d = X.shape[-2], X.shape[-1]
    #     X_flat = X.reshape(-1, d)
    #
    #     num_samples = int(kwargs.get("num_samples", 256))
    #     with gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(num_samples):
    #         preds = self.likelihood(self(X_flat)) if observation_noise else self(X_flat)
    #         mu, var = preds.mean, preds.variance
    #
    #     if mu.dim() > 1 and mu.shape[0] == num_samples:
    #         mu_s = mu.view(num_samples, -1)
    #         var_s = var.view(num_samples, -1)
    #         mean_flat = mu_s.mean(dim=0)
    #         second_moment = (var_s + mu_s ** 2).mean(dim=0)
    #         var_flat = (second_moment - mean_flat ** 2).clamp_min(1e-12)
    #     else:
    #         mean_flat, var_flat = mu.view(-1), var.view(-1).clamp_min(1e-12)
    #
    #     # variance inflation (tau^2 = 2)
    #     tau = (2.0) ** 0.5
    #     var_flat = (tau ** 2) * var_flat + 1e-4
    #
    #     covar = torch.diag_embed(var_flat)
    #     mvn = MultivariateNormal(mean_flat, covar)
    #     post = GPyTorchPosterior(mvn)
    #
    #     if posterior_transform is not None:
    #         post = posterior_transform(post)
    #     return post

    def posterior(
        self,
        X,
        observation_noise: bool = False,
        posterior_transform=None,
        enable_grad: Optional[bool] = None,
        **kwargs,
    ):
        """
        Compute posterior of the DeepGP.

        Args:
            X: [batch, q, d] or [q, d]
            observation_noise: include likelihood noise
            posterior_transform: optional BoTorch transform
            enable_grad:
              - True → keep gradients (for acquisition optimization)
              - False → no gradients (for plotting)
              - None → auto-detect (default)
        """
        # ✅ Auto-detect: if no flag given, match torch.is_grad_enabled()
        if enable_grad is None:
            enable_grad = torch.is_grad_enabled()

        grad_mode = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_mode, gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(
            int(kwargs.get("num_samples", 16))
        ):
            if X.dim() == 2:
                X = X.unsqueeze(0)
            batch_size, q, d = X.shape
            X_flat = X.reshape(-1, d)

            # Forward through both layers
            latent_dist = self.compressor_layer(X_flat)
            latent_mean, latent_var = latent_dist.mean, latent_dist.variance
            preds = self.likelihood(self(X_flat)) if observation_noise else self(X_flat)
            mu, var = preds.mean, preds.variance

            # ---- Handle Monte Carlo samples ----
            flat_N = batch_size * q
            if mu.numel() > flat_N:
                num_samples = mu.numel() // flat_N
                mu = mu.reshape(num_samples, flat_N)
                var = var.reshape(num_samples, flat_N)
                print(f" -> Detected {num_samples} MC samples; aggregating.")
                mu_mean = mu.mean(dim=0)
                second_moment = (var + mu ** 2).mean(dim=0)
                var_mean = (second_moment - mu_mean ** 2).clamp_min(1e-12)
                mu, var = mu_mean, var_mean
            else:
                mu = mu.reshape(-1)
                var = var.reshape(-1)

            # ---- Variance inflation ----
            tau = 2.0 ** 0.5
            var = (tau ** 2) * var + 1e-4

            # ---- Reshape to [batch_size, q] ----
            mu = mu.reshape(batch_size, q)
            var = var.reshape(batch_size, q)

            # ---- Optional diagnostics ----
            if not enable_grad:
                print(f" Output mean: μ∈[{mu.min():+.3f}, {mu.max():+.3f}], std={mu.std():.3f}")
                print(f" Output var : σ²∈[{var.min():+.6f}, {var.max():+.6f}], mean={var.mean():+.6f}")
                print("=======================================\n")

            covar = torch.diag_embed(var)
            posterior = GPyTorchPosterior(MultivariateNormal(mu, covar))
            if posterior_transform is not None:
                posterior = posterior_transform(posterior)

            if not enable_grad:
                latent_std = latent_var.mean().sqrt().item()
                print(f" Latent layer variance mean={latent_var.mean():.4f}, std={latent_var.std():.4f}")

            return posterior


def freeze_feature_extractor(model: RWGSDeepGP, freeze: bool = True):
    for p in model.compressor_layer.parameters():
        p.requires_grad = not freeze


def train_deepgp(model, train_x, train_y, epochs, base_lr, initial_num_samples=8, final_num_samples=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=base_lr / 50
    )
    base_elbo = VariationalELBO(model.likelihood, model, num_data=train_x.shape[0], beta=1e-3)
    mll = DeepApproximateMLL(base_elbo)

    print(f"\n🚀 Training DeepGP with {train_x.shape[0]} samples, {epochs} epochs, lr={base_lr:.1e}")
    for epoch in tqdm.trange(epochs, desc="Epochs"):
        model.train()
        optimizer.zero_grad()

        # Monte Carlo warmup
        num_samples = int(initial_num_samples + (final_num_samples - initial_num_samples) * (epoch / epochs))
        num_samples = max(8, min(num_samples, final_num_samples))

        for xb, yb in train_loader:
            base_elbo.beta = max(1e-4, min(1.0, epoch / epochs ** 2.0))
            with gpytorch.settings.num_likelihood_samples(num_samples):
                out = model(xb)
                loss = -mll(out, yb)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            reinitialize_inducing_points(model, train_x)

        tqdm.tqdm.write(f"Epoch {epoch + 1:03d}: loss={loss.item():.6f}, beta={base_elbo.beta:.2f}")

    return model


def reinitialize_inducing_points(model, train_x: torch.Tensor, sigma: float = 1.5):
    """
    Reinitialize inducing points for all GPLayers in a DeepGP.

    Args:
        model: RWGSDeepGP model
        train_x: current standardized training inputs
        sigma: std dev for Normal(0, sigma^2) reinitialization (default=2.0)
    """
    with torch.no_grad():
        print("\n🔁 Reinitializing inducing points (standard GPyTorch style)...")
        for layer_name in ["compressor_layer", "pre_heater_layer", "reactor_layer", "separator_layer", "output_layer"]:
            if not hasattr(model, layer_name):
                continue
            layer = getattr(model, layer_name, None)
            if layer is None or not hasattr(layer, "variational_strategy"):
                continue

            vs = layer.variational_strategy
            Z = vs.inducing_points  # shape: [M, D] or [B, M, D]
            if Z is None:
                continue

            shape = Z.shape
            dtype, device = Z.dtype, Z.device

            # --- reinit centered on data mean ---
            mean_data = train_x.mean(dim=0, keepdim=True)
            std_data = train_x.std(dim=0, keepdim=True).clamp_min(1e-12)
            new_Z = mean_data + sigma * torch.randn(shape, dtype=dtype, device=device) * std_data

            # --- assign in place ---
            Z.copy_(new_Z)

            # --- diagnostics ---
            Z_np = Z.detach().cpu().numpy().reshape(-1, Z.shape[-1])
            zmin, zmax = Z_np.min(axis=0), Z_np.max(axis=0)
            print(
                f" • {layer_name:18s}: shape={tuple(Z.shape)}, "
                f"x-range per dim = [{zmin.min():+.2f}, {zmax.max():+.2f}]"
            )
            if hasattr(layer, "covar_module"):
                ls = layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                print(f" lengthscale mean={ls.mean():.3f}, std={ls.std():.3f}")

        print("✅ Inducing points reinitialized.\n")


def evaluate_deepgp(model, test_loader, y_mean, y_std, test_y):
    model.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(256), gpytorch.settings.cholesky_jitter(1e-3):
        means, vars, lls = model.predict(test_loader)

    means_unscaled = means.mean(0) * y_std + y_mean
    test_y_unscaled = test_y.squeeze() * y_std + y_mean
    rmse = torch.mean((means_unscaled - test_y_unscaled) ** 2).sqrt()
    nll = -lls.mean().item()

    print(f"\n✅ Final Results ({train_x.shape[0]} training pts)")
    print(f"RMSE (original scale): {rmse.item():.4f}")
    print(f"NLL: {nll:.4f}")
    print(f"Avg. predictive variance: {vars.mean().item():.4f}")
    return rmse.item(), nll, vars.mean().item()


def run_learning_ackley():
    num_hidden_dims = 6
    X2_dims = 0
    X1_dims = train_x.shape[-1] - X2_dims

    model = RWGSDeepGP(
        train_x.shape,
        X1_dims,
        num_hidden_dims,
        num_inducing=min(32, train_x.shape[0]),  # adapt to small data
        inducing_source=train_x
    )

    N = train_x.shape[0]
    base_lr = 1e-3 if N > 50 else 5e-4
    epochs = 200 if N > 50 else 400
    # num_samples = 32 if N > 50 else 64
    batch_size = min(N, 64)
    initial_num_samples = 8
    final_num_samples = 64

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y.squeeze(-1)),
        batch_size=batch_size,
        shuffle=True,
    )

    rmse, nll, var = None, None, None
    with gpytorch.settings.cholesky_jitter(1e-3):
        model = train_deepgp(model, train_x, train_y, epochs, base_lr)
        rmse, nll, var = evaluate_deepgp(model, test_loader, y_mean, y_std, test_y)


def init_data(ackley, n_init=10):
    bounds = ackley.bounds
    X = torch.rand(n_init, ackley.dim, dtype=torch.float64)
    X = unnormalize(X, bounds)
    y = ackley(X).unsqueeze(-1)
    return X, y


def train_deepgp_light(
    model,
    train_x,
    train_y,
    lr: float = 7e-4,
    epochs: int = 300,
    initial_num_samples: int = 16,
    final_num_samples: int = 64,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    verbose: bool = True,
):
    model.train()
    optimizer = torch.optim.Adam(
        [
            {"params": model.compressor_layer.parameters(), "lr": lr},
            {"params": model.output_layer.parameters(), "lr": lr * 2},
            {"params": model.likelihood.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    # No cosine annealing — keep learning rate steady
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * 2, eta_min=lr / 10
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        base_elbo = VariationalELBO(model.likelihood, model, num_data=train_x.shape[0])
        mll = DeepApproximateMLL(base_elbo)

        # smoother β ramp
        beta = max(1e-3, min(1.0, (epoch / epochs) ** 1.5))
        base_elbo.beta = beta

        # ramp likelihood samples
        num_samples = int(initial_num_samples + (final_num_samples - initial_num_samples) * (epoch / epochs))
        with gpytorch.settings.num_likelihood_samples(num_samples):
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        # keep hyperparams flexible
        for m in [model.compressor_layer, model.output_layer]:
            if hasattr(m, "covar_module"):
                m.covar_module.outputscale.data.clamp_(min=0.2, max=5.0)
        if hasattr(model.likelihood, "noise_covar"):
            model.likelihood.noise_covar.noise.clamp_(1e-5, 1e1)

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"[Epoch {epoch + 1:04d}] "
                f"Loss={loss.item():.3f} | β={beta:.3f} | "
                f"samples={num_samples} | lr={scheduler.get_last_lr()[0]:.1e}"
            )

    model.eval()
    return model


def train_deepgp_light_stable(
    model,
    train_x,
    train_y,
    lr: float = 7e-4,
    epochs: int = 300,
    initial_num_samples: int = 8,
    final_num_samples: int = 48,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    patience: int = 30,
    min_delta: float = 1e-3,
    verbose: bool = True,
):
    """
    Stable DeepGP training with early stopping and clean graph handling.
    - Slower KL ramp to prevent over-regularization.
    - Explicit loss detaching to avoid memory leaks.
    - Stops if loss plateaus for patience epochs.
    """
    model.train()
    optimizer = torch.optim.Adam(
        [
            {"params": model.compressor_layer.parameters(), "lr": lr},
            {"params": model.output_layer.parameters(), "lr": lr * 2},
            {"params": model.likelihood.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        base_elbo = VariationalELBO(model.likelihood, model, num_data=train_x.shape[0])
        mll = DeepApproximateMLL(base_elbo)

        # Smooth quadratic KL ramp
        beta = max(1e-3, min(0.9, (epoch / (epochs * 1.5)) ** 2))
        base_elbo.beta = beta

        # Gradual increase in MC samples
        num_samples = int(initial_num_samples + (final_num_samples - initial_num_samples) * (epoch / epochs))
        num_samples = min(final_num_samples, max(initial_num_samples, num_samples))

        with gpytorch.settings.num_likelihood_samples(num_samples):
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss = loss.mean() if loss.ndim > 0 else loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # Detach & cleanup
        loss_val = float(loss.detach())
        del output, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Clamp hyperparameters
        for m in [model.compressor_layer, model.output_layer]:
            if hasattr(m, "covar_module"):
                m.covar_module.outputscale.data.clamp_(min=0.3, max=5.0)
        if hasattr(model.likelihood, "noise_covar"):
            model.likelihood.noise_covar.noise.clamp_(1e-5, 5e0)

        # --- Early stopping ---
        if loss_val + min_delta < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch + 1:04d}] Loss={loss_val:.3f} | β={beta:.3f} | samples={num_samples}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"⏹️ Early stopping at epoch {epoch + 1} (no improvement in {patience} epochs).")
            break

    model.eval()
    return model


def train_deepgp_light_stable_v2(
    model,
    train_x,
    train_y,
    lr: float = 7e-4,
    epochs: int = 300,
    initial_num_samples: int = 8,
    final_num_samples: int = 40,  # smaller cap
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    patience: int = 25,
    min_delta: float = 1e-3,
    reinit_on_plateau: bool = False,
    verbose: bool = True,
):
    """
    Stable DeepGP training:
    - Faster β ramp to reach meaningful values early
    - Hard sample cap (<=40)
    - Early stopping + optional reinit on plateau
    """
    model.train()
    optimizer = torch.optim.Adam([
        {"params": model.compressor_layer.parameters(), "lr": lr},
        {"params": model.output_layer.parameters(), "lr": lr * 2},
        {"params": model.likelihood.parameters(), "lr": lr},
    ], weight_decay=weight_decay)

    best_loss = float("inf")
    epochs_no_improve = 0
    plateau_triggered = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        base_elbo = VariationalELBO(model.likelihood, model, num_data=train_x.shape[0])
        mll = DeepApproximateMLL(base_elbo)

        # ✅ Faster quadratic+linear β ramp (reaches ~1.0 near ⅔ of epochs)
        frac = epoch / max(1, epochs)
        beta = max(1e-3, min(1.0, (frac ** 1.5) * 1.5))
        base_elbo.beta = beta

        # ✅ Sample cap at 40
        num_samples = int(initial_num_samples + (final_num_samples - initial_num_samples) * frac)
        num_samples = min(final_num_samples, max(initial_num_samples, num_samples))

        with gpytorch.settings.num_likelihood_samples(num_samples):
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss = loss.mean() if loss.ndim > 0 else loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        loss_val = float(loss.detach())
        del output, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Clamp and regularize
        for m in [model.compressor_layer, model.output_layer]:
            if hasattr(m, "covar_module"):
                m.covar_module.outputscale.data.clamp_(min=0.3, max=5.0)
        if hasattr(model.likelihood, "noise_covar"):
            model.likelihood.noise_covar.noise.clamp_(1e-5, 5.0)

        # --- Early stopping ---
        if loss_val + min_delta < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Optional reinitialization trigger
        if reinit_on_plateau and not plateau_triggered and epochs_no_improve >= patience // 2:
            print("🔁 Plateau detected — reinitializing inducing points mid-training.")
            reinitialize_inducing_points(model, train_x)
            plateau_triggered = True

        if verbose and (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch + 1:04d}] Loss={loss_val:.3f} | β={beta:.3f} | samples={num_samples}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"⏹️ Early stopping at epoch {epoch + 1} (no improvement in {patience} epochs).")
            break

    model.eval()
    return model


def train_deepgp_light_new(
    model,
    train_x,
    train_y,
    lr: float = 1e-3,
    epochs: int = 300,
    initial_num_samples: int = 16,
    weight_decay: float = 0.0,
    final_num_samples: int = 64,
    clip_grad: float = 1.0,
    verbose: bool = True,
):
    model.train()
    optimizer = torch.optim.Adam([
        {"params": model.output_layer.covar_module.parameters(), "lr": 3 * lr},
        {"params": model.output_layer.mean_module.parameters(), "lr": 3 * lr},
        {"params": model.compressor_layer.parameters(), "lr": lr},
        {"params": model.likelihood.parameters(), "lr": lr},
    ], lr=lr)

    base_elbo = VariationalELBO(model.likelihood, model, num_data=train_x.shape[0])
    mll = DeepApproximateMLL(base_elbo)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Monte Carlo ramp-up
        num_samples = int(initial_num_samples + (final_num_samples - initial_num_samples) * (epoch / epochs))
        num_samples = max(initial_num_samples, min(num_samples, final_num_samples))

        # Smooth KL annealing
        base_elbo.beta = max(1e-3, min(1.0, (epoch / epochs) ** 1.5))

        with gpytorch.settings.num_likelihood_samples(num_samples), gpytorch.settings.cholesky_jitter(1e-4):
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            # ⚠️ Ensure scalar loss
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # Keep hyperparams healthy
        for m in [model.compressor_layer, model.output_layer]:
            if hasattr(m, "covar_module"):
                m.covar_module.outputscale.data.clamp_(min=0.5)
        if hasattr(model.likelihood, "noise_covar"):
            model.likelihood.noise_covar.noise.clamp_(1e-5, 1e2)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"[Epoch {epoch + 1:04d}] Loss={loss.item():.3f} | β={base_elbo.beta:.3f} | samples={num_samples}")

    model.eval()
    return model


def train_deepgp_light_fast(
    model,
    train_x,
    train_y,
    lr: float = 7e-4,
    epochs: int = 80,  # 60–100 for init
    initial_num_samples: int = 4,  # 4 → 12 is plenty
    final_num_samples: int = 12,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    patience: int = 15,
    min_delta: float = 1e-3,
    batch_size: int = None,
    verbose: bool = True,
):
    model.train()
    params = [
        {"params": model.compressor_layer.parameters(), "lr": lr},
        {"params": model.output_layer.parameters(), "lr": lr * 2},
        {"params": model.likelihood.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)

    best = float("inf")
    noimp = 0
    N = train_x.shape[0]
    bs = batch_size or min(N, 8)
    xs = train_x.split(bs)
    ys = train_y.split(bs)

    for epoch in range(epochs):
        total = 0.0
        frac = epoch / max(1, epochs)
        beta = max(1e-3, min(1.0, (frac ** 1.3) * 1.4))
        num_s = int(initial_num_samples + (final_num_samples - initial_num_samples) * frac)
        num_s = max(initial_num_samples, min(final_num_samples, num_s))

        for xb, yb in zip(xs, ys):
            optimizer.zero_grad()
            base_elbo = VariationalELBO(model.likelihood, model, num_data=N)
            mll = DeepApproximateMLL(base_elbo)
            base_elbo.beta = beta

            with gpytorch.settings.num_likelihood_samples(num_s), gpytorch.settings.fast_pred_var():
                out = model(xb)
                loss = -mll(out, yb.squeeze(-1))
                loss = loss.mean() if loss.ndim > 0 else loss
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total += float(loss.detach())

        avg = total / len(xs)
        if avg + min_delta < best:
            best = avg
            noimp = 0
        else:
            noimp += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"[Epoch {epoch+1:03d}] Loss={avg:.3f} | β={beta:.3f} | samples={num_s}")

        if noimp >= patience:
            if verbose:
                print(f"⏹️ early stop @ {epoch+1}")
            break

    print("\n[Model diagnostics]")
    print(
        f" compressor: ls={model.compressor_layer.covar_module.base_kernel.lengthscale.mean():.3f}, "
        f"os={model.compressor_layer.covar_module.outputscale.mean():.3f}"
    )
    print(
        f" output: ls={model.output_layer.covar_module.base_kernel.lengthscale.mean():.3f}, "
        f"os={model.output_layer.covar_module.outputscale.mean():.3f}"
    )
    print(f" likelihood noise={model.likelihood.noise.item():.3e}\n")

    model.eval()
    return model


def finetune_last_layer_fast(model, X_scaled, y_scaled, lr=4e-4, epochs=20, verbose=False):
    freeze_feature_extractor(model, True)  # ← freeze latent layer

    # Only train output layer + likelihood
    optimizer = torch.optim.Adam([
        {"params": model.output_layer.parameters(), "lr": lr * 2},
        {"params": model.likelihood.parameters(), "lr": lr},
    ], weight_decay=1e-4)

    model.train()
    N = X_scaled.shape[0]
    for ep in range(epochs):
        optimizer.zero_grad()
        base_elbo = VariationalELBO(model.likelihood, model, num_data=N)
        mll = DeepApproximateMLL(base_elbo)
        base_elbo.beta = 0.5  # steady, no heavy ramp

        with gpytorch.settings.num_likelihood_samples(6), gpytorch.settings.fast_pred_var():
            out = model(X_scaled)
            loss = -mll(out, y_scaled.squeeze(-1))
            loss = loss.mean() if loss.ndim > 0 else loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if verbose and (ep+1) % 10 == 0:
            print(f"[ft {ep+1:02d}] Loss={float(loss):.3f}")

    model.eval()
    freeze_feature_extractor(model, False)  # leave unfrozen for next init if needed
    return model


def run_bo_ackley_original():
    torch.manual_seed(0)
    device = "cpu"

    # Problem setup
    ackley = Ackley(dim=4, negate=True).to(device)
    bounds = ackley.bounds.to(device)
    n_init = 10
    n_iter = 40

    # Initial data
    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # Standardize
        X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
        X_scaled = (train_x - X_mean) / X_std
        y_scaled = (train_y - y_mean) / y_std

        model = RWGSDeepGP(
            train_x_shape=X_scaled.shape,
            X1_dims=X_scaled.shape[-1],
            num_hidden_dims=6,
            num_inducing=min(64, train_x.shape[0]),
            inducing_source=X_scaled,
        )
        model = train_deepgp_light(
            model,
            train_x=X_scaled,
            train_y=y_scaled,
            lr=5e-4,
            epochs=300,
            initial_num_samples=8,
            final_num_samples=64,
            weight_decay=1e-4,
        )
        model.eval()

        best_f = ((train_y.max() - y_mean) / y_std).item()
        scaled_bounds = torch.stack([(bounds[0] - X_mean) / X_std, (bounds[1] - X_mean) / X_std])
        acqf = LogExpectedImprovement(model=model, best_f=best_f)

        x_test = torch.zeros(1, 1, 4, dtype=torch.float64)
        print("EI at center:", acqf(x_test))

        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=20,  # a few more restarts helps on Ackley
            raw_samples=256,
        )

        # Unscale and evaluate
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        print(new_x)
        print(new_y)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


def run_bo_ackley_warm():
    torch.manual_seed(0)
    device = "cpu"

    # Problem setup
    ackley = Ackley(dim=4, negate=True).to(device)
    bounds = ackley.bounds.to(device)
    n_init = 10
    n_iter = 40

    # Initial data
    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    # === Initial standardization ===
    X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
    y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
    X_scaled = (train_x - X_mean) / X_std
    y_scaled = (train_y - y_mean) / y_std

    # === Initialize DeepGP once ===
    model = RWGSDeepGP(
        train_x_shape=X_scaled.shape,
        X1_dims=X_scaled.shape[-1],
        num_hidden_dims=6,
        num_inducing=min(64, train_x.shape[0]),
        inducing_source=X_scaled,
    )
    model = train_deepgp_light(
        model,
        train_x=X_scaled,
        train_y=y_scaled,
        lr=5e-4,
        epochs=400,
        initial_num_samples=8,
        final_num_samples=64,
        weight_decay=1e-4,
    )
    print("✅ Initial DeepGP trained. Starting BO loop...")

    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # Standardize with *current* data
        X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
        X_scaled = (train_x - X_mean) / X_std
        y_scaled = (train_y - y_mean) / y_std

        # Fine-tune existing model on updated data (warm start)
        model.train()
        model = train_deepgp_light(
            model,
            train_x=X_scaled,
            train_y=y_scaled,
            lr=3e-4,
            epochs=150,  # only short fine-tuning
            initial_num_samples=16,
            final_num_samples=64,
            verbose=False,
        )
        model.eval()

        if (iteration + 1) % 5 == 0:
            reinitialize_inducing_points(model, X_scaled)

        # Compute best_f in scaled space
        best_f = ((train_y.max() - y_mean) / y_std).item()

        # Analytic LogEI with variance inflation handled in posterior()
        acqf = LogExpectedImprovement(model=model, best_f=best_f)
        scaled_bounds = torch.stack([
            (bounds[0] - X_mean) / X_std,
            (bounds[1] - X_mean) / X_std
        ])

        # Optimize acquisition
        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
            options={"batch_limit": 10, "maxiter": 100},
        )

        # Unscale and evaluate
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)
        print("Next point:", new_x)
        print("Value:", new_y)

        # Update data
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

        # Optional: occasionally reinit inducing points
        if (iteration + 1) % 10 == 0:
            reinitialize_inducing_points(model, (train_x - X_mean) / X_std)

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


import matplotlib.pyplot as plt
from botorch.utils.transforms import normalize, unnormalize


def plot_bo_state_1d(
    model,
    ackley,
    X_scaled,
    y_scaled,
    X_mean,
    X_std,
    y_mean,
    y_std,
    bounds,
    best_f,
    iteration,
    show_af=True,
):
    import matplotlib.pyplot as plt
    from botorch.acquisition.analytic import LogExpectedImprovement

    model.eval()

    # Dense grid in ORIGINAL space
    Xtest = torch.linspace(bounds[0].item(), bounds[1].item(), 400).unsqueeze(-1)
    Xtest_scaled = (Xtest - X_mean) / X_std

    # Use model posterior (no gradients needed)
    with torch.no_grad():
        post = model.posterior(Xtest_scaled.unsqueeze(1), enable_grad=False)
        mean = post.mean.squeeze(0).view(-1)
        var = post.variance.squeeze(0).view(-1)

    # Backtransform to original y-scale
    mean_plot = mean * y_std + y_mean
    std_plot = var.clamp_min(1e-12).sqrt() * y_std

    # Ground truth y_true
    y_true = ackley(Xtest).squeeze()

    # Acquisition on the same grid (in scaled space)
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    with torch.no_grad():
        af_vals = acqf(Xtest_scaled.unsqueeze(1)).view(-1)

    # --- Quick diagnostics ---
    print(f"[diag] mean range: {mean.min().item():.4f} .. {mean.max().item():.4f} (scaled)")
    print(f"[diag] var range : {var.min().item():.6f} .. {var.max().item():.6f} (scaled)")

    xL = ((bounds[0] + 0.1 * (bounds[1] - bounds[0]))).view(1, 1, 1)
    xR = ((bounds[0] + 0.9 * (bounds[1] - bounds[0]))).view(1, 1, 1)
    xL_s = (xL - X_mean) / X_std
    xR_s = (xR - X_mean) / X_std
    with torch.no_grad():
        mL = model.posterior(xL_s).mean.item()
        mR = model.posterior(xR_s).mean.item()
    print(f"[diag] mean at left / right (scaled): {mL:.4f} / {mR:.4f}")

    # --- Convert to numpy for plotting ---
    Xtest_np = Xtest.detach().cpu().numpy().squeeze()
    mean_np = mean_plot.detach().cpu().numpy()
    std_np = std_plot.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    af_np = af_vals.detach().cpu().numpy()
    X_samples_np = (X_mean + X_scaled * X_std).detach().cpu().numpy()
    y_samples_np = (y_scaled * y_std + y_mean).detach().cpu().numpy()

    with torch.no_grad():
        post = model.posterior(Xtest_scaled.unsqueeze(1))
        print(f"[plot debug] posterior mean std={post.mean.std():.3f}, var mean={post.variance.mean():.3f}")

    # --- Plot ---
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
    axs[1].fill_between(
        Xtest_np, mean_np - 2 * std_np, mean_np + 2 * std_np, alpha=0.2, color="blue", label="±2σ",
    )
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


def run_bo_ackley_warm_1d(n_plot_interval=5):
    torch.manual_seed(0)
    device = "cpu"

    ackley = Ackley(dim=1, negate=True).to(device)
    bounds = ackley.bounds.to(device)
    n_init, n_iter = 3, 25

    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    # --- Standardize ---
    X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
    y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
    X_scaled = (train_x - X_mean) / X_std
    y_scaled = (train_y - y_mean) / y_std

    # Ensure dtype consistency
    train_x, train_y, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds = ensure_dtype_consistency(
        train_x, train_y, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds
    )

    # --- Train initial DeepGP ---
    model = RWGSDeepGP(
        train_x_shape=X_scaled.shape,
        X1_dims=1,
        num_hidden_dims=8,
        num_inducing=8,
        inducing_source=X_scaled,
    )
    with gpytorch.settings.cholesky_jitter(1e-3):
        model = train_deepgp_light_stable_v2(
            model, X_scaled, y_scaled, lr=7e-4, epochs=300, patience=30, reinit_on_plateau=True
        )

    # ✅ Plot immediately after initial training
    print("\n📊 Plotting initial model state...")
    best_f_init = ((train_y.max() - y_mean) / y_std).item()
    plot_bo_state_1d(
        model, ackley, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds, best_f_init, iteration=0
    )

    # --- BO loop ---
    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # re-standardize
        X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
        X_scaled = (train_x - X_mean) / X_std
        y_scaled = (train_y - y_mean) / y_std

        with gpytorch.settings.cholesky_jitter(1e-3):
            # model = finetune_last_layer_fast(
            #     model, X_scaled, y_scaled,
            #     lr=4e-4, epochs=20, verbose=False
            # )
            model = train_deepgp_light_stable_v2(
                model, X_scaled, y_scaled, lr=7e-4, epochs=300, patience=30, reinit_on_plateau=True
            )

        # Reinitialize inducing points every 5 iterations
        if (iteration + 1) % 5 == 0:
            reinitialize_inducing_points(model, X_scaled)

        # --- Acquisition optimization ---
        best_f = ((train_y.max() - y_mean) / y_std).item()
        acqf = LogExpectedImprovement(model=model, best_f=best_f)
        scaled_bounds = torch.stack([
            (bounds[0] - X_mean) / X_std,
            (bounds[1] - X_mean) / X_std
        ])

        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
            options={"batch_limit": 5, "maxiter": 50},
        )

        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        # Update dataset
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

        # ✅ Plot every n_plot_interval iterations
        if (iteration + 1) % n_plot_interval == 0:
            print(f"\n📈 Plotting BO progress after {iteration + 1} iterations...")
            plot_bo_state_1d(
                model, ackley, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds, best_f, iteration + 1
            )

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


# def plot_bo_state_umap_1d(
#     model,
#     ackley,
#     X_scaled,
#     y_scaled,
#     X_mean,
#     X_std,
#     y_mean,
#     y_std,
#     bounds,
#     best_f,
#     iteration,
#     show_af=True,
#     n_grid: int = 400,
# ):
#     """
#     Visualize BO state for 4D problem using 1D UMAP embedding.
#
#     Compresses 4D inputs to 1D latent axis and plots:
#     1) true Ackley values at training points
#     2) DeepGP posterior mean ±2σ
#     3) Acquisition function (LogEI)
#     """
#
#     model.eval()
#     device = X_scaled.device
#
#     # --- Step 1: Fit UMAP on existing training data ---
#     umap_1d = UMAP(n_components=1, random_state=0, n_neighbors=10, min_dist=0.1)
#     umap_1d.fit(X_scaled.cpu().numpy())
#     X_umap = torch.tensor(umap_1d.transform(X_scaled.cpu()).squeeze(), dtype=torch.float64)
#
#     # --- Step 2: Generate a grid in latent space ---
#     u_min, u_max = X_umap.min().item(), X_umap.max().item()
#     Utest = torch.linspace(u_min, u_max, n_grid).unsqueeze(-1)
#
#     # --- Step 3: Inverse-transform via nearest-neighbor interpolation ---
#     # We’ll use local linear interpolation in the 1D UMAP space
#     from sklearn.neighbors import KNeighborsRegressor
#     knn = KNeighborsRegressor(n_neighbors=min(5, len(X_scaled)))
#     knn.fit(X_umap.cpu().numpy(), X_scaled.cpu().numpy())
#     Xtest_scaled = torch.tensor(knn.predict(Utest.cpu().numpy()), dtype=torch.float64, device=device)
#
#     # --- Step 4: Model predictions ---
#     with torch.no_grad(), gpytorch.settings.num_likelihood_samples(128):
#         preds = model.likelihood(model(Xtest_scaled))
#         mu, var = preds.mean, preds.variance
#
#     if mu.dim() > 1 and mu.shape[0] > 1:
#         mean = mu.mean(0)
#         var = (var + mu ** 2).mean(0) - mean ** 2
#     else:
#         mean = mu
#         var = var
#
#     mean = mean.view(-1) * y_std + y_mean
#     std = var.clamp_min(1e-12).sqrt().view(-1) * y_std
#
#     # --- Step 5: True Ackley on interpolated inputs ---
#     Xtest = X_mean + Xtest_scaled * X_std
#     y_true = ackley(Xtest).squeeze()
#
#     # --- Step 6: Acquisition function ---
#     acqf = LogExpectedImprovement(model=model, best_f=best_f)
#     with torch.no_grad():
#         af_vals = acqf(Xtest_scaled.unsqueeze(1)).squeeze()
#
#     # --- Step 7: Plot everything ---
#     fig, axs = plt.subplots(3 if show_af else 2, 1, figsize=(8, 8), sharex=True)
#     fig.suptitle(f"DeepGP BO (UMAP-1D view, Iter {iteration})", fontsize=14, y=0.95)
#
#     # 1️⃣ Ground truth + samples
#     axs[0].plot(Utest, y_true, "k--", label="True Ackley")
#     axs[0].scatter(X_umap, y_scaled * y_std + y_mean, color="red", label="Samples", zorder=5)
#     axs[0].set_ylabel("f(x)")
#     axs[0].legend()
#     axs[0].set_title("Objective & Samples (UMAP-1D)")
#
#     # 2️⃣ Posterior mean ± 2σ
#     axs[1].plot(Utest, mean, "b", label="Posterior mean")
#     axs[1].fill_between(Utest.squeeze(), mean - 2 * std, mean + 2 * std,
#                         color="blue", alpha=0.2, label="±2σ")
#     axs[1].scatter(X_umap, y_scaled * y_std + y_mean, color="red", s=25, label="Samples")
#     axs[1].set_ylabel("Predicted mean")
#     axs[1].set_title("DeepGP Posterior (1D UMAP)")
#
#     # 3️⃣ Acquisition
#     if show_af:
#         axs[2].plot(Utest, af_vals, "g", label="LogEI")
#         axs[2].set_xlabel("UMAP-1D embedding")
#         axs[2].set_ylabel("Acquisition")
#         axs[2].legend()
#         axs[2].set_title("Acquisition Function (1D projection)")
#
#     plt.tight_layout()
#     plt.show()


def run_bo_ackley_warm_umap_interactive(n_plot_interval: int = 5):
    """
    4D Ackley Bayesian Optimization with DeepGP and interactive 1D UMAP visualization.
    Uses Plotly for interactive plotting of the DeepGP posterior, true function, and acquisition function.
    """
    import umap
    import plotly.graph_objects as go
    from sklearn.neighbors import KNeighborsRegressor

    torch.manual_seed(0)
    device = "cpu"

    # === Problem setup ===
    ackley = Ackley(dim=4, negate=True).to(device)
    bounds = ackley.bounds.to(device)
    n_init = 10
    n_iter = 25

    # === Initial data ===
    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    # === Standardize ===
    X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
    y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
    X_scaled = (train_x - X_mean) / X_std
    y_scaled = (train_y - y_mean) / y_std

    # === Initialize DeepGP ===
    model = RWGSDeepGP(
        train_x_shape=X_scaled.shape,
        X1_dims=X_scaled.shape[-1],
        num_hidden_dims=6,
        num_inducing=min(64, train_x.shape[0]),
        inducing_source=X_scaled,
    )
    model = train_deepgp_light(
        model,
        train_x=X_scaled,
        train_y=y_scaled,
        lr=5e-4,
        epochs=400,
        initial_num_samples=8,
        final_num_samples=64,
        weight_decay=1e-4,
    )
    model.eval()
    print("✅ Initial DeepGP trained. Starting BO loop...")

    # === Fixed UMAP for consistent visualization ===
    umap_1d = umap.UMAP(n_components=1, n_neighbors=min(15, len(X_scaled)-1), random_state=0)
    umap_1d.fit(X_scaled.cpu().numpy())

    # === BO loop ===
    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # Standardize
        X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
        X_scaled = (train_x - X_mean) / X_std
        y_scaled = (train_y - y_mean) / y_std

        # Warm-start fine-tune
        model.train()
        model = train_deepgp_light(
            model,
            train_x=X_scaled,
            train_y=y_scaled,
            lr=3e-4,
            epochs=150,
            initial_num_samples=16,
            final_num_samples=64,
            verbose=False,
        )
        model.eval()

        # Periodic reinit
        if (iteration + 1) % 5 == 0:
            reinitialize_inducing_points(model, X_scaled)

        # Compute acquisition
        best_f = ((train_y.max() - y_mean) / y_std).item()
        acqf = LogExpectedImprovement(model=model, best_f=best_f)
        scaled_bounds = torch.stack([
            (bounds[0] - X_mean) / X_std,
            (bounds[1] - X_mean) / X_std
        ])

        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
        )

        # Unscale and evaluate
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

        # Interactive plot every few iterations
        if (iteration) % n_plot_interval == 0:
            plot_umap_state_matplotlib(
                model, ackley, X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, bounds, best_f, iteration + 1,
                umap_1d, device="cpu"
            )

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


def plot_umap_state_matplotlib(
    model,
    ackley,
    X_scaled,
    y_scaled,
    X_mean,
    X_std,
    y_mean,
    y_std,
    bounds,
    best_f,
    iteration,
    umap_1d=None,
    device="cpu",
    refit_umap_every: int = 5,
):
    """
    Stable visualization of BO progress using a 1D latent embedding.
    - Re-fits UMAP every refit_umap_every iterations.
    - Falls back to PCA if UMAP collapses (too low variance).
    - Normalizes Ackley curve for visualization.
    """
    import matplotlib.pyplot as plt
    import torch
    import gpytorch
    import umap
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.decomposition import PCA
    from botorch.acquisition.analytic import LogExpectedImprovement

    model.eval()

    # --- Refit UMAP (or PCA fallback) ---
    if umap_1d is None or (iteration % refit_umap_every == 0):
        n_neighbors = min(15, max(3, len(X_scaled) - 1))
        umap_1d = umap.UMAP(
            n_components=1,
            n_neighbors=n_neighbors,
            random_state=0,
            min_dist=0.1,
        )
        X_umap_np = umap_1d.fit_transform(X_scaled.cpu().numpy())
        X_umap = torch.tensor(X_umap_np.squeeze(), dtype=torch.float64, device=device)

        # 🚨 Fallback to PCA if UMAP is degenerate (flat line)
        if torch.std(X_umap) < 1e-4:
            print(f"[⚠️ UMAP collapsed at iter {iteration}] → falling back to PCA(1)")
            pca_1d = PCA(n_components=1)
            X_umap_np = pca_1d.fit_transform(X_scaled.cpu().numpy())
            X_umap = torch.tensor(X_umap_np.squeeze(), dtype=torch.float64, device=device)
            umap_1d = pca_1d  # store PCA object for later reuse
    else:
        # reuse existing UMAP or PCA
        X_umap_np = umap_1d.transform(X_scaled.cpu().numpy())
        X_umap = torch.tensor(X_umap_np.squeeze(), dtype=torch.float64, device=device)

    # --- Define latent grid ---
    u_min, u_max = X_umap.min().item(), X_umap.max().item()
    Utest = torch.linspace(u_min, u_max, 400).unsqueeze(-1)

    # --- Interpolate latent → 4D scaled inputs ---
    knn = KNeighborsRegressor(n_neighbors=min(5, len(X_scaled)))
    knn.fit(X_umap.cpu().numpy().reshape(-1, 1), X_scaled.cpu().numpy())
    Xtest_scaled = torch.tensor(
        knn.predict(Utest.cpu().numpy().reshape(-1, 1)),
        dtype=torch.float64,
        device=device,
    )

    # --- DeepGP posterior predictions ---
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(128):
        preds = model.likelihood(model(Xtest_scaled))
        mu, var = preds.mean, preds.variance

    if mu.dim() > 1 and mu.shape[0] > 1:
        mean = mu.mean(0)
        var = (var + mu ** 2).mean(0) - mean ** 2
    else:
        mean = mu
        var = var

    mean = mean.view(-1) * y_std + y_mean
    std = var.clamp_min(1e-12).sqrt().view(-1) * y_std

    # --- Acquisition function ---
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    with torch.no_grad():
        af_vals = acqf(Xtest_scaled.unsqueeze(1)).squeeze()

    # --- True Ackley (normalized for visual comparability) ---
    with torch.no_grad():
        Xtest_unscaled = X_mean + Xtest_scaled * X_std
        y_true = ackley(Xtest_unscaled).squeeze()
        y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min())
        y_true_norm = y_true_norm * (mean.max() - mean.min()) + mean.min()

    # --- Plot ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle(f"Latent 1D BO Progress (Iter {iteration})", fontsize=14, y=0.96)

    # 1️⃣ True Ackley (scaled)
    axs[0].plot(Utest.cpu(), y_true_norm.cpu(), "k--", lw=1.5, label="True Ackley (scaled for visualization)")
    axs[0].scatter(X_umap.cpu(), (y_scaled * y_std + y_mean).cpu(), c="red", s=25, label="Samples")
    axs[0].set_ylabel("f(x)")
    axs[0].legend()
    axs[0].set_title("Objective (UMAP/PCA 1D projection)")

    # 2️⃣ DeepGP posterior
    axs[1].plot(Utest.cpu(), mean.cpu(), "b", lw=2, label="Posterior mean")
    axs[1].fill_between(
        Utest.squeeze().cpu(), (mean - 2 * std).cpu(), (mean + 2 * std).cpu(),
        color="blue", alpha=0.2, label="±2σ",
    )
    axs[1].scatter(X_umap.cpu(), (y_scaled * y_std + y_mean).cpu(), color="red", s=25, label="Samples")
    axs[1].set_ylabel("Predicted mean")
    axs[1].legend()
    axs[1].set_title("DeepGP Posterior")

    # 3️⃣ Acquisition function
    axs[2].plot(Utest.cpu(), af_vals.cpu(), "g", lw=2, label="LogEI")
    axs[2].set_xlabel("Latent 1D coordinate")
    axs[2].set_ylabel("Acquisition")
    axs[2].set_title("Acquisition Function (LogEI)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    return umap_1d


def run_gp_bo_ackley():
    torch.manual_seed(0)
    device = "cpu"

    # Problem setup
    ackley = Ackley(dim=1, negate=True).to(device)
    bounds = ackley.bounds.to(device)
    n_init = 3
    n_iter = 40

    # Initial data
    train_x, train_y = init_data(ackley, n_init=n_init)
    best_y = [train_y.max().item()]

    for iteration in range(n_iter):
        print(f"\n=== BO Iteration {iteration + 1}/{n_iter} ===")

        # Standardize data
        X_mean, X_std = train_x.mean(0), train_x.std(0).clamp_min(1e-12)
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-12)
        X_scaled = (train_x - X_mean) / X_std
        y_scaled = (train_y - y_mean) / y_std

        # Fit model
        model = SingleTaskGP(train_X=X_scaled, train_Y=y_scaled)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Compute best_f in scaled space
        best_f = ((train_y.max() - y_mean) / y_std).item()

        # Acquisition
        acqf = LogExpectedImprovement(model=model, best_f=best_f)

        # Optimize in scaled space
        scaled_bounds = torch.stack([
            (bounds[0] - X_mean) / X_std,
            (bounds[1] - X_mean) / X_std
        ])
        new_x_scaled, _ = optimize_acqf(
            acq_function=acqf,
            bounds=scaled_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        # Unscale and evaluate
        new_x = X_mean + new_x_scaled * X_std
        new_y = ackley(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_y.append(train_y.max().item())
        print(f"→ Best value so far: {best_y[-1]:.4f}")

    print("\n=== Optimization Finished ===")
    print(f"Final best value: {max(best_y):.4f}")


def main():
    run_bo_ackley_warm_1d()


if __name__ == '__main__':
    main()
