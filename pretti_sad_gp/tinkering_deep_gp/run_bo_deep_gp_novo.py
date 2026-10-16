import argparse
import math
import os
from typing import Optional

import torch
import gpytorch
import matplotlib.pyplot as plt

from botorch.test_functions.synthetic import Ackley


torch.set_default_dtype(torch.float64)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    """
    A single hidden/output layer for a Deep GP using an RBF kernel.
    Inducing points are provided explicitly for flexible initialization.
    """
    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int],
        inducing_points: torch.Tensor,
    ):
        # output_dims=None means this is the final layer (single task output)
        if output_dims is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([output_dims])

        # Variational distribution / strategy
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        var_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            q_dist,
            learn_inducing_locations=True,
        )

        super().__init__(var_strategy, input_dims=input_dims, output_dims=output_dims)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepGPRegression(gpytorch.models.deep_gps.DeepGP):
    """
    Two-layer Deep GP for 1D -> hidden_dim -> 1 mapping.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        inducing_points_x: torch.Tensor,   # shape [num_inducing, input_dim]
        inducing_points_hidden: torch.Tensor,  # shape [num_inducing, hidden_dim]
        noise_init: float = 1e-3,
    ):
        super().__init__()

        # First layer: 1D -> hidden_dim
        # Expand inducing points for multitask (one set per hidden dimension)
        inducing1 = inducing_points_x.unsqueeze(0).expand(hidden_dim, -1, -1).contiguous()
        self.hidden_layer = DGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            inducing_points=inducing1,
        )

        # Final layer: hidden_dim -> 1D output
        self.last_layer = DGPHiddenLayer(
            input_dims=hidden_dim,
            output_dims=None,
            inducing_points=inducing_points_hidden,
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Small initial observation noise works better on tiny datasets
        with torch.no_grad():
            self.likelihood.noise = torch.as_tensor(noise_init).clamp_min(1e-6)

    def forward(self, x: torch.Tensor):
        h = self.hidden_layer(x)
        out = self.last_layer(h)
        return out

    # Convenience helpers for separating parameter sets
    def variational_parameters(self):
        return self.named_variational_parameters()

    def hyperparameters(self):
        return self.named_hyperparameters()


def build_training_data(n_train: int, seed: int = 0, device: torch.device = torch.device("cpu")):
    torch.manual_seed(seed)
    f = Ackley(dim=1, negate=False)  # standard Ackley, global minimum ~0

    bounds = f.bounds.to(device=device, dtype=torch.get_default_dtype())  # shape [2, 1]
    lb, ub = bounds[0], bounds[1]
    train_x = lb + (ub - lb) * torch.rand(n_train, 1, device=device, dtype=torch.get_default_dtype())
    train_y = f(train_x).view(-1)

    return f, bounds, train_x, train_y


def standardize(y: torch.Tensor):
    mean = y.mean()
    std = y.std().clamp_min(1e-6)
    y_std = (y - mean) / std
    return y_std, mean, std


def destandardize(y_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return y_std * std + mean


def train_model(
    model: DeepGPRegression,
    train_x: torch.Tensor,
    train_y_std: torch.Tensor,
    epochs: int = 800,
    num_likelihood_samples: int = 16,
    lr_ngd: float = 0.1,
    lr_adam: float = 0.01,
    verbose: bool = True,
):
    # Deep variational ELBO
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y_std.numel())
    )

    variational_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), num_data=train_y_std.numel(), lr=lr_ngd
    )
    hyperparameter_optimizer = torch.optim.Adam(
        model.hyperparameters(), lr=lr_adam
    )

    model.train()
    model.likelihood.train()

    for i in range(1, epochs + 1):
        variational_optimizer.zero_grad(set_to_none=True)
        hyperparameter_optimizer.zero_grad(set_to_none=True)

        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            output = model(train_x)
            loss = -mll(output, train_y_std)
        loss.backward()

        variational_optimizer.step()
        hyperparameter_optimizer.step()

        if verbose and (i % max(1, iters // 10) == 0 or i == 1):
            print(f"[{i:04d}/{iters}] ELBO loss: {loss.item():.4f}")

    return model


def make_figure(
    f,
    bounds: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: DeepGPRegression,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    n_grid: int = 400,
    n_posterior_samples: int = 6,
    fig_path: str = "dgp_ackley.png",
):
    model.eval()
    model.likelihood.eval()

    # Dense grid over input bounds
    Xmin = bounds[0, 0].item()
    Xmax = bounds[1, 0].item()
    test_x = torch.linspace(Xmin, Xmax, n_grid, dtype=train_x.dtype, device=train_x.device).unsqueeze(-1)

    # True function
    with torch.no_grad():
        f_true = f(test_x).view(-1).cpu()

    # Predictive distribution (standardized space)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
        pred_dist = model.likelihood(model(test_x))

    mean = pred_dist.mean.view(-1)
    lower, upper = pred_dist.confidence_region()
    # Back to original (unstandardized) scale
    mean = destandardize(mean, y_mean, y_std).cpu()
    lower = destandardize(lower.view(-1), y_mean, y_std).cpu()
    upper = destandardize(upper.view(-1), y_mean, y_std).cpu()

    # Sample posterior functions and de-standardize
    with torch.no_grad():
        samples = pred_dist.rsample(torch.Size([n_posterior_samples]))  # [S, n]
        samples = destandardize(samples, y_mean, y_std).cpu()

    # Matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    # (a) True function + observed samples
    ax = axes[0]
    ax.plot(test_x.cpu().view(-1).numpy(), f_true.numpy(), lw=2, label="Ackley (true)")
    ax.scatter(train_x.cpu().view(-1).numpy(), train_y.cpu().numpy(), s=40, marker="o", label="Observations")
    ax.set_title("(a) Test function and samples")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (b) Deep GP posterior
    ax = axes[1]
    ax.plot(test_x.cpu().view(-1).numpy(), mean.numpy(), lw=2, label="DGP posterior mean")
    ax.fill_between(
        test_x.cpu().view(-1).numpy(),
        lower.numpy(),
        upper.numpy(),
        alpha=0.25,
        label="95% confidence",
        linewidth=0,
    )
    # Draw a few posterior samples
    xs = test_x.cpu().view(-1).numpy()
    for s in range(min(n_posterior_samples, samples.shape[0])):
        ax.plot(xs, samples[s].view(-1).numpy(), lw=1, alpha=0.8)

    # Overlay training points
    ax.scatter(train_x.cpu().view(-1).numpy(), train_y.cpu().numpy(), s=40, marker="o", color="k", label="Observations")

    ax.set_title("(b) Deep GP posterior")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {os.path.abspath(fig_path)}")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n-train", type=int, default=3, help="Number of training points (3–100 recommended)")
    # parser.add_argument("--iters", type=int, default=800, help="Training iterations")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # parser.add_argument("--hidden-dim", type=int, default=2, help="Width of the hidden GP layer")
    # parser.add_argument("--num-inducing", type=int, default=-1, help="Inducing points per layer; -1 = auto (5x n_train, clipped to [16, 128])")
    # parser.add_argument("--double", action="store_true", help="Use float64 for extra stability on tiny datasets")
    # args = parser.parse_args()

    n_train = 3
    seed = 42
    hidden_dim = 6
    epochs = 200

    f, bounds, train_x, train_y = build_training_data(n_train=n_train, seed=seed, device=device)
    train_y_std, y_mean, y_std = standardize(train_y)

    num_inducing = int(max(16, min(128, 5 * n_train)))

    inducing_x = torch.linspace(bounds[0, 0], bounds[1, 0], num_inducing, device=device, dtype=train_x.dtype).unsqueeze(-1)
    inducing_hidden = torch.randn(num_inducing, hidden_dim, device=device, dtype=train_x.dtype) * 0.5

    # Model
    model = DeepGPRegression(
        input_dim=1,
        hidden_dim=hidden_dim,
        inducing_points_x=inducing_x,
        inducing_points_hidden=inducing_hidden,
        noise_init=1e-3,
    ).to(device)

    model = train_model(
        model,
        train_x=train_x,
        train_y_std=train_y_std,
        epochs=epochs,
        num_likelihood_samples=16,
        lr_ngd=0.1,
        lr_adam=0.01,
        verbose=True,
    )

    # Plot
    make_figure(
        f=f,
        bounds=bounds,
        train_x=train_x,
        train_y=train_y,
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        n_grid=400,
        n_posterior_samples=6,
        fig_path="dgp_ackley.png",
    )


if __name__ == "__main__":
    main()
