import os
import math
import torch
import tqdm
import gpytorch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List

from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
)
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.test_functions import Ackley

# os.environ['CI'] = '1'
#
# --- Configuration ---
CONFIG = {
    "num_epochs": 1 if 'CI' in os.environ else 20,
    "num_datapoints": 200,
    "num_hidden_dims": 6,
    "num_inducing": 128,
    "learning_rate": 0.1,
    "seed": 42,
}

SETUP = {
    "noise_ratio": 0.0,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Data Generation ---
def generate_data(num_points: int, noise_ratio: float):
    """Generates synthetic data and standardizes it."""
    target_function = Ackley(dim=1)
    train_x = torch.linspace(0, 1, num_points)

    scaled_input = 32.768 * (2 * train_x.unsqueeze(-1) - 1)
    base_y = target_function(scaled_input)  # / 5 - 2.5
    noise = torch.randn(base_y.size()) * noise_ratio

    # Raw target
    train_y = torch.stack([base_y + noise], -1)

    # --- FEATURE: Standardization ---
    # Calculate mean and std
    train_y_mean = train_y.mean()
    train_y_std = train_y.std()

    # Standardize (Z-score normalization)
    train_y_standardized = (train_y - train_y_mean) / train_y_std

    train_x = train_x.unsqueeze(-1)

    # Return the standardized data AND the scaler stats
    return train_x, train_y_standardized, train_y_mean, train_y_std

# --- Model Definition ---
class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims: int, output_dims: int, num_inducing: int = 128, linear_mean: bool = True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        if linear_mean:
            self.mean_module = LinearMean(input_dims)

            # IDENTITY INITIALIZATION
            self.mean_module.weights.data.fill_(0.0)
            self.mean_module.bias.data.fill_(0.0)
            min_dim = min(input_dims, output_dims)
            self.mean_module.weights.data[..., :min_dim, :min_dim].diagonal(dim1=-2, dim2=-1).fill_(1.0)
        else:
            self.mean_module = ConstantMean()

        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overridden to support skip-connections (concatenating inputs).
        This allows passing the original input 'x' to deeper layers.
        """
        if len(other_inputs):
            # If the input is a distribution (from previous DGP layer), sample from it
            if isinstance(x, MultitaskMultivariateNormal) or isinstance(x, MultivariateNormal):
                x = x.rsample()

            # Expand other inputs (like the original train_x) to match likelihood samples
            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            # Concatenate along the feature dimension
            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class MultitaskDeepGP(DeepGP):
    def __init__(self, input_dim: int, num_tasks: int, hidden_dims: int):
        super().__init__()

        # Layer 1: Expansion to hidden dimensions
        self.hidden_layer = DGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dims,
            linear_mean=True
        )

        # Layer 2: Hidden + Original Input -> Output Tasks
        # (Skip connection architecture)
        self.last_layer = DGPHiddenLayer(
            input_dims=hidden_dims + input_dim,
            output_dims=num_tasks,
            linear_mean=False
        )

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        # Pass through first layer
        hidden_rep1 = self.hidden_layer(inputs)

        # Pass through second layer, concatenating original inputs (Skip connection)
        output = self.last_layer(hidden_rep1, inputs)
        return output

    def predict(self, test_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and variance for test data."""
        with torch.no_grad():
            # to_data_independent_dist removes cross-covariance terms for marginal prediction
            preds = self.likelihood(self(test_x)).to_data_independent_dist()

        # Average over the Monte Carlo samples (dimension 0)
        return preds.mean.mean(0), preds.variance.mean(0)


# --- Training Routine ---
def train_model(model: MultitaskDeepGP, train_x: torch.Tensor, train_y: torch.Tensor, epochs: int, lr: float):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch")

    loss_history = []

    for _ in epochs_iter:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        epochs_iter.set_postfix(loss=loss.item())

    return loss_history


def plot_results(model: MultitaskDeepGP, train_x, train_y, y_mean, y_std):
    model.eval()
    num_tasks = train_y.size(-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 100).unsqueeze(-1)
        mean, var = model.predict(test_x)

        # --- FEATURE: Un-Standardize Predictions ---
        # Scale back to original dimensions for plotting
        mean = mean * y_std + y_mean
        std = var.sqrt() * y_std
        lower = mean - 2 * std
        upper = mean + 2 * std

        # Also un-standardize the training data for the scatter plot
        train_y_original = train_y * y_std + y_mean

    # Convert to numpy
    train_x_np = train_x.squeeze(-1).numpy()
    train_y_np = train_y_original.numpy()  # Use the un-scaled data
    test_x_np = test_x.squeeze(-1).numpy()
    mean_np = mean.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()

    # Create Subplots
    fig = make_subplots(rows=1, cols=num_tasks, subplot_titles=[f'Task {t + 1}' for t in range(num_tasks)])

    for task in range(num_tasks):
        # Confidence Interval
        fig.add_trace(go.Scatter(x=test_x_np, y=lower_np[:, task], mode='lines', line=dict(width=0), showlegend=False,
                                 hoverinfo='skip'), row=1, col=task + 1)
        fig.add_trace(go.Scatter(x=test_x_np, y=upper_np[:, task], mode='lines', line=dict(width=0), fill='tonexty',
                                 fillcolor='rgba(0, 0, 255, 0.2)', name='Confidence', hoverinfo='skip'), row=1,
                      col=task + 1)

        # Mean
        fig.add_trace(go.Scatter(x=test_x_np, y=mean_np[:, task], mode='lines', line=dict(color='blue'), name='Mean'),
                      row=1, col=task + 1)

        # Observed Data
        fig.add_trace(
            go.Scatter(x=train_x_np, y=train_y_np[:, task], mode='lines+markers', marker=dict(color='black', size=4),
                       line=dict(color='black', width=1), name='Observed Data'), row=1, col=task + 1)

    fig.update_layout(title="Deep GP Regression (Ackley) - Identity Init & Standardized", height=600,
                      width=800 * num_tasks, showlegend=True, template="plotly_white")
    fig.show()


if __name__ == "__main__":
    set_seed(CONFIG["seed"])

    # 1. Prepare Data (Capturing mean/std)
    train_x, train_y, y_mean, y_std = generate_data(CONFIG["num_datapoints"], SETUP["noise_ratio"])
    print(f"Data Shape: X={train_x.shape}, Y={train_y.shape}")
    print(f"Standardization stats: Mean={y_mean:.4f}, Std={y_std:.4f}")

    # 2. Initialize Model
    input_dim = train_x.shape[-1]
    num_tasks = train_y.shape[-1]

    model = MultitaskDeepGP(
        input_dim=input_dim,
        num_tasks=num_tasks,
        hidden_dims=CONFIG["num_hidden_dims"]
    )

    # 3. Train
    print("Starting training...")
    losses = train_model(model, train_x, train_y, CONFIG["num_epochs"], CONFIG["learning_rate"])

    # 4. Visualize (Passing mean/std)
    print("Plotting results...")
    plot_results(model, train_x, train_y, y_mean, y_std)

    print(f"Likelihood noise: {model.likelihood.noise.item()}")