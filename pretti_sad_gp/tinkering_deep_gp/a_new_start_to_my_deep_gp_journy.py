import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader
from botorch.test_functions import Ackley
from math import floor
import numpy as np
import plotly.graph_objects as go
import os
import torch
import tqdm
import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt


# --- Configuration ---
n_points = 15000  # Reduced for clearer visualization in browser (use 15000 if needed)
num_hidden_dims = 10
num_epochs = 5  # RESTORED: Needs ~100 to fit Ackley
num_samples = 10  # RESTORED: Needs >1 for stable gradients on complex functions

# --- Data Generation ---
target_function = Ackley(dim=1)
X = torch.linspace(-32.768, 32.768, n_points).reshape([n_points, 1])
y = target_function(X).unsqueeze(-1)
num_tasks = y.size(-1)

# --- Scaling ---
# Scale X to [-1, 1]
X_min, X_max = X.min(0)[0], X.max(0)[0]
X_scaled = 2 * ((X - X_min) / (X_max - X_min)) - 1

# Scale y to Mean 0, Std 1
y_mean, y_std = y.mean(0), y.std(0)
y_scaled = (y - y_mean) / y_std

# --- Split ---
train_n = int(floor(0.8 * len(X)))
train_x = X_scaled[:train_n, :].contiguous()
test_x = X_scaled[train_n:, :].contiguous()
train_y = y_scaled[:train_n].contiguous()
test_y_scaled = y_scaled[train_n:].contiguous()
test_y_raw = y[train_n:].contiguous()  # Keep raw for metric calc

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_dataset = TensorDataset(test_x, test_y_scaled)
test_loader = DataLoader(test_dataset, batch_size=1024)


# --- Model Definition ---

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
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
        self.mean_module = LinearMean(input_dims) if linear_mean else ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []

            # Iterate over the loader passed as an argument
            for x_batch, y_batch in loader:
                # Pass the BATCH (x_batch) to the model, not the loader
                preds = self.likelihood(self(x_batch)).to_data_independent_dist()

                # Mean over the DeepGP samples (dimension 0)
                mus.append(preds.mean.mean(0))
                variances.append(preds.variance.mean(0))

                # Fix dimensions for Log Likelihood calculation
                # Multitask/DeepGP expects y to be [batch, tasks], not [batch]
                if y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(-1)

                # Calculate log prob using the predictive distribution
                lls.append(preds.log_prob(y_batch).mean(0))

        return torch.cat(mus), torch.cat(variances), torch.cat(lls)


model = MultitaskDeepGP(train_x.shape)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

for i in epochs_iter:
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

model.eval()

full_dataset = TensorDataset(X_scaled, y_scaled)  # Pass y_scaled just for shape/compatibility
full_loader = DataLoader(full_dataset, batch_size=1024, shuffle=False)  # No shuffle for plotting!

with gpytorch.settings.num_likelihood_samples(num_samples):
    # all_means_scaled, all_vars_scaled, _ = model.predict(full_loader)
    test_means_scaled, _, test_lls = model.predict(test_loader)

# --- Metrics (Test Set Only) ---
nll_standardized = -test_lls.mean()
mse_scaled = torch.mean(torch.pow(test_means_scaled.mean(0) - test_y_scaled, 2))
smse = mse_scaled / test_y_scaled.var()
test_means_raw = (test_means_scaled.mean(0) * y_std) + y_mean
rmse_raw = torch.mean(torch.pow(test_means_raw - test_y_raw, 2)).sqrt()

print("\n" + "-" * 30)
print(f"  SMSE: {smse.item():.4f}")
print(f"  NLL:  {nll_standardized.item():.4f}")
print(f"  RMSE Raw: {rmse_raw.item():.4f}")
print("-" * 30)

# --- Visualizing (Full Domain) ---

# 1. Aggregate Statistics (Scaled Space)
mean_pred_scaled = test_means_scaled.mean(0)
# Total Variance = Expected Variance + Variance of Expectations
total_var_scaled = test_means_scaled.mean(0) + test_means_scaled.var(0)
std_pred_scaled = total_var_scaled.sqrt()

# 2. Un-scale Everything to Original Units
# X-axis
x_plot_raw = X.squeeze().numpy()

# Y-axis (Prediction)
mean_pred_raw = (mean_pred_scaled * y_std) + y_mean
std_pred_raw = std_pred_scaled * y_std  # Scale std, don't shift

# Y-axis (Confidence Intervals)
lower = (mean_pred_raw - 1.96 * std_pred_raw).numpy()
upper = (mean_pred_raw + 1.96 * std_pred_raw).numpy()
mean_line = mean_pred_raw.numpy()

# Y-axis (True Data - Unscale train_x back to original domain)
# Formula: (x_scaled + 1) / 2 * (max - min) + min
train_x_raw = (train_x + 1) / 2 * (X_max - X_min) + X_min
train_y_raw = (train_y * y_std) + y_mean

test_x_raw = (test_x + 1) / 2 * (X_max - X_min) + X_min
test_y_raw_vis = (test_y_scaled * y_std) + y_mean

# --- Plotly Chart ---
fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=np.concatenate([x_plot_raw, x_plot_raw[::-1]]),
#     y=np.concatenate([upper, lower[::-1]]),
#     fill='toself',
#     fillcolor='rgba(0, 100, 250, 0.2)',
#     line=dict(color='rgba(255,255,255,0)'),
#     hoverinfo="skip",
#     name='95% Confidence Interval'
# ))

# 2. Mean Prediction
fig.add_trace(go.Scatter(
    x=x_plot_raw,
    y=mean_line,
    mode='lines',
    line=dict(color='blue', width=3),
    name='Mean Prediction'
))

# 3. Training Data
fig.add_trace(go.Scatter(
    x=train_x_raw.squeeze().numpy(),
    y=train_y_raw.squeeze().numpy(),
    mode='markers',
    marker=dict(color='black', size=3, opacity=0.5),
    name='Training Data'
))

# 4. Test Data
fig.add_trace(go.Scatter(
    x=test_x_raw.squeeze().numpy(),
    y=test_y_raw_vis.squeeze().numpy(),
    mode='markers',
    marker=dict(color='red', size=3, opacity=0.5),
    name='Test Data'
))

fig.update_layout(
    title=f'Deep GP Fit (RMSE: {rmse_raw.item():.2f})',
    xaxis_title='Input X',
    yaxis_title='Output y',
    template='plotly_white',
    height=600,
    width=1000
)

fig.show()