import torch
import gpytorch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import copy

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


# --- 1. Benchmark ---
class Benchmark1D:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
        self.a, self.b, self.c = 20, 0.2, 2 * np.pi

    def func(self, x):
        z = (x - 0.5) * 10
        term1 = -self.a * torch.exp(-self.b * torch.abs(z))
        term2 = -torch.exp(torch.cos(self.c * z))
        return term1 + term2 + self.a + np.exp(1)

    def observe(self, x):
        return self.func(x) + torch.randn_like(x) * self.noise_std


# --- 2. DeepGP Classes ---
class DeepGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', kernel='rbf', is_hidden=True):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        if is_hidden:
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=batch_shape
            )
        else:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=batch_shape
            )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)

        ls_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        if kernel == 'matern2.5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                                                        lengthscale_prior=ls_prior)
        elif kernel == 'matern1.5':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                                                        lengthscale_prior=ls_prior)
        else:
            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims,
                                                     lengthscale_prior=ls_prior)

        if is_hidden:
            os_prior = gpytorch.priors.GammaPrior(3.0, 3.0)
        else:
            os_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, batch_shape=batch_shape, outputscale_prior=os_prior
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HybridDeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x, num_inducing):
        super().__init__()
        self.layer1 = DeepGPHiddenLayer(
            input_dims=train_x.shape[-1], output_dims=5, mean_type='constant',
            num_inducing=num_inducing, kernel='matern2.5', is_hidden=True
        )
        self.layer2 = DeepGPHiddenLayer(
            input_dims=self.layer1.output_dims, output_dims=None, mean_type='constant',
            num_inducing=num_inducing, kernel='matern1.5', is_hidden=False
        )

    def forward(self, x):
        if x.dim() == 2:
            x_expanded = x.expand(self.layer1.output_dims, *x.shape)
        else:
            x_expanded = x

        d1 = self.layer1(x_expanded)
        s1 = d1.rsample()

        s1_transposed = s1.transpose(0, 1)
        if s1_transposed.dim() == 3 and s1_transposed.shape[-1] == 1:
            s1_transposed = s1_transposed.squeeze(-1)

        d2 = self.layer2(s1_transposed)
        return d2


# --- 3. Training Routine (Short for Debug) ---
def train_final(train_x, train_y):
    M = train_x.size(0)
    model = HybridDeepGP(train_x, num_inducing=M).to(device, dtype)

    print("--> Initializing Latent Space...")
    latent_init = train_x.repeat(1, 5) + torch.randn_like(train_x.repeat(1, 5)) * 0.1
    latent_init_batched = latent_init.t().unsqueeze(-1)
    model.layer1.variational_strategy.inducing_points.data = latent_init_batched.clone()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)
    ).to(device, dtype)
    likelihood.noise = 0.1

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.DeepApproximateMLL(gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel()))

    # 1 ITERATION ONLY (As requested for debugging)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = -mll(model(train_x), train_y).sum()
    loss.backward()
    optimizer.step()
    print(f"Single Step Loss: {loss.item()}")

    return model, likelihood


# --- 4. Execution ---
torch.manual_seed(42)
benchmark = Benchmark1D(noise_std=0.1)
train_x = torch.linspace(0, 1, 300, device=device, dtype=dtype).unsqueeze(-1)
train_y_raw = benchmark.observe(train_x).squeeze()
train_y = (train_y_raw - train_y_raw.mean()) / train_y_raw.std()

model, likelihood = train_final(train_x, train_y)

# --- Visualization (DIAGNOSTIC MODE) ---
model.eval()
likelihood.eval()
print("\n--> Generating Diagnostic Plot...")

with torch.no_grad():
    test_x = torch.linspace(0, 1, 500, device=device, dtype=dtype).unsqueeze(-1)

    means = []
    for _ in range(5):  # Fewer samples for debug
        preds = likelihood(model(test_x))
        means.append(preds.mean)
    mean_preds = torch.stack(means).mean(0)

    # --- SANITIZE: Handle Infinite/NaN values explicitly ---
    # Replace Inf with a large number (100) so it doesn't break the plot
    mean_preds = torch.nan_to_num(mean_preds, nan=0.0, posinf=100.0, neginf=-100.0)

    # Scale back
    mean = mean_preds * train_y_raw.std() + train_y_raw.mean()

    # Safe Variance
    safe_variance = torch.nan_to_num(preds.variance, nan=0.0, posinf=100.0, neginf=0.0).clamp(min=0.0)
    std = safe_variance.sqrt() * train_y_raw.std()

    lower, upper = mean - 2 * std, mean + 2 * std

    # --- PRINT STATS (Check Terminal!) ---
    print(f"MEAN Stats: Min={mean.min():.4f}, Max={mean.max():.4f}")
    print(f"STD  Stats: Min={std.min():.4f}, Max={std.max():.4f}")

    # Convert to Numpy
    test_x_np = test_x.squeeze().cpu().numpy()
    mean_np = mean.cpu().numpy()
    upper_np, lower_np = upper.cpu().numpy(), lower.cpu().numpy()
    true_f_np = benchmark.func(test_x).squeeze().cpu().numpy()
    train_x_np = train_x.squeeze().cpu().numpy()
    train_y_np = train_y_raw.cpu().numpy()

    fig = make_subplots(rows=1, cols=1, subplot_titles=("DeepGP Diagnostic View",))

    # Plot Confidence
    fig.add_trace(
        go.Scatter(x=np.concatenate([test_x_np, test_x_np[::-1]]), y=np.concatenate([upper_np, lower_np[::-1]]),
                   fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(width=0), name='Confidence'))
    # Plot Mean
    fig.add_trace(
        go.Scatter(x=test_x_np, y=mean_np, mode='lines', line=dict(color='blue', width=2), name='DeepGP Mean'))
    # Plot Truth
    fig.add_trace(
        go.Scatter(x=test_x_np, y=true_f_np, mode='lines', line=dict(color='black', dash='dash'), name='Truth'))
    # Plot Data
    fig.add_trace(go.Scatter(x=train_x_np, y=train_y_np, mode='markers', marker=dict(color='red', size=3, opacity=0.4),
                             name='Data'))

    # FORCE AXIS RANGE (This guarantees the grid appears)
    fig.update_layout(
        title='DeepGP Diagnostic (Forced Grid)',
        template='plotly_white',
        xaxis=dict(range=[0, 1]),  # Force X range
        yaxis=dict(range=[-5, 20])  # Force Y range (approx Ackley)
    )
    fig.show()