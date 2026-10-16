import math
import torch
import gpytorch
import tqdm
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.optim import NGD
import plotly.graph_objects as go
from math import floor

torch.manual_seed(0)

def ackley(x, a=20.0, b=0.2, c=2*math.pi):
    d = x.shape[-1]
    sq_term = torch.mean(x**2, dim=-1)
    cos_term = torch.mean(torch.cos(c * x), dim=-1)
    return -a * torch.exp(-b * torch.sqrt(sq_term)) - torch.exp(cos_term) + a + math.e

X = torch.linspace(-5.0, 5.0, steps=30).unsqueeze(-1)  # [30, 1]
y = ackley(X)

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
test_x  = X[train_n:, :].contiguous()
test_y  = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

# -----------------------------
# 2) Model
# -----------------------------
class MyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=64,
                 mean_type='constant', kernel_type='rbf',
                 init_lengthscale=None, init_period=None, active_dims=None):
        if output_dims is None:
            inducing_points = torch.linspace(-5.0, 5.0, num_inducing).unsqueeze(-1)
            if input_dims > 1:
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

        self.mean_module = ConstantMean(batch_shape=batch_shape) if mean_type == 'constant' else LinearMean(input_dims)

        # ----- Kernel choices -----
        if kernel_type == 'locally_periodic':
            rbf = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims, active_dims=active_dims)
            per = PeriodicKernel(batch_shape=batch_shape, ard_num_dims=input_dims, active_dims=active_dims)
            if init_lengthscale is not None:
                rbf.initialize(lengthscale=float(init_lengthscale))
                per.initialize(lengthscale=float(init_lengthscale))
            if init_period is not None:
                per.initialize(period_length=float(init_period))
            base_kernel = rbf * per  # locally periodic
        elif kernel_type == 'periodic':
            per = PeriodicKernel(batch_shape=batch_shape, ard_num_dims=input_dims, active_dims=active_dims)
            if init_lengthscale is not None:
                per.initialize(lengthscale=float(init_lengthscale))
            if init_period is not None:
                per.initialize(period_length=float(init_period))
            base_kernel = per
        else:  # 'rbf' / default
            rbf = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims, active_dims=active_dims)
            if init_lengthscale is not None:
                rbf.initialize(lengthscale=float(init_lengthscale))
            base_kernel = rbf

        self.covar_module = ScaleKernel(base_kernel, batch_shape=batch_shape)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def __call__(self, x, *other_inputs, **kwargs):
        # Supports skip connections if you pass extra inputs: layer(out, inputs)
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            processed = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))

# More hidden dims + more inducing points for fidelity
num_hidden_dims = 10

class AckleyDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        # Layer 1: LOCALLY PERIODIC on the raw input (period ≈ 1 for cos(2πx))
        hidden_layer = MyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],      # 1D input
            output_dims=num_hidden_dims,
            num_inducing=128,
            kernel_type='locally_periodic',
            init_lengthscale=0.2,              # short to capture small-scale wiggles
            init_period=1.0                    # Ackley period
        )
        # Layer 2: RBF on features (kept simple/robust)
        last_layer = MyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,  # NOTE: if you want skip-conn -> make this +1 and call with (hidden, inputs)
            output_dims=None,
            num_inducing=64,
            kernel_type='rbf',
            init_lengthscale=0.5
        )
        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        # Encourage low observation noise (was likely too big → underfit)
        self.likelihood.initialize(noise=1e-3)

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        # If you want a skip connection, change last_layer's input_dims to hidden_dim+1 above
        # and replace the next line with: output = self.last_layer(hidden, inputs)
        output = self.last_layer(hidden)
        return output

model = AckleyDeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()

# -----------------------------
# 3) Train (full-batch) with NGD + Adam
# -----------------------------
num_epochs  = 800
num_samples = 64

# Natural gradient for variational params (fast, stable); Adam for kernel/likelihood hypers
variational_ngd = NGD(model.variational_parameters(), num_data=train_x.shape[-2], lr=0.1)
hyper_opt = torch.optim.Adam([{'params': model.hyperparameters()}], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

for _ in tqdm.tqdm(range(num_epochs), desc="Epoch"):
    with gpytorch.settings.num_likelihood_samples(num_samples):
        variational_ngd.zero_grad()
        hyper_opt.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        variational_ngd.step()
        hyper_opt.step()

# -----------------------------
# 4) Eval + Plot
# -----------------------------
@torch.no_grad()
def make_main_figure_gpytorch(
    model, ackley, Xtr, Ytr, Xte, Yte,
    num_points: int = 600, num_likelihood_samples: int = 512, x_bounds=(-5.0, 5.0)
):
    device, dtype = Xtr.device, Xtr.dtype
    x_min, x_max = map(float, x_bounds)
    xs = torch.linspace(x_min, x_max, num_points, dtype=dtype, device=device).unsqueeze(-1)
    with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
        post = model.likelihood(model(xs))
        mean_s = post.mean
        var_s  = post.variance
        mu = mean_s.mean(0)
        total_var = var_s.mean(0) + (mean_s**2).mean(0) - mu**2
        std = total_var.clamp_min(0).sqrt()

    f_true = ackley(xs).cpu().numpy()
    xline  = xs.squeeze(-1).cpu().numpy()
    mu_np  = mu.cpu().numpy()
    lo     = (mu - 1.96 * std).cpu().numpy()
    hi     = (mu + 1.96 * std).cpu().numpy()

    tx, ty = Xtr[:, 0].cpu().numpy(), Ytr.cpu().numpy()
    vx, vy = Xte[:, 0].cpu().numpy(), Yte.cpu().numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xline, y=lo, mode="lines", line=dict(width=0),
                             name="DeepGP 95% CI (low)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xline, y=hi, mode="lines", fill="tonexty", line=dict(width=0),
                             name="DeepGP 95% CI"))
    fig.add_trace(go.Scatter(x=xline, y=mu_np, mode="lines", name="DeepGP mean"))
    fig.add_trace(go.Scatter(x=xline, y=f_true, mode="lines", line=dict(dash="dash"), name="Ackley (truth)"))
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="markers", marker=dict(symbol="circle", size=7), name="Train pts"))
    fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers", marker=dict(symbol="x", size=8), name="Test pts"))
    fig.update_layout(height=500, width=900, title="Deep GP on 1D Ackley (original units)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                      margin=dict(l=50, r=20, t=70, b=50))
    fig.update_xaxes(title_text="x", range=[x_min, x_max])
    fig.update_yaxes(title_text="Objective")
    return fig

model.eval()
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(512):
    pred_dist = model.likelihood(model(test_x))
    mu = pred_dist.mean.mean(0)
    rmse = torch.mean((mu - test_y)**2).sqrt()
    test_ll = model.likelihood.log_marginal(test_y, model(test_x)).mean()
print(f"RMSE: {rmse.item():.4f}, NLL: {-test_ll.item():.4f}")

fig = make_main_figure_gpytorch(model, ackley, train_x, train_y, test_x, test_y,
                                x_bounds=(-5.0, 5.0))
fig.show()
