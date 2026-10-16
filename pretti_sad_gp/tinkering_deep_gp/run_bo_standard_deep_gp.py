import torch
import gpytorch
import tqdm
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from scipy.io import loadmat

from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf_discrete

torch.set_default_dtype(torch.float64)
device = "cpu"


class GPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant',
                 inducing_source=None):
        if inducing_source is not None:
            num_inducing = min(num_inducing, inducing_source.shape[0])

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
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

        super(GPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# class GPLayer(DeepGPLayer):
#     def __init__(
#         self,
#         input_dims,
#         output_dims,
#         num_inducing=128,
#         mean_type="constant",
#         inducing_source=None,
#     ):
#         if inducing_source is not None:
#             num_inducing = min(num_inducing, inducing_source.shape[0])
#         else:
#             num_inducing = max(5, num_inducing)
#
#         def pick_inducing_from_data(X, M, batch_shape=None):
#             M = min(M, X.shape[0])
#             idx = torch.randperm(X.shape[0])[:M]
#             Z = X[idx]
#             if batch_shape is None or len(batch_shape) == 0:
#                 return Z.clone()
#             return Z.unsqueeze(0).expand(*batch_shape, M, X.shape[-1]).contiguous()
#
#         if output_dims is None:
#             inducing_points = (
#                 pick_inducing_from_data(inducing_source, num_inducing)
#                 if inducing_source is not None
#                 else torch.randn(num_inducing, input_dims)
#             )
#             batch_shape = torch.Size([])
#         else:
#             inducing_points = (
#                 pick_inducing_from_data(inducing_source, num_inducing, torch.Size([output_dims]))
#                 if inducing_source is not None
#                 else torch.randn(output_dims, num_inducing, input_dims)
#             )
#             batch_shape = torch.Size([output_dims])
#
#         variational_distribution = CholeskyVariationalDistribution(
#             num_inducing_points=num_inducing,
#             batch_shape=batch_shape,
#         )
#
#         variational_strategy = VariationalStrategy(
#             self,
#             inducing_points,
#             variational_distribution,
#             learn_inducing_locations=True,
#         )
#
#         super(GPLayer, self).__init__(variational_strategy, input_dims, output_dims)
#
#         self.mean_module = (
#             ConstantMean(batch_shape=batch_shape)
#             if mean_type == "constant"
#             else LinearMean(input_dims)
#         )
#         self.covar_module = ScaleKernel(
#             RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
#             batch_shape=batch_shape,
#         )
#
#     def forward(self, x):
#         return MultivariateNormal(self.mean_module(x), self.covar_module(x))
#
#     def __call__(self, x, *other_inputs, **kwargs):
#         if len(other_inputs):
#             if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
#                 x = x.rsample()
#             processed_inputs = [
#                 inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
#                 for inp in other_inputs
#             ]
#             x = torch.cat([x] + processed_inputs, dim=-1)
#         return super().__call__(x, are_samples=bool(len(other_inputs)))


# =========================================================
# RWGSDeepGP (your full architecture)
# =========================================================
class RWGSDeepGP(DeepGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x_shape, X1_dims, num_hidden_dims, inducing_source=None):
        super().__init__()
        M = min(20, max(5, inducing_source.shape[0] // 2))

        self.compressor_layer = GPLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=M,
            # inducing_source=inducing_source,
        )

        self.pre_heater_layer = GPLayer(
            input_dims=self.compressor_layer.output_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=M,
            # inducing_source=inducing_source,
        )

        self.reactor_layer = GPLayer(
            input_dims=self.pre_heater_layer.output_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=M,
            # inducing_source=inducing_source,
        )

        self.separator_layer = GPLayer(
            input_dims=self.reactor_layer.output_dims,
            output_dims=num_hidden_dims,
            mean_type="linear",
            num_inducing=M,
            # inducing_source=inducing_source,
        )

        self.output_layer = GPLayer(
            input_dims=self.separator_layer.output_dims,
            output_dims=None,
            mean_type="constant",
            num_inducing=M,
            # inducing_source=inducing_source,
        )

        self.likelihood = GaussianLikelihood()
        self.X1_dims = X1_dims

    def forward(self, x):
        c = self.compressor_layer(x)
        p = self.pre_heater_layer(c)
        r = self.reactor_layer(p)
        s = self.separator_layer(r)
        return self.output_layer(s)

    def posterior(self, X, observation_noise=False, **kwargs):
        self.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(128), gpytorch.settings.cholesky_jitter(1e-3):
            mvn = (self.likelihood(self(X)).mvn) if observation_noise else self(X).mvn
        return GPyTorchPosterior(mvn)

    def train_model(self, train_X, train_Y, lr=5e-4, epochs=200):
        self.train()
        mll = VariationalELBO(self.likelihood, self, num_data=train_X.shape[0])
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm.trange(epochs, desc="Training DeepGP", leave=False)
        for _ in pbar:
            opt.zero_grad()
            with gpytorch.settings.num_likelihood_samples(16), gpytorch.settings.cholesky_jitter(1e-3):
                out = self(train_X)
                loss = -mll(out, train_Y.squeeze(-1))
            # if torch.isfinite(loss):
                loss.backward()
                opt.step()
            pbar.set_postfix(loss=float(loss.detach()))


# =========================================================
# Load data & preprocess
# =========================================================
data = torch.tensor(loadmat("elevators.mat")["data"], dtype=torch.float64)
X = data[:, :-1]
y = data[:, -1]

# robust min-max scaling
minv = X.min(0).values
X = X - minv
maxv = X.max(0).values
scale = torch.where(maxv > 1e-12, maxv, torch.ones_like(maxv))
X = 2 * (X / scale) - 1
y_mean, y_std = y.mean(), y.std().clamp_min(1e-12)
y = (y - y_mean) / y_std

# =========================================================
# Initialize small training set (BO warm start)
# =========================================================
N0 = 10
perm = torch.randperm(X.shape[0])
train_idx = perm[:N0]
pool_idx = perm[N0:]

train_X, train_Y = X[train_idx], y[train_idx].unsqueeze(-1)
pool_X, pool_Y = X[pool_idx], y[pool_idx]

# =========================================================
# Create model and BO setup
# =========================================================
model = RWGSDeepGP(train_X.shape, X1_dims=train_X.shape[-1] - 2, num_hidden_dims=10, inducing_source=train_X)

iterations = 15
best_y_hist = []

for t in range(iterations):
    # (1) Train / update model
    model.train_model(train_X, train_Y, lr=5e-4, epochs=150)

    # (2) Define acquisition function (LogEI)
    sampler = SobolQMCNormalSampler(num_samples=128)
    acq = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)

    # (3) Pick next candidate from pool
    cand_batch, _ = optimize_acqf_discrete(
        acq_function=acq,
        q=1,
        choices=pool_X,
        max_batch_size=512,
    )

    chosen = cand_batch[0]
    ci = (pool_X == chosen).all(dim=-1).nonzero(as_tuple=True)[0].item()
    y_new = pool_Y[ci]

    # (4) Update datasets
    train_X = torch.cat([train_X, chosen.unsqueeze(0)], dim=0)
    train_Y = torch.cat([train_Y, y_new.unsqueeze(0).unsqueeze(-1)], dim=0)

    pool_X = torch.cat([pool_X[:ci], pool_X[ci + 1 :]], dim=0)
    pool_Y = torch.cat([pool_Y[:ci], pool_Y[ci + 1 :]], dim=0)

    best_y = train_Y.max().item()
    best_y_hist.append(best_y)
    print(f"Iter {t+1:02d}: added y={y_new.item():.4f}, best={best_y:.4f}")

# Unstandardize final best
best_std = max(best_y_hist)
best_unstd = best_std * y_std + y_mean
print(f"\nBest standardized: {best_std:.4f}")
print(f"Best unstandardized (original scale): {best_unstd:.4f}")
