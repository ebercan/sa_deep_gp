import torch
import gpytorch
import sys
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

# DETECT DEVICE
device = torch.device("cpu")
dtype = torch.double

print(f"Running Debug on: {device}")


# --- MOCK BENCHMARK ---
class MockBenchmark:
    def __init__(self):
        self.bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)

    def observe(self, x):
        return torch.sin(x.sum(dim=-1, keepdim=True))


# --- DEEP GP CLASSES ---

class DeepGPHiddenLayer(ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing=32, mean_type='linear'):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims, dtype=dtype)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims])
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.batch_shape = torch.Size([output_dims])
        self.mean_module = LinearMean(input_dims, batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=input_dims, batch_shape=self.batch_shape),
            batch_shape=self.batch_shape
        )

    def forward(self, x):
        # FIX: Expand x to match batch dimensions (Batch, N, D)
        if x.dim() == 2:
            x = x.expand(*self.batch_shape, *x.shape)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TwoLayerDGP(gpytorch.Module):
    def __init__(self, input_dim, num_inducing=32):
        super().__init__()

        hidden_dim = 2

        # Layer 1: Hidden
        self.hidden_layer = DeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            num_inducing=num_inducing
        )

        # Layer 2: Final Output
        # We create a specific GP class for the last layer so ELBO accepts it
        last_layer_inducing = torch.randn(num_inducing, hidden_dim, dtype=dtype)
        last_layer_dist = CholeskyVariationalDistribution(num_inducing)

        class LastLayer(ApproximateGP):
            def __init__(self, ind_points, var_dist):
                strategy = VariationalStrategy(self, ind_points, var_dist, learn_inducing_locations=True)
                super().__init__(strategy)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=hidden_dim))

            def forward(self, x):
                return MultivariateNormal(self.mean_module(x), self.covar_module(x))

        self.last_layer = LastLayer(last_layer_inducing, last_layer_dist)

    def forward(self, x):
        # 1. Hidden Layer
        hidden_dist = self.hidden_layer(x)
        hidden_rep = hidden_dist.rsample()
        # Permute: (HiddenDim, N) -> (N, HiddenDim)
        hidden_rep = hidden_rep.permute(1, 0)

        # 2. Last Layer
        return self.last_layer(hidden_rep)


class DeepGPWrapper(GPyTorchModel):
    num_outputs = 1

    def __init__(self, model, likelihood, input_transform, outcome_transform):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform
        self._num_outputs = 1
        self.to(dtype=dtype, device=device)

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.model(x)

    def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            if self.input_transform is not None:
                X = self.input_transform(X)
            # Hidden
            hidden_dist = self.model.hidden_layer(X)
            hidden_samples = hidden_dist.rsample(torch.Size([32]))
            # Permute: (32, Batch, HiddenDim)
            hidden_samples = hidden_samples.permute(0, 2, 1)
            # Output
            output_dist = self.model.last_layer(hidden_samples)

            mean = output_dist.mean.mean(0).unsqueeze(-1)
            var = output_dist.variance.mean(0).unsqueeze(-1) + output_dist.mean.var(0).unsqueeze(-1)

            if self.outcome_transform is not None:
                means = mean * self.outcome_transform.std + self.outcome_transform.mean
                vars_ = var * (self.outcome_transform.std ** 2)
                return gpytorch.distributions.MultivariateNormal(means.squeeze(-1), torch.diag_embed(vars_.squeeze(-1)))
            return gpytorch.distributions.MultivariateNormal(mean.squeeze(-1), torch.diag_embed(var.squeeze(-1)))


# --- RUN ROUTINE ---

def main():
    print("Initializing Benchmark...")
    benchmark = MockBenchmark()
    n_init = 5
    n_iter = 2

    bounds = benchmark.bounds
    dim = bounds.shape[1]

    train_x = torch.rand(n_init, dim, device=device, dtype=dtype)
    train_obj = benchmark.observe(train_x)

    input_trans = Normalize(d=dim)
    outcome_trans = Standardize(m=1)

    print(f"Starting Debug Loop ({n_iter} iters)...")

    for i in range(n_iter):
        print(f"\n--- Iteration {i + 1} ---")

        num_inducing = min(16, train_x.shape[0])

        dgp_model = TwoLayerDGP(input_dim=dim, num_inducing=num_inducing).to(device=device, dtype=dtype)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)

        dgp_model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': dgp_model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.1)

        # FIX 1: Initialize MLL on the LAST LAYER (which is a GP)
        mll = VariationalELBO(likelihood, dgp_model.last_layer, num_data=train_x.size(0))

        train_x_tf = input_trans(train_x)
        train_obj_tf, _ = outcome_trans(train_obj)

        print("  Training...")
        for _ in range(5):
            optimizer.zero_grad()
            output = dgp_model(train_x_tf)

            # FIX 2: Calculate Loss = - (ELBO_LastLayer + KL_HiddenLayer)
            # mll(output, target) returns (LogLikelihood - KL_LastLayer)
            log_lik = mll(output, train_obj_tf.squeeze(-1))

            loss = -log_lik

            # Add KL divergence from the hidden layer (Auxiliary Loss)
            for name, term in dgp_model.hidden_layer.named_added_loss_terms():
                loss += term.loss()

            loss.backward()
            optimizer.step()

        print("  Optimizing Acquisition...")
        wrapped_model = DeepGPWrapper(dgp_model, likelihood, input_trans, outcome_trans)

        acq = LogExpectedImprovement(model=wrapped_model, best_f=train_obj.max())

        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds, q=1, num_restarts=2, raw_samples=8
        )

        new_x = candidates.detach()
        new_obj = benchmark.observe(new_x)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        print("  ✅ Iteration Complete.")


if __name__ == "__main__":
    main()
