import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGP
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.distributions import MultivariateNormal


# class GPLayer(DeepGPLayer):
#     def __init__(self, input_dims, output_dims, num_inducing=128,
#                  length_prior=None, outputscale_prior=None, mean_prior=None):
#         if output_dims is None:
#             inducing_points = torch.randn(num_inducing, input_dims)
#             batch_shape = torch.Size([])
#         else:
#             inducing_points = torch.randn(output_dims, num_inducing, input_dims)
#             batch_shape = torch.Size([output_dims])
#
#         variational_distribution = CholeskyVariationalDistribution(
#             num_inducing_points=num_inducing,
#             batch_shape=batch_shape
#         )
#
#         variational_strategy = VariationalStrategy(
#             self,
#             inducing_points,
#             variational_distribution,
#             learn_inducing_locations=True
#         )
#
#         super(GPLayer, self).__init__(variational_strategy, input_dims, output_dims)
#
#         self.mean_module = ConstantMean(batch_shape=batch_shape)
#         self.covar_module = ScaleKernel(
#             RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
#             batch_shape=batch_shape, ard_num_dims=None
#         )
#
#         # self.mean_module = ConstantMean(prior=mean_prior)
#         # base_kernel = RBFKernel(ard_num_dims=input_dim, lengthscale_prior=length_prior)
#         # self.covar_module = ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#
#         return MultivariateNormal(mean_x, covar_x)
#
#     def __call__(self, x, *other_inputs, **kwargs):
#         if len(other_inputs):
#             if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
#                 x = x.rsample()
#
#             processed_inputs = [
#                 inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
#                 for inp in other_inputs
#             ]
#             x = torch.cat([x] + processed_inputs, dim=-1)
#
#         return super().__call__(x, are_samples=bool(len(other_inputs)))
#
#
# class RWGSDeepGP(DeepGP):
#     def __init__(self,
#                  input_dims_x1: int,
#                  input_dims_x2: int,
#                  compressor_dims: int,
#                  pre_heater_dims: int,
#                  reactor_dims: int,
#                  num_inducing=64, #128
#                  priors_compressor=None,
#                  priors_pre_heater=None,
#                  priors_reactor=None,
#                  priors_separator=None):
#
#         super().__init__()
#
#         self.input_dim_x1 = input_dims_x1
#         self.input_dim_x2 = input_dims_x2
#
#         self.compressor_layer = GPLayer(input_dims=input_dims_x1, output_dims=compressor_dims,
#                                         # num_inducing=num_inducing,
#                                         # length_prior=None if not priors_compressor else priors_compressor.get("length"),
#                                         # outputscale_prior=None if not priors_compressor else priors_compressor.get(
#                                         #     "output"),
#                                         # mean_prior=None if not priors_compressor else priors_compressor.get("mean")
#                                         )
#
#         self.pre_heater_layer = GPLayer(input_dims=compressor_dims, output_dims=pre_heater_dims,
#                                         num_inducing=num_inducing,
#                                         # length_prior=None if not priors_pre_heater else priors_pre_heater.get("length"),
#                                         # outputscale_prior=None if not priors_pre_heater else priors_pre_heater.get(
#                                         #     "output"),
#                                         # mean_prior=None if not priors_pre_heater else priors_pre_heater.get("mean")
#                                         )
#
#         self.reactor_layer = GPLayer(input_dims=pre_heater_dims + input_dims_x2, output_dims=reactor_dims,
#                                      num_inducing=num_inducing,
#                                      # length_prior=None if not priors_reactor else priors_reactor.get("length"),
#                                      # outputscale_prior=None if not priors_reactor else priors_reactor.get("output"),
#                                      # mean_prior=None if not priors_reactor else priors_reactor.get("mean")
#                                      )
#
#         self.separator_layer = GPLayer(input_dims=reactor_dims, output_dims=1,
#                                        num_inducing=num_inducing,
#                                        # length_prior=None if not priors_separator else priors_separator.get("length"),
#                                        # outputscale_prior=None if not priors_separator else priors_separator.get(
#                                        #     "output"),
#                                        # mean_prior=None if not priors_separator else priors_separator.get("mean")
#                                        )
#
#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
#     def _split_inputs(self, X):
#         X1 = X[:, :self.input_dim_x1]
#         X2 = X[:, self.input_dim_x1:self.input_dim_x1 + self.input_dim_x2]
#         return X1, X2
#
#     def forward(self, inputs):
#         X1, X2 = self._split_inputs(inputs)
#
#         compressor_dist = self.compressor_layer(X1)
#         pre_heater_dist = self.pre_heater_layer(compressor_dist)
#
#         reactor_dist = self.reactor_layer(pre_heater_dist, X2)
#         separator_dist = self.separator_layer(reactor_dist)
#
#         return separator_dist
#
#     @torch.no_grad()
#     def predict(self, X_test, num_samples=64):
#         self.eval()
#         self.likelihood.eval()
#         # with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples):
#         with gpytorch.settings.num_likelihood_samples(num_samples):
#             preds = self.likelihood(self.forward(X_test))
#             if hasattr(preds, "rsample"):
#                 s = preds.rsample(torch.Size([num_samples]))
#                 mean = s.mean(dim=0)
#                 var = s.var(dim=0, unbiased=False)
#             else:
#                 mean = preds.mean
#                 var = preds.variance
#         return mean, var

class GPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128):
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

        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))


class RWGSDeepGP(DeepGP):
    def __init__(self,
                 input_dims_x1: int,
                 input_dims_x2: int,
                 compressor_dims: int,
                 pre_heater_dims: int,
                 reactor_dims: int,
                 num_inducing=64):

        super().__init__()

        self.input_dim_x1 = input_dims_x1
        self.input_dim_x2 = input_dims_x2

        self.compressor_layer = GPLayer(input_dims=input_dims_x1, output_dims=compressor_dims)
        self.pre_heater_layer = GPLayer(input_dims=compressor_dims, output_dims=pre_heater_dims, num_inducing=num_inducing)
        self.reactor_layer = GPLayer(input_dims=pre_heater_dims+input_dims_x1, output_dims=reactor_dims, num_inducing=num_inducing)
        self.separator_layer = GPLayer(input_dims=reactor_dims, output_dims=1, num_inducing=num_inducing)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def _split_inputs(self, X):
        X1 = X[:, :self.input_dim_x1]
        X2 = X[:, self.input_dim_x1:self.input_dim_x1 + self.input_dim_x2]
        return X1, X2

    def forward(self, inputs):
        X1, X2 = self._split_inputs(inputs)

        compressor_dist = self.compressor_layer(X1)
        pre_heater_dist = self.pre_heater_layer(compressor_dist)

        reactor_dist = self.reactor_layer(pre_heater_dist, X2)
        separator_dist = self.separator_layer(reactor_dist)

        return separator_dist

    @torch.no_grad()
    def predict(self, X_test, num_samples=64):
        self.eval()
        self.likelihood.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            preds = self.likelihood(self.forward(X_test))
            if hasattr(preds, "rsample"):
                s = preds.rsample(torch.Size([num_samples]))
                mean = s.mean(dim=0)
                var = s.var(dim=0, unbiased=False)
            else:
                mean = preds.mean
                var = preds.variance
        return mean, var
