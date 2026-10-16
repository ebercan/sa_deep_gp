import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

import ssl, certifi, urllib.request
import urllib.request
import os
from scipy.io import loadmat
from math import floor
from torch.utils.data import TensorDataset, DataLoader
import math


data = torch.Tensor(loadmat('elevators.mat')['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1

y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

class GPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
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


class RWGSDeepGP(DeepGP):
    def __init__(self, train_x_shape, X1_dims, num_hidden_dims):
        compressor_layer = GPLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type='linear',
            num_inducing=20,
        )

        # pre_heater_layer = GPLayer(
        #     input_dims=compressor_layer.output_dims,
        #     output_dims=num_hidden_dims,
        #     mean_type='linear',
        #     num_inducing=128,
        # )
        #
        # reactor_layer = GPLayer(
        #     input_dims=pre_heater_layer.output_dims,
        #     output_dims=num_hidden_dims,
        #     mean_type='linear',
        #     num_inducing=128,
        # )
        #
        # separator_layer = GPLayer(
        #     input_dims=reactor_layer.output_dims,
        #     output_dims=num_hidden_dims,
        #     mean_type='linear',
        #     num_inducing=64,
        # )

        output_layer = GPLayer(
            input_dims=compressor_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        #     num_inducing=64,
            num_inducing=20,
        )

        super().__init__()

        self.X1_dims = X1_dims

        self.compressor_layer = compressor_layer
        # self.pre_heater_layer = pre_heater_layer
        # self.reactor_layer = reactor_layer
        # self.separator_layer = separator_layer
        self.output_layer = output_layer

        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        compressor_dist = self.compressor_layer(inputs)
        # pre_heater_dist = self.pre_heater_layer(compressor_dist)
        # reactor_dist = self.reactor_layer(pre_heater_dist)
        # separator_dist = self.separator_layer(reactor_dist)
        output = self.output_layer(compressor_dist)

        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


num_hidden_dims = 10

X2_dims = 2
X1_dims = train_x.shape[-1] - X2_dims

model = RWGSDeepGP(train_x.shape, X1_dims, num_hidden_dims)

num_epochs = 10
num_samples = 10


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            # print(output)
            # print(y_batch.shape)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())

# import torch
# import gpytorch
# from torch.cuda.amp import autocast, GradScaler
#
# # -------------------------------
# # Configuration
# # -------------------------------
# num_epochs = 60
# train_samples_warm = 1
# train_samples_main = 2
# patience = 7  # early stopping
# learning_rate = 1e-4
#
# device = train_x.device  # assumes train_x and train_y are already on correct device
# train_num_data = train_x.shape[0]
#
# # -------------------------------
# # ELBO + Optimizer
# # -------------------------------
# elbo = gpytorch.mlls.VariationalELBO(
#     model.likelihood,
#     model,
#     num_data=train_num_data
# )
# mll = gpytorch.mlls.DeepApproximateMLL(elbo)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=4, verbose=False
# )
#
# scaler = GradScaler(enabled=torch.cuda.is_available())
#
# # -------------------------------
# # Training Loop
# # -------------------------------
# best_val = float('inf')
# bad_epochs = 0
#
# epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
# for epoch in epochs_iter:
#     model.train()
#     optimizer.zero_grad(set_to_none=True)
#
#     # KL warmup — slower for small data
#     elbo.beta = 0.05 + 0.95 * min(1.0, epoch / int(0.8 * num_epochs))
#
#     # fewer MC samples for stability
#     num_samples = train_samples_warm if epoch < int(0.7 * num_epochs) else train_samples_main
#
#     # forward + loss
#     with gpytorch.settings.num_likelihood_samples(num_samples), autocast(enabled=torch.cuda.is_available()):
#         out = model(train_x)
#         loss = -mll(out, train_y)
#
#     scaler.scale(loss).backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#     scaler.step(optimizer)
#     scaler.update()
#
#     train_loss = loss.item()
#     scheduler.step(train_loss)
#
#     # Early stopping on train loss (or val loss if available)
#     if train_loss + 1e-6 < best_val:
#         best_val = train_loss
#         bad_epochs = 0
#     else:
#         bad_epochs += 1
#         if bad_epochs >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break
#
#     epochs_iter.set_postfix({"loss": train_loss, "best": best_val})
#
# # -------------------------------
# # Evaluation
# # -------------------------------
# model.eval()
# with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
#     predictive_means, predictive_variances, test_lls = model.predict(test_loader)
#
# rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
# nll = -test_lls.mean().item()
#
# print(f"Final RMSE: {rmse.item():.4f}, NLL: {nll:.4f}")


# num_epochs = 600
# train_num_data = train_x.shape[-2]
# train_num_samples_warm = 16
# train_num_samples_main = 32
#
# # Two optimizers: NGD for variational params, Adam for hypers
# variational_optimizer = gpytorch.optim.NGD(
#     model.variational_parameters(),
#     num_data=train_num_data,
#     lr=0.1
# )
# hyper_optimizer = torch.optim.Adam(
#     [{'params': model.hyperparameters()}],
#     lr=3e-3
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(hyper_optimizer, mode='min', factor=0.5, patience=10)
#
# mll = gpytorch.mlls.DeepApproximateMLL(
#     gpytorch.mlls.VariationalELBO(model.likelihood, model, train_num_data)
# )
#
# epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
# for epoch in epochs_iter:
#     model.train()
#     # KL warmup: ramp β from 0.1 to 1.0 over first 20% epochs
#     warmup_frac = min(1.0, epoch / int(0.2 * num_epochs))
#     beta = 0.1 + 0.9 * min(1.0, epoch / int(0.2 * num_epochs))
#     if hasattr(mll, "mll"):
#         mll.mll.beta = beta
#     elif hasattr(mll, "base_mll"):
#         mll.base_mll.beta = beta
#
#     num_samples = train_num_samples_warm if epoch < int(0.15 * num_epochs) else train_num_samples_main
#
#     epoch_loss = 0.0
#     for x_batch, y_batch in train_loader:
#         variational_optimizer.zero_grad()
#         hyper_optimizer.zero_grad()
#
#         with gpytorch.settings.num_likelihood_samples(num_samples):
#             out = model(x_batch)
#             loss = -mll(out, y_batch)
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.hyperparameters(), max_norm=5.0)
#         variational_optimizer.step()
#         hyper_optimizer.step()
#
#         epoch_loss += loss.item() * y_batch.shape[0]
#
#     # Optional: scheduler on validation loss if you have a val_loader
#     scheduler.step(epoch_loss / train_num_data)
#
#
#
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024)

model.eval()
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(256):
    predictive_means, predictive_variances, test_lls = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
