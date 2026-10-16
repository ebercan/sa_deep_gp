import torch
import numpy as np
import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import gpytorch

import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

def ackley_1d(x):
    a, b, c = 20, 0.2, 2 * np.pi
    term1 = -a * torch.exp(-b * torch.abs(x))
    term2 = -torch.exp(torch.cos(c * x))
    return term1 + term2 + a + np.e

def matern52_kernel(x, z, lengthscale, amplitude):
    if x.dim() == 1: x = x.unsqueeze(-1)
    if z.dim() == 1: z = z.unsqueeze(-1)

    dist = torch.cdist(x, z, p=2)
    sqrt5 = np.sqrt(5.0)
    d_scaled = sqrt5 * dist / lengthscale
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k

def deep_gp_model(x, y=None):
    n = x.size(0)

    x_in = x.unsqueeze(-1)

    ls1 = pyro.sample("ls1", dist.LogNormal(-1.0, 1.0))
    amp1 = pyro.sample("amp1", dist.LogNormal(-2.0, 0.5))

    # k1_obj = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # k1_obj = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5/2))
    # k1_obj.base_kernel.lengthscale = ls1
    # k1_obj.outputscale = amp1
    # # cov1 = k1_obj(x).evaluate() + torch.eye(n) * 1e-4

    cov1 = matern52_kernel(x_in, x_in, ls1, amp1) + torch.eye(n) * 1e-4

    layer1_slope = pyro.sample("slope1", dist.Normal(1.0, 0.5))
    layer1_bias = pyro.sample("bias1", dist.Normal(0.0, 0.5))

    # h = pyro.sample("h", dist.MultivariateNormal(torch.zeros(n), cov1))
    # h_mean = layer1_slope * x + layer1_bias
    h_mean = layer1_slope * x + layer1_bias

    h = pyro.sample("h", dist.MultivariateNormal(h_mean, cov1))

    ls2 = pyro.sample("ls2", dist.LogNormal(-2.0, 0.5))  # Tighter lengthscale
    amp2 = pyro.sample("amp2", dist.LogNormal(0, 1.0))
    noise = pyro.sample("noise", dist.HalfNormal(0.3))

    # k2_obj = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # k2_obj = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5/2))
    #
    # k2_obj.base_kernel.lengthscale = ls2
    # k2_obj.outputscale = amp2
    # cov2 = k2_obj(h).evaluate() + torch.eye(n) * (noise + 1e-4)

    h_in = h.unsqueeze(-1)
    cov2 = matern52_kernel(h_in, h_in, ls2, amp2) + torch.eye(n) * (noise + 1e-4)

    pyro.sample("obs", dist.MultivariateNormal(torch.zeros(n), cov2), obs=y)


def predict(samples, x_train, x_test, y_train, latent_collapse=False):
    n_samples = samples['h'].size(0)
    all_preds = []

    for i in range(n_samples):
        h_i = samples['h'][i]

        # k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        k1.base_kernel.lengthscale, k1.outputscale = samples['ls1'][i], samples['amp1'][i]
        K_tr = k1(x_test, x_train).evaluate()
        K_rr = k1(x_train, x_train).evaluate() + torch.eye(len(x_train)) * 1e-4
        K_tt = k1(x_test, x_test).evaluate() + torch.eye(len(x_test)) * 1e-4

        L_rr = torch.linalg.cholesky(K_rr)
        weights_1 = torch.cholesky_solve((h_i - x_train).unsqueeze(1), L_rr).squeeze()

        h_test_mean = x_test + K_tr @ weights_1

        if latent_collapse:
            h_test_sample = h_test_mean
        else:
            v = torch.linalg.solve(L_rr, K_tr.t())
            h_test_cov = K_tt - v.t() @ v

            h_test_sample = dist.MultivariateNormal(h_test_mean, h_test_cov).sample()

        # k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        k2.base_kernel.lengthscale, k2.outputscale = samples['ls2'][i], samples['amp2'][i]
        noise = samples['noise'][i]

        K2_tr = k2(h_test_sample, h_i).evaluate()
        K2_rr = k2(h_i, h_i).evaluate() + torch.eye(len(x_train)) * (noise + 1e-4)

        L2_rr = torch.linalg.cholesky(K2_rr)
        weights_2 = torch.cholesky_solve(y_train.unsqueeze(1), L2_rr).squeeze()

        y_test_mean = K2_tr @ weights_2

        # y_test_mean = K2_tr @ torch.linalg.solve(K2_rr, y_train)

        all_preds.append(y_test_mean.detach().numpy())

    return np.array(all_preds)

def predict_vectorized_whitened(samples, x_train, x_test, y_train):
    """
    Fully vectorized prediction with Whitening Reconstruction.
    """
    # CHANGE: Get S from a param that definitely exists
    S = samples['ls1'].shape[0]
    N = x_train.shape[0]
    M = x_test.shape[0]

    device = samples['ls1'].device
    x_train = x_train.to(device).unsqueeze(0).unsqueeze(-1).expand(S, N, 1)  # (S, N, 1)
    x_test = x_test.to(device).unsqueeze(0).unsqueeze(-1).expand(S, M, 1)  # (S, M, 1)
    y_train = y_train.to(device).unsqueeze(0).expand(S, N)  # (S, N)

    # ==========================================
    # 1. RECONSTRUCT LATENT h (Vectorized)
    # ==========================================

    # Batch Kernels
    k1 = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([S])),
        batch_shape=torch.Size([S])
    )
    k1.base_kernel.lengthscale = samples['ls1'].view(S, 1, 1)
    k1.outputscale = samples['amp1'].view(S)

    # Compute K_rr and L_rr
    K_rr = k1(x_train, x_train).evaluate() + torch.eye(N, device=device) * 1e-4
    L_rr = torch.linalg.cholesky(K_rr)  # (S, N, N)

    # Compute Prior Mean (Slope/Bias)
    slope1 = samples['slope1'].view(S, 1, 1)
    bias1 = samples['bias1'].view(S, 1, 1)
    h_mean_train = slope1 * x_train + bias1  # (S, N, 1)

    # Reconstruct h_train from noise
    # h = mean + L @ noise
    h_noise = samples['h_noise'].unsqueeze(-1)  # (S, N, 1)
    h_train_samples = h_mean_train + L_rr @ h_noise  # (S, N, 1)

    # ==========================================
    # 2. PREDICT LAYER 1
    # ==========================================
    K_tr = k1(x_test, x_train).evaluate()
    K_tt = k1(x_test, x_test).evaluate() + torch.eye(M, device=device) * 1e-4

    # Residuals = (h_reconstructed - h_prior_mean)
    # Note: Mathematically, (h - mu) is exactly (L @ noise), so we could
    # just use L @ noise, but subtracting is safer for code consistency.
    residuals = h_train_samples - h_mean_train
    weights_1 = torch.cholesky_solve(residuals, L_rr)

    # Test Prior Mean
    h_mean_test = slope1 * x_test + bias1

    # Posterior Mean = Prior + K_tr @ weights
    h_test_mean = h_mean_test + K_tr @ weights_1

    # Posterior Variance
    v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
    h_test_cov = K_tt - v.transpose(-1, -2) @ v

    # Sample h_test
    h_test_sample = dist.MultivariateNormal(h_test_mean.squeeze(-1), h_test_cov).sample()
    h_test_input = h_test_sample.unsqueeze(-1)  # (S, M, 1)

    # ==========================================
    # 3. PREDICT LAYER 2
    # ==========================================
    k2 = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([S])),
        batch_shape=torch.Size([S])
    )
    k2.base_kernel.lengthscale = samples['ls2'].view(S, 1, 1)
    k2.outputscale = samples['amp2'].view(S)
    noise = samples['noise'].view(S, 1, 1)

    K2_tr = k2(h_test_input, h_train_samples).evaluate()
    K2_rr = k2(h_train_samples, h_train_samples).evaluate() + \
            torch.eye(N, device=device) * (noise + 1e-4)

    L2_rr = torch.linalg.cholesky(K2_rr)
    weights_2 = torch.cholesky_solve(y_train.unsqueeze(-1), L2_rr)

    y_test_mean = K2_tr @ weights_2

    return y_test_mean.squeeze(-1).detach().cpu().numpy()

def predict_vectorized_og(samples, x_train, x_test, y_train):
    """
    Fully vectorized prediction.
    Processes all MCMC samples in parallel using PyTorch batching.
    """
    S = samples['h'].shape[0]
    N = x_train.shape[0]
    M = x_test.shape[0]

    device = samples['h'].device
    x_train = x_train.to(device).unsqueeze(0).unsqueeze(-1).expand(S, N, 1)  # (S, N, 1)
    x_test = x_test.to(device).unsqueeze(0).unsqueeze(-1).expand(S, M, 1)  # (S, M, 1)
    y_train = y_train.to(device).unsqueeze(0).expand(S, N)  # (S, N)

    h_train_samples = samples['h'].unsqueeze(-1)  # (S, N, 1)

    k1 = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([S])),
        batch_shape=torch.Size([S])
    )
    k1.base_kernel.lengthscale = samples['ls1'].view(S, 1, 1)
    k1.outputscale = samples['amp1'].view(S)

    K_tr = k1(x_test, x_train).evaluate()
    K_rr = k1(x_train, x_train).evaluate() + torch.eye(N, device=device) * 1e-4
    K_tt = k1(x_test, x_test).evaluate() + torch.eye(M, device=device) * 1e-4

    L_rr = torch.linalg.cholesky(K_rr)  # (S, N, N)

    residuals = h_train_samples - x_train

    weights_1 = torch.cholesky_solve(residuals, L_rr)

    h_test_mean = x_test + K_tr @ weights_1  # (S, M, 1)

    v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))  # (S, N, M)
    h_test_cov = K_tt - v.transpose(-1, -2) @ v  # (S, M, M)

    h_test_sample = dist.MultivariateNormal(h_test_mean.squeeze(-1), h_test_cov).sample()

    h_test_input = h_test_sample.unsqueeze(-1)

    k2 = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([S])),
        batch_shape=torch.Size([S])
    )
    k2.base_kernel.lengthscale = samples['ls2'].view(S, 1, 1)
    k2.outputscale = samples['amp2'].view(S)

    noise = samples['noise'].view(S, 1, 1)

    K2_tr = k2(h_test_input, h_train_samples).evaluate()
    K2_rr = k2(h_train_samples, h_train_samples).evaluate() + \
            torch.eye(N, device=device) * (noise + 1e-4)

    L2_rr = torch.linalg.cholesky(K2_rr)

    weights_2 = torch.cholesky_solve(y_train.unsqueeze(-1), L2_rr)

    y_test_mean = K2_tr @ weights_2  # (S, M, 1)

    return y_test_mean.squeeze(-1).detach().cpu().numpy()

def predict_vectorized(samples, x_train, x_test, y_train):
    S = samples['ls1'].shape[0]
    N = x_train.shape[0]
    M = x_test.shape[0]
    device = samples['ls1'].device

    x_train = x_train.to(device).view(1, N, 1).expand(S, N, 1)
    x_test = x_test.to(device).view(1, M, 1).expand(S, M, 1)
    y_train = y_train.to(device).view(1, N).expand(S, N)

    # Reshape params
    ls1 = samples['ls1'].view(S, 1, 1)
    amp1 = samples['amp1'].view(S, 1, 1)
    ls2 = samples['ls2'].view(S, 1, 1)
    amp2 = samples['amp2'].view(S, 1, 1)
    noise = samples['noise'].view(S, 1, 1)
    slope1 = samples['slope1'].view(S, 1, 1)
    bias1 = samples['bias1'].view(S, 1, 1)

    # 1. LAYER 1 PREDICTION (Standard GP)
    # We use samples['h'] directly again (Centered approach)
    h_train_samples = samples['h'].unsqueeze(-1)  # (S, N, 1)

    K_rr = matern52_kernel(x_train, x_train, ls1, amp1) + torch.eye(N, device=device) * 1e-4
    L_rr = torch.linalg.cholesky(K_rr)

    # Residuals = h_sample - h_prior_mean
    h_prior_mean_train = slope1 * x_train + bias1
    residuals = h_train_samples - h_prior_mean_train

    weights_1 = torch.cholesky_solve(residuals, L_rr)

    K_tr = matern52_kernel(x_test, x_train, ls1, amp1)
    K_tt = matern52_kernel(x_test, x_test, ls1, amp1) + torch.eye(M, device=device) * 1e-4

    h_prior_mean_test = slope1 * x_test + bias1
    h_test_mean = h_prior_mean_test + K_tr @ weights_1

    v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
    h_test_cov = K_tt - v.transpose(-1, -2) @ v

    # Sample h_test
    h_test_sample = dist.MultivariateNormal(h_test_mean.squeeze(-1), h_test_cov).sample()
    h_test_input = h_test_sample.unsqueeze(-1)

    # 2. LAYER 2 PREDICTION
    K2_tr = matern52_kernel(h_test_input, h_train_samples, ls2, amp2)
    K2_rr = matern52_kernel(h_train_samples, h_train_samples, ls2, amp2) + \
            torch.eye(N, device=device) * (noise + 1e-4)

    L2_rr = torch.linalg.cholesky(K2_rr)
    weights_2 = torch.cholesky_solve(y_train.unsqueeze(-1), L2_rr)

    y_test_mean = K2_tr @ weights_2

    return y_test_mean.squeeze(-1).detach().cpu().numpy()

def train_and_predict_deep_gp_model(train_x, train_y, test_x):
    nuts_kernel = NUTS(deep_gp_model, jit_compile=True)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
    mcmc.run(train_x, train_y)
    samples = mcmc.get_samples()

    preds = predict_vectorized(samples, train_x, test_x, train_y)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return preds, preds.mean(axis=0), preds.std(axis=0)

def train_and_predict_standard_gp_model(train_x, train_y, test_x):
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_x_bo = train_x.unsqueeze(-1)
    train_y_bo = train_y.unsqueeze(-1)
    test_x_bo = test_x.unsqueeze(-1)

    model = SingleTaskGP(train_x_bo, train_y_bo)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll(mll)

    model.eval()
    preds = model(test_x_bo)

    mean_pred = preds.mean.detach().squeeze().numpy()
    std_pred = preds.variance.sqrt().detach().squeeze().numpy()

    return preds, mean_pred, std_pred


def main():
    torch.manual_seed(42)

    # 1. Create Raw Data (Keep these for plotting!)
    train_x_raw = torch.linspace(-5, 5, 50, dtype=torch.float64)
    train_y_raw = ackley_1d(train_x_raw) + torch.randn(50) * 0.1
    test_x_raw = torch.linspace(-6, 6, 200, dtype=torch.float64)

    x_min = train_x_raw.min()
    x_max = train_x_raw.max()
    x_range = x_max - x_min

    train_x = (train_x_raw - x_min) / x_range
    test_x = (test_x_raw - x_min) / x_range  # Use Train min/range to scale Test!

    y_mean = train_y_raw.mean()
    y_std = train_y_raw.std()
    train_y = (train_y_raw - y_mean) / y_std

    # ==========================================
    # MODEL 1: DEEP GP
    # ==========================================
    # Note: Increase samples/warmup for real results (e.g., 500/500)
    preds_norm, mean_pred_norm, std_pred_norm = train_and_predict_deep_gp_model(train_x, train_y, test_x)

    # De-normalize predictions
    # preds_norm is a Numpy Array, so this math works fine
    preds = preds_norm * y_std.numpy() + y_mean.numpy()
    mean_pred = mean_pred_norm * y_std.numpy() + y_mean.numpy()
    std_pred = std_pred_norm * y_std.numpy()

    fig = go.Figure()

    # Plot 2-Sigma Uncertainty
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_x_raw.numpy(), test_x_raw.numpy()[::-1]]),  # Use RAW X
        y=np.concatenate([mean_pred + 2 * std_pred, (mean_pred - 2 * std_pred)[::-1]]),
        fill='toself', fillcolor='rgba(0,100,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="2σ Uncertainty (Deep GP)"
    ))

    # Plot True Function (computed on RAW X)
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=ackley_1d(test_x_raw), mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Ackley'
    ))

    # Plot Deep GP Mean
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=mean_pred, mode='lines',
        line=dict(color='blue', width=3),
        name='Deep GP Mean'
    ))

    # Plot Training Data
    fig.add_trace(go.Scatter(
        x=train_x_raw, y=train_y_raw, mode='markers',
        marker=dict(color='red', size=8, line=dict(color='black', width=1)),
        name='Train Data'
    ))

    # ==========================================
    # MODEL 2: STANDARD GP
    # ==========================================
    # preds_dist is a MultivariateNormal object (we ignore it for plotting)
    preds_dist, mean_pred_norm, std_pred_norm = train_and_predict_standard_gp_model(train_x, train_y, test_x)

    # De-normalize predictions
    # FIX: We do NOT scale 'preds_dist'. We only scale the mean/std arrays.
    mean_pred_bo = mean_pred_norm * y_std.numpy() + y_mean.numpy()
    std_pred_bo = std_pred_norm * y_std.numpy()

    # Plot Standard GP Uncertainty
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_x_raw.numpy(), test_x_raw.numpy()[::-1]]),
        y=np.concatenate([mean_pred_bo + 2 * std_pred_bo, (mean_pred_bo - 2 * std_pred_bo)[::-1]]),
        fill='toself', fillcolor='rgba(0,255,100,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="2σ Uncertainty (Standard GP)"
    ))

    # Plot Standard GP Mean
    fig.add_trace(go.Scatter(
        x=test_x_raw, y=mean_pred_bo, mode='lines',
        line=dict(color='green', width=3),
        name='Standard GP Mean'
    ))

    fig.update_layout(
        title="Deep GP vs Standard GP (Ackley 1D)",
        xaxis_title="Input x", yaxis_title="Output f(x)",
        template="plotly_white", hovermode="x unified",
        legend=dict(groupclick="toggleitem")
    )

    fig.show()


if __name__ == "__main__":
    main()