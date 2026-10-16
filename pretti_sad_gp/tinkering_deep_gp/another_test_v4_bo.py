import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import warnings

# BoTorch Imports
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
torch.set_default_dtype(torch.float64)

JITTER = 1e-6


# --- Kernel Helper Functions (Unchanged) ---

def clamped_cdist(x, z):
    # Assumes x, z are at least 2D: (..., N, D)
    x2 = x.square().sum(-1, keepdim=True)
    z2 = z.square().sum(-1, keepdim=True)
    dist_sq = (x2 + z2.transpose(-1, -2) - 2 * x @ z.transpose(-1, -2)).clamp(min=1e-12)
    return dist_sq.sqrt()


def matern52_kernel(x, z, lengthscale, amplitude):
    # Assumes x, z are shape (..., N, D)
    # Scale inputs by lengthscale (ARD)
    x_scaled = x / lengthscale
    z_scaled = z / lengthscale

    dist_mat = clamped_cdist(x_scaled, z_scaled)

    if amplitude.dim() == 1:
        amplitude = amplitude.view(-1, 1, 1)

    sqrt5 = np.sqrt(5.0)
    d_scaled = sqrt5 * dist_mat
    k = amplitude * (1 + d_scaled + d_scaled.square() / 3.0) * torch.exp(-d_scaled)
    return k


# --- Dynamic Deep GP Model ---

def _deep_gp_probabilistic_model(x, y=None, hidden_dims=[2]):
    """
    hidden_dims: List of integers, e.g., [2, 2] for two hidden layers of width 2.
    """
    n = x.size(0)
    current_x = x

    # --- Hidden Layers Loop ---
    for layer_idx, width in enumerate(hidden_dims):
        input_dim = current_x.size(1)

        with pyro.plate(f"layer_{layer_idx}_params", width):
            # ARD Lengthscales per hidden neuron: (Width, Input_Dim)
            ls = pyro.sample(f"L{layer_idx}_ls", dist.LogNormal(0.0, 1.0).expand([input_dim]).to_event(1))
            amp = pyro.sample(f"L{layer_idx}_amp", dist.LogNormal(-1.0, 0.5))

            # Linear Mean weights
            slope = pyro.sample(f"L{layer_idx}_slope", dist.Normal(0.0, 0.5).expand([input_dim]).to_event(1))
            bias = pyro.sample(f"L{layer_idx}_bias", dist.Normal(0.0, 0.5))

        h_list = []
        for dim_idx in range(width):
            # Extract params for this specific neuron (dimension)
            ls_d = ls[dim_idx].unsqueeze(0)  # (1, Input_Dim)
            amp_d = amp[dim_idx]
            slope_d = slope[dim_idx]
            bias_d = bias[dim_idx]

            # Kernel
            cov = matern52_kernel(current_x, current_x, ls_d, amp_d) + torch.eye(n) * JITTER

            # Mean: Linear Mapping
            mean = (current_x @ slope_d.unsqueeze(-1)).squeeze(-1) + bias_d

            # Sample Latent Variable
            h_d = pyro.sample(f"h_{layer_idx}_{dim_idx}", dist.MultivariateNormal(mean, cov))
            h_list.append(h_d)

        # Stack to create input for next layer: (N, Width)
        current_x = torch.stack(h_list, dim=1)

    # --- Final Output Layer ---
    # Input is the output of the last hidden layer
    input_dim = current_x.size(1)

    ls_out = pyro.sample("out_ls", dist.LogNormal(-1.0, 1.0).expand([input_dim]).to_event(1))  # ARD
    amp_out = pyro.sample("out_amp", dist.LogNormal(0.0, 1.0))
    noise = pyro.sample("obs_noise", dist.HalfNormal(0.3))

    # Note: Using unsqueeze(0) on ls_out to make it (1, D) for broadcasting
    cov_out = matern52_kernel(current_x, current_x, ls_out.unsqueeze(0), amp_out)
    noise_term = noise.view(-1, 1, 1) if noise.dim() == 1 else noise
    cov_final = cov_out + torch.eye(n) * (noise_term + JITTER)

    pyro.sample("obs", dist.MultivariateNormal(torch.zeros(n), cov_final), obs=y)


# --- Class Wrapper ---

class LightweightDeepGP:
    def __init__(self, train_x, train_y, hidden_dims=[2]):
        # Enforce 2D inputs
        self.train_x = train_x.clone()
        if self.train_x.dim() == 1:
            self.train_x = self.train_x.unsqueeze(-1)

        self.train_y = train_y
        self.hidden_dims = hidden_dims  # List of layer widths
        self.samples = None

    def fit_nuts(self, num_chains=1, num_samples=1000, warmup_steps=500):
        print(f"--- Fitting Deep GP (Depth: {len(self.hidden_dims)} hidden layers) ---")
        nuts_kernel = NUTS(_deep_gp_probabilistic_model, jit_compile=False)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                    num_chains=num_chains, mp_context="spawn")

        mcmc.run(self.train_x, self.train_y, hidden_dims=self.hidden_dims)
        self.samples = mcmc.get_samples()
        print("NUTS finished.")

    def predict_torch(self, X):
        if self.samples is None: raise RuntimeError("Model not trained.")

        # Enforce 2D inputs
        if X.dim() == 1: X = X.unsqueeze(-1)

        # Determine Batch vs Single prediction
        if X.dim() == 2:
            M = X.shape[0]
            X_flat = X
        else:
            M = X.shape[0] * X.shape[1]
            X_flat = X.view(M, -1)

        device = self.samples['out_ls'].device
        S = self.samples['out_ls'].shape[0]  # Number of MCMC samples
        N = self.train_x.shape[0]

        # 1. Prepare Initial Data
        # Expand Training Data: (S, N, D_in)
        curr_x_train = self.train_x.to(device).unsqueeze(0).expand(S, -1, -1)
        # Expand Test Data: (S, M, D_in)
        curr_x_test = X_flat.to(device).unsqueeze(0).expand(S, -1, -1)

        # 2. Propagate through Hidden Layers
        for layer_idx, width in enumerate(self.hidden_dims):
            next_x_test_list = []
            next_x_train_list = []  # We need to reconstruct train samples layer-by-layer for covariance

            for dim_idx in range(width):
                # Fetch Params
                # ls shape: (S, D_prev) -> Need (S, 1, D_prev) for kernel
                ls = self.samples[f'L{layer_idx}_ls'][:, dim_idx, :].unsqueeze(1)
                amp = self.samples[f'L{layer_idx}_amp'][:, dim_idx].view(S, 1, 1)
                slope = self.samples[f'L{layer_idx}_slope'][:, dim_idx, :]  # (S, D_prev)
                bias = self.samples[f'L{layer_idx}_bias'][:, dim_idx].view(S, 1, 1)

                # Fetch Latent Samples for Training Data (S, N) -> (S, N, 1)
                h_train_samples = self.samples[f'h_{layer_idx}_{dim_idx}'].unsqueeze(-1)
                next_x_train_list.append(h_train_samples)

                # --- GP Prediction Step ---
                # 1. Covariance of Training Data (at this layer)
                K_rr = matern52_kernel(curr_x_train, curr_x_train, ls, amp) + torch.eye(N, device=device) * JITTER
                L_rr = torch.linalg.cholesky(K_rr)

                # 2. Compute Prior Mean for Training Data (Linear Mapping)
                # (S, N, D) * (S, 1, D) -> sum -> (S, N, 1)
                train_prior_mean = (curr_x_train * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias

                # 3. Compute Weights (Alpha)
                residuals = h_train_samples - train_prior_mean
                weights = torch.cholesky_solve(residuals, L_rr)

                # 4. Cross Covariance & Test Covariance
                K_tr = matern52_kernel(curr_x_test, curr_x_train, ls, amp)
                K_tt = matern52_kernel(curr_x_test, curr_x_test, ls, amp) + torch.eye(M, device=device) * JITTER

                # 5. Posterior Mean & Covariance
                test_prior_mean = (curr_x_test * slope.unsqueeze(1)).sum(-1, keepdim=True) + bias
                mu_test = test_prior_mean + K_tr @ weights

                v = torch.linalg.solve(L_rr, K_tr.transpose(-1, -2))
                cov_test = K_tt - v.transpose(-1, -2) @ v

                # 6. Sample from Posterior (Propagate Uncertainty)
                rng_state = torch.get_rng_state()
                torch.manual_seed(12345 + layer_idx * 100 + dim_idx)  # Deterministic noise per layer/dim

                # Expand dist to handle batch shape correctly
                h_dist = dist.MultivariateNormal(mu_test.squeeze(-1), cov_test)
                h_test_sample = h_dist.rsample().unsqueeze(-1)

                torch.set_rng_state(rng_state)
                next_x_test_list.append(h_test_sample)

            # Stack for next layer inputs
            curr_x_test = torch.cat(next_x_test_list, dim=2)  # (S, M, Width)
            curr_x_train = torch.cat(next_x_train_list, dim=2)  # (S, N, Width)

        # 3. Final Output Layer
        # Inputs: curr_x_test (Hidden output), curr_x_train (Hidden samples)
        ls_out = self.samples['out_ls'].unsqueeze(1)  # (S, 1, Hidden_Last_Width)
        amp_out = self.samples['out_amp'].view(S, 1, 1)
        noise = self.samples['obs_noise'].view(S, 1, 1)

        K_rr = matern52_kernel(curr_x_train, curr_x_train, ls_out, amp_out) + torch.eye(N, device=device) * (
                    noise + JITTER)
        K_tr = matern52_kernel(curr_x_test, curr_x_train, ls_out, amp_out)

        # Target Values
        y_tr = self.train_y.to(device).view(1, N).expand(S, N).unsqueeze(-1)

        # Solve
        L_rr = torch.linalg.cholesky(K_rr)
        weights = torch.cholesky_solve(y_tr, L_rr)

        y_pred = K_tr @ weights  # (S, M, 1)

        # Reshape Output
        if X.dim() == 3: return y_pred.view(S, X.shape[0], X.shape[1], 1)
        return y_pred.view(S, X.shape[0], 1)

    def predict(self, test_x):
        with torch.no_grad():
            x_in = torch.as_tensor(test_x)
            if x_in.dim() == 1: x_in = x_in.unsqueeze(-1)
            preds = self.predict_torch(x_in)
            preds = preds.squeeze(-1).cpu().numpy()
            return preds.mean(axis=0), preds.std(axis=0)


# --- BoTorch Wrapper (Unchanged) ---
class DeepGPBoTorchModel(Model):
    def __init__(self, dgp_model):
        super().__init__()
        self.dgp = dgp_model
        self._num_outputs = 1

    def posterior(self, X, observation_noise=False, **kwargs):
        preds_samples = self.dgp.predict_torch(X)
        mu = preds_samples.mean(dim=0)
        var = preds_samples.var(dim=0)
        cov = torch.diag_embed(var.squeeze(-1) + 1e-6)
        mvn = MultivariateNormal(mu.squeeze(-1), cov)
        return GPyTorchPosterior(mvn)

    @property
    def num_outputs(self):
        return self._num_outputs


def ackley_1d(x, a=20, b=0.2, c=2 * np.pi):
    if x.dim() == 1: x = x.unsqueeze(-1)
    term1 = -a * torch.exp(-b * torch.sqrt(x ** 2))
    term2 = -torch.exp(torch.cos(c * x))
    return term1 + term2 + a + np.e


def main():
    print("Initializing 1D Ackley Benchmark (Full Scale)...")
    input_dim = 1

    # Standard Ackley Bounds [-32.768, 32.768]
    bounds = torch.tensor([[-32.768], [32.768]]).double()

    # 1. Generate Initial Data
    # Increased to 50 to capture high-frequency oscillations over the larger range
    n_init = 20
    train_x_raw = torch.rand(n_init, input_dim).double()
    train_x_raw = train_x_raw * (bounds[1] - bounds[0]) + bounds[0]

    # Query Ackley
    print(f"Observing {n_init} initial points...")
    train_y_raw = ackley_1d(train_x_raw).squeeze(-1)

    # 2. Standardization
    x_mean, x_std = train_x_raw.mean(0), train_x_raw.std(0) + 1e-6
    y_mean, y_std = train_y_raw.mean(), train_y_raw.std() + 1e-6

    train_x = (train_x_raw - x_mean) / x_std
    train_y = (train_y_raw - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    print(f"Data Shapes: X={train_x.shape}, Y={train_y.shape}")

    # 3. Fit Deep GP
    # Using [4] hidden units to allow for more flexibility on the complex surface
    dgp = LightweightDeepGP(train_x, train_y, hidden_dims=[1])
    dgp.fit_nuts(num_chains=1, num_samples=1000, warmup_steps=500)

    # 4. BoTorch Optimization
    std_bounds_lower = (bounds[0].to(device) - x_mean.to(device)) / x_std.to(device)
    std_bounds_upper = (bounds[1].to(device) - x_mean.to(device)) / x_std.to(device)
    bo_bounds = torch.stack([std_bounds_lower, std_bounds_upper])

    bo_model = DeepGPBoTorchModel(dgp)
    log_ei = LogExpectedImprovement(model=bo_model, best_f=train_y.min(), maximize=False)

    print("Optimizing Acquisition Function...")
    # Increased raw_samples to ensure we find the narrow minimum in the larger space
    candidate_norm, _ = optimize_acqf(log_ei, bounds=bo_bounds, q=1, num_restarts=10, raw_samples=500)

    # 5. Denormalize & Validate
    candidate_raw = candidate_norm.cpu() * x_std + x_mean
    print(f"Next Candidate (Raw Inputs): {candidate_raw.numpy()}")

    y_new_raw = ackley_1d(candidate_raw).item()
    print(f"Observed Result: {y_new_raw}")

    # 6. Visualization (1D)
    print("Generating 1D Plot...")

    n_grid = 400
    x_grid = torch.linspace(bounds[0].item(), bounds[1].item(), n_grid).unsqueeze(-1)

    y_true = ackley_1d(x_grid).squeeze(-1).numpy()

    x_grid_norm = (x_grid - x_mean) / x_std
    mu_norm, std_norm = dgp.predict(x_grid_norm)

    mu_raw = mu_norm * y_std.item() + y_mean.item()
    std_raw = std_norm * y_std.item()

    upper_bound = mu_raw + 2 * std_raw
    lower_bound = mu_raw - 2 * std_raw

    # Numpy Conversion Fix
    x_vals = x_grid.flatten().numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_vals, x_vals[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence'
    ))

    fig.add_trace(go.Scatter(x=x_vals, y=mu_raw, mode='lines',
                             line=dict(color='blue', width=2), name='Deep GP Mean'))

    fig.add_trace(go.Scatter(x=x_vals, y=y_true, mode='lines',
                             line=dict(color='black', dash='dash'), name='Ackley (True)'))

    fig.add_trace(go.Scatter(x=train_x_raw.flatten().numpy(), y=train_y_raw.numpy(),
                             mode='markers', marker=dict(color='red', size=6), name='Training Data'))

    fig.add_trace(go.Scatter(x=candidate_raw.flatten().numpy(), y=[y_new_raw],
                             mode='markers', marker=dict(color='green', size=12, symbol='star'), name='Next Candidate'))

    fig.update_layout(title="Full Scale Ackley [-32.768, 32.768]",
                      xaxis_title="Input X", yaxis_title="Output Y", height=600)
    fig.show()


if __name__ == "__main__":
    main()