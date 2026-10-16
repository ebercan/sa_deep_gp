import torch
import torch.nn as nn
import gpytorch
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import Mean
from gpytorch.constraints import GreaterThan
from botorch.models.transforms import Standardize, Normalize
from gpytorch.priors import NormalPrior, GammaPrior, LogNormalPrior

from benchmarks import DiscontinuousPolymerizationV2
# from pretti_sad_gp.deep_gp_testing import likelihood

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
N_INIT, N_ITER, N_REPS = 19, 1, 1
TRAIN_ITER = 150
warnings.filterwarnings("ignore")


class LearnablePhysicsMean(Mean):
    def __init__(self, prior_gp):
        super().__init__()
        self.prior_gp = prior_gp
        # self.prior_gp.eval()
        for p in self.prior_gp.parameters():
            p.requires_grad = False
        self.register_parameter("scaling", nn.Parameter(torch.tensor([1.0], device=DEVICE, dtype=DTYPE)))
        self.register_parameter("bias", nn.Parameter(torch.tensor([0.0], device=DEVICE, dtype=DTYPE)))

    def forward(self, x):
        self.prior_gp.eval()
        res = self.prior_gp(x).mean
        return self.scaling * res + self.bias


class PhysicsWarper(gpytorch.Module):
    """
    Layer 1: Deterministic or Probabilistic Warping based on LF Physics.
    Replaces the 'illegal' GP1.
    """

    def __init__(self, prior_gp, joint=False):
        super().__init__()
        self.prior_gp = prior_gp
        self.prior_gp.eval()
        for p in self.prior_gp.parameters(): p.requires_grad = False

        # JOINT PARAMS: alpha * z + beta
        # If joint=True, these learn to correct the 'skew' of the physics
        self.register_parameter("scaling", nn.Parameter(torch.tensor([1.0], device=DEVICE, dtype=DTYPE)))
        self.register_prior("scaling_prior", LogNormalPrior(0.0, 0.5), "scaling")

        self.register_parameter("bias", nn.Parameter(torch.tensor([0.0], device=DEVICE, dtype=DTYPE)))
        self.register_prior("bias_prior", NormalPrior(0.0, 1.0), "bias")

        self.scaling.requires_grad = joint
        self.bias.requires_grad = joint

    @property
    def scaling(self):
        return torch.exp(self.raw_scaling)

    def forward(self, x, return_dist=False):
        self.prior_gp.eval()
        dist = self.prior_gp(x)
        if return_dist:
            return dist
        return dist.mean


class HiddenGPLayer(GPyTorchModel):
    def __init__(self, prior_gp, joint=False):
        super().__init__()
        self.prior_gp = prior_gp  # This is your pre-trained LF GP
        for p in self.prior_gp.parameters(): p.requires_grad = False

        self.register_parameter("raw_scaling", nn.Parameter(torch.tensor([0.0], device=DEVICE)))
        self.register_parameter("bias", nn.Parameter(torch.tensor([0.0], device=DEVICE)))

        self.register_prior("scaling_prior", LogNormalPrior(0.0, 0.5), "scaling")
        self.register_prior("bias_prior", NormalPrior(0.0, 1.0), "bias")

        self.raw_scaling.requires_grad = joint
        self.bias.requires_grad = joint

    @property
    def scaling(self): return torch.exp(self.raw_scaling)

    def forward(self, x):
        self.prior_gp.eval()
        dist = self.prior_gp(x)

        mean = self.scaling * dist.mean + self.bias
        variance = (self.scaling ** 2) * dist.variance

        return gpytorch.distributions.MultivariateNormal(mean, torch.diag_embed(variance))


class DeepPosterior(Posterior):
    def __init__(self, gp1, gp2, X):
        self.gp1, self.gp2, self.X = gp1, gp2, self.X_check(X)

    def X_check(self, X): return X if X.ndim > 1 else X.unsqueeze(0)

    @property
    def mean(self):
        with torch.no_grad(), gpytorch.settings.prior_mode(True):
            # Pass through GP1
            res1 = self.gp1(self.X[..., :2])
            # UNDO STANDARDIZATION: Convert z-scores back to physical concentration [0, 1]
            z_raw = self.gp1.outcome_transform.untransform(res1.mean.unsqueeze(-1))[0]
            # Feed physical concentration to GP2
            return self.gp2(torch.cat([z_raw, self.X[..., 2:]], dim=-1)).mean

    @property
    def variance(self):
        with torch.no_grad(), gpytorch.settings.prior_mode(True):
            res1 = self.gp1(self.X[..., :2])
            z_raw = self.gp1.outcome_transform.untransform(res1.mean.unsqueeze(-1))[0]
            return self.gp2(torch.cat([z_raw, self.X[..., 2:]], dim=-1)).variance

    def rsample_from_base_samples(self, sample_shape, base_samples):
        with gpytorch.settings.prior_mode(True):
            post1 = self.gp1.posterior(self.X[..., :2])
            z_s = post1.rsample_from_base_samples(torch.Size(), base_samples[..., 0:1])
            if z_s.ndim < 3: z_s = z_s.unsqueeze(-1)
            # UNTRANSFORM SAMPLES: Ensure fantasy samples match the physical scale GP2 expects
            z_s_raw = self.gp1.outcome_transform.untransform(z_s)[0]
            x3 = self.X[..., 2:].expand(z_s_raw.shape[:-1] + (1,))
            return self.gp2.posterior(torch.cat([z_s_raw, x3], dim=-1)).rsample_from_base_samples(torch.Size(),
                                                                                                  base_samples[...,
                                                                                                  1:2])

    # def rsample(self, sample_shape=torch.Size(), base_samples=None):
    #     if base_samples is None:
    #         base_samples = torch.randn(sample_shape + self.base_sample_shape, device=self.device, dtype=self.dtype)
    #     return self.rsample_from_base_samples(sample_shape, base_samples)


class IntegratedDeepGP(GPyTorchModel):
    def __init__(self, train_x, train_y, prior_gps, joint=False, latent=False):
        super().__init__()
        self.train_inputs = (train_x,)
        self.train_targets = train_y

        self.joint = joint
        self.latent = latent
        self.num_samples = 512 if latent else 1

        self.layer1 = HiddenGPLayer(prior_gps[0], joint=joint)

        with torch.no_grad():
            z_est = self.warper(train_x[:, :2], return_dist=False)
            gp2_input = torch.cat([z_est.unsqueeze(-1), train_x[:, 2:]], dim=-1)

        self.gp2 = SingleTaskGP(
            gp2_input,
            train_y[:, 1:2],
            input_transform=Normalize[2],
            outcome_transform=Standardize(m=1)
        )
        self.gp2.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))

        self.gp2.mean_module = LearnablePhysicsMean(prior_gps[1])
        self.gp2.mean_module.register_prior("sc_prior", LogNormalPrior(0.0, 0.5), "scaling")
        self.gp2.mean_module.register_prior("b_prior", NormalPrior(0.0, 1.0), "bias")


        # if self.latent:
        #     dist1 = self.lf_gp1(x[..., :2])
        #     z_samples = dist1.rsample(torch.Size([self.num_samples]))
        #
        # self.gp1 = SingleTaskGP(train_x[:, :2], train_y[:, :1],
        #                         input_transform=Normalize(d=2),
        #                         outcome_transform=Standardize(m=1)
        #                         )
        #
        # self.gp1.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
        # self.gp1.mean_module = LearnablePhysicsMean(prior_gps[0])
        # self.gp2 = SingleTaskGP(torch.cat([train_y[:, :1], train_x[:, 2:]], -1), train_y[:, 1:2],
        #                         input_transform=Normalize(d=2),
        #                         outcome_transform=Standardize(m=1)
        #                         )
        # self.gp2.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
        # self.gp2.mean_module = LearnablePhysicsMean(prior_gps[1])

    @property
    def num_outputs(self):
        return 1

    def forward(self, x):
        z_dist = self.layer1(x[..., :2])

        # 2. LATENT LOGIC
        if self.latent:
            z_samples = z_dist.rsample(torch.Size([self.num_samples]))
            if z_samples.ndim == 2: z_samples = z_samples.unsqueeze(-1)

            # z_warped = z_samples * self.warper.scaling + self.warper.bias

            # x_exp = x.expand(self.num_samples, *x.shape)
            # gp2_in = torch.cat([z_warped, x_exp[..., 2:]], dim=-1)

            x_rest = x[..., 2:].expand(self.num_samples, *x[..., 2:].shape)
            gp2_in = torch.cat([z_samples, x_rest], dim=-1)

        else:
            # z_warped = z_mean * self.warper.scaling + self.warper.bias
            gp2_in = torch.cat([z_dist.mean.unsqueeze(-1), x[..., 2:]], dim=-1)

        return self.gp2(gp2_in)

    def posterior(self, X, observation_noise=False, **kwargs):
        # if self.is_joint: return DeepPosterior(self.gp1, self.gp2, X)
        # with gpytorch.settings.prior_mode(True):
        #     m1_res = self.gp1(X[..., :2])
        #     m1_raw = self.gp1.outcome_transform.untransform(m1_res.mean.unsqueeze(-1))[0]
        #     return self.gp2.posterior(torch.cat([m1_raw, X[..., 2:]], dim=-1))
        mvn = self.forward(X)
        if self.latent:
            return mvn
        return self.gp2.posterior(mvn.mean, mvn.covariance_matrix, observation_noise=observation_noise)


def train_and_track(model, train_x, train_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # mll1 = ExactMarginalLogLikelihood(model.gp1.likelihood, model.gp1)
    # mll2 = ExactMarginalLogLikelihood(model.gp2.likelihood, model.gp2)
    mll = ExactMarginalLogLikelihood(model.gp2.likelihood, model.gp2)

    model.train()

    for _ in range(TRAIN_ITER):
        optimizer.zero_grad()
        output = model(train_x)

        if model.latent:
            y_exp = train_y[:, 1].expand(model.num_samples, -1)
            loss = -mll(output, y_exp).mean
        else:
            loss = -mll(output, train_y[:, 1])

        loss.backward()
        optimizer.step()
    model.eval()

    sc1 = model.warper.scaling.item()
    sc2 = model.gp2.mean_module.scaling.item()
    print(f"    Health -> Warper_Scale: {sc1:.3f} | Output_Scale: {sc2:.3f}")


def visualize_layers(model, hf_env, train_x, train_y, name):
    # Updated to handle new architecture
    res = 40
    x_mono = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)
    z_grid = torch.linspace(0, 1, res, device=DEVICE, dtype=DTYPE)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Layer 1 (Physics)", "Full Model", "Layer 2 (Reaction)"))

    def compute_scenario(cat, T):
        with torch.no_grad(), gpytorch.settings.prior_mode(False):  # Standard Posterior

            # Layer 1 Visualization
            X1 = torch.stack([torch.full_like(x_mono, cat), x_mono], -1)
            # Access prior directly for visualization
            p1_dist = model.warper(X1, return_dist=True)
            m1, s1 = p1_dist.mean, p1_dist.variance.sqrt()

            # Full Model
            Xf = torch.stack([torch.full_like(x_mono, cat), x_mono, torch.full_like(x_mono, T)], -1)
            pf = model.posterior(Xf)
            mf, sf = pf.mean.squeeze(), pf.variance.sqrt().squeeze()

            # Layer 2 Visualization (Approximation on grid)
            X2 = torch.stack([z_grid, torch.full_like(z_grid, T)], -1)
            p2 = model.gp2.posterior(X2)
            m2, s2 = p2.mean.squeeze(), p2.variance.sqrt().squeeze()

            # Ground Truths
            tz = hf_env.step_1_mixing(cat, x_mono.cpu().numpy())
            tyf = hf_env.step_3_reaction(tz, hf_env.step_2_thermal(tz, T))
            ty2 = hf_env.step_3_reaction(z_grid.cpu().numpy(), hf_env.step_2_thermal(z_grid.cpu().numpy(), T))

            mc = torch.abs(train_x[:, 0] - cat) < 0.15
            mt = torch.abs(train_x[:, 2] - T) < 0.15

        return (m1, s1, tz, mf, sf, tyf, m2, s2, ty2, mc, mt)

    m1, s1, t1, mf, sf, tf, m2, s2, t2, mc, mt = compute_scenario(0.5, 0.5)

    def add_subplot_group(fig, x, m, s, t, obs_x, obs_y, col, color, lbl):
        group_color = 'blue' if col == 1 else 'purple' if col == 2 else 'green'
        fig.add_trace(
            go.Scatter(x=x.cpu(), y=(m + 1.96 * s).cpu(), line=dict(width=0), showlegend=False, legendgroup=lbl), row=1,
            col=col)
        fig.add_trace(go.Scatter(x=x.cpu(), y=(m - 1.96 * s).cpu(), fill='tonexty', fillcolor=color, line=dict(width=0),
                                 showlegend=False, legendgroup=lbl), row=1, col=col)
        fig.add_trace(go.Scatter(x=x.cpu(), y=m.cpu(), name=lbl, line=dict(color=group_color), legendgroup=lbl), row=1,
                      col=col)
        fig.add_trace(go.Scatter(x=x.cpu(), y=t, name=f"True {lbl}", line=dict(color='black', dash='dot')), row=1,
                      col=col)
        fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='markers', name=f'Obs {lbl}',
                                 marker=dict(symbol='x', size=8, color=group_color)), row=1, col=col)

    add_subplot_group(fig, x_mono, m1, s1, t1, train_x[mc, 1].cpu(), train_y[mc, 0].cpu(), 1, 'rgba(0,0,255,0.15)',
                      'Mixing (Z)')
    add_subplot_group(fig, x_mono, mf, sf, tf, train_x[mc & mt, 1].cpu(), train_y[mc & mt, 1].cpu(), 2,
                      'rgba(255,0,255,0.15)', 'Full Model')
    add_subplot_group(fig, z_grid, m2, s2, t2, train_y[mt, 0].cpu(), train_y[mt, 1].cpu(), 3, 'rgba(0,255,0,0.15)',
                      'Reaction (Y)')

    frames = []
    for cat in cat_steps:
        for T in T_steps:
            f_m1, f_s1, f_t1, f_mf, f_sf, f_tf, f_m2, f_s2, f_t2, f_mc, f_mt = compute_scenario(cat, T)
            frames.append(go.Frame(
                data=[
                    go.Scatter(y=(f_m1 + 1.96 * f_s1).cpu()), go.Scatter(y=(f_m1 - 1.96 * f_s1).cpu()),
                    go.Scatter(y=f_m1.cpu()), go.Scatter(y=f_t1),
                    go.Scatter(x=train_x[f_mc, 1].cpu(), y=train_y[f_mc, 0].cpu()),
                    go.Scatter(y=(f_mf + 1.96 * f_sf).cpu()), go.Scatter(y=(f_mf - 1.96 * f_sf).cpu()),
                    go.Scatter(y=f_mf.cpu()), go.Scatter(y=f_tf),
                    go.Scatter(x=train_x[f_mc & f_mt, 1].cpu(), y=train_y[f_mc & f_mt, 1].cpu()),
                    go.Scatter(y=(f_m2 + 1.96 * f_s2).cpu()), go.Scatter(y=(f_m2 - 1.96 * f_s2).cpu()),
                    go.Scatter(y=f_m2.cpu()), go.Scatter(y=f_t2),
                    go.Scatter(x=train_y[f_mt, 0].cpu(), y=train_y[f_mt, 1].cpu())
                ],
                name=f"c{cat}t{T}"
            ))

    sliders = [
        dict(active=1, currentvalue={"prefix": "Catalyst: "}, pad={"t": 50}, steps=[
            dict(method="animate", label=f"{c:.1f}",
                 args=[[f"c{c}t0.5"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]) for c in
            cat_steps
        ]),
        dict(active=1, currentvalue={"prefix": "Temperature: "}, pad={"t": 150}, steps=[
            dict(method="animate", label=f"{t:.1f}",
                 args=[[f"c0.5t{t}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]) for t in T_steps
        ])
    ]
    fig.frames = frames
    fig.update_layout(sliders=sliders, title=f"Physics-Guided Layers: {name}", template="plotly_white", height=600)
    fig.show()


def run_benchmark():
    hf_env = DiscontinuousPolymerizationV2(mode='HF')
    lf_env = DiscontinuousPolymerizationV2(mode='LF')

    x_lf = torch.rand(200, 3, device=DEVICE, dtype=DTYPE)
    y_lf = lf_env.observe_full(x_lf)

    p1 = train_gp(x_lf[:, :2], y_lf[:, :1])
    p2 = train_gp(torch.cat([y_lf[:, :1], x_lf[:, 2:]], -1), y_lf[:, 1:2])

    results = {}

    scenarios = [
        ("Sequential", False, False),  # joint=False, latent=False
        ("Joint", True, False),  # joint=True,  latent=False
        ("Latent", True, True)  # joint=True,  latent=True
    ]

    results = {"Sequential": [], "Joint": [], "BASELINE": []}
    seeds = [torch.randint(0, 10000, (1,)).item() for _ in range(N_REPS)]
    for mode_name, is_joint, is_latent in scenarios:
        results[mode_name] = []
        for r in range(N_REPS):
            torch.manual_seed(seeds[r])

            ix = torch.rand(N_INIT, 3, device=DEVICE, dtype=DTYPE)
            iy = hf_env.observe_full(ix)
            cx, cy = ix.clone(), iy.clone()

            trace = [cy[:, 1].max().item()]
            print(f"\n>>> REP {r} | {mode_name}")

            for it in range(N_ITER):
                m = IntegratedDeepGP(cx, cy, [p1, p2], joint=is_joint, latent=is_latent)

                train_and_track(m, cx, cy)

                acq = qLogExpectedImprovement(m, cy[:, 1].max(), sampler=IIDNormalSampler(64))
                nx, _ = optimize_acqf(acq, hf_env.bounds, 1, 5, 64)

                ny = hf_env.observe_full(nx)
                cx, cy = torch.cat([cx, nx]), torch.cat([cy, ny])

                trace.append(cy[:, 1].max().item())
                print(f"  It {it + 1} | Best: {trace[-1]:.4f}")

            results[mode_name].append(trace)

    plot_final(results)

def train_gp(x, y):
    # if y.ndim == 1: y = y.unsqueeze(-1)
    m = SingleTaskGP(x, y,
                     input_transform=Normalize(d=x.shape[1]),
                     outcome_transform=Standardize(m=y.shape[1])
                     )
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    fit_gpytorch_mll(mll)

    ls = m.covar_module.base_kernel.lengthscale.detach().squeeze().tolist()
    print(f"  GP trained. Lengthscales: {ls} | Noise: {m.likelihood.noise.item():.2e}")

    return m

def plot_final(res):
    fig = go.Figure()
    for name, reps in res.items():
        if not reps: continue
        arr = np.array(reps)
        mean = np.mean(arr, axis=0)
        x = list(range(len(mean)))
        fig.add_trace(go.Scatter(x=x, y=mean, name=name))
    fig.show()

if __name__ == "__main__":
    run_benchmark()