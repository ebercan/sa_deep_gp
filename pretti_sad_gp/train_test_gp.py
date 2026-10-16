"""Interactive check of an LF GP fit for PolymerizationV2 (helper study).

It fits a SingleTaskGP from (x0, x1) to 20 * C_eff on 200 random LF points and shows
the posterior mean +- 2 std along x0, with a slider for x1. Observations within one
slider step are overlaid. Run from `pretti_sad_gp/`; the figure opens in Plotly.
"""

from benchmarks import DiscontinuousPolymerizationV2
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize


lf_env = DiscontinuousPolymerizationV2(device=torch.device('cpu'))
x_lf = torch.rand(200, 3, dtype=torch.float64)
# Columns: [C_eff (step 1), T_sys (step 2), yield (step 3)] of the LF model
y_lf = 20 * torch.cat(lf_env.observe_all_steps(x_lf, mode='LF'), dim=1)

m = SingleTaskGP(train_X=x_lf[:, :2], train_Y=y_lf[:, :1],
                 input_transform=Normalize(d=2), outcome_transform=Standardize(m=1),
                 )

mll = ExactMarginalLogLikelihood(m.likelihood, m)
mll = fit_gpytorch_mll(mll)

import torch
import numpy as np
import plotly.graph_objects as go

n_x1 = 100
n_slider = 25
x1_lin = torch.linspace(0, 1, n_x1)
x2_lin = torch.linspace(0, 1, n_slider)
step_size = (x2_lin[1] - x2_lin[0]).item()

frames = []
m.eval()

for val_x2 in x2_lin:
    test_x = torch.stack([x1_lin, torch.full_like(x1_lin, val_x2)], dim=-1)

    with torch.no_grad():
        posterior = m.posterior(test_x)
        mean = posterior.mean.squeeze().numpy()
        std = posterior.variance.sqrt().squeeze().numpy()

    upper = mean + 2 * std
    lower = mean - 2 * std

    mask = (x_lf[:, 1] >= val_x2 - step_size) & (x_lf[:, 1] <= val_x2 + step_size)
    obs_x = x_lf[mask, 0].numpy()
    obs_y = y_lf[mask, 0].numpy()

    frames.append(go.Frame(
        data=[
            go.Scatter(x=x1_lin.numpy(), y=mean, mode='lines', line=dict(color='blue'), name='GP Mean'),
            go.Scatter(x=x1_lin.numpy(), y=upper, mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=x1_lin.numpy(), y=lower, mode='lines', fill='tonexty', fillcolor='rgba(0,0,255,0.2)',
                       line=dict(width=0), name=r'$2\sigma$ Confidence'),
            go.Scatter(x=obs_x, y=obs_y, mode='markers', marker=dict(color='red', size=6), name='Observations')
        ],
        name=f"{val_x2:.2f}"
    ))

init_val = x2_lin[0]
init_mask = (x_lf[:, 1] >= init_val - step_size) & (x_lf[:, 1] <= init_val + step_size)
with torch.no_grad():
    init_post = m.posterior(torch.stack([x1_lin, torch.full_like(x1_lin, init_val)], dim=-1))
    im, istd = init_post.mean.squeeze().numpy(), init_post.variance.sqrt().squeeze().numpy()

fig = go.Figure(
    data=[
        go.Scatter(x=x1_lin.numpy(), y=im, mode='lines', line=dict(color='blue'), name='GP Mean'),
        go.Scatter(x=x1_lin.numpy(), y=im + 2 * istd, mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=x1_lin.numpy(), y=im - 2 * istd, mode='lines', fill='tonexty', fillcolor='rgba(0,0,255,0.2)',
                   line=dict(width=0), name=r'$2\sigma$ Confidence'),
        go.Scatter(x=x_lf[init_mask, 0].numpy(), y=y_lf[init_mask, 0].numpy(), mode='markers',
                   marker=dict(color='red', size=6), name='Observations')
    ],
    layout=go.Layout(
        title="GP Posterior Slice over $x_1$",
        xaxis=dict(title="$x_1$", range=[0, 1]),
        yaxis=dict(title="Target Output", range=[y_lf[:, 0].min().item() - 0.5, y_lf[:, 0].max().item() + 0.5]),
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': '$x_2$: '},
            'pad': {'t': 50},
            'steps': [
                {
                    'label': f"{v:.2f}",
                    'method': 'animate',
                    'args': [[f"{v:.2f}"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                } for v in x2_lin
            ]
        }]
    ),
    frames=frames
)

fig.show()