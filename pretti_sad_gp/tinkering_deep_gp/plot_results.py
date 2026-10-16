import torch
import glob
import os
import plotly.graph_objects as go


def visualize_comparison(results_dir):
    fig = go.Figure()
    files = glob.glob(os.path.join(results_dir, "*.pt"))

    for f in files:
        data = torch.load(f)
        label = os.path.basename(f).replace(".pt", "")
        fig.add_trace(go.Scatter(y=data['history_best'], mode='lines', name=label))

    fig.update_layout(title="Optimization Convergence", xaxis_title="Iteration", yaxis_title="Best Objective")
    fig.show()