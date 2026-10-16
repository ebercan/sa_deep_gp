# BoTorchified DeepGP with proper input/outcome transforms, forward, posterior
import torch
import gpytorch
from typing import Optional, List, Tuple

from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.priors import GammaPrior
from gpytorch.priors import LogNormalPrior

from botorch.models.model import Model  # for type hint parity
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior

from torch.utils.data import TensorDataset, DataLoader
from botorch.test_functions import Ackley

import plotly.graph_objects as go
import tqdm

@torch.no_grad()
def kl_by_layer(model):
    names = [n for (n, *_ ) in model.hidden_specs] + ["output"]
    out = {}
    for n in names:
        layer = getattr(model, n) if n != "output" else model.output_layer
        out[n] = layer.variational_strategy.kl_divergence()
    return out

@torch.no_grad()
def latent_std_on_train(model, X_int):
    x = X_int
    for name, out_dim, _ in model.hidden_specs:
        mvn = getattr(model, name)(x)
        m = mvn.mean
        if m.dim()==2 and m.size(0)==out_dim: m = m.transpose(-1, -2)
        x = m
    return float(x.std())

def _collapse_samples_to_moments(mu: torch.Tensor, var: torch.Tensor):
    """
    Collapse all leading sample/batch dims to [1, N] using law of total variance.
    Last dim is treated as the event dim (N).
    """
    # ensure var is present
    if var is None:
        var = torch.zeros_like(mu)

    # [S..., N] -> [S_total, N]
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
        var = var.unsqueeze(0)

    N = mu.size(-1)
    if mu.dim() == 2:
        mu_f, var_f = mu, var
    else:
        S_total = int(torch.tensor(mu.shape[:-1]).prod().item())
        mu_f = mu.reshape(S_total, N)
        var_f = var.reshape(S_total, N)

    mu_m = mu_f.mean(dim=0, keepdim=True)           # [1, N]
    var_m = var_f.mean(dim=0, keepdim=True)         # [1, N]
    mu_v = mu_f.var(dim=0, unbiased=False, keepdim=True)  # [1, N]
    return mu_m, var_m + mu_v

class GPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32, mean_type="constant", inducing_source=None):
        def pick_from_data(X, M, D, batch_shape=None):
            M = min(M, X.shape[0])
            Z = X[torch.randperm(X.shape[0])[:M]]
            if Z.shape[-1] != D:
                Z = torch.randn(M, D, dtype=X.dtype, device=X.device)
            return Z if batch_shape is None else Z.unsqueeze(0).expand(*batch_shape, M, D).contiguous()

        if inducing_source is not None:
            num_inducing = min(num_inducing, inducing_source.shape[0])

        if output_dims is None:
            batch_shape = torch.Size([])
            inducing_points = (torch.randn(num_inducing, input_dims)
                               if inducing_source is None else pick_from_data(inducing_source, num_inducing, input_dims))
        else:
            batch_shape = torch.Size([output_dims])
            inducing_points = (torch.randn(output_dims, num_inducing, input_dims)
                               if inducing_source is None else pick_from_data(inducing_source, num_inducing, input_dims, batch_shape))

        q = NaturalVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        vs = VariationalStrategy(
            self,
            inducing_points,
            q,
            learn_inducing_locations=True,
        )

        super().__init__(vs, input_dims, output_dims)

        self.mean_module = ConstantMean(batch_shape=batch_shape) if mean_type == "constant" else LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

        # Sensible init on normalized inputs X∈[0,1]
        self.covar_module.base_kernel.initialize(lengthscale=0.2)
        self.covar_module.initialize(outputscale=1.0)

        # Light priors to prevent pathologies
        # inside GPLayer.__init__ after creating self.covar_module
        # self.covar_module.base_kernel.register_prior(
        #     "lengthscale_prior",
        #     LogNormalPrior(loc=-1.609, scale=0.5),  # median ~ 0.2
        #     "lengthscale",
        # )
        # self.covar_module.register_prior(
        #     "outputscale_prior",
        #     LogNormalPrior(loc=0.0, scale=0.5),  # median ~ 1.0
        #     "outputscale",
        # )

    def forward(self, X):
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))

class BoTorchDeepGP(DeepGP, Model):
    """
    DeepGP that behaves like BoTorch models re: transforms & posterior:
      - input_transform: applied in `forward` when training, in `posterior` when evaluating
      - outcome_transform: targets transformed for training; predictions untransformed in `posterior`
    Hidden layers are specified as [(name, out_dim[, mean_type])].
    """
    def __init__(
        self,
        input_dims: int,
        hidden_specs: List[Tuple[str, int, str]] | List[Tuple[str, int]],
        num_inducing: int | List[int] = 32,
        inducing_source: Optional[torch.Tensor] = None,     # should be TRANSFORMED X if you pass one
        output_mean_type: str = "constant",
        noise_prior: gpytorch.priors.Prior = GammaPrior(2.0, 40.0),
        init_noise: float = 1e-3,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ):
        super().__init__()
        self.input_dims = input_dims

        # Register BoTorch transforms as submodules (move with .to(), saved in state_dict)
        self.input_transform = input_transform
        if self.input_transform is not None:
            self.add_module("input_transform", self.input_transform)
        self.outcome_transform = outcome_transform
        if self.outcome_transform is not None:
            self.add_module("outcome_transform", self.outcome_transform)

        parsed = []
        for spec in hidden_specs:
            if len(spec) == 2:
                name, out_dim = spec; mtype = "linear"
            else:
                name, out_dim, mtype = spec
            parsed.append((str(name), int(out_dim), str(mtype)))
        self.hidden_specs = parsed
        self.hidden_layers = torch.nn.ModuleList()

        total_layers = len(parsed) + 1
        if isinstance(num_inducing, int):
            Ms = [num_inducing] * total_layers
        else:
            if len(num_inducing) == total_layers:
                Ms = list(num_inducing)
            elif len(num_inducing) == len(parsed):
                Ms = list(num_inducing) + [num_inducing[-1]]
            else:
                raise ValueError("num_inducing must be int or list of length len(hidden_specs) or len(hidden_specs)+1")

        prev = input_dims
        for i, (name, out_dim, mtype) in enumerate(parsed):
            layer = GPLayer(prev, out_dim, num_inducing=Ms[i], mean_type=mtype, inducing_source=inducing_source)
            setattr(self, name, layer)
            self.hidden_layers.append(layer)
            prev = out_dim

        self.output_layer = GPLayer(prev, None, num_inducing=Ms[-1], mean_type=output_mean_type,
                                    inducing_source=inducing_source)

        # Likelihood lives in transformed outcome space
        self.likelihood = GaussianLikelihood(noise_prior=noise_prior)
        self.likelihood.noise_covar.initialize(noise=init_noise)

    def set_train_data(self, inputs=None, targets=None, strict: bool = True, **kwargs):
        """
        BoTorch-compatible:
        - Called by BoTorch internals as set_train_data(X_tf, strict=False) to refresh transformed inputs.
        - You can also call set_train_data(X, Y) yourself (original units); we’ll store the *transformed* copies.
        """
        # Case A: BoTorch internal update (inputs only; already transformed)
        if targets is None:
            if inputs is not None:
                if isinstance(inputs, torch.Tensor):
                    self.train_inputs = (inputs,)
                else:
                    # tuple/list of inputs
                    self.train_inputs = tuple(inputs)
            return self

        # Case B: user-style call (original X, Y) -> store transformed versions
        self.train()  # ensure forward applies input transform
        X_int = self._transform_X_for_forward(inputs)
        Y_int = self._transform_Y_for_training(targets)
        self.train_inputs = (X_int,)
        self.train_targets = Y_int.squeeze(-1)
        # keep originals around if you want
        self.X_train = inputs
        self.Y_train = targets
        return self

    def _transform_X_for_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Only apply the input transform here when TRAINING.
        (In EVAL, posterior(...) applies the transform before calling forward.)
        """
        if self.input_transform is not None and self.training:
            # Respect transform_on_train flag inside the transform itself
            X = self.input_transform(X)
        return X

    def _transform_Y_for_training(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform targets for training (like SingleTaskGP stores transformed targets).
        """
        if self.outcome_transform is None:
            return Y
        Yt, _ = self.outcome_transform(Y, None)
        return Yt

    def _untransform_posterior(self, mean: torch.Tensor, var: Optional[torch.Tensor]):
        if self.outcome_transform is None:
            return mean, var
        self.outcome_transform.eval()
        return self.outcome_transform.untransform(mean, var)

    def forward(self, X: torch.Tensor) -> MultivariateNormal:
        X = self._transform_X_for_forward(X)  # train-time transform only
        out = X
        for (name, _out_dim, _mtype) in self.hidden_specs:
            out = getattr(self, name)(out).rsample()  # <= THIS was missing
        out = self.output_layer(out)
        return out

    # ----------- BoTorch-style posterior -----------
    @torch.no_grad()
    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        num_likelihood_samples: int = 256,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        Eval path:
          - apply input_transform in eval mode
          - run DeepGP; include likelihood if observation_noise=True
          - collapse MC sample dims
          - untransform with outcome_transform
          - wrap as a GPyTorchPosterior (diagonal covariance)
        """
        self.eval()

        # Apply input transform in EVAL path
        if self.input_transform is not None:
            self.input_transform.eval()
            X = self.input_transform(X)

        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples), \
                gpytorch.settings.cholesky_jitter(1e-3):
            latent = self(X)
            dist = self.likelihood(latent) if observation_noise else latent

        # Collapse sample dims -> [1, N]
        mu, var = _collapse_samples_to_moments(dist.mean, dist.variance)
        mu = mu.squeeze(0).unsqueeze(-1)   # [N,1]
        var = var.squeeze(0).unsqueeze(-1) # [N,1] (marginal)

        # Untransform back to original units
        mu_orig, var_orig = self._untransform_posterior(mu, var)  # each [N,1]
        mu_orig = mu_orig.squeeze(-1)                              # [N]
        var_orig = None if var_orig is None else var_orig.squeeze(-1)  # [N]

        # Build a diagonal MVN for the Posterior wrapper (DeepGP off-diagonals not reliable)
        if var_orig is None:
            # no variance -> treat as tiny diag to keep shapes consistent
            cov = torch.diag_embed(torch.zeros_like(mu_orig))
        else:
            cov = torch.diag_embed(var_orig.clamp_min(0.0))

        mvn = MultivariateNormal(mean=mu_orig, covariance_matrix=cov)
        return GPyTorchPosterior(mvn)

    def make_std_train_loader(self, X_orig, Y_orig, batch_size=64, shuffle=True):
        self.train()
        Y_int = self._transform_Y_for_training(Y_orig)
        ds = TensorDataset(X_orig, Y_int.squeeze(-1))
        # For introspection, also return the transformed copy:
        X_int = self._transform_X_for_forward(X_orig)
        self.train_inputs = (X_int,)  # what BoTorch expects if something inspects it
        self.train_targets = Y_int.squeeze(-1)
        return DataLoader(ds, batch_size=min(batch_size, X_orig.shape[0]), shuffle=shuffle), X_int, Y_int


def train_botorch_deepgp(
    model: BoTorchDeepGP,
    train_X_orig: torch.Tensor,
    train_Y_orig: torch.Tensor,
    epochs: int = 400,
    base_lr: float = 5e-4,             # keep small for hypers
    initial_num_samples: int = 16,     # sample *through layers*
    final_num_samples: int = 64,
    batch_size: int = 64,
):
    loader, X_int, Y_int = model.make_std_train_loader(train_X_orig, train_Y_orig, batch_size=batch_size, shuffle=True)

    nvd_params = []
    for m in model.modules():
        if isinstance(m, NaturalVariationalDistribution):
            nvd_params += list(m.parameters())
    nvd_ids = {id(p) for p in nvd_params}
    hyp_params = [p for p in model.parameters() if id(p) not in nvd_ids]

    ngd  = gpytorch.optim.NGD(nvd_params, num_data=X_int.shape[0], lr=0.1)
    adam = torch.optim.Adam(hyp_params, lr=base_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=epochs, eta_min=base_lr/50)

    base_elbo = VariationalELBO(model.likelihood, model, num_data=X_int.shape[0])
    mll = DeepApproximateMLL(base_elbo)

    # reinitialize_inducing_points(model, X_int)

    print(f"\n🚀 Training DeepGP with {X_int.shape[0]} samples, {epochs} epochs, lr={base_lr:.1e}")

    for epoch in range(epochs):
        model.train()
        base_elbo.beta = 1.0 if epoch >= int(0.6*epochs) else 0.1 + 0.9*epoch/int(0.6*epochs)

        for xb, yb in loader:
            ngd.zero_grad();
            adam.zero_grad()
            with gpytorch.settings.variational_cholesky_jitter(double_value=1e-2), \
                    gpytorch.settings.num_likelihood_samples(128):
                out = model(xb)
                elbo = mll(out, yb)  # can be vector
                loss = -elbo.sum()  # <-- make scalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hyp_params, 1.0)
            ngd.step();
            adam.step()

        sch.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d}: loss={loss.item():.6f}, beta={base_elbo.beta:.3f}")
            print_hypers(model)
            print(f"hidden_out_std={latent_std_on_train(model, X_int):.4f}")
            print("KL per layer:", kl_by_layer(model))

    return model

@torch.no_grad()
def print_hypers(model):
    def show(layer, name):
        ls = layer.covar_module.base_kernel.lengthscale.detach().view(-1).cpu().numpy()
        os = layer.covar_module.outputscale.detach().view(-1).cpu().numpy()
        print(f"{name:12s} | lengthscale={ls.mean():.3g} (min={ls.min():.3g}, max={ls.max():.3g}) | outputscale={os.mean():.3g}")
    for (name, *_), layer in zip(model.hidden_specs, model.hidden_layers):
        show(layer, name)
    show(model.output_layer, "output")
    print(f"noise       | {model.likelihood.noise.item():.3g}")

@torch.no_grad()
def reinitialize_inducing_points(model: BoTorchDeepGP, train_X_internal: torch.Tensor):
    def source_for_index(i: int) -> torch.Tensor:
        """
        Return an [N, D_i] design for layer i (i=0 -> input).
        Ensures features are last-dim (N, D).
        """
        if i == 0:
            x = train_X_internal
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            return x  # [N, Din]

        x = train_X_internal  # [N, Din]
        with gpytorch.settings.num_likelihood_samples(1):
            for j, (name, out_dim, _mt) in enumerate(model.hidden_specs):
                if j >= i:
                    break
                mvn = getattr(model, name)(x)  # mean shape from DeepGP is [out_dim, N]
                m = mvn.mean
                if m.dim() == 2 and m.size(0) == out_dim:
                    m = m.transpose(-1, -2)  # -> [N, out_dim]
                elif m.dim() == 1:
                    m = m.unsqueeze(-1)      # -> [N, 1]
                x = m
        return x  # [N, D_i]

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
        layers = [getattr(model, name) for name, *_ in model.hidden_specs] + [model.output_layer]

        for i, layer in enumerate(layers):
            if not hasattr(layer, "variational_strategy"):
                continue

            vs = layer.variational_strategy
            target = vs.inducing_points            # shape (..., M, D)
            target_shape = target.shape
            M = target_shape[-2]
            D = target_shape[-1]
            is_batched = (target.dim() == 3)
            B = target_shape[0] if is_batched else 1

            Xsrc = source_for_index(i)             # [N, D?]
            if Xsrc.dim() == 1:
                Xsrc = Xsrc.unsqueeze(-1)
            # Coerce feature dim to D (transpose / slice / pad)
            if Xsrc.size(-1) != D:
                if Xsrc.size(0) == D and Xsrc.size(-1) != D:
                    Xsrc = Xsrc.transpose(-1, -2)
                if Xsrc.size(-1) != D:
                    if Xsrc.size(-1) > D:
                        Xsrc = Xsrc[..., :D]
                    else:
                        pad = D - Xsrc.size(-1)
                        Xsrc = torch.cat([Xsrc, Xsrc.new_zeros(Xsrc.size(0), pad)], dim=-1)

            N = Xsrc.size(0)
            if N >= M:
                idx = torch.randperm(N, device=Xsrc.device)[:M]
                Z = Xsrc[idx]                      # (M, D)
            else:
                idx = torch.randint(0, N, (M,), device=Xsrc.device)
                Z = Xsrc[idx]                      # (M, D)

            if is_batched:
                Z = Z.unsqueeze(0).expand(B, M, D).contiguous()  # (B, M, D)

            target.copy_(Z.to(dtype=target.dtype, device=target.device))

    tqdm.tqdm.write("🔁 Reinitialized inducing points.")

@torch.no_grad()
def make_main_figure(model: BoTorchDeepGP, ackley, Xtr, Ytr, Xte, Yte, num_points: int = 400):
    model.eval()
    D = Xtr.shape[-1]; device, dtype = Xtr.device, Xtr.dtype
    x_min, x_max = Xtr[:, 0].min().item(), Xtr[:, 0].max().item()
    xs = torch.linspace(x_min, x_max, num_points, dtype=dtype, device=device).unsqueeze(-1)

    post = model.posterior(xs, observation_noise=True, num_likelihood_samples=256)

    # Always use the MVN and its diagonal for std
    mvn = post.distribution            # == post.mvn
    mu = mvn.mean.view(-1)             # [num_points]
    std = mvn.covariance_matrix.diagonal(dim1=-1, dim2=-2).clamp_min(0).sqrt().view(-1)

    f_true = ackley(xs).cpu().numpy()
    xline = xs.squeeze(-1).cpu().numpy()
    mu_np = mu.cpu().numpy()
    lo = (mu - 1.96 * std).cpu().numpy()
    hi = (mu + 1.96 * std).cpu().numpy()
    tx = Xtr[:, 0].cpu().numpy(); ty = Ytr.squeeze(-1).cpu().numpy()
    vx = Xte[:, 0].cpu().numpy(); vy = Yte.squeeze(-1).cpu().numpy()

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
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="Objective")
    return fig

def run():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    ackley = Ackley(dim=1, negate=True).to(device)
    N = 32
    bounds = ackley.bounds.clone()
    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N, 1, dtype=torch.float64, device=device)
    Y = ackley(X).unsqueeze(-1)

    ntr = int(0.8 * N)
    Xtr, Xte = X[:ntr].contiguous(), X[ntr:].contiguous()
    Ytr, Yte = Y[:ntr].contiguous(), Y[ntr:].contiguous()

    # BoTorch transforms (you can swap for different ones)
    from botorch.models.transforms import Normalize, Standardize as OutcomeStandardize
    x_tf = Normalize(d=1, bounds=bounds)   # maps X to [0,1]
    y_tf = OutcomeStandardize(m=1)         # standardize Y using train stats, unstandardize on posterior

    inducing_src = x_tf(Xtr)

    M = min(ntr, 64)
    model = BoTorchDeepGP(
        input_dims=1,
        # hidden_specs=None,
        hidden_specs=[("h1", 3, "constant")],
        num_inducing=M,
        inducing_source=inducing_src,
        input_transform=x_tf,
        outcome_transform=y_tf,
    )

    base_lr = 5e-3
    epochs = 400
    model = train_botorch_deepgp(model, Xtr, Ytr, epochs=epochs, base_lr=base_lr, batch_size=min(ntr, 64))

    post_te = model.posterior(Xte, observation_noise=True, num_likelihood_samples=256)
    mu_te = post_te.mean
    rmse = torch.mean((mu_te - Yte.squeeze(-1))**2).sqrt().item()
    print(f"\n✅ RMSE (original scale): {rmse:.4f}")

    fig = make_main_figure(model, ackley, Xtr, Ytr, Xte, Yte, num_points=400)
    fig.show()

if __name__ == "__main__":
    run()
