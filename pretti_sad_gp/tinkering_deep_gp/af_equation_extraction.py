# export_singletaskgp_flexible.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math, json, torch
from gpytorch.kernels import RBFKernel, MaternKernel

# ----------------- helpers -----------------

def _get_noise(likelihood) -> float:
    if hasattr(likelihood, "noise_covar") and hasattr(likelihood.noise_covar, "noise"):
        return float(likelihood.noise_covar.noise.detach().mean())
    if hasattr(likelihood, "noise"):
        return float(likelihood.noise.detach().mean())
    raise RuntimeError("Cannot find Gaussian noise parameter on likelihood.")

def _read_outcome_standardize(model) -> Tuple[float,float]:
    y_mean, y_std = 0.0, 1.0
    ot = getattr(model, "outcome_transform", None)
    if ot is None:
        return y_mean, y_std
    # BoTorch changed attr names across versions
    for m_name, s_name in [("mean", "std"), ("means", "stds")]:
        if hasattr(ot, m_name) and hasattr(ot, s_name):
            y_mean = float(getattr(ot, m_name).detach().view(-1)[0])
            y_std  = float(getattr(ot, s_name).detach().view(-1)[0])
            return y_mean, y_std
    return y_mean, y_std

def _read_input_bounds(model) -> Optional[Dict[str,List[float]]]:
    it = getattr(model, "input_transform", None)
    if it is None or not hasattr(it, "bounds") or it.bounds is None:
        return None
    b = it.bounds.detach().double()
    return {"lower": b[0].tolist(), "upper": b[1].tolist()}

# ----------------- main export -----------------

def export_singletaskgp_flexible(model) -> Dict[str, Any]:
    """
    Works for SingleTaskGP with covar_module = ScaleKernel(RBFKernel|MaternKernel(ARD)),
    Gaussian likelihood, single output.
    Returns constants to build:
       μ(x)  = c + Σ_i β_i φ_i(x)
       σ²(x) = s_var - Σ_{i,j} G_ij φ_i(x) φ_j(x)
    in ORIGINAL output units. If an input Normalize was used, 'bounds' is included
    so expressions can inline raw→normalized x.
    """
    model.eval()
    if not hasattr(model, "likelihood") or model.likelihood is None:
        raise ValueError("Model must have a Gaussian likelihood (SingleTaskGP).")
    model.likelihood.eval()

    # Training data as seen by model
    X_raw = model.train_inputs[0].detach().double()
    y_t   = model.train_targets.detach().double()

    # Apply input transform (e.g., Normalize) to get kernel-space coords
    X = X_raw
    if getattr(model, "input_transform", None) is not None:
        model.input_transform.eval()
        X = model.input_transform(X_raw).detach().double()

    # Mean at training points (std-space)
    mX = model.mean_module(X).detach().double()

    # Covariance matrix on training inputs (includes outputscale)
    K = model.covar_module(X).evaluate().detach().double()

    # Noise (homoskedastic for SingleTaskGP)
    noise = _get_noise(model.likelihood)
    n = X.shape[0]
    I = torch.eye(n, dtype=torch.double, device=K.device)

    # α and M via Cholesky
    L = torch.linalg.cholesky(K + noise * I)
    alpha = torch.cholesky_solve((y_t - mX).unsqueeze(-1), L).squeeze(-1)  # [n]
    M = torch.cholesky_solve(I, L)  # (K + noise I)^(-1)  [n,n]

    # Kernel specifics
    scale_kernel = model.covar_module
    outputscale = float(scale_kernel.outputscale.detach())
    base = scale_kernel.base_kernel
    if isinstance(base, RBFKernel):
        k_type = "rbf"
        nu = None
    elif isinstance(base, MaternKernel):
        k_type = "matern"
        nu = float(base.nu)
        if nu not in (0.5, 1.5, 2.5):
            raise NotImplementedError(
                f"Matérn ν={nu} not supported for string export (needs Bessel K_ν). "
                "Use ν ∈ {0.5, 1.5, 2.5}."
            )
    else:
        raise ValueError(f"Unsupported base kernel {type(base).__name__}; only RBF or Matérn supported.")

    # ARD lengthscales as 1D list
    lengthscales = base.lengthscale.detach().view(-1).double().tolist()

    # Constant mean (std-space)
    if hasattr(model.mean_module, "constant"):
        c_std = float(model.mean_module.constant.detach())
    else:
        c_std = 0.0

    # Undo Standardize on outputs
    y_mean, y_std = _read_outcome_standardize(model)

    # Map constants to ORIGINAL output units
    # μ_orig = y_std * (c_std + s * α^T φ) + y_mean  =>  c = y_mean + y_std*c_std, β_i = y_std*s*α_i
    # σ²_orig = y_std² * (s - s² φ^T M φ) = s_var - Σ_ij G_ij φ_i φ_j,
    #   with s_var = y_std²*s, G = y_std²*s²*M
    c = y_mean + y_std * c_std
    betas = (y_std * outputscale * alpha).double().tolist()
    G = ( (y_std ** 2) * (outputscale ** 2) * M ).double().tolist()
    s_var = y_std ** 2 * outputscale

    export = {
        "kernel": k_type,               # "rbf" or "matern"
        "nu": nu,                       # Matérn only
        "X": X.tolist(),                # training inputs in kernel space
        "lengthscales": lengthscales,   # ARD ℓ_d
        "c": c,                         # constant in original units
        "betas": betas,                 # β_i
        "G": G,                         # for σ²(x) = s_var - Σ_ij G_ij φ_i φ_j
        "s_var": s_var,                 # leading scalar for variance
        "bounds": _read_input_bounds(model),  # raw→normalized if Normalize used
        "note": "Pass RAW x; expressions inline Normalize if bounds present.",
    }
    return export

# ----------------- rendering to strings -----------------

def _xn_expr(var_names: List[str], bounds: Optional[Dict[str, List[float]]]) -> List[str]:
    """Return strings for x_d in KERNEL SPACE. If bounds present, inline Normalize."""
    if bounds is None: return var_names
    xn = []
    for d, name in enumerate(var_names):
        lb, ub = bounds["lower"][d], bounds["upper"][d]
        xn.append(f"(({name}) - ({lb})) / ({ub} - {lb})")
    return xn

def _phi_str(kernel: str, xk: List[str], Xi: List[float], ls: List[float], nu: Optional[float]) -> str:
    # r^2 = Σ_d ((x_d - X_i,d)^2 / ℓ_d^2); r = sqrt(r^2)
    parts = [f"(({xk[d]}) - ({Xi[d]}))^2 / ({ls[d]}^2)" for d in range(len(ls))]
    rsq = " + ".join(parts) if parts else "0"
    if kernel == "rbf":
        return f"exp(-0.5*({rsq}))"
    elif kernel == "matern":
        r = f"sqrt({rsq})"
        if abs(nu - 0.5) < 1e-9:
            return f"exp(-({r}))"
        elif abs(nu - 1.5) < 1e-9:
            c = f"{math.sqrt(3):.15g}"
            return f"(1 + {c}*{r})*exp(-{c}*{r})"
        elif abs(nu - 2.5) < 1e-9:
            c = f"{math.sqrt(5):.15g}"
            return f"(1 + {c}*{r} + {(5.0/3.0):.15g}*({r})^2)*exp(-{c}*{r})"
        else:
            raise NotImplementedError("Only ν ∈ {0.5,1.5,2.5} supported.")
    else:
        raise ValueError(kernel)

def render_mu_expression(exp: Dict[str, Any], var_names: Optional[List[str]] = None) -> str:
    D = len(exp["lengthscales"])
    if var_names is None:
        var_names = [f"x{d+1}" for d in range(D)]
    xk = _xn_expr(var_names, exp.get("bounds"))
    terms = [f"{exp['c']}"]
    for i, beta in enumerate(exp["betas"]):
        phi = _phi_str(exp["kernel"], xk, exp["X"][i], exp["lengthscales"], exp.get("nu"))
        terms.append(f"({beta})*({phi})")
    return " + ".join(terms) if terms else "0"

def render_sigma2_expression(exp: Dict[str, Any], var_names: Optional[List[str]] = None, eps: float = 0.0) -> str:
    D = len(exp["lengthscales"])
    if var_names is None:
        var_names = [f"x{d+1}" for d in range(D)]
    xk = _xn_expr(var_names, exp.get("bounds"))
    # Precompute φ_i strings
    phi = [ _phi_str(exp["kernel"], xk, exp["X"][i], exp["lengthscales"], exp.get("nu")) for i in range(len(exp["X"])) ]
    # Σ_ij G_ij φ_i φ_j
    quad_terms = []
    G = exp["G"]
    n = len(G)
    for i in range(n):
        for j in range(n):
            gij = G[i][j]
            if gij != 0.0:
                quad_terms.append(f"({gij})*({phi[i]})*({phi[j]})")
    quad = " + ".join(quad_terms) if quad_terms else "0"
    s_expr = f"({exp['s_var']}) - ({quad})"
    return f"max({s_expr}, {eps})" if eps and eps > 0 else s_expr

def render_logei_expression(
    exp: Dict[str, Any],
    f_best: float,
    xi: float = 0.0,
    var_names: Optional[List[str]] = None,
    eps_sigma: float = 1e-12,
) -> str:
    """logEI(x) with Φ via erf (solver-friendly)."""
    mu = render_mu_expression(exp, var_names)
    sigma2 = render_sigma2_expression(exp, var_names, eps=eps_sigma**2)
    sigma = f"sqrt({sigma2})"
    a = f"({mu}) - ({f_best}) - ({xi})"
    z = f"({a})/({sigma})"
    inv_sqrt_2pi = 0.3989422804014327
    phi = f"{inv_sqrt_2pi}*exp(-0.5*({z})*({z}))"
    Phi = f"0.5*(1 + erf(({z})/sqrt(2)))"
    inner = f"({z})*({Phi}) + ({phi})"
    return f"log(max({sigma}, {eps_sigma})) + log({inner})"

# ----------------- optional Python callables & IO -----------------

def make_mu_var_callables(exp: Dict[str, Any]):
    """Return Python functions mu(x_raw) and var(x_raw) for quick checks."""
    X = exp["X"]; ls = exp["lengthscales"]; kernel = exp["kernel"]; nu = exp.get("nu")
    c = exp["c"]; betas = exp["betas"]; G = exp["G"]; s_var = exp["s_var"]
    bounds = exp.get("bounds", None)
    D = len(ls)

    def _to_kernel_space(x):
        if bounds is None: return x
        return [(x[d] - bounds["lower"][d]) / (bounds["upper"][d] - bounds["lower"][d]) for d in range(D)]

    def _phi_vec(x):
        xn = _to_kernel_space(x)
        out = []
        for i in range(len(X)):
            acc = 0.0
            for d in range(D):
                acc += ((xn[d] - X[i][d])**2) / (ls[d]**2)
            if kernel == "rbf":
                out.append(math.exp(-0.5 * acc))
            else:  # matern
                r = math.sqrt(acc)
                if abs(nu - 0.5) < 1e-9:
                    out.append(math.exp(-r))
                elif abs(nu - 1.5) < 1e-9:
                    out.append((1 + math.sqrt(3.0)*r) * math.exp(-math.sqrt(3.0)*r))
                elif abs(nu - 2.5) < 1e-9:
                    out.append((1 + math.sqrt(5.0)*r + (5.0/3.0)*r*r) * math.exp(-math.sqrt(5.0)*r))
                else:
                    raise NotImplementedError
        return out

    def mu(x: List[float]) -> float:
        phi = _phi_vec(x)
        return c + sum(betas[i] * phi[i] for i in range(len(phi)))

    def var(x: List[float]) -> float:
        phi = _phi_vec(x)
        # s_var - φ^T G φ
        qt = 0.0
        n = len(phi)
        for i in range(n):
            gi = G[i]
            for j in range(n):
                qt += gi[j] * phi[i] * phi[j]
        return max(s_var - qt, 0.0)

    return mu, var

def save_export(path: str, exp: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)

# ----------------- usage example -----------------
if __name__ == "__main__":
    """
    After training your SingleTaskGP `model` (RBF or Matérn ν∈{1/2,3/2,5/2}):

        from botorch.models import SingleTaskGP
        # model = ...  # your trained model

        exp = export_singletaskgp_flexible(model)

        # Solver-ready strings (variables default to x1..xD, RAW scale):
        mu_str   = render_mu_expression(exp)
        sig2_str = render_sigma2_expression(exp, eps=1e-16)  # tiny clamp recommended
        logei_str = render_logei_expression(exp, f_best=1.234, xi=0.0)

        print("mu(x)    =", mu_str)
        print("sigma2(x)=", sig2_str)
        print("logEI(x) =", logei_str)

        # Optional: quick numerical sanity checks in Python
        mu_fun, var_fun = make_mu_var_callables(exp)
        print("μ([0.1,0.2]) =", mu_fun([0.1, 0.2]))
        print("σ²([0.1,0.2]) =", var_fun([0.1, 0.2]))

        save_export("gp_export.json", exp)
    """
    pass
