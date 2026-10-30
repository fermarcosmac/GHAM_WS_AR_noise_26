"""Bayesian identification of the original parametric Wiener--AR model.

This module deliberately has no optimizer-iteration/result-history interface.
``run_bi_hmc`` runs a separate, fixed-budget HMC experiment and returns posterior
draws, a physical-coordinate posterior mean, diagnostics, and total elapsed time.
No true coefficient, true innovation, or simulated-noise standard deviation is
accepted by the inference API.

The likelihood uses an exact causal IIR recursion.  SciPy's C implementation is
wrapped in a first-derivative PyTorch autograd function; its finite-record adjoint
is exact, rather than an FFT/circular-filter approximation.  This CPU/float64
implementation is appropriate for the small parameter vectors in these studies.
It supports known observed input prehistory and an exact finite-burn-in AR noise
covariance, as well as stationary, known-zero, and conditional noise likelihoods.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.special import ndtri
from scipy.stats import rankdata
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from wiener_system import WienerDimensions, polynomial_nonlinearity, split_theta


class _ExactLFilter(torch.autograd.Function):
    """Zero-state causal filter, with numerator including its zero-lag tap.

    For A(q)y=B(q)x and output adjoint g, lambda=A(q)^(-T)g.  The
    denominator derivative is -sum(lambda[t]*y[t-i]), the numerator derivative
    sum(lambda[t]*x[t-j]), and the input derivative B(q)^T lambda.  Reversing
    the record implements both transpose filters with the correct boundaries.
    Higher derivatives are intentionally unsupported; ordinary HMC needs only
    first derivatives of its log density.
    """

    @staticmethod
    def forward(ctx, x, denominator, numerator):
        if x.device.type != "cpu":
            raise ValueError("Exact HMC filtering operates on CPU")
        xn = np.asarray(x.detach().numpy(), dtype=np.float64)
        an = np.r_[1.0, denominator.detach().numpy()]
        bn = np.asarray(numerator.detach().numpy(), dtype=np.float64)
        yn = lfilter(bn, an, xn)
        result = torch.as_tensor(yn.copy(), dtype=x.dtype)
        ctx.save_for_backward(x, denominator, numerator, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, denominator, numerator, y = ctx.saved_tensors
        xn, yn = x.detach().numpy(), y.detach().numpy()
        an = np.r_[1.0, denominator.detach().numpy()]
        bn = numerator.detach().numpy()
        adjoint = lfilter([1.0], an, grad_output.detach().numpy()[::-1])[::-1].copy()
        dx = lfilter(bn, [1.0], adjoint[::-1])[::-1].copy()
        da = np.asarray([-np.dot(adjoint[i:], yn[:-i]) if i < len(yn) else 0.0
                         for i in range(1, len(an))])
        db = np.asarray([np.dot(adjoint[j:], xn[:len(xn)-j]) if j < len(xn) else 0.0
                         for j in range(len(bn))])
        return (torch.as_tensor(dx, dtype=x.dtype),
                torch.as_tensor(da, dtype=denominator.dtype),
                torch.as_tensor(db, dtype=numerator.dtype))


def exact_lfilter(x: torch.Tensor, denominator: torch.Tensor,
                  numerator: torch.Tensor) -> torch.Tensor:
    """Differentiable causal IIR filtering with a monic denominator."""
    return _ExactLFilter.apply(x, denominator, numerator)


def reflection_to_coefficients(coordinates: torch.Tensor) -> torch.Tensor:
    """Real coordinates -> tanh reflection coefficients -> stable polynomial."""
    previous = coordinates.new_empty(0)
    for reflection in torch.tanh(coordinates).unbind():
        previous = torch.cat((previous + reflection * previous.flip(0), reflection.reshape(1)))
    return previous


def coefficients_to_coordinates(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse of the real Schur step-up map; rejects unstable initial values."""
    current = coefficients.detach().clone()
    reflection = torch.empty_like(current)
    for order in range(current.numel(), 0, -1):
        kappa = current[-1]
        if not torch.isfinite(kappa) or abs(float(kappa)) >= 1.0:
            raise ValueError("BI_HMC initialization must have strictly stable denominators")
        reflection[order-1] = kappa
        current = (current[:-1] - kappa * current[:-1].flip(0)) / (1-kappa.square())
    if not torch.all(torch.isfinite(reflection)):
        raise ValueError("Could not invert the initial stable denominator")
    return torch.atanh(reflection)


def denominator_log_jacobian(coordinates: torch.Tensor) -> torch.Tensor:
    """Log |d physical polynomial / d real coordinates|, all orders incl. zero."""
    # Stable identities avoid subtracting tanh(z)^2 from one near a boundary.
    log_plus = math.log(2.0) - F.softplus(-2.0 * coordinates)
    log_minus = math.log(2.0) - F.softplus(2.0 * coordinates)
    answer = (log_plus + log_minus).sum()
    for order in range(2, coordinates.numel()+1):
        answer = answer + (order//2)*log_plus[order-1] + ((order-1)//2)*log_minus[order-1]
    return answer


DEFAULT_HMC_CONFIG = {
    "chains": 4, "warmup": 500, "samples": 1000, "leapfrog_steps": 10,
    "step_size": 0.01, "target_accept": 0.8, "adapt_step_size": True,
    "adapt_mass": True, "mass_window": 50, "initial_jitter": 0.1,
    "initial_optimization_steps": 30, "initial_variance": 0.01,
    "variance_prior_shape": 3.0, "variance_prior_scale": 0.002,
    "coefficient_prior_mean": 0.0, "coefficient_prior_std": 1.0,
    "seed": 20261030, "progress_every": 100, "divergence_threshold": 1000.0,
    "max_rhat": 1.01, "min_ess": 100.0, "noise_initialization": "stationary",
    "min_step_size": 1e-7, "max_step_size": 1.0, "cpu_threads": 1,
    "metric_regularization_weight": 5.0, "metric_regularization_variance": 1e-6,
    "metric_min_variance": 1e-10, "metric_max_variance": 1e3,
}


def resolve_hmc_config(method_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve and validate the sampler settings independently of K_max."""
    cfg = copy.deepcopy(DEFAULT_HMC_CONFIG)
    supplied = dict(method_cfg or {})
    supplied.update(supplied.pop("hmc", {}))
    cfg.update(supplied)
    for key in ("chains", "samples", "leapfrog_steps", "mass_window", "progress_every"):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"BI_HMC {key} must be a positive integer")
        cfg[key] = int(value)
    for key in ("warmup", "initial_optimization_steps", "cpu_threads"):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"BI_HMC {key} must be a nonnegative integer")
        cfg[key] = int(value)
    for key in ("step_size", "initial_variance", "variance_prior_shape", "variance_prior_scale",
                "divergence_threshold", "min_step_size", "max_step_size", "min_ess",
                "metric_regularization_weight", "metric_regularization_variance",
                "metric_min_variance", "metric_max_variance"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"BI_HMC {key} must be finite and positive")
        cfg[key] = float(cfg[key])
    if not 0 < float(cfg["target_accept"]) < 1:
        raise ValueError("BI_HMC target_accept must lie strictly between zero and one")
    if not 0 <= float(cfg["initial_jitter"]) or not math.isfinite(float(cfg["initial_jitter"])):
        raise ValueError("BI_HMC initial_jitter must be finite and nonnegative")
    if not 1 <= float(cfg["max_rhat"]):
        raise ValueError("BI_HMC max_rhat must be at least one")
    if cfg["min_step_size"] > cfg["max_step_size"]:
        raise ValueError("BI_HMC min_step_size exceeds max_step_size")
    if cfg["metric_min_variance"] > cfg["metric_max_variance"]:
        raise ValueError("BI_HMC metric_min_variance exceeds metric_max_variance")
    if cfg["noise_initialization"] not in {"stationary", "zero", "conditional", "finite_burn_in"}:
        raise ValueError("Unknown BI_HMC noise_initialization")
    cfg["seed"] = int(cfg["seed"])
    return cfg


def _coefficient_vector(value, count, name):
    vector = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if vector.numel() == 1:
        vector = vector.repeat(count)
    if vector.numel() != count or not torch.all(torch.isfinite(vector)):
        raise ValueError(f"{name} must be a finite scalar or a vector of length {count}")
    return vector


class BayesianWienerARModel:
    """A differentiable physical prior and exact finite-record Gaussian model."""

    def __init__(self, r, c, dims, config, *, input_history=None, noise_burn_in=None):
        self.dims, self.config = dims, config
        if min(dims.na, dims.nb, dims.nd) < 0 or dims.nf < 1 or dims.nb < 1:
            raise ValueError("BI_HMC requires na,nd>=0, nb>=1 and nf>=1")
        self.r = torch.as_tensor(r).detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
        self.c = torch.as_tensor(c).detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
        if self.r.numel() != self.c.numel() or not self.r.numel():
            raise ValueError("BI_HMC input and output must be nonempty, equal-length vectors")
        if not torch.isfinite(self.r).all() or not torch.isfinite(self.c).all():
            raise ValueError("BI_HMC observations must be finite")
        history = self.r.new_empty(0) if input_history is None else torch.as_tensor(input_history).detach().to(
            device="cpu", dtype=torch.float64).reshape(-1).clone()
        if not torch.isfinite(history).all():
            raise ValueError("Observed input prehistory must be finite")
        self.input_count = history.numel()
        self.full_input = torch.cat((history, self.r))
        self.prior_mean = _coefficient_vector(config["coefficient_prior_mean"], dims.n_params, "coefficient_prior_mean")
        self.prior_std = _coefficient_vector(config["coefficient_prior_std"], dims.n_params, "coefficient_prior_std")
        start = 0
        for block, count in (("a", dims.na), ("b", dims.nb), ("f", dims.nf-1), ("d", dims.nd)):
            if f"prior_{block}_std" in config:
                self.prior_std[start:start+count] = _coefficient_vector(config[f"prior_{block}_std"], count, f"prior_{block}_std")
            start += count
        if not torch.all(self.prior_std > 0):
            raise ValueError("BI_HMC coefficient prior standard deviations must be positive")
        self.noise_policy = config["noise_initialization"]
        if noise_burn_in is not None:
            if isinstance(noise_burn_in, bool) or int(noise_burn_in) != noise_burn_in or noise_burn_in < 0:
                raise ValueError("noise_burn_in must be a nonnegative integer")
            self.noise_policy = "finite_burn_in"
            self.noise_burn_in = int(noise_burn_in)
        else:
            self.noise_burn_in = int(config.get("noise_burn_in", 0))
        if self.noise_burn_in < 0:
            raise ValueError("noise_burn_in must be nonnegative")
        self.initial_count = min(dims.nd, self.r.numel())
        if self.noise_policy == "conditional" and self.r.numel() <= dims.nd:
            raise ValueError("Conditional AR likelihood needs more observations than noise coefficients")
        p = dims.nd
        self.yw_masks = torch.zeros((p+1, p+1, p), dtype=torch.float64)
        for k in range(p+1):
            for i in range(1, p+1):
                self.yw_masks[k, abs(k-i), i-1] = 1.0
        index = torch.arange(self.initial_count)
        self.initial_lags = (index[:, None] - index[None, :]).abs()
        self.evaluation_count = 0

    def transform(self, position):
        na, nb, nf = self.dims.na, self.dims.nb, self.dims.nf
        za = position[:na]
        b = position[na:na+nb]
        f = position[na+nb:na+nb+nf-1]
        zd = position[na+nb+nf-1:-1]
        theta = torch.cat((reflection_to_coefficients(za), b, f, reflection_to_coefficients(zd)))
        return theta, torch.exp(position[-1]), denominator_log_jacobian(za)+denominator_log_jacobian(zd)+position[-1]

    def inverse(self, theta, variance):
        a, b, f, d = split_theta(theta, self.dims)
        if not torch.isfinite(theta).all() or not math.isfinite(float(variance)) or float(variance) <= 0:
            raise ValueError("BI_HMC initial coefficients must be finite and variance positive")
        return torch.cat((coefficients_to_coordinates(a), b, f,
                          coefficients_to_coordinates(d), theta.new_tensor([math.log(float(variance))])))

    def signal(self, theta):
        a, b, f, _ = split_theta(theta, self.dims)
        numerator = torch.cat((theta.new_zeros(1), b))
        alpha_full = exact_lfilter(self.full_input, a, numerator)
        alpha = alpha_full[self.input_count:]
        return alpha, polynomial_nonlinearity(alpha, f)

    def initial_noise_covariance(self, d):
        count = self.initial_count
        if self.noise_policy == "finite_burn_in":
            horizon = self.noise_burn_in + count
            impulse = d.new_zeros(horizon)
            impulse[0] = 1.0
            response = exact_lfilter(impulse, d, d.new_ones(1))
            times = self.noise_burn_in + torch.arange(count)
            lags = times[:, None] - torch.arange(horizon)[None, :]
            response_matrix = response[lags.clamp(min=0)] * (lags >= 0)
            return response_matrix @ response_matrix.T
        p = d.numel()
        system = torch.eye(p+1, dtype=d.dtype) + torch.einsum("kji,i->kj", self.yw_masks, d)
        rhs = d.new_zeros(p+1)
        rhs[0] = 1.0
        gamma = torch.linalg.solve(system, rhs)
        return gamma[self.initial_lags]

    def likelihood_terms(self, theta):
        """Return Q, log|unit covariance| and number of Gaussian contributions."""
        _, beta = self.signal(theta)
        e = self.c-beta
        d = split_theta(theta, self.dims)[3]
        residual = e
        for lag, coefficient in enumerate(d.unbind(), 1):
            if lag < e.numel():
                residual = residual + coefficient * torch.cat((e.new_zeros(lag), e[:-lag]))
        if not d.numel() or self.noise_policy == "zero" or (
                self.noise_policy == "finite_burn_in" and self.noise_burn_in == 0):
            return residual.square().sum(), e.new_zeros(()), e.numel()
        count = self.initial_count
        if self.noise_policy == "conditional":
            return residual[count:].square().sum(), e.new_zeros(()), e.numel()-count
        covariance = self.initial_noise_covariance(d)
        factor = torch.linalg.cholesky(covariance)
        initial = torch.linalg.solve_triangular(factor, e[:count, None], upper=False).reshape(-1)
        q = initial.square().sum()+residual[count:].square().sum()
        return q, 2*torch.log(torch.diag(factor)).sum(), e.numel()

    def log_prob(self, position, *, include_jacobian=True):
        self.evaluation_count += 1
        theta, variance, log_jacobian = self.transform(position)
        # In real arithmetic tanh maps every finite coordinate to the open
        # interval. Floating point can round extreme coordinates to +/-1;
        # reject that numerical boundary, and any unrepresentable variance,
        # rather than clipping and silently changing the density.
        a_coordinates = position[:self.dims.na]
        d_coordinates = position[self.dims.na+self.dims.nb+self.dims.nf-1:-1]
        if (not torch.isfinite(theta).all() or not torch.isfinite(variance) or variance <= 0
                or torch.any(torch.abs(torch.tanh(a_coordinates)) >= 1)
                or torch.any(torch.abs(torch.tanh(d_coordinates)) >= 1)):
            return position.sum()*0-float("inf")
        q, logdet, count = self.likelihood_terms(theta)
        xi, zeta = self.config["variance_prior_shape"], self.config["variance_prior_scale"]
        prior_quadratic = ((theta-self.prior_mean)/self.prior_std).square().sum()
        value = -(count/2+xi+1)*position[-1]-(q/2+zeta)/variance-logdet/2-prior_quadratic/2
        return value+log_jacobian if include_jacobian else value


def _value_gradient(model, position):
    # get model output and gradient at given candidate parameters (position in phase space, HMC interpretation)
    variable = position.detach().requires_grad_(True)
    try:
        value = model.log_prob(variable) # here is where the main logic happens!
        if not torch.isfinite(value):
            return float("-inf"), torch.zeros_like(position)
        gradient, = torch.autograd.grad(value, variable)
        if not torch.isfinite(gradient).all():
            return float("-inf"), torch.zeros_like(position)
        return float(value.detach()), gradient.detach()
    except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return float("-inf"), torch.zeros_like(position)


def _transition(model, position, value, gradient, inverse_mass, step, leapfrog_steps, generator, threshold):
    momentum = torch.randn(position.shape, generator=generator, dtype=position.dtype)/torch.sqrt(inverse_mass)
    energy = -value+0.5*float((momentum.square()*inverse_mass).sum())
    proposal = position.clone()
    momentum = momentum+0.5*step*gradient
    candidate_value, candidate_gradient = value, gradient
    for index in range(leapfrog_steps):
        proposal = proposal+step*inverse_mass*momentum
        candidate_value, candidate_gradient = _value_gradient(model, proposal)
        if not math.isfinite(candidate_value):
            return position, value, gradient, 0.0, False, True, float("inf")
        momentum = momentum+(0.5 if index == leapfrog_steps-1 else 1.0)*step*candidate_gradient
    delta = -candidate_value+0.5*float((momentum.square()*inverse_mass).sum())-energy
    probability = math.exp(min(0.0, -delta)) if math.isfinite(delta) else 0.0
    accepted = float(torch.rand((), generator=generator)) < probability
    divergent = not math.isfinite(delta) or abs(delta) > threshold
    if accepted:
        return proposal.detach(), candidate_value, candidate_gradient, probability, True, divergent, delta
    return position, value, gradient, probability, False, divergent, delta


def _reasonable_step(model, position, value, gradient, inverse_mass, configured_step, cfg, generator):
    step = min(cfg["max_step_size"], max(cfg["min_step_size"], configured_step))
    target = float(cfg["target_accept"])
    for _ in range(12):
        result = _transition(model, position, value, gradient, inverse_mass, step, 1,
                             generator, cfg["divergence_threshold"])
        probability = result[3]
        direction = 1 if probability > target else -1
        if _ == 0:
            initial_direction = direction
        elif direction != initial_direction:
            break
        next_step = min(cfg["max_step_size"], max(cfg["min_step_size"], step*(2 if direction > 0 else 0.5)))
        if next_step == step:
            break
        step = next_step
    return step


def _map_initialize(model, initial, steps):
    if not steps:
        return initial.detach(), {"evaluations": 0, "used": False}
    variable = initial.detach().clone().requires_grad_(True)
    best_position, best_value = initial.detach().clone(), float("-inf")
    evaluations = 0
    optimizer = torch.optim.LBFGS([variable], max_iter=steps, max_eval=steps*5,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-6)
    def closure():
        nonlocal best_position, best_value, evaluations
        optimizer.zero_grad()
        evaluations += 1
        try:
            value = model.log_prob(variable, include_jacobian=False)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite initial optimization proposal")
            if float(value.detach()) > best_value:
                best_value, best_position = float(value.detach()), variable.detach().clone()
            loss = -value
            loss.backward()
            if variable.grad is None or not torch.isfinite(variable.grad).all():
                raise ValueError("Nonfinite initial optimization gradient")
            return loss
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            # This finite fallback is only for the optional initialization
            # optimizer; it is never used in the HMC target or transitions.
            loss = 1e100+variable.square().sum()
            loss.backward()
            return loss
    try:
        optimizer.step(closure)
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    return best_position, {"evaluations": evaluations, "used": math.isfinite(best_value)}


def _split(chains):
    half = chains.shape[1]//2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _rhat(chains):
    m, n = chains.shape
    within = np.var(chains, axis=1, ddof=1).mean()
    between = n*np.var(chains.mean(axis=1), ddof=1)
    if within == 0:
        # An exactly constant continuous-parameter chain contains no evidence
        # of exploration. It must not be reported as perfect convergence.
        return float("nan") if between == 0 else float("inf")
    return float(np.sqrt(((n-1)*within/n+between/n)/within))


def _ess(chains):
    m, n = chains.shape
    if n < 4:
        return float("nan")
    centered = chains-chains.mean(axis=1, keepdims=True)
    nfft = 1 << (2*n-1).bit_length()
    transform = np.fft.rfft(centered, n=nfft, axis=1)
    covariance = np.fft.irfft(transform*np.conjugate(transform), n=nfft, axis=1)[:, :n]/n
    within = np.var(chains, axis=1, ddof=1).mean()
    var_plus = (n-1)*within/n + (np.var(chains.mean(axis=1), ddof=1) if m > 1 else 0.0)
    if not var_plus > 0:
        return float("nan")
    rho = 1-(within-covariance.mean(axis=0))/var_plus
    rho[0] = 1
    total, last = 0.0, float("inf")
    for index in range(0, n-1, 2):
        pair = rho[index]+rho[index+1]
        if pair <= 0:
            break
        pair = min(pair, last)
        total += pair
        last = pair
    tau = max(1.0, -1+2*total)
    return float(min(m*n, m*n/tau))


def posterior_diagnostics(samples):
    """Rank-normalized folded split Rhat, bulk/tail ESS and mean MCSE.

    These are screening diagnostics, not a proof that all posterior modes were
    found. Very short or single-chain runs intentionally fail Rhat screening.
    """
    values = np.asarray(samples, dtype=float)
    chains, draws, parameters = values.shape
    rhat = np.full(parameters, np.nan)
    bulk, tail, mean_ess = (np.full(parameters, np.nan) for _ in range(3))
    if chains >= 2 and draws >= 8:
        for parameter in range(parameters):
            raw = _split(values[:, :, parameter])
            count = raw.size
            ranked = ndtri((rankdata(raw.ravel()).reshape(raw.shape)-0.375)/(count+0.25))
            folded = np.abs(raw-np.median(raw))
            folded = ndtri((rankdata(folded.ravel()).reshape(raw.shape)-0.375)/(count+0.25))
            rhat[parameter] = np.maximum(_rhat(ranked), _rhat(folded))
            bulk[parameter] = _ess(ranked)
            lo, hi = np.quantile(raw, [0.05, 0.95])
            tail[parameter] = np.minimum(_ess((raw <= lo).astype(float)), _ess((raw >= hi).astype(float)))
            mean_ess[parameter] = _ess(raw)
    elif draws >= 8:
        for parameter in range(parameters):
            raw = values[:, :, parameter]
            mean_ess[parameter] = _ess(raw)
    std = np.std(values.reshape(-1, parameters), axis=0, ddof=1)
    return {"rhat": rhat, "bulk_ess": bulk, "tail_ess": tail,
            "mean_ess": mean_ess, "posterior_std": std,
            "mean_mcse": std/np.sqrt(mean_ess)}


def _stable(theta, dims):
    a, _, _, d = split_theta(theta, dims)
    return all(not block.numel() or np.all(np.abs(np.roots(np.r_[1., block.detach().numpy()])) < 1)
               for block in (a, d))


def _run_bi_hmc(r: torch.Tensor, c: torch.Tensor, dims: WienerDimensions,
               method_cfg: dict[str, Any] | None = None, *, initial_theta=None,
               input_history=None, noise_burn_in: int | None = None, seed: int | None = None,
               progress_callback: Callable[[dict[str, Any]], None] | None = None,
               verbose: bool = False) -> dict[str, Any]:
    """Run independent HMC chains and return their physical posterior mean.

    Parameters
    ----------
    r : torch.Tensor, shape (N,)
        Known external input/excitation over the retained observation record:
        this is u_t in the manuscript, NOT an output residual or innovation.
        Sample r[t] is aligned with observed output c[t]. Earlier observed
        input samples, if available, belong in ``input_history`` instead.
    c : torch.Tensor, shape (N,)
        Measured noisy output: c_t = beta_t(theta) + e_t. Must have the same
        length as r. Both records must be finite; inference copies them to
        CPU float64 and does not modify the supplied tensors.
    dims : WienerDimensions
        Model orders na, nb, nf, nd. There are P = na + nb + (nf - 1) + nd
        free physical coefficients, ordered as (a1,...,a_na, b1,...,b_nb,
        f2,...,f_nf, d1,...,d_nd); f1 is fixed to one. The innovation variance
        is a separate additional unknown. Orders are not inferred here.
    method_cfg : dict or None
        Sampler/prior settings merged with DEFAULT_HMC_CONFIG. In particular,
        chains=C, warmup=W, samples=S, and leapfrog_steps control the sampling
        budget; Gaussian coefficient and inverse-gamma variance prior settings
        define the physical prior. Optimizer K_max is not used.
    initial_theta : array-like of shape (P,), optional, keyword-only
        Starting PHYSICAL coefficient vector in the order above. Denominators
        must be strictly stable. If omitted, use zero coefficients except
        b1=1. Each chain jitters the corresponding unconstrained coordinates
        and may run the configured initialization optimization. The starting
        variance comes from method_cfg['initial_variance'].
    input_history : array-like of shape (H,), optional, keyword-only
        Observed input samples immediately preceding r, in chronological
        order. Each candidate signal filter is replayed over history followed
        by r, starting from a known zero signal state before the history.
        Without history, the assumed signal state is zero at r's first sample.
        No true internal signal state is supplied.
    noise_burn_in : nonnegative int or None, optional, keyword-only
        Number of discarded noise samples between a known zero AR-noise start
        and c's first retained sample. If supplied, overrides the configured
        noise policy with the exact finite-burn-in initial covariance. Only
        the count is used, never the discarded innovations or noise values.
        If None, use the configured noise_initialization policy.
    seed : int or None, optional, keyword-only
        Override the configured base sampler seed. Each chain derives its own
        local generator from this seed; the global PyTorch RNG is unchanged.
    progress_callback : callable(dict) or None, optional, keyword-only
        Receives sampling heartbeats containing phase, sampling_phase
        ('warmup' or 'sampling'), chain, transition, max_transitions,
        chain_transition, step_size, acceptance_probability, elapsed_seconds.
        Chain numbers and transition counts are one-based/completed counts.
    verbose : bool, optional, keyword-only
        Print heartbeats when no progress_callback is supplied.

    Returns
    -------
    dict
        All draws below exclude warm-up. C denotes chains, S retained draws
        per chain, and P the physical coefficient count.

        theta_hat : torch.Tensor, shape (P,)
            Mean of the physical coefficient draws, returned on r's original
            device/dtype. It is NOT the transformed mean of sampling positions.
        posterior_samples : CPU float64 tensor, shape (C, S, P)
            Retained original-coefficient draws, preserving chain identity.
        variance_samples : CPU float64 tensor, shape (C, S)
            Retained innovation VARIANCE draws (sigma_nu squared, not its SD).
        variance_hat : float
            Mean of variance_samples.
        posterior_interval : CPU float64 tensor, shape (2, P)
            Physical marginal 2.5% and 97.5% posterior quantiles.
        posterior_log_density : CPU float64 tensor, shape (C, S)
            Physical-coordinate log posterior at each draw, up to a constant;
            the sampling-coordinate Jacobian has been removed here.
        sampling_positions : CPU float64 tensor, shape (C, S, P + 1)
            Retained unconstrained coordinates (z_a, b, f, z_d, log variance).
            These are sampler draws, not optimizer parameter trajectories.
        final_alpha, final_beta : tensors, shape (N,)
            Internal signal and noise-free output evaluated at theta_hat,
            including input history; returned on r's original device/dtype.
        status, diagnostics_passed, point_estimate_valid : str, bool, bool
            Status is 'sampling_complete', 'sampling_warning', or 'failed'.
            Diagnostic checks and a finite/stable physical mean determine the
            flags; a finite posterior does not automatically pass diagnostics.
        sampler_diagnostics : dict
            Rhat, ESS, MCSE, acceptance/divergence information, per-chain
            adapted settings/seeds, warnings, and evaluation counts.
        resolved_config, sampler_seed, point_estimator : dict, int, str
            Actual settings/base seed and the label 'posterior_mean'.
        sampling_transitions, retained_draws : int, int
            Total counts C*(W+S) and C*S, respectively.
        forward_model, noise_initialization, noise_burn_in,
        input_history_length, signal_initialization
            Explicit model and beginning-of-record conventions used in the fit.
        total_time : float
            Elapsed seconds including model setup, initialization optimization,
            warm-up, sampling and posterior diagnostics. The public run_bi_hmc
            wrapper additionally includes thread-setting setup/restoration.

    True parameters and true innovations are deliberately not inference inputs;
    the comparison wrapper uses them only afterwards to assess the estimate.
    """
    # Initialization and configuration
    started = time.perf_counter()
    cfg = resolve_hmc_config(method_cfg)
    if seed is not None:
        cfg["seed"] = int(seed)
    model = BayesianWienerARModel(r, c, dims, cfg, input_history=input_history, noise_burn_in=noise_burn_in)
    output_device, output_dtype = r.device, r.dtype
    if initial_theta is None:
        initial_theta = torch.zeros(dims.n_params, dtype=torch.float64)
        initial_theta[dims.na] = 1.0
    theta0 = torch.as_tensor(initial_theta).detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
    if theta0.numel() != dims.n_params:
        raise ValueError(f"BI_HMC initial_theta must contain {dims.n_params} values")
    center = model.inverse(theta0, cfg["initial_variance"])
    count, warmup, retained = cfg["chains"], cfg["warmup"], cfg["samples"]
    dimension = center.numel()
    posterior = torch.empty((count, retained, dims.n_params), dtype=torch.float64)
    variances = torch.empty((count, retained), dtype=torch.float64)
    log_posterior = torch.empty((count, retained), dtype=torch.float64)
    positions = torch.empty((count, retained, dimension), dtype=torch.float64)
    accepted = np.zeros((count, retained), dtype=bool)
    probabilities = np.zeros((count, retained))
    divergences = np.zeros((count, retained), dtype=bool)
    energy_errors = np.zeros((count, retained))
    chain_info = []
    completed = 0
    maximum = count*(warmup+retained)

    # Main loop (one per sampling Markov Chain)
    for chain in range(count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((cfg["seed"] + chain*1000003) % (2**63-1))
        position = center+float(cfg["initial_jitter"])*torch.randn(center.shape, generator=generator, dtype=center.dtype) # slightly different starting point for each chain
        position, map_info = _map_initialize(model, position, cfg["initial_optimization_steps"])
        value, gradient = _value_gradient(model, position)
        if not math.isfinite(value):
            position = center.clone()
            value, gradient = _value_gradient(model, position)
        if not math.isfinite(value):
            raise ValueError("BI_HMC initial log posterior or gradient is nonfinite; inspect priors and initialization")
        inverse_mass = torch.ones_like(position)
        step = _reasonable_step(model, position, value, gradient, inverse_mass,
                                cfg["step_size"], cfg, generator) if cfg["adapt_step_size"] else cfg["step_size"]
        mu, average_log_step, hbar, adapt_iteration = math.log(10*step), math.log(step), 0.0, 0
        window_positions = []
        window = min(cfg["mass_window"], max(5, warmup//3))
        mass_updates, warmup_divergences = 0, 0
        frozen_step = step
        for transition in range(warmup+retained):
            is_warmup = transition < warmup
            leaps = int(torch.randint(max(1, cfg["leapfrog_steps"]//2), cfg["leapfrog_steps"]+1,
                                      (), generator=generator))
            position, value, gradient, probability, did_accept, divergent, error = _transition(
                model, position, value, gradient, inverse_mass, step, leaps, generator, cfg["divergence_threshold"])
            completed += 1
            if is_warmup:
                warmup_divergences += int(divergent)
                window_positions.append(position.clone())
                if cfg["adapt_step_size"]:
                    adapt_iteration += 1
                    eta = 1/(adapt_iteration+10)
                    hbar = (1-eta)*hbar+eta*(cfg["target_accept"]-probability)
                    log_step = mu-math.sqrt(adapt_iteration)*hbar/0.05
                    log_step = min(math.log(cfg["max_step_size"]), max(math.log(cfg["min_step_size"]), log_step))
                    weight = adapt_iteration**(-0.75)
                    average_log_step = weight*log_step+(1-weight)*average_log_step
                    step = math.exp(log_step)
                can_update_mass = (cfg["adapt_mass"] and transition+1 >= 10 and len(window_positions) >= window
                                   and transition+1 <= warmup-max(5, warmup//5))
                if can_update_mass:
                    draws = torch.stack(window_positions)
                    n = draws.shape[0]
                    variance = draws.var(dim=0, unbiased=True)
                    # Momentum covariance M is the inverse of this vector.
                    # Position variance is therefore M^-1, not M.
                    # A large absolute ridge masks the very different posterior
                    # scales at long record lengths, freezing slow coordinates.
                    # This regularizes only the warm-up metric, never the prior.
                    weight = cfg["metric_regularization_weight"]
                    inverse_mass = ((n*variance+weight*cfg["metric_regularization_variance"])/(n+weight)).clamp(
                        cfg["metric_min_variance"], cfg["metric_max_variance"])
                    mass_updates += 1
                    window_positions.clear()
                    if cfg["adapt_step_size"]:
                        step = _reasonable_step(model, position, value, gradient, inverse_mass, step, cfg, generator)
                        mu, average_log_step, hbar, adapt_iteration = math.log(10*step), math.log(step), 0.0, 0
                if transition == warmup-1:
                    if cfg["adapt_step_size"]:
                        step = math.exp(average_log_step)
                    frozen_step = step
            else:
                draw = transition-warmup
                theta, variance, jacobian = model.transform(position)
                posterior[chain, draw] = theta.detach()
                variances[chain, draw] = variance.detach()
                positions[chain, draw] = position
                log_posterior[chain, draw] = value-float(jacobian.detach())
                accepted[chain, draw], probabilities[chain, draw] = did_accept, probability
                divergences[chain, draw], energy_errors[chain, draw] = divergent, error
            if completed % cfg["progress_every"] == 0 or transition in (warmup-1, warmup+retained-1):
                event = {"phase": "sampling", "sampling_phase": "warmup" if is_warmup else "sampling",
                         "transition": completed, "max_transitions": maximum, "chain": chain+1,
                         "chains": count, "chain_transition": transition+1, "step_size": step,
                         "acceptance_probability": probability, "elapsed_seconds": time.perf_counter()-started}
                if progress_callback is not None:
                    progress_callback(event)
                elif verbose:
                    print(f"BI_HMC chain {chain+1}/{count} {event['sampling_phase']}: "
                          f"{completed}/{maximum} transitions, {event['elapsed_seconds']:.1f}s, "
                          f"step={step:.3g}, accept={probability:.3f}", flush=True)
        chain_info.append({"chain": chain+1, "seed": (cfg["seed"]+chain*1000003) % (2**63-1),
                           "step_size": frozen_step, "inverse_mass": inverse_mass.numpy().copy(),
                           "mass_updates": mass_updates, "warmup_divergences": warmup_divergences,
                           "initial_optimization": map_info})

    combined = torch.cat((posterior, variances[..., None]), dim=2).numpy()
    diagnostic = posterior_diagnostics(combined)
    warnings = []
    if count < 2:
        warnings.append("At least two chains are required for Rhat screening")
    if retained < 8:
        warnings.append("Too few retained draws for reliable diagnostics")
    if not np.all(np.isfinite(diagnostic["rhat"])) or not np.all(diagnostic["rhat"] <= cfg["max_rhat"]):
        warnings.append("Rhat threshold not satisfied for all coefficients and variance")
    if not np.all(np.isfinite(diagnostic["bulk_ess"])) or not np.all(diagnostic["bulk_ess"] >= cfg["min_ess"]):
        warnings.append("Bulk ESS threshold not satisfied")
    if not np.all(np.isfinite(diagnostic["tail_ess"])) or not np.all(diagnostic["tail_ess"] >= cfg["min_ess"]):
        warnings.append("Tail ESS threshold not satisfied")
    if divergences.any():
        warnings.append("Post-warmup divergent transitions require investigation")
    if np.any(probabilities.mean(axis=1) < 0.5):
        warnings.append("Low post-warmup mean acceptance probability")
    theta_hat = posterior.mean(dim=(0, 1))
    alpha, beta = model.signal(theta_hat)
    finite = bool(torch.isfinite(theta_hat).all() and torch.isfinite(beta).all())
    stable_mean = _stable(theta_hat, dims)
    if not stable_mean:
        warnings.append("Physical posterior mean is outside the stable denominator domain; no projection was applied")
    point_valid = finite and stable_mean
    passed = not warnings and point_valid
    diagnostic.update({"warnings": warnings, "diagnostics_passed": passed,
                       "acceptance_rate": accepted.mean(axis=1),
                       "mean_acceptance_probability": probabilities.mean(axis=1),
                       "retained_divergences": divergences.sum(axis=1),
                       "divergence_mask": divergences, "energy_error": energy_errors,
                       "chain_info": chain_info, "target_evaluations": model.evaluation_count,
                       "posterior_mean_stable": stable_mean})
    result = {
        "theta_hat": theta_hat.to(device=output_device, dtype=output_dtype),
        "posterior_samples": posterior, "variance_samples": variances,
        "posterior_log_density": log_posterior, "sampling_positions": positions,
        "variance_hat": float(variances.mean()), "posterior_interval": torch.quantile(posterior.reshape(-1, dims.n_params),
                                                                                       torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0),
        "final_alpha": alpha.detach().to(device=output_device, dtype=output_dtype),
        "final_beta": beta.detach().to(device=output_device, dtype=output_dtype),
        "status": "sampling_complete" if passed else ("sampling_warning" if finite else "failed"),
        "diagnostics_passed": passed, "point_estimate_valid": point_valid,
        "point_estimator": "posterior_mean", "sampler_diagnostics": diagnostic,
        "resolved_config": cfg, "sampler_seed": cfg["seed"],
        "sampling_transitions": maximum, "retained_draws": count*retained,
        "forward_model": "exact_causal_iir", "noise_initialization": model.noise_policy,
        "noise_burn_in": model.noise_burn_in, "input_history_length": model.input_count,
        "signal_initialization": "zero_before_observed_input_history" if model.input_count else "zero_at_record_start",
    }
    result["total_time"] = time.perf_counter()-started
    diagnostic["effective_samples_per_second"] = diagnostic["mean_ess"]/result["total_time"]
    return result


def run_bi_hmc(r: torch.Tensor, c: torch.Tensor, dims: WienerDimensions,
               method_cfg: dict[str, Any] | None = None, *, initial_theta=None,
               input_history=None, noise_burn_in: int | None = None, seed: int | None = None,
               progress_callback: Callable[[dict[str, Any]], None] | None = None,
               verbose: bool = False) -> dict[str, Any]:
    """Run the separate HMC inference loop, including all setup in wall time.

    See ``_run_bi_hmc`` for the physical boundary conventions. ``cpu_threads``
    defaults to one for small matrix operations; zero preserves the caller's
    setting. The previous process-wide PyTorch thread count is restored even
    if inference raises. Like PyTorch's own thread setting, this optional
    context is intended for the repository's sequential method runner, not
    concurrent Python threads sharing one PyTorch runtime.
    """
    started = time.perf_counter()
    config = resolve_hmc_config(method_cfg)
    previous_threads = torch.get_num_threads()
    desired_threads = config["cpu_threads"]
    changed = desired_threads > 0 and previous_threads != desired_threads
    try:
        if changed:
            torch.set_num_threads(desired_threads)
        result = _run_bi_hmc(r, c, dims, config, initial_theta=initial_theta,
                            input_history=input_history, noise_burn_in=noise_burn_in,
                            seed=seed, progress_callback=progress_callback, verbose=verbose)
    finally:
        if changed:
            torch.set_num_threads(previous_threads)
    result["total_time"] = time.perf_counter()-started
    result["sampler_diagnostics"]["effective_samples_per_second"] = (
        result["sampler_diagnostics"]["mean_ess"]/result["total_time"])
    return result
