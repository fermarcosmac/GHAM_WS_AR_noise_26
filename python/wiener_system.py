from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from torch.func import jacrev


@dataclass(frozen=True)
class WienerDimensions:
    na: int
    nb: int
    nf: int
    nd: int

    @property
    def n_params(self) -> int:
        return self.na + self.nb + (self.nf - 1) + self.nd


def lfilter_1d(waveform: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor) -> torch.Tensor:
    waveform = waveform.reshape(1, -1)
    a_coeffs = a_coeffs.reshape(1, -1)
    b_coeffs = b_coeffs.reshape(1, -1)

    # torchaudio.functional.lfilter requires numerator and denominator
    # coefficient tensors to have the same trailing dimension. Zero-padding
    # the shorter side preserves the intended transfer function.
    target_len = max(a_coeffs.shape[-1], b_coeffs.shape[-1])
    if a_coeffs.shape[-1] < target_len:
        a_coeffs = torch.nn.functional.pad(a_coeffs, (0, target_len - a_coeffs.shape[-1]))
    if b_coeffs.shape[-1] < target_len:
        b_coeffs = torch.nn.functional.pad(b_coeffs, (0, target_len - b_coeffs.shape[-1]))

    filtered = torchaudio.functional.lfilter(waveform, a_coeffs, b_coeffs, clamp=False, batching=True)
    return filtered.reshape(-1)


def polynomial_nonlinearity(alpha: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
    beta = alpha
    for power, coeff in enumerate(f_v, start=2):
        beta = beta + coeff * alpha.pow(power)
    return beta


def build_lag_matrix(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    T = x.numel()
    Xlag = torch.zeros((T, max_lag), dtype=x.dtype, device=x.device)
    for lag in range(1, max_lag + 1):
        Xlag[lag:, lag - 1] = x[:-lag]
    return Xlag


def build_state_matrix(alpha_hat: torch.Tensor, e_hat: torch.Tensor, r: torch.Tensor, dims: WienerDimensions) -> torch.Tensor:
    alpha_lags = build_lag_matrix(alpha_hat, dims.na)
    r_lags = build_lag_matrix(r, dims.nb)
    e_lags = build_lag_matrix(e_hat, dims.nd)

    if dims.nf > 1:
        nonlinear_terms = torch.stack([alpha_hat.pow(p) for p in range(2, dims.nf + 1)], dim=1)
    else:
        nonlinear_terms = torch.zeros((r.numel(), 0), dtype=r.dtype, device=r.device)

    return torch.cat([-alpha_lags, r_lags, nonlinear_terms, -e_lags], dim=1)


def split_theta(theta: torch.Tensor, dims: WienerDimensions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_a_end = dims.na
    idx_b_end = idx_a_end + dims.nb
    idx_f_end = idx_b_end + (dims.nf - 1)
    a_v = theta[:idx_a_end]
    b_v = theta[idx_a_end:idx_b_end]
    f_v = theta[idx_b_end:idx_f_end]
    d_v = theta[idx_f_end:]
    return a_v, b_v, f_v, d_v


def simulate_wiener(r: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor, dims: WienerDimensions) -> torch.Tensor:
    a_v, b_v, f_v, d_v = split_theta(theta, dims)
    den_lin = torch.cat([torch.ones(1, dtype=theta.dtype, device=theta.device), a_v])
    num_lin = torch.cat([torch.zeros(1, dtype=theta.dtype, device=theta.device), b_v])
    alpha = lfilter_1d(r, den_lin, num_lin)

    beta = polynomial_nonlinearity(alpha, f_v)

    den_ar = torch.cat([torch.ones(1, dtype=theta.dtype, device=theta.device), d_v])
    num_ar = torch.ones(1, dtype=theta.dtype, device=theta.device)
    e = lfilter_1d(nu, den_ar, num_ar)
    return beta + e


def parameter_to_loss(theta: torch.Tensor, r: torch.Tensor, nu: torch.Tensor, c: torch.Tensor, dims: WienerDimensions) -> torch.Tensor:
    c_hat = simulate_wiener(r, nu, theta, dims)
    residual = c - c_hat
    return 0.5 * torch.sum(residual.pow(2))


def extract_loss_derivatives(
    theta: torch.Tensor,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims: WienerDimensions,
    max_order: int = 1,
) -> dict[str, torch.Tensor]:
    if max_order < 1 or max_order > 3:
        raise ValueError("Only derivative orders 1..3 are supported.")

    loss_fn = lambda th: parameter_to_loss(th, r, nu, c, dims)
    derivatives: dict[str, torch.Tensor] = {"loss": loss_fn(theta)}

    grad = jacrev(loss_fn)(theta)
    derivatives["grad"] = grad

    if max_order >= 2:
        hess = jacrev(jacrev(loss_fn))(theta)
        derivatives["hess"] = hess

    if max_order >= 3:
        jac3 = jacrev(jacrev(jacrev(loss_fn)))(theta)
        derivatives["jac3"] = jac3

    return derivatives


def fit_init_vector(vec_in: Any, target_len: int, fill_value: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(vec_in, (int, float)):
        return torch.full((target_len,), float(vec_in), dtype=dtype, device=device)

    vec = torch.as_tensor(vec_in, dtype=dtype, device=device).reshape(-1)
    out = torch.full((target_len,), fill_value, dtype=dtype, device=device)
    n_copy = min(target_len, vec.numel())
    out[:n_copy] = vec[:n_copy]
    return out


def initialize_method_state(
    method_cfg: dict[str, Any],
    r: torch.Tensor,
    c: torch.Tensor,
    dims: WienerDimensions,
) -> dict[str, torch.Tensor]:
    n = dims.n_params
    device = r.device
    dtype = r.dtype
    T = r.numel()

    alpha_hat = torch.full((T,), 1e-6, dtype=dtype, device=device)
    e_hat = torch.full((T,), 1e-6, dtype=dtype, device=device)
    theta_hat = torch.full((n,), 1e-6, dtype=dtype, device=device)

    init_cfg = method_cfg.get("initialization")
    if init_cfg is None:
        return {"alpha_hat": alpha_hat, "e_hat": e_hat, "theta_hat": theta_hat}

    theta_init = init_cfg.get("theta_init")
    if theta_init is not None:
        theta_hat = fit_init_vector(theta_init, n, 1e-6, dtype=dtype, device=device)

    if init_cfg.get("mode", "").lower() == "physical":
        a0 = torch.zeros(dims.na, dtype=dtype, device=device)
        b0 = torch.zeros(dims.nb, dtype=dtype, device=device)
        if dims.nb >= 1:
            b0[0] = 1.0

        if "linear_init_a" in init_cfg:
            a0 = fit_init_vector(init_cfg["linear_init_a"], dims.na, 0.0, dtype=dtype, device=device)
        if "linear_init_b" in init_cfg:
            b0 = fit_init_vector(init_cfg["linear_init_b"], dims.nb, 0.0, dtype=dtype, device=device)
            if dims.nb >= 1 and torch.count_nonzero(b0).item() == 0:
                b0[0] = 1.0

        den_lin_0 = torch.cat([torch.ones(1, dtype=dtype, device=device), a0])
        num_lin_0 = torch.cat([torch.zeros(1, dtype=dtype, device=device), b0])
        alpha_hat = lfilter_1d(r, den_lin_0, num_lin_0)

        e_mode = init_cfg.get("e_init_mode", "output_residual").lower()
        if e_mode == "output_residual":
            e_hat = c - alpha_hat
        elif e_mode == "zeros":
            e_hat = torch.zeros(T, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown e_init_mode '{e_mode}'.")

    return {"alpha_hat": alpha_hat, "e_hat": e_hat, "theta_hat": theta_hat}


def generate_example_data(
    dims: WienerDimensions,
    theta_true: torch.Tensor,
    lambda_g: int,
    burn_in: int,
    sigma_nu: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    N_total = lambda_g + burn_in
    r_full = torch.randn(N_total, dtype=dtype, device=device)
    nu_full = sigma_nu * torch.randn(N_total, dtype=dtype, device=device)
    c_full = simulate_wiener(r_full, nu_full, theta_true, dims)

    keep = slice(burn_in, burn_in + lambda_g)
    return {
        "r": r_full[keep],
        "nu": nu_full[keep],
        "c": c_full[keep],
        "r_full": r_full,
        "nu_full": nu_full,
        "c_full": c_full,
    }
