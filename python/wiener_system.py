from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.func import jacfwd, jacrev


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
    waveform_batched = waveform.reshape(1, 1, -1)
    a_batched = a_coeffs.reshape(1, -1)
    b_batched = b_coeffs.reshape(1, -1)
    filtered = lfilter_via_fsm(waveform_batched, b_batched, a_batched)
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


def params_to_loss(theta: torch.Tensor, r: torch.Tensor, nu: torch.Tensor, c: torch.Tensor, dims: WienerDimensions) -> torch.Tensor:
    c_hat = simulate_wiener(r, nu, theta, dims)
    residual = c - c_hat
    return torch.mean(residual.pow(2))


def parameter_to_loss(theta: torch.Tensor, r: torch.Tensor, nu: torch.Tensor, c: torch.Tensor, dims: WienerDimensions) -> torch.Tensor:
    return params_to_loss(theta, r, nu, c, dims)


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

    loss_fn = lambda th: params_to_loss(th, r, nu, c, dims)
    loss = loss_fn(theta)
    grad = jacrev(loss_fn)(theta)

    derivatives: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "grad": grad.detach(),
    }

    if max_order >= 2:
        hess = jacfwd(jacrev(loss_fn))(theta)
        derivatives["hess"] = hess.detach()

    if max_order >= 3:
        jac3 = jacfwd(jacfwd(jacrev(loss_fn)))(theta)
        derivatives["jac3"] = jac3.detach()

    # TODO: implement higher order derivatives when needed

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

def lfilter_via_fsm(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor = None):
    """Use the frequency sampling method to approximate an IIR filter.
    The filter will be applied along the final dimension of x.
    Args:
        x (torch.Tensor): Time domain signal with shape (bs, 1, timesteps)
        b (torch.Tensor): Numerator coefficients with shape (bs, N).
        a (torch.Tensor): Denominator coefficients with shape (bs, N).
    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, 1, timesteps)
    """
    bs, chs, seq_len = x.size()  # enforce shape
    assert chs == 1

    # round up to nearest power of 2 for FFT
    fft_len = (2 * x.shape[-1]) - 1
    n_fft = 1 << (fft_len - 1).bit_length()

    # move coefficients to same device as x
    b = b.type_as(x)

    if a is None:
        # directly compute FFT of numerator coefficients
        H = torch.fft.rfft(b, n_fft)
    else:
        a = a.type_as(x)
        # compute complex response as ratio of polynomials
        H = fft_freqz(b, a, n_fft=n_fft)

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y


def fft_freqz(b, a, n_fft: int = 512):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y
