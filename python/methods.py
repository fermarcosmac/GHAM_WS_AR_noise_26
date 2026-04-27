from __future__ import annotations

from typing import Any

import torch

from wiener_system import extract_loss_derivatives, params_to_loss


def _build_step_scale_vector(
    method_cfg: dict[str, Any],
    dims,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    scale = torch.ones(dims.n_params, dtype=dtype, device=device)
    d_step_multiplier = float(method_cfg.get("d_step_multiplier", 1.0))
    if dims.nd > 0 and d_step_multiplier != 1.0:
        scale[-dims.nd :] = d_step_multiplier
    return scale


def method_parameter_update(
    method_name: str,
    method_cfg: dict[str, Any],
    phi_hat: torch.Tensor | None,
    c: torch.Tensor,
    theta_hat: torch.Tensor,
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    dims,
    iter_idx: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    theta_new = theta_hat
    aux_state: dict[str, Any] = {"status": "ok", "message": "", "derivatives": None}
    name = method_name.upper()

    if name == "WS-GGI":
        theta_new, derivatives = ws_ggi_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name == "WS-GNI":
        theta_new, derivatives = ws_gni_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-GGHAM-1-DH", "WS_GGHAM_1_DH"}:
        theta_new, derivatives = ws_ggham_1_dh_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-GGHAM-2-I", "WS_GGHAM_2_I"}:
        theta_new, derivatives = ws_ggham_2_i_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-GGHAM-2-DH", "WS_GGHAM_2_DH"}:
        theta_new, derivatives = ws_ggham_2_dh_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-GGHAM-2-H", "WS_GGHAM_2_H"}:
        theta_new, derivatives = ws_ggham_2_h_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-LGHAM-1-TIK", "WS_LGHAM_1_TIK"}:
        theta_new, derivatives = ws_lgham_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims, order=1)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    if name in {"WS-LGHAM-3-TIK", "WS_LGHAM_3_TIK"}:
        theta_new, derivatives = ws_lgham_update(theta_hat, method_cfg, r=r, nu=nu, c=c, dims=dims, order=3)
        aux_state["derivatives"] = derivatives
        return theta_new, aux_state

    derivative_order = int(method_cfg.get("derivative_order", 1))
    aux_state["derivatives"] = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=min(derivative_order, 3))
    aux_state["status"] = "not_implemented"
    aux_state["message"] = (
        f"{method_name} placeholder selected. "
        "Autodiff derivatives were extracted for the current parameter-to-loss mapping, "
        "but the manual update rule is not implemented yet."
    )
    return theta_new, aux_state


def ws_ggi_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    step_size = float(method_cfg.get("step_size", method_cfg.get("step_scale", 1.0)))

    theta_param = torch.nn.Parameter(theta_hat.detach().clone())
    loss = params_to_loss(theta_param, r, nu, c, dims) # forward_pass and basis for higher-order jacobians
    loss.backward()
    gradient = theta_param.grad.detach().clone()
    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat - step_size * step_scale * gradient

    derivatives = {
        "loss": loss.detach(),
        "grad": gradient.detach(),
    }
    return theta_new.detach().clone(), derivatives


def ws_gni_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eta = float(method_cfg.get("eta", method_cfg.get("step_size", 1.0)))
    lambda_reg = float(method_cfg.get("hessian_regularization", 0.0))

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    hessian_reg = hessian + lambda_reg * torch.eye(
        hessian.shape[0],
        dtype=hessian.dtype,
        device=hessian.device,
    )

    try:
        direction = torch.linalg.solve(hessian_reg, gradient.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        direction = torch.linalg.lstsq(hessian_reg, gradient.unsqueeze(-1)).solution.squeeze(-1)

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat - eta * step_scale * direction
    derivatives["hess_reg"] = hessian_reg.detach()
    derivatives["newton_direction"] = direction.detach()
    return theta_new.detach().clone(), derivatives


def ws_ggham_1_dh_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eta = float(method_cfg.get("eta", method_cfg.get("step_size", 1.0)))
    lambda_reg = float(method_cfg.get("hessian_regularization", 0.0))

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    # GHAM gradient-root convention uses +Phi'r in linearized form.
    gradient_root = -gradient
    diag_h = torch.diagonal(hessian) + lambda_reg
    safe_diag_h = torch.where(diag_h.abs() > 1e-12, diag_h, torch.full_like(diag_h, 1e-12))
    direction = gradient_root / safe_diag_h

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat + eta * step_scale * direction

    derivatives["gradient_root"] = gradient_root.detach()
    derivatives["diag_hess_reg"] = safe_diag_h.detach()
    derivatives["gham_direction"] = direction.detach()
    return theta_new.detach().clone(), derivatives


def ws_ggham_2_i_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eta = float(method_cfg.get("eta", method_cfg.get("step_size", 1.0)))
    lambda_reg = float(method_cfg.get("hessian_regularization", 0.0))

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)

    # TODO: use step_scale (different per each parameter)
    theta_new = theta_hat - step_scale * ((2*eta*torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device) - eta**2*hessian) @ gradient)
    #theta_new = theta_hat - (2*torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)@step_scale - hessian@(step_scale**2)) @ gradient

    derivatives["gradient"] = gradient.detach()
    derivatives["hess"] = hessian.detach()
    return theta_new.detach().clone(), derivatives


def ws_ggham_2_dh_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eta = float(method_cfg.get("eta", method_cfg.get("step_size", 1.0)))
    lambda_reg = float(method_cfg.get("hessian_regularization", 0.0))

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    diag_h = torch.diagonal(hessian) + lambda_reg
    safe_diag_h = torch.where(diag_h.abs() > 1e-12, diag_h, torch.full_like(diag_h, 1e-12))

    theta_1 = - eta * (gradient / safe_diag_h)
    delta = 2.0 * theta_1 - eta * ((hessian @ theta_1) / safe_diag_h)

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat + step_scale * delta

    derivatives["gradient"] = gradient.detach()
    derivatives["diag_hess_reg"] = safe_diag_h.detach()
    derivatives["theta_1"] = theta_1.detach()
    derivatives["gham_delta"] = delta.detach()
    return theta_new.detach().clone(), derivatives


def ws_ggham_2_h_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eta = float(method_cfg.get("eta", method_cfg.get("step_size", 1.0)))
    lambda_reg = float(method_cfg.get("hessian_regularization", 0.0))

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    hessian_reg = hessian + lambda_reg * torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)

    try:
        theta_1 = -eta * torch.linalg.solve(hessian_reg, gradient.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        theta_1 = -eta * torch.linalg.lstsq(hessian_reg, gradient.unsqueeze(-1)).solution.squeeze(-1)

    h_theta_1 = hessian @ theta_1
    try:
        correction = torch.linalg.solve(hessian_reg, h_theta_1.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        correction = torch.linalg.lstsq(hessian_reg, h_theta_1.unsqueeze(-1)).solution.squeeze(-1)

    # NOTE: if hessian is "nice" (i.e. invertible and well-conditioned), this should be just theta_1
    delta = 2.0 * theta_1 - eta * correction

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat + step_scale * delta

    derivatives["gradient"] = gradient.detach()
    derivatives["hess_reg"] = hessian_reg.detach()
    derivatives["theta_1"] = theta_1.detach()
    derivatives["h_correction"] = correction.detach()
    derivatives["gham_delta"] = delta.detach()
    return theta_new.detach().clone(), derivatives


def _solve_loss_root_inverse(
    gradient_root: torch.Tensor,
    rhs_scalar: torch.Tensor,
    method_cfg: dict[str, Any],
) -> torch.Tensor:
    solver = str(method_cfg.get("inverse_solver", "tikhonov")).lower()
    rhs = rhs_scalar.to(dtype=gradient_root.dtype, device=gradient_root.device)
    gg = torch.dot(gradient_root, gradient_root)

    if solver == "pinv":
        if gg.abs().item() <= 1e-15:
            return torch.zeros_like(gradient_root)
        return gradient_root * (rhs / gg)

    if solver == "tikhonov":
        lambda_reg = float(method_cfg.get("tikhonov_lambda", 0.0))
        denom = gg + lambda_reg
        if denom.abs().item() <= 1e-15:
            return torch.zeros_like(gradient_root)
        return gradient_root * (rhs / denom)

    raise ValueError(f"Unknown WS-LGHAM inverse_solver '{solver}'.")


def _get_eps0(method_cfg: dict[str, Any], *, default: float = 0.0) -> float:
    eps0 = method_cfg.get("eps_0", method_cfg.get("epsilon0", default))
    if isinstance(eps0, str) and eps0.lower() == "automatic":
        return float(method_cfg.get("best_error_so_far", default))
    return float(eps0)


def ws_lgham_update(
    theta_hat: torch.Tensor,
    method_cfg: dict[str, Any],
    *,
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    dims,
    order: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if order < 1 or order > 3:
        raise ValueError("WS-LGHAM supports only orders 1 to 3.")

    eta = float(method_cfg.get("eta", 0.05))
    eps0 = _get_eps0(method_cfg, default=0.0)

    derivatives = extract_loss_derivatives(theta_hat, r, nu, c, dims, max_order=2)
    loss = derivatives["loss"]
    gradient = derivatives["grad"]
    hessian = derivatives["hess"]

    # Gradient-root convention matches the linearized MATLAB formulas.
    rhs1 = loss - eps0
    theta_1 = - eta * _solve_loss_root_inverse(gradient, rhs1, method_cfg)
    delta = theta_1.clone()

    theta_2 = torch.zeros_like(theta_hat)
    if order >= 2:
        rhs2 = torch.dot(gradient, theta_1)
        theta_2 = theta_1 - eta * _solve_loss_root_inverse(gradient, rhs2, method_cfg)
        delta = delta + theta_2

    if order >= 3:
        rhs3 = 0.5 * torch.dot(theta_1, hessian @ theta_1) + torch.dot(gradient, theta_2)
        theta_3 = theta_2 - eta * _solve_loss_root_inverse(gradient, rhs3, method_cfg)
        delta = delta + theta_3
        derivatives["theta_3"] = theta_3.detach()

    step_scale = _build_step_scale_vector(method_cfg, dims, dtype=theta_hat.dtype, device=theta_hat.device)
    theta_new = theta_hat + step_scale * delta

    derivatives["gradient"] = gradient.detach()
    derivatives["theta_1"] = theta_1.detach()
    derivatives["theta_2"] = theta_2.detach()
    derivatives["lgham_delta"] = delta.detach()
    return theta_new.detach().clone(), derivatives
