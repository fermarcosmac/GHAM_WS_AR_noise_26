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
    phi_hat: torch.Tensor,
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
