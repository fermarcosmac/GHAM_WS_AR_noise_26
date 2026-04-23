from __future__ import annotations

from typing import Any

import torch

from wiener_system import extract_loss_derivatives


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
        theta_new = ws_ggi_update(phi_hat, c, theta_hat, method_cfg)
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


def ws_ggi_update(phi_hat: torch.Tensor, c: torch.Tensor, theta_hat: torch.Tensor, method_cfg: dict[str, Any]) -> torch.Tensor:
    step_scale = float(method_cfg["step_scale"])
    fro_sq = torch.linalg.matrix_norm(phi_hat, ord="fro").pow(2).clamp_min(torch.finfo(phi_hat.dtype).eps)
    delta = step_scale / fro_sq
    residual = c - phi_hat @ theta_hat
    return theta_hat + delta * (phi_hat.transpose(0, 1) @ residual)
