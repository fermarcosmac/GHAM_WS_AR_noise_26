"""Compact, detached diagnostics and reproducibility metadata for comparison runs."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import numpy as np
import torch


def array_digest(*arrays) -> str:
    """SHA-256 of concatenated little-endian float64 vectors (r, c for data)."""
    digest = hashlib.sha256()
    for value in arrays:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        digest.update(np.asarray(value, dtype="<f8").reshape(-1).tobytes())
    return digest.hexdigest()


def resolve_method_config(cfg: dict, experiment: dict | None = None) -> dict:
    cfg = deepcopy(cfg)
    experiment = experiment or {}
    if "initialization" in experiment:
        cfg["initialization"] = deepcopy(experiment["initialization"])
    cfg.setdefault("initialization", {}).setdefault("policy", "legacy")
    for section, defaults in (
        ("stopping", {"gradient_tolerance": 1e-6, "loss_tolerance": 1e-8}),
        ("diagnostics", {"enabled": True, "save_matrices": False}),
    ):
        cfg[section] = {**defaults, **cfg.get(section, {}), **experiment.get(section, {})}
    return cfg


def config_json(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, allow_nan=False)


def matrix_metrics(matrix: torch.Tensor) -> dict[str, float]:
    """Keep singularity distinct from missing data, using a scale-relative rank test."""
    a = matrix.detach().cpu().double().numpy()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Diagnostics require a finite square matrix.")
    singular = np.linalg.svd(a, compute_uv=False)
    tol = max(a.shape) * torch.finfo(matrix.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tol))
    eigen = np.linalg.eigvalsh(0.5 * (a + a.T))
    return {
        "condition": float(singular[0] / singular[-1]) if rank == a.shape[0] else float("inf"),
        "sigma_min": float(singular[-1]), "sigma_max": float(singular[0]),
        "rank": float(rank), "rank_tolerance": float(tol),
        "singular": float(rank < a.shape[0]),
        "eigen_min": float(eigen[0]), "eigen_max": float(eigen[-1]),
        "negative_count": float(np.count_nonzero(eigen < -tol)),
        "symmetry_error": float(np.linalg.norm(a-a.T) / max(np.linalg.norm(a), np.finfo(float).tiny)),
    }


def step_diagnostics(derivatives: dict, aux: dict, cfg: dict, evaluation_iteration: int) -> dict:
    row: dict[str, Any] = {"evaluation_iteration": evaluation_iteration, "available": 0.,
                           "solver_kind": "none", "diagnostic_status": "ok"}
    for key in ("gradient_norm", "proposed_step_norm", "applied_step_norm", "projected",
                "rejected", "solver_fallback", "inverse_denominator"):
        value = aux.get(key, derivatives.get(key, float("nan")))
        row[key] = float(value)
    if "grad" in derivatives:
        row["gradient_norm"] = float(torch.linalg.vector_norm(derivatives["grad"]))
    row["regularization"] = float(cfg.get("hessian_regularization", cfg.get("tikhonov_lambda", 0.)))
    row["d_step_multiplier"] = float(cfg.get("d_step_multiplier", 1.))
    if not cfg["diagnostics"]["enabled"]:
        row["diagnostic_status"] = "disabled"
        return row
    try:
        if "hess" in derivatives:
            row.update({"raw_"+k: v for k, v in matrix_metrics(derivatives["hess"]).items()})
            row["available"] = 1.
            scales = cfg["diagnostics"].get("parameter_scales")
            if scales is not None:
                s = torch.as_tensor(scales, dtype=derivatives["hess"].dtype, device=derivatives["hess"].device)
                if s.shape != (derivatives["hess"].shape[0],) or not torch.all(torch.isfinite(s) & (s > 0)):
                    raise ValueError("parameter_scales must contain one finite positive scale per parameter")
                scaled = s[:, None] * derivatives["hess"] * s[None, :]
                row["scaled_condition"] = matrix_metrics(scaled)["condition"]
        solver = derivatives.get("hess_reg")
        if solver is not None:
            row["solver_kind"] = "regularized_hessian"
        elif "diag_hess_reg" in derivatives:
            solver = torch.diag(derivatives["diag_hess_reg"])
            row["solver_kind"] = "safeguarded_diagonal"
        elif "inverse_denominator" in derivatives:
            row["solver_kind"] = "gradient_row"
        if solver is not None:
            row.update({"solver_"+k: v for k, v in matrix_metrics(solver).items()})
    except (ValueError, np.linalg.LinAlgError) as exc:
        row["diagnostic_status"] = str(exc)
    return row


def pack_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        return {"evaluation_iteration": np.array([], dtype=float)}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    return {k: np.asarray([r.get(k, "" if k in {"solver_kind", "diagnostic_status"} else np.nan) for r in rows],
                          dtype=object if k in {"solver_kind", "diagnostic_status"} else float)
            for k in keys}


def summarize_diagnostics(d: dict) -> dict:
    out = {}
    for prefix in ("raw", "solver"):
        values = np.asarray(d.get(prefix+"_condition", []), dtype=float)
        values = values[~np.isnan(values)]  # Inf is evidence of singularity, not missing data.
        if values.size:
            out.update({prefix+"_condition_initial": values[0], prefix+"_condition_final": values[-1],
                        prefix+"_condition_peak": values.max(), prefix+"_sample_count": values.size,
                        prefix+"_singular_fraction": np.mean(np.isinf(values))})
            negative = np.asarray(d[prefix+"_negative_count"])
            out[prefix+"_negative_curvature_fraction"] = np.mean(negative[np.isfinite(negative)] > 0)
    for key in ("solver_fallback", "projected", "rejected"):
        out[key+"_count"] = float(np.nansum(d.get(key, [])))
    return out
