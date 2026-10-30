from __future__ import annotations

import sys
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import (
    ROOT as PY_ROOT,
    build_param_names,
    build_plot_labels,
    choose_dtype,
    default_device,
    load_json_config,
    make_valid_method_key,
    now,
    plot_method_metrics,
    plot_parameter_trajectories,
    set_seed,
)
from diagnostics import (array_digest, config_json, resolve_method_config, step_diagnostics, pack_diagnostics, summarize_diagnostics)
from methods import method_parameter_update
from wiener_system import (
    WienerDimensions,
    _resolve_fsm_lengths,
    extract_loss_derivatives,
    build_state_matrix,
    estimate_innovations_from_state,
    fsm_kwargs_from_config,
    generate_example_data,
    initialize_method_state,
    parameter_to_loss,
    simulate_wiener,
    update_latent_state_from_innovations,
)

PYTHON_EXTRA_METHODS = ["WS-GGHAM-2-I", "WS-GGHAM-2-H"]
FINITE_RESULT_STATUSES = {"ok", "converged", "max_iterations", "stalled", "sampling_complete", "sampling_warning"}


def inference_data_context(data: dict, burn_in: int = 0) -> dict:
    """Observed prehistory and noise-start policy; never expose true innovations."""
    history = data.get("input_prehistory")
    if history is None and "r_full" in data:
        history = data["r_full"][:burn_in]
    context = {"noise_burn_in": int(data.get("noise_burn_in", burn_in))}
    if history is not None:
        context["input_prehistory"] = history
    return context


def _sampling_seed(method_cfg: dict, metadata: dict) -> int:
    """Independent of truth/noise amplitude and stable across method ordering."""
    base = int(method_cfg.get("seed", 20261030))
    # Pair the sampler RNG across noise/resolution conditions of a data replicate.
    replicate = int(metadata.get("initialization_index", metadata.get("mc_index", 1))) - 1
    return (base + 104729 * replicate) % (2**31 - 1)


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def run_bayesian_identification(
    method_cfg: dict, r: torch.Tensor, nu: torch.Tensor, c: torch.Tensor,
    theta_true: torch.Tensor, dims: WienerDimensions, *, run_metadata: dict | None = None,
    data_context: dict | None = None, verbose: bool = True, progress_callback=None,
) -> dict[str, Any]:
    """Run HMC separately, then score only its physical posterior-mean estimate.

    Truth and the simulated innovations enter assessment below, never run_bi_hmc.
    Empty optimizer histories intentionally prevent interpreting draws as updates.
    """
    from bayesian_hmc import run_bi_hmc, resolve_hmc_config

    started = time.perf_counter()
    cfg = resolve_hmc_config(resolve_method_config(method_cfg))
    metadata, context = dict(run_metadata or {}), dict(data_context or {})
    sampler_seed = _sampling_seed(cfg, metadata)
    init_cfg = cfg.get("initialization", {})
    initial = init_cfg.get("theta0")
    if initial is None:
        initial = np.zeros(dims.n_params)
        if dims.nb:
            initial[dims.na] = 1.
    initial = np.asarray(initial, dtype=float).reshape(-1)
    if initial.size != dims.n_params or not np.isfinite(initial).all():
        raise ValueError("BI_HMC initialization.theta0 must contain one finite value per physical coefficient")

    def report(event):
        event = {**event, "method": "BI_HMC", "method_kind": "bayesian",
                 "record_length": len(r), "run_metadata": metadata}
        if progress_callback is not None:
            progress_callback(event)
        elif verbose and event.get("phase") == "sampling":
            completed = event.get("transition", 0)
            print(f"BI_HMC: chain {event.get('chain', '?')}; {event.get('sampling_phase', 'sampling')}; "
                  f"{completed}/{event.get('max_transitions', '?')} transitions; "
                  f"{event.get('elapsed_seconds', 0.):.1f}s elapsed", flush=True)

    if verbose:
        print("BI_HMC: separate HMC loop; final estimator = physical posterior mean", flush=True)
    sample = run_bi_hmc(r, c, dims, cfg, initial_theta=initial,
        input_history=context.get("input_prehistory", context.get("input_history")),
        noise_burn_in=context.get("noise_burn_in"), seed=sampler_seed, progress_callback=report)
    theta_hat = _numpy(sample["theta_hat"]).astype(float).reshape(-1)
    valid = bool(sample.get("point_estimate_valid", np.isfinite(theta_hat).all()))
    truth = theta_true.detach().cpu().numpy().reshape(-1)
    norm_truth = np.linalg.norm(truth)
    final_error = float(100. * np.linalg.norm(theta_hat-truth)/norm_truth) if valid and norm_truth > 0 else np.nan
    output = np.full(len(r), np.nan)
    if valid:
        with torch.no_grad():
            estimate = torch.as_tensor(theta_hat, dtype=r.dtype, device=r.device)
            output = simulate_wiener(r, nu, estimate, dims).detach().cpu().numpy()
    observed = c.detach().cpu().numpy()
    output_error = observed-output
    final_rmse = float(np.sqrt(np.mean(output_error**2))) if np.isfinite(output_error).all() else np.nan
    samples = _numpy(sample.get("posterior_samples", [])).astype(float)
    variance_samples = _numpy(sample.get("variance_samples", [])).astype(float)
    metadata.update({"data_sha256": array_digest(r, c), "dataset_id": array_digest(r, c),
        "initialization_sha256": array_digest(initial), "backend": "python", "dtype": "torch.float64",
        "device": "cpu", "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
        "sampler_seed": sampler_seed, "point_estimate": "posterior_mean",
        "forward_model": "exact_iir", "noise_burn_in": sample.get("noise_burn_in", -1),
        "noise_initialization": sample.get("noise_initialization", "stationary"),
        "signal_initialization": sample.get("signal_initialization", "zero_at_record_start"),
        "input_history_length": sample.get("input_history_length", 0),
        "input_history_sha256": array_digest(context.get("input_prehistory", context.get("input_history", [])))})
    resolved_cfg = sample.get("resolved_config", cfg)
    resolved_json = config_json(resolved_cfg)
    metadata["config_sha256"] = hashlib.sha256(resolved_json.encode("utf-8")).hexdigest()
    metadata["run_id"] = ":".join((metadata["dataset_id"], "BI_HMC", metadata["initialization_sha256"],
                                   metadata["config_sha256"], str(sampler_seed)))
    if r.is_cuda:
        torch.cuda.synchronize(r.device)
    total_wall = time.perf_counter()-started
    result = {"name": "BI_HMC", "method_kind": "bayesian", "is_bayesian": True,
        "point_estimate": "posterior_mean", "status": sample["status"],
        "status_msg": sample.get("status_msg", sample.get("message", "; ".join(sample.get("sampler_diagnostics", {}).get("warnings", [])) or "Sampling diagnostics passed.")),
        "theta_hat": theta_hat, "theta0": initial.copy(), "theta_hist": np.empty((dims.n_params, 0)),
        **{key: np.array([], dtype=float) for key in ("param_err", "rmse_hist", "est_loss_hist", "rel_change", "iter_time", "cum_time")},
        "active_mask": np.array([], dtype=bool), "final_err": final_error,
        "total_time": total_wall, "total_wall_time": total_wall,
        "inference_wall_time": float(sample.get("total_time", total_wall)),
        "diagnostic_time": 0., "iterations": 0, "update_count": 0, "accepted_updates": 0,
        "stopping_iteration": -1, "sampling_transitions": int(sample.get("sampling_transitions", 0)),
        "posterior_samples": samples, "posterior_variance_samples": variance_samples,
        "posterior_log_density": _numpy(sample.get("posterior_log_density", [])),
        "posterior_draw_count": int(samples.shape[0]*samples.shape[1]) if samples.ndim == 3 else 0,
        "sampler_seed": sampler_seed, "sampler_diagnostics": sample.get("sampler_diagnostics", {}),
        "diagnostics_passed": bool(sample.get("diagnostics_passed", False)), "point_estimate_valid": valid,
        "final_output_rmse": final_rmse, "output_estimate": output, "output_error": output_error,
        "noise_free_output_estimate": _numpy(sample.get("final_beta", [])),
        "final_innovation_variance": float(variance_samples.mean()) if variance_samples.size else np.nan,
        "run_metadata": metadata, "resolved_config_json": resolved_json,
        "objective_definition": "gaussian_innovation_likelihood_exact_iir",
        "rmse_definition": "true_innovation_reconstruction", "diagnostics": pack_diagnostics([]),
        "diagnostic_summary": {}, "derivative_snapshots": []}
    if samples.ndim == 3 and samples.size:
        flat = samples.reshape(-1, dims.n_params)
        result["posterior_std"] = flat.std(axis=0, ddof=1)
        result["posterior_interval"] = np.quantile(flat, [.025, .975], axis=0)
    if verbose:
        print(f"BI_HMC: {result['status']}; {result['posterior_draw_count']} retained draws; "
              f"final parameter error {final_error:.5g}%; total wall time {total_wall:.3f}s", flush=True)
    return result


def save_comparison_results(
    output_file: Path,
    results: list[dict[str, Any]],
    *,
    experiment_name: str,
    mode: str,
    selected_methods: list[str],
    dims: WienerDimensions,
    theta_true: torch.Tensor,
    lambda_g: int,
    K_max: int,
    conv_threshold: float,
    sigma_nu: float,
    burn_in: int,
    enforce_fixed_iterations: bool,
    seed: int,
    n_mc: int = 1,
    example_data: dict[str, Any] | None = None,
    mc_summary: dict[str, Any] | None = None,
    all_runs: list | None = None,
    resolved_experiment: dict | None = None,
    study: dict | None = None,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError("Saving MATLAB-compatible results requires scipy. Install scipy and rerun this script.") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    param_names = build_param_names(dims.na, dims.nb, dims.nf, dims.nd)
    param_labels = build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd)
    def mat_struct(items):
        # Bayesian and optimization results share final scores but have different
        # supplementary fields. Preserve the union, regardless of method order.
        fields = list(dict.fromkeys(field for item in items for field in item))
        output = np.empty((1, len(items)), dtype=[(field, "O") for field in fields])
        for idx, result in enumerate(items):
            for field in fields:
                value = result.get(field, np.empty((0, 0)))
                if value is None:
                    value = np.empty((0, 0))
                if field == "derivative_snapshots":
                    value = np.asarray(value, dtype=object)
                output[field][0, idx] = value
        return output
    mat_results = mat_struct(results)

    experiment = {
        "name": experiment_name,
        "mode": mode,
        "seed": seed,
        "N_MC": n_mc,
        "lambda_g": lambda_g,
        "K_max": K_max,
        "conv_threshold": conv_threshold,
        "sigma_nu": sigma_nu,
        "burn_in": burn_in,
        "enforce_fixed_iterations": enforce_fixed_iterations,
        "dims": {"na": dims.na, "nb": dims.nb, "nf": dims.nf, "nd": dims.nd},
        "theta_true": theta_true.detach().cpu().numpy().reshape(-1, 1),
        "param_names": np.asarray(param_names, dtype=object).reshape(1, -1),
        "param_labels": np.asarray(param_labels, dtype=object).reshape(1, -1),
        "selected_methods": np.asarray(selected_methods, dtype=object).reshape(1, -1),
    }

    payload = {
        "schema_version": 2,
        "resolved_experiment_json": config_json(resolved_experiment or {}),
        "parametrization": "FSM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "results": mat_results,
    }
    if example_data is not None:
        payload["example_data"] = example_data
    if mc_summary is not None:
        payload["mc_summary"] = mc_summary
    if study is not None:
        payload["study"] = study

    cells = np.empty((1, len(all_runs or [results])), dtype=object)
    for idx, run in enumerate(all_runs or [results]):
        cells[0, idx] = mat_struct(run)
    payload["all_runs"] = cells
    savemat(output_file, payload, long_field_names=True)
    print(f"\nSaved FSM results to: {output_file}")


def run_identification_method(
    method_name: str, method_cfg: dict[str, Any], r: torch.Tensor, nu: torch.Tensor,
    c: torch.Tensor, theta_true: torch.Tensor, dims: WienerDimensions,
    lambda_g: int, K_max: int, conv_threshold: float, *,
    enforce_fixed_iterations: bool = False, verbose: bool = True,
    run_metadata: dict | None = None,
    data_context: dict | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    if method_name.upper() == "BI_HMC":
        return run_bayesian_identification(method_cfg, r, nu, c, theta_true, dims,
            run_metadata=run_metadata, data_context=data_context, verbose=verbose,
            progress_callback=progress_callback)
    start_time = now()
    if K_max < 1 or conv_threshold < 0:
        raise ValueError("K_max must be positive and conv_threshold nonnegative")
    method_cfg = resolve_method_config(method_cfg)
    method_cfg.setdefault("innovation_estimator", "fsm_inverse")
    method_cfg.setdefault("stability_projection", True)
    method_cfg.setdefault("stability_radius", 0.98)
    method_cfg.setdefault("d_step_multiplier", 1.)
    lengths = _resolve_fsm_lengths(len(r), **fsm_kwargs_from_config(method_cfg))
    method_cfg.update(dict(zip(("fsm_tau_g", "fsm_tau_h", "fsm_n_freq"), lengths)))
    method_cfg["run_settings"] = {"K_max": K_max, "conv_threshold": conv_threshold,
                                  "enforce_fixed_iterations": enforce_fixed_iterations}
    if "LGHAM" in method_name.upper():
        target = method_cfg.get("eps_0", method_cfg.get("epsilon0", 0.))
        method_cfg["resolved_epsilon0"] = float(method_cfg.get("best_error_so_far", 0.)) if str(target).lower() == "automatic" else float(target)
    init = initialize_method_state(method_cfg, r, c, dims)
    theta_hat = init["theta_hat"]
    theta0 = theta_hat.detach().cpu().numpy().copy()
    alpha_hat, e_hat = init["alpha_hat"], init["e_hat"]
    fsm_kwargs = fsm_kwargs_from_config(method_cfg)
    innovation_mode = method_cfg["innovation_estimator"]
    if innovation_mode not in {"state_residual", "fsm_inverse"}:
        raise ValueError(f"Unknown innovation_estimator {innovation_mode!r}")
    nu_hat = torch.zeros_like(c)  # FSM loss reconstructs innovations from measured data.
    metadata = dict(run_metadata or {})
    metadata.update({"data_sha256": array_digest(r, c), "initialization_sha256": array_digest(theta0),
                     "backend": "python", "dtype": str(r.dtype), "device": str(r.device),
                     "torch_version": str(torch.__version__), "python_version": sys.version.split()[0]})
    metadata["dataset_id"] = metadata["data_sha256"]
    metadata["config_sha256"] = hashlib.sha256(config_json(method_cfg).encode("utf-8")).hexdigest()
    metadata["run_id"] = ":".join((metadata["dataset_id"], method_name, metadata["initialization_sha256"], metadata["config_sha256"]))
    history = {k: [] for k in ("param_err", "rmse_hist", "est_loss_hist", "rel_change", "iter_time", "cum_time", "active_mask")}
    theta_history = []
    rows, snapshots = [], []
    status, message = "max_iterations", "Iteration budget exhausted."
    updates = accepted = 0
    stopping_iteration = -1
    diagnostic_time = 0.

    def record(theta, elapsed=0., change=0., active=True):
        theta_history.append(theta.detach().cpu().numpy().copy())
        history["param_err"].append(float(torch.linalg.vector_norm(theta-theta_true) / torch.linalg.vector_norm(theta_true) * 100.))
        history["rmse_hist"].append(float(torch.sqrt(torch.mean((c-simulate_wiener(r, nu, theta, dims)).square()))))
        history["est_loss_hist"].append(float(parameter_to_loss(theta, r, nu_hat, c, dims, **fsm_kwargs)))
        history["rel_change"].append(change)
        history["iter_time"].append(elapsed)
        history["cum_time"].append((history["cum_time"][-1] if history["cum_time"] else 0.) + elapsed)
        history["active_mask"].append(active)

    record(theta_hat)
    if not np.isfinite([history["rmse_hist"][0], history["est_loss_hist"][0]]).all():
        status, message = "failed", "Initial objective or reconstruction is non-finite."
    else:
        for k in tqdm(range(1, K_max), desc=method_name, unit="iter", leave=False, disable=not verbose):
            t0 = now()
            phi_hat = None
            if innovation_mode == "state_residual":
                phi_hat = build_state_matrix(alpha_hat, e_hat, r, dims)
            updates += 1
            try:
                candidate, aux = method_parameter_update(method_name, method_cfg, phi_hat, c, theta_hat,
                                                         r=r, nu=nu_hat, dims=dims, iter_idx=k)
            except (RuntimeError, FloatingPointError) as exc:
                status, message = "failed", str(exc)
                break
            derivatives = aux.get("derivatives") or {}
            td = now()
            rows.append(step_diagnostics(derivatives, aux, method_cfg, k-1))
            if method_cfg["diagnostics"]["enabled"] and method_cfg["diagnostics"]["save_matrices"]:
                snapshots.append({"evaluation_iteration": k-1, **{key: value.detach().cpu().numpy()
                                  for key, value in derivatives.items() if isinstance(value, torch.Tensor)}})
            step_diag_time = now()-td
            diagnostic_time += step_diag_time
            if aux["status"] == "not_implemented":
                status, message = "skipped", aux["message"]
                break
            if aux["status"] == "invalid_update" or not torch.all(torch.isfinite(candidate)):
                status, message = "invalid_update", aux.get("message", "Non-finite update")
                break
            candidate_loss = parameter_to_loss(candidate, r, nu_hat, c, dims, **fsm_kwargs)
            candidate_output = simulate_wiener(r, nu, candidate, dims)
            if not torch.isfinite(candidate_loss) or not torch.all(torch.isfinite(candidate_output)):
                rows[-1]["rejected"] = 1.
                rows[-1]["applied_step_norm"] = 0.
                status, message = "invalid_update", "Candidate objective or reconstruction is non-finite."
                break
            change = float(torch.linalg.vector_norm(candidate-theta_hat)/(torch.linalg.vector_norm(theta_hat)+1e-15))
            theta_hat = candidate
            accepted += 1
            if innovation_mode == "state_residual":
                nu_hat = estimate_innovations_from_state(phi_hat, c, theta_hat)
                latent = update_latent_state_from_innovations(theta_hat, r, nu_hat, dims)
                alpha_hat, e_hat = latent["alpha_hat"], latent["e_hat"]
            small_step = change < conv_threshold
            if small_step:
                if "LGHAM" in method_name.upper():
                    target = float(method_cfg.get("resolved_epsilon0", method_cfg.get("eps_0", 0.)))
                    satisfied = abs(float(candidate_loss)-target) <= method_cfg["stopping"]["loss_tolerance"] * max(1., abs(target))
                else:
                    g = extract_loss_derivatives(theta_hat, r, nu_hat, c, dims, max_order=1, **fsm_kwargs)["grad"]
                    satisfied = float(torch.linalg.vector_norm(g)) <= method_cfg["stopping"]["gradient_tolerance"]
                status = "converged" if satisfied else "stalled"
                message = "Small step and residual tolerance satisfied." if satisfied else "Small step without satisfying the residual tolerance."
                stopping_iteration = k
            record(theta_hat, change=change)
            elapsed = max(0., now()-t0-step_diag_time)
            history["iter_time"][-1] = elapsed
            history["cum_time"][-1] += elapsed
            if progress_callback is not None:
                progress_callback({"phase": "iteration", "method": method_name, "iteration": k,
                                   "max_updates": K_max-1, "record_length": lambda_g,
                                   "run_metadata": run_metadata or {}})
            if small_step:
                break
    if r.is_cuda:
        torch.cuda.synchronize(r.device)
    wall_time = now()-start_time
    # Retain the last finite state on failure; never report its error as a successful final score.
    valid = status in {"converged", "max_iterations", "stalled"}
    actual_samples = len(theta_history)
    if enforce_fixed_iterations and valid:
        while len(theta_history) < K_max:
            theta_history.append(theta_history[-1].copy())
            for key in history:
                value = False if key == "active_mask" else (0. if key in {"iter_time", "rel_change"} else history[key][-1])
                history[key].append(value)
    packed = pack_diagnostics(rows)
    result = {"name": method_name, "status": status, "status_msg": message,
              "method_kind": "optimization", "is_bayesian": False, "point_estimate": "terminal_iterate",
              "theta_hat": theta_hat.detach().cpu().numpy(), "theta0": theta0,
              "theta_hist": np.asarray(theta_history).T,
              **{key: np.asarray(value) for key, value in history.items()},
              "final_err": history["param_err"][actual_samples-1] if valid else float("nan"),
              "total_time": max(0., wall_time-diagnostic_time), "total_wall_time": wall_time,
              "diagnostic_time": diagnostic_time, "iterations": len(theta_history),
              "update_count": updates, "accepted_updates": accepted, "stopping_iteration": stopping_iteration,
              "run_metadata": metadata, "resolved_config_json": config_json(method_cfg),
              "objective_definition": "mean_squared_innovation", "rmse_definition": "true_innovation_reconstruction",
              "diagnostics": packed, "diagnostic_summary": summarize_diagnostics(packed),
              "derivative_snapshots": snapshots}
    result["point_estimate_valid"] = valid
    result["diagnostics_passed"] = status == "converged"
    result["final_output_rmse"] = history["rmse_hist"][actual_samples-1] if valid else float("nan")
    if verbose:
        print(f"{method_name}: {status}; {accepted} accepted updates. {message}")
    return result


def load_experiment_config(experiment_name: str) -> dict[str, Any]:
    config_file = PY_ROOT.parent / "configs" / f"{experiment_name}.json"
    if not config_file.exists():
        config_file = PY_ROOT.parent / "configs" / f"{experiment_name}.JSON"
    if not config_file.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_file}")
    return load_json_config(config_file)


def build_dims_and_theta(cfg: dict[str, Any], *, dtype: torch.dtype, device: torch.device) -> tuple[WienerDimensions, torch.Tensor]:
    system_cfg = cfg["system"]
    dims = WienerDimensions(
        na=int(system_cfg["na"]),
        nb=int(system_cfg["nb"]),
        nf=int(system_cfg["nf"]),
        nd=int(system_cfg.get("nd", 0)),
    )
    theta_true = torch.tensor(system_cfg["theta_true"], dtype=dtype, device=device)
    if theta_true.numel() != dims.n_params:
        raise ValueError(f"theta_true has {theta_true.numel()} entries, but dims require {dims.n_params}.")
    return dims, theta_true


def select_python_methods(exp_cfg: dict[str, Any]) -> list[str]:
    selected_methods = list(exp_cfg.get("python_selected_methods", exp_cfg["selected_methods"]))
    for method_name in PYTHON_EXTRA_METHODS if exp_cfg.get("include_python_extra_methods", True) else []:
        if method_name not in selected_methods:
            selected_methods.append(method_name)
    return selected_methods


def pad_vector(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.full(length, np.nan)
    if values.size >= length:
        return values[:length]
    return np.pad(values, (0, length - values.size), mode="edge")


def run_method_set(
    selected_methods: list[str],
    method_configs: dict[str, Any],
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    theta_true: torch.Tensor,
    dims: WienerDimensions,
    lambda_g: int,
    K_max: int,
    conv_threshold: float,
    enforce_fixed_iterations: bool,
    *,
    verbose: bool,
    experiment_cfg: dict | None = None,
    run_metadata: dict | None = None,
    data_context: dict | None = None,
    progress_callback=None,
) -> list[dict[str, Any]]:
    results = []
    best_error_so_far = float("inf")
    for method_name in selected_methods:
        method_cfg = resolve_method_config(method_configs.get(make_valid_method_key(method_name), {}), experiment_cfg)
        is_bayesian = method_name.upper() == "BI_HMC"
        if is_bayesian:
            from bayesian_hmc import resolve_hmc_config
            method_cfg = resolve_hmc_config(method_cfg)
        progress_event = {"method": method_name, "record_length": lambda_g,
                          "max_updates": K_max-1, "run_metadata": run_metadata or {},
                          "method_kind": "bayesian" if is_bayesian else "optimization"}
        if is_bayesian:
            progress_event["max_transitions"] = int(method_cfg.get("chains", 4))*(int(method_cfg.get("warmup", 500))+int(method_cfg.get("samples", 1000)))
        if progress_callback is not None:
            progress_callback({**progress_event, "phase": "method_start"})
        if "LGHAM" in method_name.upper() and np.isfinite(best_error_so_far):
            method_cfg["best_error_so_far"] = best_error_so_far
        if verbose:
            print("\n" + "=" * 72)
            print(f"Running method: {method_name}")
            print("=" * 72)
        method_result = run_identification_method(
            method_name,
            method_cfg,
            r,
            nu,
            c,
            theta_true,
            dims,
            lambda_g,
            K_max,
            conv_threshold,
            enforce_fixed_iterations=enforce_fixed_iterations,
            verbose=verbose,
            run_metadata=run_metadata,
            data_context=data_context,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback({**progress_event, "phase": "method_end",
                               "iteration": method_result.get("sampling_transitions", method_result["accepted_updates"]),
                               "status": method_result["status"]})
        results.append(method_result)
        loss_curve = method_result.get("est_loss_hist", np.array([]))
        if "LGHAM" not in method_name.upper() and method_result["status"] in {"converged", "max_iterations", "stalled"} and loss_curve.size:
            candidate_error = float(loss_curve[-1])
            if np.isfinite(candidate_error) and candidate_error < best_error_so_far:
                best_error_so_far = candidate_error
    return results


def summarize_montecarlo(mc_runs: list[list[dict[str, Any]]], selected_methods: list[str], theta_true: torch.Tensor, K_max: int) -> dict[str, Any]:
    theta_true_np = theta_true.detach().cpu().numpy().reshape(-1)
    n_methods = len(selected_methods)
    n_mc = len(mc_runs)
    n_params = theta_true_np.size
    param_err_runs = np.full((n_methods, n_mc, K_max), np.nan)
    rmse_runs = np.full((n_methods, n_mc, K_max), np.nan)
    final_param_errors = np.full((n_methods, n_mc, n_params), np.nan)
    final_param_estimates = np.full((n_methods, n_mc, n_params), np.nan)
    final_errors = np.full((n_methods, n_mc), np.nan)
    total_time_runs = np.full((n_methods, n_mc), np.nan)
    total_wall_time_runs = np.full((n_methods, n_mc), np.nan)
    final_output_rmse_runs = np.full((n_methods, n_mc), np.nan)
    diagnostics_passed = np.zeros((n_methods, n_mc), dtype=bool)
    point_estimate_valid = np.zeros((n_methods, n_mc), dtype=bool)
    statuses = np.full((n_methods, n_mc), "missing", dtype=object)
    run_ids = np.full((n_methods, n_mc), "", dtype=object)

    for run_idx, run_results in enumerate(mc_runs):
        by_name = {result["name"]: result for result in run_results}
        for method_idx, method_name in enumerate(selected_methods):
            result = by_name[method_name]
            statuses[method_idx, run_idx] = result["status"]
            run_ids[method_idx, run_idx] = result["run_metadata"]["run_id"]
            if result["status"] in {"converged", "max_iterations", "stalled"}:
                param_err_runs[method_idx, run_idx, :] = pad_vector(result["param_err"], K_max)
                rmse_runs[method_idx, run_idx, :] = pad_vector(result["rmse_hist"], K_max)
            final_errors[method_idx, run_idx] = result["final_err"]
            total_time_runs[method_idx, run_idx] = result["total_time"]
            total_wall_time_runs[method_idx, run_idx] = result.get("total_wall_time", result["total_time"])
            final_output_rmse_runs[method_idx, run_idx] = result.get("final_output_rmse", np.nan)
            diagnostics_passed[method_idx, run_idx] = result.get("diagnostics_passed", result["status"] == "converged")
            valid = result["status"] in FINITE_RESULT_STATUSES and result.get("point_estimate_valid", True)
            point_estimate_valid[method_idx, run_idx] = valid
            if valid and result["theta_hat"].size:
                theta_hat = np.asarray(result["theta_hat"], dtype=float).reshape(-1)
                final_param_estimates[method_idx, run_idx, :] = theta_hat
                final_param_errors[method_idx, run_idx, :] = theta_hat - theta_true_np

    def moments(values):
        finite = np.isfinite(values)
        count = finite.sum(axis=1)
        mean = np.divide(np.where(finite, values, 0.).sum(axis=1), count,
                         out=np.full(count.shape, np.nan), where=count > 0)
        squares = np.where(finite, values-mean[:, None, :], 0.)**2
        variance = np.divide(squares.sum(axis=1), count-1,
                             out=np.full(count.shape, np.nan), where=count > 1)
        stderr = np.sqrt(variance/np.maximum(count, 1))
        return mean, stderr, count

    param_mean, param_stderr, param_count = moments(param_err_runs)
    rmse_mean, rmse_stderr, rmse_count = moments(rmse_runs)
    return {
        "statuses": statuses,
        "run_ids": run_ids,
        "method_names": np.asarray(selected_methods, dtype=object).reshape(1, -1),
        "param_err_runs": param_err_runs,
        "param_err_mean": param_mean,
        "param_err_stderr": param_stderr,
        "param_err_count": param_count,
        "rmse_runs": rmse_runs,
        "rmse_mean": rmse_mean,
        "rmse_stderr": rmse_stderr,
        "rmse_count": rmse_count,
        "final_param_errors": final_param_errors,
        "final_param_estimates": final_param_estimates,
        "final_errors": final_errors,
        "total_time_runs": total_time_runs,
        "total_wall_time_runs": total_wall_time_runs,
        "final_output_rmse_runs": final_output_rmse_runs,
        "diagnostics_passed": diagnostics_passed,
        "point_estimate_valid": point_estimate_valid,
    }


def main() -> None:
    experiment_name = "example_CSTR"

    device = default_device()
    dtype = choose_dtype()
    exp_cfg = load_experiment_config(experiment_name)
    dims, theta_true = build_dims_and_theta(exp_cfg, dtype=dtype, device=device)

    mode = exp_cfg.get("mode", "EXAMPLE").upper()
    seed = int(exp_cfg.get("seed", 42))
    n_mc = int(exp_cfg.get("N_MC", 1))
    selected_methods = select_python_methods(exp_cfg)
    settings = exp_cfg["settings"]
    lambda_g = int(settings["lambda_g"])
    K_max = int(settings["K_max"])
    conv_threshold = float(settings["conv_threshold"])
    sigma_nu = float(settings["sigma_nu"])
    burn_in = int(settings["burn_in"])
    enforce_fixed_iterations = bool(settings.get("enforce_fixed_iterations", mode == "MONTECARLO"))
    make_local_plots = bool(exp_cfg.get("make_local_plots", mode == "EXAMPLE"))
    config_file = PY_ROOT / exp_cfg.get("python_optim_config", "optim_configs.json")
    method_configs = load_json_config(config_file)["methods"]
    results_dir = PY_ROOT.parent / "results" / experiment_name
    results_file = results_dir / f"comparison_fsm_{mode.lower()}.mat"

    set_seed(seed)
    data = generate_example_data(dims, theta_true, lambda_g, burn_in, sigma_nu, device=device, dtype=dtype)
    r, nu, c = data["r"], data["nu"], data["c"]
    results = run_method_set(
        selected_methods,
        method_configs,
        r,
        nu,
        c,
        theta_true,
        dims,
        lambda_g,
        K_max,
        conv_threshold,
        enforce_fixed_iterations,
        verbose=True,
        experiment_cfg=exp_cfg, run_metadata={"data_seed": seed, "mc_index": 1, "experiment_name": experiment_name},
        data_context=inference_data_context(data, burn_in),
    )

    mc_summary = None
    mc_runs = [results]
    if mode == "MONTECARLO":
        mc_runs = [results]
        for mc_idx in range(1, n_mc):
            run_seed = seed + mc_idx
            print(f"\nMonte Carlo run {mc_idx + 1}/{n_mc} (seed {run_seed})")
            set_seed(run_seed)
            data_i = generate_example_data(dims, theta_true, lambda_g, burn_in, sigma_nu, device=device, dtype=dtype)
            mc_runs.append(
                run_method_set(
                    selected_methods,
                    method_configs,
                    data_i["r"],
                    data_i["nu"],
                    data_i["c"],
                    theta_true,
                    dims,
                    lambda_g,
                    K_max,
                    conv_threshold,
                    enforce_fixed_iterations,
                    verbose=False,
                    experiment_cfg=exp_cfg, run_metadata={"data_seed": run_seed, "mc_index": mc_idx+1, "experiment_name": experiment_name},
                    data_context=inference_data_context(data_i, burn_in),
                )
            )
        mc_summary = summarize_montecarlo(mc_runs, selected_methods, theta_true, K_max)

    print("\n" + "=" * 92)
    print("FINAL METHOD SUMMARY")
    print("=" * 92)
    print(f"{'Method':<14} {'Status':<18} {'Final err(%)':<12} {'Final d1':<12} {'Wall time (s)':<12}")
    print("-" * 92)
    for result in results:
        final_d = float("nan")
        if result["theta_hat"].size:
            final_d = result["theta_hat"][-1]
        print(f"{result['name']:<14} {result['status']:<18} {result['final_err']:<12.5f} {final_d:<12.5f} {result['total_wall_time']:<12.5f}")

    save_comparison_results(
        results_file,
        results,
        experiment_name=experiment_name,
        mode=mode,
        selected_methods=selected_methods,
        dims=dims,
        theta_true=theta_true,
        lambda_g=lambda_g,
        K_max=K_max,
        conv_threshold=conv_threshold,
        sigma_nu=sigma_nu,
        burn_in=burn_in,
        enforce_fixed_iterations=enforce_fixed_iterations,
        seed=seed,
        n_mc=n_mc,
        example_data={
            "r": r.detach().cpu().numpy().reshape(-1, 1),
            "nu": nu.detach().cpu().numpy().reshape(-1, 1),
            "c": c.detach().cpu().numpy().reshape(-1, 1),
            "input_prehistory": data["r_full"][:burn_in].detach().cpu().numpy(),
            "noise_burn_in": burn_in,
        },
        mc_summary=mc_summary, all_runs=mc_runs, resolved_experiment=exp_cfg,
    )

    if make_local_plots:
        plot_method_metrics(results)
        plot_parameter_trajectories(results, theta_true.detach().cpu().numpy(), build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd))
        bayesian = [item for item in results if item.get("method_kind") == "bayesian"]
        if bayesian:
            from bayesian_plots import plot_bayesian_posteriors, plot_bayesian_outputs, plot_final_error_runtime
            for item in bayesian:
                plot_bayesian_posteriors(item, theta_true.detach().cpu().numpy(),
                                        build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd))
                plot_bayesian_outputs(item, c.detach().cpu().numpy())
            plot_final_error_runtime(results)

        import matplotlib.pyplot as plt

        plt.show()
        print("\nDone. All figures generated.")
    else:
        print("\nDone. Results saved for MATLAB plotting.")


if __name__ == "__main__":
    main()
