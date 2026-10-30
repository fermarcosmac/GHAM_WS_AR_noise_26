"""Initialization, noise, and record-length studies for Python identification methods."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from scipy.signal import lfilter

from common import build_param_names, build_plot_labels, default_device, load_json_config
from comparison_torch import (build_dims_and_theta, load_experiment_config, run_method_set,
                              save_comparison_results, select_python_methods, summarize_montecarlo)
from diagnostics import config_json
from study_progress import StudyProgress

ROOT = Path(__file__).resolve().parent.parent
FINITE_STATUSES = {"converged", "stalled", "max_iterations", "ok",
                   "sampling_complete", "sampling_warning"}
DIAGNOSTIC_PASS_STATUSES = {"converged", "sampling_complete"}
INITIAL_CONDITION_POLICIES = {
    "optimization": "zero_state_fit_after_burn_in",
    "BI_HMC": "known_zero_signal_start_before_observed_input_prehistory; finite_noise_start_covariance",
}


def positive_integer(value, name):
    number = int(value)
    if number < 1 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def nominal_theta(dims):
    theta = np.zeros(dims.n_params)
    if dims.nb:
        theta[dims.na] = 1.
    return theta


def parameter_vector(value, dims, name, default=None):
    values = np.asarray(default if value is None else value, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(dims.n_params, values.item())
    if values.size != dims.n_params or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {dims.n_params} finite entries")
    return values


def stable(theta, dims, radius=1.):
    blocks = (theta[:dims.na], theta[-dims.nd:] if dims.nd else [])
    return all(np.all(np.abs(np.roots(np.r_[1., block])) < radius) for block in blocks)


def sample_initializations(dims, study):
    count = positive_integer(study["n_runs"], "n_runs")
    center = parameter_vector(study.get("center"), dims, "center", nominal_theta(dims))
    scale = parameter_vector(study.get("scale", .5), dims, "scale")
    radius = float(study.get("stability_radius", .95))
    if np.any(scale < 0) or not np.any(scale > 0) or not 0 < radius < 1:
        raise ValueError("scale must be nonnegative and nonzero; stability_radius must lie in (0, 1)")
    distribution = study.get("distribution", "uniform")
    if distribution not in {"uniform", "normal"}:
        raise ValueError("distribution must be uniform or normal")
    rng = np.random.default_rng(int(study.get("initialization_seed", 314159)))
    draws = []
    for _ in range(positive_integer(study.get("max_draws", 100000), "max_draws")):
        z = rng.uniform(-1, 1, dims.n_params) if distribution == "uniform" else rng.standard_normal(dims.n_params)
        candidate = center + scale*z
        if stable(candidate, dims, radius):
            draws.append(candidate)
            if len(draws) == count:
                return np.asarray(draws).T
    raise ValueError("Could not draw enough stable starts; change center/scale or increase max_draws")


def reference_data(cfg, dims, theta, length, seed, device):
    """Recursive IIR truth, independent of the estimator's finite frequency grid."""
    burn = int(cfg["settings"]["burn_in"])
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(length+burn)
    nu = float(cfg["settings"]["sigma_nu"])*rng.standard_normal(length+burn)
    th = theta.detach().cpu().numpy()
    alpha = lfilter(np.r_[0., th[dims.na:dims.na+dims.nb]], np.r_[1., th[:dims.na]], u)
    beta = alpha.copy()
    for power, coefficient in enumerate(th[dims.na+dims.nb:dims.na+dims.nb+dims.nf-1], 2):
        beta += coefficient*alpha**power
    d = th[-dims.nd:] if dims.nd else np.empty(0)
    c = beta+lfilter([1.], np.r_[1., d], nu)
    data = {key: torch.as_tensor(values[burn:].copy(), dtype=torch.float64, device=device)
            for key, values in (("r", u), ("nu", nu), ("c", c))}
    # Input prehistory is observed experimental information, not latent truth.
    # HMC replays it using each candidate's coefficients from the known zero
    # signal start; it never receives the true internal state or innovations.
    data["input_prehistory"] = torch.as_tensor(u[:burn].copy(), dtype=torch.float64, device=device)
    data["noise_burn_in"] = burn
    return data


def prefix_data(data, length):
    """Shorten the measured record without shortening its known prehistory."""
    return {key: value[:length] if key in {"r", "nu", "c"} else value
            for key, value in data.items()}


def matlab_data(data):
    return {key: value.detach().cpu().numpy().reshape(-1, 1)
            if isinstance(value, torch.Tensor) else value for key, value in data.items()}


def _setup(cfg, study, device):
    cfg = copy.deepcopy(cfg)
    if "methods" in study:
        methods = study["methods"]
        if not isinstance(methods, list) or not methods or any(not isinstance(m, str) for m in methods):
            raise ValueError("methods must be a nonempty list of method names")
        if len(set(methods)) != len(methods):
            raise ValueError("methods must not contain duplicate names")
        cfg["python_selected_methods"] = list(methods)
        cfg["include_python_extra_methods"] = False
    if "K_max" in study:
        cfg["settings"]["K_max"] = positive_integer(study["K_max"], "K_max")
    cfg["settings"]["enforce_fixed_iterations"] = False
    dims, theta = build_dims_and_theta(cfg, dtype=torch.float64, device=device)
    methods = select_python_methods(cfg)
    configs = load_json_config(ROOT/"python"/cfg.get("python_optim_config", "optim_configs.json"))["methods"]
    if "bayesian_hmc" in study:
        overrides = study["bayesian_hmc"]
        if not isinstance(overrides, dict) or "hmc" in overrides:
            raise ValueError("bayesian_hmc must be a flat object of HMC setting overrides")
        base = copy.deepcopy(configs.get("BI_HMC", {}))
        nested = base.pop("hmc", {})
        configs["BI_HMC"] = {**base, **nested, **copy.deepcopy(overrides)}
    return cfg, dims, theta, methods, configs


def _run(cfg, methods, configs, dims, theta, data, metadata, verbose, progress):
    settings = cfg["settings"]
    return run_method_set(methods, configs, data["r"], data["nu"], data["c"], theta, dims,
                          len(data["r"]), int(settings["K_max"]), float(settings["conv_threshold"]),
                          False, verbose=verbose, experiment_cfg=cfg, run_metadata=metadata,
                          progress_callback=progress if progress.enabled else None,
                          data_context={"input_prehistory": data.get("input_prehistory"),
                                        "noise_burn_in": data.get("noise_burn_in")})


def _save_case(path, cfg, study, methods, dims, theta, data, runs, name, mode):
    settings = cfg["settings"]
    summary = summarize_montecarlo(runs, methods, theta, int(settings["K_max"]))
    study = {**study, "initial_condition_policies": INITIAL_CONDITION_POLICIES,
             "sampler_seed_policy": "method_seed_plus_deterministic_repetition_metadata; independent_data_rng"}
    save_comparison_results(Path(path), runs[0], experiment_name=name, mode=mode,
        selected_methods=methods, dims=dims, theta_true=theta, lambda_g=len(data["r"]),
        K_max=int(settings["K_max"]), conv_threshold=float(settings["conv_threshold"]),
        sigma_nu=float(settings["sigma_nu"]), burn_in=int(settings["burn_in"]),
        enforce_fixed_iterations=False, seed=int(study["data_seed"]), n_mc=len(runs),
        example_data=matlab_data(data),
        mc_summary=summary, all_runs=runs, resolved_experiment=cfg, study=study)


def run_initialization_study(cfg, study, *, experiment_name="example_CSTR", output_file=None,
                             device=torch.device("cpu"), verbose=False):
    cfg, dims, theta, methods, configs = _setup(cfg, study, device)
    study = copy.deepcopy(study)
    study.update(type="initialization", randomized_variable="parameter_initialization",
                 data_seed=int(study.get("data_seed", cfg.get("seed", 42))))
    study["initialization_seed"] = int(study.get("initialization_seed", 314159))
    study["center"] = parameter_vector(study.get("center"), dims, "center", nominal_theta(dims)).tolist()
    study["scale"] = parameter_vector(study.get("scale", .5), dims, "scale").tolist()
    progress = StudyProgress("FSM initialization", methods, positive_integer(study["n_runs"], "n_runs"),
                             [int(cfg["settings"]["lambda_g"])], 1, study)
    progress.stage("Preparing stable starts and the fixed data record")
    starts = sample_initializations(dims, study)
    data = reference_data(cfg, dims, theta, int(cfg["settings"]["lambda_g"]), study["data_seed"], device)
    runs = []
    for i, start in enumerate(starts.T):
        cfg["initialization"] = {"policy": "consistent", "theta0": start.tolist()}
        runs.append(_run(cfg, methods, configs, dims, theta, data,
            {"study_type": "initialization", "initialization_index": i+1,
             "initialization_seed": int(study["initialization_seed"]), "data_seed": study["data_seed"]}, verbose, progress))
    study["initializations"] = starts
    study.update(terminal_metadata(runs, methods))
    path = Path(output_file or ROOT/"results"/experiment_name/"initialization_fsm.mat")
    progress.stage("Summarizing and saving initialization results")
    _save_case(path, cfg, study, methods, dims, theta, data, runs, experiment_name, "INITIALIZATION")
    progress.finish(path)
    return path


def terminal_arrays(runs, methods, n_params):
    estimates = np.full((len(methods), len(runs), n_params), np.nan)
    statuses = np.empty((len(methods), len(runs)), dtype=object)
    configs = np.empty(statuses.shape, dtype=object)
    ids = np.empty(statuses.shape, dtype=object)
    for r, results in enumerate(runs):
        by_name = {result["name"]: result for result in results}
        for m, name in enumerate(methods):
            result = by_name[name]
            statuses[m, r] = result["status"]
            configs[m, r] = result["resolved_config_json"]
            ids[m, r] = result["run_metadata"]["dataset_id"]
            values = np.asarray(result["theta_hat"]).reshape(-1)
            if (result["status"] in FINITE_STATUSES and result.get("point_estimate_valid", True)
                    and np.isfinite(values).all()):
                estimates[m, r] = values
    return estimates, statuses, configs, ids


def terminal_metadata(runs, methods):
    """Keep sampler validity and total runtime separate from optimizer updates."""
    shape = (len(methods), len(runs))
    wall_time = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    bayesian = np.zeros(shape, dtype=bool)
    passed = np.zeros(shape, dtype=bool)
    seeds = np.full(shape, np.nan)
    diagnostics = np.full(shape, "", dtype=object)
    for r, results in enumerate(runs):
        by_name = {result["name"]: result for result in results}
        for m, name in enumerate(methods):
            result = by_name[name]
            wall_time[m, r] = float(result.get("total_wall_time", result.get("total_time", np.nan)))
            bayesian[m, r] = bool(result.get("is_bayesian", result.get("method_kind") == "bayesian"))
            values = np.asarray(result["theta_hat"]).reshape(-1)
            valid[m, r] = (result["status"] in FINITE_STATUSES
                           and result.get("point_estimate_valid", True)
                           and np.isfinite(values).all())
            if bayesian[m, r]:
                passed[m, r] = bool(result.get("diagnostics_passed", result["status"] == "sampling_complete"))
                metadata = result.get("run_metadata", {})
                seed = result.get("sampler_seed", metadata.get("sampler_seed"))
                if seed is not None:
                    seeds[m, r] = float(seed)
                diagnostic_values = result.get("sampler_diagnostics", result.get("posterior_diagnostics", {}))
                diagnostics[m, r] = config_json(json_compatible(diagnostic_values))
    return {"total_wall_time": wall_time, "total_time_runs": wall_time.copy(),
            "point_estimate_valid": valid, "is_bayesian": bayesian,
            "sampler_diagnostics_passed": passed, "sampling_seeds": seeds,
            "sampler_diagnostics_json": diagnostics}


def json_compatible(value):
    """Represent undefined sampler diagnostics as JSON null, not invalid NaN."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def bias_statistics(estimates, statuses, theta_true, population="finite"):
    """MC intervals describe uncertainty of the mean error, not parameter intervals."""
    if population not in {"finite", "converged"}:
        raise ValueError("summary_population must be finite or converged")
    values = np.asarray(estimates, dtype=float).copy()
    finite_runs = np.isfinite(values).all(axis=2)
    diagnostic_pass = np.isin(statuses, list(DIAGNOSTIC_PASS_STATUSES))
    selected = finite_runs & (diagnostic_pass if population == "converged" else True)
    values[~selected] = np.nan
    errors = values-np.asarray(theta_true).reshape(1, 1, -1)
    count = np.isfinite(errors).sum(axis=1)
    mean = np.divide(np.nansum(errors, axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)
    centered = np.where(np.isfinite(errors), errors-mean[:, None, :], 0.)
    variance = np.divide(np.sum(centered**2, axis=1), count-1,
                         out=np.full(mean.shape, np.nan), where=count > 1)
    mcse = np.sqrt(variance/np.maximum(count, 1))
    mse = np.divide(np.nansum(errors**2, axis=1), count, out=np.full(mean.shape, np.nan), where=count > 0)
    return {"bias": mean, "variance": variance, "rmse": np.sqrt(mse), "mcse": mcse,
            "ci_low": mean-1.96*mcse, "ci_high": mean+1.96*mcse, "count": count,
            "finite_count": finite_runs.sum(axis=1), "selected_count": selected.sum(axis=1),
            "converged_count": diagnostic_pass.sum(axis=1),
            "sampling_complete_count": (statuses == "sampling_complete").sum(axis=1),
            "sampling_warning_count": (statuses == "sampling_warning").sum(axis=1),
            "skipped_count": (statuses == "skipped").sum(axis=1),
            "failed_count": (~np.isin(statuses, list(FINITE_STATUSES)+["skipped"])).sum(axis=1)}


def run_noise_study(cfg, study, *, experiment_name="example_CSTR", output_file=None,
                    device=torch.device("cpu"), verbose=False):
    cfg, dims, theta, methods, configs = _setup(cfg, study, device)
    study = copy.deepcopy(study)
    count = positive_integer(study["n_runs"], "n_runs")
    levels = np.asarray(study["sigma_nu_values"], dtype=float)
    if (levels.ndim != 1 or not levels.size or not np.isfinite(levels).all()
            or np.any(levels < 0) or np.any(np.diff(levels) <= 0)):
        raise ValueError("sigma_nu_values must be nonempty, finite, nonnegative, unique, and increasing")
    start = parameter_vector(study.get("theta0"), dims, "theta0", nominal_theta(dims))
    if not stable(start, dims):
        raise ValueError("theta0 denominators must be stable")
    population = study.get("summary_population", "finite")
    if population not in {"finite", "converged"}:
        raise ValueError("summary_population must be finite or converged")
    seed = int(study.get("data_seed", cfg.get("seed", 42)))
    length = positive_integer(cfg["settings"]["lambda_g"], "lambda_g")
    cfg.update(mode="MONTECARLO", seed=seed, N_MC=count, make_local_plots=False)
    cfg["initialization"] = {"policy": "consistent", "theta0": start.tolist()}
    study.update(type="noise", varied_variable="innovation_standard_deviation",
                 data_seed=seed, theta0=start.tolist(), sigma_nu_values=levels.tolist(),
                 summary_population=population)
    path = Path(output_file or ROOT/"results"/experiment_name/"noise_fsm.mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(levels), len(methods), count)
    estimates = np.full(shape+(dims.n_params,), np.nan)
    statuses = np.full(shape, "not_run", dtype=object)
    resolved_configs = np.full(shape, "", dtype=object)
    dataset_ids = np.full(shape, "", dtype=object)
    case_files = np.full(len(levels), "", dtype=object)
    summaries, metadata_cases = [], []
    progress = StudyProgress("FSM noise", methods, count, [length]*len(levels), 1, study)
    for j, sigma in enumerate(levels):
        cfg["settings"]["sigma_nu"] = float(sigma)
        progress.stage(f"Noise level {j+1}/{len(levels)}: sigma_nu={sigma:g}")
        runs = []
        first_data = None
        for r in range(count):
            # Reset the same stream at each level: only innovation amplitude changes,
            # including the pre-burn-in samples used by the recursive noise filter.
            data = reference_data(cfg, dims, theta, length, seed+r, device)
            if first_data is None:
                first_data = data
            runs.append(_run(cfg, methods, configs, dims, theta, data,
                {"study_type": "noise", "noise_level_index": j+1, "sigma_nu": float(sigma),
                 "mc_index": r+1, "data_seed": seed+r}, verbose, progress))
        arrays = terminal_arrays(runs, methods, dims.n_params)
        estimates[j], statuses[j], resolved_configs[j], dataset_ids[j] = arrays
        metadata_cases.append(terminal_metadata(runs, methods))
        summaries.append(bias_statistics(arrays[0], arrays[1], theta.cpu().numpy(), population))
        if study.get("save_case_results", True):
            case_file = path.parent/(path.stem+f"_sigma{j+1:02d}.mat")
            case_study = {**study, "type": "noise_case", "sigma_nu": float(sigma), "noise_level_index": j+1}
            _save_case(case_file, cfg, case_study, methods, dims, theta, first_data,
                       runs, experiment_name, "MONTECARLO")
            case_files[j] = str(case_file.resolve())
    statistics = {key: np.asarray([cell[key] for cell in summaries]) for key in summaries[0]}
    run_details = {key: np.stack([cell[key] for cell in metadata_cases]) for key in metadata_cases[0]}
    saved = {"type": "noise", "varied_variable": "innovation_standard_deviation",
             "sigma_nu_values": levels, "innovation_variances": levels**2,
             "method_names": np.asarray(methods, dtype=object), "n_runs": count,
             "record_length": length, "theta_true": theta.cpu().numpy(), "theta0": start,
             "param_labels": np.asarray(build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd), dtype=object),
             "param_names": np.asarray(build_param_names(dims.na, dims.nb, dims.nf, dims.nd), dtype=object),
             "final_param_estimates": estimates, "statuses": statuses, "dataset_ids": dataset_ids,
             "resolved_configs": resolved_configs, "summary_population": population,
             "data_seeds": seed+np.arange(count), "case_files": case_files,
             "data_pairing": "matched_input_and_standard_innovations",
             "config_json": config_json(study), "initial_condition_policy": "method_specific",
             "initial_condition_policies": INITIAL_CONDITION_POLICIES,
             "sampler_seed_policy": "method_seed_plus_deterministic_repetition_metadata; independent_data_rng",
             "posterior_storage": "full_draws_in_case_files_when_save_case_results_is_true",
             **statistics, **run_details}
    progress.stage("Saving noise study results")
    savemat(path, {"schema_version": 2, "parametrization": "FSM", "study": saved,
                   "experiment_name": experiment_name, "resolved_experiment_json": config_json(cfg)}, long_field_names=True)
    progress.finish(path)
    return path


def run_record_length_study(cfg, study, *, experiment_name="example_CSTR", output_file=None,
                           device=torch.device("cpu"), verbose=False):
    cfg, dims, theta, methods, configs = _setup(cfg, study, device)
    study = copy.deepcopy(study)
    count = positive_integer(study["n_runs"], "n_runs")
    lengths = [positive_integer(n, "record_lengths") for n in study["record_lengths"]]
    if not lengths or lengths != sorted(set(lengths)):
        raise ValueError("record_lengths must be nonempty, unique, and increasing")
    resolutions = study.get("fsm_resolutions", [{"label": "configured"}])
    if not resolutions or len({r["label"] for r in resolutions}) != len(resolutions):
        raise ValueError("FSM resolution labels must be nonempty and unique")
    # Avoid changing the FIR length implicitly when a shorter record is fitted.
    for resolution in resolutions:
        if (any(method != "BI_HMC" for method in methods)
                and max(resolution.get("fsm_tau_g", 1), resolution.get("fsm_tau_h", 1)) > lengths[0]):
            raise ValueError("Every record must be at least as long as the configured FSM responses")
    start = parameter_vector(study.get("theta0"), dims, "theta0", nominal_theta(dims))
    if not stable(start, dims):
        raise ValueError("theta0 denominators must be stable")
    cfg["initialization"] = {"policy": "consistent", "theta0": start.tolist()}
    seed = int(study.get("data_seed", 271828))
    population = study.get("summary_population", "finite")
    if population not in {"finite", "converged"}:
        raise ValueError("summary_population must be finite or converged")
    path = Path(output_file or ROOT/"results"/experiment_name/"record_length_fsm.mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(resolutions), len(lengths), len(methods), count)
    estimates = np.full(shape+(dims.n_params,), np.nan)
    statuses = np.full(shape, "not_run", dtype=object)
    resolved_configs = np.full(shape, "", dtype=object)
    dataset_ids = np.full(shape, "", dtype=object)
    case_files = np.full((len(resolutions), len(lengths)), "", dtype=object)
    # Draw each full record once; all lengths and resolutions share its prefixes.
    progress = StudyProgress("FSM record_length", methods, count, lengths, len(resolutions), study)
    progress.stage(f"Preparing {count} independent records of N={lengths[-1]}")
    full_records = []
    for r in range(count):
        full_records.append(reference_data(cfg, dims, theta, lengths[-1], seed+r, device))
        progress.stage(f"Prepared data record {r+1}/{count}", force=(r+1 == count))
    summaries, metadata_cases = [], []
    for k, resolution in enumerate(resolutions):
        case_cfgs = copy.deepcopy(configs)
        for method_cfg in case_cfgs.values():
            method_cfg.update({key: value for key, value in resolution.items() if key != "label"})
        case_summaries, case_details = [], []
        for j, length in enumerate(lengths):
            runs = []
            for r, full in enumerate(full_records):
                data = prefix_data(full, length)
                runs.append(_run(cfg, methods, case_cfgs, dims, theta, data,
                    {"study_type": "record_length", "record_length": length, "mc_index": r+1,
                     "data_seed": seed+r, "resolution": resolution["label"]}, verbose, progress))
            progress.stage(f"Summarizing {resolution['label']}, N={length}")
            arrays = terminal_arrays(runs, methods, dims.n_params)
            estimates[k, j], statuses[k, j], resolved_configs[k, j], dataset_ids[k, j] = arrays
            case_details.append(terminal_metadata(runs, methods))
            case_summaries.append(bias_statistics(arrays[0], arrays[1], theta.cpu().numpy(), population))
            if study.get("save_case_results", False):
                case_file = path.parent/(path.stem+f"_{k+1}_n{length}.mat")
                case_study = {**study, "type": "record_length_case", "data_seed": seed,
                              "record_length": length, "resolution": resolution["label"]}
                _save_case(case_file, cfg, case_study, methods,
                           dims, theta, prefix_data(full_records[0], length),
                           runs, experiment_name, "MONTECARLO")
                case_files[k, j] = str(case_file.resolve())
        summaries.append(case_summaries)
        metadata_cases.append(case_details)
    statistics = {key: np.asarray([[cell[key] for cell in row] for row in summaries]) for key in summaries[0][0]}
    run_details = {key: np.stack([np.stack([cell[key] for cell in row]) for row in metadata_cases])
                   for key in metadata_cases[0][0]}
    # Descriptive b(N)=intercept+slope/N fit; no independent-length inference is claimed.
    coefficients = np.full((len(resolutions), len(methods), dims.n_params, 2), np.nan)
    for k in range(len(resolutions)):
        for m in range(len(methods)):
            for p in range(dims.n_params):
                y = statistics["bias"][k, :, m, p]
                keep = np.isfinite(y)
                if keep.sum() >= 3:
                    coefficients[k, m, p] = np.linalg.lstsq(
                        np.column_stack((np.ones(keep.sum()), 1/np.asarray(lengths)[keep])), y[keep], rcond=None)[0]
    saved = {"type": "record_length", "record_lengths": np.asarray(lengths),
             "resolution_labels": np.asarray([r["label"] for r in resolutions], dtype=object),
             "method_names": np.asarray(methods, dtype=object), "n_runs": count,
             "theta_true": theta.cpu().numpy(), "theta0": start,
             "param_labels": np.asarray(build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd), dtype=object),
             "param_names": np.asarray(build_param_names(dims.na, dims.nb, dims.nf, dims.nd), dtype=object),
             "final_param_estimates": estimates, "statuses": statuses, "dataset_ids": dataset_ids,
             "resolved_configs": resolved_configs, "summary_population": population,
             "data_seeds": seed+np.arange(count), "inverse_n_coefficients": coefficients,
             "case_files": case_files, "config_json": config_json(study),
             "initial_condition_policy": "method_specific",
             "initial_condition_policies": INITIAL_CONDITION_POLICIES,
             "sampler_seed_policy": "method_seed_plus_deterministic_repetition_metadata; independent_data_rng",
             "posterior_storage": "full_draws_in_case_files_when_save_case_results_is_true",
             **statistics, **run_details}
    progress.stage("Saving record-length results")
    savemat(path, {"schema_version": 2, "parametrization": "FSM", "study": saved,
                   "experiment_name": experiment_name, "resolved_experiment_json": config_json(cfg)}, long_field_names=True)
    progress.finish(path)
    return path


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="example_1")
    parser.add_argument("--study-config", type=Path, default=ROOT/"configs"/"studies.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        help="Innovation standard deviations (noise study only)")
    parser.add_argument("--methods", nargs="+", help="Exact Python method set, including BI_HMC when requested")
    parser.add_argument("--hmc-config", type=Path,
                        help="Flat JSON object overriding this study's BI_HMC settings")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable study progress messages")
    args = parser.parse_args()
    study = load_json_config(args.study_config)[kind+"_study"]
    if args.no_progress:
        study["progress"] = False
    if args.methods is not None:
        study["methods"] = args.methods
    if args.hmc_config is not None:
        overrides = load_json_config(args.hmc_config)
        if not isinstance(overrides, dict) or "hmc" in overrides:
            parser.error("--hmc-config must contain a flat JSON object of HMC settings")
        existing = study.get("bayesian_hmc", {})
        if not isinstance(existing, dict):
            parser.error("The study's bayesian_hmc settings must be a JSON object")
        study["bayesian_hmc"] = {**existing, **overrides}
    if args.runs is not None:
        study["n_runs"] = args.runs
    if args.iterations is not None:
        study["K_max"] = args.iterations
    if args.lengths is not None:
        if kind != "record_length":
            parser.error("--lengths is only available for the record-length study")
        study["record_lengths"] = args.lengths
    if args.noise_levels is not None:
        if kind != "noise":
            parser.error("--noise-levels is only available for the noise study")
        study["sigma_nu_values"] = args.noise_levels
    runner = {"initialization": run_initialization_study, "noise": run_noise_study,
              "record_length": run_record_length_study}[kind]
    runner(load_experiment_config(args.experiment), study, experiment_name=args.experiment,
           output_file=args.output, device=torch.device("cpu") if args.cpu else default_device(), verbose=args.verbose)
