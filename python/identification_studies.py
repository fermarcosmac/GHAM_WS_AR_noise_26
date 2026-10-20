"""Initialization sensitivity and record-length studies using the existing FSM runners."""
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
FINITE_STATUSES = {"converged", "stalled", "max_iterations", "ok"}


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
    return {key: torch.as_tensor(values[burn:].copy(), dtype=torch.float64, device=device)
            for key, values in (("r", u), ("nu", nu), ("c", c))}


def _setup(cfg, study, device):
    cfg = copy.deepcopy(cfg)
    if "K_max" in study:
        cfg["settings"]["K_max"] = positive_integer(study["K_max"], "K_max")
    cfg["settings"]["enforce_fixed_iterations"] = False
    dims, theta = build_dims_and_theta(cfg, dtype=torch.float64, device=device)
    methods = select_python_methods(cfg)
    configs = load_json_config(ROOT/"python"/cfg.get("python_optim_config", "optim_configs.json"))["methods"]
    return cfg, dims, theta, methods, configs


def _run(cfg, methods, configs, dims, theta, data, metadata, verbose, progress):
    settings = cfg["settings"]
    return run_method_set(methods, configs, data["r"], data["nu"], data["c"], theta, dims,
                          len(data["r"]), int(settings["K_max"]), float(settings["conv_threshold"]),
                          False, verbose=verbose, experiment_cfg=cfg, run_metadata=metadata,
                          progress_callback=progress if progress.enabled else None)


def _save_case(path, cfg, study, methods, dims, theta, data, runs, name, mode):
    settings = cfg["settings"]
    summary = summarize_montecarlo(runs, methods, theta, int(settings["K_max"]))
    save_comparison_results(Path(path), runs[0], experiment_name=name, mode=mode,
        selected_methods=methods, dims=dims, theta_true=theta, lambda_g=len(data["r"]),
        K_max=int(settings["K_max"]), conv_threshold=float(settings["conv_threshold"]),
        sigma_nu=float(settings["sigma_nu"]), burn_in=int(settings["burn_in"]),
        enforce_fixed_iterations=False, seed=int(study["data_seed"]), n_mc=len(runs),
        example_data={k: v.detach().cpu().numpy().reshape(-1, 1) for k, v in data.items()},
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
            if result["status"] in FINITE_STATUSES and np.isfinite(values).all():
                estimates[m, r] = values
    return estimates, statuses, configs, ids


def bias_statistics(estimates, statuses, theta_true, population="finite"):
    """MC intervals describe uncertainty of the mean error, not parameter intervals."""
    if population not in {"finite", "converged"}:
        raise ValueError("summary_population must be finite or converged")
    values = np.asarray(estimates, dtype=float).copy()
    finite_runs = np.isfinite(values).all(axis=2)
    selected = finite_runs & ((statuses == "converged") if population == "converged" else True)
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
            "converged_count": (statuses == "converged").sum(axis=1),
            "skipped_count": (statuses == "skipped").sum(axis=1),
            "failed_count": (~np.isin(statuses, list(FINITE_STATUSES)+["skipped"])).sum(axis=1)}


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
        if max(resolution.get("fsm_tau_g", 1), resolution.get("fsm_tau_h", 1)) > lengths[0]:
            raise ValueError("Every record must be at least as long as the configured FSM responses")
    start = parameter_vector(study.get("theta0"), dims, "theta0", nominal_theta(dims))
    if not stable(start, dims):
        raise ValueError("theta0 denominators must be stable")
    cfg["initialization"] = {"policy": "consistent", "theta0": start.tolist()}
    seed = int(study.get("data_seed", 271828))
    population = study.get("summary_population", "finite")
    path = Path(output_file or ROOT/"results"/experiment_name/"record_length_fsm.mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(resolutions), len(lengths), len(methods), count)
    estimates = np.full(shape+(dims.n_params,), np.nan)
    statuses = np.full(shape, "not_run", dtype=object)
    resolved_configs = np.full(shape, "", dtype=object)
    dataset_ids = np.full(shape, "", dtype=object)
    # Draw each full record once; all lengths and resolutions share its prefixes.
    progress = StudyProgress("FSM record_length", methods, count, lengths, len(resolutions), study)
    progress.stage(f"Preparing {count} independent records of N={lengths[-1]}")
    full_records = []
    for r in range(count):
        full_records.append(reference_data(cfg, dims, theta, lengths[-1], seed+r, device))
        progress.stage(f"Prepared data record {r+1}/{count}", force=(r+1 == count))
    summaries = []
    for k, resolution in enumerate(resolutions):
        case_cfgs = copy.deepcopy(configs)
        for method_cfg in case_cfgs.values():
            method_cfg.update({key: value for key, value in resolution.items() if key != "label"})
        case_summaries = []
        for j, length in enumerate(lengths):
            runs = []
            for r, full in enumerate(full_records):
                data = {key: value[:length] for key, value in full.items()}
                runs.append(_run(cfg, methods, case_cfgs, dims, theta, data,
                    {"study_type": "record_length", "record_length": length, "mc_index": r+1,
                     "data_seed": seed+r, "resolution": resolution["label"]}, verbose, progress))
            progress.stage(f"Summarizing {resolution['label']}, N={length}")
            arrays = terminal_arrays(runs, methods, dims.n_params)
            estimates[k, j], statuses[k, j], resolved_configs[k, j], dataset_ids[k, j] = arrays
            case_summaries.append(bias_statistics(arrays[0], arrays[1], theta.cpu().numpy(), population))
            if study.get("save_case_results", False):
                case_file = path.parent/(path.stem+f"_{k+1}_n{length}.mat")
                _save_case(case_file, cfg, {"type": "record_length_case", "data_seed": seed}, methods,
                           dims, theta, {key: value[:length] for key, value in full_records[0].items()},
                           runs, experiment_name, "MONTECARLO")
        summaries.append(case_summaries)
    statistics = {key: np.asarray([[cell[key] for cell in row] for row in summaries]) for key in summaries[0][0]}
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
             "config_json": config_json(study), "initial_condition_policy": "zero_state_fit_after_burn_in",
             **statistics}
    progress.stage("Saving record-length results")
    savemat(path, {"schema_version": 2, "parametrization": "FSM", "study": saved,
                   "experiment_name": experiment_name, "resolved_experiment_json": config_json(cfg)}, long_field_names=True)
    progress.finish(path)
    return path


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="example_CSTR")
    parser.add_argument("--study-config", type=Path, default=ROOT/"configs"/"studies.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable study progress messages")
    args = parser.parse_args()
    study = load_json_config(args.study_config)[kind+"_study"]
    if args.no_progress:
        study["progress"] = False
    if args.runs is not None:
        study["n_runs"] = args.runs
    if args.iterations is not None:
        study["K_max"] = args.iterations
    if args.lengths is not None:
        if kind != "record_length":
            parser.error("--lengths is only available for the record-length study")
        study["record_lengths"] = args.lengths
    runner = run_initialization_study if kind == "initialization" else run_record_length_study
    runner(load_experiment_config(args.experiment), study, experiment_name=args.experiment,
           output_file=args.output, device=torch.device("cpu") if args.cpu else default_device(), verbose=args.verbose)
