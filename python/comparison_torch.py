from __future__ import annotations

import sys
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
    print_iteration_table,
    set_seed,
)
from methods import method_parameter_update
from wiener_system import WienerDimensions, generate_example_data, initialize_method_state, simulate_wiener


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
) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError("Saving MATLAB-compatible results requires scipy. Install scipy and rerun this script.") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    param_names = build_param_names(dims.na, dims.nb, dims.nf, dims.nd)
    param_labels = build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd)
    result_fields = (
        "name",
        "status",
        "status_msg",
        "theta_hat",
        "theta_hist",
        "param_err",
        "rel_change",
        "rmse_hist",
        "iter_time",
        "cum_time",
        "final_err",
        "total_time",
        "iterations",
    )
    mat_results = np.empty((1, len(results)), dtype=[(field, "O") for field in result_fields])

    for idx, result in enumerate(results):
        for field in result_fields:
            value = result[field]
            if field == "theta_hat":
                value = np.asarray(value, dtype=float).reshape(-1, 1)
            elif field in {"param_err", "rel_change", "rmse_hist", "iter_time", "cum_time"}:
                value = np.asarray(value, dtype=float).reshape(-1, 1)
            elif field == "theta_hist":
                value = np.asarray(value, dtype=float)
            mat_results[field][0, idx] = value

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
        "parametrization": "FSM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "results": mat_results,
    }
    if example_data is not None:
        payload["example_data"] = example_data
    if mc_summary is not None:
        payload["mc_summary"] = mc_summary

    savemat(output_file, payload, long_field_names=True)
    print(f"\nSaved FSM results to: {output_file}")


def run_identification_method(
    method_name: str,
    method_cfg: dict[str, Any],
    r: torch.Tensor,
    nu: torch.Tensor,
    c: torch.Tensor,
    theta_true: torch.Tensor,
    dims: WienerDimensions,
    lambda_g: int,
    K_max: int,
    conv_threshold: float,
    *,
    enforce_fixed_iterations: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    init_state = initialize_method_state(method_cfg, r, c, dims)
    theta_hat = init_state["theta_hat"]

    theta_hist = torch.zeros((dims.n_params, K_max), dtype=r.dtype, device=r.device)
    param_err = torch.full((K_max,), torch.nan, dtype=r.dtype, device=r.device)
    rel_change = torch.full((K_max,), torch.nan, dtype=r.dtype, device=r.device)
    rmse_hist = torch.full((K_max,), torch.nan, dtype=r.dtype, device=r.device)
    iter_time = torch.full((K_max,), torch.nan, dtype=r.dtype, device=r.device)
    cum_time = torch.full((K_max,), torch.nan, dtype=r.dtype, device=r.device)

    status = "ok"
    status_msg = ""
    iter_count = K_max
    has_converged = False
    converged_iter: int | None = None
    derivative_snapshots: list[dict[str, np.ndarray]] = []

    theta_hist[:, 0] = theta_hat
    param_err[0] = torch.linalg.vector_norm(theta_hat - theta_true) / torch.linalg.vector_norm(theta_true) * 100.0
    c_sim_0 = simulate_wiener(r, nu, theta_hat, dims)
    rmse_hist[0] = torch.sqrt(torch.mean((c - c_sim_0).pow(2)))
    rel_change[0] = 0.0
    iter_time[0] = 0.0
    cum_time[0] = 0.0

    for k in tqdm(range(1, K_max), desc=method_name, unit="iter", leave=False):
        if enforce_fixed_iterations and has_converged:
            theta_hist[:, k] = theta_hat
            param_err[k] = param_err[k - 1]
            rel_change[k] = 0.0
            rmse_hist[k] = rmse_hist[k - 1]
            iter_time[k] = 0.0
            cum_time[k] = cum_time[k - 1]
            continue

        t0 = now()
        theta_new, aux_state = method_parameter_update(
            method_name,
            method_cfg,
            None,
            c,
            theta_hat,
            r=r,
            nu=nu,
            dims=dims,
            iter_idx=k,
        )

        if aux_state["derivatives"] is not None:
            snapshot = {name: value.detach().cpu().numpy() for name, value in aux_state["derivatives"].items() if name != "loss"}
            derivative_snapshots.append(snapshot)

        if aux_state["status"] == "not_implemented":
            status = "skipped"
            status_msg = aux_state["message"]
            iter_count = k
            break

        rel_change[k] = torch.linalg.vector_norm(theta_new - theta_hat) / (torch.linalg.vector_norm(theta_hat) + 1e-15)
        theta_hat = theta_new
        theta_hist[:, k] = theta_hat

        param_err[k] = torch.linalg.vector_norm(theta_hat - theta_true) / torch.linalg.vector_norm(theta_true) * 100.0
        c_sim = simulate_wiener(r, nu, theta_hat, dims)
        rmse_hist[k] = torch.sqrt(torch.mean((c - c_sim).pow(2)))

        iter_time[k] = now() - t0
        cum_time[k] = cum_time[k - 1] + iter_time[k]

        if rel_change[k].item() < conv_threshold:
            if enforce_fixed_iterations:
                has_converged = True
                converged_iter = k
            else:
                iter_count = k + 1
                converged_iter = k
                break

    if status == "ok":
        theta_hist = theta_hist[:, :iter_count]
        param_err = param_err[:iter_count]
        rel_change = rel_change[:iter_count]
        rmse_hist = rmse_hist[:iter_count]
        iter_time = iter_time[:iter_count]
        cum_time = cum_time[:iter_count]
        final_err = float(param_err[-1].item())
        total_time = float(cum_time[-1].item())
        theta_out = theta_hat.detach().cpu().numpy()
    else:
        theta_hist = torch.zeros((dims.n_params, 0), dtype=r.dtype, device=r.device)
        param_err = torch.zeros(0, dtype=r.dtype, device=r.device)
        rel_change = torch.zeros(0, dtype=r.dtype, device=r.device)
        rmse_hist = torch.zeros(0, dtype=r.dtype, device=r.device)
        iter_time = torch.zeros(0, dtype=r.dtype, device=r.device)
        cum_time = torch.zeros(0, dtype=r.dtype, device=r.device)
        final_err = float("nan")
        total_time = 0.0
        theta_out = np.array([])

    if status_msg:
        print(f"  {status_msg}")

    if status == "ok" and verbose:
        if converged_iter is not None:
            if enforce_fixed_iterations:
                print(f"*** {method_name} converged at iteration {converged_iter} (holding parameters for remaining iterations) ***")
            else:
                print(f"*** {method_name} converged at iteration {converged_iter} ***")
        print_iteration_table(
            build_param_names(dims.na, dims.nb, dims.nf, dims.nd),
            theta_hist.detach().cpu().numpy(),
            param_err.detach().cpu().numpy(),
            cum_time.detach().cpu().numpy(),
            title=f"{method_name} Iteration Summary",
        )

    return {
        "name": method_name,
        "status": status,
        "status_msg": status_msg,
        "theta_hat": theta_out,
        "theta_hist": theta_hist.detach().cpu().numpy(),
        "param_err": param_err.detach().cpu().numpy(),
        "rel_change": rel_change.detach().cpu().numpy(),
        "rmse_hist": rmse_hist.detach().cpu().numpy(),
        "iter_time": iter_time.detach().cpu().numpy(),
        "cum_time": cum_time.detach().cpu().numpy(),
        "final_err": final_err,
        "total_time": total_time,
        "iterations": theta_hist.shape[1],
        "derivative_snapshots": derivative_snapshots,
    }


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
) -> list[dict[str, Any]]:
    results = []
    for method_name in selected_methods:
        method_cfg = method_configs[make_valid_method_key(method_name)]
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
        )
        results.append(method_result)
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

    for run_idx, run_results in enumerate(mc_runs):
        by_name = {result["name"]: result for result in run_results}
        for method_idx, method_name in enumerate(selected_methods):
            result = by_name[method_name]
            param_err_runs[method_idx, run_idx, :] = pad_vector(result["param_err"], K_max)
            rmse_runs[method_idx, run_idx, :] = pad_vector(result["rmse_hist"], K_max)
            final_errors[method_idx, run_idx] = result["final_err"]
            if result["theta_hat"].size:
                theta_hat = np.asarray(result["theta_hat"], dtype=float).reshape(-1)
                final_param_estimates[method_idx, run_idx, :] = theta_hat
                final_param_errors[method_idx, run_idx, :] = theta_hat - theta_true_np

    return {
        "method_names": np.asarray(selected_methods, dtype=object).reshape(1, -1),
        "param_err_runs": param_err_runs,
        "param_err_mean": np.nanmean(param_err_runs, axis=1),
        "param_err_stderr": np.nanstd(param_err_runs, axis=1, ddof=1) / np.sqrt(max(n_mc, 1)),
        "rmse_runs": rmse_runs,
        "rmse_mean": np.nanmean(rmse_runs, axis=1),
        "rmse_stderr": np.nanstd(rmse_runs, axis=1, ddof=1) / np.sqrt(max(n_mc, 1)),
        "final_param_errors": final_param_errors,
        "final_param_estimates": final_param_estimates,
        "final_errors": final_errors,
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
    selected_methods = exp_cfg["selected_methods"]
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
    )

    mc_summary = None
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
                )
            )
        mc_summary = summarize_montecarlo(mc_runs, selected_methods, theta_true, K_max)

    print("\n" + "=" * 92)
    print("FINAL METHOD SUMMARY")
    print("=" * 92)
    print(f"{'Method':<14} {'Status':<12} {'Final err(%)':<12} {'Final d1':<12} {'CPU time (s)':<12}")
    print("-" * 92)
    for result in results:
        final_d = float("nan")
        if result["theta_hat"].size:
            final_d = result["theta_hat"][-1]
        print(f"{result['name']:<14} {result['status']:<12} {result['final_err']:<12.5f} {final_d:<12.5f} {result['total_time']:<12.5f}")

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
        },
        mc_summary=mc_summary,
    )

    if make_local_plots:
        plot_method_metrics(results)
        plot_parameter_trajectories(results, theta_true.detach().cpu().numpy(), build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd))

        import matplotlib.pyplot as plt

        plt.show()
        print("\nDone. All figures generated.")
    else:
        print("\nDone. Results saved for MATLAB plotting.")


if __name__ == "__main__":
    main()
