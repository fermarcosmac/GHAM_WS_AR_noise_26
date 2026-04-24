from __future__ import annotations

import sys
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
from wiener_system import WienerDimensions, build_state_matrix, generate_example_data, initialize_method_state, lfilter_1d, simulate_wiener


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
) -> dict[str, Any]:
    init_state = initialize_method_state(method_cfg, r, c, dims)
    alpha_hat = init_state["alpha_hat"]
    e_hat = init_state["e_hat"]
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
        phi_hat = build_state_matrix(alpha_hat, e_hat, r, dims)
        theta_new, aux_state = method_parameter_update(
            method_name,
            method_cfg,
            phi_hat,
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

        residual_eq = c - phi_hat @ theta_new
        a_hat = theta_new[:dims.na]
        b_hat = theta_new[dims.na:dims.na + dims.nb]
        d_hat = theta_new[dims.na + dims.nb + (dims.nf - 1):]

        den_ar_hat = torch.cat([torch.ones(1, dtype=r.dtype, device=r.device), d_hat])
        e_hat = lfilter_1d(residual_eq, den_ar_hat, torch.ones(1, dtype=r.dtype, device=r.device))

        den_lin_hat = torch.cat([torch.ones(1, dtype=r.dtype, device=r.device), a_hat])
        num_lin_hat = torch.cat([torch.zeros(1, dtype=r.dtype, device=r.device), b_hat])
        alpha_hat = lfilter_1d(r, den_lin_hat, num_lin_hat)

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

    if status == "ok":
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


def main() -> None:
    set_seed(42)
    device = default_device()
    dtype = choose_dtype()

    config_file = PY_ROOT / "optim_configs.json"
    selected_methods = [
        "RGLS",
        "WS-GNI",
        "WS-GGI",
        "WS-GGHAM-1-dH",
        "WS-GGHAM-2-I",
        "WS-GGHAM-2-dH",
        "WS-LGHAM-1-TIK",
        "WS-LGHAM-3-TIK",
    ]

    dims = WienerDimensions(na=2, nb=2, nf=2, nd=1)
    theta_true = torch.tensor([-0.31, -0.27, 0.23, 0.98, 0.32, -0.40], dtype=dtype, device=device)

    lambda_g = 1000
    K_max = 240
    conv_threshold = 1e-8
    sigma_nu = 0.10
    burn_in = 100
    enforce_fixed_iterations = False

    data = generate_example_data(dims, theta_true, lambda_g, burn_in, sigma_nu, device=device, dtype=dtype)
    r, nu, c = data["r"], data["nu"], data["c"]
    method_configs = load_json_config(config_file)["methods"]

    results = []
    for method_name in selected_methods:
        method_cfg = method_configs[make_valid_method_key(method_name)]
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
        )
        results.append(method_result)

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

    plot_method_metrics(results)
    plot_parameter_trajectories(results, theta_true.detach().cpu().numpy(), build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd))

    import matplotlib.pyplot as plt

    plt.show()
    print("\nDone. All figures generated.")


if __name__ == "__main__":
    main()
