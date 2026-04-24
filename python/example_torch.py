from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import ROOT as PY_ROOT, build_param_names, build_plot_labels, choose_dtype, default_device, load_json_config, make_valid_method_key, now, print_iteration_table, set_seed
from methods import method_parameter_update
from wiener_system import WienerDimensions, build_state_matrix, generate_example_data, initialize_method_state, parameter_to_loss, simulate_wiener


def main() -> None:
    set_seed(42)
    device = default_device()
    dtype = choose_dtype()

    dims = WienerDimensions(na=2, nb=2, nf=2, nd=1)
    theta_true = torch.tensor([-0.31, -0.27, 0.23, 0.98, 0.32, -0.40], dtype=dtype, device=device)

    lambda_g = 1000*2
    burn_in = 100
    sigma_nu = 0.10*5
    K_max = 240*1
    conv_threshold = 1e-8
    method_name = "WS-GNI"
    config_file = PY_ROOT / "optim_configs.json"

    # Generate measured data from the true system
    data = generate_example_data(dims, theta_true, lambda_g, burn_in, sigma_nu, device=device, dtype=dtype)
    r, nu, c = data["r"], data["nu"], data["c"]

    method_cfg = load_json_config(config_file)["methods"][make_valid_method_key(method_name)]

    init_state = initialize_method_state(method_cfg, r, c, dims)
    alpha_hat = init_state["alpha_hat"]
    e_hat = init_state["e_hat"]
    theta_hat = init_state["theta_hat"]

    theta_hist = torch.zeros((dims.n_params, K_max), dtype=dtype, device=device)
    param_err = torch.zeros(K_max, dtype=dtype, device=device)
    rel_change = torch.zeros(K_max, dtype=dtype, device=device)
    rmse_hist = torch.zeros(K_max, dtype=dtype, device=device)
    iter_time = torch.zeros(K_max, dtype=dtype, device=device)
    cum_time = torch.zeros(K_max, dtype=dtype, device=device)

    param_names = build_param_names(dims.na, dims.nb, dims.nf, dims.nd)
    converged_iter: int | None = None

    for k in tqdm(range(K_max), desc=method_name, unit="iter"):
        t0 = now()
        phi_hat = build_state_matrix(alpha_hat, e_hat, r, dims)
        theta_new, _ = method_parameter_update(method_name, method_cfg, phi_hat, c, theta_hat, r=r, nu=nu, dims=dims, iter_idx=k)

        residual_eq = c - phi_hat @ theta_new
        a_hat = theta_new[:dims.na]
        b_hat = theta_new[dims.na:dims.na + dims.nb]
        d_hat = theta_new[dims.na + dims.nb + (dims.nf - 1):]

        den_ar_hat = torch.cat([torch.ones(1, dtype=dtype, device=device), d_hat])
        e_hat = simulate_ar_noise(residual_eq, den_ar_hat)

        den_lin_hat = torch.cat([torch.ones(1, dtype=dtype, device=device), a_hat])
        num_lin_hat = torch.cat([torch.zeros(1, dtype=dtype, device=device), b_hat])
        alpha_hat = simulate_linear_block(r, den_lin_hat, num_lin_hat)

        rel_change[k] = torch.linalg.vector_norm(theta_new - theta_hat) / (torch.linalg.vector_norm(theta_hat) + 1e-15)
        theta_hat = theta_new
        theta_hist[:, k] = theta_hat
        param_err[k] = torch.linalg.vector_norm(theta_hat - theta_true) / torch.linalg.vector_norm(theta_true) * 100.0
        c_sim = simulate_wiener(r, nu, theta_hat, dims)
        rmse_hist[k] = torch.sqrt(torch.mean((c - c_sim).pow(2)))
        iter_time[k] = now() - t0
        cum_time[k] = iter_time[k] if k == 0 else cum_time[k - 1] + iter_time[k]

        if k > 0 and rel_change[k].item() < conv_threshold:
            theta_hist = theta_hist[:, : k + 1]
            param_err = param_err[: k + 1]
            rmse_hist = rmse_hist[: k + 1]
            iter_time = iter_time[: k + 1]
            cum_time = cum_time[: k + 1]
            converged_iter = k + 1
            break

    theta_np = theta_hat.detach().cpu().numpy()
    true_np = theta_true.detach().cpu().numpy()
    theta_hist_np = theta_hist.detach().cpu().numpy()
    param_err_np = param_err.detach().cpu().numpy()
    rmse_np = rmse_hist.detach().cpu().numpy()
    cum_time_np = cum_time.detach().cpu().numpy()

    print()
    if converged_iter is not None:
        print(f"*** {method_name} converged at iteration {converged_iter} ***")
    print_iteration_table(param_names, theta_hist_np, param_err_np, cum_time_np, title=f"{method_name} Iteration Summary")
    print("\n" + "=" * 60)
    print("FINAL PARAMETER ESTIMATES vs. TRUE VALUES")
    print("=" * 60)
    print(f"{'Param':<8}  {'Estimated':<12}  {'True':<12}")
    print("-" * 40)
    for name, est, true in zip(param_names, theta_np, true_np):
        print(f"{name:<8}  {est:<12.5f}  {true:<12.5f}")
    print("-" * 40)
    print(f"Final err(%) = {param_err[-1].item():.5f}")
    print(f"Final loss   = {parameter_to_loss(theta_hat, r, nu, c, dims).item():.5f}")

    c_sim_final = simulate_wiener(r, nu, theta_hat, dims).detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()
    labels = build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd)

    fig = plt.figure(figsize=(10, 7.5))
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(np.arange(lambda_g), c_np, "b-", lw=0.8, label="Observed Output")
    ax1.plot(np.arange(lambda_g), c_sim_final, "r.", ms=3.0, label=f"Simulated Output / {method_name}")
    ax1.set_title(f"Observation and Simulation Output of {method_name}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("c(t)")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.semilogy(np.arange(1, len(param_err_np) + 1), param_err_np, "b-", lw=1.8)
    ax2.set_title("Parameter Error Convergence")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("err (%)")
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.semilogy(np.arange(1, len(rmse_np) + 1), rmse_np, "b-", lw=1.5, label="RMSE")
    ax3.set_title("RMSE vs Iteration")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Error")
    ax3.grid(True)
    ax3.legend(loc="best")
    fig.suptitle(f"{method_name}: Wiener System with AR Noise - Example 1 (PyTorch)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig2, axes = plt.subplots(int(np.ceil(dims.n_params / 3)), 3, figsize=(9, 5.5))
    axes = np.atleast_1d(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, dims.n_params))
    for i in range(dims.n_params):
        axes[i].plot(np.arange(1, theta_hist_np.shape[1] + 1), theta_hist_np[i], color=colors[i], lw=1.5)
        axes[i].axhline(true_np[i], color="k", ls="--", lw=1.2)
        axes[i].set_title(f"Parameter {labels[i]}")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
        axes[i].legend(["Estimate", "True"], loc="best")
    for i in range(dims.n_params, len(axes)):
        axes[i].axis("off")
    fig2.suptitle(f"Parameter Convergence Trajectories - {method_name} (PyTorch)", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    plt.show()


def simulate_linear_block(r: torch.Tensor, den_lin: torch.Tensor, num_lin: torch.Tensor) -> torch.Tensor:
    from wiener_system import lfilter_1d

    return lfilter_1d(r, den_lin, num_lin)


def simulate_ar_noise(residual_eq: torch.Tensor, den_ar: torch.Tensor) -> torch.Tensor:
    from wiener_system import lfilter_1d

    num_ar = torch.ones(1, dtype=residual_eq.dtype, device=residual_eq.device)
    return lfilter_1d(residual_eq, den_ar, num_ar)


if __name__ == "__main__":
    main()
