from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import ROOT as PY_ROOT, build_param_names, build_plot_labels, choose_dtype, default_device, load_json_config, make_valid_method_key, method_figure_label, now, print_iteration_table, set_seed
from methods import method_parameter_update
from wiener_system import (
    WienerDimensions,
    estimate_innovations_from_signal_residual,
    fsm_kwargs_from_config,
    generate_example_data,
    initialize_method_state,
    parameter_to_loss,
    simulate_wiener,
)


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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def run_bayesian_example(args, exp_cfg, configs, data, dims, theta_true):
    """A separate inference branch: save posteriors and final scores, not updates."""
    from comparison_torch import (inference_data_context, run_method_set,
                                  save_comparison_results, summarize_montecarlo)
    from bayesian_plots import plot_bayesian_posteriors, plot_bayesian_outputs

    cfg = dict(configs.get("BI_HMC", {}))
    if args.hmc_config:
        cfg.update(load_json_config(args.hmc_config))
    for option in ("chains", "warmup", "samples", "leapfrog_steps"):
        value = getattr(args, option)
        if value is not None:
            cfg[option] = value
    if args.sampler_seed is not None:
        cfg["seed"] = args.sampler_seed
    configs = {**configs, "BI_HMC": cfg}
    settings = exp_cfg["settings"]
    result = run_method_set(["BI_HMC"], configs, data["r"], data["nu"], data["c"],
        theta_true, dims, int(settings["lambda_g"]), int(settings["K_max"]),
        float(settings["conv_threshold"]), False, verbose=True, experiment_cfg=exp_cfg,
        run_metadata={"data_seed": int(exp_cfg.get("seed", 42)), "mc_index": 1,
                      "experiment_name": args.experiment},
        data_context=inference_data_context(data, int(settings["burn_in"])))[0]
    truth = theta_true.detach().cpu().numpy()
    names = build_param_names(dims.na, dims.nb, dims.nf, dims.nd)
    intervals = result.get("posterior_interval", np.full((2, dims.n_params), np.nan))
    print("\nFINAL PHYSICAL POSTERIOR MEAN AND 95% CREDIBLE INTERVALS")
    print(f"{'Parameter':<10} {'Mean':>12} {'Truth':>12} {'Lower':>12} {'Upper':>12}")
    for i, name in enumerate(names):
        print(f"{name:<10} {result['theta_hat'][i]:>12.6f} {truth[i]:>12.6f} "
              f"{intervals[0,i]:>12.6f} {intervals[1,i]:>12.6f}")
    print(f"Innovation variance posterior mean: {result['final_innovation_variance']:.6g}")
    print(f"Final parameter error: {result['final_err']:.6g}%")
    print(f"Final reconstruction RMSE: {result['final_output_rmse']:.6g}")
    print(f"Total wall-clock time: {result['total_wall_time']:.3f} s")
    diagnostics = result["sampler_diagnostics"]
    print(f"Sampling status: {result['status']}")
    for warning in diagnostics.get("warnings", []):
        print(f"  Sampling diagnostic: {warning}")
    for key in ("rhat", "bulk_ess", "tail_ess", "acceptance_rate", "retained_divergences"):
        if key in diagnostics:
            print(f"{key}: {np.asarray(diagnostics[key])}")

    output = Path(args.output or PY_ROOT.parent/"results"/"local"/f"{args.experiment}_bi_hmc.mat")
    summary = summarize_montecarlo([[result]], ["BI_HMC"], theta_true, int(settings["K_max"]))
    save_comparison_results(output, [result], experiment_name=args.experiment, mode="EXAMPLE",
        selected_methods=["BI_HMC"], dims=dims, theta_true=theta_true,
        lambda_g=len(data["r"]), K_max=int(settings["K_max"]),
        conv_threshold=float(settings["conv_threshold"]), sigma_nu=float(settings["sigma_nu"]),
        burn_in=int(settings["burn_in"]), enforce_fixed_iterations=False,
        seed=int(exp_cfg.get("seed", 42)), example_data={
            **{key: data[key].detach().cpu().numpy().reshape(-1, 1) for key in ("r", "nu", "c")},
            "input_prehistory": data["r_full"][:int(settings["burn_in"])].detach().cpu().numpy(),
            "noise_burn_in": int(settings["burn_in"])},
        all_runs=[[result]], mc_summary=summary, resolved_experiment=exp_cfg)
    report = {"experiment": args.experiment, "method": "BI_HMC", "point_estimate": "posterior_mean",
              "parameter_names": names, "theta_true": truth, "theta_hat": result["theta_hat"],
              "posterior_std": result.get("posterior_std", []), "posterior_interval": intervals,
              "final_err_percent": result["final_err"], "final_output_rmse": result["final_output_rmse"],
              "innovation_variance_mean": result["final_innovation_variance"],
              "total_wall_time": result["total_wall_time"], "status": result["status"],
              "diagnostics_passed": result["diagnostics_passed"], "sampler_diagnostics": diagnostics,
              "run_metadata": result["run_metadata"], "resolved_config": json.loads(result["resolved_config_json"])}
    report_file = output.with_suffix(".summary.json")
    report_file.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    figure_dir = Path(args.figure_dir or PY_ROOT.parent/"figs"/"local"/"bi_hmc"/args.experiment)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = [("posterior", plot_bayesian_posteriors(result, truth, build_plot_labels(dims.na,dims.nb,dims.nf,dims.nd))),
               ("output_comparison", plot_bayesian_outputs(result, data["c"].detach().cpu().numpy(),
                                  title=f"BI_HMC / {args.experiment}: final parameter estimate"))]
    for name, figure in figures:
        if figure is not None:
            figure.savefig(figure_dir/f"{name}.png", dpi=160)
            figure.savefig(figure_dir/f"{name}.pdf")
    print(f"Saved summary: {report_file}")
    print(f"Saved posterior/output figures: {figure_dir}")
    if args.no_show:
        for _, figure in figures:
            if figure is not None:
                plt.close(figure)
    else:
        plt.show()
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Single Wiener-system identification example; BI_HMC uses its own sampling loop.")
    parser.add_argument("--experiment", default="example_CSTR")
    parser.add_argument("--method", help="For example BI_HMC or WS-GGI; otherwise use the configured method.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--length", type=int, help="Override retained record length in memory.")
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--sampler-seed", type=int)
    for option in ("chains", "warmup", "samples", "leapfrog-steps"):
        parser.add_argument("--"+option, type=int, help="BI_HMC sampling-budget override.")
    parser.add_argument("--hmc-config", type=Path, help="JSON object overriding BI_HMC method settings.")
    parser.add_argument("--output", type=Path, help="BI_HMC MATLAB-compatible output file.")
    parser.add_argument("--figure-dir", type=Path, help="BI_HMC posterior/output figure directory.")
    parser.add_argument("--no-show", action="store_true", help="Save Bayesian figures without opening windows.")
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")

    # Change method or experiment name here, or use command-line arguments to override them.
    experiment_name = args.experiment
    default_method_name = "BI_HMC" # Default method for this example; can be overridden by --method argument.

    device = torch.device("cpu") if args.cpu else default_device()
    dtype = choose_dtype()
    exp_cfg = load_experiment_config(experiment_name)
    if args.length is not None:
        if args.length < 1:
            parser.error("--length must be positive")
        exp_cfg["settings"]["lambda_g"] = args.length
    if args.data_seed is not None:
        exp_cfg["seed"] = args.data_seed
    dims, theta_true = build_dims_and_theta(exp_cfg, dtype=dtype, device=device)

    seed = int(exp_cfg.get("seed", 52))
    settings = exp_cfg["settings"]
    lambda_g = int(settings["lambda_g"])
    burn_in = int(settings["burn_in"])
    sigma_nu = float(settings["sigma_nu"])
    K_max = int(settings["K_max"])
    conv_threshold = float(settings["conv_threshold"])
    method_name = args.method or exp_cfg.get("python_example_method", exp_cfg.get("example_method", default_method_name))
    config_file = PY_ROOT / exp_cfg.get("python_optim_config", "optim_configs.json")

    # Generate measured data from the true system
    set_seed(seed)
    data = generate_example_data(dims, theta_true, lambda_g, burn_in, sigma_nu, device=device, dtype=dtype)
    r, nu, c = data["r"], data["nu"], data["c"]

    method_configs = load_json_config(config_file)["methods"]
    if method_name.upper() == "BI_HMC":
        return run_bayesian_example(args, exp_cfg, method_configs, data, dims, theta_true)
    method_cfg = method_configs[make_valid_method_key(method_name)]
    fsm_kwargs = fsm_kwargs_from_config(method_cfg)

    init_state = initialize_method_state(method_cfg, r, c, dims)
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
        innovation_state = estimate_innovations_from_signal_residual(theta_hat, r, c, dims, **fsm_kwargs)
        theta_new, _ = method_parameter_update(
            method_name,
            method_cfg,
            None,
            c,
            theta_hat,
            r=r,
            nu=innovation_state["nu_hat"],
            dims=dims,
            iter_idx=k,
        )

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
    final_innovation_state = estimate_innovations_from_signal_residual(theta_hat, r, c, dims, **fsm_kwargs)
    print(f"Final loss   = {parameter_to_loss(theta_hat, r, final_innovation_state['nu_hat'], c, dims, **fsm_kwargs).item():.5f}")

    c_sim_final = simulate_wiener(r, nu, theta_hat, dims).detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()
    labels = build_plot_labels(dims.na, dims.nb, dims.nf, dims.nd)

    fig = plt.figure(figsize=(10, 7.5))
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(np.arange(lambda_g), c_np, "b-", lw=0.8, label="Observed Output")
    ax1.plot(np.arange(lambda_g), c_sim_final, "r.", ms=3.0, label=f"Simulated Output / {method_figure_label(method_name)}")
    ax1.set_title(f"Observation and Simulation Output of {method_figure_label(method_name)}")
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
    fig.suptitle(f"{method_figure_label(method_name)}: {experiment_name} (PyTorch)", fontsize=13, fontweight="bold")
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
    fig2.suptitle(f"Parameter Convergence Trajectories - {method_figure_label(method_name)} / {experiment_name} (PyTorch)", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    if args.no_show:
        plt.close("all")
    else:
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
