"""Posterior and terminal-estimate figures; draws are never optimizer iterates."""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt


def plot_bayesian_posteriors(result: dict, theta_true: np.ndarray, labels: list[str]):
    samples = np.asarray(result.get("posterior_samples", []), dtype=float)
    if samples.ndim != 3 or not samples.size:
        return None
    flat = samples.reshape(-1, samples.shape[-1])
    variances = np.asarray(result.get("posterior_variance_samples", []), dtype=float).reshape(-1)
    count = flat.shape[1] + bool(variances.size)
    columns = min(3, count)
    fig, axes = plt.subplots(math.ceil(count/columns), columns,
                             figsize=(4.1*columns, 2.9*math.ceil(count/columns)), squeeze=False)
    estimate = np.asarray(result["theta_hat"]).reshape(-1)
    for j, ax in enumerate(axes.flat):
        if j >= count:
            ax.set_visible(False)
            continue
        values = flat[:, j] if j < flat.shape[1] else variances
        low, high = np.quantile(values, [.025, .975])
        ax.hist(values, bins="auto", density=True, color="C0", alpha=.7, label="Posterior draws")
        ax.axvspan(low, high, color="C0", alpha=.12, label="95% credible interval")
        if j < flat.shape[1]:
            ax.axvline(theta_true[j], color="black", ls="--", label="True coefficient")
            ax.axvline(estimate[j], color="C3", label="Posterior mean")
            ax.set_xlabel(f"${labels[j]}$")
        else:
            ax.axvline(values.mean(), color="C3", label="Posterior mean")
            ax.set_xlabel(r"$\sigma_\nu^2$")
        ax.set_ylabel("Density")
        ax.grid(alpha=.2)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"BI_HMC posterior distributions ({result['status']})")
    fig.tight_layout()
    return fig


def plot_bayesian_outputs(result: dict, observed: np.ndarray, *, title: str = ""):
    output = np.asarray(result.get("output_estimate", []), dtype=float).reshape(-1)
    if not output.size:
        return None
    observed = np.asarray(observed).reshape(-1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(observed, lw=.8, label="Observed output")
    axes[0].plot(output, lw=.8, alpha=.8, label="Posterior-mean parameter reconstruction")
    axes[0].set_ylabel(r"$c_t$")
    axes[0].legend(loc="best")
    axes[0].set_title(title or "BI_HMC final-estimate output comparison")
    axes[1].plot(observed-output, lw=.8, color="C3")
    axes[1].set_ylabel("Reconstruction error")
    axes[1].set_xlabel("Sample")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def plot_final_error_runtime(results: list[dict]):
    """One final score per method against its complete measured wall time."""
    finite = [r for r in results if np.isfinite(r.get("final_err", np.nan))
              and np.isfinite(r.get("total_wall_time", r.get("total_time", np.nan)))]
    if not finite:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in finite:
        elapsed = result.get("total_wall_time", result["total_time"])
        marker = "D" if result.get("method_kind") == "bayesian" else "o"
        ax.scatter(elapsed, result["final_err"], marker=marker, label=result["name"])
    ax.set_xlabel("Total wall-clock time (s)")
    ax.set_ylabel("Final parameter error (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
