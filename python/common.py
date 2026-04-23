from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_valid_method_key(method_name: str) -> str:
    return method_name.replace("-", "_")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype() -> torch.dtype:
    return torch.float64


def now() -> float:
    return time.perf_counter()


def build_param_names(na: int, nb: int, nf: int, nd: int) -> list[str]:
    names = [f"a{i}" for i in range(1, na + 1)]
    names += [f"b{i}" for i in range(1, nb + 1)]
    names += [f"f{i}" for i in range(2, nf + 1)]
    names += [f"d{i}" for i in range(1, nd + 1)]
    return names


def build_plot_labels(na: int, nb: int, nf: int, nd: int) -> list[str]:
    labels = [f"a_{i}" for i in range(1, na + 1)]
    labels += [f"b_{i}" for i in range(1, nb + 1)]
    labels += [f"f_{i}" for i in range(2, nf + 1)]
    labels += [f"d_{i}" for i in range(1, nd + 1)]
    return labels


def print_result_header(param_names: list[str]) -> None:
    print("Iter    ", end="")
    for name in param_names:
        print(f"{name:<10}  ", end="")
    print(f"{'err(%)':<10}  {'time (s)':<10}")
    print("-" * (12 * (len(param_names) + 2)))


def plot_method_metrics(results: list[dict[str, Any]]) -> None:
    valid_results = [res for res in results if res["status"] == "ok"]
    if not valid_results:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)

    for color, result in zip(colors, valid_results):
        iter_axis = np.arange(result["iterations"], dtype=float)
        axes[0].plot(iter_axis, np.maximum(result["param_err"], np.finfo(float).eps), color=color, lw=1.6, label=result["name"])
        axes[1].plot(result["cum_time"], np.maximum(result["param_err"], np.finfo(float).eps), color=color, lw=1.6, label=result["name"])
        axes[2].plot(iter_axis, np.maximum(result["rmse_hist"], np.finfo(float).eps), color=color, lw=1.6, label=result["name"])
        axes[3].plot(result["cum_time"], np.maximum(result["rmse_hist"], np.finfo(float).eps), color=color, lw=1.6, label=result["name"])

    axes[0].set_title("Parameter Error vs Iteration")
    axes[1].set_title("Parameter Error vs Compute Time")
    axes[2].set_title("RMSE vs Iteration")
    axes[3].set_title("RMSE vs Compute Time")

    axes[0].set_ylabel("err (%)")
    axes[1].set_ylabel("err (%)")
    axes[2].set_ylabel("RMSE")
    axes[3].set_ylabel("RMSE")

    axes[0].set_xlabel("Iteration")
    axes[1].set_xlabel("Compute time (s)")
    axes[2].set_xlabel("Iteration")
    axes[3].set_xlabel("Compute time (s)")

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("Wiener System Identification - PyTorch Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()


def plot_parameter_trajectories(results: list[dict[str, Any]], theta_true: np.ndarray, param_labels: list[str]) -> None:
    valid_results = [res for res in results if res["status"] == "ok"]
    if not valid_results:
        return

    n = len(theta_true)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for idx in range(n):
        ax = axes[idx]
        for color, result in zip(colors, valid_results):
            iter_axis = np.arange(result["iterations"], dtype=float)
            ax.plot(iter_axis, result["theta_hist"][idx, :], color=color, lw=1.4, label=result["name"])
        ax.axhline(theta_true[idx], color="k", ls="--", lw=1.0, label="True")
        ax.set_title(f"Parameter {param_labels[idx]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(param_labels[idx])
        ax.grid(True)
        ax.legend(loc="best")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Parameter Trajectories Across Methods", fontsize=12, fontweight="bold")
    fig.tight_layout()
