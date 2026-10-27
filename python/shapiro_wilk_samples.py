"""Shapiro-Wilk backend for MATLAB versions without swtest (before R2026b).

MATLAB passes finite samples through temporary JSON files. This uses SciPy's
Shapiro-Wilk implementation, not the different Shapiro-Francia test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import shapiro


def analyze(samples):
    statistics, probabilities, statuses = [], [], []
    for sample in samples:
        values = np.asarray(sample, dtype=float).ravel()
        if not np.all(np.isfinite(values)):
            raise ValueError("The MATLAB caller must supply finite samples only.")
        w, p = None, None
        if len(values) < 3:
            status = "insufficient samples"
        elif len(values) > 5000:
            status = "sample size exceeds calibrated range"
        elif np.ptp(values) == 0:
            status = "constant sample; test undefined"
        else:
            result = shapiro(values)
            w, p = float(result.statistic), float(result.pvalue)
            status = "tested"
        statistics.append(w)
        probabilities.append(p)
        statuses.append(status)
    return {"W": statistics, "P": probabilities, "Status": statuses,
            "Engine": "scipy.stats.shapiro", "Version": scipy.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload["samples"])
    args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
