"""Verify MATLAB win tests independently and build manuscript LaTeX tables.

Run after mlab/run_parameter_error_win_analysis.m. Input MAT files are read only.
Requires NumPy, SciPy, and statsmodels; no optimization experiments are rerun.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "manuscript"


def load_and_verify(study: str, benchmark: str, parametrization: str) -> dict:
    stem = f"parameter_error_wins_{benchmark}_{study}_{parametrization}"
    with (ROOT / "figs" / f"{stem}.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    info = json.loads((ROOT / "figs" / f"{stem}_metadata.json").read_text())
    data = loadmat(info["SourceFile"], simplify_cells=True,
                   variable_names=["experiment", "mc_summary"])
    summary = data["mc_summary"]
    names = list(np.atleast_1d(summary["method_names"]))
    errors = np.asarray(summary["final_errors"], dtype=float)
    estimates = np.asarray(summary["final_param_estimates"], dtype=float)
    truth = np.asarray(data["experiment"]["theta_true"], dtype=float).ravel()
    recomputed = 100 * np.linalg.norm(estimates - truth, axis=2) / np.linalg.norm(truth)
    np.testing.assert_allclose(errors, recomputed, rtol=1e-10, atol=1e-10, equal_nan=True)
    assert errors.shape[1] == 30, f"Expected 30 trials in {stem}"
    valid = np.isfinite(errors)
    if "statuses" in summary:
        valid &= np.isin(summary["statuses"], ["ok", "converged", "stalled", "max_iterations"])
    raw = []
    for row in rows:
        a, b = names.index(row["OptimizerA"]), names.index(row["OptimizerB"])
        joint = valid[a] & valid[b]
        x, y = errors[a, joint], errors[b, joint]
        tolerance = np.maximum(info["AbsoluteTieTolerance"],
            info["TieULPs"] * np.spacing(np.maximum(1, np.maximum(np.abs(x), np.abs(y)))))
        wins = int(np.sum(x-y < -tolerance))
        losses = int(np.sum(x-y > tolerance))
        ties = int(np.sum(np.abs(x-y) <= tolerance))
        assert (wins, losses, ties) == tuple(int(row[k]) for k in ("WinsA", "WinsB", "Ties"))
        assert wins+losses+ties == int(row["ValidPairs"])
        assert int(row["ValidPairs"]) + sum(int(row[k]) for k in
            ("AInvalidOnly", "BInvalidOnly", "BothInvalid")) == 30
        probability = binomtest(wins, wins+losses).pvalue if wins+losses else 1.0
        if not np.any(joint):
            assert np.isnan(float(row["RawP"]))
        else:
            np.testing.assert_allclose(float(row["RawP"]), probability, rtol=1e-12, atol=1e-15)
        raw.append(probability)
    _, adjusted, _, _ = multipletests(raw, alpha=info["Alpha"], method="holm")
    for row, p in zip(rows, adjusted):
        if np.isfinite(float(row["RawP"])):
            np.testing.assert_allclose(float(row["AdjustedP"]), p, rtol=1e-12, atol=1e-15)
            assert bool(int(row["Significant"])) == (p < info["Alpha"])
    print(f"Verified {stem}: {sum(int(r['Significant']) for r in rows)}/{len(rows)} significant")
    return dict(stem=stem, rows=rows, info=info, errors=errors, names=names,
                benchmark=benchmark, parametrization=parametrization, study=study)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def tex_p(value: float) -> str:
    if not np.isfinite(value):
        return r"\mathrm{NA}"
    if value >= 0.001:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"


def matrix_table(case: dict) -> str:
    info, rows = case["info"], case["rows"]
    active = [i for i, skip in enumerate(info["Skipped"]) if not skip]
    names = [info["MethodNames"][i] for i in active]
    labels = [info["MethodLabels"][i] for i in active]
    lookup = {(r["OptimizerA"], r["OptimizerB"]): r for r in rows}
    benchmark = "Example 1" if case["benchmark"] == "example_1" else "CSTR"
    study_label = "Monte Carlo" if case["study"] == "montecarlo" else "initialization ablation"
    key = f"{case['study']}_{case['benchmark']}_{case['parametrization']}"
    lines = [r"\begin{table}[!htbp]", r"\centering",
        rf"\caption{{Pairwise {study_label} results: {benchmark}, {case['parametrization'].upper()}. "
        r"Each upper-triangular entry gives row wins:losses:ties, followed by the Holm-adjusted $p$-value. "
        r"$\uparrow$ denotes a significant advantage for the row rule; $\downarrow$ for the column rule. "
        r"An unmarked $p$-value is inconclusive at $\alpha=0.05$.}",
        rf"\label{{tab:win_{key}}}", r"\begingroup", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{@{{}}l*{{{len(names)}}}{{c}}@{{}}}}",
        r"\toprule", "Rule [valid/30] & " + " & ".join(str(i+1) for i in range(len(names))) + r" \\",
        r"\midrule"]
    for i, (name, label) in enumerate(zip(names, labels)):
        cells = []
        for j, other in enumerate(names):
            if j < i:
                cells.append("")
            elif j == i:
                cells.append("--")
            else:
                row = lookup[(name, other)]
                p = float(row["AdjustedP"])
                if not np.isfinite(p):
                    cells.append(r"$\mathrm{NA}$")
                    continue
                arrow = ""
                if int(row["Significant"]):
                    arrow = r"\uparrow\," if row["Winner"] == name else r"\downarrow\,"
                counts = "{:}".join(row[k] for k in ("WinsA", "WinsB", "Ties"))
                cells.append(r"\shortstack{$" + counts + r"$\\$" + arrow + tex_p(p) + r"$}")
        n = info["FiniteCounts"][active[i]]
        lines.append(f"{i+1}: {tex_escape(label)} [{n}/30] & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\endgroup",
        r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize",
        r"The valid count includes finite usable estimates; a paired test uses only jointly valid runs. "
        r"Its effective binomial sample size is wins plus losses. "
        r"All finite extreme values are retained. "
        r"Blank lower-triangular cells omit symmetric duplicates."])
    excluded = [info["MethodLabels"][i] for i, skip in enumerate(info["Skipped"]) if skip]
    if excluded:
        lines.append("Unavailable rule: " + ", ".join(map(tex_escape, excluded)) + ".")
    lines.extend([r"\end{minipage}", r"\end{table}", ""])
    return "\n".join(lines)


MC_TEXT = r"""% Requires amsmath, amssymb, booktabs, and graphicx; merge win_test_references.bib.
% Generated from verified MATLAB CSV exports by python/report_parameter_error_wins.py.
\subsection{Statistical comparison across independent realizations}
\label{subsec:montecarlo_win_tests}
For each benchmark and parametrization, the optimizers were compared using
the same 30 independent input/noise realizations. The outcome for rule $a$
in realization $r$ was the final relative parameter error
\begin{equation}
 E_{a,r}=100\frac{\|\hat{\boldsymbol\theta}_{a,r}-\boldsymbol\theta^\star\|_2}
 {\|\boldsymbol\theta^\star\|_2}.
\end{equation}
For every unordered pair $(a,b)$, we counted the realizations in which each
rule produced the lower error and performed a two-sided exact paired sign
test~\cite{DixonMood1946}. Under the null hypothesis of equal win probabilities,
$W_{ab}\mid(W_{ab}+L_{ab})\sim\operatorname{Binomial}(W_{ab}+L_{ab},1/2)$;
ties were excluded from the binomial count. Numerical ties were defined by
$|E_{a,r}-E_{b,r}|\leq10\,\operatorname{eps}(\max\{1,E_{a,r},E_{b,r}\})$,
where $\operatorname{eps}(x)$ denotes floating-point spacing at $x$.
All finite outcomes, including extreme values, were retained. Each group
contained eight applicable rules and 28 comparisons; Holm's
correction~\cite{Holm1979} controlled the family-wise error rate at $0.05$
separately within each benchmark--parametrization combination.
Superiority denotes a significantly greater frequency of lower errors;
the test does not establish a lower arithmetic mean or a practically
important reduction. The legacy Monte Carlo files preserve pairing by
realization column, but do not contain per-run hashes for retrospective
verification. All applicable rules have 30 finite outcomes in these files.

Tables~\ref{tab:win_montecarlo_example_1_fsm}--\ref{tab:win_montecarlo_example_cstr_pl}
report all pairwise outcomes. For Example~1/FSM, none of the 28 comparisons
remained significant. For Example~1/PL, GHAM-2-I and LHAM-1-TIK outperformed
WS-GGI in 28/30 and 25/30 realizations, respectively
($p_{\mathrm H}=2.43\times10^{-5}$ and $8.77\times10^{-3}$); the other
26 comparisons were inconclusive. For CSTR/FSM, 26 comparisons were
significant. LHAM-1-TIK and LHAM-3-TIK each outperformed GI and all evaluated
GHAM rules, but neither showed a significant advantage over Newton
($p_{\mathrm H}=0.585$ and $0.401$). LHAM-1-TIK outperformed LHAM-3-TIK
in 23/30 realizations ($p_{\mathrm H}=0.0157$), although their median paired
error difference was only $1.97\times10^{-4}$ percentage points in favor
of LHAM-1-TIK. For CSTR/PL, 18 comparisons were significant; WS-GNI,
RLS, GHAM-1-dH, and GHAM-2-dH each outperformed LHAM-1-TIK in 27/30
realizations ($p_{\mathrm H}=1.26\times10^{-4}$).
Thus, the evidence supports specific update-rule advantages and does not
establish one optimizer as uniformly superior. Inconclusive comparisons
do not establish equivalence.

"""


INIT_TEXT = r"""% Requires amsmath, amssymb, booktabs, and graphicx; merge win_test_references.bib.
% Generated from verified MATLAB CSV exports by python/report_parameter_error_wins.py.
\subsection{Ablation: sensitivity to initialization}
\label{subsec:initialization_win_tests}
Initialization sensitivity was assessed separately using 30 randomly drawn
starting points per benchmark and parametrization, with a fixed input/noise
record. All rules received the same starting point within each paired trial;
the saved metadata were checked for shared data and initialization.
The final relative parameter error and two-sided exact paired sign test
were defined as in Section~\ref{subsec:montecarlo_win_tests}, with a separate
Holm family of 28 comparisons for each benchmark--parametrization combination
and $\alpha=0.05$. All finite extreme outcomes were included in the tests,
including observations omitted from the boxplot display.
Accuracy comparisons were conditional on both rules returning valid finite
estimates; failed runs were not counted as accuracy wins for their competitors.
The valid counts in the tables therefore qualify every superiority claim,
and these tests alone do not establish a higher probability of successful
completion or robustness across different data records.

Tables~\ref{tab:win_initialization_example_1_fsm}--\ref{tab:win_initialization_example_cstr_pl}
contain all comparisons. The numbers of significant pairs were 18/28 and
14/28 for Example~1/FSM and Example~1/PL, and 22/28 and 24/28 for CSTR/FSM
and CSTR/PL, respectively. For Example~1/PL, LHAM-3-TIK produced 30 finite
outcomes and significantly lower errors than RLS, WS-GNI, GHAM-1-dH,
GHAM-2-dH, WS-GGI, and GHAM-2-I on their jointly valid trials; its comparison
with LHAM-1-TIK was inconclusive. Conversely, for CSTR/PL, RLS, WS-GNI,
GHAM-1-dH, and GHAM-2-dH each significantly outperformed LHAM-3-TIK on
jointly valid trials. RLS and WS-GNI returned only 14 and 16 finite estimates
in each PL benchmark, whereas both LHAM variants returned 30.
For Example~1/FSM, GHAM-2-H and GHAM-2-I frequently attained lower errors
than the LHAM rules, but GHAM-2-I returned only 18 finite estimates and
GHAM-2-H included a very large finite error. Frequent accuracy wins therefore
do not imply protection against rare failures or extreme errors.
For CSTR/FSM, both LHAM rules significantly outperformed all evaluated GHAM
rules, while neither significantly outperformed Newton or GI.
The observed initialization sensitivity is consequently specific to the
benchmark, parametrization, and chosen robustness criterion.

"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for study, text in [("montecarlo", MC_TEXT), ("initialization", INIT_TEXT)]:
        cases = [load_and_verify(study, benchmark, parametrization)
                 for benchmark in ("example_1", "example_cstr")
                 for parametrization in ("fsm", "pl")]
        all_cases.extend(cases)
        # These numeric summaries are stated in the interpretation above.
        expected = [0, 2, 26, 18] if study == "montecarlo" else [18, 14, 22, 24]
        actual = [sum(int(r["Significant"]) for r in c["rows"]) for c in cases]
        if actual != expected:
            raise RuntimeError("Results changed: revise the manuscript interpretation before regenerating.")
        output = text + "\n".join(matrix_table(c) for c in cases)
        (DOCS / f"{study}_win_tests.tex").write_text(output, encoding="utf-8")
    print("Independent SciPy/statmodels verification passed for all eight cases.")
    print("Wrote montecarlo_win_tests.tex and initialization_win_tests.tex.")


if __name__ == "__main__":
    main()
