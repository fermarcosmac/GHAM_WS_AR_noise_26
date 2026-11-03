# Integration and verification notes

The source manuscript is `C:/Users/ferma/Downloads/GHAM_JFI_paper_REBUTTAL-2.pdf`.
Its printed page 26 (PDF page 28) contains the empty Sections 5.4 and 5.5.
The original PDF has not been modified.

## Insertions

- `section5_hypothesis_tests.tex` replaces the Section 5.4 placeholder. It includes
  five result tables and the four fresh Monte Carlo comparison matrices.
- `section5_initialization_statistics.tex` supplies the separate Section 5.5
  ablation requested earlier. It includes valid-output accounting, comparison
  tables, and the four initialization matrices.
- `section5_normality_figures.tex` optionally supplies the four full histogram
  figures as supplementary material. Normality results are already tabulated
  in the main subsection; these additional figures are not in the short preview.
- Merge `section5_statistics_references.bib` into the manuscript bibliography,
  keeping only one definition of citation keys already present.
- The snippets require `amsmath`, `amssymb`, `booktabs`, and `graphicx`. Figure
  paths are relative to the repository root. Upload the corresponding PDFs
  from `output/experimental_statistics/` when moving the source to another system.

The Experimental Results introduction currently mentions only a paired exact
sign test. Replace that sentence with:

> Additionally, paired exact sign tests and independent Wilcoxon rank-sum
> tests, selected according to whether the methods share the same observed
> records, were used to assess differences in final parameter error across
> the 30 Monte Carlo realizations (Section 5.4).

## What was checked

The current `run_parameter_error_win_analysis` presentation function was run
with `statistics_only=true` through its own implementation, exporting to a new
folder. No identification or HMC experiments were rerun; no input MAT files
or pre-existing figures were overwritten. The folder also holds normalized
sample snapshots, per-pair CSV tables, run-level tables, metadata, and a run log.

`python/write_experimental_statistics_section.py` reconstructs the error metric
from the saved physical parameter estimates, independently reproduces exact
binomial p-values (SciPy), independent rank-sum p-values (SciPy), and Holm
adjustments (statsmodels), and checks all exported normality results.
It then regenerates the LaTeX tables directly from the verified exports.
The interpretation paragraphs describe this result snapshot. If experiments
change, review the narrative when regenerating the tables; numerical
verification does not automatically update scientific interpretations.

## Interpretation limits retained in the draft

- Shapiro-Wilk is a marginal distribution diagnostic. It neither verifies
  independence/pairing nor proves that a nonparametric test is correct. A
  paired t-test concerns the distribution of paired differences, which was
  not tested here. Test choice was not switched according to normality p-values.
- Rank-sum is used for the separate Python BI-HMC reference against MATLAB PL
  samples. Its null is equal distributions. The rank effect supplies direction;
  it does not prove a uniform quantile ordering or an arithmetic-mean difference.
- Reused BI-HMC values in PL/FSM are the same Bayesian experiment, not additional
  independent repetitions. Thus the normality table has 36 method/case entries
  but only 34 distinct sample sets (30 of those 34 reject normality).
- The current Example 1/PL Monte Carlo exporter retains GHAM-2-H as an
  unavailable row: 45 planned contrasts, 36 testable, nine unit adjustment
  inputs. This conservative Holm family is preserved exactly and described
  in the table note. It must not be reported as 45 performed tests.
- Some legacy PL files lack realization hashes. Their optimizer pairing relies
  on the saved column convention; the manuscript explicitly discloses this.
- In the initialization study, Python and MATLAB use different fixed records.
  Cross-implementation rank-sum contrasts therefore do not isolate robustness
  to initialization under identical observed data. Paired FSM contrasts do.
- Finite-output counts include stalled/iteration-limited results and valid
  Bayesian point estimates with sampler warnings, following the existing
  result policy. Error superiority is conditional on valid outputs and is not
  a success-rate test, computational-cost test, or consistency proof.

## Reproduction

From the repository root, run MATLAB with `mlab` on its path:

```matlab
run_parameter_error_win_analysis(struct( ...
    'fig_dir',fullfile(pwd,'output','experimental_statistics'), ...
    'show_figures',false));
```

The normalized JSON snapshots used by the independent verifier were exported
by `tmp/run_current_statistics.m`. To refresh them together with the figures,
run that script. Then:

```text
python python/write_experimental_statistics_section.py
pdflatex -output-directory=tmp/section5_statistics docs/manuscript/section5_statistics_preview.tex
bibtex tmp/section5_statistics/section5_statistics_preview
pdflatex -output-directory=tmp/section5_statistics docs/manuscript/section5_statistics_preview.tex
pdflatex -output-directory=tmp/section5_statistics docs/manuscript/section5_statistics_preview.tex
```

The final compiled preview is `output/pdf/experimental_statistics_subsections.pdf`.
