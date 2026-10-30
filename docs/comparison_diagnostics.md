# Comparison initialization, outcomes, and diagnostics

This describes the first two resubmission preparation steps. Separate
[initialization and record-length runners](identification_studies.md) extend
the harness without changing its optimizer update rules. Existing result
files do not acquire diagnostics retroactively; generate new files to obtain them.

## Configuration

The comparison runners read `configs/example_1.json` or
`configs/example_CSTR.json`, together with the backend's `optim_configs.json`.
The existing experiment selection remains at the top of each runner. MATLAB's
`comparison` and `plot_comparison_results` are now functions callable without
arguments, with optional overrides for small runs or separate output files.

The default initialization policy is `legacy`: existing method-specific
initialization and numerical update rules are retained. In that policy,
`mode: physical` initializes latent signals separately from `theta_init`, as in
the original code. To give every method an explicit, physically consistent start,
add this experiment-level block (this six-parameter example is a unit-delay
linear starting model, not a recommended tuned initialization):

```json
"initialization": {
  "policy": "consistent",
  "theta0": [0, 0, 1, 0, 0, 0]
}
```

The parameter order is `[a1...ana, b1...bnb, f2...fnf, d1...dnd]`.
The vector must have the correct length, finite entries, and stable signal and
noise denominators. Both backends assign the whole vector to the optimizer,
derive `alpha` from its linear model, and initialize `e = c - beta(alpha)`.
Python uses its configured FSM approximation; MATLAB uses its recursive filter.
An experiment-level initialization replaces the method-level initialization.
Remove that experiment block to restore the original method-specific setup.

The following settings are supported at experiment or method level; experiment
settings override corresponding method settings:

```json
"stopping": {
  "gradient_tolerance": 1e-6,
  "loss_tolerance": 1e-8
},
"diagnostics": {
  "enabled": true,
  "save_matrices": false
}
```

Optionally set `diagnostics.parameter_scales` to one positive finite scale per
parameter. It reports conditioning of `S H S`, for `theta = S z`, alongside the
raw condition number. This is an observational diagnostic; it does not rescale
the optimization. Full derivative/matrix snapshots are opt-in because retaining
them for every Monte Carlo run increases file size.

## Stopping and iteration accounting

`K_max` keeps its original meaning: the maximum number of history samples,
including iteration zero, so there are at most `K_max - 1` parameter updates.
`settings.conv_threshold` still controls the relative-step threshold.

| Status | Meaning |
| --- | --- |
| `converged` | Small relative step and the applicable residual tolerance passed. |
| `stalled` | Small relative step without passing the residual tolerance. |
| `max_iterations` | The configured iteration budget was exhausted. |
| `invalid_update` | Non-finite proposed parameters, objective, or reconstructed state; the last finite parameters are retained for inspection. |
| `failed` | A numerical exception or invalid initial objective prevented completion. |
| `skipped` | The method is not implemented in this backend. |

Gradient-based FSM methods check the gradient at the accepted parameters. PL
methods check the gradient of the current fixed-regressor subproblem at the
accepted parameters. This PL test is not a claim of stationarity of a global
nonlinear objective. Loss-root methods check the residual relative to their
resolved target, with tolerance `loss_tolerance * max(1, abs(target))`.
The existing automatic loss target is retained and recorded; it depends on
the preceding methods and their order.

`update_count` counts attempted updates, `accepted_updates` counts accepted
ones, and `stopping_iteration` is the zero-based state index at a small-step
stop (`-1` otherwise). `iterations` is the stored history length for compatibility.
When fixed-length output is requested, the terminal state is padded as before,
but `active_mask` marks padding false and diagnostics are never padded.
Failures have `final_err = NaN`; their last finite parameters remain inspectable
but are excluded from Monte Carlo final estimates/errors. Finite stalled and
iteration-limited results remain in performance summaries, with their statuses
saved separately. Standard errors use the number of finite observations and are
undefined (`NaN`) for fewer than two observations. `param_err_count` and
`rmse_count` save those observation counts per method and history sample.

## Saved schema (version 2)

The existing top-level fields and original metric arrays remain available.
New top-level fields are `schema_version`, `resolved_experiment_json`, and
`all_runs`: a MATLAB cell array containing a complete result struct array for
each Monte Carlo realization (also present for a single run). `results` remains
the first realization for compatibility. Each method result adds:

- `theta0`, `est_loss_hist`, `active_mask`, update counts, and stopping outcome.
- `resolved_config_json`, including stopping/diagnostic defaults, iteration
  settings, resolved automatic loss targets, and the actual Python FSM lengths.
- `run_metadata`: data seed/run index when supplied by the experiment runner,
  backend/runtime information, SHA-256 hashes of data, initialization, and
  configuration, a `dataset_id`, and a deterministic `run_id`.
- `diagnostics`, `diagnostic_summary`, and optional `derivative_snapshots`.
- `total_wall_time`, `diagnostic_time`, and `total_time` (wall time minus the
  diagnostic extraction/export-to-memory interval). `total_time` includes
  objective/metric evaluation and stopping checks; it is not pure solver time.
  Python timing now includes per-fit configuration and initialization; file
  writing and plotting remain outside it. Legacy MATLAB optimizer timing
  retains its existing initialization convention. Cross-method final-score
  tables prefer `total_wall_time`, including diagnostics, over `total_time`.

### Bayesian final-only results

Python `BI_HMC` uses a separate sampler loop with `method_kind="bayesian"` and
`point_estimate="posterior_mean"`. Its optimizer histories are empty; iteration
summaries are NaN and HMC draws are never counted as optimizer updates. The
same original-coefficient `theta_hat`, terminal errors, and output RMSE enter
final comparison tables, boxplots and paired hypothesis tests. Posterior draws,
innovation-variance draws, sampler diagnostics and transition counts are saved
separately. Mixed MATLAB structs preserve the union of result fields, in any
method order.

`sampling_complete` records passing sampler diagnostics; `sampling_warning`
records unresolved diagnostics with finite draws. Only a valid physical point
estimate can enter finite-error summaries. A warning may be included in the
finite cohort, like a stalled/budget-limited optimizer, but is excluded from
the passed/converged cohort. Posterior distributions use their own plots and
are excluded from optimizer trajectories and Hessian figures.

`total_wall_time` includes the complete inference setup, initialization,
warm-up, retained sampling and diagnostics, plus terminal assessment in the
shared wrapper. `mc_summary.total_wall_time_runs` and
`mc_summary.final_output_rmse_runs` support comparisons without creating
Bayesian iteration curves. See [the implementation guide](bayesian_hmc_implementation.md)
for boundaries, physical priors, the exact-IIR forward model and configuration.

Data hashes use concatenated `r` and measured `c` as little-endian float64 bytes.
Equal seeds across MATLAB and Python do **not** imply equal generated data.
The identifiers allow checking actual data equality; this change does not add a
shared cross-language data generator. Use the saved configuration together with
the repository revision that produced a run for reproducibility.

FSM `est_loss_hist` is mean squared reconstructed innovation. PL
`est_loss_hist` is half the sum of squared residuals with the refreshed
regressor. PL additionally logs `update_loss_before/after` for each fixed
regressor subproblem. These objectives have different normalizations and
meanings; compare them within a backend. The existing RMSE is explicitly labeled
`true_innovation_reconstruction`; it is not a held-out prediction metric.

## Hessian interpretation

Each diagnostic sample includes `evaluation_iteration`. Sample zero is evaluated
at `theta0`, before update one. There is no additional Hessian evaluation after
the last update. Consequently, summary `*_condition_final` means the last
evaluated Hessian, which may precede the final stored parameter vector.

`raw_*` fields describe the FSM objective Hessian or PL `Phi' * Phi`.
`solver_*` fields describe the regularized full matrix or actual regularized /
safeguarded diagonal used for solving. Identity-operator GHAM has a raw Hessian
but no Hessian inverse. Loss-root GHAM uses a gradient-row inverse, recorded as
`solver_kind = gradient_row`, not a Hessian solve. Unregularized MATLAB RGLS
solves the rectangular least-squares system; its design condition is reported
separately. Gradient-only GGI does not acquire extra Hessian differentiation.

The metrics include 2-norm condition number, smallest/largest singular value,
numerical rank, rank tolerance, minimum/maximum signed eigenvalue, count of
negative eigenvalues, and symmetry error. Rank tolerance is
`n * machine_epsilon * sigma_max`. Rank deficiency is represented by `Inf`
condition number; missing diagnostics use `NaN` or absent fields. Negative
curvature is reported separately, since a well-conditioned Hessian can be
indefinite. Summaries retain initial, final, and peak condition, singular fraction,
negative-curvature fraction, and projection/rejection/solver-fallback counts.

Projection and fallback flags reflect actual update decisions. Diagnostics do
not add damping or modify the step. MATLAB reconstructs its small PL matrices
from the same regressor in the separately timed diagnostic interval. Python
uses detached derivatives already computed by the optimizer. The nominal
gradient-row denominator follows each existing implementation: Python uses
`g'g + lambda`, whereas MATLAB `lsqminnorm` uses `g'g + lambda^2`.

## Plotting and small runs

Run `plot_comparison_results` in MATLAB after generating results. It accepts old
files without diagnostics and recognizes both old `ok` and new outcome labels.
One diagnostic figure per parametrization overlays all methods in six panels,
using consistent method colors across PL and FSM. The top row contains objective, raw-Hessian condition, and solver-matrix
condition. The bottom row contains signed curvature, gradient norm, and step
norm. Optional scaled-Hessian curves are dotted in the raw-Hessian panel.
Singularity and event fractions remain available in the CSV summaries. Each figure exports as `optimization_<experiment>_<mode>_<pl|fsm>.pdf`.
The panels show objective, condition number, signed curvature, gradient
norm and step norm. Monte
Carlo curves use medians and interquartile bands over observed finite values;
singularities excluded from logarithmic condition curves remain in the saved
diagnostics and CSV summaries. Padded terminal samples are excluded. Partial failed runs are
retained in diagnostic plots, and per-run statuses/counts are in the CSV summary.

```matlab
addpath('mlab');
cfg = jsondecode(fileread('configs/example_1.json'));
cfg.mode = 'EXAMPLE';
cfg.settings.lambda_g = 64;
cfg.settings.K_max = 6;
cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
p = comparison(cfg, struct('output_file','results/diagnostic_check/pl.mat', ...
    'make_local_plots',false));
plot_comparison_results(struct('data_files',{{'results/diagnostic_check/pl.mat'}}, ...
    'fig_dir','figs/diagnostic_check','export_figures',true,'diagnostics_only',true));
```

Python's `run_method_set(..., experiment_cfg=cfg, run_metadata=...)` provides the
same programmatic override, and `save_comparison_results(..., all_runs=...,
resolved_experiment=cfg)` exports the complete run collection. The MATLAB helper
`plot_optimization_diagnostics` also returns figure handles and the summary table.

## Validation

With PyTorch, NumPy, SciPy, tqdm, and Matplotlib available:

```text
python -B -m unittest discover -s python -p "test_*.py" -v
```

In MATLAB:

```matlab
addpath('mlab');
test_comparison_diagnostics;
% Optional: also validate a freshly generated Python schema-2 result file.
test_comparison_diagnostics('path/to/fsm_result.mat');
```

These use small data sets and temporary output folders, cover singular and
indefinite matrices, explicit initialization, unchanged updates with diagnostics
disabled, stopping/rejection, all-run export, and MATLAB plots. They do not run
the manuscript-scale experiments.
