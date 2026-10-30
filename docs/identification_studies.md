# Initialization, noise sensitivity, and record-length studies

Run these from the repository root. The existing comparison experiments and
their output filenames are unchanged. Studies use the same configured update
rules, stopping rules, stability projections, and automatic loss targets as the
comparison runners for optimization methods. Python also supports the separate
`BI_HMC` sampler described below. Unimplemented methods are retained as `skipped`.

## Including BI_HMC in Python studies

`BI_HMC` participates in initialization, noise-sensitivity and record-length
studies through the shared method dispatcher, with its own HMC sampling loop.
It estimates the original physical coefficients with an exact causal IIR model,
not the finite-frequency FSM approximation used by the optimization methods.
The saved backend tag remains `FSM` for compatibility with the Python result
readers; `run_metadata.forward_model: exact_iir` identifies the actual model.

The final estimator is the posterior mean of the retained physical coefficient
draws. Parameter errors, signed bias, RMSE, final-error boxplots and hypothesis
tests operate on that one estimate per dataset/trial, just as they operate on
the terminal estimate of an optimizer. A posterior draw is not a new dataset or
an optimization update. HMC therefore has no synthetic `theta_hist`, objective
history or optimizer-conditioning trajectory.

### Select methods and configure the separate sampling budget

All three Python study CLIs accept an exact `--methods` list:

```text
python python/initialization_study.py --experiment example_1 --methods WS-GGI BI_HMC --runs 3 --cpu --output results/local/initialization_bi.mat
python python/noise_study.py --experiment example_1 --methods WS-GGI BI_HMC --runs 3 --noise-levels 0.05 0.1 0.2 --cpu --output results/local/noise_bi.mat
python python/record_length_study.py --experiment example_1 --methods WS-GGI BI_HMC --runs 3 --lengths 256 512 1024 --cpu --output results/local/record_length_bi.mat
```

These commands use the experiment's configured record length and full HMC
settings, so they are not smoke tests. Run the short single-example smoke command
in the [README](../README.md#bayesian-inference-with-bi_hmc) first.
The same exact selection can be set in an individual Python study block:

```json
{
  "methods": ["WS-GGI", "BI_HMC"],
  "n_runs": 3,
  "data_seed": 271828,
  "summary_population": "finite",
  "save_case_results": true
}
```

Merge these fields into the relevant `<kind>_study` object; retain that study's
required grid/distribution fields. A `methods` override disables automatic
addition of the two legacy Python extra methods. Without an override, selection
continues to come from the benchmark configuration; HMC is not added implicitly.

Sampler settings come from `methods.BI_HMC` in the backend method JSON,
normally `python/optim_configs.json`. The study's `K_max` / CLI `--iterations`
only changes optimizer update budgets. It does not change HMC `chains`, `warmup`,
`samples` or `leapfrog_steps`. Override sampler settings locally with a flat
`bayesian_hmc` object inside the relevant study block, for example:

```json
{
  "methods": ["BI_HMC"],
  "bayesian_hmc": {
    "chains": 2,
    "warmup": 10,
    "samples": 20,
    "leapfrog_steps": 3,
    "initial_optimization_steps": 3
  }
}
```

This is a smoke-only budget, insufficient for posterior inference. The CLI
`--hmc-config` accepts a JSON file containing just that flat override object,
without a `bayesian_hmc` wrapper; it merges over study-file overrides. For
example, after saving the inner object as `results/local/hmc_smoke.json`:

```text
python python/record_length_study.py --experiment example_1 --methods BI_HMC --runs 1 --lengths 32 64 --hmc-config results/local/hmc_smoke.json --cpu --output results/local/record_length_bi_smoke.mat
```

The production method JSON is unchanged. The API accepts the same
`study["bayesian_hmc"]` dictionary. A complete alternative method file can still
be selected through `cfg["python_optim_config"]`.
The [implementation guide](bayesian_hmc_implementation.md) documents all priors,
adaptation controls, diagnostics and sampling-budget settings.

### Random streams, initial conditions and computational comparison

The data generator uses its own NumPy stream. Random initializations use a
separate stream controlled by `initialization_seed`. The sampler uses the
configured HMC seed plus a deterministic offset for the initialization or data
realization index. Noise levels, record lengths and FSM-resolution cases reuse
the sampler stream for the same realization. This pairing does not make the
posteriors identical: each is conditioned on its own observed record.
Actual per-run sampler seeds are saved, independently of the data seed.

Each random initialization supplies the same physical starting vector to every
selected method for that trial, including the starting center for the HMC
chains. Any configured HMC initial optimization, chain jitter and warm-up remain
part of that method and its measured runtime. The initialization experiment
keeps the measured record fixed; it assesses computational sensitivity, not
finite-sample bias across independent records.

The generator starts the full signal and noise filters from zero and then
discards `burn_in` samples. Python now preserves the observed input prefix as
`example_data.input_prehistory` and the discarded count as
`example_data.noise_burn_in`. HMC replays this input with each candidate's
coefficients, starting at the known zero signal state. It uses the exact finite
initial noise covariance after that burn-in, computed from the candidate noise
coefficients. Neither the true latent signal state nor the true innovation
prehistory is supplied to inference. Shorter record-length prefixes keep the
entire observed input prehistory. Existing optimization methods retain their
zero-state fitting convention at the retained record's start; these policies
are recorded separately in `study.initial_condition_policies`.

Use per-fit **total wall-clock time**, including initialization, warm-up,
retained sampling, diagnostics and final scoring. The study arrays
`total_wall_time` and `total_time_runs` both use the method's complete
`total_wall_time`, not optimizer time with diagnostic work subtracted. Saving and
plot generation are outside per-method fit timing. Compare these totals, not
HMC time per transition against optimizer time per update. Console progress
names HMC warm-up/sampling transitions explicitly and keeps completed-fit counts
common to both method types.

### Validity cohorts and saved posterior information

- `sampling_complete`: finite posterior result and the configured sampler
  diagnostics passed.
- `sampling_warning`: a posterior result exists, but diagnostics are unresolved;
  a finite, valid point estimate can still be evaluated in the default finite
  cohort. This is not a claim of satisfactory posterior exploration.
- `failed`: no usable inference result. A nonfinite or unstable posterior-mean
  point estimate has `point_estimate_valid: false` and is excluded from scores;
  it is not silently projected into the stable domain.

`summary_population: finite` includes valid finite estimates from both sampling
statuses, as well as finite stalled/iteration-limited optimizers.
`summary_population: converged` includes optimizer `converged` and HMC
`sampling_complete` only. `converged_count` follows this combined passing cohort;
`sampling_complete_count` and `sampling_warning_count` retain the separate HMC
outcomes. Counts always refer to attempted trials. Tight-budget smoke runs may
pass export checks while correctly retaining `sampling_warning`.

Python study aggregates add arrays with the same axes as `statuses`:

| Array | Meaning |
| --- | --- |
| `total_wall_time`, `total_time_runs` | Full fit seconds for each method and trial. |
| `point_estimate_valid` | Whether the returned final estimate is eligible for scoring. |
| `is_bayesian` | Distinguishes posterior inference from optimization. |
| `sampler_diagnostics_passed` | HMC diagnostic pass flag; false for non-HMC methods. |
| `sampling_seeds` | Actual HMC seed; NaN for optimization methods. |
| `sampler_diagnostics_json` | Per-fit diagnostics; undefined numeric values are JSON `null`. |

For noise studies the axes are level by method by realization. For record-length
studies they are resolution by length by method by realization. Initialization
stores these fields in `study` with method by initialization axes, alongside
the full `all_runs` comparison payload.

Initialization files always retain every posterior in `all_runs`. Noise and
record-length aggregates retain final estimates and diagnostic summaries;
`save_case_results: true` additionally saves full comparison case files and their
paths in `study.case_files`. Each HMC case result contains `posterior_samples`
(chain by retained draw by coefficient), `posterior_variance_samples` (chain by
retained draw), the point estimate, diagnostics and reproducibility metadata.
Set `save_case_results` false only when full posterior retention is unnecessary.
The aggregate's `posterior_storage` describes this policy. Repeated-data bias
intervals still describe uncertainty of the mean estimator error across trials;
they are not the credible intervals within one fitted posterior.

## Console progress and estimated time remaining

All studies log progress by default, independently of optimizer verbosity.
Messages identify the current initialization/realization, noise level, record length and FSM
resolution where applicable, and method. Every method start and finish is
reported, including skipped/failed fits and early stopping. During a long fit,
iteration updates are printed at most once every 10 seconds, at completed
iteration boundaries. HMC reports its warm-up/sampling transitions rather than
accepted optimizer updates. One slow update or HMC transition can therefore
delay the next message.
Python output is flushed immediately, including when redirected to a log file.

The completed-fit count and percentage cover the entire study. Elapsed time
includes preparation and saving; the approximate ETA covers remaining method
fits. ETA warms up until every selected method has a completed timing sample,
then uses its observed seconds per sample and the remaining record lengths.
Different FSM resolutions, stopping behavior, and machine load can change the
estimate. Preparation, summarizing, and saving stages are also announced, followed
by the total elapsed time and saved path on completion.

All study blocks in `configs/studies.json` accept `progress` (default `true`) and
`progress_interval_seconds` (positive, default `10`). For example:

```matlab
initialization_study('example_1',struct('study',struct('progress_interval_seconds',30)));
% Disable study progress messages:
record_length_study('example_1',struct('study',struct('progress',false)));
```

```text
python python/record_length_study.py --experiment example_1 --no-progress
```

`--verbose` still enables the existing optimizer detail; it is not needed to see
study progress. Console logging adds a small amount of wall-time overhead without
changing optimizer updates, stopping rules, seeds, or run identifiers.

## Random initialization, fixed data

```matlab
addpath('mlab');
initialization_study('example_CSTR');
```

```text
python python/initialization_study.py --experiment example_CSTR
```

These save `results/example_CSTR/initialization_pl.mat` and
`initialization_fsm.mat`, respectively. Both support `example_1` as well.
Each backend generates one input/noise/output record. Every trial reuses that
record, and all optimizers receive the same parameter vector for that trial.
Latent signals are initialized consistently with that vector (`alpha` from the
linear model and `e = c - beta`). The polynomial's fixed linear coefficient is
not randomized. This measures sensitivity to parameter initialization on a
single realized data set, not bias over independent measurement records.

Edit `initialization_study` in `configs/studies.json`:

- `n_runs`: number of distinct random trials (default 30).
- `initialization_seed`: random stream for parameter draws; independent of data.
- `data_seed`: optional fixed data seed; otherwise use the experiment seed.
- `center`: a parameter vector, or `null` for `[a=0, b1=1, remaining b=0, f=0, d=0]`.
- `scale`: scalar or one entry per parameter. For `uniform`, this is the half-width
  around `center`; for `normal`, it is the standard deviation.
- `distribution`: `uniform` or `normal`.
- `stability_radius`: reject draws with any signal/noise pole outside this radius.
  Rejection conditions the specified distribution on stability; coefficients are
  not clipped or projected during sampling.
- `max_draws`: total proposal limit. If too few stable proposals can be obtained,
  the runner reports an error instead of silently reducing the trial count.
- `K_max`: optional override of the experiment's iteration budget.

For example, a small run without changing configuration files:

```matlab
initialization_study('example_1',struct('study',struct('n_runs',5,'K_max',20)));
```

```text
python python/initialization_study.py --experiment example_1 --runs 5 --iterations 20 --cpu
```

Plot both backends after running them:

```matlab
plot_comparison_results(struct('study','initialization','experiment_name','example_CSTR'));
```

One figure per parametrization shows **final parameter estimates**, grouped by
optimizer, with a dashed line at each true parameter.
An ordinary `plot_comparison_results` call also includes available initialization
files for the selected experiment by default. Use `include_initialization: false`
to disable this discovery; an explicit `data_files` selection is always respected.
The boxplot window is named `<PL|FSM> initialization final parameter estimates`. Initialization plots keep
the full range rather than clipping extreme finite outcomes. A separate stacked
count figure reports converged, stalled, iteration-limited, failed, and skipped
trials. Empty groups have no finite estimates. Finite stalled and iteration-limited
outputs are included in boxplots, so the count plot is part of their interpretation.
The existing diagnostic panels remain available for these results.

`study.initializations` stores the complete parameter bank (parameter by trial).
`all_runs` stores every trial's histories, settings, initialization/data hashes,
and outcomes. Equal seeds across MATLAB and NumPy do not produce identical
data or initializations; pairing is guaranteed within each backend.

## Noise sensitivity, paired records, fixed initialization

```matlab
noise_study('example_1');
```

```text
python python/noise_study.py --experiment example_1
```

Both benchmarks are supported. Configure `noise_study` in `configs/studies.json`:

- `sigma_nu_values`: innovation standard deviations, default `[0.05, 0.1, 0.2, 0.5, 1.0]`.
  Values must be finite, nonnegative, unique, and increasing; zero noise is allowed.
- `n_runs`: runs **per optimizer and noise level**, default 1. Increase this
  for independent data realizations at each level.
- `data_seed`: stream seeds start here and increase by one per realization.
- `theta0`: fixed common parameter start; `null` uses the nominal unit-delay start.
- `K_max`: optional iteration-budget override; record length comes from the benchmark.
- `summary_population`: `finite` (default) or `converged`, as for record length below.
- `save_case_results`: default `true`, retaining full comparison histories at each level.

MATLAB accepts the shared `options.study`, `study_config`, `experiment_cfg`,
`method_configs`, `output_file`, and `verbose` options. Python exposes
`run_noise_study(cfg, study, ...)` and accepts the shared study CLI options plus
`--noise-levels` to override the standard deviations.

Within each backend, each realization uses the same Gaussian input and unit-variance
innovation draws at every level. Resetting the same stream also preserves the
pre-burn-in noise samples. Recursive IIR filtering generates truth independently
of FSM approximation, and the standard deviation scales innovations **before**
the AR noise filter. Thus `sigma_nu**2` is innovation power, rather than the
variance of the colored output disturbance. The initial parameter vector, record
length, methods, FSM resolution, and optimizer settings stay fixed across levels.
Latent signals are initialized consistently for each observed record.

The aggregate files `results/<experiment>/noise_pl.mat` and `noise_fsm.mat` use
the same schema in both backends:

- `study.final_param_estimates`: level by method by realization by parameter.
- `study.statuses`, `dataset_ids`, `resolved_configs`: level by method by realization.
- `study.bias`, `variance`, `rmse`, `mcse`, `ci_low`, `ci_high`, `count`:
  level by method by parameter.
- `study.finite_count`, `selected_count`, `converged_count`, `failed_count`,
  `skipped_count`: level by method.
- `study.sigma_nu_values`, `innovation_variances`, `data_seeds`, `theta0`,
  `theta_true`, `method_names`, `record_length`, `case_files`, and `config_json`:
  sweep coordinates and reproducibility information.

MATLAB may omit trailing singleton dimensions; SciPy's `simplify_cells=True`
also squeezes singleton axes. Failed/skipped estimates remain NaN and their
statuses remain visible. Bias intervals describe Monte Carlo uncertainty of the
mean parameter error, conditional on the selected population. Matched levels are
correlated; no independent-level significance test is claimed.

By default, `noise_<pl|fsm>_sigma01.mat` through `_sigma05.mat` retain full
comparison payloads, including every trial's histories, data IDs, initialization,
diagnostics, and per-run noise-level metadata. Indices follow `sigma_nu_values`;
the second default file uses 0.1. Custom output filenames supply the corresponding
prefix. Set `save_case_results` false for aggregate-only runs.

Plot the complete sweep for both parametrizations without running identification:

```matlab
plot_comparison_results(struct('study','noise','experiment_name','example_1', ...
    'fig_dir',fullfile('figs','local'),'export_figures',true));
```

This reads `noise_pl.mat` and `noise_fsm.mat`. Each backend has a figure of final
relative parameter error versus innovation standard deviation, with one panel
per optimizer, and a figure of final parameter estimates with dashed true-value
references. The default single run shows points and curves without MC intervals.
With repeated runs, plots show means and pointwise approximate 95% MC intervals
only where at least two estimates are retained. Overall error averages the
per-run relative norms, `100*norm(theta_hat-theta_true)/norm(theta_true)`.
The configured summary population is respected; failed/skipped estimates are
excluded and retained counts are shown where fits are missing.

Exports are `noise_error_<experiment>_<pl|fsm>.pdf`,
`noise_estimates_<experiment>_<pl|fsm>.pdf`, and
`noise_summary_<experiment>_<pl|fsm>.csv`. The CSV includes retained counts and
optimization outcomes at each level. Direct use is
`[figures, report] = plot_noise_study(payload_or_filename, options)`.

Plot individual levels using the comparison plotter with explicit files:

```matlab
files = {fullfile('results','example_1','noise_pl_sigma02.mat'), ...
         fullfile('results','example_1','noise_fsm_sigma02.mat')};
plot_comparison_results(struct('data_files',{files}, ...
    'fig_dir',fullfile('figs','local','noise_sigma02'), ...
    'win_test',struct('enabled',false),'normality',struct('enabled',false)));
```

Matching seed numbers across MATLAB and Python do not produce identical records.

## Record length, independent data, fixed initialization

After choosing an initialization and adequate iteration budget from the first
study, put that vector in `record_length_study.theta0` in `configs/studies.json`.
`null` uses the nominal unit-delay start described above, never the true vector.

```matlab
record_length_study('example_CSTR');
```

```text
python python/record_length_study.py --experiment example_CSTR
```

Results are `record_length_pl.mat` and `record_length_fsm.mat` in the experiment's
results folder. Plot them using:

```matlab
plot_comparison_results(struct('study','record_length','experiment_name','example_CSTR'));
```

The study configures `n_runs`, `data_seed`, increasing `record_lengths`, `theta0`,
optional `K_max`, `summary_population`, and `save_case_results`.
Python also accepts `--methods`, `--hmc-config`, `--runs`, `--iterations`, `--lengths`, `--cpu`, `--output`, and
`--study-config`. MATLAB accepts an `options.study` struct of overrides,
`options.study_config` (JSON path), `options.output_file`, and
`options.experiment_cfg` (a full experiment struct).

For each independent realization, the runners generate the longest record once
and fit nested prefixes. All methods, lengths, and FSM resolutions share those
prefixes. The input and innovations are independent Gaussian sequences; output
truth is generated with recursive IIR filtering, independently of the FSM fit's
frequency grid. The optimization methods retain the current zero-state boundary
convention after discarding the generator's burn-in; HMC uses the preserved
input prehistory and finite initial noise covariance described above.
Thus boundary/transient effects are included
in the finite-record behavior. Changing the longest configured length can change
the noise realization; the full study settings and seeds define reproduction.

Python's `fsm_resolutions` compares fixed approximation settings (default:
128 coefficients/1024 frequencies and 256 coefficients/2048 frequencies).
When FSM optimization methods are selected, every record must be at least as long as both configured FIR responses to avoid
silently changing resolution at shorter lengths. PL has one `recursive_PL` case.
No FSM response length is optimized as a statistical parameter. HMC does not
depend on this grid. Resolution axes remain present for paired comparisons;
HMC-only studies do not impose the optimizer's minimum FIR-response length.
For a given record, the repeated resolution axis therefore represents the same
HMC model and paired sampler stream. Select one resolution label when those
duplicate HMC fits are unnecessary.

The saved arrays contain, for each resolution, length, method, and parameter:

- signed mean error `bias = mean(theta_hat - theta_true)`;
- sample `variance`, empirical parameter `rmse`, and finite `count`;
- `mcse = sample_std / sqrt(count)` and pointwise normal-approximation intervals
  `bias +/- 1.96*mcse` (undefined with fewer than two estimates);
- counts of finite outputs, converged outputs, failures, and skips;
- all final estimates, statuses, resolved per-run optimizer settings, and data IDs.

Statistics arrays are resolution by length by method by parameter; estimate
arrays additionally have a realization dimension before parameter. Count arrays
without a parameter coordinate are resolution by length by method.

With `summary_population: finite` (default), statistics describe finite returned
outputs including `stalled`, `max_iterations` and valid `sampling_warning`.
With `converged`, only optimizer `converged` and HMC `sampling_complete`
outcomes are used. **Both are conditional summaries when trials are
excluded**; outcome counts always refer to all attempted trials. Do not interpret
survivor-only means as the unconditional bias of a reliable estimator.

Figures show signed bias with Monte Carlo intervals and parameter RMSE, with
`box on` for each axes. Outcome diagnostics are tables, one per resolution, with
counts and percentages by record length and method. They separate converged,
stalled, iteration-limited, failed/invalid, and skipped trials, and report finite
outputs and the population actually used in the bias/RMSE summary. The optional
second output of `plot_record_length_study` is the numeric outcome table. Export
also writes a separate `record_length_outcomes_<experiment>_<pl|fsm>.csv` and
replaces the former outcome-curve PDFs with table PDFs. The parameter-statistics
CSV keeps its existing `FailedOrSkipped` column for compatibility. These intervals
quantify Monte Carlo uncertainty of the mean error; they are not confidence
intervals for parameters from one measured record. Shared prefixes correlate
lengths; no independent-length significance test is performed.

`inverse_n_coefficients[...,0:2]` (Python indexing) stores an unweighted descriptive
fit `bias(N) = intercept + slope/N` where at least three usable lengths exist.
It has no inferred standard errors. A nonzero intercept is not a proof of
inconsistency: finite FSM error, insufficient convergence, constraints, and local
minima can also leave persistent offsets. Tighten optimization and compare FSM
resolutions before attributing the observed error to statistical finite-sample bias.

`save_case_results: true` additionally writes the full comparison histories for
each length/resolution. Otherwise the aggregate retains terminal data/settings
without the much larger histories. Existing comparison results are never used
as a substitute for newly generated study records.

The number of realizations should be chosen from the achieved MCSE; the default
100 is an initial experiment size, not a precision guarantee. Increase it until
the intervals can resolve the signed bias of interest. Increasing record length
alone does not improve Monte Carlo precision relative to a bias that shrinks
faster than the standard deviation.

## Small validation runs

```text
python -B -m unittest discover -s python -p "test_*.py" -v
```

```matlab
addpath('mlab');
test_identification_studies;
test_noise_study;
test_plot_noise_study;
% Optional: exercise MATLAB plots on freshly generated FSM study files:
test_identification_studies('path/to/initialization_fsm.mat','path/to/record_length_fsm.mat');
```

Tests check fixed data versus changing initialization, shared starts, stable
sampling, reproducibility, signed bias/MCSE calculations, nested record pairing,
MAT-file dimensions, outcome filtering, and plot dispatch. They use short records
and temporary outputs; they do not run the full manuscript experiments.
