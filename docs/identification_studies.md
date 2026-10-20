# Initialization sensitivity and record-length studies

Run these from the repository root. The existing comparison experiments and
their output filenames are unchanged. Studies use the same configured update
rules, stopping rules, stability projections, and automatic loss targets as the
comparison runners. Unimplemented methods are retained as `skipped`.

## Console progress and estimated time remaining

Both studies log progress by default, independently of optimizer verbosity.
Messages identify the current initialization/realization, record length and FSM
resolution where applicable, and optimizer. Every method start and finish is
reported, including skipped/failed fits and early stopping. During a long fit,
iteration updates are printed at most once every 10 seconds, at completed
iteration boundaries. One slow iteration can therefore delay the next message.
Python output is flushed immediately, including when redirected to a log file.

The completed-fit count and percentage cover the entire study. Elapsed time
includes preparation and saving; the approximate ETA covers remaining method
fits. ETA warms up until every selected method has a completed timing sample,
then uses its observed seconds per sample and the remaining record lengths.
Different FSM resolutions, stopping behavior, and machine load can change the
estimate. Preparation, summarizing, and saving stages are also announced, followed
by the total elapsed time and saved path on completion.

Both blocks in `configs/studies.json` accept `progress` (default `true`) and
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
optimizer, with a dashed line at each true parameter. Initialization plots keep
the full range rather than clipping extreme finite outcomes. A separate stacked
count figure reports converged, stalled, iteration-limited, failed, and skipped
trials. Empty groups have no finite estimates. Finite stalled and iteration-limited
outputs are included in boxplots, so the count plot is part of their interpretation.
The existing diagnostic panels remain available for these results.

`study.initializations` stores the complete parameter bank (parameter by trial).
`all_runs` stores every trial's histories, settings, initialization/data hashes,
and outcomes. Equal seeds across MATLAB and NumPy do not produce identical
data or initializations; pairing is guaranteed within each backend.

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
Python also accepts `--runs`, `--iterations`, `--lengths`, `--cpu`, `--output`, and
`--study-config`. MATLAB accepts an `options.study` struct of overrides,
`options.study_config` (JSON path), `options.output_file`, and
`options.experiment_cfg` (a full experiment struct).

For each independent realization, the runners generate the longest record once
and fit nested prefixes. All methods, lengths, and FSM resolutions share those
prefixes. The input and innovations are independent Gaussian sequences; output
truth is generated with recursive IIR filtering, independently of the FSM fit's
frequency grid. Fitting retains the current zero-state boundary convention after
discarding the generator's burn-in. Thus boundary/transient effects are included
in the finite-record behavior. Changing the longest configured length can change
the noise realization; the full study settings and seeds define reproduction.

Python's `fsm_resolutions` compares fixed approximation settings (default:
128 coefficients/1024 frequencies and 256 coefficients/2048 frequencies).
Every record must be at least as long as both configured FIR responses to avoid
silently changing resolution at shorter lengths. PL has one `recursive_PL` case.
No FSM response length is optimized as a statistical parameter.

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
outputs including `stalled` and `max_iterations`. With `converged`, only outcomes
marked converged are used. **Both are conditional summaries when trials are
excluded**; outcome counts always refer to all attempted trials. Do not interpret
survivor-only means as the unconditional bias of a reliable estimator.

Figures show signed bias with Monte Carlo intervals, parameter RMSE, and outcome
fractions. Export also writes a CSV with statistics and counts. These intervals
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
% Optional: exercise MATLAB plots on freshly generated FSM study files:
test_identification_studies('path/to/initialization_fsm.mat','path/to/record_length_fsm.mat');
```

Tests check fixed data versus changing initialization, shared starts, stable
sampling, reproducibility, signed bias/MCSE calculations, nested record pairing,
MAT-file dimensions, outcome filtering, and plot dispatch. They use short records
and temporary outputs; they do not run the full manuscript experiments.
