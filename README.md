# Wiener-system identification with AR output noise

MATLAB implements the PL parametrization; Python/PyTorch implements the FSM
parametrization. The repository includes individual examples, optimizer
comparisons, initialization-sensitivity experiments, noise-sensitivity studies, record-length studies, and
MATLAB plots for results from both backends.

Run the commands below **from the repository root**. Full experiments can be
expensive: check `lambda_g`, `K_max`, the number of repetitions, and the selected
methods before launching them. The short examples below override these settings
in memory and write to ignored `results/local/` and `figs/local/` directories.

## Contents

- [Installation and layout](#installation-and-layout)
- [Choose an entry point](#choose-an-entry-point)
- [Single examples and comparisons](#single-examples-and-comparisons)
- [Initialization sensitivity](#initialization-sensitivity)
- [Noise sensitivity](#noise-sensitivity)
- [Record length and finite-sample bias](#record-length-and-finite-sample-bias)
- [Configuration reference](#configuration-reference)
- [Method availability and controls](#method-availability-and-controls)
- [Ablations and parameter sweeps](#ablations-and-parameter-sweeps)
- [Plotting and exports](#plotting-and-exports)
- [Results and reproducibility](#results-and-reproducibility)
- [Validation and repository hygiene](#validation-and-repository-hygiene)

## Installation and layout

For Python, use an environment with PyTorch, NumPy, SciPy, Matplotlib, and tqdm:

```text
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) lists direct dependencies; it is not a
locked environment for reproducing the paper. Use a suitable PyTorch installation
for your CPU/CUDA platform. FSM derivatives require PyTorch automatic
differentiation, including higher derivatives for some methods. The comparison
runner normally chooses CUDA when available and uses float64; the study CLIs
provide `--cpu`.

For MATLAB, add `mlab` to the path. Use a release supporting functions such as
`jsondecode`, `tiledlayout`, `exportgraphics`, and `lsqminnorm`. The test commands
at the end check the installed environment on short experiments.

| Location | Purpose |
| --- | --- |
| [configs/](configs/) | Shared benchmark definitions and study settings. |
| [mlab/](mlab/) | PL runners, optimizer configuration, tests, and plots for both backends. |
| [python/](python/) | FSM runners, automatic derivatives, optimizer configuration, and tests. |
| [results/](results/) | Saved experiment MAT files, grouped by benchmark. |
| [figs/](figs/) | Exported figures and source diagrams. |
| [docs/comparison_diagnostics.md](docs/comparison_diagnostics.md) | Initialization, stopping, diagnostic definitions, and result schema. |
| [docs/identification_studies.md](docs/identification_studies.md) | Study design, statistical interpretation, and saved array dimensions. |
| [docs/manuscript/](docs/manuscript/) | Manuscript sources and section previews. |

## Choose an entry point

Open [mlab/entry.m](mlab/entry.m) for a commented menu of MATLAB experiment,
study, ablation, and plotting calls. Select a benchmark in its setup section,
run that section, then uncomment and run the block you need.

| Task | MATLAB / PL | Python / FSM |
| --- | --- | --- |
| Interactive standalone demo | [example.m](mlab/example.m) | [example_torch.py](python/example_torch.py) |
| One record or Monte Carlo method comparison | [comparison.m](mlab/comparison.m) | [comparison_torch.py](python/comparison_torch.py) |
| Fixed data, randomized parameter starts | [initialization_study.m](mlab/initialization_study.m) | [initialization_study.py](python/initialization_study.py) |
| Innovation standard deviation, paired records | [noise_study.m](mlab/noise_study.m) | [noise_study.py](python/noise_study.py) |
| Independent records at several lengths | [record_length_study.m](mlab/record_length_study.m) | [record_length_study.py](python/record_length_study.py) |
| Compare methods, estimates, and conditioning | [plot_comparison_results.m](mlab/plot_comparison_results.m) | Load the Python MAT files in the MATLAB plotter. |
| Plot bias, RMSE, and outcome tables | [plot_record_length_study.m](mlab/plot_record_length_study.m) | Load the Python MAT files in the MATLAB plotter. |

The standalone demos and ordinary Python comparison script have **no command-line
argument parser**. Select their experiment in the source, or use the APIs below.
The three Python study scripts have command-line options and `--help`.

## Single examples and comparisons

### Standalone interactive demos

```matlab
run(fullfile('mlab','example.m'));
```

```text
python python/example_torch.py
```

The MATLAB demo currently selects `example_1`; Python selects `example_CSTR`.
Change `experiment_name` inside the relevant script to switch benchmark.
These demos display figures and retain their original initialization and
relative-step stopping behavior; they do not save the comparison MAT schema.
MATLAB's demo explicitly implements WS-GGI, even if its configured method label
is changed. Python selects `python_example_method`, then `example_method`, then
its built-in WS-GGHAM-1-dH default. Experiment-level initialization, stopping, and
diagnostic overrides described below belong to the comparison/study runners.
For a saved, consistently initialized single example, use `mode = EXAMPLE` in a
comparison instead.

### MATLAB comparison: short, saved example

```matlab
addpath('mlab');
cfg = jsondecode(fileread('configs/example_1.json'));
cfg.mode = 'EXAMPLE';
cfg.selected_methods = {'WS-GGI','WS-GNI'};
cfg.settings.lambda_g = 128;
cfg.settings.K_max = 5;
cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
options = struct('experiment_name','quickstart', ...
    'output_file',fullfile(pwd,'results','local','quickstart_pl.mat'), ...
    'make_local_plots',false,'verbose',false);
p = comparison(cfg, options);
```

`comparison()` without arguments uses `example_CSTR`. To run a full configured
Monte Carlo comparison explicitly:

```matlab
addpath('mlab');
cfg = jsondecode(fileread('configs/example_CSTR.json'));
cfg.mode = 'MONTECARLO';
cfg.N_MC = 30;
p = comparison(cfg,struct('experiment_name','example_CSTR','make_local_plots',false));
```

`EXAMPLE` fits one measured record. `MONTECARLO` changes the input and noise
realizations, reusing each realization across all selected methods. It retains
the configured method initialization across repetitions. Use the separate
initialization study when the starts, rather than the records, should vary.

MATLAB `comparison(cfg, options)` supports:

| Option | Meaning |
| --- | --- |
| `experiment_name` | Output label/directory; it does not select a benchmark configuration. |
| `output_file` | Explicit MAT-file path. |
| `make_local_plots`, `verbose` | Toggle runner figures and progress output. |
| `method_configs` | Override the optimizer **methods struct**, e.g. `optim.methods` after reading the backend JSON. |
| `save_results` | Set false to return the payload without writing it. |
| `data_runs` | Supply records as cells of structs containing `r`, `nu`, and `c`, matched to the repetition count. |
| `initializations` | Parameter-by-trial bank for `INITIALIZATION` mode; normally managed by the study wrapper. |

### Python comparison: configured run or exact method subset

```text
python python/comparison_torch.py
```

The script's `main()` currently selects `example_CSTR`. Edit that source-level
selection to use `example_1`; edit the benchmark JSON for ordinary comparison
settings. `python_selected_methods`, if provided, takes precedence over
`selected_methods`. The ordinary Python runner and study runners also append
WS-GGHAM-2-I and WS-GGHAM-2-H. To run an **exact subset**, call `run_method_set`
directly, as in this complete short example from a Python session at the repo root:

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd() / "python"))
from common import load_json_config, set_seed
from comparison_torch import (
    build_dims_and_theta, load_experiment_config, run_method_set,
    save_comparison_results,
)
from wiener_system import generate_example_data

cfg = load_experiment_config("example_1")
cfg["mode"] = "EXAMPLE"
cfg["settings"].update(lambda_g=128, K_max=5)
cfg["initialization"] = {"policy": "consistent", "theta0": [0, 0, 1, 0, 0, 0]}
s = cfg["settings"]
device = torch.device("cpu")
dims, theta = build_dims_and_theta(cfg, dtype=torch.float64, device=device)
optim = load_json_config(Path("python") / cfg["python_optim_config"])["methods"]
methods = ["WS-GGI", "WS-GNI"]
cfg["selected_methods"] = methods
set_seed(cfg["seed"])
data = generate_example_data(
    dims, theta, s["lambda_g"], s["burn_in"], s["sigma_nu"],
    device=device, dtype=torch.float64,
)
results = run_method_set(
    methods, optim, data["r"], data["nu"], data["c"], theta, dims,
    s["lambda_g"], s["K_max"], s["conv_threshold"], False,
    verbose=False, experiment_cfg=cfg, run_metadata={"data_seed": cfg["seed"]},
)
save_comparison_results(
    Path("results/local/quickstart_fsm.mat"), results,
    experiment_name="quickstart", mode="EXAMPLE", selected_methods=methods,
    dims=dims, theta_true=theta, lambda_g=s["lambda_g"], K_max=s["K_max"],
    conv_threshold=s["conv_threshold"], sigma_nu=s["sigma_nu"],
    burn_in=s["burn_in"], enforce_fixed_iterations=False, seed=cfg["seed"],
    all_runs=[results], resolved_experiment=cfg,
    example_data={key: data[key].cpu().numpy().reshape(-1, 1)
                  for key in ("r", "nu", "c")},
)
```

## Initialization sensitivity

These runners hold one measured record fixed and draw a different stable
parameter vector for each repetition. Within a repetition, every optimizer gets
the same start, including consistently reconstructed latent signals.

```matlab
addpath('mlab');
initialization_study('example_CSTR');
% Short run, without editing JSON:
initialization_study('example_1',struct('study',struct('n_runs',5,'K_max',20), ...
    'output_file',fullfile(pwd,'results','local','initialization_pl.mat')));
```

```text
python python/initialization_study.py --experiment example_CSTR
python python/initialization_study.py --experiment example_1 --runs 5 --iterations 20 --cpu --output results/local/initialization_fsm.mat
python python/initialization_study.py --help
```

Configure `initialization_study` in [configs/studies.json](configs/studies.json):

| Setting | Meaning |
| --- | --- |
| `n_runs` | Number of random starts (default 30). |
| `initialization_seed` | Seed for parameter draws, independent of the measured record. |
| `data_seed` | Optional fixed-data seed; otherwise use the benchmark seed. |
| `center` | Parameter vector; `null` uses zero coefficients except `b1 = 1`. |
| `distribution` | `uniform` or `normal`. |
| `scale` | Scalar or parameter vector: uniform half-width or normal standard deviation. |
| `stability_radius` | Reject proposed starts with signal/noise poles outside this radius. |
| `max_draws` | Maximum proposals before reporting insufficient stable draws. |
| `K_max` | Optional iteration-budget override. |

Sampling conditions the requested distribution on stability; it does not clip
coefficients. The optimizer's later stability projection is a separate setting.
The fixed polynomial coefficient `f1 = 1` is not randomized. Plot **final parameter
estimates** across starts, with true-value reference lines, using the commands in
[Plotting and exports](#plotting-and-exports).

## Noise sensitivity

The `noise_study` block in [configs/studies.json](configs/studies.json) tests
innovation standard deviations **[0.05, 0.1, 0.2, 0.5, 1.0]**, with the nominal
0.1 second. Innovation power is the square of each value. Each optimizer runs
once per level by default (`n_runs: 1`), using a fixed common `theta0` (`null` uses
the nominal `b1 = 1` start), and the benchmark's record length and optimizer
settings. An optional `K_max` overrides the iteration budget.

```matlab
addpath('mlab');
noise_study('example_1');
% Short local experiment:
cfg = jsondecode(fileread('configs/example_1.json'));
cfg.settings.lambda_g = 256;
noise_study('example_1',struct('experiment_cfg',cfg, ...
    'study',struct('n_runs',1,'K_max',10), ...
    'output_file',fullfile(pwd,'results','local','noise_pl.mat')));
```

```text
python python/noise_study.py --experiment example_1
python python/noise_study.py --experiment example_1 --runs 1 --iterations 10 --cpu --output results/local/noise_fsm.mat
python python/noise_study.py --help
```

The input and standard-normal innovation draws (including burn-in) are matched
across noise levels using `data_seed + realization - 1`. Only innovation amplitude
changes; all methods share each resulting record and initialization. Pairing is
within a backend; MATLAB and NumPy streams differ.

Outputs are `results/<experiment>/noise_pl.mat` and `noise_fsm.mat`, containing
final estimates, statuses, signed bias, variance, parameter RMSE, Monte Carlo
standard errors, outcome counts, seeds, and resolved settings. By default,
`save_case_results: true` also saves full histories in `noise_<pl|fsm>_sigma01.mat`
through `_sigma05.mat`; the second case has `sigma_nu = 0.1`. These per-level files
work with `plot_comparison_results` through explicit `data_files`. The aggregate
is for analysis of the sweep; see [array dimensions](docs/identification_studies.md#noise-sensitivity-paired-records-fixed-initialization).

Override `sigma_nu_values`, `theta0`, `n_runs`, `data_seed`, `summary_population`,
and `save_case_results` in the study block or MATLAB `options.study`. Python also
accepts `--noise-levels 0.05 0.1 0.2 0.5 1.0`.

## Record length and finite-sample bias

Use this after choosing a start and iteration budget from the initialization
study. Set `record_length_study.theta0` in `configs/studies.json` to that fixed
vector; `null` uses the nominal `b1 = 1` start, not the true parameters.

```matlab
addpath('mlab');
record_length_study('example_CSTR');
% Short run; FSM-specific resolutions are ignored by the PL backend:
record_length_study('example_1',struct('study',struct( ...
    'n_runs',3,'K_max',5,'record_lengths',[256 512 1024]), ...
    'output_file',fullfile(pwd,'results','local','record_length_pl.mat')));
```

```text
python python/record_length_study.py --experiment example_CSTR
python python/record_length_study.py --experiment example_1 --runs 3 --iterations 5 --lengths 256 512 1024 --cpu --output results/local/record_length_fsm.mat
python python/record_length_study.py --help
```

| Setting in `record_length_study` | Meaning |
| --- | --- |
| `n_runs`, `data_seed` | Independent data repetitions and their seed. |
| `record_lengths` | Increasing retained record lengths. |
| `theta0`, `K_max` | Common fixed start and optional iteration budget. |
| `summary_population` | `finite` (default, includes finite stalled/iteration-limited outputs) or `converged`. |
| `fsm_resolutions` | Python cases with `label`, `fsm_tau_g`, `fsm_tau_h`, and `fsm_n_freq`. |
| `save_case_results` | Save full per-case comparison histories in addition to the aggregate. |

Each independent repetition generates the longest record once, then uses nested
prefixes shared by all methods, lengths, and FSM resolutions. Truth is generated
by recursive IIR filtering independently of the FSM grid. Python requires every
length to accommodate both configured FIR lengths; the defaults compare
128/1024 and 256/2048 coefficient/frequency resolutions. MATLAB uses one
`recursive_PL` case.

Outputs include signed mean parameter error, sample variance, RMSE, Monte Carlo
standard error, pointwise normal-approximation intervals, final estimates, and
outcome counts. Excluding unsuccessful trials makes these conditional summaries.
The descriptive `intercept + slope/N` fit is not a proof of asymptotic
inconsistency; convergence, local minima, boundaries, and FSM approximation can
also produce persistent offsets. See the [study guide](docs/identification_studies.md)
for interpretation and array dimensions.

### Shared study interfaces

All Python study CLIs accept `--experiment`, `--study-config`, `--runs`,
`--iterations`, `--output`, `--cpu`, and `--verbose`; only record length accepts
`--lengths`, and only noise accepts `--noise-levels`. A custom study JSON must
contain the corresponding `<kind>_study` block.
CLI overrides take precedence over the study file.

Study progress is **on by default**, without `--verbose`: method start/finish
messages and periodic iteration updates include the current repetition/case,
completed-fit count, elapsed time, and approximate ETA. Configure `progress`
(default `true`) and `progress_interval_seconds` (default `10`) in any study
block. MATLAB can override these via `options.study`; Python also accepts
`--no-progress`. ETA warms up after the first fit of every method and estimates
remaining optimization time from method-specific runtimes and record lengths.
Updates appear at iteration boundaries; preparation and saving are reported
separately. See [progress details](docs/identification_studies.md#console-progress-and-estimated-time-remaining).

MATLAB wrappers accept `options.study` (study overrides), `study_config` (JSON
path), `experiment_cfg` (full benchmark struct), `method_configs` (methods struct),
`output_file`, and `verbose`. Python's
[identification_studies.py](python/identification_studies.py) exposes
`run_initialization_study(cfg, study, ...)`, `run_noise_study(cfg, study, ...)`, and
`run_record_length_study(cfg, study, ...)` for programmatic runs. All accept
`experiment_name`, `output_file`, `device`, and
`verbose` keyword arguments. Study wrappers manage initialization themselves and
disable padded histories.

## Configuration reference

Shared benchmark files are [example_1.json](configs/example_1.json) and
[example_CSTR.json](configs/example_CSTR.json). Optimizer controls live separately
in [MATLAB optim_configs.json](mlab/optim_configs.json) and
[Python optim_configs.json](python/optim_configs.json).

### Benchmark settings

| Field | Purpose |
| --- | --- |
| `mode` | `EXAMPLE` or `MONTECARLO` for ordinary comparisons. Study wrappers manage their own modes. |
| `seed` | Input/noise random seed; ordinary Monte Carlo uses `seed + repetition - 1`. |
| `N_MC` | Number of ordinary Monte Carlo records. |
| `selected_methods` | Ordered method names; see Python's extra-method behavior above. |
| `python_selected_methods` | Optional Python-specific selection. |
| `make_local_plots` | Ordinary comparison runner's immediate figures. |
| `matlab_optim_config`, `python_optim_config` | Backend optimizer JSON paths, resolved relative to `mlab/` or `python/`. |
| `system.na`, `nb`, `nf`, `nd` | Linear denominator/numerator orders, polynomial degree, and noise AR order. |
| `system.theta_true` | True benchmark vector in the parameter order below. |
| `settings.lambda_g` | Retained sample count, **not** a regularization parameter. |
| `settings.burn_in` | Generator samples discarded before fitting. |
| `settings.sigma_nu` | Innovation **standard deviation**, not variance. |
| `settings.K_max` | Comparison history capacity including state zero: at most `K_max - 1` updates. |
| `settings.conv_threshold` | Relative-step threshold used with the stopping residual tests. |
| `settings.enforce_fixed_iterations` | Pad terminal histories; does not force extra optimization updates. |

Parameter vectors use `[a1...ana, b1...bnb, f2...fnf, d1...dnd]`, of length
`na + nb + nf - 1 + nd`. The fixed coefficient `f1 = 1` is omitted. The
six-entry vectors in this README apply to the supplied benchmark dimensions;
adapt their lengths when changing the model orders.

### Initialization, stopping, and diagnostic overrides

Add these blocks at benchmark level to apply them across compared methods:

```json
{
  "initialization": {"policy": "consistent", "theta0": [0, 0, 1, 0, 0, 0]},
  "stopping": {"gradient_tolerance": 1e-6, "loss_tolerance": 1e-8},
  "diagnostics": {"enabled": true, "save_matrices": false}
}
```

These are blocks to merge into a benchmark, not a complete experiment JSON.
Without the override, the default initialization policy is `legacy`, retaining
the method-specific latent-state setup. `consistent` assigns the whole stable
finite `theta0` and derives the latent signals from it. An experiment-level
initialization **replaces** the method initialization. `stopping` and `diagnostics`
merge in order: defaults, method, experiment. Arbitrary experiment-level fields
such as `eta` do not override optimizer controls; change the method configuration.

Optional `diagnostics.parameter_scales` supplies positive scales for reporting
the conditioning of `S H S` alongside the raw Hessian. It does not rescale the
optimizer. Matrix snapshots increase result size. Gradient-only WS-GGI does not
perform extra Hessian differentiation, so absent Hessian diagnostics are expected.
Definitions and stopping-status semantics are in the
[diagnostics guide](docs/comparison_diagnostics.md).

For legacy initialization, these method-level fields retain the original setup:

| Field under `initialization` | Effect |
| --- | --- |
| `theta_init` | Scalar broadcast or full parameter vector; default `1e-6`. |
| `mode: physical` | Initialize the linear latent signal from `linear_init_a` and `linear_init_b`, independently of `theta_init`. |
| `linear_init_a`, `linear_init_b` | Initial linear-filter coefficients; defaults are zero denominator coefficients and `b1 = 1`. |
| `e_init_mode` | With physical mode: `output_residual` (`c - alpha`) or `zeros`. |
| `alpha_init`, `e_init` | MATLAB only, outside physical mode: scalar or sample-length latent-state arrays. |

Without physical initialization, latent signals otherwise start at `1e-6`.
Use `consistent` when the parameter vector should determine the latent state.

## Method availability and controls

Display names contain hyphens; JSON keys under `methods` use underscores, e.g.
`WS-GNI` corresponds to `methods.WS_GNI`. A requested unsupported method is saved
as `skipped`, rather than silently substituted.

| Method | MATLAB / PL | Python / FSM | Main active controls |
| --- | --- | --- | --- |
| RGLS | Implemented | Placeholder; skipped | PL `regularization.enabled`, `regularization.lambda`, `stability_projection`. |
| WS-GGI | Implemented | Implemented | PL `step_scale`, `d_step_multiplier`; FSM `step_size` (fallback `step_scale`), `d_step_multiplier`. |
| WS-GNI | Implemented | Implemented | PL `step_size`; FSM `eta` (fallback `step_size`); both `hessian_regularization`, `d_step_multiplier`. |
| WS-GGHAM-1-dH | Implemented | Implemented | `eta`, `hessian_regularization`; FSM also `d_step_multiplier`. |
| WS-GGHAM-2-dH | Implemented | Implemented | `eta`, `hessian_regularization`; FSM also `d_step_multiplier`. |
| WS-GGHAM-2-H | Skipped | Implemented | FSM `eta`, `hessian_regularization`, `d_step_multiplier`. |
| WS-GGHAM-2-I | Implemented | Implemented | `eta`; FSM also `d_step_multiplier`; no Hessian inverse to regularize. |
| WS-LGHAM-1-TIK | Implemented | Implemented | `eta`, `inverse_solver`, `tikhonov_lambda`, `eps_0` (fallback `epsilon0`); FSM also `d_step_multiplier`. |
| WS-LGHAM-3-TIK | Implemented | Implemented | Same loss-root controls. |

PL WS-GGI normalizes its step using the regressor norm; FSM WS-GGI uses an
unnormalized nonlinear gradient. Loss and regularization scales also differ
between backends. Do not transfer tuned step sizes or damping values assuming
equivalence. PL's `d_step_multiplier` applies to GGI/GNI, not every GHAM rule.

Loss-root `inverse_solver` accepts `pinv` or `tikhonov`. The gradient-row
denominator is `g'g + lambda` in Python and `g'g + lambda^2` through MATLAB's
`lsqminnorm` regularization factor. Without an explicit target, the target comes
from the best preceding non-loss-root method; with no such result it falls back
to zero. Method order therefore matters. MATLAB additionally dispatches
WS-LGHAM-2 / WS-LGHAM-2-TIK with an appropriate method config and supports
`order` 1–3. Python's implemented loss-root names select orders 1 and 3.

Python applies configured `stability_projection` to implemented methods, with
`stability_radius` controlling the pole radius. PL currently applies projection
in RGLS. This differs from rejecting unstable random initializations.

### FSM approximation and reconstruction

| Python method field | Effect |
| --- | --- |
| `fsm_tau_g` / `tau_g` | Signal-model FIR response length. |
| `fsm_tau_h` / `tau_h` | Noise-model response length for `1/D`, used in reconstruction; also affects the shared frequency grid. |
| `fsm_n_freq` / `n_freq` | Frequency-grid size. |
| `innovation_estimator` | `fsm_inverse` or `state_residual` for auxiliary innovation reconstruction. |

Ordinary runs default to response lengths up to 128, bounded by the record
length, and an automatically sized power-of-two grid when not specified. Study
resolution cases explicitly override these fields. The optimized innovation
criterion applies the candidate AR polynomial to the output residual;
`innovation_estimator` does not switch that objective.

Method names dispatch the implementations. Descriptive fields such as
`linear_operator`, `hessian_mode`, and `derivative_order` are not general switches
for constructing arbitrary update rules. `d_regularization_weight` currently
does not change the FSM objective; changing it is not an implemented ablation.

## Ablations and parameter sweeps

There is no dedicated ablation CLI. Use in-memory configuration copies and save
one named result per case. Keep data seeds, starts, selected-method order, stopping
rules, and budgets fixed when isolating one optimizer control. Full resolved
settings and data hashes are saved by the comparison runners.

| Study | What to change | Recommended entry point |
| --- | --- | --- |
| Initialization distribution / spread | `center`, `distribution`, `scale`, sampling radius | Initialization-study wrapper. |
| Noise intensity | `noise_study.sigma_nu_values` | Noise-study wrapper, with paired input/innovation draws and a fixed start. |
| Record length | `record_lengths` | Record-length wrapper, which pairs nested data prefixes. |
| FSM accuracy | `fsm_resolutions` or per-method FIR/grid fields | Record-length study for bias; comparison API for optimization behavior. |
| Step size / damping | Method-specific active fields in the table above | MATLAB `options.method_configs`; Python `run_method_set` with copied configs. |
| Inverse / homotopy order | Supported method names, `inverse_solver`, target controls | Comparison API; preserve preceding baselines when automatic targets are desired. |
| Stability projection | Active backend/method projection settings | Comparison API; include failure/outcome counts. |
| Diagnostic overhead | `diagnostics.enabled`, `save_matrices` | Matched comparison runs; compare `total_time`, `diagnostic_time`, and `total_wall_time`. |
| Stopping sensitivity | Residual tolerances, relative-step threshold, iteration budget | Comparison or study wrapper with the same data/starts. |

Example: a small PL Hessian-damping ablation using the same three data records in
every case, without modifying the tuned JSON files:

```matlab
addpath('mlab');
cfg = jsondecode(fileread('configs/example_1.json'));
cfg.mode = 'MONTECARLO';
cfg.N_MC = 3;
cfg.selected_methods = {'WS-GNI'};
cfg.settings.lambda_g = 128;
cfg.settings.K_max = 5;
cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
base = jsondecode(fileread('mlab/optim_configs.json'));
values = [0 1e-4 1e-2];
for k = 1:numel(values)
    optim = base.methods;
    optim.WS_GNI.hessian_regularization = values(k);
    comparison(cfg,struct('experiment_name','gni_ablation','method_configs',optim, ...
        'make_local_plots',false,'verbose',false, ...
        'output_file',fullfile(pwd,'results','local',sprintf('gni_reg_%02d.mat',k))));
end
```

For Python, start with the complete `run_method_set` example above, use
`copy.deepcopy(optim)` for each case, change e.g.
`case_optim["WS_GNI"]["hessian_regularization"]`, and pass `case_optim` to the
runner. Reuse `data`, `cfg`, and `theta` for optimizer-only ablations, and save
each case to a separate filename. For a noise sweep, regenerate data after
resetting the same seed for each `sigma_nu`; report both performance and outcomes.
Equal seeds across MATLAB and Python do not imply equal records or parameter draws.

## Plotting and exports

### Ordinary comparisons and conditioning

The MATLAB plotter reads both PL and FSM MAT files. Explicit paths are the most
reliable way to select a benchmark or custom run:

```matlab
addpath('mlab');
files = {fullfile(pwd,'results','local','quickstart_pl.mat'), ...
         fullfile(pwd,'results','local','quickstart_fsm.mat')};
plot_comparison_results(struct('data_files',{files}, ...
    'fig_dir',fullfile(pwd,'figs','local'),'export_figures',true));
```

With no options, `plot_comparison_results` uses the `experiment_name` and `mode`
defaults near the top of the function and exports to `figs/`. It also discovers
available `initialization_fsm.mat` / `initialization_pl.mat` files for that
experiment and plots their final-parameter estimate boxplots and outcome counts.
Use `options.experiment_name` to switch the experiment, `mode: INITIALIZATION`
(or `study: initialization`) for study-only plots, or explicit `data_files` for
an exact file selection. `include_initialization: false` disables automatic
initialization discovery alongside ordinary comparisons.

| Option | Purpose |
| --- | --- |
| `data_files` | Cell array of one or more MAT-file paths; overrides study auto-discovery. |
| `fig_dir` | Export directory. |
| `export_figures` | Export PDFs/CSV summaries (default true in the main plotter). |
| `diagnostics_only` | Only optimizer-conditioning figures and summaries. |
| `experiment_name`, `mode` | Select standard experiment files; modes are `EXAMPLE`, `MONTECARLO`, or `INITIALIZATION`. |
| `include_initialization` | Include available initialization-study files alongside ordinary comparisons (default true). |
| `study` | Select only standard `initialization` or `record_length` study files. |

Ordinary figures include parameter trajectories, estimation/reconstruction
metrics, available output traces, Monte Carlo summaries, and final parameter-error
boxplots. One diagnostic figure per parametrization overlays the selected
methods in six panels: objective, raw-Hessian condition number, solver-matrix
condition number, signed curvature, gradient norm, and applied-step norm.
Monte Carlo diagnostic curves use finite medians and interquartile bands.
Singular conditions remain in the saved diagnostics/CSV even when omitted from
logarithmic curves. Old MAT files without diagnostics remain readable.

The lower-level helpers are also callable:

```matlab
payload = load(fullfile('results','local','quickstart_pl.mat'));
[figures, report] = plot_optimization_diagnostics( ...
    payload,fullfile(pwd,'figs','local'),'quickstart',true);
% report is the per-run diagnostic summary table.
```

`summarize_optimization_diagnostics` is the summary helper;
`matrix_diagnostics` computes matrix metrics. They are not experiment runners.
Python comparison figures use `common.plot_method_metrics` and
`common.plot_parameter_trajectories`; `make_local_plots` controls the ordinary
comparison runner's local plotting. Use MATLAB for the combined saved-result
figures and exports.

### Initialization and record-length plots

After generating default-named study files for either or both backends:

```matlab
addpath('mlab');
plot_comparison_results(struct('study','initialization','experiment_name','example_CSTR'));
plot_comparison_results(struct('study','record_length','experiment_name','example_CSTR'));
```

For custom output paths, pass `data_files` as above; study type is detected from
the saved payload. Initialization boxplots show final **estimates**, retain
outliers, and add true-value lines; separate outcome-count plots expose failures,
skips, and nonconverged finite runs. Record-length figures show signed bias with
Monte Carlo intervals and parameter RMSE with boxed axes. Each resolution also
gets an outcome table with counts and percentages by record length and optimizer:
finite, converged, stalled, iteration-limited, failed/invalid, skipped, and used
in the bias/RMSE summary. Failed and skipped runs are reported separately.
The direct record-length helper accepts a path or loaded payload:

```matlab
[figures,outcomes] = plot_record_length_study(fullfile('results','local','record_length_fsm.mat'), ...
    struct('fig_dir',fullfile(pwd,'figs','local'),'export_figures',true));
```

The direct diagnostic and record-length helpers default to no export; the main
plotter defaults to export. `outcomes` is a numeric MATLAB table. Exported
summaries include diagnostic/bias CSVs and a separate
`record_length_outcomes_<experiment>_<pl|fsm>.csv` with counts and percentages.
The existing `record_length_outcomes_*.pdf` exports now contain tables.
The plotter visualizes results; it does not perform new hypothesis tests.

### Source diagrams

[figs/DRAWIO/build_gotham_flowchart.py](figs/DRAWIO/build_gotham_flowchart.py)
regenerates the GOTHAM flowchart's editable `.drawio` source:

```text
python figs/DRAWIO/build_gotham_flowchart.py
```

This is a diagram builder, not an experiment or PDF-export runner. Other diagram
sources and existing exports are under `figs/`.

## Results and reproducibility

Default output locations:

| Run | MATLAB | Python |
| --- | --- | --- |
| Ordinary comparison | `results/<experiment>/comparison_pl_<mode>.mat` | `results/<experiment>/comparison_fsm_<mode>.mat` |
| Initialization study | `results/<experiment>/initialization_pl.mat` | `results/<experiment>/initialization_fsm.mat` |
| Record-length study | `results/<experiment>/record_length_pl.mat` | `results/<experiment>/record_length_fsm.mat` |

Ordinary mode suffixes are lowercase (`example`, `montecarlo`). Existing files at
the chosen path are overwritten: use distinct labels/output paths for sweeps.
Schema-2 comparisons retain `results` for the first realization, `all_runs` for
every realization, the resolved experiment/configuration, starts, histories,
status/counts, run/data hashes, and optional derivative diagnostics. Study files
also contain their study design and aggregate statistics. Keep the repository
revision with results; saved settings alone do not identify the implementation.

FSM estimation loss is mean squared innovation; PL logs half the squared-residual
sum for its refreshed regressor, with additional fixed-regressor update losses.
Their raw objective values are not directly comparable. The existing
`true_innovation_reconstruction` RMSE uses the synthetic true innovation as an
evaluation reference; it is not a held-out prediction score. Outcomes such as
`stalled`, `max_iterations`, `failed`, and `skipped` must accompany error summaries.
See the [schema and timing definitions](docs/comparison_diagnostics.md) for details.

## Validation and repository hygiene

Short checks, independent of manuscript-scale runs:

```text
python -B -m unittest discover -s python -p "test_*.py" -v
```

```matlab
addpath('mlab');
test_comparison_diagnostics;
test_identification_studies;
test_noise_study;
% Optional cross-backend import/plot checks on freshly generated Python files:
test_comparison_diagnostics('path/to/fsm_result.mat');
test_identification_studies('path/to/initialization_fsm.mat', ...
                           'path/to/record_length_fsm.mat');
```

Tests cover diagnostic matrices, stopping, initialization, reproducibility,
record pairing, study statistics, MAT schemas, and plotting on small data.
The optional paths are placeholders: replace them with actual generated files.

Use `results/local/` and `figs/local/` for disposable runs; they are ignored along
with Python bytecode, virtual environments, MATLAB autosaves, `.validation/`, and
`tmp/`. Existing scientific results and figure sources remain versioned. Add
intentional publication artifacts explicitly outside the local directories, and
keep benchmark configurations separate from one-off in-memory overrides. Do not
commit environment-specific MATLAB preferences or generated Python caches.
