# BI_HMC: Bayesian Wiener-AR identification

`BI_HMC` infers the original rational-block coefficients `a,b`, polynomial
coefficients `f2,...,f_nf` with `f1=1`, output-AR coefficients `d`, and a positive
innovation variance. The model orders are arbitrary and fixed for a fit,
with `na,nd>=0`, `nb>=1`, and `nf>=1` for a nontrivial signal path.
This is ordinary Bayesian inference with fixed Gaussian coefficient priors,
not sparse Bayesian model selection.

## Point estimate and comparison contract

The reported coefficient vector is the **posterior mean in physical coordinates**:

\[
\widehat\theta=\frac{1}{CS}\sum_{c=1}^{C}\sum_{s=1}^{S}\theta_{cs}.
\]

The mean minimizes posterior expected squared coefficient error over unrestricted
coefficient decisions. It is computed after transforming each retained draw,
not by transforming the mean of the sampling coordinates. It is not the most
densely sampled draw or a coordinate-dependent MAP approximation. This choice
matches the existing signed-error, RMSE, final-error boxplot, and paired-test
comparisons. See [Stan's decision-analysis discussion](https://mc-stan.org/docs/stan-users-guide/decision-analysis.html)
for the expected-loss interpretation of a Bayesian point decision.

The innovation variance is saved separately; it is not appended to the
coefficient vector used by existing parameter-error comparisons, since those
methods do not estimate an equivalent variance parameter.

HMC transitions are **not optimizer iterations**. A Bayesian result has
`method_kind="bayesian"`, `point_estimate="posterior_mean"`, and an empty
`theta_hist` with shape `[parameter,0]`. Its `param_err`, `rmse_hist`,
`est_loss_hist`, `iter_time`, and `cum_time` are empty, and `iterations`,
`update_count`, and `accepted_updates` are zero. Sampling counts, posteriors,
and diagnostics have their own fields. No HMC trace is padded into an
optimization-convergence curve.

For denominator orders above two, the stable coefficient domain need not be
convex. A coefficient-wise posterior mean can therefore be unstable despite
stable draws. The implementation checks the final mean, never projects it,
and marks an unusable mean as invalid for output/scoring populations while
retaining the coefficients, posterior, and explicit warning for inspection.

## Inference model and numerical method

`python/bayesian_hmc.py` owns the complete sampler. Its inference API accepts
observations, dimensions, priors, starts, and observed input history. It does
not accept true parameters, true innovations, or the benchmark's true noise
standard deviation.

1. Replay the known input prehistory and retained input through the candidate
   causal IIR block, starting from the specified known zero state before that
   input history. Apply the polynomial to obtain the candidate noise-free mean.
2. Form residuals `e=c-beta`, whiten them with the candidate output-AR
   polynomial, and include the selected initial-noise contribution in `Q`.
3. Evaluate the normalized Gaussian innovation likelihood, independent fixed
   Gaussian coefficient priors on the stable domain, and inverse-gamma
   innovation-variance prior.
4. Transform real sampling coordinates through `tanh` reflection coefficients
   and the Schur recursion for both denominators. Transform log variance
   through `exp`. Add the complete transformation log determinant.
5. For each chain, optionally optimize the **physical** log posterior for a
   finite initialization budget. Run warm-up with dual-averaging step-size
   adaptation and optional diagonal geometry adaptation. Retained sampling
   freezes both settings. A randomized bounded leapfrog count avoids using one
   fixed trajectory length at every transition.
6. Keep repeated states when proposals are rejected. Pool retained physical
   draws only for the posterior summary; retain chain identity for diagnostics.

This is fixed-budget HMC, **not NUTS**. The initialization optimization is a
numerical starting aid, not the reported estimator. Its full cost is included.

Filtering uses SciPy's exact causal `lfilter` under a custom PyTorch autograd
function. Its finite-record adjoint differentiates inputs, numerator, and
denominator without building a Python recursion graph for every observation.
Only first derivatives are supported and needed by HMC. The existing FSM
optimizer loss, derivative engines, and update rules are unchanged.

The sampler runs on CPU in float64, including when a comparison's data was
generated on CUDA. `cpu_threads` temporarily limits PyTorch threads and restores
the caller's setting, including on failure. Device/dtype and version metadata
are saved. Different devices and other concurrent work can affect elapsed time.

## Boundaries and forward-model conventions

The shared runners pass observed input prehistory and the number of discarded
noise-start samples. These are experimental inputs and boundary information,
not hidden internal states or true innovations.

With `noise_burn_in` supplied, the initial retained AR residual block uses the
exact covariance after that finite zero-start burn-in. It is computed from the
candidate AR impulse response, including its dependence on the candidate `d`.
Without that context, the explicitly selected `noise_initialization` policy is
used: `stationary`, `zero`, `conditional`, or configured `finite_burn_in`.
Stationary covariance uses Yule-Walker equations; conditional likelihood omits
the initial block and uses the corresponding smaller sample count.

Without observed input prehistory, the signal model explicitly starts at zero
at the retained record's beginning. Unknown signal initial states are not
sampled in this implementation. Supply the known experimental history, or
extend the model with initial-state nuisance variables for data requiring it.

The existing standalone/example generator uses a frequency-sampling filter
approximation. The study generator uses an exact recursive IIR. BI_HMC fits the
same original coefficient model through an exact IIR likelihood; it does not
inherit an FSM grid or truncated impulse response. This target/boundary
distinction should be reported with comparisons, especially for short records.

## Output assessment versus inference

After inference is finished, the shared wrapper alone receives the benchmark
truth and innovations to compute ordinary assessment scores:

- `final_err`: the existing percentage coefficient-norm error.
- `final_output_rmse`: the existing `simulate_wiener(r,nu,theta_hat,dims)`
  reconstruction RMSE, using the common legacy assessment convention.
- `output_estimate` and `output_error`: the corresponding reconstructed output
  and measured-minus-reconstructed difference.
- `noise_free_output_estimate`: the actual fitted exact-IIR mean response,
  including known observed input history.

The shared reconstruction resets filtering at the retained record and uses the
existing FSM simulation, just as other comparison methods do. It is an
oracle-assisted **assessment** when true innovations are available, not a
predictive model for an unknown future record. It does not feed back into HMC.
Keep it distinct from the fitted noise-free mean response and its residuals.

## Configuration

The `BI_HMC` entry in `python/optim_configs.json` provides normal-run settings.
It is opt-in and is not silently added to all existing method lists.

| Setting | Meaning |
| --- | --- |
| `chains`, `warmup`, `samples` | Number of chains and transitions per chain; `samples` excludes warm-up. |
| `leapfrog_steps` | Upper trajectory-step count; each transition randomizes its leapfrog count within the configured range. |
| `step_size`, `target_accept` | Initial integration step and warm-up acceptance target. |
| `adapt_step_size`, `adapt_mass`, `mass_window` | Warm-up adaptation controls. Retained sampling does not adapt. |
| `metric_regularization_weight`, `metric_regularization_variance`, `metric_min_variance`, `metric_max_variance` | Numerical regularization/bounds for the adapted position-variance metric in sampling coordinates. These do not change the posterior prior or likelihood. |
| `initial_jitter`, `initial_optimization_steps` | Dispersed chain starts and optional physical-posterior optimization budget. |
| `initialization.theta0` | Shared original-coefficient start when explicitly configured. Otherwise a stable nominal start is constructed from dimensions. |
| `initial_variance` | Starting innovation variance, independent of the benchmark truth. |
| `coefficient_prior_mean`, `coefficient_prior_std` | Scalar or parameter-length physical prior settings. |
| `prior_a_std`, `prior_b_std`, `prior_f_std`, `prior_d_std` | Optional scalar or block-length prior scale overrides. |
| `variance_prior_shape`, `variance_prior_scale` | Inverse-gamma shape and scale: density proportional to `s^(-shape-1)*exp(-scale/s)`. |
| `noise_initialization`, `noise_burn_in` | Boundary policy when a runner has not supplied explicit burn-in context. |
| `seed`, `progress_every` | Base sampler seed and heartbeat interval in transitions. |
| `max_rhat`, `min_ess`, `divergence_threshold` | Explicit diagnostics/energy-error thresholds. |
| `cpu_threads` | Temporary PyTorch CPU thread count; zero preserves the caller setting. |

The default prior scales are fixed choices in the coefficient units, not
learned sparse precisions. The variance prior is fixed across noise-intensity
conditions. Reconsider its scale for different units or independently measured
noise ranges; do not set it from hidden simulation truth.

The innovation-variance prior now uses inverse-gamma shape `3` and scale
`0.002`. Its mean is `scale/(shape-1) = 0.001`, ten times smaller than the
benchmark innovation variance `0.1^2 = 0.01`. Its standard deviation is also
`0.001`. This replaces the earlier scale `0.02`, whose prior mean happened to
equal the benchmark truth. The prior remains fixed across datasets and noise
levels; experiment settings do not determine its parameters.

`K_max`, optimization stopping tolerances, Hessian regularization, and fixed
optimizer-history padding do not govern BI_HMC sampling.

## Running the example and studies

From the repository root, in a Python environment containing the existing
requirements:

```text
python python/example_torch.py --experiment example_CSTR --method BI_HMC --cpu --no-show
```

This uses the configured record length and sampler budget, saves a MATLAB file
and a compact JSON summary under `results/local/`, and writes posterior and
final-output figures under `figs/local/bi_hmc/`. Use `--output` and
`--figure-dir` for explicit paths. Add `--sampler-seed` to change inference
randomness independently of `--data-seed`.

A wiring smoke test, intentionally insufficient for posterior inference:

```text
python python/example_torch.py --method BI_HMC --cpu --length 64 --chains 2 --warmup 12 --samples 10 --leapfrog-steps 2 --no-show
```

`--hmc-config path/to/settings.json` accepts a JSON object of method overrides.
Direct `--chains`, `--warmup`, `--samples`, `--leapfrog-steps`, and
`--sampler-seed` options take precedence over that object.

For comparisons, include `BI_HMC` in `python_selected_methods`. Set
`include_python_extra_methods=false` for an exact method list. It can be mixed
with optimizers in the same `run_method_set` call; only its final-score fields
participate in common accuracy summaries. The three study entry points also
accept `--methods BI_HMC ...`; see the README and identification-study notes
for initialization, noise, and record-length/bias settings.

## Stored posteriors, validity, and time

Per-result fields include:

| Field | Shape / interpretation |
| --- | --- |
| `posterior_samples` | `[chain,retained_draw,physical_parameter]` |
| `posterior_variance_samples` | `[chain,retained_draw]` |
| `posterior_log_density` | Physical log posterior at each retained draw, up to a constant. |
| `theta_hat`, `posterior_std`, `posterior_interval` | Physical mean, SD, and `[2,parameter]` equal-tailed 95% intervals. |
| `sampling_transitions`, `posterior_draw_count` | Separate total transition and retained-draw counts. |
| `sampler_diagnostics` | Rank-normalized/folded split Rhat; bulk, tail and mean ESS; MCSE; acceptance; divergences; per-chain adapted settings; evaluation counts. |
| `sampler_seed`, `run_metadata` | Actual seed, dataset/start/config identifiers, boundary policies, and runtime environment. |
| `total_wall_time` | Complete wrapper time including initialization, warm-up, sampling, diagnostics, transfers and terminal scoring. |
| `inference_wall_time` | Core inference time including its setup, initialization, adaptation, draws and posterior diagnostics. |

`sampling_complete` means that configured sampler checks passed.
`sampling_warning` means finite estimates exist but at least one check failed;
it is not an optimizer-convergence claim. `point_estimate_valid` is independent
of the diagnostic-pass flag. Invalid or failed estimates are excluded from
finite-score populations. Finite warning estimates can enter the same finite
population that includes stalled or budget-limited optimizers, with warnings
reported separately. The passed/converged population accepts
`sampling_complete`, not `sampling_warning`.

Monte Carlo summaries expose `final_param_estimates`, `final_param_errors`,
`final_errors`, `final_output_rmse_runs`, `total_wall_time_runs`, and validity
flags. BI_HMC's optimizer-trajectory summaries remain NaN. Mixed-method MAT
exports use the union of result fields regardless of method ordering.

Prefer `total_wall_time` / `total_wall_time_runs` for comparisons across all
methods. Legacy optimizer `total_time` excludes diagnostic overhead, so it is
retained for compatibility but is not the cross-framework timing measure.
For BI_HMC, `total_time` equals the complete `total_wall_time`. File/figure
writing and plotting are outside the recorded identification time, consistently
with the other methods. No per-HMC-transition time is presented as an optimizer
iteration cost.

## Code map and validation

- `python/bayesian_hmc.py`: exact filter adjoint, physical density,
  transformations, boundary covariance, HMC, adaptation and posterior diagnostics.
- `python/comparison_torch.py`: independent Bayesian dispatch, terminal
  assessment, seed/context routing, mixed exports and common summaries.
- `python/example_torch.py`: CLI selection and sampler controls, posterior
  tables, MAT/JSON saving and final-output demonstration.
- `python/bayesian_plots.py`: posterior marginals, final output comparisons and
  terminal accuracy versus total runtime. `python/common.py` keeps optimizer
  trajectory figures separate.
- `python/identification_studies.py`, `python/study_progress.py`: repeated
  study execution, contexts, final scores, timing, diagnostics and sampling
  heartbeats. The small study entry scripts retain their existing dispatch.
- MATLAB comparison/study plotting and statistical helpers: accept Bayesian
  final scores/statuses, show posterior marginals, prefer total wall time and
  omit Bayesian optimizer/Hessian curves.
- `README.md`, `docs/identification_studies.md`, and this note document usage,
  interpretation, schemas and implementation choices.

Validation commands:

```text
python -B -m unittest discover -s python -p "test_bayesian*.py"
python -B -m unittest discover -s python -p "test_*.py"
```

The focused tests cover causal-filter gradcheck; complete-posterior gradients
under each boundary policy; dense covariance versus innovation likelihoods;
general-order stability/inversion/Jacobians; an independently integrated linear
posterior; seed and thread-state isolation; frozen retained adaptation;
non-oracle integration; mixed exports; and all three study pipelines.
MATLAB's `test_bayesian_comparison_plots` and `test_bayesian_study_reports`
validate posterior plotting, final-score handling and timing reports. Tiny-budget smoke tests
validate wiring, not MCMC convergence, finite-sample unbiasedness or interval
coverage. Longer runs must still be diagnosed using their saved results.

### Validated example, 30 October 2026

This recorded run predates the prior adjustment above: it used inverse-gamma
shape `3`, scale `0.02` (prior mean `0.01`). The saved MAT/JSON configuration
preserves those original settings. Its numerical results are not a rerun with
the current, smaller prior mean.

The following command ran the configured 10,000-observation `example_CSTR`
benchmark with four chains, 500 warm-up transitions and 1,000 retained draws
per chain, on CPU in the existing `DAFx26` environment (Python 3.10.19,
PyTorch 2.5.1, NumPy 2.2.6, SciPy 1.15.2):

```text
python python/example_torch.py --method BI_HMC --experiment example_CSTR --cpu --no-show --output results/local/bi_hmc/example_CSTR.mat --figure-dir figs/local/bi_hmc/example_CSTR
```

| Result | Value |
| --- | --- |
| Status | `sampling_complete` |
| Retained draws | 4,000 |
| Final coefficient error | 0.306424% |
| Shared reconstruction RMSE | 0.0157418 |
| Total wall-clock time | 320.242 seconds |
| Largest rank-normalized/folded split Rhat | 1.005235 |
| Smallest bulk ESS | 4,000 (the implementation conservatively caps ESS at the raw draw count) |
| Smallest tail ESS | 2,120 |
| Post-warm-up divergences | 0 |
| Mean innovation variance | 0.00986354 |

The first validation run exposed insufficient mixing in `d` and log variance
despite accurate point estimates. Reducing the numerical ridge in the warm-up
metric preserved the different posterior scales and resolved the configured
checks on the rerun. This is adaptation of computational geometry, not a
change to the prior or likelihood. Metric regularization controls are saved
in the resolved configuration.

The MAT file contains the full retained posterior and diagnostics; the adjacent
`.summary.json` contains numerical estimates and the run configuration.
`posterior.png/.pdf` and `output_comparison.png/.pdf` contain the final figures.
The reconstruction figure uses the shared assessment convention described
above, including the retained-record reset transient. This is one successful
validation record, not a repeated-record bias/coverage or timing benchmark.

The complete Python suite passed **38 tests**, including all existing Python
tests and the new Bayesian tests. MATLAB checks exercise mixed/Bayesian-only
plots, hypothesis-test cohorts, noise plots, record-length outcome/runtime CSVs,
and round-trip loading of a real Python posterior file.
