# Boundary handling in Wiener-AR identification

**Controlled numerical checks and implementation decision**

9 November 2026

## 1. Main findings and scope

The earlier BI-HMC and optimization comparisons used different assumptions at
the beginning of the retained data record. BI-HMC replayed observed input
prehistory and included an initial AR-noise covariance contribution. The
optimization methods restarted their fitting filters at zero after the
data-generation burn-in. This difference explains a substantial part of the
observed parameter-error advantage in the two tested benchmarks.

Two directions of intervention support that interpretation. Giving the FSM
criterion input-history replay and a conditional AR likelihood brought its
well-optimized estimates close to BI-HMC. Conversely, removing prehistory and
initial-covariance treatment from BI-HMC moved its posterior-mean errors close
to the zero-state optimization criterion. Priors and the preliminary posterior
optimization were retained in the reverse test.

These are finite-record numerical findings. They establish neither general
Bayesian superiority nor asymptotic inconsistency of the optimization methods.
The current implementation deliberately uses a common zero-state fitting
convention for comparability; the simulator's burn-in is unchanged.

The main experiments used the saved `example_1` and `example_CSTR` records:
30 independent records per benchmark, 10,000 retained observations per record,
100 discarded generation samples, and innovation standard deviation 0.1.
Inputs and outputs were matched by saved hashes. Ground truth was used for
assessment, not as an inference input. The error reported in the tables is

\[
E_\theta=100\,\frac{\|\widehat\theta-\theta_{\mathrm{true}}\|_2}
                        {\|\theta_{\mathrm{true}}\|_2}.
\]

Table entries are arithmetic means of this percentage error across the stated
records. They are not componentwise signed bias, parameter RMSE, or posterior
credible intervals. The separately estimated innovation variance is excluded
from the coefficient-error vector.

## 2. What differs at the record boundary?

Write the signal and disturbance as

\[
\alpha_t=\frac{B(z^{-1})}{A(z^{-1})}u_t,\qquad
\beta_t=\alpha_t+\sum_{j=2}^{n_f}f_j\alpha_t^j,\qquad
c_t=\beta_t+e_t,\qquad D(z^{-1})e_t=\nu_t.
\]

The input is named `r` in Python. The model is strictly causal in its linear
signal block, and the innovations are independent zero-mean Gaussian samples.

| Fitting protocol | Signal calculation | AR-noise likelihood |
| --- | --- | --- |
| Original optimization convention; current common convention | Start the signal filter at zero at the first retained input. | Set pre-record residuals to zero; score all retained innovations. |
| Former BI-HMC convention | Replay the observed input prefix, then retain the fitted signal over the measurement interval. | Integrate the unknown noise state through its finite-burn-in initial covariance. |
| Conditional protocol tested in FSM | Replay the same observed input prefix. | Use the first noise-order output observations for conditioning; score subsequent innovations. |

Increasing the simulator's burn-in does not make the signal state zero at the
retained-record boundary. With an ongoing input, a stable filter generally
carries a nonzero state into that interval. Replaying input history reconstructs
that candidate-dependent state. It must be recalculated when the candidate
coefficients change; initializing it once at the starting parameter vector is
not equivalent.

For a candidate coefficient vector, define residuals and reconstructed innovations:

\[
e_t(\theta)=c_t-\beta_t(\theta),\qquad
\widehat\nu_t(\theta)=e_t(\theta)+\sum_{i=1}^{n_d}d_i e_{t-i}(\theta).
\]

Under the common zero-state convention, unavailable pre-record residuals are
zero. Every retained sample contributes to

\[
Q_0(\theta)=\sum_{t=1}^{N}\widehat\nu_t(\theta)^2,\qquad
\log L_0=-\frac{N}{2}\log(2\pi\sigma_\nu^2)
          -\frac{Q_0(\theta)}{2\sigma_\nu^2}.
\]

The finite-record AR whitening matrix is unit lower triangular, so its
determinant is one. The output disturbance is still colored Gaussian noise;
zero initial states do not make the output noise white. It is the whitened
innovation sequence that is modeled as independent.

For the former finite-start likelihood, let \(m=\min(n_d,N)\), and let
\(\Gamma_d\) be the covariance of the first \(m\) noise residuals divided
by innovation variance. The initial part instead contributes

\[
Q_{\mathrm{full}}(\theta)=e_{1:m}^{\mathsf T}\Gamma_d^{-1}e_{1:m}
            +\sum_{t=m+1}^{N}\widehat\nu_t(\theta)^2,
\qquad \log L_{\mathrm{full}}=-\frac{N}{2}\log(2\pi\sigma_\nu^2)
          -\frac{\log|\Gamma_d|}{2}-\frac{Q_{\mathrm{full}}(\theta)}{2\sigma_\nu^2}.
\]

Both the residual mean and this covariance depend on candidate coefficients.
The conditional FSM experiment used input-history replay but omitted the first
\(n_d\) likelihood contributions. Here \(n_d=1\), so it scored 9,999
innovations. That tested alternative is distinct from the zero-state protocol
now adopted in production, which scores all 10,000 retained samples.

## 3. Numerical evidence

### 3.1 Separating optimization accuracy, priors, and initial states

The following controls used all 30 matched records per benchmark. Maximum
likelihood omitted both coefficient and innovation-variance priors and profiled
the variance. Posterior modes used the same physical priors as BI-HMC.
Refinement optimized the stated physical objective, without a sampling-coordinate
Jacobian term. A second refinement checked for residual numerical improvement.

| Procedure, 30 records per benchmark | Example 1 error (%) | CSTR error (%) |
| --- | ---: | ---: |
| Original saved FSM WS-GNI | 1.24945 | 2.25868 |
| Further optimization of the unchanged FSM criterion | 1.24945 | 2.05071 |
| Maximum likelihood with input replay and finite initial-noise covariance | 0.68207 | 0.79182 |
| Physical posterior mode with those full initial-state conventions | 0.68251 | 0.79155 |
| Saved BI-HMC posterior mean | 0.68245 | 0.78663 |
| Physical posterior mode with both states reset | 1.24788 | 2.05552 |
| Maximum likelihood with both states reset | 1.24945 | 2.05072 |

A well-optimized likelihood fit already reproduces nearly the saved Bayesian
mean accuracy when its initial-state conventions match. On these records,
posterior averaging and the selected priors are not required to reproduce that
average accuracy. This does not rule out their value for uncertainty estimation,
shorter records, weaker excitation, or other noise regimes.

For the likelihood controls, changing both boundary conventions reduced mean
error by 0.56738 percentage points in Example 1 and 1.25890 points in CSTR.
Paired percentile-bootstrap 95% intervals were [0.15741, 1.09255] and
[0.57499, 2.08678], respectively, using 20,000 resamples and seed 901109.
These quantify variation across records, not posterior uncertainty.

### 3.2 Which boundary component matters most?

| Physical posterior-mode treatment, 30 records | Example 1 error (%) | CSTR error (%) |
| --- | ---: | ---: |
| Signal prehistory and finite initial-noise covariance | 0.68251 | 0.79155 |
| Signal prehistory, but zero initial-noise state | 0.68289 | 0.79216 |
| Signal reset, but finite initial-noise covariance | 1.38812 | 1.75863 |
| Both states reset | 1.24788 | 2.05552 |

Once signal prehistory is included, changing only the noise-start covariance
has a small mean effect: 0.00038 and 0.00061 percentage points. The large effect
is associated with the signal restart. When that signal restart is wrong, the
two boundary components interact; their effects must not be added independently.

### 3.3 Literal FSM implementation test

This test retained the archived WS-GNI starting vectors, tuning and budget:
300 stored states, corresponding to 299 updates. It retained the FSM response
length of 128 and frequency grid of 1,024, and used the existing autodiff
gradient/Hessian and update rule. Only the signal-history replay and conditional
residual criterion were changed in an isolated process.

| Procedure, 30 records per benchmark | Example 1 error (%) | CSTR error (%) |
| --- | ---: | ---: |
| Original saved FSM WS-GNI | 1.24945 | 2.25868 |
| Conditional boundary-adjusted WS-GNI, original budget | 0.68255 | 1.07168 |
| Same corrected FSM criterion, additional L-BFGS refinement | 0.68255 | 0.79235 |
| Saved BI-HMC posterior mean | 0.68245 | 0.78663 |

The unchanged baseline reproduced the saved estimates. The conditional FSM
loss matched an exact IIR calculation to numerical precision at the checked
points. Finite-difference checks of its Hessian had relative discrepancies of
approximately 1.34e-9 and 2.13e-10.

The boundary change alone brought Example 1 to the Bayesian mean-error level.
CSTR retained numerical optimization error at the original budget: terminal
gradient norms were approximately 0.0026-0.0032. Refining that same corrected
criterion reduced them below 2.3e-8 and brought its average coefficient error
close to BI-HMC. The refinement used no Bayesian prior or posterior optimization.

### 3.4 Reverse experiment: remove BI-HMC's boundary treatment

The first five saved records of each benchmark were selected before examining
the reverse-test outcomes. Both normal and zero-state BI-HMC were rerun on
each record with paired sampler seeds. Priors, starting vectors, coordinate
jitter, 30-step L-BFGS initializer, adaptation settings and sampling budgets
were held fixed within each pair. Starting vectors were preserved for each
method, rather than made identical across different methods.

Example 1 retained four chains and CSTR retained one chain. Each chain used
300 warm-up transitions and 500 retained draws. The experiment therefore
comprised 20 complete HMC fits. The normal version replayed input history and
used finite-start noise covariance; the other version used neither treatment.

| Procedure, five matched records per benchmark | Example 1 error (%) | CSTR error (%) |
| --- | ---: | ---: |
| Fresh BI-HMC with its former boundary treatment | 0.58645 | 0.54489 |
| Fresh BI-HMC with signal and noise states reset | 1.35521 | 1.41459 |
| Original saved FSM WS-GNI on those same records | 1.35864 | 1.70701 |
| Well-optimized zero-state maximum likelihood on those records | 1.35865 | 1.40229 |

Removing the treatment increased mean HMC error by 0.76876 and 0.86970
percentage points. The five-record paired-bootstrap 95% intervals were
[0.15224, 1.52306] and [-0.01949, 2.23209]. The CSTR interval includes zero,
so this small MCMC subset alone does not establish a precise or statistically
significant population mean increase at the 95% level. Individual records can
improve by chance even when the aggregate error rises.

The zero-state HMC results closely follow the zero-state optimization target.
The higher original CSTR WS-GNI error includes additional finite-budget
optimization error, consistent with the separate refinement controls.

There were no retained divergences. Seven of ten Example 1 fits passed all
configured diagnostics; one normal and two zero-state fits exceeded the
1.01 R-hat threshold, with maxima 1.0295 and 1.0115 respectively. CSTR's
single-chain runs cannot provide multi-chain R-hat screening. Their minimum
mean-ESS values were approximately 348 with normal treatment and 493 with
zero states. These are exploratory point-estimation checks, not a claim of
fully certified posterior convergence or interval coverage.

The largest relative differences between HMC means and their corresponding
physical posterior modes were 0.019% in Example 1 and 0.066% in CSTR,
normalized by the mode-vector norm. Their small size relative to the observed
error shifts, together with the 30-record target controls, supports a change
of fitted statistical target as the main explanation rather than a large
sampling error in the reported means.

## 4. Why a short transient affects the AR coefficient

At the true coefficients, a signal restart leaves a deterministic discrepancy
between the actual signal path and the restarted one. The apparent output
residual contains that discrepancy as well as AR noise. AR whitening then
also acts on the discrepancy, so fitting the noise coefficients can partly
fit the transient's temporal structure.

Post-hoc diagnostics at the true signal parameters gave first-sample RMS
restart discrepancies of 1.62 in Example 1 and 1.27 in CSTR, against innovation
standard deviation 0.1. Most discrepancy energy was concentrated in the first
several samples. These diagnostics used truth only for evaluation.

| RMSE of the AR coefficient d1 across 30 posterior-mode fits | Example 1 | CSTR |
| --- | ---: | ---: |
| Both states reset | 0.02316 | 0.03163 |
| Input history and finite initial-noise covariance | 0.00992 | 0.00960 |

Signal-block coefficient RMSEs changed much less. For CSTR, the empirical
signed mean error in d1 decreased from approximately +0.01867 to +0.00075.
For Example 1, dispersion rather than a large signed mean shift accounted
for much of the effect. Thus an increase in mean norm error should not simply
be called statistical bias.

For a fixed stable model with finite total startup-error energy, its
contribution to an averaged criterion can vanish as the record grows. An
asymptotic inconsistency claim would require separate analysis of the target,
identifiability, excitation, stability and convergence conditions. These
finite-record experiments do not supply such a proof.

## 5. Preliminary optimization is a separate issue

A separate five-record-per-benchmark test compared single-chain HMC with and
without the 30-step initializer, keeping 300 warm-up transitions and 500
retained draws. Example 1 mean errors were 0.58537% with initialization and
0.58450% without. CSTR mean errors were 0.54489% and 49.69271%: two
uninitialized chains had errors above 123% and effective sample sizes near 3.

Those failures concern finite-budget sampling robustness. They do not show
that a converged Bayesian estimator has a 49.7% statistical error. The
initializer can prevent poor exploration even though a sufficiently optimized
likelihood fit can attain similar point accuracy. It was therefore retained
unchanged in both the reverse-boundary experiment and the production change.
Its cost belongs in total wall-clock time; HMC transitions are not comparable
one-for-one with optimizer updates.

## 6. Current implementation and rerunning studies

BI-HMC now uses zero signal and AR-noise states at the first retained sample
and scores every retained innovation. The input-history API, finite-start and
stationary covariance machinery, and conditional-noise option were removed.
Obsolete boundary settings raise an error. Priors, smooth stability coordinates,
Jacobian compensation, initialization optimization and physical posterior-mean
estimation are unchanged.

Saved input prehistory and its hash are retained as data provenance for
Python/MATLAB replay checks, without entering inference. Metadata distinguishes
the generation count `data_burn_in` from inference's `noise_burn_in=0` and
`input_history_length=0`. The policy is named `zero_state_fit_after_burn_in`.
The numerical distinction between exact IIR evaluation in HMC and FSM
approximation in the optimizers remains.

From the repository root, these commands run both benchmarks and both
parametrizations, with FSM first and PL through MATLAB:

```text
python python/run_studies.py --study montecarlo --cpu
python python/run_studies.py --study initialization --cpu
python python/run_studies.py --study noise --cpu
python python/run_studies.py --study record-length --cpu
```

Monte Carlo and initialization export matched Python records/starts for PL.
Noise and record-length studies retain their existing backend-specific record
generators and pairing rules. Add `--dry-run` to preview, or use `--experiment`
and `--parametrization` to select a subset. Existing result files were not
relabelled; regenerate them before presenting the new convention.

## 7. Validation, provenance and interpretation limits

The Python implementation passed 46 tests, including dense-Gaussian likelihood
checks, gradient/Hessian checks, stability transforms, posterior summaries,
real tiny study runs, shared-record export and command-plan validation.
Changing prehistory provenance while holding the retained data fixed produced
identical posterior draws, verifying that history no longer affects inference.

The MATLAB R2026a retry on 9 November 2026 passed all four Python-to-PL study
checks: Monte Carlo, initialization, noise sensitivity and record length.
The first two verified shared-record replay and paired exact-sign tests;
the latter two verified reuse of saved BI-HMC estimates in PL summaries.
The existing shared-BI-HMC alignment, cohort, timing and hypothesis tests also
passed. Separate existing checks for posterior/comparison plots, study outcome
and runtime reports, and paired parameter-error tests passed as well.
That separate plotting batch completed every assertion but stalled during
shutdown; its two isolated batch processes were stopped after checking the
success markers, without targeting the interactive MATLAB session.
The smoke fixtures used two repetitions, 16-24 observations, one
optimizer plus BI-HMC, and deliberately tiny sampling budgets. They validate
integration, not posterior convergence or experimental accuracy. All fixture
files were removed; existing study results were preserved.

The retry exposed a MATLAB singleton-method summary-shape error in the
record-length path. Explicit allocation now preserves the parameter axis in
record-length and noise statistics. A new regression check passed for both
studies, recomputing the single-method summaries independently and checking
the inverse-record-length fit. This is a summary-storage correction, not a
change to an optimizer or to the boundary convention. Earlier startup errors
5201 were avoided by allowing normal MATLAB preference and licensing access.

The historical ablations were isolated from production code. Their temporary
scripts and outputs were deleted as requested; the numerical summaries above
are transcribed from the checked session results, and this report is not a raw
chain archive. During the reverse test, the CSTR comparison file was updated
by another process. Its starting version was read from Git and verified
against the initial checksum; the newly updated file was preserved. The
five-record HMC and 30-record control tables therefore refer to the original
matched records, not a mixture of result revisions.

The inference is limited to these benchmark settings and representative FSM
WS-GNI comparisons. It does not rank every optimizer at every record length,
noise intensity or initial parameter vector. Aligning boundary assumptions
removes a demonstrated comparison confound; differences in priors, numerical
optimization, approximation accuracy and posterior uncertainty remain legitimate
subjects for the regenerated experiments.

Related implementation and usage sources: [BI-HMC implementation](bayesian_hmc_implementation.md),
[study design](identification_studies.md), [likelihood code](../python/bayesian_hmc.py),
[experiment launcher](../python/run_studies.py), and [repository README](../README.md).
