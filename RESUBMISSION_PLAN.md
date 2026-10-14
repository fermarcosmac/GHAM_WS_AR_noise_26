# JFI resubmission plan: GOTHAM for Wiener systems with AR noise

Prepared 14 October 2026. Manuscript ID: FI-D-26-02046.

Status: analysis and proposed work only. The manuscript, identification programs, configurations, and saved experiment results have not been changed. Reviewer suggestions are treated as requests to assess and answer, not instructions that automatically determine the scientific conclusions.

## 1. Overall assessment

All three reviewers recognize the originality of the operator-theoretic framework. The central contribution should remain the unification and systematic construction of identification algorithms. The experiments should establish when the resulting configurations are useful, rather than attempt to demonstrate universal superiority.

Reviewer 1 recommends minor revision, but Reviewer 3 raises substantial methodological issues. The editor's decision letter was not supplied, so no overall editorial decision is inferred. A persuasive response needs both a shorter presentation and stronger evidence. Adding paragraphs and more plots without restructuring would worsen Reviewer 1's principal concern.

There are 17 numbered comments, plus unnumbered concerns about bias, computational cost, theoretical convergence, identifiability, and experimental fairness. Those general concerns need explicit responses too.

Recommended order: verify the criterion and evaluation protocol; resolve theoretical scope and implementation details; select and tune external comparators; run a coordinated experiment set; analyze statistics and figures; shorten and polish; complete the response letter.

## 2. Findings that determine the revision priorities

Locations below refer to section/equation/table numbers in the supplied PDF. Physical PDF pages include the graphical abstract and highlights.

### A. The output metric needs an explicit definition and new validation

Section 5 describes output reconstruction RMSE, but the current Python and MATLAB comparison programs reconstruct the output using the actual synthetic innovation sequence. For example, `python/comparison_torch.py`, lines 184 and 252, calls `simulate_wiener(r, nu, theta_hat, dims)`. The plotting program also uses this reconstruction. This is a legitimate simulation diagnostic if clearly labeled, but it is not ordinary prediction from observed inputs and past outputs. It can explain why the reported error is much smaller than the innovation standard deviation.

The Python estimation loss itself discards the supplied innovation argument and computes innovations from measured data (`python/wiener_system.py`, `params_to_loss`). Thus, use of the true innovation in evaluation does not by itself establish that the estimator uses unavailable information. Audit estimation, tuning, and evaluation separately.

Required revision: use independent test records and distinguish (i) deterministic signal reconstruction against the known clean signal in simulation; (ii) one-step prediction of the measured output using only information available at prediction time; and, if retained, (iii) reconstruction driven by the true synthetic innovation, explicitly identified as an auxiliary diagnostic. Future measured output must not enter a purported free-run simulation.

### B. The loss-root target creates an additional dependency and cost

Section 5 states that epsilon_0 comes from the minimum loss reached by other algorithms. The Python driver passes the best previous non-LHAM final loss into LHAM on each record. Its recorded LHAM runtime excludes the preceding algorithms' cost. That is a hybrid workflow and must be reported as such, or replaced by a self-contained target-selection rule.

Plan: investigate a training-data-only target or adaptive target strategy, with explicit safeguards; compare it against the existing externally calibrated target as an ablation. A noise-variance estimate is a candidate reference level, not a guaranteed finite-sample optimum. If calibration still requires preliminary optimization, include that cost and information in the comparison.

The mathematical statement in Section 4.3 also needs tightening. A prescribed loss level does not in general guarantee a root or a unique minimizer: a level below the minimum is unattainable, and a level above it may define many solutions. Preserve the algebraic connection to Polyak steps while separating it from root-existence and convergence claims.

### C. Parameter error is not an estimate of statistical bias

Sections 5 and 6 attribute nonzero limiting parameter error to bias. An individual trajectory or the mean norm of estimation error does not establish bias. Estimate the signed Monte Carlo bias vector, its uncertainty, and its change with data length. Separate finite-sample variability, optimization error, FSM approximation error, and possible structural mismatch.

High parameter error with low output error is not sufficient evidence of overfitting. It may reflect weak sensitivity, compensating parameter changes, differences between signal and noise parameters, or a limited input distribution. A training/test gap and sensitivity diagnostics can distinguish these explanations.

### D. Some theory is conditional, and identifiability is assumed

Section 3 assumes an analytic homotopy; it does not establish general convergence of the implemented outer iteration. Appendix B explicitly assumes identifiability. Fixing f_1 = 1 removes one scale ambiguity but does not prove that all remaining parameters are identifiable.

Separate four questions: convergence of the infinite homotopy series; accuracy of a truncated homotopy expansion; convergence of repeated finite-order updates; and statistical identification of the true system. Establish appropriately limited results and state the conditions, rather than promise global convergence for every generated variant.

### E. Reproducibility details affect fairness and interpretation

The local saved Monte Carlo files reproduce the parameter-error means in Tables 5-8. They contain per-run final parameter vectors, parameter-error and RMSE histories, and times. There are deliberately unimplemented/omitted method rows with NaNs; these must be distinguished from numerical failures.

Further audit items:

- The code uses a mean-square FSM loss, while Eq. (93) uses half a sum of squares. The conversion changes gradients, step sizes, regularization, and epsilon_0.
- Noise-parameter step multipliers and stability projection appear in the implementation/configuration and need complete documentation.
- The FSM defaults resolve to truncation lengths 128 and a frequency grid of 1024 for the current record length, but the experimental settings do not fully document these choices. Confirm which defaults generated each saved run.
- The current innovation criterion filters the signal residual directly with D. Clarify whether tau_h is relevant to estimation at all, or only to the alternative reconstruction in Section 2. Remove obsolete dependencies if appropriate.
- Curves are plotted with standard errors in the local plotting program, whereas captions describe standard deviations. Verify and correct.
- The timing routines measure elapsed time. Label it wall-clock time rather than CPU time unless actual CPU time is measured.
- The nominal 300 stored iterations include initialization and may include repeated values after early convergence. Report actual updates and actual stopping behavior.
- Identical seed integers in MATLAB and Python do not establish identical data. Shared input/output arrays are needed for paired PL-versus-FSM comparisons.
- Verify whether the baseline labeled RLS is a repeated batch least-squares update and describe it accurately. Keep Newton, Gauss-Newton, and published WS-GNI terminology distinct.
- The CSTR case is a simulated Wiener model using published CSTR-associated parameters. Do not describe it as new experimental reactor measurements.

## 3. Comment-by-comment response map

Priority P0 means a dependency for valid interpretation; P1 means substantive revision; P2 means presentation or reporting. Every row remains pending until its evidence and manuscript changes are complete.

| Comment | Assessment and planned action | Manuscript target and completion evidence | Priority |
|---|---|---|---|
| R1.1: Length, notation, numerous variants, no selection guidance | Agree. Retain one general recursion, essential equivalences, a compact notation table, and a practical method-selection guide. Move long expansions, repeated proofs, exhaustive variant formulas, and repetitive trajectories to supplementary material. Keep two or three representative GOTHAM configurations prominent; retain the full comparison in the supplement. | Sections 2-5; compact replacement for Table 2; shorter main text with all central assumptions preserved. | P1 |
| R1.2: Recent sparse Bayesian and deep-learning comparisons | Add relevant external methods under the same experimental protocol, provisionally one sparse Bayesian method, one neural identification method, and one close colored-noise/filtering method. Screen model and data assumptions before committing to implementations. Historical equivalence baselines remain useful. | Introduction; experimental protocol; new comparator table and reproduced results. Cite recent work, but distinguish literature discussion from an actual controlled comparison. | P1 |
| R2.1: Replace first-person wording | Apply the requested objective style contextually: “the authors” where appropriate, or direct formulations such as “The framework yields...”. Check highlights, abstract, main text, and conclusions. | Final language pass without mechanical, awkward substitutions. | P2 |
| R2.2: Method flowchart | Add a Wiener-specific flowchart showing observed data, preprocessing/initialization, PL or FSM criterion, residual/operator choice, deformation loop, regularized solve, admissibility/acceptance check, stopping, and returned estimate. Show the distinct roles of homotopy order m and outer iteration k. | Section 4 alongside a shorter Algorithm 1; diagram consistent with the final implementation. | P2 |
| R2.3: Output-error plots | Replace repetitive output overlays with signed error traces and a compact summary on a prespecified test record/window. Define the reference signal and prediction mode explicitly. Do not cherry-pick a favorable run. | Section 5; residual plots for both benchmarks with common scales and clear captions. In this PDF Fig. 4 is already an aggregate error curve; respond to the intended request rather than argue about numbering. | P1 |
| R2.4: Compare with other researchers | Address jointly with R1.2. Prefer reproductions on common data. Values copied from papers with different noise, data length, tuning, or initialization may be contextual only. | Same comparator experiment; separate response-letter entry pointing to shared evidence. | P1 |
| R2.5: Percentage-improvement graphs | For a lower-is-better metric E, use 100(E_baseline - E_method)/E_baseline, with a declared comparator and aggregation rule. Retain negative values, uncertainty, absolute metrics, and all relevant scenarios. Define handling of zero/near-zero denominators. | Compact percentage-change figure; no universal “improvement” claim where results are equal or worse. | P2 |
| R2.6: Statistical tests for 30 runs | Use run-level pairing on genuinely shared data. Prespecify primary methods/metrics and the multiple-comparison family. Use paired tests with appropriate assumptions, effect estimates, confidence intervals, and multiplicity adjustment. Treat nested random starts as nested observations. | Statistical protocol, table of adjusted p-values and effect sizes, complete run counts and failure reporting. | P1 |
| R2.7: Advantages and limitations | Add a focused discussion: unification, operator flexibility, derivative/order trade-offs; local convergence limits, target/tuning dependence, conditioning, finite-data uncertainty, computational overhead, and restrictions of the tested systems. | New discussion subsection and revised conclusions tied to observed evidence. | P1 |
| R3.1: Generalized inverse and FSM regularization | Specify the actual solve, its regularization/thresholds, scaling, and failure handling. Distinguish singularity from indefiniteness; an arbitrary positive diagonal shift need not make a Hessian positive definite. Compare damping/regularization settings and record accepted steps. | Sections 3.4 and 4.4; solver specification and a focused regularization ablation. | P0 |
| R3.2: High-order HODE complexity, especially M > 2 | Give costs in record length, parameter dimension, FSM grid/truncation length, and homotopy order. Separate derivative construction, factorization, correction solves, memory, and calibration. Compare M = 1, 2, 3 in one consistent family; do not compare different residual/operator choices and attribute everything to order. | Complexity subsection and matched timing/accuracy study; clarify different derivative requirements for loss-root and gradient-root formulations. | P1 |
| R3.3: Fixed initial values | Add nominal, nearby perturbed, and broader stable initializations independent of ground truth. Share starting vectors across comparable methods and report failure/success rates, attained loss, test error, and time. | Initialization-robustness study; reproducible generation rule and constraints. | P1 |
| R3.4: eta, lambda, tau_g, tau_h, N_omega selection | Provide an actionable tuning procedure, ranges, defaults, selection metric, and validation budget. Include epsilon_0, stability limits, step scaling, and stopping thresholds. Separate optimizer regularization from FSM numerical accuracy. | Consolidated settings table and sensitivity plots; account for criterion normalization and any irrelevant tau_h setting. | P0 |
| R3.5: Parameter/output-error disagreement | Reassess after redefining evaluation. Split signal-path and noise-path errors, estimate signed bias, evaluate fresh records, and examine parameter sensitivity/correlation. Treat overfitting and compensation as hypotheses to test. | Diagnostic discussion linked to corrected Tables 5-8, bias and validation results. | P0 |
| R3.6: Missing filtering literature | Read the four suggested papers and add a concise comparison of system architecture, noise, measurement type, and estimation principle. Explain the relation of the FSM innovation criterion to filtering/whitening and prediction error. Only use directly compatible methods as numerical baselines, or clearly disclose adaptations. | Revised introduction/related-work paragraph and comparison table. Binary/quantized or Hammerstein results need not be treated as interchangeable Wiener-AR benchmarks. | P1 |
| R3.7: Slow CSTR FSM convergence | Measure scaled innovation-Jacobian singular values, Hessian eigenvalues/conditioning, gradient norms, step norms, damping and accepted steps. Vary nonlinearity strength and parameter scaling separately while controlling SNR and budget. Compare time-to-target. | A diagnostic experiment distinguishing conditioning, curvature, scaling, tuning, and nonlinear strength; replace unsupported causal assertions. | P1 |
| R3.8: Only sigma_nu = 0.1 | Test lower and higher innovation noise levels for both systems. Proposed initial grid: 0.01, 0.05, 0.10, 0.20; confirm relevance in a pilot and report achieved output SNR. Retain noise-free data only as a separate diagnostic if useful. | Noise-robustness plot/table with 30 independent records per principal condition and fixed declared tuning policy. | P1 |
| R1 general: Bias and computational cost | Address explicitly in addition to the two numbered comments. | Signed-bias analysis, sample-length check, controlled runtime study, and discussion. | P0/P1 |
| R3 general: Convergence, identifiability, fairness | Address explicitly; do not assume the eight numbered responses cover the opening assessment. | Conditional convergence statement, identifiability assumptions and numerical evidence, and reproducible common-data protocol. | P0 |

## 4. Work sequence and completion conditions

### Phase 0: Establish the experimental contract

1. Archive the submitted manuscript, current results, source revision, and exact configurations as a reference.
2. Trace every reported metric to its formula, data, and implementation; resolve the discrepancies in Section 2 above.
3. Define training, validation, and independent test records. A practical starting design retains 10,000 training samples and uses separate validation/test records; freeze lengths and seeds before final runs.
4. Generate the data once and share the saved arrays across languages, methods, and parametrizations. Store per-record identifiers and hashes as well as seeds.
5. Define admissible parameter sets, stopping criteria, failures, runtime boundaries, and tuning budgets. Separate initialization, fitting, target calibration, and evaluation time.

Completion condition: every algorithm consumes permitted data, every metric has a single meaning, and the same experiment can be reproduced. This phase answers parts of R3.4/R3.5 and prevents invalid comparisons later.

### Phase 1: Tighten the mathematical claims and numerical specification

Plan a short local convergence result for a specific update under explicit smoothness, bounded positive-definite preconditioning, and step-acceptance/step-size conditions, if those conditions can be shown for the actual implementation. Do not extend that result automatically to every higher-order update or to the changing PL outer map. A small step alone is not proof of convergence to a stationary point.

For identifiability, state the normalization, correct model orders, stable monic noise filter, suitable excitation, independence assumptions, minimal/coprime signal-path representation, and any nonlinearity restrictions needed. Examine the rank and scaling of the innovation sensitivity matrix. Numerical full rank supports local empirical identifiability at the tested points; it does not prove global identifiability.

Resolve the role of generalized/regularized solves: if a right-hand side lies outside the range of L, the computed step need not solve the original HODE exactly. State the approximation and its effect on any convergence argument. Likewise, LASSO or constrained solution maps should not be described as linear generalized inverses without qualification.

Audit the Taylor-series convergence assumption in Section 3.3 and the Neumann-series argument in Appendix A. Specify sufficient conditions for the expansion actually used; do not infer every required series property solely from filter stability or real analyticity along [0,1]. These are internal mathematical checks, additional to the numbered reviewer comments.

For FSM, quantify truncation/aliasing accuracy under a specified stable parameter region and verify it by comparison with the recursive model on identical inputs. A grid-doubling check and a tail-error tolerance provide practical selection guidance. Separate this numerical approximation from identifiability and statistical consistency.

Completion condition: each theorem/proposition has explicit scope, all implemented safeguards are described, and no empirical observation is used as a general proof.

### Phase 2: Select external comparators and freeze tuning

Retain the historical methods that establish the equivalences. Add a small, defensible external comparator set covering filtering/prediction error, sparse Bayesian identification, and neural identification. Final selection requires a methods/code feasibility check, not just a publication-year match.

Screen each candidate for: Wiener versus other architecture; colored measurement noise versus process noise; ordinary versus binary/quantized outputs; required input excitation; parameter normalization; available implementation; training budget; and compatible evaluation tasks. Methods without matching physical coefficients should receive output-performance and cost comparisons, not artificial parameter-error scores.

Select representative GOTHAM methods using validation performance and intended use, then freeze their names before the final test evaluation. Use all variants in a supplement to support the algorithm-generation contribution. Compare the same criterion with an ordinary well-tuned optimizer where possible, to separate the value of the parametrization from the value of the GOTHAM update.

Completion condition: one comparator table specifying assumptions, implementation source/adaptations, tuning budget, and reported metrics for every method.

### Phase 3: Run coordinated experiments

| Experiment | Proposed scope | Main questions answered |
|---|---|---|
| Controlled comparison | Two benchmarks; common records; 30 independent training/validation/test realizations per principal condition; representative methods plus external baselines. | R1.2, R2.3-6, R3 fairness and R3.5. |
| Noise robustness | Both systems; proposed sigma_nu grid 0.01, 0.05, 0.10, 0.20; record actual SNR; fixed tuning learned on separate data, with per-noise retuning shown separately if needed. | R3.8, general robustness. |
| Initialization robustness | Nominal initialization plus prespecified nearby and broad stable draws. Pilot with a small set of independent records; then use approximately 10 starts per record on the selected final subset, subject to measured runtime. | R3.3, local convergence limitations. |
| Regularization/FSM accuracy | Selected methods; regularization grid and truncation/frequency refinement; include a recursive-model reference. | R3.1, R3.4, FSM approximation and conditioning. |
| CSTR diagnosis | Log conditioning/curvature/scaling diagnostics; vary nonlinearity strength independently; control SNR and data length. | R3.7, R3.5. |
| Order and cost | Same residual and auxiliary-operator family at M = 1, 2, 3; representative record lengths and, if feasible, a larger parameter dimension. | R3.2 and Reviewer 1's runtime concern. |
| Bias versus sample length | Suggested lengths 2,500, 10,000, 40,000; representative methods; signed parameter bias and confidence intervals across runs. | Reviewer 1's bias concern, R3.5, revised conclusions. |

Use a staged design rather than the full Cartesian product of methods, starts, noise levels, and lengths. First check correctness and runtime on pilot data, then freeze the confirmatory experiments. Keep pilot/tuning records out of the final inferential test set.

Completion condition: every requested new claim has a corresponding experiment, all failures remain visible, and the run-level output is saved before aggregation.

### Phase 4: Statistical analysis and revised figures

The unit of replication is the independently generated experiment, not an iteration or an individual time sample. For multiple starts within one data realization, summarize within the realization or use an explicitly hierarchical analysis.

For planned pairwise comparisons on shared data, use paired differences. A Wilcoxon signed-rank test is an option when its symmetry assumptions are reasonable; a paired permutation approach needs its own exchangeability assumptions. A sign test is a possible fallback for a weaker direction-of-effect claim. Prespecify the test rule, two-sided alternatives by default, alpha = 0.05, and Holm adjustment across the declared comparison family. Report effect sizes and 95% confidence intervals as well as adjusted p-values. Handle ties and numerically negligible differences explicitly. An omnibus Friedman analysis is optional when a broader within-scenario ranking question is relevant; do not pool heterogeneous conditions indiscriminately.

Do not select the best GOTHAM configuration on the same test runs used to claim superiority. Do not interpret a nonsignificant difference as equivalence; equivalence would require a practical margin and suitable analysis. A hypothesis test also cannot establish estimator consistency, which is an asymptotic property.

Diebold-Mariano tests concern forecast-loss comparisons along time and require attention to serial dependence. They are not an automatic replacement for paired analysis of 30 independent experiment-level scores. The first reviewer-suggested DOI mentions this test, but its setting must be distinguished from the present design.

For percentage change, use paired per-run ratios when denominators are meaningful, or a clearly labeled ratio of aggregate metrics with uncertainty computed by resampling whole paired runs. Never mix the two definitions without explanation.

Proposed main-text figures: method flowchart; test-error traces; accuracy-versus-wall-time curves; noise/initialization robustness; one compact percentage-change panel. Combine closely related panels. Put full parameter trajectories, all variant comparisons, and extended sensitivity grids in the supplement. Keep the ordinary absolute-error tables beside relative changes.

Completion condition: conclusions remain supported after multiplicity adjustment and practical-effect assessment; captions match the plotted uncertainty; equal or unfavorable results are reported honestly.

### Phase 5: Shorten, revise claims, and assemble the resubmission

The supplied PDF contains 42 physical pages. Set an editorial target of a materially shorter main text (provisionally about 20-25% shorter, not a claimed journal requirement) by removing duplication and moving detailed material to the supplement. Do not achieve this through smaller typography.

Suggested structure: focused introduction and related work; concise Wiener-AR problem/criterion; general GOTHAM recursion and limited theoretical results; practical implementation, flowchart, and method-selection guide; common experimental protocol and results; advantages/limitations; conclusions.

The practical selection guide should state which configuration is a low-cost starting point, when curvature information is useful, when regularization is required, how to choose M, and when FSM is justified by the identification objective. Final recommendations must follow the new results.

Reassess phrases such as “state of the art,” “biased limiting estimate,” “stable,” and “fastest.” Distinguish accuracy at convergence, speed per iteration, wall-clock efficiency, numerical stability, and statistical reliability. A close final tie can still support the unification contribution.

Prepare separate responses for all 17 numbered comments and the general concerns. Each entry should contain the concern, a concise response, the exact change, supporting evidence, and final manuscript/supplement locations. Where a suggestion is inappropriate, explain the scientific reason and the alternative evidence supplied. Draft responses alongside each completed work item, then update all locations after final typesetting.

Final package: clean manuscript; marked manuscript; response letter; supplementary methods/results; reproducibility instructions and fixed experiment configurations. The workspace currently contains numerical code and results but no manuscript `.tex` or bibliography source, so those sources will be needed for the later editing phase.

## 5. What can be reused, and what requires fresh work

| Available now | Use and limit |
|---|---|
| Four saved Monte Carlo datasets matching Tables 5-8 | Initial paired within-dataset comparisons, signed-bias estimates, distributions, and exploratory plots. Treat as exploratory because they were already used for tuning/reporting. |
| Per-run final parameter estimates | Can be evaluated on fresh independent inputs after confirming model conventions and provenance; this may avoid refitting for an initial validation check. |
| Per-run error histories and total times | Useful for diagnosis, but current RMSE needs explicit relabeling and timing omits some calibration costs. |
| Existing theory, equivalence results, and figures | Preserve the core contribution; shorten proofs/duplicate figures and update claims. |

Fresh work is needed for externally reproduced baselines, robust independent-data comparisons, initialization/noise sweeps, properly controlled timing, condition-number/curvature diagnostics, and sample-length evidence. Reuse does not remove the need to verify shared data and paired run identities.

## 6. Literature screening notes

These are candidates and reading priorities, not claims that all have been fully read or that their methods are directly compatible.

- [An Application of Partial Update Kalman Filter for Bilinear System Modelling](https://link.springer.com/article/10.1007/s13369-024-09313-x): publisher abstract/preview verified. The setting is bilinear identification and the abstract mentions Diebold-Mariano, independent runs, and cross-validation. Useful methodological context; not automatically a Wiener-AR comparator. Detailed methodology needs full-text inspection.
- [Reviewer-suggested DOI 10.1016/j.aeue.2025.155943](https://doi.org/10.1016/j.aeue.2025.155943): publisher full text could not be accessed in this planning pass. Inspect before describing or adopting its statistical procedure; do not guess the test from the recommendation.
- [Data Filtering-Based Maximum Likelihood Gradient Iterative Identification for Hammerstein Systems With Asymmetric Preload Nonlinearity](https://onlinelibrary.wiley.com/doi/full/10.1002/rnc.70657): publisher abstract verified; useful for R3.6, with an explicit architecture distinction.
- Reviewer 3's other three suggested titles remain full-text screening tasks: “Composite Error Learning Estimator for Wiener-Hammerstein Systems With Time-Delay Under Binary-Valued Measurements”; “Hierarchical generalized extended parameter identification for multivariable equation-error ARMA-like systems by using the filtering identification idea”; and “Adaptive Learning-Based Quantized Parameter Identification of Nonlinear Sandwich-Like Systems With Memory Nonlinearity.” Do not treat their measurement regimes as equivalent merely because they concern nonlinear identification.
- [Sparse Bayesian deep learning for dynamic system identification](https://doi.org/10.1016/j.automatica.2022.110489): publisher search preview identifies a potentially relevant intersection of the two families requested by R1.2. Verify the complete method, code, and noise assumptions before selection.
- [An Iterative Min-Min Optimization Method for Sparse Bayesian Learning](https://proceedings.mlr.press/v235/wang24al.html): 2024 primary conference source verified; an SBL candidate requiring a suitable regression/model representation, not a ready-made Wiener-AR algorithm.
- [Identification for nonlinear systems modelled by deep long short-term memory networks based Wiener model](https://doi.org/10.1016/j.ymssp.2024.111631): publisher search preview verified. Its two-stage procedure uses particular excitation signals, so compatibility with the current random-input protocol must be assessed.
- [dynoNet](https://arxiv.org/abs/2006.02250): author preprint offers a structurally relevant neural benchmark candidate. It is older, so it cannot alone substantiate a claim of current state-of-the-art coverage.
- [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html): reference for paired-test assumptions, ties, zeros, and numerical handling. Final inferential choices should be justified by the experiment design, not by software defaults.

## 7. Source record

- Manuscript: `C:/Users/ferma/Downloads/GHAM_JFI_paper_REBUTTAL.pdf`.
- Reviewer comments: `C:/Users/ferma/.codex/attachments/16e0d0f7-4c04-43e4-bc0c-85f932e9dfa6/Texto pegado.txt`.
- Implementations inspected: `python/comparison_torch.py`, `python/wiener_system.py`, `python/methods.py`, `python/common.py`, `mlab/comparison.m`, `mlab/plot_comparison_results.m`.
- Settings inspected: `configs/example_1.json`, `configs/example_CSTR.json`, `python/optim_configs.json`.
- Saved results inspected: the PL and FSM `comparison_*_montecarlo.mat` files under `results/example_1/` and `results/example_CSTR/`.

First work item when revisions begin: settle the estimation/evaluation definitions and epsilon_0 protocol, addressing R3.5 together with the general fairness concern. This makes the later error plots, comparisons, hypothesis tests, and computational-cost claims interpretable.
