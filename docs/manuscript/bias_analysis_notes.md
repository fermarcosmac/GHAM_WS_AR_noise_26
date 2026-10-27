# Population objective and completed record-length analysis

The appendix now treats both PL and FSM population objectives, reviews the
scope of analytical bias expressions for configured iterative estimators with
supporting references, and discusses the completed record-length results.

## Insertion and figures

Insert appendix_bias_analysis.tex after the existing appendices without
issuing another appendix command. It uses the external label
app:fsm_innovation_energy and requires amsmath, amssymb, graphicx, and booktabs.
The figure filenames assume that figs/ is under the manuscript root.

Eight mean-error and RMSE PDFs are included in four paired figures: PL and
FSM128 for each benchmark. FSM256 remains in the finite/converged outcome
table at N=10000. The complete outcome
breakdowns remain in figs/record_length_outcomes_*.csv and their original PDFs.
Figures are regenerated from the saved MAT studies using the user's revised
mlab/plot_record_length_study.m. RMSE panels now use logarithmic vertical
axes; signed mean-error panels retain linear vertical axes capped at [-1,1],
while preserving tighter automatic ranges. Outlying plotted values and
error-bar portions are clipped, without modifying their statistics. Repeated axis
labels are suppressed by the plotting function. The appendix captions and
discussion of extreme values reflect this display. No estimation study is rerun.

The standalone preview provides the external appendix reference as D,
searches for figures relative to the repository root, and uses the supplied
bias_analysis_references.bib. Merge its six citation entries into the full
manuscript bibliography.

The omitted CSTR FSM256 curves coincide visually with FSM128. Example 1
regular curves, especially at large N, also agree, but short-record extreme
outliers can differ substantially. The text states this qualification rather
than claiming every curve is identical.

## Optional aggregate-bias figures

The existing appendix and its parameter-wise figures are retained. The separate
aggregate_bias_preview.tex previews an alternative with one optimizer per panel,
using the Euclidean norm of the six-component signed mean-error vector. The
norm is taken after averaging each parameter error over retained runs. It is
neither the average error norm nor total parameter RMSE. Parameters use their
original scale, with equal weights; no division by true parameters is used.
This is a plug-in Monte Carlo estimate of the bias norm, so finite-run
uncertainty also affects its magnitude, especially when the true bias is small.
The parameter-wise 95% intervals are not transferred to this nonlinear norm.

mlab/plot_record_length_aggregate_bias.m exports six additional PDFs and four
CSV reports from the saved studies. plot_record_length_study.m generates both
views, appending the aggregate figures after all original bias/RMSE/outcome
figures, and exposes the aggregate report as an optional third output.
The aggregate axes are linear on [0,1], retaining tighter ranges; upward
triangles identify clipped values. Missing/skipped optimizers remain missing,
and changing retained counts are indicated in the corresponding panels.
The separate preview is output/pdf/aggregate_bias_comparison.pdf.

## Theory and source scope

- The ideal PL innovation regression has population loss equal to the
  innovation variance plus a quadratic parameter-error term.
- For an actually reconstructed frozen PL row, the exact expansion retains
  its effective-error cross moment q_k. The shift R_k^{-1} q_k concerns the
  population surrogate minimizer; it is not a finite-record bias formula for
  the whole algorithm. Row-index dependence is suppressed; record-averaged
  moments give the corresponding expected batch criterion.
- The stationary FSM innovation identity is retained, with separate
  qualifications for finite signal approximation and initial states.
- Ljung and Soderstrom (1983), Theory and Practice of Recursive Identification,
  MIT Press, ISBN 9780262120951:
  https://mitpress.mit.edu/9780262120951/theory-and-practice-of-recursive-identification/
- Soderstrom, Ljung and Gustavsson (1978), Automatica 14(3), 231-244,
  doi:10.1016/0005-1098(78)90088-2, studies recursive-method convergence:
  https://www.lunduniversity.lu.se/publication/0c6b86f2-43bf-4f03-b05d-5feb1cd34f56
- Ljung (1981), Automatica 17(1), 89-99,
  doi:10.1016/0005-1098(81)90086-8, treats general recursive prediction-error
  algorithms and simplified-gradient approximations.
- Soderstrom (2007), doi:10.1016/j.automatica.2006.11.025, supports the
  regressor-contamination mechanism; https://user.it.uu.se/~ts/eivplenaut02.pdf
- Ni, Gilson and Garnier (2013), doi:10.1049/iet-cta.2012.0548, concerns
  reconstruction and IV consistency for continuous-time Hammerstein-Wiener
  models with coloured output noise.
- Janczak and Korbicz (2019), doi:10.2478/amcs-2019-0042, gives an analytical
  correlation/inconsistency mechanism for a transformed Wiener regression.
  Its disturbance placement differs from the present output-AR model.

These sources establish substantial method-specific theory. They are not
presented as proving that a common exact finite-record bias formula exists,
or that joint reconstruction alone causes bias in the implemented methods.

## Interpretation checked against saved results

- Each saved study uses 100 realizations, N=500,1000,2000,5000,10000,
  K_max=300, and theta0=(0,0,1,0,0,0). The seed bank is 271828:271927.
- The MAT study metadata, not the older top-level N_MC field in the FSM
  experiment metadata, determines the actual number of realizations.
- Independent records and shared prefixes apply within each implementation;
  MATLAB PL and Python FSM records are not asserted to be paired.
- All finite returns enter statistics, including stalled and iteration-limited
  estimates. Failed outputs are excluded, making some summaries conditional.
- Intervals are pointwise normal-approximation Monte Carlo intervals of the
  mean error, not confidence intervals for one record or simultaneous bands.
- Example 1 FSM GHAM-2-I has only 7 finite outputs at N=10000 at both
  resolutions; 93 failures must accompany any interpretation of its curves.
- Example 1 FSM128 WS-GNI has a b1 RMSE approximately 1.38e14 at N=500;
  extreme finite outliers are retained in the statistics and logarithmic
  RMSE panels, while the capped linear mean-error panels clip values and
  error-bar portions outside the displayed range.
- CSTR offsets survive the tested FSM resolution change; this does not
  identify their cause or establish inconsistency.
- A configured convergence flag can mean loss-target attainment and does
  not certify unbiasedness or a global innovation-energy minimum.

Analytical bias is described as generally unavailable in closed form for
the complete implementation, not as mathematically impossible.
