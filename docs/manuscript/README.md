# Condensed Section 4 and associated appendices

## Files and insertion

- `section4_revised.tex`: replace the supplied Section 4 in full, including its
  old large table and flowchart. This fragment has no document preamble.
- `section4_appendices.tex`: append after the existing Appendices A--G. It adds
  three sections, which will be numbered H--J in the supplied manuscript.
  Do not issue a second `\appendix` command when appendix mode is already active.
- `section4_preview.tex`: a flattened, standalone reading/compilation copy of
  both fragments. Its external reference numbers and three bibliography entries
  are supplied only to make this preview readable. Do not paste its preamble,
  preview notes, or reference definitions into the manuscript.

In the manuscript, use the existing theorem definitions and bibliography.
The fragments need `amsmath`, `amssymb`, `amsthm`, `booktabs`, `graphicx`, and
the package that already supplies `\mathscr` (for example, `mathrsfs`).
Keep the existing citation keys `Lv2026`, `Wang2020`, and `Polyak1969`.

For example, from a manuscript whose root contains these files:

```latex
% At the original location of Section 4:
\input{section4_revised}

% Later, after the existing Appendix G:
\input{section4_appendices}
```

The optional flowchart block in the last appendix uses the existing
`figs/GOTHAM_flowchart.pdf`. It creates no placeholder when that graphic is
unavailable, as in this repository. If the full manuscript uses another figure
path, update it there. References to that figure require the graphic to exist.

## Editorial structure

The new main section has three subsections:

1. The two implementable identification criteria and a common notation.
2. The explicit first- and second-order gradient-root updates in a compact table,
   their classical special cases, and the auxiliary-operator remark.
3. The first three loss-root corrections, the Polyak connection, and the
   target/inverse/implementation remarks.

The appendices contain:

- **H:** PL residual coefficients, arbitrary-order recursion, WS-GGI and
  fixed-regressor Gauss--Newton proofs, full-rank and rank-deficient least-squares
  relations, second-order specializations, and the effects of regularization.
- **I:** row pseudoinverse and Tikhonov derivations, loss-root coefficients and
  corrections, arbitrary-order PL scalar residuals, the Polyak proof, and
  attainable-target conditions.
- **J:** the FSM signal/innovation definitions, exact gradient/Hessian structure,
  gradient- and loss-root residual coefficients, the common update rules, and
  automatic differentiation at higher orders.

Existing FSM, innovation-energy, and general HODE appendices are referenced
rather than duplicated. All active `\label` names from the supplied Section 4
are retained, with no duplicate labels. Labels on material moved to an appendix
now resolve there. The old PL/FSM subsection labels both identify the common
criteria subsection; the old update-summary label identifies the gradient-root
subsection containing the compact table. Check the wording of any external
sentence that refers specifically to the former subsection titles.

## Mathematical and attribution qualifications

- The common gradient-root formulas through order two and loss-root formulas
  through order three apply to either smooth loss. Higher nonlinear derivatives
  first enter at gradient-root order three and loss-root order four.
- The full-Hessian order-two reduction follows from
  `Q^dagger Q Q^dagger = Q^dagger`. It applies to both parametrizations with
  that inverse, but generally fails for a shifted or otherwise regularized
  inverse. The appendix states the extra term for a shifted inverse.
- For a frozen quadratic loss, the arbitrary-order gradient-root construction
  is equivalent to repeated preconditioned gradient steps with the regressor
  and solution operator unchanged. No separate novelty or acceleration claim
  is made for those algebraically identical updates.
- The least-squares result is a batch, fixed-regressor identity. A rank-deficient
  pseudoinverse correction retains the current null-space component. A damped
  Newton correction and ridge least squares are also distinct regularized rules.
- The WS-GGI statement retains the supplied citation to Lv et al. and specifies
  matching regressors and scalar step normalization. WS-GNI is described as the
  local fixed-regressor Gauss--Newton comparator. The Wang et al. (2020) paper
  concerns a broader adaptive filtering/recursive identification scheme, so it
  is retained as related background, not as an assertion that the entire
  algorithm is identical to the batch update. Verify the intended WS-GNI
  primary citation before claiming a stronger literature equivalence.
- A nonnegative target does not by itself guarantee a loss root. The appendix
  gives exact attainability conditions for unconstrained PL least squares and
  separates target attainment, stationarity, and parameter accuracy.
- Classical Polyak equivalence uses the optimal loss value. An estimated target
  or lower bound produces the corresponding substituted formula without
  automatically inheriting the classical convergence theorem.
- The manuscript convention `J = (1/2) sum r_t^2` is preserved. The current
  Python comparison code instead uses `mean(r_t^2)`. Reconciling manuscript and
  implementation normalizations belongs in the experimental description:
  gradient step sizes, regularization, and loss targets depend on this choice.

The mathematical derivations are written for a fixed linear solution operator.
Sparse or constrained maps can be nonlinear in their right-hand side, so the
text explicitly excludes using linearity identities for them without further
justification. No optimizer or experiment code is changed by this revision.
