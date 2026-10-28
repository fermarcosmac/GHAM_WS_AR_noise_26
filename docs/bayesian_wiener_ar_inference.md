# Bayesian inference for the normalized parametric Wiener-AR model

Research and mathematical design, screened on 28 October 2026. This note defines
an inference problem and a possible implementation; it does not implement a new
estimator or claim that the proposed adaptation is an existing published method.

## Literature findings

Liu et al. (2024) is the closest recent equation-level match verified in this
search. The table records the remaining differences. No inspected source was
verified to combine direct inference
for all original rational coefficients `a,b`, polynomial coefficients `f`, and
unknown additive output-AR coefficients `d`, with optional sparsity. This is a
bounded search finding, not a claim that no such paper exists.

| Reference | Verified formulation and relevance | Difference from the requested model |
| --- | --- | --- |
| Q. Liu, J.-H. Li, H. Wang, J.-X. Zeng and Y. Chai, **Stochastic Variational Bayesian Learning of Wiener Model in the Presence of Uncertainty**, *Acta Automatica Sinica* 50(6), 1185-1198, 2024. [Publisher/full text](https://www.aas.net.cn/article/doi/10.16383/j.aas.c210925), DOI `10.16383/j.aas.c210925`. | Equations (2)-(5) introduce a rational block, replace it by FIR coefficients, and use a finite nonlinear basis with unknown coefficients. | Gaussian internal noise, Student-t measurement noise; not a learned output-AR model. The shared precision prior is not coefficient-specific ARD. |
| J. Li, F. Zhang, C. Wang and Y. Liu, **Identification of Wiener-Hammerstein Model Using Stochastic Variational Bayesian Learning**, ICIRA 2024 proceedings, published online 22 January 2025, pp. 147-158. [Publisher](https://link.springer.com/chapter/10.1007/978-981-96-0783-9_11), DOI `10.1007/978-981-96-0783-9_11`. | Recent SVI for block-oriented models with process and measurement noise. | Publisher abstract inspected; detailed polynomial/rational parametrization and AR-noise treatment not verified from full text. A related lead, not a confirmed exact match. |
| F. Li, Y. Zhu, S. Qian and Y. Gu, **Parameters Identification of Wiener Nonlinear Systems with Gaussian Mixed Measurement Noises**, *Circuits, Systems, and Signal Processing* 45, 953-976, 2026; online 26 August 2025. [Publisher](https://link.springer.com/article/10.1007/s00034-025-03294-4), DOI `10.1007/s00034-025-03294-4`. | Parametric ARMA dynamic block and neuro-fuzzy static block. | Linear parameters are estimated by correlation analysis; VB updates nonlinear weights and noise precision. Not joint Bayesian inference of a rational/polynomial/AR-error model. Publisher abstract inspected. |
| R. S. Risuleo, F. Lindsten and H. Hjalmarsson, **Semi-Parametric Kernel-Based Identification of Wiener Systems**, CDC 2018, pp. 3874-3879. [Author manuscript](https://www2.it.uu.se/katalog/freli660/publications/RisuleoLH_2018.pdf), DOI `10.1109/CDC.2018.8619482`. | Bayesian impulse-response model and a finite nonlinear basis; empirical Bayes with sampling and EM. | Nonlinearity coefficients are estimated, but the linear block uses a GP impulse-response representation rather than direct original `a,b` inference. |
| A. L. Cedeno, R. A. Gonzalez, R. Carvajal and J. C. Aguero, **Identification of Wiener state-space models utilizing Gaussian sum smoothing**, *Automatica* 166, 111707, 2024. [Publisher](https://www.sciencedirect.com/science/article/pii/S0005109824002012), DOI `10.1016/j.automatica.2024.111707`. | Joint estimation of linear state-space parameters and piecewise-polynomial nonlinearity coefficients. | Maximum likelihood/EM, not Bayesian parameter inference. Useful recent parametric comparison; posterior state smoothing alone does not make parameter estimation Bayesian. |
| S. Vakili, M. Mazo Jr. and P. Mohajerin Esfahani, **Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model**, arXiv preprint first submitted in 2025. [Primary manuscript](https://arxiv.org/abs/2504.05490). | Bayesian learning of coefficients in a nonlinear observation function. | Linear dynamics are assumed known, so it does not solve the joint problem here. |
| R. Manavalan and F. Tronarp, **Barron-Wiener-Laguerre models**, arXiv preprint, February 2026. [Primary manuscript](https://arxiv.org/abs/2602.13098). | Recent probabilistic Wiener-Laguerre modeling with Bayesian nonlinear-map inference. | Laguerre features and Barron-type nonlinear approximators; not the manuscript's original rational coefficients, polynomial coefficient vector and learned output-AR model. |

Two useful foundations are [Lindsten, Schon and Jordan (2013)](https://doi.org/10.1016/j.automatica.2013.03.021)
for Bayesian Wiener inference, and [Tipping (2001)](https://www.jmlr.org/papers/v1/tipping01a.html)
for coefficient-specific relevance learning. Ordinary Bayesian regularization,
continuous sparse shrinkage, empirical-Bayes pruning, and discrete model
selection should be distinguished when describing the eventual baseline.

## Match the manuscript before changing the inference method

The model is the one in Section 2.1, equations (1)-(6), of
`C:/Users/ferma/Downloads/GHAM_JFI_paper_REBUTTAL.pdf`:

\[
 A(q^{-1};a)=1+\sum_{i=1}^{n_a}a_iq^{-i},\qquad
 B(q^{-1};b)=\sum_{i=1}^{n_b}b_iq^{-i},\qquad
 D(q^{-1};d)=1+\sum_{i=1}^{n_d}d_iq^{-i},
\]
\[
 \alpha_t=\frac{B(q^{-1};b)}{A(q^{-1};a)}u_t,
 \qquad
 \beta_t=\alpha_t+\sum_{j=2}^{n_f}f_j\alpha_t^j,
 \qquad
 c_t=\beta_t+e_t,
 \qquad
 D(q^{-1};d)e_t=\nu_t.
\]

Keep `f1 = 1`, no constant nonlinear term, the input delay in `B`, and the
parameter order

\[
 \theta=(a_1,\ldots,a_{n_a},b_1,\ldots,b_{n_b},
         f_2,\ldots,f_{n_f},d_1,\ldots,d_{n_d})^\top.
\]

The normalization assumes a nonzero true linear polynomial coefficient. Known
orders, suitable excitation and a nonredundant transfer-function representation
are still needed for identification; normalization alone is not a proof of
identifiability.

For the likelihood below, explicitly assume independent Gaussian innovations

\[
 \nu_t\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma_\nu^2),
 \qquad \nu\ \text{independent of }u.
\]

Whiteness alone would not imply this likelihood. The Gaussian innovation
assumption matches the Gaussian synthetic experiments, and leaves the output
disturbance `e` temporally correlated.

### The state-space output matrix is not generally the identity

In Lindsten's model, `x_t` is an `n_x`-dimensional state while the input of the
static nonlinearity is scalar. Therefore `C` is a row vector. Their paper already
fixes `C = [1,0,...,0]` in an observable realization. A scalar rational IIR block
does not require `C = I`; that choice produces a vector output when `n_x > 1`.
For `n_x = 1`, `C = 1` is the special scalar case. Using `B/A` directly avoids
state-space realization ambiguity.

Also remove the independent process noise in the *signal* block when matching
this manuscript. Its signal dynamics are deterministic given parameters, input
and initial state. The stochastic dynamics are in the separate output-error AR
filter. Keeping Lindsten's positive process-noise covariance in the signal block
would define a different data-generating model.

## Conditional innovation likelihood

For any candidate `theta`, simulate the IIR signal recursively:

\[
 \alpha_t=-\sum_{i=1}^{n_a}a_i\alpha_{t-i}
          +\sum_{i=1}^{n_b}b_i u_{t-i}.
\]

Compute

\[
 e_t(\theta)=c_t-\beta_t(\theta),\qquad
 r_t(\theta)=e_t(\theta)+\sum_{i=1}^{n_d}d_i e_{t-i}(\theta).
\]

Both the nonlinear response and AR filtering depend on the candidate
parameters. Filtering must be applied after forming the complete output
residual; it cannot be moved through the static nonlinearity.

For a conditional-likelihood implementation, treat the first `n_d` observed
outputs as boundary data and retain indices `t = t0,...,N`, with
`t0 >= n_d + 1`. The signal prehistory/initial state is either known or included
among the unknowns. Let `M = N-t0+1`. The conditional likelihood is

\[
 L(\theta,\sigma_\nu^2)
 =(2\pi\sigma_\nu^2)^{-M/2}
 \exp\left[-\frac{1}{2\sigma_\nu^2}
                 \sum_{t=t_0}^{N}r_t(\theta)^2\right].
\]

There is no extra parameter-dependent whitening determinant in this conditional
likelihood: the monic AR map from retained observations to innovations is
triangular with unit diagonal. Discarding the density of the boundary data makes
this a conditional-likelihood analysis, not the exact full-record stationary
posterior.

For a stationary full-record alternative, the complete noise vector satisfies

\[
 e\mid d,\sigma_\nu^2\sim\mathcal N(0,\sigma_\nu^2 K_d),
\]

where `K_d` is the stationary covariance for unit innovation variance. Then

\[
 -\log L=\frac{N}{2}\log(2\pi\sigma_\nu^2)
       +\frac12\log|K_d|
       +\frac{e^\top K_d^{-1}e}{2\sigma_\nu^2}.
\]

Its initial-noise density/covariance term depends on `d` and cannot be dropped.
The simple Gaussian conditional for `d` below applies to the conditional
likelihood, not unchanged to this stationary alternative.

The initial signal state can be included with a proper prior when pre-record
inputs are unavailable. A warm-up exclusion reduces transients but does not
make unknown initial conditions known.

## Priors and posterior

For a first fixed-order Bayesian baseline, use proper, weakly informative
Gaussian priors on `b` and the free polynomial coefficients, with priors on `a`
and `d` supported on stable monic denominators. Choose scales from declared
training-data scaling or scientific prior knowledge, not the unknown truth.

For optional sparse shrinkage on the nonlinear coefficients, use

\[
 f_j\mid\lambda_j\sim\mathcal N(0,\lambda_j^{-1}),\qquad
 \lambda_j\sim\operatorname{Gamma}(h_0,k_0),\quad j=2,\ldots,n_f,
\]
\[
 \sigma_\nu^2\sim\operatorname{IG}(s_0,r_0),
 \qquad h_0,k_0,s_0,r_0>0.
\]

Here Gamma uses shape/rate, and the inverse-Gamma density is proportional to
`s^(-s0-1) exp(-r0/s)` for variance `s > 0`. The priors on `f` are independent
of the innovation variance in this specification. Consequently the variance
update below has no extra prior-parameter-count contribution.

Large `lambda_j` shrinks `f_j` toward zero. This continuous hierarchy does not
give positive posterior probability to an exactly zero coefficient. Exact term
inclusion needs a spike-and-slab prior or a declared selection rule. Empirical
Bayes/ARD pruning is another option, but maximization over hyperparameters is
different from integrating their uncertainty. Do not shrink the fixed `f1 = 1`.

With `lambda = (lambda_2,...,lambda_nf)`, the conditional-likelihood posterior is

\[
 p(\theta,\sigma_\nu^2,\lambda\mid c,u)
 \propto L(\theta,\sigma_\nu^2)
 p(a)p(b)p(d)p(\sigma_\nu^2)
 \prod_{j=2}^{n_f}p(f_j\mid\lambda_j)p(\lambda_j).
\]

If the signal initial state is inferred, append it to the unknowns and multiply
this expression by its declared proper prior.

This is an adaptation of standard Bayesian regression with autoregressive
errors to the normalized parametric Wiener structure. It is not asserted to be
the likelihood or algorithm of any one of the literature entries above.

## Conditional inference blocks

Write `f = (f2,...,f_nf)` and `Lambda = diag(lambda_2,...,lambda_nf)`.
All filtered matrices below retain only rows `t0,...,N`, after the required
histories have been formed.

### Polynomial coefficients: Gaussian conditional

Given `a,b,d`, construct the power matrix with rows

\[
 V_t=(\alpha_t^2,\ldots,\alpha_t^{n_f}),
 \qquad z=D_d(c-\alpha),\qquad X=D_d V.
\]

The term `c-alpha` preserves the fixed first-order coefficient. The AR filter is
applied to every column of `V`, including the available pre-retention histories.
The regression is exactly

\[
 z=Xf+\nu.
\]

Thus

\[
 f\mid\text{rest}\sim\mathcal N(m_f,S_f),\qquad
 S_f=(\Lambda+X^\top X/\sigma_\nu^2)^{-1},\qquad
 m_f=S_fX^\top z/\sigma_\nu^2.
\]

These are posterior distributions of the actual polynomial coefficients, not
unconstrained NARX monomial products. Their dependence on uncertain `a,b,d`
is propagated by drawing them conditionally throughout the sampler.

### AR coefficients: Gaussian conditional restricted to stability

Given the current signal parameters, form

\[
 E_t=(e_{t-1},\ldots,e_{t-n_d}),\qquad
 e_t=-E_t d+\nu_t.
\]

For a prior `N(m_d0,V_d0)` restricted to stable denominators, the unrestricted
Gaussian update is

\[
 S_d=(V_{d0}^{-1}+E^\top E/\sigma_\nu^2)^{-1},\qquad
 m_d=S_d(V_{d0}^{-1}m_{d0}-E^\top e/\sigma_\nu^2).
\]

Sample this Gaussian conditional restricted to stable `D`. For a low-order
noise model, rejection sampling can be practical; otherwise use a valid
Metropolis update. Projection of arbitrary samples onto the stable set is not
a correct posterior-sampling step. A stable reflection-coefficient
parametrization is an alternative; declare the prior in those coordinates or
include the appropriate change-of-variable Jacobian.

### Innovation variance and shrinkage: conjugate conditionals

\[
 \sigma_\nu^2\mid\text{rest}
 \sim\operatorname{IG}\left(s_0+\frac M2,
                  r_0+\frac12\sum_{t=t_0}^{N}r_t^2\right),
\]
\[
 \lambda_j\mid f_j
 \sim\operatorname{Gamma}\left(h_0+\frac12,
                                    k_0+\frac12 f_j^2\right).
\]

For a non-sparse baseline, fix the polynomial prior precisions and omit their
hyperparameter sampling.

### Original IIR coefficients: nonlinear conditional

The conditional density of `a,b` is proportional to the innovation likelihood
times `p(a)p(b)`. It is generally non-Gaussian because the recursive signal is
fed through the polynomial. Use a valid Metropolis or Hamiltonian update, with
stability enforced through the prior support or a declared transformation.
Any unknown signal initial state can be updated in this block.

Useful exact-recursion sensitivity equations, for fixed initial conditions,
are

\[
 \frac{\partial\alpha_t}{\partial a_j}
 =-\alpha_{t-j}-\sum_i a_i
            \frac{\partial\alpha_{t-i}}{\partial a_j},\qquad
 \frac{\partial\alpha_t}{\partial b_j}
 =u_{t-j}-\sum_i a_i
            \frac{\partial\alpha_{t-i}}{\partial b_j}.
\]

For a signal-block parameter `w` in `a,b`,

\[
 \frac{\partial r_t}{\partial w}
 =-D_d\left[F'(\alpha_t)\frac{\partial\alpha_t}{\partial w}\right],
 \quad F'(\alpha)=1+\sum_{j=2}^{n_f}j f_j\alpha^{j-1},
\]
\[
 \frac{\partial r_t}{\partial f_j}=-D_d[\alpha_t^j],\qquad
 \frac{\partial r_t}{\partial d_j}=e_{t-j}.
\]

Automatic differentiation can compute the same derivatives. They refer to the
exact recursive IIR model; FSM derivatives refer to the chosen finite-frequency
approximation instead.

## Recommended first algorithm

Use Metropolis-within-Gibbs with multiple initialized chains:

1. Update `a,b` and any unknown signal initial state using a valid nonlinear
   posterior transition; recompute `alpha`.
2. Draw `f` from its Gaussian conditional.
3. Recompute `beta,e`; draw `d` from its stability-restricted conditional.
4. Recompute innovations; draw the innovation variance.
5. Draw the optional coefficient-specific shrinkage precisions.
6. Repeat, assess chain convergence and effective sample sizes, then retain
   posterior samples for parameter and prediction summaries.

The exact conditional-likelihood target is preserved by valid conditional
updates, subject to the usual MCMC convergence requirements. No separate
particle trajectory sampler is required for this deterministic signal path.
The unknown internal waveform is recomputed from each parameter draw.

In particular, all entries of the manuscript's original `theta` participate in
the inference. The Gaussian conditional substeps do not make the full posterior
Gaussian or remove its nonlinear parameter dependence.

## Optional collapsed inference

The small polynomial block can also be integrated out analytically. For fixed
`a,b,d,lambda,sigma2`, define `z,X` as above and

\[
 S=\sigma_\nu^2 I+X\Lambda^{-1}X^\top.
\]

Then the marginal conditional likelihood is

\[
 z\mid a,b,d,\lambda,\sigma_\nu^2\sim\mathcal N(0,S),
 \quad
 -\log L_{\rm marginal}
 =\frac M2\log(2\pi)+\frac12\log|S|+\frac12z^\top S^{-1}z.
\]

Remember that both `z` and `X` depend on `d`, and `X` depends on `a,b`.
Sampling the remaining unknowns and subsequently drawing `f` from its
conditional yields joint posterior draws. Maximizing this objective over
hyperparameters instead gives an empirical-Bayes alternative.

Use the matrix determinant lemma and Woodbury identity to work in the
`(nf-1)`-dimensional coefficient space rather than constructing a dense
`M`-by-`M` matrix. In particular,

\[
 \log|S|=M\log\sigma_\nu^2-\log|\Lambda|
       +\log|\Lambda+X^\top X/\sigma_\nu^2|.
\]

## Comparison and implementation boundaries

- Start with the same declared `na,nb,nf,nd` and normalization as the existing
  methods. Use proper regularizing priors first. A larger nonlinear dictionary
  with sparse shrinkage is a separate model-selection experiment.
- For physical-parameter inference, use the exact recursive `B/A` signal and
  monic AR innovation filter. If comparing only optimization/inference on the
  existing FSM approximation, keep its frequency grid and truncation fixed for
  every method and identify the posterior as belonging to that approximate
  forward model.
- The existing `prediction_error_innovations` in
  `python/wiener_system.py` already forms the candidate AR-filtered output
  residual, but its signal response is FSM based. Priors, variance inference,
  posterior sampling and convergence diagnostics would be additional work.
- Report posterior means/MAP estimates explicitly and propagate draws through
  the forward model for posterior predictive summaries. Evaluating the model
  at posterior mean parameters is generally different from averaging the
  model output over parameter draws.
- Use matching one-step or free-run prediction tasks. A one-step Wiener-AR
  predictor is `beta_t - sum_i d_i (c_{t-i}-beta_{t-i})`.
- Assess parameter error, residual whiteness, uncertainty calibration and wall
  time separately. Bayesian priors and sparse shrinkage can introduce
  finite-sample bias; posterior inference does not imply unbiased estimates.
- Validate implementation against the analytic polynomial and AR conditional
  updates, finite-difference gradients, and a small synthetic record before
  launching the existing Monte Carlo studies.

## Algebra verification performed for this design

A synthetic two-pole, two-numerator-coefficient, cubic-polynomial, AR(2) record
was used to verify the innovation/design identity, the exact-recursion
sensitivities, the unrestricted Gaussian conditional means, and the collapsed
determinant formula. Maximum central-finite-difference sensitivity discrepancy
was `2.34e-10`; conditional-mean stationarity discrepancies were below
`3.5e-14`; the two determinant calculations differed by `1.14e-13`. These checks
validate the displayed algebra, not MCMC convergence or estimator performance.
