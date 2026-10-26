# Simplified bias appendix

Insert appendix_bias_analysis.tex after the existing appendices. It refers to
the supplied innovation-energy appendix through app:fsm_innovation_energy.
Add the three entries in bias_analysis_references.bib to the paper bibliography.

The appendix now contains only the general population-objective identity,
a short explanation of possible bias from reconstructed regressors, and a
visible placeholder plus comments for the running record-length study.
The second-order expansion and benchmark-specific derivations are removed.

## Supporting references and their scope

- Soderstrom (2007), Automatica 43(6), 939-958,
  doi:10.1016/j.automatica.2006.11.025.
  Author manuscript: https://user.it.uu.se/~ts/eivplenaut02.pdf
  Sections 9-10, especially equations (66)-(70), show the orthogonality issue
  and how regressor contamination shifts the least-squares solution.
- Ni, Gilson and Garnier (2013), IET Control Theory & Applications 7(9),
  1276-1286, doi:10.1049/iet-cta.2012.0548.
  Official full text:
  https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-cta.2012.0548
  Section 3.1, equations (36)-(45) and Lemma 1, connects instrumental-variable
  orthogonality with iterative reconstruction of unknown internal signals.
- Janczak and Korbicz (2019), International Journal of Applied Mathematics
  and Computer Science 29(3), 571-580, doi:10.2478/amcs-2019-0042.
  University-hosted paper:
  https://zbc.uz.zgora.pl/Content/86007/AMCS_2019_29_3_12.pdf
  Section 3.2, equations (31)-(33), gives an explicit Wiener-regression
  inconsistency mechanism. Its noise acts before the nonlinearity, so the
  appendix explicitly identifies it as an analogy, not a proof for the
  present output-AR model.

Joint estimation is not described as sufficient to establish bias.
The effective-regression-error identity is an application to this paper's
PL representation. No source is presented as proving that the implemented
PL or FSM estimator must be biased.

The standalone preview uses D for the external innovation-energy appendix,
as in the existing Section 4 preview. The full manuscript resolves the
actual appendix label normally.
