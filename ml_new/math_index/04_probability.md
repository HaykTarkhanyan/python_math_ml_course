# Probability — Math Reference

## Homework / Quarto modules (`math/`)

- [16_probability_intro.qmd](../../math/16_probability_intro.qmd) — **Probability basics** — Sample space, events, axioms, conditional probability, independence, Bayes' theorem, total probability.
- [17_probability_exp_var_inequalities.qmd](../../math/17_probability_exp_var_inequalities.qmd) — **Expectation, variance, inequalities** — Expectation linearity, variance, Markov / Chebyshev / Cauchy-Schwarz inequalities, Jensen for E.
- [18_probability_corr_cov.qmd](../../math/18_probability_corr_cov.qmd) — **Covariance & correlation** — Covariance matrix, Pearson correlation, properties, multivariate normal hint.
- [19_probability_distributions.qmd](../../math/19_probability_distributions.qmd) — **Distributions** — Bernoulli, Binomial, Geometric, Poisson, Uniform, Normal, Exponential, Beta, Gamma; pmf/pdf/cdf; moment generating function (intro).
- [20_probability_convergence_modes_lln_clt.qmd](../../math/20_probability_convergence_modes_lln_clt.qmd) — **Convergence, LLN, CLT** — Convergence in probability vs distribution vs almost sure; weak and strong LLN; central limit theorem (statement + applications).

## Lecture PDFs (`math/Lectures/`)

- [L09_Probability__Independence__Bayes_Rule.pdf](../../math/Lectures/L09_Probability__Independence__Bayes_Rule.pdf) — Probability, independence, Bayes (module 16).
- [L10_Random_Variables.pdf](../../math/Lectures/L10_Random_Variables.pdf) — Random variables (modules 16-17 bridge).
- [L11_Expected_Value__Variance.pdf](../../math/Lectures/L11_Expected_Value__Variance.pdf) — Expectation + variance (module 17).
- [L12_Covariance__Correlation.pdf](../../math/Lectures/L12_Covariance__Correlation.pdf) — Cov + correlation (module 18).
- [L13_Distributions.pdf](../../math/Lectures/L13_Distributions.pdf) — Distributions (module 19).
- [L14_Convergence__LLN__CLT.pdf](../../math/Lectures/L14_Convergence__LLN__CLT.pdf) — Convergence + LLN + CLT (module 20).

## Homework write-ups (`math/Homeworks/`)

- `hw_08_prob_1.pdf` — basics
- `hw_09_prob_2_rvs.pdf` — random variables
- `hw_10_prob_3_exp_var_inequalities.pdf`
- `hw_11_prob_4_corr_cov.pdf`
- `hw_12_prob_5_distribs.pdf`
- `hw_13_prob_6_convergence_lln_clt.pdf`

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L01 i.i.d. assumption (when adding the missing frame I flagged) | module 16 + 20 |
| L01 / L02 P(x, y) joint distribution language | module 16 |
| L02 noise model `y = f(x) + epsilon, epsilon ~ N(0, sigma^2)` | module 19 (Normal) |
| L02 squared error ↔ Gaussian MLE | modules 19 + 23 (stat MLE) |
| Logistic regression (Bernoulli loss) | module 19 |
| Naive Bayes | modules 16 (Bayes), 19 (distributions) |
| Decision trees (entropy / Gini) | module 28 (info theory) but probability prereq from module 16 |
| Bias-variance decomposition | module 17 (expectation, variance) |
| Sampling / Monte Carlo | module 20 (LLN) + stat L7 (sampling distributions) |
| Confidence intervals for metrics | module 20 (CLT) + stat L8 |
| Bayesian neural networks / VI | modules 16, 19 (priors) + stat L6 (MAP, conjugate) |
| Diffusion models (forward noising process) | module 19 (Normal) + module 20 (limits) |
| LLM token sampling (softmax → categorical) | module 19 |

## Notes

- Module 19 distributions list does NOT include Dirichlet, Categorical, Multinomial, Beta-Binomial — these are still flagged as missing in the ML checklist's "Probability foundations" tier. When ML topics need them, currently must teach inline.
- Module 16 is heavier on combinatorics than typical probability courses; the qmd reflects that mix.
- L09-L14 PDFs are single-artefact delivery decks (no tex source).
