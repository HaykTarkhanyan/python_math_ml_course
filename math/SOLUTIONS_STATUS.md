# Math homework solutions — status tracker

Last updated: 2026-04-30

Tracks every `math/*.qmd` file (excluding optim 08–15 per user instruction):
- whether solutions exist inline (in the qmd as collapsible callouts)
- whether they exist elsewhere (separate `.tex`/`.ipynb` solution package)
- whether nothing is solved yet
- pedagogical review status (by Claude / by user)

## Legend

- **inline** = at least some `::: {.callout-tip collapse="true" title="Solution"}` blocks in the qmd
- **external** = solutions live in `Lectures/stat/hw_*_solutions.{tex,ipynb,pdf}`
- **none** = no solutions anywhere
- **skeleton** = qmd has no problems yet (just materials/links section)
- **partial** = some problems solved, others not

## Table

| File                                          | Problems | Inline solutions | External solutions               | Status         | Reviewed by Claude    | Reviewed by user      |
|-----------------------------------------------|----------|------------------|----------------------------------|----------------|-----------------------|-----------------------|
| 00_sets_comb_funcs.qmd                        | 14       | 14/14            | -                                | inline         | 2026-04-30            | -                     |
| 01_linear_algebra_vectors.qmd                 | 13       | 10/13 (skip 06,08,11) | -                           | inline         | -                     | yes (early session)   |
| 02_linear_algebra_matrices.qmd                | 12       | 12/12            | -                                | inline         | -                     | -                     |
| 03_linear_algebra_concepts.qmd                | 12       | 11/12 (skip 15)       | -                           | inline         | -                     | yes (early session)   |
| 04_calc_lim_continuity_deriv.qmd              | 0 (PDF)  | -                | Armenian textbook (page 76)      | skeleton       | -                     | -                     |
| 05_calc_extrema_convexity_taylor.qmd          | 8        | 8/8              | -                                | inline         | -                     | -                     |
| 06_calc_integrals.qmd                         | 0 (PDF)  | -                | Armenian textbook (page 91)      | skeleton       | -                     | -                     |
| 07_calc_multivar.qmd                          | 6        | 4/6 (skip 01,03)      | -                           | inline         | -                     | -                     |
| 08_optim_univar.qmd                           | -        | -                | -                                | excluded       | -                     | -                     |
| 09_optim_prereq__gradient_descent.qmd         | -        | -                | -                                | excluded       | -                     | -                     |
| 10_optim_momentum_first_order_algs.qmd        | -        | -                | -                                | excluded       | -                     | -                     |
| 11_optim_second_order.qmd                     | -        | -                | -                                | excluded       | -                     | -                     |
| 12_derivative_free.qmd                        | -        | -                | -                                | excluded       | -                     | -                     |
| 13_evolutionary.qmd                           | -        | -                | -                                | excluded       | -                     | -                     |
| 14_bayesian.qmd                               | -        | -                | -                                | excluded       | -                     | -                     |
| 15_multicriteria_optimization.qmd             | -        | -                | -                                | excluded       | -                     | -                     |
| 16_probability_intro.qmd                      | 38       | 34/38 (skip 10 image, 22-24 video bonus) | -          | inline         | 2026-04-30            | -                     |
| 17_probability_exp_var_inequalities.qmd       | 12       | 12/12            | -                                | inline         | 2026-04-30            | -                     |
| 18_probability_corr_cov.qmd                   | 7        | 7/7              | -                                | inline         | -                     | -                     |
| 19_probability_distributions.qmd              | 14       | 14/14            | -                                | inline         | -                     | -                     |
| 20_probability_convergence_modes_lln_clt.qmd  | 4        | 4/4              | -                                | inline         | -                     | -                     |
| 21_stat_fundamentals.qmd                      | 3        | 1/3 (skip 01,02 exploratory)     | -                                | partial        | 2026-05-01            | -                     |
| 22_stat_estimators.qmd                        | 7        | 7/7              | -                                | inline         | 2026-05-01            | -                     |
| 23_stat_mle_map.qmd                           | 11       | 0/11 + 1 worked example | hw_05_06_solutions.{tex,pdf} (Normal-Normal MAP only) | partial | -        | -                     |
| 24_stat_confidence_intervals.qmd              | 5        | 5/5              | hw_07_08_solutions.{tex,ipynb,pdf} | inline       | -                     | -                     |
| 25_stat_hypothesis_testing.qmd                | 5        | 5/5              | hw_09_solutions.{tex,ipynb,pdf}  | inline         | -                     | -                     |
| 26_stat_classical_tests.qmd                   | 6        | 0/6              | -                                | none           | -                     | -                     |
| 27_stat_anova_ab_testing.qmd                  | 4        | 0/4              | -                                | none           | -                     | -                     |
| 28_stat_how_to_lie.qmd                        | 0        | -                | -                                | skeleton       | -                     | -                     |
| 29_info_theory_entropy_kl.qmd                 | 0        | -                | -                                | skeleton       | -                     | -                     |
| 30_info_theory_ml.qmd                         | 0        | -                | -                                | skeleton       | -                     | -                     |
| 31_curse_of_dimensionality.qmd                | 0        | -                | -                                | skeleton       | -                     | -                     |

## Roll-up

- **Inline complete (full coverage):** 00, 02, 05, 17, 18, 19, 20, 22, 24, 25 — **10 files**
- **Inline partial (some skipped):** 01, 03, 07, 16, 21 — **5 files** (skipped problems are image-, video-, or exploration-based)
- **Partial (mix of inline + external + unsolved):** 23 — **1 file**
- **Problems exist but no solutions:** 26, 27 — **2 files**
- **Skeleton (problems live in external Armenian PDF or not yet written):** 04, 06, 28, 29, 30, 31 — **6 files**
- **Excluded per user instruction:** 08–15 (optim) — **8 files**

## Pedagogical review notes (Claude, 2026-04-30)

See full review in conversation. Key forward-concept findings:

- **HW 16 problem 38 (Distinct Stickers Var)** uses **covariance** load-bearing — covariance is HW 18 territory. Same issue in HW 17 problem 10 (literal duplicate of HW 16 #38).
- **HW 16 problems 26–38** (RV/expectation/variance/Markov/Chebyshev) have no supporting lectures in HW 16 itself; they overlap heavily with HW 17 problems 01–10 (8 problems duplicated). Section is labeled "Սա հետոյվա" (this is for later) in HW 16, suggesting user is aware.
- **HW 16 problem 11** (Computing Pi) mentions MCMC, importance sampling, Bayesian inference in flavor text — forward references but not load-bearing.
- **HW 17 problem 12** (Chebyshev commute) mentions Hoeffding/Chernoff bounds — forward but flavor-only.
- **HW 00**: clean, no forward-concept issues. Solutions live entirely within sets / combinatorics / function-property tools.
- All other findings are minor flavor-text asides (ROC curves, Bellman equations, Kelly criterion, etc.) that don't load-bear on the math.
