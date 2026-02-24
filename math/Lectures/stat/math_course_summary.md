# Math Folder — Comprehensive Content Summary

Generated 2026-02-24 by studying all `.qmd` modules, `.tex` lectures, homeworks, and plan files.

---

## 1. QMD Modules Overview (00–25)

### Linear Algebra (Modules 00–03) — COMPLETE

| Module | Topic | Problems | Key ML Connections |
|--------|-------|----------|--------------------|
| 00 | Sets, combinatorics, functions | ~14 | Data splitting, hyperparameter grids, ReLU/sigmoid |
| 01 | Vectors, norms, dot product, cosine similarity | ~13 | Feature normalization, L1/L2 regularization, kNN, word embeddings |
| 02 | Matrices, transformations, determinants | ~6 | Geometric transformations, normal equation for regression |
| 03 | Linear systems, independence, eigenvalues, diagonalization | ~15 | GPS, regression, multicollinearity, PCA, tensors |

### Calculus (Modules 04–07) — PARTIAL

| Module | Topic | Status | Notes |
|--------|-------|--------|-------|
| 04 | Limits, continuity, derivatives | Partial | References external textbook PDF |
| 05 | Extrema, convexity, Taylor series | Partial | ~3 worked examples, optimization box problem |
| 06 | Integrals | Partial | References external material |
| 07 | Multivariate calculus, gradient descent | Partial | Gradient descent, Hessians, contour plots, boat-in-Sevan problem |

### Optimization (Modules 08–15) — MOSTLY TODO

| Module | Topic | Status |
|--------|-------|--------|
| 08 | Golden Section Search, Brent's method | **Complete** — implementation notebooks, homework |
| 09 | Prerequisite & Gradient Descent | Skeleton |
| 10 | Momentum & First-Order Algorithms | Skeleton |
| 11 | Second-Order Optimization | Skeleton |
| 12 | Derivative-Free Methods | Skeleton |
| 13 | Evolutionary Algorithms | Skeleton |
| 14 | Bayesian Optimization | Skeleton |
| 15 | Multi-criteria Optimization | Skeleton |

### Probability (Modules 16–20) — COMPLETE

| Module | Topic | Problems | Key Content |
|--------|-------|----------|-------------|
| 16 | Probability basics, Bayes, Monty Hall | ~25 | Discrete/geometric probability, birthday paradox, urns, Bayes applications |
| 17 | Expectation, variance, inequalities | ~12 | LOTUS, optimal stopping, St. Petersburg paradox, coupon collector |
| 18 | Covariance, correlation | 7 | Pearson vs Spearman, outlier effects, portfolio diversification |
| 19 | Distributions (discrete + continuous) | 14 | Distribution identification, mystery distributions, interactive tools |
| 20 | Convergence, LLN, CLT | — | Convergence modes, vanishing spike, Borel-Cantelli |

### Statistics (Modules 21–25) — PARTIAL/SKELETON

| Module | Topic | Status |
|--------|-------|--------|
| 21 | Foundations & Descriptive Statistics | Partial — data viz, EDA, survivorship bias, Simpson's paradox |
| 22 | Properties of Estimators | Partial — Poisson exp family, sufficiency, bias/variance, Fisher info, CR |
| 23 | MLE and MAP | Skeleton |
| 24 | Confidence Intervals & Asymptotics | Skeleton |
| 25 | Hypothesis Testing | Skeleton |

---

## 2. Beamer Lecture Presentations

### Statistics Lecture Series (`math/Lectures/stat/`)

| File | Session | Frames | Topic |
|------|---------|--------|-------|
| `01_stat.tex` | 1 | 24 | Foundations: prob vs stat, population/sample, i.i.d., plug-in, loss/risk/ERM |
| `02_stat.tex` | 2 | 23 | Descriptive: center, spread, quantiles, boxplots, skewness, kurtosis, ECDF, Anscombe, Simpson |
| `03_stat.tex` | 3 | 34 | Estimator properties: bias, variance, MSE, consistency, sufficiency, exp family, Lehmann-Scheffe |
| `04_stat.tex` | 4 | 26 | Fisher information, Cramer-Rao bound, efficiency, admissibility, Stein's paradox, minimax |
| `05_stat.tex` | 5 | 35 | Point estimation: MoM, MLE (Bernoulli/Normal/Poisson/Exp), invariance, cross-entropy, MLE failures, numerical MLE |

**Next up (planned, .tex not yet written):**

| Future File | Session | Topic |
|-------------|---------|-------|
| `06_stat.tex` | 6 | MAP, conjugate priors, regularization = MAP, MLE vs MAP vs Bayesian |
| `07_stat.tex` | 7 | EM algorithm, GMMs, K-means as hard EM |
| `08_stat.tex` | 8 | MCMC, Metropolis-Hastings, Gibbs, Bayesian computation |
| `09_stat.tex` | 9 | Sampling distributions, delta method, confidence intervals |
| `10_stat.tex` | 10 | Bootstrap, permutation tests |
| `11_stat.tex` | 11 | Hypothesis testing, power analysis, multiple testing |
| `12_stat.tex` | 12 | LRT, t-tests, chi-squared, nonparametric tests |
| `13_stat.tex` | 13 | Regression inference (OLS + logistic) |
| `14_stat.tex` | 14 | GLMs: exp family + link functions, Poisson regression |
| `15_stat.tex` | 15 | ANOVA, AIC/BIC, model comparison |
| `16_stat.tex` | 16 | Causal inference, DAGs, potential outcomes |

### Optimization Lecture Series (`math/Lectures/optim/`)

| File | Topic | Status |
|------|-------|--------|
| `01_univariate.tex` | Convex sets/functions, golden section, Brent's | Complete |
| `02_prereqs.tex` | Prerequisites for optimization | Exists |
| `03_gd_step_size.tex` | Gradient descent, step size | Exists |
| `04_momentum_adam.tex` | Momentum, Adam optimizer | Exists |
| `05_derivative_free.tex` | Derivative-free methods | Exists |
| `06_evolutionary.tex` | Evolutionary algorithms | Exists |

---

## 3. Homeworks (PDF + Xournal++ sources)

| File | Covers Module(s) |
|------|-----------------|
| `hw_00_sets_comb_funcs.pdf` | 00 — Sets, combinatorics, functions |
| `hw_01_vectors.pdf` | 01 — Vectors |
| `hw_02_matrices_det_inverse.pdf` | 02 — Matrices |
| `hw_03_calc_1.pdf` | 04 — Limits, derivatives |
| `hw_05_calc_2_extrema_convexity_taylor.pdf` | 05 — Extrema, convexity |
| `hw_06_integrals.pdf` | 06 — Integrals |
| `hw_07_multivar_calc.pdf` | 07 — Multivariate calculus |
| `hw_08_optim_univar.pdf` | 08 — Univariate optimization |
| `hw_18_prob_1.pdf` | 16 — Probability basics |
| `hw_19_prob_2_rvs.pdf` | 17 — Random variables |
| `hw_20_prob_3_exp_var_inequalities.pdf` | 17 — Expectation, variance |
| `hw_21_prob_4_corr_cov.pdf` | 18 — Covariance, correlation |
| `hw_22_prob_5_distribs.pdf` | 19 — Distributions |
| `hw_23_prob_6_convergence_lln_clt.pdf` | 20 — Convergence, LLN, CLT |

---

## 4. Key Course Design Principles

- **Bilingual**: Armenian + English throughout
- **Problem-centric**: Real-world scenarios with multi-part questions, difficulty ratings (cheese emojis)
- **ML connections woven naturally**: Each math topic ties to ML (not forced)
- **Intuition-first**: No heavy proofs; computation and visualization over formalism
- **Diverse examples**: Medicine, finance, science, engineering, polling — not ML-only
- **Visual-heavy presentations**: All diagrams are TikZ/pgfplots (no external images)
- **Progressive prerequisite chain**: LA → Calculus → Optimization → Probability → Statistics

---

## 5. Prerequisite Flow

```
Modules 00-03 (Linear Algebra)
    ├─→ Module 03 regression → Stat L13 (regression inference)
    ├─→ Module 01 norms → Stat L6 (regularization = MAP)
    └─→ Module 02 transformations → PCA concepts

Modules 04-07 (Calculus)
    ├─→ Module 05 convexity → Stat L1 (loss landscapes, ERM)
    ├─→ Module 07 Hessian → Stat L4 (Fisher information)
    └─→ Module 07 Taylor → Stat L9 (delta method)

Module 08 (Optimization)
    └─→ Stat L5 (numerical MLE: gradient ascent, Newton-Raphson)

Modules 16-20 (Probability)
    ├─→ Module 16 Bayes → Stat L6 (MAP, priors)
    ├─→ Module 17 E[X], Var → Stat L3-L4 (bias, variance, MSE, CR)
    ├─→ Module 19 distributions → Stat L5 (MLE for specific distributions)
    └─→ Module 20 CLT → Stat L9 (sampling distributions, CIs)
```

---

## 6. Current Status at a Glance

| Area | Completion |
|------|-----------|
| Linear Algebra modules (00-03) | 100% |
| Calculus modules (04-07) | ~40% |
| Optimization modules (08-15) | ~15% (only 08 complete) |
| Probability modules (16-20) | 100% |
| Statistics modules (21-25) | ~20% |
| Stat lectures written (.tex) | 5 of 16 planned (31%) |
| Homeworks | 14 PDFs covering modules 00-08, 16-20 |
