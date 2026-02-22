# Statistics Lectures — Outline & Plan

## Completed Lectures

### Lecture 1: Foundations (`01_stat.tex`, 24 frames)

Population vs sample, parameter vs statistic, estimand/estimator/estimate, i.i.d. assumption, survivorship bias, plug-in principle, loss functions (L1/L2/L0), risk & ERM, connection to ML.

---

### Lecture 2: Descriptive Statistics (`02_stat.tex`, 23 frames)

Measures of center & spread, robust vs non-robust, histograms & bin width, quantiles & boxplots, skewness & kurtosis, ECDF, choosing the right plot, Anscombe's quartet, Datasaurus Dozen, Simpson's paradox.

**Homework:** (1) Data visualization topic, (2) EDA on real dataset, (3) survivorship bias & Simpson's paradox examples

---

### Lecture 3: Properties of Estimators (`03_stat.tex`, 34 frames)

Bias (X-bar unbiased, dividing by n biased, Bessel's correction) → Variance & MSE (dartboard analogy, MSE = Bias² + Var proof, when biased beats unbiased) → Bias-variance tradeoff (+ ML connection) → Consistency (sufficient conditions) → Sufficiency (Fisher-Neyman factorization, minimal sufficiency, Rao-Blackwell) → Exponential family (completeness, Lehmann-Scheffe).

**Homework:** (1) X-bar unbiased + MSE, (2) sigma-hat biased, (3) shrinkage MSE optimization, (4) Poisson sufficiency via factorization

---

### Lecture 4: Fisher Information & Cramer-Rao (`04_stat.tex`, 26 frames)

Recap of L3 → Why lower variance matters → From data to likelihood (i.i.d. → product → log) → Score function → Fisher information (two forms, coin flip intuition, sharp vs flat) → Cramer-Rao bound → CR efficiency → Regularity conditions (+ when CR fails: Uniform counterexample) → Admissibility (Pareto dominance) → Shrinkage → Stein's paradox (James-Stein formula, why d >= 3) → Minimax → Three philosophies.

**Homework:** (1) Poisson Fisher info + CR + efficiency, (2) Exponential: both CR forms, (3) admissibility/minimax MSE sketch

---

### Lecture 5: Point Estimation — MoM & MLE (`05_stat.tex`, 35 frames)

Recap of L4 → MoM (idea, Normal example, Uniform failure) → MLE idea & recipe → Examples: Bernoulli, **Normal (full derivation → MSE analysis → MLE = least squares)**, Poisson, Exponential → MoM vs MLE → Invariance, identifiability → MLE & sufficiency → MLE in exponential families → Why MLE works (consistency, asymptotic normality, efficiency) → MLE achieves CR bound → **MLE & cross-entropy (logistic regression)** → **When MLE goes wrong** (small n, boundary, Neyman-Scott, overfitting) → **Numerical MLE** (gradient ascent, Newton-Raphson).

**Homework:** (1) Geometric MLE + efficiency, (2) Normal MLE = MoM via exp family, (3) Uniform(0,theta) MLE — not exp family, (4) Poisson MLE simulation

---

## Next Up

### Lecture 6: MAP & Bayesian Estimation (`xx_stat.tex` → `06_stat.tex`, 21 frames)

Recap of L5 → Bayes' theorem for parameters → Prior × likelihood ∝ posterior → Conjugate priors (Beta-Binomial, Normal-Normal) → **Conjugate priors as pseudo-observations** → MAP estimation → When does the prior matter? → Regularization = MAP (Gaussian → Ridge, Laplace → Lasso) → **MLE vs MAP overfitting demo** → MLE vs MAP vs full Bayesian → When to use what.

Builds on: L5 (MLE), Module 16 (Bayes' theorem), Module 01 (norms), Module 05 (regularization).

**Homework:** (1) Normal conjugacy MAP derivation, (2) Beta-Binomial MAP under 3 priors, (3) Ridge closed-form = MAP

---

## Future Lectures

### Lecture 7: The EM Algorithm

*"What if you can't observe everything you need for MLE?"*

Incomplete data & latent variables → If we knew the labels, MLE would be easy → E-step (expected sufficient statistics) → M-step (maximize expected complete-data log-likelihood) → Two-coin example → **Gaussian Mixture Models** (full derivation) → EM as coordinate ascent on ELBO → Convergence (monotonic, local optima) → K-means as hard EM → When to use EM vs numerical MLE (L5).

### Lecture 8: MCMC & Bayesian Computation

*"The posterior exists but we can't write it down. How do we sample from it?"*

Why MAP isn't enough (need full posterior for uncertainty) → Monte Carlo integration → Rejection sampling → **Metropolis-Hastings** algorithm → **Gibbs sampling** → Diagnostics (trace plots, autocorrelation, R-hat, effective sample size) → Practical tools (PyMC, Stan) → Variational inference as alternative (brief).

### Lecture 9: Sampling Distributions & Confidence Intervals

*"A poll reports 52% +/- 3%. Where does the +/- come from?"*

Sampling distributions → Standard error → Asymptotic normality of MLE → **Delta method** (variance of transformed estimators) → Pivotal quantities → Wald confidence intervals → Exact vs asymptotic CIs → **Bayesian credible intervals vs frequentist CIs** (interpretation, coverage, when they agree/disagree).

### Lecture 10: Bootstrap & Permutation Tests

*"One dataset, a complicated statistic, no SE formula. Now what?"*

Bootstrap idea & CIs (percentile, BCa), when bootstrap fails, permutation tests, cross-validation vs bootstrap.

### Lecture 11: Hypothesis Testing

*"A new drug lowers blood pressure by 3 mmHg. Real effect or noise?"*

H0/H1, test statistics, Type I/II errors → p-values (interpretation & misinterpretation) → **Power analysis & sample size planning** (effect size, power curves, designing studies) → Multiple testing (Bonferroni, BH-FDR).

### Lecture 12: Likelihood Ratio Tests & Classical Tests

*"Which test do I actually use?"*

**LRT as unifying framework** (Neyman-Pearson lemma, -2 log Λ ~ χ²) → z-test & t-test (one-sample, paired, Welch's) **as special cases of LRT** → Chi-squared tests (goodness-of-fit, independence) → Nonparametric alternatives (Mann-Whitney, Wilcoxon).

### Lecture 13: Regression Inference

OLS inference → Gauss-Markov theorem → SE / t-tests / p-values for coefficients → Diagnostics (residuals, QQ, leverage, Cook's distance) → R² and adjusted R² → **Logistic regression as MLE** (connects back to L5 cross-entropy slide) → Inference for logistic regression coefficients.

### Lecture 14: Generalized Linear Models

*"OLS and logistic regression look different — but they're the same machine."*

Three components: random (exp family), systematic (linear predictor), link function → Canonical links (identity, logit, log) → OLS as GLM (Normal + identity) → Logistic as GLM (Bernoulli + logit) → **Poisson regression** (counts + log link) → MLE for GLMs via IRLS → Deviance & model comparison → Connects: exp family (L3), MLE (L5), regression (L13).

### Lecture 15: ANOVA, Model Comparison & Applications

ANOVA via F-test → Prediction vs confidence intervals → **Information criteria (AIC, BIC)** for model selection → A/B testing → Reproducibility crisis.

### Lecture 16: Causal Inference

*"Correlation is not causation — so how do we get causation?"*

Correlation ≠ causation (motivating examples) → Potential outcomes / Rubin causal model → Average Treatment Effect (ATE) → **Randomized Controlled Trials** (the gold standard) → Confounders & Simpson's paradox revisited (from L2) → **DAGs** (directed acyclic graphs, d-separation) → Conditioning vs intervening (do-calculus basics) → Observational studies: matching, propensity scores → Instrumental variables (brief) → Connection to A/B testing.

---

## Session Mapping

| Session | File | Topic |
|---------|------|-------|
| 1 | `01_stat.tex` | Foundations: population, sample, loss, ERM |
| 2 | `02_stat.tex` | Descriptive statistics, EDA, ECDF, pitfalls |
| 3 | `03_stat.tex` | Bias, variance, MSE, consistency, sufficiency, exp family |
| 4 | `04_stat.tex` | Fisher info, Cramer-Rao, admissibility, Stein's paradox |
| 5 | `05_stat.tex` | MoM, MLE, cross-entropy, MLE failures, numerical MLE |
| 6 | `06_stat.tex` | MAP, conjugate priors, regularization = MAP, overfitting |
| 7 | `07_stat.tex` | EM algorithm, GMMs, K-means as hard EM |
| 8 | `08_stat.tex` | MCMC, Metropolis-Hastings, Gibbs, Bayesian computation |
| 9 | `09_stat.tex` | Sampling distributions, delta method, CIs (freq & Bayes) |
| 10 | `10_stat.tex` | Bootstrap, permutation tests |
| 11 | `11_stat.tex` | Hypothesis testing, power analysis, multiple testing |
| 12 | `12_stat.tex` | LRT framework, t-tests, chi-squared, nonparametric |
| 13 | `13_stat.tex` | Regression inference (OLS + logistic) |
| 14 | `14_stat.tex` | GLMs: exp family + link functions, Poisson regression |
| 15 | `15_stat.tex` | ANOVA, AIC/BIC, model comparison |
| 16 | `16_stat.tex` | Causal inference, DAGs, potential outcomes, propensity |

---

## Cross-references to Existing Modules

| Stat topic | Builds on |
|-----------|-----------|
| ERM, loss functions | Module 05 (convexity, loss landscapes) |
| Regularization as MAP | Module 01 (norms), Module 05 (regularized regression) |
| MLE for distributions | Module 19 (Bernoulli, Poisson, Normal, Exponential) |
| MLE = MSE loss, cross-entropy | Module 03/05 (regression, loss functions) |
| Logistic regression as MLE | Module 04 (logistic regression), L5 (cross-entropy) |
| EM algorithm, GMMs | Module 18 (mixture models), L5 (MLE), L6 (latent variables) |
| MCMC, Gibbs sampling | Module 16 (conditional probability), L6 (Bayesian) |
| Sampling distributions, CLT | Module 20 (LLN, CLT) |
| Fisher information | Module 07 (Hessians, second derivatives) |
| Bias-variance decomposition | Module 17 (expectation, variance) |
| OLS, normal equation | Module 03 (linear regression) |
| Bayes' theorem, priors | Module 16 (conditional probability) |
| Delta method | Module 07 (Taylor expansion, derivatives) |
| GLMs, link functions | L3 (exp family), L5 (MLE), L13 (regression) |
| Causal inference, DAGs | L2 (Simpson's paradox), L11 (hypothesis testing) |
