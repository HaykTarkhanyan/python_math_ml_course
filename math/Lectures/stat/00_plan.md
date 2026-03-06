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

### Lecture 6: MAP & Bayesian Estimation (`06_stat.tex`, 25 frames)

Recap of L5 → Bayes' theorem for parameters → Prior × likelihood ∝ posterior → Conjugate priors (Beta-Binomial, Normal-Normal, Gamma-Poisson, Gamma-Exp) → **Conjugate priors as pseudo-observations** → **Sequential Bayesian updating** → **Choosing priors** (informative, weakly informative, Jeffreys) → MAP estimation → When does the prior matter? → Regularization = MAP (Gaussian → Ridge, Laplace → Lasso, **Elastic Net**) → **MLE vs MAP overfitting demo** → MLE vs MAP vs full Bayesian → **Empirical Bayes** → When to use what.

Builds on: L5 (MLE), L4 (Fisher info for Jeffreys), Module 16 (Bayes' theorem), Module 01 (norms), Module 05 (regularization).

**Homework:** (1) Normal conjugacy MAP derivation, (2) Beta-Binomial MAP under 3 priors, (3) Ridge closed-form = MAP

---

### Lecture 7: Confidence Intervals (`07_stat.tex`, 28 frames)

Recap of L5 & L6 → Sampling distributions (hat-theta is random) → The sqrt(n) law → Standard error (known vs estimated) → **Wald CI** → What 95% confidence really means → Width determinants → CI for proportions (Wald, **Wilson interval**) → Sample size planning → **t-interval** (unknown sigma) → **General MLE CI recipe** → **Bayesian credible intervals vs frequentist CIs** → **Delta method for CIs**.

Builds on: L5 (MLE, asymptotic normality), L4 (Fisher info, CR bound), Module 20 (CLT).

**Homework:** (1) Lightbulb CIs at 3 levels, (2) Poll CI + sample size, (3) Exponential CI via Fisher + delta method, (4) Wald vs Wilson simulation

---

### Lecture 8: The Bootstrap (`08_stat.tex`, 30 frames)

Recap of L7 → The bootstrap idea (resample with replacement) → Bootstrap algorithm → **Bootstrap SE** → **Bootstrap bias estimation** → Why bootstrap works (plug-in principle) → **Bootstrap CIs** (Normal, **Percentile**, **BCa**) → Comparing CI methods → **Parametric vs nonparametric bootstrap** → Bootstrap for correlation → **When bootstrap fails** (extremes, small n, dependence, non-smooth) → **Permutation tests** (shuffle labels, null distribution, p-value) → Bootstrap vs permutation → How many replicates? → Python code.

Builds on: L7 (CIs, SE), L5 (MLE for Uniform — bootstrap failure), L3 (bias-variance tradeoff).

**Homework:** (1) Bootstrap SE/CI for median, (2) Bootstrap SE converges to S/sqrt(n), (3) A/B test: permutation + bootstrap, (4) Bootstrap failure for Uniform max

---

## Future Lectures

### Lecture 9: Hypothesis Testing

*"A new drug lowers blood pressure by 3 mmHg. Real effect or noise?"*

H0/H1, test statistics, Type I/II errors → p-values (interpretation & misinterpretation) → **Power analysis & sample size planning** (effect size, power curves, designing studies) → Multiple testing (Bonferroni, BH-FDR).

### Lecture 10: Likelihood Ratio Tests & Classical Tests

*"Which test do I actually use?"*

**LRT as unifying framework** (Neyman-Pearson lemma, -2 log Λ ~ χ²) → z-test & t-test (one-sample, paired, Welch's) **as special cases of LRT** → Chi-squared tests (goodness-of-fit, independence) → Nonparametric alternatives (Mann-Whitney, Wilcoxon).

### Lecture 11: Regression Inference

OLS inference → Gauss-Markov theorem → SE / t-tests / p-values for coefficients → Diagnostics (residuals, QQ, leverage, Cook's distance) → R² and adjusted R² → **Logistic regression as MLE** (connects back to L5 cross-entropy slide) → Inference for logistic regression coefficients.

### Lecture 12: Generalized Linear Models

*"OLS and logistic regression look different — but they're the same machine."*

Three components: random (exp family), systematic (linear predictor), link function → Canonical links (identity, logit, log) → OLS as GLM (Normal + identity) → Logistic as GLM (Bernoulli + logit) → **Poisson regression** (counts + log link) → MLE for GLMs via IRLS → Deviance & model comparison → Connects: exp family (L3), MLE (L5), regression (L11).

### Lecture 13: ANOVA, Model Comparison & Applications

ANOVA via F-test → Prediction vs confidence intervals → **Information criteria (AIC, BIC)** for model selection → A/B testing → Reproducibility crisis.

### Lecture 14: Causal Inference

*"Correlation is not causation — so how do we get causation?"*

Correlation ≠ causation (motivating examples) → Potential outcomes / Rubin causal model → Average Treatment Effect (ATE) → **Randomized Controlled Trials** (the gold standard) → Confounders & Simpson's paradox revisited (from L2) → **DAGs** (directed acyclic graphs, d-separation) → Conditioning vs intervening (do-calculus basics) → Observational studies: matching, propensity scores → Instrumental variables (brief) → Connection to A/B testing.

*Note: EM algorithm and MCMC/Bayesian computation topics may be added as supplementary lectures if time permits.*

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
| 7 | `07_stat.tex` | Sampling distributions, CIs (Wald, Wilson, t, delta method) |
| 8 | `08_stat.tex` | Bootstrap, bootstrap CIs (Normal/Percentile/BCa), permutation tests |
| 9 | `09_stat.tex` | Hypothesis testing, power analysis, multiple testing |
| 10 | `10_stat.tex` | LRT framework, t-tests, chi-squared, nonparametric |
| 11 | `11_stat.tex` | Regression inference (OLS + logistic) |
| 12 | `12_stat.tex` | GLMs: exp family + link functions, Poisson regression |
| 13 | `13_stat.tex` | ANOVA, AIC/BIC, model comparison |
| 14 | `14_stat.tex` | Causal inference, DAGs, potential outcomes, propensity |

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
| GLMs, link functions | L3 (exp family), L5 (MLE), L11 (regression) |
| Causal inference, DAGs | L2 (Simpson's paradox), L9 (hypothesis testing) |
