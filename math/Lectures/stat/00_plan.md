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

### Lecture 7: Sampling Distributions (`07_stat.tex`, 28 frames)

Recap of L5 & L6 → **θ̂ is random** (thought experiment, diagram) → But we only have one sample (theory / simulation / bootstrap) → **Monte Carlo simulation** (recipe, Normal example, effect of n) → Every estimator has a sampling distribution (mean vs median vs variance) → MLE for Exp(λ) → **CLT** (statement, Uniform demo, Exponential demo, how large must n be?) → CLT for MLEs (Bernoulli, Poisson, Exp) → **SD ≠ SE** (the most confused pair) → SE of X̄ = σ/√n (derivation) → **√n law** (diminishing returns) → SE for any MLE via Fisher info → Plug-in SE → Known vs estimated SE (z vs t) → **Fisher info ↔ sampling distribution** (sharp vs flat) → The full picture (analytical vs computational paths) → Python Monte Carlo code.

Builds on: L4 (Fisher info, CR bound), L5 (MLE, asymptotic normality), Module 20 (CLT).

**Homework:** (1) SE of p̂ two ways, (2) Poisson sampling distribution simulation, (3) IQ sample size planning, (4) Uniform MLE SE — why asymptotic formula fails

---

### Lecture 8: Confidence Intervals & The Bootstrap (`08_stat.tex`, 30 frames)

**Part I — Confidence Intervals (analytical path):** What is a CI (definition + many-CIs visualization) → Wald CI construction → Example (lightbulb) → What 95% confidence really means (correct vs wrong interpretation) → Width determinants (confidence level, σ, n) → CI for proportions (Wald + Wilson, when Wald fails) → Sample size planning → **t-interval** + general MLE CI recipe → CI vs credible interval (frequentist vs Bayesian) → **Delta method** for transformations.

**Part II — The Bootstrap (computational path):** When formulas don't exist → Core insight (ideal vs bootstrap world) → Bootstrap algorithm (5 steps) → Visualizing bootstrap samples → Bootstrap SE + distribution → **Bootstrap CIs** (Normal, Percentile, **BCa**) → Why it works (plug-in principle) → Parametric vs nonparametric → When bootstrap fails (extremes, small n, dependence, non-smooth).

**Part III — Two Paths, One Goal:** Analytical vs bootstrap comparison (pros/cons) → When to use which → Python code.

Builds on: L7 (sampling distributions, SE, CLT), L5 (MLE), L4 (Fisher info), L6 (credible intervals), L3 (bias-variance).

*Note: Permutation tests moved to L9/L10 (after hypothesis testing introduces p-values).*

---

### Lecture 9: Hypothesis Testing (`09_stat.tex`, 38 frames)

*"A new drug lowers blood pressure by 3 mmHg. Real effect or noise?"*

**Part I — The Logic of Testing:** Running example (drug trial: n=100, 3 mmHg difference) → Innocent until proven guilty (courtroom analogy) → H0/H1 → Test statistic (z-statistic) → Rejection region & significance level.

**Part II — Two Types of Errors:** Type I/II error table → Real-world costs (medicine, justice, spam).

**Part III — p-Values:** Definition & visualization → What p-values are NOT (4 common misinterpretations) → Statistical vs practical significance (effect size matters) → The cliff-edge problem (p=0.049 vs 0.051, ASA statement).

**Part IV — Power Analysis:** Power definition & visualization (two overlapping distributions) → Four knobs (effect size, n, α, σ) → Cohen's d (small/medium/large) → Power curves → Sample size formula.

**Part V — Permutation Tests:** If H0 is true, labels don't matter → Algorithm → Permutation distribution visualization → Pros and cons.

**Part VI — Multiple Testing:** The problem (1-(1-α)^m) → Bonferroni vs Benjamini-Hochberg (BH step-up visualization).

Builds on: L7 (sampling distributions, SE), L8 (CIs, bootstrap), L6 (Bayesian for p-value misconception), Module 16 (Bayes' theorem).

**Homework:** (1) Coffee shop z-test + CI agreement, (2) Sample size planning for d=0.3, (3) Gene testing with Bonferroni/BH, (4) Permutation test for two teaching methods.

---

### Lecture 10: Classical Tests & the LRT Framework (`10_stat.tex`, 30 frames)

*"Which test do I actually use?"*

**Part I — The Unifying Principle:** Nested models (restricted vs full) → Likelihood ratio statistic (Λ) → Wilks' theorem ($-2\log\Lambda \to \chi^2_k$) → The χ² distribution (density curves, key facts, decision rule).

**Part II — Tests for Means:** One-sample t-test (coffee shop example) → Paired t-test (blood pressure before/after, why pairing matters) → Two-sample t-test (pooled + **Welch's** — default to Welch's) → Teaching methods example → Test for proportions (z-test, coin flip example, two-proportion formula).

**Part III — Chi-Squared Tests:** Goodness-of-fit (die fairness example with bar chart) → Test of independence (smoking/cancer 2×2 contingency table, expected counts formula).

**Part IV — When Assumptions Fail:** Mann–Whitney U test (rank-based, visualization of ranking procedure) → Wilcoxon signed-rank test (paired nonparametric, bar chart of differences).

**Decision Flowchart:** TikZ decision tree (data type → number of groups → paired? → specific test).

Builds on: L9 (hypothesis testing, p-values, power), L7 (sampling distributions, CLT), L5 (MLE), L4 (Fisher info).

**Homework:** (1) One-sample t + CI (manufacturer weight claim), (2) Paired vs unpaired t on runners' times, (3) χ² GoF on transport preferences, (4) χ² independence on study method vs exam result.

---

## Future Lectures

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
| 7 | `07_stat.tex` | Sampling distributions, Monte Carlo, CLT in action, SD vs SE |
| 8 | `08_stat.tex` | CIs (Wald, Wilson, t, delta) + bootstrap (SE, percentile, BCa) |
| 9 | `09_stat.tex` | Hypothesis testing, p-values, power, permutation tests, multiple testing |
| 10 | `10_stat.tex` | Classical tests & LRT: t-tests, chi-squared, nonparametric, decision flowchart |
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
