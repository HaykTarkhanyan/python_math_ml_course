# Statistics — Lecture Plan

> **Prerequisites:** probability (discrete + continuous distributions, expectation, variance, covariance/correlation, Bayes' theorem, LLN/CLT sketches), linear algebra (normal equation, matrix operations, eigenvalues), calculus (gradients, optimization basics), Python.
>
> **Design principle:** statistics is the science of learning from data under uncertainty. Every section earns its place by answering a real question — from medicine, engineering, science, business, or ML. The goal is a complete statistical thinker, not a toolbox memorizer.

---

## 0. Foundations: what statistics is and why it's hard *(1 lecture)*

**Motivating question:** *A poll says 52% support candidate A (n = 1000). A clinical trial says drug B reduces symptoms by 15% (n = 200). How much should you trust these numbers?*

- Population vs sample; parameters (estimands) vs statistics (estimators)
- The i.i.d. assumption — and real-world violations (time dependence, selection bias, non-response, distribution shift)
- The plug-in principle: replace the unknown distribution with the empirical one
- Loss functions → risk → empirical risk minimization (ERM)
  - Mean minimizes squared-error loss, median minimizes absolute-error loss, mode minimizes 0-1 loss
  - Why the choice of loss reflects your values, not just math
- Examples across domains: polling, quality control in manufacturing, clinical dosing, train/test splits in ML
- **Practical:** Given a dataset, compute empirical risk under different losses in Python; observe how the "best" summary changes with the loss

> **Bridge:** We now have point summaries — but a single number hides a lot. How do we see the full picture?

---

## 1. Descriptive statistics and empirical distributions *(1 lecture)*

**Motivating question:** *You get a spreadsheet with 10,000 rows. What do you look at first — and what can fool you?*

- Sample moments (mean, variance, skewness, kurtosis) — what each reveals
- Robust alternatives: median, IQR, MAD — why outliers matter
  - Heavy-tailed data in practice: income distributions, insurance claims, network traffic
- Quantiles, percentiles, boxplots
  - Applications: grading curves, risk thresholds (VaR in finance), anomaly detection
- Empirical CDF and its properties
  - Visual tool: comparing distributions (Kolmogorov–Smirnov intuition)
- Anscombe's quartet / Datasaurus Dozen — summary statistics can lie, always plot your data
- **Practical:** EDA on a real dataset: histograms, boxplots, ECDF, spotting data quality issues before any analysis

> **Bridge:** Descriptive stats summarize data. But we want to *learn parameters* — how do we go from data to estimates?

---

## 2. Point estimation: MLE and MAP *(2 lectures)*

**Motivating question:** *A factory produces lightbulbs. You test 50 and find a mean lifetime of 1,200 hours. What can you say about the true mean lifetime — and how confident should you be in your method?*

### Lecture 2a: Method of Moments and Maximum Likelihood

- Method of Moments — quick intuition, when it works, when it's awkward
- The likelihood function: "how plausible is this parameter value, given the data I observed?"
  - Log-likelihood and why we prefer it (numerical stability, sums > products)
  - Score function = gradient of log-likelihood
- Maximum Likelihood Estimation (MLE)
  - Worked examples: Bernoulli (coin fairness), Normal (measurement error), Poisson (rare events), Exponential (waiting times)
  - Invariance property: MLE of g(θ) = g(MLE of θ)
- Identifiability: when can we even hope to recover θ? Failure modes (mixture models, overparameterization)
- **Practical:** Implement MLE for a Gaussian from scratch; compare with `scipy.stats.fit`; fit a Poisson to real count data (e.g., goals per match, earthquake counts)

### Lecture 2b: MAP, Priors, and the Bayesian perspective

- Bayesian twist: what if we have prior information?
  - Prior × Likelihood ∝ Posterior — the update rule
  - MAP = mode of the posterior
- When does the prior matter? (small n vs large n — the data eventually overwhelms the prior)
- The regularization connection:
  - Gaussian prior → L2 penalty (Ridge); Laplace prior → L1 penalty (Lasso)
  - Why this matters: shrinkage, sparsity, and preventing overfitting in any regression setting
- MLE vs MAP vs full Bayesian — different philosophies, practical tradeoffs
- **Practical:** Estimate a coin's bias with different priors; visualize how the posterior evolves as data accumulates; show Ridge regression as MAP with varying prior strength

> **Bridge:** We can now *estimate* parameters. But how good are our estimates? Can we trust them?

---

## 3. Estimator quality: bias, variance, and the tradeoff *(1 lecture)*

**Motivating question:** *Two labs estimate the same physical constant using different methods. One is systematically off but consistent; the other is unbiased but noisy. Which is better?*

- Bias of an estimator: E[θ̂] − θ
  - Example: why we divide by n−1 for sample variance (Bessel's correction)
  - Example: biased but useful estimators (James–Stein, shrinkage)
- Variance of an estimator: how much does θ̂ jump between samples?
- MSE = Bias² + Variance — the fundamental decomposition
  - The tradeoff: sometimes accepting a little bias buys a large variance reduction
  - This shows up everywhere: regularization, model complexity, smoothing
- Consistency: θ̂ → θ as n → ∞ (connects to LLN from Module 20)
- Efficiency and the Cramér–Rao lower bound
  - "No unbiased estimator can have variance below 1/I(θ)" — a fundamental limit of estimation
  - MLE is asymptotically efficient (preview of Section 4)
- Sufficient statistics *(optional):* when can we compress data without losing information about θ?
- **Practical:** Simulate the bias-variance tradeoff — fit polynomials of degree 1, 3, 10, 20 to noisy data; plot bias², variance, and MSE vs complexity; connect to the same tradeoff in ridge regression

> **Bridge:** We know how to measure estimator quality in theory. But in practice, we need *intervals*, not just point estimates.

---

## 4. Uncertainty quantification and asymptotics *(2 lectures)*

**Motivating question:** *A poll reports "52% ± 3%". A lab reports "9.81 ± 0.02 m/s²". Where do those ± numbers come from — and what do they actually mean?*

### Lecture 4a: Sampling distributions, standard errors, Fisher information

- Sampling distribution of an estimator (θ̂ is a random variable!)
  - Demo: repeatedly sample, compute mean, plot histogram → it's Normal (CLT callback)
- Standard error = std of the sampling distribution
  - SE of the mean: σ/√n — the √n law and why quadrupling the sample only halves the error
- Fisher information: how much does one observation tell you about θ?
  - Expected information = E[score²] = −E[∂²ℓ/∂θ²]
  - Curvature intuition: sharp likelihood peak → high information → low uncertainty
  - Observed vs expected information — when they differ
- **Practical:** Compute Fisher information for Bernoulli, Normal; verify Cramér–Rao bound numerically via simulation

### Lecture 4b: Confidence intervals and asymptotic normality of MLE

- The MLE is asymptotically Normal: θ̂_MLE ~ N(θ, 1/nI(θ))
  - This is *why* MLE is so central — it gives you estimates *and* uncertainty
- Wald confidence intervals: θ̂ ± z · SE
- Delta method *(as needed):* CI for transformations of θ (e.g., odds ratio from log-odds, relative risk)
- Likelihood-based intervals *(optional):* profile likelihood, the "2 log-likelihood units" rule
- Confidence intervals vs credible intervals — what each actually promises
  - Frequentist: "95% of intervals built this way contain θ"
  - Bayesian: "Given the data and prior, θ is in this interval with 95% probability"
  - Neither is "wrong" — they answer different questions
- **Practical:** Compute 95% CI for a proportion (election poll, defect rate, test accuracy); compare Wald, Wilson, and bootstrap intervals; observe how they diverge at small n or extreme proportions

> **Bridge:** What if we can't derive the sampling distribution analytically? We simulate it.

---

## 5. Resampling methods *(1–2 lectures)*

**Motivating question:** *You have one dataset, a complicated statistic, and no formula for its standard error. Now what?*

- The bootstrap idea: treat your sample as a mini-population, resample from it
  - Nonparametric bootstrap: sample with replacement, compute statistic, repeat B times
  - Bootstrap distribution ≈ sampling distribution
- Bootstrap confidence intervals
  - Percentile method (simple, intuitive)
  - Basic (pivotal) method
  - BCa *(optional — mention it exists, no derivation)*
- When bootstrap fails and why: dependent data, heavy tails, small n, statistics at boundaries (e.g., max)
- Permutation (randomization) tests — the logic:
  - "If the null is true, the labels don't matter — so shuffle and see"
  - Exact p-value by enumeration or Monte Carlo approximation
  - Examples: treatment vs control, before vs after, model A vs model B
- Cross-validation vs bootstrap — different purposes:
  - CV: estimate prediction error (evaluation)
  - Bootstrap: estimate parameter uncertainty (inference)
- **Practical:** Bootstrap the median income from survey data; permutation test for a treatment effect; compare results with analytical formulas where available

> **Bridge:** We can now quantify uncertainty. But how do we make *decisions* — "is this effect real or noise?"

---

## 6. Hypothesis testing framework *(1–2 lectures)*

**Motivating question:** *A new drug lowers blood pressure by 3 mmHg in a trial. Is that a real effect or just random variation?*

### Core framework

- The logic: assume nothing is happening (H₀), see if the data contradicts it
- Null hypothesis H₀ vs alternative H₁
- Test statistic: a single number that measures "how far from H₀"
- Rejection region and significance level α
- Type I error (false alarm) and Type II error (missed detection)
  - Concrete costs: approving an ineffective drug vs missing an effective one; convicting the innocent vs freeing the guilty
- Power = P(reject H₀ | H₁ true) — why underpowered studies waste everyone's time
- Effect size and sample size planning
  - "With n=20, you can't detect small effects. With n=20,000, everything is 'significant'."

### p-values: the most misunderstood number in science

- Correct: P(data this extreme or more | H₀ true)
- **Not**: P(H₀ is true | data) — that's the transpose fallacy (connects to prosecutor's fallacy from Module 19!)
- p = 0.049 vs p = 0.051 — the cliff-edge problem
- Statistical significance ≠ practical significance (a drug that costs $10,000 and lowers blood pressure by 0.1 mmHg)
- The ASA statement on p-values *(brief)*

### Multiple testing

- The problem: test 20 hypotheses, one will be "significant" by chance
  - Genomics: testing 20,000 genes. Drug trials: testing multiple endpoints. A/B testing: testing many variants.
- Family-wise error rate: Bonferroni (conservative), Holm (step-down, less conservative)
- False Discovery Rate: Benjamini–Hochberg — control the *fraction* of false discoveries
- **Practical:** Simulate 1000 null experiments; show that ~50 are "significant" at α = 0.05; apply Bonferroni and BH corrections; discuss which is appropriate when

> **Bridge:** The framework is abstract — let's see the concrete tests that fall out of it.

---

## 7. Classical tests as instances of the framework *(1–2 lectures)*

**Motivating question:** *Which test do I actually use?* — a decision tree, not a menu to memorize.

### Comparing means

- z-test (known σ) → t-test (unknown σ): one-sample, paired, two-sample
  - When to use paired vs unpaired (before/after treatment; same subjects vs different groups)
  - Welch's t-test: don't assume equal variances (almost always prefer this)
  - Assumptions check: normality (QQ-plots), what to do when it fails
  - Nonparametric alternatives *(brief):* Wilcoxon signed-rank, Mann-Whitney U
- Examples: clinical trial (drug vs placebo), education (teaching method A vs B), engineering (old process vs new)

### Comparing distributions / categories

- Chi-squared goodness-of-fit: "does my data follow this distribution?"
  - Examples: are dice fair? Does website traffic follow a Poisson? Are predicted probabilities well-calibrated?
- Chi-squared test of independence: "are these two categorical variables related?"
  - Examples: smoking × lung cancer, gender × hiring, feature × target

### The unifying view

- Likelihood Ratio Test (LRT): −2 log(L₀/L₁) ~ χ²
  - Nested model comparison — the most general framework
  - Every test above is a special case (or close approximation)
  - This is the tool that scales: any two nested models, any likelihood
- **Practical:** Apply t-test, permutation test, and bootstrap test to the same question; see when they agree and disagree; discuss which assumptions each relies on

> **Bridge:** Tests compare simple hypotheses. Linear models let us ask richer questions while controlling for confounders.

---

## 8. Linear models as a unifying inference engine *(2 lectures)*

**Motivating question:** *Does smoking cause lung cancer — or do smokers just happen to be different in other ways? How do we disentangle effects?*

### Lecture 8a: Inference in linear regression

- Review: OLS estimates β̂ = (X'X)⁻¹X'y (connects to Module 03, normal equation)
- The statistical lens: β̂ is a *random variable* with a sampling distribution
  - β̂ ~ N(β, σ²(X'X)⁻¹) under Gauss-Markov assumptions
  - Standard errors, t-tests, p-values for each coefficient
  - "Controlling for age and income, is education significantly associated with health outcomes?"
- Diagnostics: residual plots, QQ-plots, leverage, Cook's distance
  - What to do when assumptions fail: transformations, robust SE, nonparametric approaches
- R² and adjusted R² — what they do and don't tell you

### Lecture 8b: Model comparison, ANOVA, and beyond

- Multiple regression: dummy variables for categories, interaction terms
  - Interpreting interactions: "the effect of X depends on the level of Z"
- ANOVA as model comparison: "does adding these variables significantly improve the model?"
  - F-test = comparing nested linear models (connects to LRT from Section 7)
  - One-way, two-way ANOVA as special cases
- Prediction intervals vs confidence intervals
  - CI for E[y|x]: where does the regression *line* live?
  - PI for a new y: where will the next *observation* land?
  - Why PIs are always wider than CIs — and when each matters
- Logistic regression as MLE *(if time permits)*
  - Binary outcomes → Bernoulli likelihood → iteratively reweighted least squares
  - Connects everything: MLE, regularization (MAP), model comparison (LRT)
- **Practical:** Full inference pipeline on a real dataset — fit a model, interpret coefficients, test significance, check diagnostics, compute prediction intervals, compare nested models

> **Bridge:** We've built the full statistical toolkit. Let's see it in action on applied problems.

---

## 9. Applications and integration *(1–2 lectures, capstone-style)*

**Motivating question:** *You have the tools — now how do you actually use them to make decisions under uncertainty?*

### Experimental design and A/B testing

- Randomization: why it's the gold standard and when you can't do it (observational vs experimental)
- Sample size planning: "how many subjects/users/measurements do I need?"
- A/B testing: design, analysis, pitfalls
  - Why "peeking" at p-values inflates false positives
  - Sequential testing *(brief mention)*
- Guardrail metrics: watching for harm while optimizing the primary outcome

### Evaluating and comparing complex procedures

- Uncertainty for any metric via bootstrap: accuracy, AUC, median survival time, whatever you need
- Comparing two methods on the same data:
  - Paired tests and permutation tests
  - McNemar's test for paired binary outcomes
- The reproducibility crisis and what it means for science
  - Pre-registration, effect sizes over p-values, replication

### Putting it all together

- End-to-end case study: from raw data → EDA → modeling → statistical comparison → decision
- **Practical / capstone project:** Students receive a dataset and a question; must produce a short report using the full pipeline — descriptive stats, estimation, confidence intervals, hypothesis tests, model comparison, and a clearly stated conclusion with uncertainty

---

## Appendix: Lecture-to-session mapping (suggested pacing)

| Session | Section | Topic |
|---------|---------|-------|
| 1 | 0 | Foundations: population, sample, loss, ERM |
| 2 | 1 | Descriptive statistics, EDA, empirical CDF |
| 3 | 2a | Method of Moments, MLE |
| 4 | 2b | MAP, priors, regularization as MAP |
| 5 | 3 | Bias, variance, MSE, Cramér–Rao |
| 6 | 4a | Sampling distributions, Fisher information |
| 7 | 4b | Asymptotic normality of MLE, confidence intervals |
| 8 | 5 | Bootstrap, permutation tests |
| 9 | 6 | Hypothesis testing, p-values, multiple testing |
| 10 | 7 | t-tests, chi-squared, LRT |
| 11 | 8a | Inference in linear regression |
| 12 | 8b | ANOVA, model comparison, prediction intervals |
| 13–14 | 9 | Applications, A/B testing, capstone |

**~14 sessions total.** Sections marked *(optional)* can be trimmed to fit ~12 sessions if needed.

---

## Cross-references to existing modules

| Stat topic | Builds on |
|-----------|-----------|
| ERM, loss functions | Module 05 (convexity, loss landscapes) |
| Regularization as MAP | Module 01 (norms), Module 05 (regularized regression) |
| MLE for known distributions | Module 19 (Bernoulli, Poisson, Normal, Exponential) |
| Sampling distributions, CLT | Module 20 (LLN, CLT) |
| Fisher information, curvature | Module 07 (Hessians, second derivatives) |
| Bias-variance decomposition | Module 17 (expectation, variance) |
| Normal equation, OLS | Module 03 (linear regression, normal equation) |
| Bayes' theorem, posteriors | Module 16 (conditional probability, Bayes) |
| Covariance in regression | Module 18 (covariance, correlation) |
