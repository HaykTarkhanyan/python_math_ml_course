# Statistics Lectures — Outline & Plan

## Completed Lectures

### Lecture 1: Foundations (`01_stat.tex`, 24 slides)

| # | Slide | Section |
|---|-------|---------|
| 1 | Title | |
| 2 | How much should you trust a number? | Motivation |
| 3 | Probability vs Statistics | Forward vs inverse problem |
| 4 | Why the inverse problem is harder | |
| 5 | Population vs Sample | Core definitions |
| 6 | Parameter vs Statistic | |
| 7 | The Triple: Estimand / Estimator / Estimate | |
| 8 | Discussion (Armenia polling) | |
| 9 | Discussion: Answers | |
| 10 | The i.i.d. Assumption | Sampling assumptions |
| 11 | When does i.i.d. hold? | |
| 12 | When does i.i.d. break? | |
| 13 | Survivorship Bias | |
| 14 | The Plug-in Principle | Estimation ideas |
| 15 | Plug-in in Action | |
| 16 | The Summarization Problem | Loss & Risk |
| 17 | Three Losses, Three Optimal Summaries | |
| 18 | Visualizing the Losses | |
| 19 | Mean vs Median: Sensitivity to Outliers | |
| 20 | The Mean Can Mislead | |
| 21 | Risk and Empirical Risk | ERM |
| 22 | ERM in Machine Learning | |
| 23 | One Principle, Many Names | Unification |
| 24 | Questions? | |

**Homework:** none (end-of-lecture practical mentioned verbally)

---

### Lecture 2: Descriptive Statistics (`02_stat.tex`, 23 slides)

| # | Slide | Section |
|---|-------|---------|
| 1 | Title | |
| 2 | You get a spreadsheet with 10,000 rows... | Motivation |
| 3 | Goals of Descriptive Statistics | |
| 4 | Measures of Center | Center |
| 5 | Measures of Spread | Spread |
| 6 | Robust vs Non-Robust: Visual | |
| 7 | How a Histogram Is Built | Visualization |
| 8 | Bin Width Matters | |
| 9 | Quantiles and Percentiles | Quantiles |
| 10 | Boxplot Anatomy | |
| 11 | Boxplot Hides Bimodality | |
| 12 | Skewness: Measuring Asymmetry | Shape |
| 13 | Kurtosis: Tail Heaviness | |
| 14 | The Empirical CDF | ECDF |
| 15 | ECDF: Why It's Powerful | |
| 16 | Choosing the Right Plot | |
| 17 | Anscombe's Quartet (1973) | Pitfalls |
| 18 | The Datasaurus Dozen (2017) | |
| 19 | Simpson's Paradox | |
| 20 | Simpson's Paradox: UC Berkeley | |
| 21 | Simpson's Paradox: Why It Happens | |
| 22 | Homework | |
| 23 | Questions? | |

**Homework:** (1) Data visualization topic, (2) EDA on a Kaggle/armstat.am dataset, (3) examples of survivorship bias & Simpson's paradox

---

### Lecture 3: Properties of Estimators (`03_stat.tex`, 33 slides)

| # | Slide | Section |
|---|-------|---------|
| 1 | Title | |
| 2 | We use estimators every day. Are they any good? | Motivation |
| 3 | Bias: Is the Estimator Centered on the Truth? | **Bias** |
| 4 | Worked Example: Is X-bar Unbiased for mu? | |
| 5 | Worked Example: Why Dividing by n Is Biased (Bessel) | |
| 6 | Bias: Summary table | |
| 7 | The Dartboard Analogy | **Variance & MSE** |
| 8 | Variance of an Estimator | |
| 9 | MSE = Bias^2 + Variance: Derivation | |
| 10 | When Biased Beats Unbiased | |
| 11 | The Bias-Variance Tradeoff | **Tradeoff** |
| 12 | Consistency: Getting It Right Eventually | **Consistency** |
| 13 | Sufficient Conditions for Consistency | |
| 14 | Sufficiency: Can We Compress the Data? | **Sufficiency** |
| 15 | How to Check: Fisher-Neyman Factorization | |
| 16 | Minimal Sufficiency and Why It Matters | |
| 17 | Finding Minimal Sufficient Statistics (likelihood ratio) | |
| 18 | The Exponential Family: A Unifying Framework | **Exponential Family** |
| 19 | Why Exponential Families Are Special (completeness, Lehmann-Scheffe) | |
| 20 | Can We Do Better? The Fundamental Question | **Fisher Info & CR** |
| 21 | The Score Function: How Sensitive Is the Model? | |
| 22 | Fisher Information: How Informative Is One Observation? | |
| 23 | Intuition: Sharp vs Flat Log-Likelihood | |
| 24 | Cramer-Rao Lower Bound | |
| 25 | Regularity Conditions: When Does CR Apply? (Uniform counterexample) | |
| 26 | Cramer-Rao: Checking Efficiency (table) | |
| 27 | Admissibility | **Admissibility & Minimax** |
| 28 | Stein's Paradox (1956) | |
| 29 | Minimax Estimators | |
| 30 | Three Philosophies of Estimation | |
| 31 | Summary: How to Judge an Estimator | **Summary** |
| 32 | Homework | |
| 33 | Questions? | |

**Key constraint:** This lecture only uses knowledge from Lectures 1-2 (plug-in estimators). No MLE, MAP, regularization, or Bayesian concepts.

**Homework:** (1) X-bar unbiased + MSE, (2) sigma-hat biased, (3) Poisson Fisher info + CR + efficiency, (4) shrinkage estimator MSE optimization, (5) Poisson sufficiency via factorization

---

### Lecture 4: Point Estimation — MLE (`04_stat.tex`, 24 slides)

| # | Slide | Section |
|---|-------|---------|
| 1 | Title | |
| 2 | The Estimation Problem | Motivation (callbacks to L3) |
| 3 | From Data to Parameters | MoM vs MLE diagram |
| 4 | Method of Moments (MoM) | **MoM** |
| 5 | MoM Example: Normal Distribution | (Bessel callback) |
| 6 | The Likelihood Function | **Likelihood** |
| 7 | Likelihood: Coin Flip Example | |
| 8 | Log-Likelihood: Why We Prefer It | (score function callback to L3) |
| 9 | Maximum Likelihood Estimation | **MLE** |
| 10 | MLE: Bernoulli | |
| 11 | MLE: Normal | (bias callback to L3) |
| 12 | MLE: Poisson | |
| 13 | MLE: Exponential | |
| 14 | MLE: Summary of Examples | (CR bound teaser) |
| 15 | Invariance Property | **MLE Properties** |
| 16 | Identifiability | |
| 17 | Visualizing Non-Identifiability | |
| 18 | MLE and Sufficient Statistics | **MLE Meets L3** |
| 19 | MLE in Exponential Families | |
| 20 | Why MLE Works: Big Theoretical Guarantees | |
| 21 | MLE Achieves the Cramér–Rao Bound | |
| 22 | Practical: Implement MLE | **Practical** |
| 23 | Homework | |
| 24 | Questions? | |

**Homework:** (1) Geometric MLE + efficiency, (2) Normal MLE=MoM via exponential family, (3) Uniform(0,θ) MLE — not exp family, (4) Poisson simulation verifying unbiasedness + variance

---

## Ready but Not Yet Numbered

### MAP Lecture (`xx_stat.tex`, ~16 slides → future `05_stat.tex`)
Bayes' theorem for parameters, prior x likelihood = posterior, MAP, regularization connection (Gaussian -> Ridge, Laplace -> Lasso), MLE vs MAP vs full Bayesian.

---

## Future Lectures

### Lecture 6: Sampling Distributions & Confidence Intervals (~2 sessions)

**Motivating question:** *A poll reports "52% +/- 3%". Where does the +/- come from?*

- Sampling distribution of an estimator (theta-hat is a random variable)
  - Demo: repeatedly sample, compute mean, plot histogram -> Normal (CLT callback)
- Standard error = std of the sampling distribution
  - SE of the mean: sigma/sqrt(n), the sqrt(n) law
- Asymptotic normality of MLE: theta-hat ~ N(theta, 1/(nI(theta)))
  - This is why MLE is central: estimates AND uncertainty
- Wald confidence intervals: theta-hat +/- z * SE
- Delta method: CI for transformations of theta (odds ratio, relative risk)
- Confidence intervals vs credible intervals (frequentist vs Bayesian)
- **Practical:** 95% CI for a proportion; compare Wald, Wilson, and bootstrap intervals

### Lecture 7: Bootstrap & Permutation Tests (~1-2 sessions)

**Motivating question:** *You have one dataset, a complicated statistic, and no formula for its SE. Now what?*

- The bootstrap idea: treat sample as mini-population, resample with replacement
- Bootstrap confidence intervals (percentile, pivotal, BCa)
- When bootstrap fails: dependent data, heavy tails, small n, boundary statistics
- Permutation (randomization) tests
  - "If the null is true, the labels don't matter -- shuffle and see"
  - Exact p-value by enumeration or Monte Carlo
- Cross-validation vs bootstrap (evaluation vs inference)
- **Practical:** Bootstrap the median; permutation test for treatment effect

### Lecture 8: Hypothesis Testing (~1-2 sessions)

**Motivating question:** *A new drug lowers blood pressure by 3 mmHg. Real effect or noise?*

- The logic: assume nothing is happening (H0), see if data contradicts it
- Null H0 vs alternative H1, test statistic, rejection region, significance level alpha
- Type I error (false alarm) vs Type II error (missed detection)
- Power, effect size, sample size planning
- p-values: the most misunderstood number in science
  - Correct: P(data this extreme | H0 true)
  - NOT: P(H0 true | data)
  - Statistical significance != practical significance
- Multiple testing: Bonferroni, Holm, Benjamini-Hochberg (FDR)
- **Practical:** Simulate 1000 null experiments; show ~50 are "significant" at alpha=0.05

### Lecture 9: Classical Tests (~1-2 sessions)

**Motivating question:** *Which test do I actually use?*

- z-test -> t-test: one-sample, paired, two-sample (Welch's)
  - Assumptions: normality (QQ-plots), equal variance
  - Nonparametric alternatives: Wilcoxon, Mann-Whitney
- Chi-squared goodness-of-fit, chi-squared test of independence
- Likelihood Ratio Test (LRT): the unifying framework
  - Every test above is a special case
- **Practical:** t-test, permutation test, and bootstrap test on the same question

### Lecture 10: Inference in Linear Regression (~1-2 sessions)

**Motivating question:** *Does smoking cause lung cancer -- or do smokers just happen to be different?*

- OLS estimates beta-hat = (X'X)^{-1}X'y (review from Module 03)
- beta-hat ~ N(beta, sigma^2 (X'X)^{-1}) under Gauss-Markov
- Standard errors, t-tests, p-values for each coefficient
- Diagnostics: residual plots, QQ-plots, leverage, Cook's distance
- R^2 and adjusted R^2

### Lecture 11: ANOVA & Model Comparison (~1-2 sessions)

- Multiple regression: dummy variables, interactions
- ANOVA as model comparison via F-test (connects to LRT)
- Prediction intervals vs confidence intervals
- Logistic regression as MLE (if time permits)
- **Practical:** Full inference pipeline on a real dataset

### Lecture 12: Applications & Capstone (~1-2 sessions)

- A/B testing: design, analysis, pitfalls (peeking, sequential testing)
- Sample size planning
- Comparing methods: paired tests, McNemar's test
- Reproducibility crisis, pre-registration
- **Capstone:** End-to-end case study from raw data to conclusion

---

## Lecture-to-Session Mapping

| Session | File | Topic |
|---------|------|-------|
| 1 | `01_stat.tex` | Foundations: population, sample, loss, ERM |
| 2 | `02_stat.tex` | Descriptive statistics, EDA, ECDF, pitfalls |
| 3 | `03_stat.tex` | Properties: bias, variance, MSE, consistency, sufficiency, exp family, CR |
| 4 | `04_stat.tex` | MLE: MoM, likelihood, MLE worked examples, MLE meets L3 (sufficiency, efficiency) |
| 5 | `xx_stat.tex` -> `05_stat.tex` | MAP: priors, posterior, regularization connection |
| 6-7 | `06_stat.tex` (to create) | Sampling distributions, asymptotic normality, confidence intervals |
| 8 | `07_stat.tex` (to create) | Bootstrap, permutation tests |
| 9-10 | `08_stat.tex` (to create) | Hypothesis testing, p-values, multiple testing |
| 10-11 | `09_stat.tex` (to create) | Classical tests: t-test, chi-squared, LRT |
| 11-12 | `10_stat.tex` (to create) | Linear regression inference |
| 12-13 | `11_stat.tex` (to create) | ANOVA, model comparison |
| 13-14 | `12_stat.tex` (to create) | Applications, A/B testing, capstone |

**~14 sessions total.**

---

## Cross-references to Existing Modules

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
