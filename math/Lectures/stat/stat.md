# Statistics — Lecture Plan

> **Prerequisites:** probability (discrete + continuous distributions, expectation, variance, covariance/correlation, Bayes' theorem, LLN/CLT sketches), linear algebra (normal equation, matrix operations, eigenvalues), calculus (gradients, optimization basics), Python.
>
> **Design principle:** statistics is the science of learning from data under uncertainty. Every section earns its place by answering a real question — from medicine, engineering, science, business, or ML. The goal is a complete statistical thinker, not a toolbox memorizer.

---

## 0. Foundations: what statistics is and why it's hard *(1 lecture)*

**Motivating question:** *A poll says 52% support candidate A (n = 1000). A clinical trial says drug B reduces symptoms by 15% (n = 200). How much should you trust these numbers?*

### 0.1 Opening: probability vs statistics

- Probability: given the rules of the game (a fair die), predict what you'll see → forward problem
- Statistics: given what you saw (data), figure out the rules → inverse problem
- This is harder — the inverse problem is usually ill-posed (many rules could produce the same data)
- Discussion: "You flip a coin 10 times and get 7 heads. Is the coin fair?" — collect student intuitions, don't resolve yet. This question will take the entire course to answer properly.

### 0.2 Population, sample, parameter, statistic

- **Population:** the complete collection we care about (all voters, all lightbulbs from this factory, all patients with condition X)
  - Can be finite (voters in a country) or conceptually infinite (all future measurements from an instrument)
- **Parameter:** a fixed (unknown) number that describes the population
  - Notation: θ, μ, σ², p — these are what we want to learn
  - Examples: true approval rating of candidate A; true mean lifetime of a lightbulb; true probability a drug works
- **Sample:** the subset we actually observe — $X_1, X_2, \ldots, X_n$
  - This is what we have. Everything else is inference.
- **Statistic (= estimator):** any function of the sample — $T(X_1, \ldots, X_n)$
  - Examples: sample mean $\bar{X}$, sample variance $S^2$, sample proportion $\hat{p}$, sample median
  - Key idea: a statistic is a *random variable* (it depends on the random sample). A parameter is a fixed number. Confusing these two is the source of most beginner mistakes.
- **Estimand vs estimator vs estimate:** the target (θ), the recipe ($\bar{X}$), and the number you got (42.7)

- Discussion prompt: "Armenia's population is ~3 million. A polling agency surveys 1,000 people and reports '62% support policy X.' What's the population? The parameter? The sample? The statistic?"

### 0.3 The i.i.d. assumption

- Most of classical statistics assumes the sample is **independent and identically distributed**
  - Independent: knowing $X_1$ tells you nothing about $X_2$
  - Identically distributed: every $X_i$ comes from the same distribution $F$
- When does i.i.d. hold (approximately)?
  - Random sampling from a large population
  - Repeated independent measurements of the same quantity
  - Controlled experiments with proper randomization
- When does it break?
  - **Time dependence:** stock prices, weather, patient health over time
  - **Spatial correlation:** neighboring sensors, neighboring houses
  - **Selection bias:** surveying only people who answered the phone; studying only patients who came to the hospital
  - **Non-response bias:** people who refuse to answer may differ systematically
  - **Distribution shift:** the data you trained on comes from a different era/population than the data you'll apply to
- Not a disaster — just means you need different tools (time series, spatial stats, causal inference). But if you pretend non-i.i.d. data is i.i.d., your conclusions can be wildly wrong.

### 0.4 The plug-in principle

- The big idea: we don't know the true distribution $F$, so replace it with the empirical distribution $\hat{F}_n$
  - $\hat{F}_n$ puts mass $1/n$ on each observed data point
  - Want the population mean $\mu = \mathbb{E}_F[X]$? Plug in: $\hat{\mu} = \mathbb{E}_{\hat{F}}[X] = \bar{X}$
  - Want the population variance? Plug in: $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$
  - Want the population CDF $F(t) = P(X \le t)$? Plug in: $\hat{F}_n(t) = \frac{\#\{X_i \le t\}}{n}$
- This is surprisingly powerful and is the foundation of much of nonparametric statistics
- But it raises questions: how *good* is the plug-in? How far can $\hat{F}_n$ be from $F$? (Glivenko–Cantelli theorem says $\hat{F}_n \to F$ uniformly — connects to LLN from Module 20)

### 0.5 Loss, risk, and optimal summaries

- Suppose you must summarize the population with a single number $a$. How do you choose?
- Depends on what "error" means to you — this is formalized by a **loss function** $L(\theta, a)$:
  - **Squared-error loss:** $L(\theta, a) = (\theta - a)^2$
    - Optimal summary: the **mean** — penalizes large errors quadratically
  - **Absolute-error loss:** $L(\theta, a) = |\theta - a|$
    - Optimal summary: the **median** — robust to outliers
  - **0-1 loss:** $L(\theta, a) = \mathbf{1}[\theta \ne a]$
    - Optimal summary: the **mode** — the most common value
- **Risk** $= \mathbb{E}[L(\theta, \hat{\theta})]$ — the average loss over repeated samples
- **Empirical risk** = average loss on the data you have — this is what we can actually compute
- **Empirical Risk Minimization (ERM):** pick the estimator that minimizes empirical risk
  - This is the principle behind least squares, maximum likelihood, and most ML training procedures
- The choice of loss is not just math — it reflects values:
  - Medical dosing: overestimate vs underestimate a dose have very different consequences (asymmetric loss)
  - Predicting house prices: MAE vs MSE gives different answers when there are mansions in the data
  - Spam filter: false positive (blocking a real email) vs false negative (letting spam through)

- Exercise (in-class): "You're predicting tomorrow's temperature for an outdoor event. The caterer needs a number to plan. Would you report the mean or median of the forecast distribution? What if you're planning evacuation thresholds for a flood?"

### 0.6 The big picture: what statistics will give us

- We now know *what* we're trying to do: learn about θ from data
- The rest of the course answers four questions:
  1. **Estimation:** What's our best guess for θ? (Lectures 2–3)
  2. **Uncertainty:** How confident are we? (Lectures 4–5)
  3. **Decisions:** Is the effect real or noise? (Lectures 6–7)
  4. **Models:** How do multiple variables relate? (Lecture 8)
- Preview: we'll use all of these together in real applications (Lecture 9)

### 0.7 Practical / take-home exercise

- Given a dataset (e.g., city temperatures, exam scores, or household incomes):
  - Compute the sample mean, median, and mode
  - Compute empirical risk under squared-error, absolute-error, and 0-1 loss for each summary
  - Observe: the mean minimizes squared-error risk, the median minimizes absolute-error risk
  - Add one extreme outlier and repeat — watch the mean shift dramatically while the median barely moves
  - Discuss: "Which summary would you report, and why? Does the answer depend on context?"

> **Bridge:** We now have point summaries — but a single number hides a lot. How do we see the full picture?

---

## 1. Descriptive statistics and empirical distributions *(1 lecture)*

**Motivating question:** *You get a spreadsheet with 10,000 rows. What do you look at first — and what can fool you?*

### 1.1 Why descriptive statistics matter

- Any serious analysis starts by *looking at the data* — before any model, any test, any estimation
- Goals of descriptive statistics:
  - Summarize the center, spread, and shape of a distribution
  - Detect anomalies: outliers, missing data patterns, impossible values
  - Compare groups or time periods visually
  - Generate hypotheses (what looks interesting?) before testing them (is it real?)
- Descriptive ≠ inferential: we're describing *this sample*, not (yet) drawing conclusions about the population

### 1.2 Measures of center

- **Sample mean:** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$
  - Pro: uses all the data, is the MLE for μ under normality, minimizes squared error
  - Con: sensitive to outliers — a single extreme value can move it a lot
  - Example: mean income in a room with 9 teachers and 1 billionaire
- **Sample median:** the middle value (or average of two middle values)
  - Pro: robust — up to 50% of the data can be corrupted before it breaks (high breakdown point)
  - Con: ignores magnitude of values beyond the middle; less efficient than mean for normal data
- **Sample mode:** most frequent value (mainly useful for categorical data)
- **Trimmed mean:** discard the top and bottom k%, compute the mean of the rest — a compromise between mean and median
- Discussion: "Armenia's mean household income vs median household income — which is higher? Why? Which tells you more about a 'typical' family?"

### 1.3 Measures of spread

- **Sample variance:** $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$
  - Why $n-1$? (Bessel's correction — we'll prove this is unbiased in Lecture 3. For now: we used up one "degree of freedom" estimating the mean.)
  - **Sample standard deviation:** $S = \sqrt{S^2}$ — same units as the data
- **Range:** max − min — simple but extremely sensitive to outliers
- **Interquartile range (IQR):** $Q_3 - Q_1$ — the range of the middle 50%
  - Robust: outliers don't affect it (unless more than 25% of data is extreme)
- **Median Absolute Deviation (MAD):** median of $|X_i - \text{median}|$
  - Even more robust than IQR; a natural companion to the median
- **Coefficient of variation (CV):** $S / \bar{X}$ — dimensionless measure of relative spread
  - Useful for comparing variability across different scales (e.g., heights in cm vs weights in kg)

### 1.4 Quantiles, percentiles, and boxplots

- **Quantile** $q_p$: the value below which a fraction $p$ of the data falls
  - $q_{0.5}$ = median, $q_{0.25} = Q_1$, $q_{0.75} = Q_3$
  - Percentile = quantile × 100 (90th percentile = $q_{0.9}$)
- Interpolation: with finite data, quantiles aren't unique — different conventions exist (Python's `np.quantile` has 9 methods). Usually doesn't matter much for large n.
- **Boxplot anatomy:**
  - Box: $Q_1$ to $Q_3$ (the middle 50%)
  - Line inside: median
  - Whiskers: typically extend to the most extreme point within $1.5 \times IQR$ from the box
  - Points beyond whiskers: potential outliers (flagged, not necessarily wrong)
- **When boxplots shine:** comparing distributions across groups side by side (salaries by department, scores by school, measurements by lab)
- **When boxplots mislead:** bimodal distributions look unimodal in a boxplot — always pair with a histogram or violin plot

- Applications beyond statistics:
  - Finance: Value at Risk (VaR) = a quantile of the loss distribution ("What's the worst 5% scenario?")
  - Medicine: growth charts for children (3rd, 50th, 97th percentile)
  - Education: standardized test scores reported as percentiles

### 1.5 Shape: skewness and kurtosis

- **Skewness:** measures asymmetry
  - $\text{Skew} = \frac{1}{n}\sum\left(\frac{X_i - \bar{X}}{S}\right)^3$
  - Positive skew: long right tail (income, house prices, file sizes)
  - Negative skew: long left tail (exam scores with a hard ceiling, age at retirement)
  - Zero skew: symmetric (not necessarily normal — uniform is symmetric too)
- **Kurtosis:** measures tail heaviness (not "peakedness" — a common misconception)
  - Normal distribution has kurtosis = 3 (or excess kurtosis = 0)
  - High kurtosis → more extreme outliers than you'd expect from a normal
  - Why it matters: financial returns have high kurtosis — assuming normality underestimates risk (this contributed to the 2008 financial crisis)
- These are useful descriptors but fragile with small samples — interpret cautiously

### 1.6 The empirical CDF

- **Definition:** $\hat{F}_n(t) = \frac{1}{n}\#\{X_i \le t\}$ — the fraction of observations at or below $t$
  - A step function that jumps by $1/n$ at each data point
- **Properties:**
  - Non-decreasing, right-continuous, goes from 0 to 1
  - $\hat{F}_n(t) \to F(t)$ for every $t$ as $n \to \infty$ (Glivenko–Cantelli — the "fundamental theorem of statistics")
  - At any point $t$: $\hat{F}_n(t) \sim \text{Binomial}(n, F(t))/n$ — we can put error bars on it
- **Visual power:** plot two ECDFs to compare distributions — more informative than histograms for comparison because it's parameter-free (no bin width choice)
- **Kolmogorov–Smirnov statistic (intuition only):** $D_n = \sup_t |\hat{F}_n(t) - F_0(t)|$
  - "What's the biggest gap between the empirical and theoretical CDF?"
  - Small $D_n$ → data is consistent with $F_0$; large $D_n$ → something differs
  - We'll formalize this as a test in Lecture 7; for now it's a visual tool

### 1.7 Summary statistics can lie — always plot your data

- **Anscombe's quartet (1973):** four datasets with nearly identical summary statistics (mean, variance, correlation, regression line) but wildly different scatter plots
  - One is linear, one is curved, one has an outlier, one is clustered with one extreme point
  - Moral: never trust a number without a picture
- **Datasaurus Dozen (2017):** same idea, pushed to the extreme — 13 datasets (including a dinosaur) all with the same means, SDs, and correlation
- **Simpson's paradox:** an aggregate trend can reverse when you split by a subgroup
  - Classic example: UC Berkeley admissions appeared to discriminate against women overall, but favored women within each department — women were applying to more competitive departments
  - This is not just a curiosity — it changes real decisions in medicine, policy, and science
  - Connection to Lecture 8: controlling for confounders in regression is how we handle this

### 1.8 Practical / take-home exercise

- **EDA on a real dataset** (suggested: a dataset with known quirks — Titanic, Palmer Penguins, or a messy real-world CSV):
  - Compute mean, median, SD, IQR, skewness for each numeric variable
  - Make histograms — identify skewed variables, potential outliers, multimodal distributions
  - Make boxplots by group (e.g., survival by class, body mass by species)
  - Plot ECDFs for two subgroups on the same axes — where do they differ most?
  - Find at least one case where a summary statistic is misleading and a plot reveals the truth
  - Bonus: construct your own Anscombe-style pair — two tiny datasets with the same mean and variance but different shapes

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
