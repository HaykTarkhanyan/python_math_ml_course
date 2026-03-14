# Statistics Lectures — Comprehensive Audit Report

## Overview

Audit of 15 statistics lecture files (`01_stat.tex` through `15_stat.tex`) covering:
1. Mistakes (mathematical errors, typos, inconsistencies)
2. Coverage improvements
3. Pedagogical improvements
4. Homework assignments

---

## 1. Mistakes & Inconsistencies

### 1.1 Cross-Reference Errors (HIGH PRIORITY)

**L4 (`04_stat.tex`, lines 946–951): "What We Haven't Covered" section has wrong lecture numbers.**

| Listed in L4 | Actual |
|---|---|
| "Computational methods: EM algorithm, MCMC (Lectures 7–8)" | EM/MCMC are **not covered** in the course at all. L7 = Sampling Distributions, L8 = CIs & Bootstrap |
| "Confidence intervals (Lecture 9)" | CIs are in **Lecture 8** |
| "Bootstrap (Lecture 10)" | Bootstrap is in **Lecture 8** |
| "Hypothesis testing (Lectures 11–12)" | Hypothesis testing is **Lectures 9–10** |

**Fix:** Rewrite the entire "What We Haven't Covered" section to match the actual curriculum.

**L5 (`05_stat.tex`, line 1103): Wrong lecture reference.**
- Says: `"More in Lecture~7 (Confidence Intervals)"`
- Should be: `"More in Lecture~8 (Confidence Intervals)"`

**L6 (`06_stat.tex`, line 1221): Inaccurate "Next" slide.**
- Says: `"Next: Lecture~7 --- Sampling distributions and confidence intervals"`
- Should be: `"Next: Lecture~7 --- Sampling distributions"`
- (CIs are a separate lecture, L8)

### 1.2 Misleading Subtitle

**L3 (`03_stat.tex`, line 30): Subtitle includes "Cramér–Rao" but L3 doesn't cover it.**
- Subtitle: `Bias · Variance · MSE · Consistency · Sufficiency · Cramér–Rao`
- L3 only foreshadows CR ("coming soon") — the actual CR content is in L4.
- **Fix:** Remove `Cramér–Rao` from L3 subtitle, or replace with `Exponential Family`.

### 1.3 Missing Homework

**L1 (`01_stat.tex`): The only lecture with no homework slide.**
- All other lectures (L2–L14) have homework slides.
- L1 covers foundational concepts (population vs sample, plug-in principle, loss functions, ERM) that warrant practice problems.
- **Fix:** Add a homework slide to L1 (see Section 4 below).

### 1.4 Minor LaTeX Issues

- **`\usepackage{fontenc}` without encoding option** (all 15 files): Loads OT1 (the default anyway). Harmless but redundant. Should be `\usepackage[T1]{fontenc}` or removed entirely.
- **`\usepackage{pifont}` loaded but `\ding{}` not used** in L10, L11, L13: Unnecessary package import.
- **L5 line 884:** References "Cramér–Rao bound from Lectures~3–4" — CR bound is only in L4, not L3. Minor but imprecise. Should say "Lecture~4".

### 1.5 Mathematical Verification

All major formulas were verified and found **correct**:
- MSE = Bias² + Variance decomposition (L3)
- Bessel's correction derivation (L3)
- MSE comparison: $\hat\sigma^2_n$ vs $S^2$ — claim that biased estimator has lower MSE for all $n \geq 2$ (L3) ✓
- Cramér–Rao bound: $\text{Var}(\hat\theta) \geq 1/(nI(\theta))$ (L4) ✓
- F-statistic: $F = \text{MS}_B/\text{MS}_W \sim F_{k-1, N-k}$ (L11) ✓
- SS decomposition: $\text{SS}_T = \text{SS}_B + \text{SS}_W$ with correct df (L11) ✓
- p-value definition includes "assuming $H_0$ is true" (L9) ✓
- Deviance definition and AIC = D + 2p (L13) ✓
- ANOVA formulas (L11) ✓
- Regression inference formulas (L12) ✓
- GLM three-component framework (L13) ✓
- CR bound notation is consistent across L4, L5, L7, L15 ✓

---

## 2. Coverage Improvements

### 2.1 Missing Topics (Recommended Additions)

| Topic | Priority | Suggested Location | Notes |
|---|---|---|---|
| **EM Algorithm** | Medium | After L6 or as L6 appendix | L4 mentions it in "What We Haven't Covered" but it's never taught. Important for mixture models. |
| **Cross-validation** | Medium | L12 or L13 | Critical for model selection in practice, only AIC is mentioned for GLMs |
| **Robust statistics** | Low | L2 extension | Trimmed mean, Huber estimators — briefly mentioned (MAD) but not developed |
| **Survival analysis** | Low | After L13 | Common in applied stats, not covered |
| **Mixed-effects models** | Low | After L13 | Important for hierarchical data |
| **Multivariate statistics** | Low | Separate lecture | PCA, multivariate normal — referenced via Stein's paradox but not taught |

### 2.2 Topics Covered Well

The curriculum excels at:
- **Exponential family thread** (L3 → L5 → L6 → L13): Consistently built up across lectures
- **MLE thread** (L5 → L7 → L8 → L10 → L12 → L13): Unified framework
- **Practical applications**: A/B testing (L11), real-world examples throughout
- **Bootstrap** (L8): Thorough treatment with BCa
- **Causal inference** (L14): Modern and relevant capstone

### 2.3 Coverage Depth Adjustments

| Lecture | Current Depth | Suggestion |
|---|---|---|
| L4 (Stein's paradox) | Detailed | Could be trimmed — interesting but rarely used in practice; space could go to more CR examples |
| L6 (Empirical Bayes) | Brief mention | Could expand — increasingly important in practice (genomics, A/B testing) |
| L8 (BCa bootstrap) | Detailed | Good depth for a course that emphasizes computational methods |
| L13 (IRLS) | Algorithm steps | Consider adding a worked iteration example |
| L14 (IV) | Brief | Could expand — instrumental variables are foundational for econometrics |

---

## 3. Pedagogical Improvements

### 3.1 Lecture Sequencing

The current sequencing is **generally excellent**. The flow from foundations → estimation theory → estimation methods → uncertainty quantification → testing → modeling → causation is logical and well-motivated.

**One potential improvement:** L7 (Sampling Distributions) could come earlier, perhaps before L5 (MLE). Understanding that estimators have distributions is conceptually prior to studying MLE properties. However, the current ordering (properties → MLE → sampling distributions) also works because students need MLE examples to make sampling distributions concrete. **Recommendation: Keep current order** but add a brief motivating example of sampling distributions at the end of L3.

### 3.2 Worked Examples

| Lecture | Worked Examples | Assessment |
|---|---|---|
| L1 | Income distribution, dice | Good |
| L2 | Anscombe's quartet, Simpson's | Excellent |
| L3 | Dartboard analogy, Normal variance | Good |
| L4 | Coin flip Fisher info | Good, but could add more |
| L5 | Bernoulli, Normal (full), Poisson, Exp | Excellent |
| L6 | Beta-Binomial, Normal-Normal | Good |
| L7 | Normal, Uniform, Exponential CLT demos | Excellent |
| L8 | Lightbulb CI, bootstrap visualization | Excellent |
| L9 | Drug trial running example | Excellent |
| L10 | Coffee shop, blood pressure, die fairness | Excellent |
| L11 | Fertilizer ANOVA, button color A/B | Good |
| L12 | House prices | Good |
| L13 | Bike rentals Poisson | Good, could add more |
| L14 | Ice cream/drowning, surgery | Good |

### 3.3 Difficulty Ramps

- **L3 → L4 is the steepest jump** in the course (Rao-Blackwell → Fisher info → CR → admissibility → Stein's paradox). Consider slowing down by splitting across two sessions.
- **L9 → L10 → L11** is well-paced with increasing complexity.
- **L12 → L13** is a natural progression from specific (OLS, logistic) to general (GLM framework).

### 3.4 Recap Effectiveness

All lectures L3+ have recap slides referencing prior lectures. These are effective and consistently formatted. The recap slides correctly reference the right prior lectures (unlike the L4 "What We Haven't Covered" section).

### 3.5 Suggested Pedagogical Additions

1. **"Why should I care?" opening** for L3 and L4 — these are the most theoretical lectures and would benefit from a motivating real-world hook at the start (similar to L9's drug trial framing).

2. **More "intuition first, formula second" approach** in L4 (Fisher Information). The concept of "information = curvature of log-likelihood" deserves a visual before the formal definition.

3. **Comparative summary tables** at the end of each "block" (e.g., end of L6 comparing MoM vs MLE vs MAP; end of L10 comparing all test types). Some exist but not systematically.

4. **Python code snippets** in L3 and L4 — most other lectures have them, but these theoretical lectures lack concrete code examples.

5. **"Common mistakes" boxes** throughout — L15 has a great "Top Mistakes" slide; distributing these warnings into individual lectures would be more effective.

---

## 4. Homework Assignments

### L1: Foundations (NEW — currently missing)

1. **Population vs sample:** A factory claims its light bulbs last 1000 hours on average ($\mu = 1000$). You test 50 bulbs and find $\bar{X} = 980$. Identify: (a) the population, (b) the sample, (c) the parameter, (d) the statistic, (e) the estimator, (f) the estimate.

2. **Plug-in principle:** For data $X = \{2, 3, 5, 7, 8\}$, compute the plug-in estimates of (a) the mean, (b) the variance (using $\hat{F}_n$), (c) the median, (d) the 0.75 quantile. Compare (b) with Bessel-corrected $S^2$.

3. **Loss functions:** For the dataset $\{1, 2, 3, 4, 100\}$: (a) Find the value $c$ that minimizes the empirical risk under $L_2$ loss. (b) Find the value that minimizes $L_1$ loss. (c) Explain which loss is more appropriate here and why.

### L2: Descriptive Statistics (EXISTING — 3 tasks, consider adding)

4. **(Optional addition)** Create ECDF plots for two datasets and use the Kolmogorov-Smirnov idea (max vertical distance) to argue whether they come from the same distribution.

### L3: Properties of Estimators (EXISTING — 4 tasks, well-designed)

No changes needed.

### L4: Fisher Information & CR (EXISTING — 3 tasks, consider adding)

4. **(Optional addition)** For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known: compute $I(\mu)$, find the CR bound, and show that $\bar{X}$ is efficient.

### L5: Point Estimation (EXISTING — 4 tasks, well-designed)

No changes needed.

### L6: MAP & Bayesian (EXISTING — 3 tasks, consider adding)

4. **(Optional addition)** Implement sequential Bayesian updating: start with Beta(1,1) prior, observe coin flips one at a time, plot how the posterior evolves. Compare the MAP estimate after 10, 50, and 200 flips.

### L7: Sampling Distributions (EXISTING — 4 tasks, well-designed)

No changes needed.

### L8: CIs & Bootstrap (EXISTING — 4 tasks, well-designed)

No changes needed.

### L9: Hypothesis Testing (EXISTING — 4 tasks, well-designed)

No changes needed.

### L10: Classical Tests & LRT (EXISTING — 4 tasks, well-designed)

No changes needed.

### L11: ANOVA & A/B Testing (EXISTING — 4 tasks, well-designed)

No changes needed.

### L12: Regression Inference (EXISTING — 4 tasks, well-designed)

No changes needed.

### L13: GLMs (EXISTING — 4 tasks, well-designed)

No changes needed.

### L14: Causal Inference (EXISTING — 4 tasks, well-designed)

No changes needed.

---

## 5. Summary of Action Items

### Must Fix (Errors)
1. [ ] **L4 lines 946–951:** Fix "What We Haven't Covered" lecture numbers to match actual curriculum
2. [ ] **L5 line 1103:** Change "Lecture~7" to "Lecture~8" for CIs reference
3. [ ] **L6 line 1221:** Change "Sampling distributions and confidence intervals" to "Sampling distributions"

### Should Fix (Improvements)
4. [ ] **L3 line 30:** Remove "Cramér–Rao" from subtitle (it's an L4 topic)
5. [ ] **L1:** Add homework slide (3 problems, see Section 4)
6. [ ] **L5 line 884:** Change "Lectures~3–4" to "Lecture~4" for CR bound reference

### Nice to Have
7. [ ] Fix `\usepackage{fontenc}` → `\usepackage[T1]{fontenc}` across all files
8. [ ] Remove unused `\usepackage{pifont}` from L10, L11, L13
9. [ ] Add Python code examples to L3 and L4
10. [ ] Add "Why should I care?" opening hooks to L3 and L4
