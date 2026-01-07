0. Foundations: statistics mindset for ML

Population vs sample; i.i.d. assumption and violations

Parameters (estimands) vs estimators; plug-in principle

Loss functions, risk, empirical risk minimization (ERM)

Mean / median / mode as risk minimizers under different losses

1. Descriptive summaries and empirical distributions

Sample moments; robust summaries

Quantiles, percentiles; boxplots and outliers

Empirical CDF; basic concentration intuition (optional)

2. Point estimation and likelihood

Method of moments (quick intuition + limitations)

Likelihood and log-likelihood; score function

Maximum Likelihood Estimation (MLE)

Invariance, constraints/reparameterization

Identifiability and common failure modes

Maximum A Posteriori (MAP) estimation

Priors, posteriors, posterior mode

Regularization as MAP (ridge ↔ Gaussian prior, lasso ↔ Laplace prior)

3. Estimator quality and optimality

Bias, variance, MSE; bias–variance decomposition

Consistency and asymptotic notions (high-level)

Efficiency and the Cramér–Rao bound (when assumptions hold)

Sufficient statistics (optional, as a unifying concept)

4. Uncertainty quantification and asymptotics

Sampling distributions and standard errors

CLT-based approximations; delta method (as needed)

Fisher information

Expected vs observed information; curvature intuition

Asymptotic normality of MLE; Wald intervals

Likelihood-based intervals (profile likelihood) (optional)

Confidence intervals vs credible intervals (conceptual contrast)

5. Resampling methods

Bootstrap for standard errors and confidence intervals

Percentile / basic (BCa optional)

When bootstrap fails (dependence, heavy tails, small n pathologies)

Permutation (randomization) tests

Cross-validation vs bootstrap (evaluation vs inference)

6. Hypothesis testing framework

Null vs alternative; test statistics; rejection regions

Type I / Type II errors; power; effect size; sample size planning

p-values: correct interpretation and common fallacies

Multiple testing and selection effects

Bonferroni / Holm

Benjamini–Hochberg (FDR)

7. Core classical tests as instances

z-test and t-test (one-sample, paired, two-sample)

Assumptions; Welch correction; nonparametric alternatives (optional)

Chi-squared tests

Goodness-of-fit

Independence in contingency tables

Likelihood Ratio Tests (LRT) (recommended as a unifying tool)

8. Linear models as a unifying inference engine

Simple linear regression: estimation, inference, diagnostics

Multiple regression; dummy variables; interactions

ANOVA as regression hypothesis testing (model comparison view)

Prediction intervals vs confidence intervals

Logistic regression as MLE/MAP (optional, time permitting)

9. ML-facing integration (capstone-style)

Regularization, bias–variance, and generalization tradeoffs

Uncertainty for metrics (accuracy/AUC) via bootstrap

Significance of model improvements via paired tests/permutation tests

Practical experimental design for ML (A/B testing patterns, guardrails)