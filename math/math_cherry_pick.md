# Math Cherry-Pick for ML

A focused subset of the course's math topics, picked for someone whose goal is machine learning rather than a full math degree. Course chapter order is pedagogical (prereqs first). This list is utility-first: what you'll actually hit when training models, reading papers, or debugging.

Site root: <https://hayktarkhanyan.github.io/python_math_ml_course/>

---

## Tier 1 - Non-negotiable (you'll use this every week)

These topics show up the moment you open PyTorch, scikit-learn, or any ML paper. If you skip these you'll be guessing at hyperparameters and copying training loops you don't understand.

- **Vectors and dot products** - the geometry of features, similarity, projections
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/01_linear_algebra_vectors.html>
- **Matrices** - linear layers, batch operations, weight tensors
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/02_linear_algebra_matrices.html>
- **Eigenvalues, eigendecomposition, SVD** - PCA, spectral methods, low-rank approximations (LoRA), conditioning
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/03_linear_algebra_concepts.html>
- **Derivatives, limits, continuity** - gradients are derivatives, full stop
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/04_calc_lim_continuity_deriv.html>
- **Extrema, convexity, Taylor series** - why convex losses are nice, why second-order methods exist
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/05_calc_extrema_convexity_taylor.html>
- **Multivariate calculus (gradients, Jacobians, Hessians)** - backpropagation lives here
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/07_calc_multivar.html>
- **Gradient descent** - the algorithm under every neural net
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/09_optim_prereq__gradient_descent.html>
- **Momentum, Adam, first-order optimizers** - what your `torch.optim.Adam(...)` call actually does
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/10_optim_momentum_first_order_algs.html>
- **Probability basics (events, conditional probability, Bayes)** - read any paper, you'll see this
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/16_probability_intro.html>
- **Expectation, variance, inequalities** - loss functions are expectations, regularization is variance control
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/17_probability_exp_var_inequalities.html>
- **Distributions (Gaussian, Bernoulli, Categorical, etc.)** - generative models, likelihoods, noise assumptions
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/19_probability_distributions.html>
- **MLE and MAP** - cross-entropy = negative log-likelihood, L2 regularization = Gaussian prior. This is the bridge from probability to loss functions.
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/23_stat_mle_map.html>
- **Entropy and KL divergence** - cross-entropy loss, VAEs, RLHF, knowledge distillation
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/28_info_theory_entropy_kl.html>
- **Information theory in ML** - direct application: cross-entropy, mutual information, ELBO
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/29_info_theory_ml.html>

## Tier 2 - Strongly recommended (you'll hit this within your first year)

Useful for understanding why ML works, debugging models that go sideways, and reading slightly older or more theoretical papers.

- **Univariate optimization** - intuition for line search, learning rate schedules
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/08_optim_univar.html>
- **Correlation and covariance** - feature engineering, PCA, Gaussian processes
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/18_probability_corr_cov.html>
- **Convergence modes, LLN, CLT** - why empirical risk minimization works, bootstrap intuition
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/20_probability_convergence_modes_lln_clt.html>
- **Estimators (bias, variance, consistency)** - the bias-variance tradeoff is a statistics concept, not an ML one
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/22_stat_estimators.html>
- **Curse of dimensionality** - why kNN dies in high dimensions, why embeddings need to be learned
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/30_curse_of_dimensionality.html>
- **How to lie with statistics** - critical reading skill for ML papers, benchmarks, and product metrics
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/27_stat_how_to_lie.html>

## Tier 3 - Useful but specialized

Worth knowing exists. Read when the use case arises rather than upfront.

- **Sets, combinatorics, functions** - prereq refresher, skip if comfortable
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/00_sets_comb_funcs.html>
- **Integrals** - mostly handled by frameworks; needed for continuous probability proofs
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/06_calc_integrals.html>
- **Second-order optimization (Newton, L-BFGS)** - rare in deep learning, common in classical ML and scientific computing
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/11_optim_second_order.html>
- **Bayesian optimization** - hyperparameter tuning, AutoML
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/14_bayesian.html>
- **Statistics fundamentals (samples, sampling distributions)** - foundation for the estimators chapter
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/21_stat_fundamentals.html>
- **Confidence intervals** - reporting model performance, A/B test results
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/24_stat_confidence_intervals.html>
- **Hypothesis testing** - A/B tests, model comparison
  <https://hayktarkhanyan.github.io/python_math_ml_course/math/25_stat_hypothesis_testing.html>

## Tier 4 - Skip unless you have a specific reason

Important math, but rarely the thing standing between you and a working ML model.

- **Derivative-free optimization** - <https://hayktarkhanyan.github.io/python_math_ml_course/math/12_derivative_free.html>
- **Evolutionary algorithms** - <https://hayktarkhanyan.github.io/python_math_ml_course/math/13_evolutionary.html>
- **Multicriteria optimization** - <https://hayktarkhanyan.github.io/python_math_ml_course/math/15_multicriteria_optimization.html>
- **Classical hypothesis tests (t-test, chi-square, ANOVA)** - <https://hayktarkhanyan.github.io/python_math_ml_course/math/26_stat_classical_tests.html>

---

## Suggested minimum path (about 15 chapters)

If you only have time for the bare minimum, do these in order:

1. `01_linear_algebra_vectors`
2. `02_linear_algebra_matrices`
3. `03_linear_algebra_concepts` (eigen, SVD)
4. `04_calc_lim_continuity_deriv`
5. `05_calc_extrema_convexity_taylor`
6. `07_calc_multivar`
7. `09_optim_prereq__gradient_descent`
8. `10_optim_momentum_first_order_algs`
9. `16_probability_intro`
10. `17_probability_exp_var_inequalities`
11. `19_probability_distributions`
12. `22_stat_estimators`
13. `23_stat_mle_map`
14. `28_info_theory_entropy_kl`
15. `29_info_theory_ml`

That's the spine. Everything else is depth.
