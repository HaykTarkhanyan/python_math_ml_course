# Optimization — Math Reference

## Homework / Quarto modules (`math/`)

- [08_optim_univar.qmd](../../math/08_optim_univar.qmd) — **Univariate optimization** — Golden section search (GSS), bracketing, Brent's method, unimodality.
- [09_optim_prereq__gradient_descent.qmd](../../math/09_optim_prereq__gradient_descent.qmd) — **Prereqs + gradient descent** — Quadratic forms, positive definiteness, condition number, vanilla GD, learning rate, convergence.
- [10_optim_momentum_first_order_algs.qmd](../../math/10_optim_momentum_first_order_algs.qmd) — **Momentum + first-order** — Heavy-ball momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW.
- [11_optim_second_order.qmd](../../math/11_optim_second_order.qmd) — **Second-order methods** — Newton's method, Quasi-Newton (BFGS, L-BFGS), trust regions, line search.
- [12_derivative_free.qmd](../../math/12_derivative_free.qmd) — **Derivative-free** — Nelder-Mead simplex, pattern search, when gradients don't exist or are noisy.
- [13_evolutionary.qmd](../../math/13_evolutionary.qmd) — **Evolutionary algorithms** — GA basics (selection, crossover, mutation), differential evolution, CMA-ES.
- [14_bayesian.qmd](../../math/14_bayesian.qmd) — **Bayesian optimization** — Surrogate models (GPs), acquisition functions (EI, UCB), warm-starting; relevant for HP search.
- [15_multicriteria_optimization.qmd](../../math/15_multicriteria_optimization.qmd) — **Multi-objective** — Pareto fronts, scalarization, dominated solutions, NSGA-II.

## Lecture decks with `.tex` source (`math/Lectures/optim/`)

- [01_univariate.tex](../../math/Lectures/optim/01_univariate.tex) — Univariate optim (matches module 08).
- [02_prereqs.tex](../../math/Lectures/optim/02_prereqs.tex) — Multivariate prereqs (matches module 09 first half). Companion notebook: `02_common_functions.ipynb`.
- [03_gd_step_size.tex](../../math/Lectures/optim/03_gd_step_size.tex) — GD step size (matches module 09 second half). Companion notebooks: `03_gd_step_in_class.ipynb`, `03_gd_step_size.ipynb`.
- [04_momentum_adam.tex](../../math/Lectures/optim/04_momentum_adam.tex) — Momentum, Adam, AdamW (matches module 10). Companion notebook: `04_adam_in_class.ipynb` (Armenian: `04_adamy_ev_xarebnery.ipynb`).
- [05_derivative_free.tex](../../math/Lectures/optim/05_derivative_free.tex) — Derivative-free (matches module 12). Companion notebooks: `05_deriv_free_in_class.ipynb`, `05_derivative_free_helper.ipynb`.
- [06_evolutionary.tex](../../math/Lectures/optim/06_evolutionary.tex) — Evolutionary (matches module 13). Companion notebooks: `06_genetic_alg_in_class.ipynb`, `06_genetic_algorithm.ipynb`.

Companion benchmark notebook: `benchmark_first_order_methods.ipynb`. Upstream slide source: `math/Lectures/optim/slides_lmu/`.

## Lecture coverage notes

- Modules 11 (second-order), 14 (Bayesian), 15 (multicriteria) — **no dedicated lecture deck yet**. Modules exist as qmds only; if delivering this material, build from the qmd content or upstream slides in `slides_lmu/`.
- Module 11 (second-order) was discussed informally per `course-completion-status.md` memory but not formally lectured.

## Homework write-ups (`math/Homeworks/`)

- `hw_07_optim_univar.pdf` — module 08

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L02 gradient descent recap | modules 09 + optim L02 + L03 |
| L02 learning rate sensitivity demo | optim L03 (notebooks demo this directly) |
| L02 / NN training step (SGD) | module 09 |
| NN optimizer chapter (Adam etc.) | module 10 + optim L04 |
| L05 hyperparameter tuning (random search, grid) | modules 14 (Bayesian opt is the formal version), 13 (evolutionary HP search) |
| Bayesian HP optimization (Optuna etc.) | module 14 |
| Neural Architecture Search (NAS) | modules 13, 14 |
| Multi-objective trade-offs (accuracy vs latency vs cost) | module 15 |
| Convex optimization claims in regularization lectures | module 09 (PSD, condition number) |

## Notes

- The `.tex` decks under `Lectures/optim/` are the maintained source; the `.pdf` versions are compiled artefacts. Edit the tex.
- `benchmark_first_order_methods.ipynb` shows GD vs momentum vs Adam side-by-side on standard test functions (Rosenbrock, Beale, etc.) — use directly in ML decks rather than rebuilding visuals.
- Upstream LMU slides (`slides_lmu/`) are the original source for some of the optim decks. Cite when reusing.
- All optim notebooks are bilingual class notebooks; `*_in_class.ipynb` are the empty versions for live coding.
