---
name: Full Project Structure
description: Complete directory layout of the math/ML/Python course repo with all key paths and file counts
type: project
---

## Root
`c:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course\`

## Top-Level Structure
```
├── python/              # 18 Python modules (01-18) + template + mini_projects/
├── python_libs/         # 18 library modules (NumPy, Pandas, Matplotlib, etc.)
├── math/
│   ├── 00-25_*.qmd      # 26 math/prob/stat homework modules
│   ├── Lectures/
│   │   ├── stat/         # 16 stat presentations (01-16_stat.tex) + plan + audit
│   │   └── optim/        # 6 optimization presentations + ~10 Jupyter notebooks
│   ├── Homeworks/        # ~16 PDF handouts + Xournal++ (.xopp) source files
│   └── assets/           # Images, knn.csv, 2 Jupyter notebooks
├── ml/                   # 6 chapter dirs + Datasets/ + Projects/ (mostly skeleton)
├── misc/                 # Google Colab guide, dl4nlp, prep materials
├── docs/                 # Rendered Quarto output → GitHub Pages
├── ma/                   # Isolated uv/Python project (gensim, ipykernel)
├── _extensions/          # Custom Quarto extensions
├── bibliography/         # BibTeX references
├── background_photos/    # Presentation photos
├── assets/               # Root-level logos/images
├── memes/                # Fun content
├── _quarto.yml           # Quarto book config (cosmo/darkly theme, 25+ chapters)
├── Makefile              # `make push` — render changed files + git push
├── render_only_changed.py # Smart incremental Quarto rendering
├── index.qmd             # Landing page
└── README.md             # Course overview
```

## Statistics Lectures (Primary Work Area)
```
math/Lectures/stat/
├── stat.md              # Full curriculum plan (10 sections, ~500 lines)
├── 00_plan.md           # Session-by-session outline with homework descriptions
├── AUDIT_REPORT.md      # Comprehensive audit with findings (all major items fixed)
├── 01_stat.tex (25 frames) — Foundations: prob vs stat, population/sample, i.i.d., ERM
├── 02_stat.tex (23 frames) — Descriptive stats, ECDF, Anscombe, Simpson
├── 03_stat.tex (34 frames) — Estimator properties: bias, variance, MSE, sufficiency
├── 04_stat.tex (29 frames) — Fisher info, Cramér-Rao, efficiency, admissibility
├── 05_stat.tex (42 frames) — Point estimation: MoM, MLE
├── 06_stat.tex (39 frames) — MAP estimation, conjugacy, regularization connection
├── 07_stat.tex (39 frames) — Sampling distributions, CLT, standard errors
├── 08_stat.tex (33 frames) — Confidence intervals & bootstrap
├── 09_stat.tex (43 frames) — Hypothesis testing, p-values, power, multiple testing
├── 10_stat.tex (28 frames) — Classical tests & LRT, decision flowchart
├── 11_stat.tex (34 frames) — ANOVA & A/B testing
├── 12_stat.tex (31 frames) — Regression inference, diagnostics
├── 13_stat.tex (29 frames) — GLMs: exponential family, link functions, Poisson
├── 14_stat.tex (37 frames) — Causal inference: DAGs, potential outcomes, propensity
├── 15_stat.tex (17 frames) — Course recap
├── 16_stat.tex (67 frames) — How to Lie with Statistics
└── Total: 541 frames across 16 lectures
```

## Optimization Lectures
```
math/Lectures/optim/
├── 01_univariate.tex/.pdf   — Golden section search, Brent's method
├── 02_prereqs.tex/.pdf      — Optimization prerequisites
├── 03_gd_step_size.tex/.pdf — Gradient descent step size
├── 04_momentum_adam.tex     — Adam, momentum methods
├── 05_derivative_free.tex/.pdf — Derivative-free optimization
├── 06_evolutionary.tex      — Evolutionary algorithms
└── ~10 Jupyter notebooks    — In-class demos (GSS, GD, Adam, genetic algorithms)
```

## Cross-References (Stat Lectures ↔ Math Modules)
- ERM, loss functions ← Module 05 (extrema, convexity)
- Regularization as MAP ← Modules 01 (norms), 05
- MLE for distributions ← Module 19
- Sampling distributions, CLT ← Module 20
- Fisher info, curvature ← Module 07 (Hessians)
- Bias-variance ← Module 17
- Normal equation, OLS ← Module 03
- Bayes' theorem ← Module 16
- Covariance in regression ← Module 18
