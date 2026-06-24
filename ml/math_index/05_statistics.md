# Statistics — Math Reference

This section is the most lecture-heavy: 12 stat decks with TeX source, all topic-annotated in the canonical outline at [`math/Lectures/stat/00_lecture_outline.md`](../../math/Lectures/stat/00_lecture_outline.md). When in doubt, **read that outline first** — it lists topics, frame counts, homework breakdowns, and cross-references. The section below is the quick map.

## Homework / Quarto modules (`math/`)

- [21_stat_fundamentals.qmd](../../math/21_stat_fundamentals.qmd) — **Foundations & descriptive stats** — Population vs sample, parameter vs statistic, descriptive stats, ECDF, boxplots, Anscombe's quartet, Simpson's paradox.
- [22_stat_estimators.qmd](../../math/22_stat_estimators.qmd) — **Estimator properties** — Bias, variance, MSE, consistency, sufficiency, Fisher info, Cramér-Rao. (Solutions in [22_stat_estimators_solutions.md](../../math/22_stat_estimators_solutions.md).)
- [23_stat_mle_map.qmd](../../math/23_stat_mle_map.qmd) — **MLE & MAP** — MoM, MLE recipe, examples (Bernoulli, Normal, Poisson, Exponential), MAP, conjugate priors, regularization = MAP.
- [24_stat_confidence_intervals.qmd](../../math/24_stat_confidence_intervals.qmd) — **Sampling distributions, CIs, bootstrap** — Sampling distribution, SE vs SD, Wald / Wilson / t CIs, bootstrap (normal / percentile / BCa).
- [25_stat_hypothesis_testing.qmd](../../math/25_stat_hypothesis_testing.qmd) — **Hypothesis testing** — H0/H1, p-values, Type I/II, power, permutation tests, multiple testing (Bonferroni, BH).
- [26_stat_classical_tests.qmd](../../math/26_stat_classical_tests.qmd) — **Classical tests, ANOVA, A/B testing** — t-tests (one-sample, paired, two-sample, Welch's), chi-squared (GoF, independence), Mann-Whitney, Wilcoxon, one-way ANOVA, post-hoc, A/B testing pitfalls.
- [27_stat_how_to_lie.qmd](../../math/27_stat_how_to_lie.qmd) — **How to lie with statistics** — Misleading charts, cherry-picking, p-hacking, survivorship bias, base-rate fallacy, presentation tricks.

## Lecture decks with TeX source (`math/Lectures/stat/`)

All decks have `.tex` source and a `_notes.pdf` variant with speaker notes baked in. Frame counts are from the outline.

- [01_stat.tex](../../math/Lectures/stat/01_stat.tex) — **Foundations** (24 frames) — Population/sample, ERM, loss, plug-in principle, survivorship bias. Pairs with module 21.
- [02_stat.tex](../../math/Lectures/stat/02_stat.tex) — **Descriptive stats** (23 frames) — Center, spread, histograms, quantiles, skewness/kurtosis, ECDF, Datasaurus, Simpson's paradox. Pairs with module 21.
- [03_stat.tex](../../math/Lectures/stat/03_stat.tex) — **Estimator properties** (34 frames) — Bias, variance, MSE = Bias² + Var, dartboard, bias-variance tradeoff, consistency, sufficiency (Fisher-Neyman), exponential family. Pairs with module 22.
- [04_stat.tex](../../math/Lectures/stat/04_stat.tex) — **Fisher info + Cramér-Rao** (26 frames) — Score function, Fisher info (two forms), CR bound, regularity, admissibility, Stein's paradox. Pairs with module 22.
- [05_stat.tex](../../math/Lectures/stat/05_stat.tex) — **MoM + MLE** (35 frames) — Includes MLE = least squares, MLE = cross-entropy (logistic), when MLE fails, numerical MLE. Pairs with module 23.
- [06_stat.tex](../../math/Lectures/stat/06_stat.tex) — **MAP + Bayesian** (25 frames) — Conjugate priors as pseudo-observations, regularization = MAP (Gaussian → Ridge, Laplace → Lasso). Pairs with module 23.
- [07_stat.tex](../../math/Lectures/stat/07_stat.tex) — **Sampling distributions** (28 frames) — Monte Carlo, CLT, SE vs SD (most-confused pair), √n law, Fisher info ↔ sampling distribution. Pairs with module 24.
- [08_stat.tex](../../math/Lectures/stat/08_stat.tex) — **CIs + bootstrap** (30 frames) — Wald/Wilson/t CIs, delta method, bootstrap algorithm + BCa, when bootstrap fails. Pairs with module 24.
- [09_stat.tex](../../math/Lectures/stat/09_stat.tex) — **Hypothesis testing** (38 frames) — Drug-trial running example, p-values, power, Cohen's d, permutation, Bonferroni vs BH. Pairs with module 25.
- [10_stat.tex](../../math/Lectures/stat/10_stat.tex) — **Classical tests + ANOVA + A/B** (~59 frames, merged L10+L11) — LRT, t-tests, chi-squared, nonparametric, decision flowchart, one-way ANOVA, Tukey vs Bonferroni, eta², A/B testing pitfalls (peeking, Simpson's, novelty). Pairs with module 26.
- [11_stat.tex](../../math/Lectures/stat/11_stat.tex) — (folded into 10's merged deck per the outline; still a standalone file).
- [12_stat.tex](../../math/Lectures/stat/12_stat.tex) — **Regression inference** (32 frames) — LINE assumptions, t-test on coefficients, F-test, R² adj, Gauss-Markov, diagnostics, logistic regression as MLE. **DEFERRED** to ML; canonical location: `ml/deferred_lectures/deferred_regression_inference.tex`.

Two further deferred stat lectures live entirely under ML:
- `ml/deferred_lectures/deferred_glms.tex` — GLMs (29 frames): exponential family + link functions, Poisson regression, IRLS, deviance.
- `ml/deferred_lectures/deferred_causal_inference.tex` — Causal inference (37 frames): RCTs, potential outcomes, DAGs, propensity scores, do-calculus intro.

## Homework write-ups (`math/Homeworks/`)

- `hw_14_stat_1_estimator_properties.pdf` — module 22
- `hw_15_stat_map_mle.pdf` — module 23
- `hw_16_cis_boostrap.pdf` — module 24
- `hw_17_hypotesis_testing.pdf` — module 25

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L01 risk vs empirical risk framing | stat L1 (ERM) |
| L02 squared error as Gaussian MLE | stat L5 + module 19 |
| L03 Ridge = Gaussian MAP, Lasso = Laplace MAP | stat L6 (explicit derivation there) |
| L04 metrics critique (R² lies) | module 27, stat L1 |
| L04 bias-variance | stat L3 (full MSE = Bias² + Var derivation) |
| L05 cross-validation as resampling | stat L8 (bootstrap is sibling) |
| L05b nested resampling vs CIs | stat L8 |
| Logistic regression as MLE / cross-entropy | stat L5 (this is where it's derived) |
| Confusion matrix + hypothesis-testing framing | stat L9 |
| Classifier metrics + Type I / Type II | stat L9 |
| A/B testing in production | stat L10 (full Part IX on A/B testing) |
| Causal inference / observational confounding | deferred causal inference deck |
| GLMs as unifying view of OLS + logistic + Poisson | deferred GLMs deck |
| Probabilistic interpretations of regularizers | stat L6 |
| Calibration (when added to ML checklist) | currently missing — see stat L9 / stat outline for adjacent material |

## Notes

- **Stat outline is the canonical reference.** When the index is unclear, open [`00_lecture_outline.md`](../../math/Lectures/stat/00_lecture_outline.md) — it lists every part of every deck.
- **Three deferred decks live under ML** (`ml/deferred_lectures/`): regression inference (L12), GLMs (L13), causal inference (L14). They were originally planned as stat content but moved into the ML chapter when they tied more naturally to ML lectures (Week 7 buffer week per syllabus).
- **L11 merge:** The original plan was L11 = ANOVA + post-hoc, but it was merged into L10. The `11_stat.tex` file still exists but the canonical deck is `10_stat.tex`.
- All stat decks are LMU-derived in style (Beamer dove theme, `popblue`/`sampred`/`paramgreen` palette per the existing convention). When linking from ML, the visual style is consistent.
- Stat decks heavily cite back to math modules (16 = Bayes, 19 = distributions, 20 = CLT). The cross-reference table in the outline lists every backward link.
