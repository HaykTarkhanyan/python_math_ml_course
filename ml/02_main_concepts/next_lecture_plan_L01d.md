# Next lecture plan — L01d Validation, CV, Underfit/Overfit

90 minutes. Combines L01d (validation/CV) with a slice of L01d2 (overfit/underfit). Reuses three pedagogical assets from `ml/Chapter 1 Regression Main Concepts/Code/02_Regression_Main_Concepts.ipynb`.

## Reusable demos from the existing notebook

### Demo 1 — "3 points + degree-10 polynomial" (cells 165-183)
- Sort the synthetic data, take the first 3 points (`df_small`)
- Fit a degree-10 polynomial — passes through all 3 perfectly (zero training error)
- Predict on all 200 points — the polynomial is wildly wrong everywhere else
- Side-by-side plot + bar chart of small-data MSE vs all-data MSE

The cleanest "training error lies" demo. Zero training MSE that is totally useless.

### Demo 2 — Train/test split as the answer (cells 184-191)
- 30 sample points, degree-40 polynomial
- `train_test_split(test_size=0.2, random_state=509)`
- 4-bar chart: Train-Line / Test-Line / Train-Poly / Test-Poly
- Poly has near-zero train MSE but huge test MSE — train-test gap as the overfit fingerprint

### Demo 3 — The cultural analogies (cells 195-196)
- Overfit = ancient astrologers fitting a rule to every celestial coincidence ("Mars in Taurus + half moon + left-handed prince → harvest")
- Underfit = medieval bloodletting (one too-simple rule for every disease)
- Memorable; students retain these for years

## Topic list (90 min)

### Part 1 — Why training error lies (~20 min)
1. Open with Demo 1 — zero training error, useless model
2. Generalize: any sufficiently flexible model can memorize any training set
3. Name the failure mode: overfitting
4. Introduce the opposite — underfitting — with a too-simple model on the same data
5. The two cultural analogies (astrology + bloodletting) for retention

### Part 2 — Train/test split as the fix (~20 min)
6. Demo 2 — the 4-bar chart showing train-test gap for line vs polynomial
7. Mechanics: `train_test_split`, `random_state`, `test_size`, why we set the seed
8. Critical rules: test set looked at exactly once, no preprocessing on full data, no peeking

### Part 3 — Why train/test isn't enough (~15 min)
9. The hyperparameter-tuning trap (predict-first frame from the L01d deck)
10. Three-way split: train / val / test
11. Practical rule: test set is for the final number you put on a slide, that's it

### Part 4 — Cross-validation (~25 min)
12. "One validation set is noisy" — motivation for k-fold
13. k-fold mechanics + the "1,2,3,4,5 → 4 train + 1 val" sketch from cell 215
14. Choosing k = 5 vs 10, the bias-variance trade-off of CV itself
15. `cross_val_score`, `KFold`, `RepeatedKFold` in code
16. Stratified CV for imbalanced classes (the 99% / 1% example from cell 216)
17. Brief mention of `GroupKFold` and `TimeSeriesSplit` (one frame each)

### Part 5 — Tying it back (~10 min)
18. Validation curves + learning curves (already in L01d deck)
19. Decision flowchart: which CV variant when
20. Preview: bias-variance decomposition is the THEORY behind the train-test gap (next lecture, L01d2)

## Additions to the demos before re-using them

- Demo 1: run with several polynomial degrees on the 3 points (2, 5, 10). Show all three pass perfectly. Predict on full data, only the linear one survives.
- Demo 2: sweep degrees 1, 2, 3, 5, 10, 20, 40. Plot train MSE vs test MSE as a line chart. The crossover point is the overfitting threshold — same U-shape as L01b's "The U-shape of test error" frame, now empirical.
- Add a CV demo: take Demo 2's setup, replace train/test with 5-fold CV, show that the CV-mean degree pick is statistically defensible.

## HW after this lecture

Reuse Demo 2's structure as student work:
- Take HW1's polynomial regression
- Run 5-fold CV at degrees 1..15
- Plot mean test MSE +/- std per degree
- Find the degree where CV-mean is minimized — this is the "correct" complexity, found honestly
- Bonus: `RepeatedKFold(n_repeats=5)` for variance reduction

## What's NOT this lecture

- Bias-variance decomposition (L01d2) — conceptual frame for what CV measures, next lecture
- Regularization (L03) — the fix for what CV reveals
- HP tuning (L01e) — builds on CV + adds Ridge's alpha as the canonical knob
- Regression metrics (L01f) — wait until after regularization
