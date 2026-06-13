# Pedagogical review: L01d, L01d2, L01e, L01f, L01g, L01h

Review of the 6 upcoming regression-chapter slide decks, focused on **missing teaching opportunities and topics** — not on the existing content (which is generally strong).

Frame counts as of the review: L01d=29, L01d2=26, L01e=26, L01f=30, L01g=22, L01h=22.

---

## L01d — Validation & CV

The most thorough of the six. Real gaps:

1. **Nested cross-validation is absent.** The single biggest standard topic missing. When you tune hyperparams with CV AND evaluate with the same CV, you're overfitting the validation procedure — even the "honest" test score is inflated. The fix is outer-CV-for-evaluation, inner-CV-for-tuning. Hastie/Tibshirani treat this as critical. Without it, students will publish optimistic numbers and not know why.

2. **The "1-SE rule" is missing.** When CV scores are noisy, pick the simplest model within 1 standard error of the best. Tibshirani's classic recommendation, prevents complexity creep. One frame.

3. **`RepeatedKFold` mentioned but not motivated.** Repeated k-fold reduces variance of the CV estimate by averaging multiple shuffles. Worth a sentence on "when k-fold scores are jumpy, repeat the CV."

4. **LOOCV omitted entirely.** Extreme of k-fold (k=n). Pedagogically useful as the boundary case: low bias, high variance, and shows why we settle on k=5 or k=10.

5. **OOB (out-of-bag) error from bootstrap.** A non-CV alternative — sample with replacement, evaluate on unsampled rows. Random forests use this natively, free regularization. At least mention as a parallel paradigm.

---

## L01d2 — Bias-variance

The deepest of the six. Hard to find real gaps. Minor:

1. **Bootstrap as the practical tool for *measuring* bias/variance.** The deck explains the decomposition but never says "here's how you actually compute these on real data." A frame on "fit your model on B bootstrap samples → compute the variance and bias of predictions at a fixed test point" would close the loop between theory and lab.

2. **Connection between regularization strength and validation curve shape.** L01d has validation curves; L01d2 has the bias-variance tradeoff. A frame that overlays the two — "the validation curves you'll actually see ARE the bias-variance tradeoff curve" — would tie L01d2 to L01d2's purpose.

---

## L01e — Hyperparameter tuning

Has the modern stack (grid, random, Bayesian/Optuna). Gaps:

1. **Hyperband / Successive Halving is mentioned but not visualized.** sklearn now ships `HalvingGridSearchCV` and `HalvingRandomSearchCV` (since 0.24). The pedagogy is gorgeous: spawn many cheap trials, kill the bottom half each round, double the budget for survivors. One frame with a "bracket diagram" would land it.

2. **Search space design intuition is missing.** Students grid-search learning rates as `[0.001, 0.01, 0.1]` (linear) instead of log-uniform — and they choose ranges that don't include the optimum. A frame on "use log-uniform for LR / regularization / depth, integer for n_estimators, conditional spaces for layered HPs" would prevent the most common search mistake.

3. **Pruning / early stopping in HP search.** Optuna's pruners (`MedianPruner`, `HyperbandPruner`) kill bad trials early. Frame is "you don't need to run every trial to completion."

4. **Reproducibility test.** When `best_params` changes meaningfully across random seeds of the tuner, the tuning is overfit. Quick diagnostic: "rerun with 3 seeds, see if the same HPs win."

5. **Multi-fidelity vs full-fidelity tradeoff.** Train on a 10% subsample first to filter — then tune the survivors at full fidelity.

---

## L01f — Regression metrics

Very comprehensive (30 frames). The "residual diagnostics" + "prediction intervals" coverage is unusually deep for an intro course. Gaps:

1. **Quantile (pinball) loss is absent.** When you care about the 90th-percentile prediction, not the mean, you minimize pinball loss. Modern tabular models (LightGBM has it built in) routinely use this. Worth one frame — "what if you don't want the mean?"

2. **MASE (Mean Absolute Scaled Error) for time series.** Standard in forecasting (Hyndman et al.): scale MAE by the naive forecast's MAE. Resolves the "MAPE explodes near zero" problem for time-series targets in a way SMAPE doesn't. Worth a frame in the magnitude section.

3. **Decision: train metric vs evaluation metric.** L01f opens with "loss vs metric: what's the difference" — could close the loop with explicit guidance on when the two should match and when they should diverge.

4. **Concordance / Kendall's tau for ranking tasks.** Already mentioned in the rank section. Could be expanded.

---

## L01g — Feature engineering

22 frames, mostly covers what to do. Missing:

1. **Binning / discretization.** Equal-width vs equal-frequency vs decision-tree-derived bins. For linear models on non-linear features, binning is the cheapest fix. The frame on "polynomial features" handles smooth nonlinearity; binning handles step-function nonlinearity. Worth a dedicated frame.

2. **Domain knowledge as the #1 source of features.** The deck lists "sources of new features" — but the most important source ("ask a domain expert / look at the data") is implicit. A frame with two or three concrete examples ("hospital admissions data: a nurse will tell you that bed occupancy at admission matters; logs show that day-of-week is the dominant signal in support tickets") makes this real.

3. **Test-time computation cost.** Features that require a DB lookup, an API call, or a pre-aggregation pipeline are problematic in production. Worth a slide — connects to the feature-stores frame.

4. **Leakage IN feature engineering specifically.** The "6 pitfalls" frame mentions it but a concrete example ("computing the rolling 30-day mean using rows from AFTER the target date") teaches the subtle case better.

---

## L01h — Feature selection

22 frames covering filter / embedded / wrapper. Gaps:

1. **Mutual information as a filter is absent.** sklearn ships `mutual_info_regression` and `mutual_info_classif`. Strictly more powerful than Pearson for non-monotonic relationships. Should sit next to "univariate filters."

2. **Feature selection vs dimensionality reduction (PCA).** Important conceptual distinction students conflate. Selection keeps the original features (interpretable); PCA creates linear combinations (not interpretable). One frame, big payoff.

3. **mRMR (minimum Redundancy Maximum Relevance)** — popular in industry, handles the "two highly correlated relevant features" case explicitly. Worth a sentence at minimum.

4. **The iterative loop**: engineer many → select few → repeat. The L01g→L01h flow is treated as one-shot. Real practice loops between them. Worth saying explicitly.

---

## Cross-deck opportunities

1. **A single worked example that recurs across L01d → L01d2 → L01e → L01f → L01h** would compound the learning. Right now each deck picks its own demo. The cheese factory worked for L07 (classification); something analogous for regression (rental prices? bike-share counts?) would let you reference back: "remember the bias-variance plot from L01d2 — this is what the validation curve from L01d looks like for that case."

2. **"What you'll publish vs what's true"** — recurring theme. Every deck has a place to call out the gap between the metric students report and the metric that matters. Could be a small recurring callout box.

---

## Highest-leverage adds if applying only two

- **Nested CV in L01d** — single most standard topic missing, and the failure mode it prevents is one students will hit in their first ML project.
- **Hyperband visualization in L01e** — cleanest pedagogical win, with sklearn support that's only 4 years old so it's still "fresh" content.
