# L11 Boosting — Deck Outline / Design (DRAFT v3)

Planning doc for the **Boosting** deck (AdaBoost + Gradient Boosting), house style
of `ml_new/02_main_concepts_continued/04_overfitting_cross_validation.tex` (the
validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> STATUS: DRAFT v3 (pedagogical review folded in). Spine stable; pending go-ahead.
> Lecture-numbering note: "L01d" = `04_overfitting_cross_validation.tex`;
> "L01c" = `03_data_preprocessing.tex`.

## Chapter-level conventions (shared across L09-L12, recorded in each file)
- **Canonical datasets:** Titanic (classification) + synthetic Yerevan rent-vs-area
  (regression / residuals). New datasets must earn their place.
- **Running-project HW:** one Titanic task across the chapter — single tree (L09) ->
  RF (L10) -> hand-built GBM (L11) -> tuned LightGBM (L12), same score each step.
  IMPORTANT (v3): the by-hand GB rounds use the REGRESSION toy (Yerevan); the
  sklearn/library GB steps run on Titanic (classification). State this in the HW.
- **Math-depth policy:** intuition + worked example in the body; heavier formalism
  boxed/optional. Boosting is where this matters most — see design note.
- **RF-vs-boosting diagram:** reuse the parallel-vs-sequential trees picture from
  L10's bridge as THIS deck's hook.
- **Verification debt:** library defaults/facts verified at BUILD time.

## Scope
AdaBoost (one conceptual frame) -> Gradient Boosting (the core) -> regularization ->
practice. The LIBRARIES (XGBoost/LightGBM/CatBoost) are L12. Follows L10.

## Sources
- **ml_old**: `L07 Boosting.pdf` (39 pp, SLDS-LMU) — AdaBoost; boosting question
  (Kearns/Schapire); forward stagewise additive modeling; gradient boosting as
  function-space gradient descent; pseudo-residuals (= -gradient; = y-f for L2);
  line search; regularization (M / depth / shrinkage); stochastic GB; illustrations.
  `Boosting with and without Line Search.ipynb` (code). Front slide recommends
  StatQuest videos + the 3 libraries.
- **Upstream (raw)**: `ml_old/slides-sl.pdf` (boosting chapter).
- **Web**: Northeastern "Gentle Intro to Gradient Boosting"; AnalyticsVidhya
  GB-vs-AdaBoost. Intuition: AdaBoost re-weights misclassified points; GB fits each
  new tree to residuals (= negative gradient of the loss); trees added one at a
  time, earlier trees frozen.

## Design note (the key call for this deck)
The old SLDS deck is theory-first. INVERT it: residual-fitting intuition + worked
example is the spine; function-space gradient descent is the FORMAL view, introduced
gently via the relabeling "the residual you fit by hand IS the negative gradient."
Do not let formalism sprawl. The hardest jump in the whole chapter lives here; it is
scaffolded across three beats below, not crammed into one frame.

## Proposed spine (~23-25 frames, L01d style)

### Hook
1. Bagging built trees INDEPENDENTLY (L10). What if each new tree FIXES the previous
   ones' mistakes? Reuse the parallel-vs-sequential trees diagram from L10.

### Section 1 — The boosting idea
2. History + the boosting question (weak learner -> strong learner; Kearns/Schapire).
3. Boosting vs bagging table (parallel/independent + variance-cut vs
   sequential/dependent + bias-cut).

### Section 2 — AdaBoost (one conceptual frame, v3)
4. AdaBoost, conceptually: up-weight the points you got wrong, down-weight the rest,
   repeat; final prediction = vote weighted by each stump's accuracy. Small 2D
   moving-boundary viz. Bridge line: "AdaBoost = boosting with EXPONENTIAL loss;
   gradient boosting generalizes to ANY loss" — same framework, not two tricks. (The
   full alpha = 0.5 ln((1-err)/err) arithmetic -> HW/optional, not a body frame:
   half-doing the numbers is worse than one clean conceptual frame.)

### Section 3 — Gradient boosting (the core, scaffolded across 3 beats)
5. **Residual-fitting worked example** on the synthetic Yerevan rent-vs-area toy
   (continuity L09 frame 12): init = mean(y) -> residuals -> fit a stump to the
   residuals -> add with learning rate eta -> recompute residuals -> repeat.
   Step-by-step frames.
6. **Predict-first:** after 200 rounds, what is the TRAINING error? the TEST error?
   Boosting can drive train error to ~0; eta + #trees control overfitting. Explicit
   contrast: RF -> more trees is SAFE; boosting -> more trees CAN overfit (regularized
   by small eta + early stopping). This contrast is itself a key teaching point.
7. **The reframe (v3, makes the cliff gentle):** for squared loss, the residual y-f
   you have been fitting by hand IS the negative gradient of the loss -> "you were
   already doing gradient descent without knowing it." (predict-first relabeling.)
8. **Generalize (v3):** for losses other than L2, fit the NEGATIVE GRADIENT instead
   of the raw residual (L1/Huber -> robustness). Optional box: function-space GD view.
9. **Classification GB (dedicated frame, v3):** predict in LOG-ODDS space; the
   pseudo-residual is `y - p`; squash with the sigmoid. Taught at the same grain as
   regression GB so the Titanic HW is actually supported (this was the v2 gap).

### Section 4 — Regularization
10. Three knobs — number of trees M (early stopping), tree depth (interaction order),
    learning rate / shrinkage eta. **Show the eta/M tradeoff (v3):** predict-first
    "if I halve eta, what happens to the #trees I need?" + a two-curve sketch (small
    eta = slow/smooth/generalizes; large eta = fast/jagged/overfits).
11. Stochastic gradient boosting (row subsampling) — the bagging+boosting hybrid that
    XGBoost/LightGBM use by default (foreshadow L12).

### Section 5 — Practice & bridge
12. sklearn `GradientBoostingClassifier` / `HistGradientBoostingClassifier`; key
    params; early stopping on a validation set (callback L01d). Tie back to L09's
    single-tree LightGBM ("now n_estimators=100, learning_rate small").
13. Bridge -> L12: XGBoost / LightGBM / CatBoost are engineered gradient boosting; L12
    also closes with **stacking** (the third ensemble strategy: a meta-learner over
    different model types).
    Recap + HW (running project: hand-build 3 rounds of GB on the REGRESSION toy,
    then sklearn GBM on Titanic; compare to the L10 RF score).

## Open decisions (pending)
1. AdaBoost: the one conceptual frame above (recommended) or skip entirely to GB?
2. Function-space GD: keep as an optional box only (recommended) or a full frame for
   the math-inclined?

## Changelog (v2 -> v3, pedagogical review)
1. The hard gradient idea split across 3 beats (residual=neg gradient relabeling ->
   generalize to any loss -> classification GB), fixing the chapter's difficulty cliff.
2. Classification GB promoted to a dedicated frame (log-odds, pseudo-residual y-p) so
   the classification Titanic HW is supported (was only "analogous" in v2).
3. AdaBoost reduced to ONE clean conceptual frame + exponential-loss bridge; the full
   arithmetic moved to HW/optional (no half-done numbers).
4. eta/M tradeoff now shown (predict-first + sketch), not just stated.
5. Explicit RF-more-trees-safe vs boosting-more-trees-can-overfit contrast added.
6. Reuse the parallel-vs-sequential diagram as the hook; HW regression/classification
   split made explicit; naming/file references updated.
