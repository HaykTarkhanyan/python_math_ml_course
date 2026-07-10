# L11 Boosting — Deck Outline / Design

Interview-driven rebuild (2026-06-23). Supersedes the prior AI-authored v3 draft:
the instructor chose a fresh build. Headline requirement: **step-by-step animations**
of the gradient-boosting process. House style of
`ml/02_main_concepts/06_overfitting_cross_validation.tex` (L01d).

> Lecture-numbering note: "L01c" = `03_data_preprocessing.tex`,
> "L01d" = `06_overfitting_cross_validation.tex`. Follows **L09 Decision Trees** (built)
> and **L10 Random Forests** (built). Libraries (XGBoost/LightGBM/CatBoost) are **L12**.

## Interview decisions (2026-06-23)

- **Animations (the headline):** animate **gradient boosting on the Yerevan regression
  toy** as a **2-panel, 5-6 round** sequence (Beamer overlays from a generated figure
  series): left = data + cumulative ensemble fit + the stump just added; right = the
  **residuals shrinking** toward zero. (NOT animating AdaBoost or classification.)
- **AdaBoost:** **one conceptual frame** (reweighting idea + "AdaBoost = boosting with
  exponential loss" bridge). Full `alpha` arithmetic -> HW/optional.
- **Math depth:** **FULL formal frame** for function-space gradient descent, **including
  the line-search step** (Friedman's true algorithm: pseudo-residual -> fit h ->
  line-search gamma -> update). The instructor wants the real algorithm here.
- **Loss functions:** **concept only** -- "swap the loss, swap the pseudo-residual" with
  ONE example (Huber/absolute -> robust). No big table.
- **Classification GB:** a **dedicated static frame** (log-odds, pseudo-residual
  `y - p`, sigmoid) so the Titanic HW is supported. Not animated.
- **Regularization:** **full** -- three knobs (M + early stopping, depth, learning rate
  eta) + the **eta/M tradeoff** (predict-first + figure) + **stochastic GB** as a concept
  frame (subsampling regularizes + speeds up; one-line "libraries default to this" pointer).
- **Practice:** **trimmed to essentials** -- ONE `[fragile]` frame (GB in sklearn +
  early stopping, conceptual params only). Tuning recipes + the library APIs are L12.

## L11 / L12 boundary (instructor flagged L12 is the deep library session)
- **L11 owns:** the gradient-boosting *algorithm* + the math (residuals, function-space
  GD with line search, any-loss principle, classification via log-odds), the animation,
  general regularization (M / depth / eta / subsampling), and just-enough sklearn to run it.
- **L12 owns:** XGBoost / LightGBM / CatBoost specifics, histogram/leaf-wise growth,
  native categorical handling, the tuning-order recipe, the bake-off, and stacking.
- **Do NOT** pre-spend L12 material in L11. Name the libraries only in the bridge.

## Review fixes applied (independent adversarial review, 2026-06-23)
1. **Frame 10 (C1):** present the generic Friedman algorithm, then an `armorange` box
   reconciling with sklearn's **per-leaf** optimal step (then shrink by `eta`); for squared
   loss the leaf step = residual mean = frame 6. Removes the formal-vs-by-hand inconsistency.
2. **Frame 9 (C2):** show `L = 1/2(y-F)^2 => -dL/dF = y-F` so the residual = negative
   gradient with no stray factor of 2.
3. **Frame 12 (C3):** `F` lives in log-odds; sigmoid applied ONCE at the end; stop at
   `y - p` (the per-leaf value is a weighted ratio, not a mean -- do not imply otherwise).
4. **Frame 3 (S3):** add the mirror -- RF bags DEEP (high-variance) trees; boosting boosts
   SHALLOW (high-bias) trees.
5. **Frame 8 / `boost_overfit` (S2):** state on-slide that the overfit demo uses a
   deliberately over-powered config (not the depth-1 animation).
6. **`boost_anim` (S1):** pin `eta` (~0.3-0.5), default `init`=mean, EYEBALL the 6 rounds
   before committing; frame 6's by-hand numbers must match round 1's actual split.
7. **Frame 14 (S4):** "halve eta -> ~double trees" is a small-`eta` rule of thumb, not
   exact; the figure shows the general trend only.
8. **Frame 15 (S5):** corrected -- row/column subsampling is NOT on by default (XGBoost
   `subsample` / LightGBM `bagging_fraction` = 1.0); the libraries expose it (L12).
9. **Frame 4 (S6):** AdaBoost & GB are both forward stagewise additive models (footnote).
- **Datasets:** synthetic **Yerevan rent-vs-area** (regression: animation + by-hand) +
  **Titanic** (classification GB + sklearn). Chapter convention.
- **Hook:** reuse the **parallel-vs-sequential trees diagram** introduced at the end of L10.

## Chapter-level conventions (shared across L09-L12)

- **Canonical datasets:** Titanic + synthetic Yerevan rent-vs-area. New datasets must earn their place.
- **Running-project HW:** single tree (L09) -> RF (L10) -> hand-built GBM (L11) -> tuned
  LightGBM (L12), comparing the SAME score each step. By-hand GB uses the REGRESSION toy;
  the sklearn/library GB steps run on Titanic (classification). Score continuity: keep
  **accuracy** (the L09-L10 running score) + also report F1/ROC-AUC on Titanic.
- **Math-depth policy:** intuition + worked example in the body; heavier formalism boxed
  -- BUT for L11 the instructor wants the function-space GD frame done in full.
- **RF-vs-boosting diagram:** the parallel-vs-sequential picture from L10's bridge is THIS
  deck's hook.
- **Style:** match L01d/L09/L10 -- Armenian-flag palette (`armblue` theory, `armred`
  warning/data, `armorange` watch-out, `paramgreen` good/Next), `\fcolorbox{..}{..!8}`
  boxes, `\pause` predict-first, hook before `\tableofcontents`, `[plain]` section
  transitions, Recap + HW, `% Provenance` block. Preamble `\input{../preamble}`; code via
  the L09 `\lstset`. Animations = generated figure series shown as `\only<k>` overlays.
- **Verification debt:** library/sklearn facts verified at BUILD time (Context7).

## Callbacks / cross-references

- **L10** (random forests): bagging grows trees INDEPENDENTLY and averages (cuts
  variance). Boosting grows them SEQUENTIALLY to fix mistakes (cuts bias). Reuse the
  parallel-vs-sequential diagram. Contrast: more trees safe in RF, can overfit in boosting.
- **L09** (trees): the base learner is a shallow tree / stump (callback the "tree is the
  atom" + single-tree LightGBM frame); regression-tree step function (frame 12 of L09)
  seeds the residual picture.
- **L01d** (CV / early stopping): pick #trees by a validation set / CV; the train-vs-test
  overfitting U-curve.
- **Forward -> L12:** XGBoost / LightGBM / CatBoost = engineered gradient boosting;
  stochastic GB (subsampling) is their default; L12 also closes with stacking.

## Frame-by-frame outline (~19 content frames + outline + 4 transitions; animation adds ~6 overlay pages)

### Hook (before outline)
1. **From averaging to fixing mistakes.** Reuse the parallel-vs-sequential diagram (L10):
   RF grew trees in parallel and averaged. *What if each new tree instead FIXES what the
   previous ones got wrong?* That is **boosting**.

### Outline frame (`\tableofcontents`)

### Section 1 - The boosting idea  [transition]
2. **The boosting question.** Can many **weak** learners (each barely better than chance)
   be combined into one **strong** learner? (Kearns & Schapire.) Boosting's answer: yes,
   if each one focuses on what the others miss.
3. **Bagging vs boosting** (table, callback L10). Parallel / independent / cuts
   **variance** (RF) vs sequential / dependent / cuts **bias** (boosting). **Mirror of
   L10:** RF bags **deep** (high-variance) trees; boosting boosts **shallow** (high-bias)
   trees -- each only needs to nudge the fit, and bias is removed across rounds. Flag up
   front: more trees is SAFE in RF but **can overfit** in boosting (we will see it).
4. **AdaBoost (one conceptual frame).** Up-weight the points you got wrong, down-weight
   the rest, refit, repeat; final prediction = vote weighted by each stump's accuracy.
   Small TikZ schematic (misclassified points drawn bigger each round). Bridge line:
   **"AdaBoost = boosting with exponential loss; gradient boosting generalizes to ANY
   loss"** -- both are forward stagewise additive models (AdaBoost predates the gradient
   view), not two unrelated tricks. (The `alpha = 0.5 ln((1-err)/err)` weight
   arithmetic -> HW.)

### Section 2 - Gradient boosting  [transition]
5. **The idea: fit the residuals.** Start with `F_0 = mean(y)`. Compute residuals
   `r = y - F_0`. Fit a **small tree** to the residuals. Add it (scaled by learning rate
   `eta`): `F_1 = F_0 + eta * tree`. Recompute residuals; repeat. Each tree learns
   *what is still wrong*.
6. **By hand (Yerevan rent-vs-area).** Real numbers from the figure script: `F_0 = mean(y)`;
   residuals for a few points; the first stump splits at `area = t` and predicts the two
   residual means; `F_1 = F_0 + eta * stump`. One clean round of arithmetic.
7. **Watch it fit (ANIMATED, 2-panel, 6 rounds).** Overlay pages `boost_anim_1..6`:
   - **Left:** Yerevan data + the **cumulative ensemble fit** `F_k` (a step function that
     grows more steps) + the **new stump** just added.
   - **Right:** the **residuals** `r_k = y - F_k` collapsing toward zero.
   Caption per round: "round k: fit the leftover error, add a (shrunken) stump, repeat."
8. **Predict-first: after many rounds, what are train and test error?** Reveal: training
   error drives toward **~0** (boosting can memorize); test error drops then **RISES** ->
   **boosting CAN overfit**. Explicit contrast with L10 (RF: more trees safe). Controlled
   by small `eta` + **early stopping**. Figure `boost_overfit` -- say on the slide that this
   uses a **deliberately over-powered** config (deeper trees + `eta=1`, many rounds) to
   *force* the failure mode you would normally avoid; the depth-1 animation does not overfit.
9. **The reframe (predict-first relabel).** For squared loss `L = 1/2 (y - F)^2`, one line:
   `dL/dF = -(y - F)`, so `-dL/dF = y - F` = the residual **exactly** (the `1/2` is the
   convention that kills the stray factor of 2). So the residual you have been fitting **IS**
   the negative gradient. *"You were already doing gradient descent -- in function space --
   without knowing it."*
10. **Function-space gradient descent (FULL formal frame).** Algorithm box (generic
    Friedman): loss `L(y, F)`; **pseudo-residual** `r_im = -[partial L / partial F]_{F =
    F_{m-1}}`; fit base learner `h_m` to the pseudo-residuals; **line search** for the step
    `gamma_m`; update `F_m = F_{m-1} + eta * gamma_m * h_m`. Squared loss recovers frame 9's
    plain residual. `armorange` reconciliation box: **sklearn refines this** -- it sets
    **each leaf** to its own optimal step (a per-leaf line search), then shrinks all leaves
    by `eta`; for squared loss that leaf-optimal step is just the leaf's **residual mean**,
    exactly what frame 6 did by hand. (Reconciles frames 6, 10, 16.)
11. **Any loss you like (concept).** Swap the loss -> swap the pseudo-residual. One
    example: **absolute / Huber -> robust to outliers** (the tree chases the sign /
    clipped residual instead of the raw residual). No table -- just the principle.
12. **Classification GB (dedicated, static).** The model `F` accumulates in **log-odds**
    space; loss = log-loss; **pseudo-residual = `y - p`** where `p = sigmoid(F)`. Add trees
    in log-odds; the sigmoid is applied **once at the end** to read off a probability (not
    per tree). Same machinery, different loss -- this is what the Titanic HW runs. (Stop at
    `y - p`; the per-leaf value here is a weighted ratio, not a plain mean -- do not imply it is.)

### Section 3 - Regularization  [transition]
13. **Three knobs.** Number of trees **M** (+ early stopping), **tree depth** (interaction
    order; depth 1 = main effects, deeper = interactions), **learning rate / shrinkage
    `eta`**. Each trades fit vs generalization.
14. **The `eta`/M tradeoff (predict-first + figure).** Predict-first: "halve `eta` -- what
    happens to the number of trees you need?" Reveal: **roughly double** (a rule of thumb in
    the small-`eta` regime, not exact). Small `eta` = slow, smooth, generalizes; large `eta`
    = fast, jagged, overfits. Figure `boost_learning_rate` shows the *general* trend
    (smaller `eta` needs more trees); it does not claim to prove the exact 2x.
15. **Stochastic gradient boosting.** Subsample **rows** (and **columns**) each round ->
    faster and a regularizer (a bagging+boosting hybrid; Friedman 2002). The big libraries
    **expose this** and it is a common recipe -- but it is **NOT on by default** (XGBoost
    `subsample` and LightGBM `bagging_fraction` both default to 1.0). Details in L12.

### Section 4 - Practice & bridge  [transition]
16. **sklearn (essentials only).** `[fragile]` `GradientBoostingClassifier` (or
    `HistGradientBoostingClassifier`) with just the core params + **early stopping**
    (`validation_fraction`, `n_iter_no_change`). Tie back to L09's single-tree LightGBM.
    `armorange` note: **the tuning recipe and the XGBoost/LightGBM/CatBoost APIs are L12 --
    this frame just gets gradient boosting running.**
17. **Bridge -> L12 (the deep library session).** XGBoost / LightGBM / CatBoost are
    **engineered** gradient boosting (speed, built-in regularization, native categorical
    handling) -- L12 is where we tune them properly and run the bake-off, and it closes the
    chapter with **stacking**. `paramgreen` Next box.

### Wrap-up (outside the sections)
18. **Recap.** Boosting = sequential trees, each fitting the (pseudo-)residual = negative
    gradient; cuts bias; `eta` + #trees + early stopping control overfitting.
19. **HW11 (running project).** By hand: 3 rounds of GB on the **Yerevan** regression toy
    (init = mean, residual, stump, update). sklearn: `GradientBoostingClassifier` on
    **Titanic**; compare to the L10 RF score (same metric: accuracy + F1/ROC-AUC); tune
    `learning_rate` + early stopping. **Bonus:** the AdaBoost `alpha` arithmetic; force
    overfitting (train -> 0, test U) then early-stop; try `subsample < 1`.

## Animation & figures (`py_src/make_boost_figures.py` -> `fig/`, `ma` venv)
- **`boost_anim_1..6`** (the headline animation): 2-panel per round. Synthetic 1D Yerevan
  rent-vs-area (~50 pts, smooth nonlinear + noise, seed 509). `GradientBoostingRegressor`,
  depth-1 stumps, **default `init` (= mean, so `F_0 = mean(y)`)**, `staged_predict` for
  `F_1..F_6`. **Pin `eta` (~0.3-0.5) so all 6 rounds are visibly distinct -- EYEBALL the
  rendered series before committing (too small = panels look identical; too large = jagged).
  If depth-1 looks too flat over 6 rounds, raise `eta` (do not silently switch to depth-2
  without a note).** Left: data + `F_k` step function + the round-`k` stump increment.
  Right: residuals `r_k`. Shown via `\only<k>` overlays on frame 7.
- **`boost_overfit`**: a **deliberately over-powered** config (deeper trees + `eta=1`, many
  rounds, train/test split on the toy) so train MSE -> ~0 and test MSE makes a clear U;
  annotate "early-stop here" (frame 8). NOT the depth-1 animation config.
- **`boost_learning_rate`**: test error vs #trees for two `eta` (e.g. `1.0` vs `0.1`),
  showing smaller `eta` needs more trees + generalizes. General trend only (frame 14 -- do
  not over-claim the exact 2x).
- **By-hand numbers** (frame 6): the script logs `F_0 = mean(y)`, **round 1's actual split
  threshold + the two leaf values + the `eta` used**, and a couple of residuals -- frame 6
  must use exactly these so it MATCHES animation round 1 (same split, same `eta`).
- TikZ schematics (no matplotlib): parallel-vs-sequential hook diagram (redrawn from L10);
  AdaBoost reweighting schematic (frame 4); the function-space GD algorithm box (frame 10).
- Palette: Armenian flag (data points blue, fit/ensemble red or orange, residuals);
  single-series plots per the slide-style rule.

## Sources (carried; re-verify library facts at build)
- **ml_old:** `L07 Boosting.pdf` (SLDS-LMU) -- AdaBoost; boosting question; forward
  stagewise additive modeling; gradient boosting as function-space gradient descent;
  pseudo-residuals (= -gradient; = y-f for L2); line search; regularization (M / depth /
  shrinkage); stochastic GB. `Boosting with and without Line Search.ipynb`.
- **Web:** Northeastern "Gentle Intro to Gradient Boosting"; AnalyticsVidhya GB-vs-AdaBoost.
- **VERIFY at build (Context7 / sklearn):** `GradientBoostingClassifier/Regressor` +
  `HistGradientBoostingClassifier/Regressor` params (`learning_rate`, `n_estimators`,
  `max_depth`, `subsample`, `validation_fraction`, `n_iter_no_change`); `staged_predict`
  for the animation; AdaBoost `alpha` formula; Huber/absolute pseudo-residuals.
- **Design note:** invert the theory-first SLDS deck -- residual-fitting intuition +
  the animation are the spine; the formal function-space GD frame comes AFTER the picture,
  as the "what you were really doing" payoff (per instructor: do it in full).
