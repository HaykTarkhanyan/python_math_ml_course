# L10 Random Forests — Deck Outline / Design

Interview-driven rebuild (2026-06-23). Supersedes the prior AI-authored v3 draft:
the instructor chose a fresh build, so this outline is shaped by a 3-round interview,
then two review passes (self-review + one independent adversarial review). House style
of `ml/02_main_concepts/04_overfitting_cross_validation.tex` (the
validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> Lecture-numbering note: "L01c" = `03_data_preprocessing.tex`,
> "L01d" = `04_overfitting_cross_validation.tex`. Short labels kept until the
> numeric-prefix migration is final. Follows **L09 Decision Trees** (built).

## Interview decisions (2026-06-23)

- **Length:** run as long as needed, ~22-24 content frames. No forced cuts (L09 precedent).
- **Datasets:** **Titanic** (classification through-line) + **synthetic Yerevan
  rent-vs-area** (regression — a `RandomForestRegressor` example so averaging vs
  voting is both shown). No third dataset.
- **By-hand frame:** **bootstrap + aggregation** — see frame 5 (computable: draws a few
  bootstrap samples, shows in-bag/OOB per sample, aggregates GIVEN per-tree predictions).
- **Variance math:** **keep the boxed, color-coded formula**
  `Var = rho*sigma^2 + (1-rho)/M * sigma^2` — own frame (8), stated not derived, with
  the M->inf floor and the per-tree-variance caveat.
- **Feature importance:** **default impurity-based only**, brief. Permutation / SHAP /
  the full treatment are a **dedicated later lecture** (feature selection) — forward-point.
- **Extra Trees:** **one-liner** in the knobs frame. No dedicated frame.
- **Proximities:** **one-liner** folded into the strengths frame. No dedicated frame.
- **OOB error:** **dedicated frame**, placed inside the Bagging section.
- **Hook:** **quantified single-tree instability** (callback L09 — perturb the data,
  count how predictions flip).
- **Practice:** one `[fragile]` sklearn snippet **+ a key-knobs frame**.
- **Predict-first frames (all four):** (1) fewer features per split, (2) more trees
  never overfits, (3) single-tree instability at the hook, (4) averaging cuts
  variance not bias.

## Review fixes applied

**Round 1 (self-review):** by-hand reframed to be computable; sections consolidated
6 -> 3; the ~37% given one owner; the variance-not-bias predict-first moved off the
intuition frame; "why trees" point added; "more trees never overfits" made precise;
formula flagged stated-not-derived; hook figure = flip-count bar.

**Round 2 (independent adversarial review, 2026-06-23):**
1. **Formula floor (C1/C2).** Frame 8 now states `sigma^2` = per-tree variance (held
   fixed in the formula), gives the limit `Var -> rho*sigma^2` as M->inf (= the plateau
   height), and warns that lowering `rho` via feature subsampling COSTS a small rise in
   `sigma^2` (each tree gets worse) -> there is an OPTIMAL `max_features`, not
   `max_features=1`. Frame 14's plateau is annotated `~ rho*sigma^2`, tying curve to formula.
2. **By-hand (S2).** Frame 5 draws ~3 bootstrap samples (one tree each, each with its
   own in-bag/OOB), THEN aggregates the given per-tree predictions — fixes the "one
   sample -> five votes" conflation.
3. **Overloaded formula frame (S1).** Split: frame 7 = predict-first intuition
   (variance not bias), frame 8 = the boxed formula. One idea per frame.
4. **L01d anchor (S3).** Frame 6 frames the CV `1/k` callback as the NAIVE HOPE (you'd
   get `sigma^2/M` IF trees were independent) that frame 8 then corrects via `rho` — not
   as a clean analogy.
5. **`max_features` defaults (C3).** Frame 13 states explicitly: classifier default
   `sqrt(P)` MATCHES the recommendation; regressor default = all features (1.0), does
   NOT match the `P/3` recommendation -> often lower it by hand.
6. **Extra Trees (S4).** Knobs one-liner corrected: random split thresholds AND (by
   default) no bootstrapping -> faster, more randomization.
7. **Class imbalance (S5, partial).** `class_weight="balanced"` one-liner added to the
   knobs frame (Titanic is imbalanced; L09 handed this forward). NOT changing the
   running-project comparison metric unilaterally — see build-time TODO.
8. **OOB precision (M1).** Frame 10 says "comparable to a held-out estimate, slightly
   pessimistic," not "~3-fold-CV-like" stated as fact.
9. **37% is a large-n limit (M2).** Noted on frames 5 and 9 (the tiny by-hand samples
   will not show exactly 37%).

## Build-time TODO (carried from review)
- **Running-project metric:** check what score L09's HW actually compares on Titanic.
  Titanic is imbalanced (~38% positive); raw accuracy contradicts the L12 metrics
  lecture. Make L10's HW compare the SAME score as L09 and prefer F1 / ROC-AUC over
  raw accuracy. Reconcile across L09-L12 so "the same score each step" is a sound one.
- Verify the bridge diagram `rf_parallel_vs_sequential` is actually BUILT in L10
  (frame 19), since L11's hook reuses it.

## Chapter-level conventions (shared across L09-L12, recorded in each file)

- **Canonical datasets:** Titanic + synthetic Yerevan rent-vs-area. Any other dataset
  must EARN its place.
- **Running-project HW:** one Titanic task carried across the chapter — single tree
  (L09) -> random forest (L10) -> hand-built GBM (L11) -> tuned LightGBM (L12),
  comparing the SAME score each step. Each deck's HW is one stage.
- **Math-depth policy:** intuition + worked example in the body; heavier formalism in
  a boxed / optional frame.
- **RF-vs-boosting diagram:** the recurring "parallel trees (RF) vs sequential trees
  (boosting)" picture is introduced in THIS deck's bridge and reused as L11's hook.
- **Style:** match L01d — hand-drawn TikZ / pgfplots + real matplotlib for data,
  Armenian-flag palette (`armblue` theory, `armred` test/warning, `armorange`
  highlight/watch-out, `paramgreen` good/goal), `\fcolorbox{...}{...!8}` callout
  boxes, `\pause` predict-first frames, hook before `\tableofcontents`, Recap + HW
  frames, `% Provenance` block at the bottom. Preamble via `\input{../preamble}`,
  code via the `listings` `\lstset` from L01d.
- **Verification debt:** library defaults verified at BUILD time (Context7).
- **File:** write to `ml/ch3_trees/L10_random_forests.tex`, overwriting the
  skeleton. Compile from inside `ch3_trees/`.

## Callbacks / cross-references to thread in

- **L09** (decision trees): single trees are unstable / high-variance; ensembles fix
  it. The hook quantifies exactly the instability L09 warned about. Geometric limits
  (axis-aligned boxes, poor extrapolation) are INHERITED here, not new.
- **L01c** (`03_data_preprocessing.tex`): RF needs no scaling (scale-invariant); but
  in sklearn categoricals still need encoding.
- **L01d** (`04_overfitting_cross_validation.tex`): CV variance reduction by ~1/k is
  the NAIVE-HOPE anchor for "why averaging helps" (corrected by `rho`); OOB ~ a
  built-in held-out set; pick knobs by CV.
- **Forward -> L11/L12:** boosting cuts BIAS by sequential correction (vs RF cutting
  VARIANCE by independent averaging); more trees is SAFE in RF but CAN overfit in boosting.
- **Forward -> feature-selection lecture:** permutation importance + SHAP live there.

## Frame-by-frame outline (~21 content frames + outline + 3 section transitions)

### Hook (before outline)
1. **Single trees are unstable (predict-first #3).** Callback L09. Refit a tree on a
   slightly perturbed Titanic sample; **predict-first**: "how much do the predictions
   change?" Reveal: a lot. Line: *"We'll kill this variance by averaging many trees."*
   Figure `rf_instability` (bar: how many test predictions flipped under resampling).

### Outline frame (`\tableofcontents`)

### Section 1 — Bagging: averaging many trees  [transition]
2. **Bagging = Bootstrap AGGregating.** Train many models on bootstrap samples, then
   combine. Bagging is GENERAL (any model); RF = bagging specialized to trees +
   per-split feature subsampling (foreshadow).
3. **Bootstrap sampling.** Sample n rows WITH replacement -> each tree sees a slightly
   different dataset. Small TikZ visual of one resample.
4. **Aggregate.** Regression = average predictions (Yerevan rent); classification =
   majority vote / average predicted probabilities (Titanic).
5. **By-hand: bootstrap composition + aggregation** (worked-numbers frame). Tiny dataset
   rows {1..6}. Draw **three** bootstrap samples (e.g. {2,2,3,5,5,6}, {1,1,3,4,4,6},
   {2,3,3,5,6,6}); for each, compute its in-bag set and its OOB leftovers (e.g. sample 1
   -> OOB {1,4}). One line: *"each sample leaves a couple of rows out — we'll use that."*
   Then, **given** each of the three trees' predictions (the fits are assumed, NOT
   derived here), aggregate: 3 votes -> majority; or 3 numbers -> mean. Note (M2): with
   only 6 rows the out-fraction varies; ~37% is the large-n limit (frame 9).
6. **Why averaging helps (intuition).** Picture: correlated errors don't cancel,
   independent ones do. Anchor (framed as the NAIVE HOPE): callback L01d — IF the trees
   were independent you'd cut variance by `1/M` (like CV's `1/k`); they are NOT
   independent, which frame 8 quantifies. **"Why trees" point:** bagging only helps
   HIGH-variance learners — bagging a stable, low-variance model (e.g. linear
   regression) barely moves it; that is why we bag DEEP (high-variance) trees.
7. **Predict-first #4: does averaging fix bias?** Predict-first: *"can averaging fix an
   UNDERFITTING (biased) model?"* Reveal: no — averaging attacks **variance** and leaves
   **bias** ~unchanged. (That is why the formula on the next frame has no bias term.)
   Pure intuition, no formula yet.
8. **The variance formula (boxed, color-coded).** State (NOT derive — say so on the
   slide): `Var = rho*sigma^2 + (1-rho)/M * sigma^2`, where `sigma^2` = variance of a
   SINGLE tree (held fixed here) and `rho` = average pairwise correlation between trees.
   Color-code: `(1-rho)/M * sigma^2` = *"the part averaging kills"* (push M up);
   `rho*sigma^2` = *"the floor"* — as `M -> inf`, `Var -> rho*sigma^2`. `armorange`
   caveat: lowering `rho` (next section) is not free — it slightly RAISES `sigma^2`
   (each tree gets worse), so there is an OPTIMAL amount of decorrelation, not
   `max_features=1`. This is the deck's hardest frame; it gets its own.
9. **The ~63/37 split (owner of the derivation).** Each bootstrap leaves ~37% of rows
   out: `(1 - 1/n)^n -> 1/e ~= 0.368` (the large-n limit; small n varies — cf. frame 5).
   Those are the OOB rows -> next frame.
10. **Out-of-bag (OOB) error.** Each row is scored only by the trees that never saw it
    -> a free held-out validation set, no separate split. `oob_score=True`. Comparable
    to a held-out estimate, slightly pessimistic (each row is scored by only ~37% of the
    trees). (Note: OOB is a *bagging* property; taught here while the 37% is hot, though
    it is often presented as an RF feature.)

### Section 2 — From bagging to Random Forest  [transition]
11. **Bagged trees are correlated.** The same strong feature dominates the top split in
    every tree -> `rho` stays high -> the `rho*sigma^2` floor (frame 8) stays high. We
    must DEcorrelate the trees.
12. **Random Forest (Breiman) - predict-first #1: fewer features per split.**
    Boxed definition: *Random Forest = a bag of trees where each split may only use a
    RANDOM subset of `F` features.* Predict-first: "does restricting each split make the
    trees better or worse?" Students predict worse. Reveal: each tree IS individually
    slightly worse (higher `sigma^2`), but they DISAGREE more (lower `rho`) -> the floor
    drops faster than `sigma^2` rises -> the average wins. (This is exactly the
    `rho`-down / `sigma^2`-up trade the formula warned about on frame 8.)
13. **How many features per split?** Recommended `F = sqrt(P)` (classification),
    `P/3` (regression). `armorange` watch-out: classifier sklearn default `max_features="sqrt"`
    MATCHES the recommendation; **regressor sklearn default = all features
    (`max_features=1.0`, since v1.1)** — does NOT match `P/3`, so you often lower it by
    hand. VERIFY both at build.

### Section 3 — Random Forest in practice  [transition]
14. **Ensemble size - predict-first #2: more trees never overfits.** Predict-first:
    test error as M grows? Reveal: variance drops then **plateaus** at `~ rho*sigma^2`
    (annotate the curve, tying it to frame 8). Precise claim: the *number of trees* does
    not overfit (Breiman: error converges as M->inf); deeper trees / noisy features
    still can. Explicit contrast: **boosting CAN overfit** with more trees (foreshadow
    L11). Figure `rf_n_estimators` (accuracy / OOB vs `n_estimators` on **Titanic**).
15. **Feature importance (default only, brief).** sklearn impurity-based
    `.feature_importances_` bar chart on Titanic (sex / fare dominant), value labels.
    One-line caveat: biased to high-cardinality / correlated features. `paramgreen`
    forward-pointer: *permutation importance + SHAP get their own lecture later.*
16. **sklearn in practice.** `[fragile]` `RandomForestClassifier` / `RandomForestRegressor`
    on Titanic. No scaling needed (callback L01c); categoricals still need encoding.
17. **Key knobs.** `n_estimators` (more = better then plateau, costs compute),
    `max_features` (the decorrelation knob), `max_depth` / `min_samples_leaf` (per-tree
    complexity), `oob_score`, `n_jobs` (trees are independent -> embarrassingly
    parallel), **`class_weight="balanced"`** (Titanic is imbalanced; callback L09).
    Param table. **Extra Trees one-liner:** `ExtraTreesClassifier` picks split thresholds
    at RANDOM and (by default) does NOT bootstrap -> faster, more randomization,
    sometimes lower variance.
18. **Strengths & weaknesses** (two-column). + strong default, robust, parallelizable,
    OOB built-in, little tuning needed. - less interpretable than a single tree,
    memory / inference cost; **geometric limits INHERITED from trees** ("averaging
    boxes still gives boxes" -> still axis-aligned; extrapolation still poor).
    **Proximities one-liner:** RF also yields a sample-similarity matrix (how often two
    rows share a leaf) -> outlier detection / missing-value imputation.
19. **Bridge -> boosting (L11).** Introduce the recurring **parallel-vs-sequential
    trees diagram** (BUILT here; L11's hook reuses it): RF cuts **variance** by
    averaging INDEPENDENT trees; boosting cuts **bias** by SEQUENTIAL correction.
    Figure `rf_parallel_vs_sequential` (TikZ).

### Wrap-up (outside the sections)
20. **Recap** + `paramgreen` workflow box (RF = strong, low-tuning default; knobs that
    matter: `n_estimators`, `max_features`; OOB for free validation).
21. **HW (running project).** Titanic RF stage: fit a `RandomForestClassifier`, compare
    **OOB vs CV**, compare to the **L09 single-tree** result on the SAME score (use a
    proper metric for imbalanced Titanic — F1 / ROC-AUC, not raw accuracy; see build-time
    TODO), read off default importances. **Regression:** `RandomForestRegressor` on the
    Yerevan toy; plot error vs `n_estimators` to see the plateau. **Bonus:** try
    `ExtraTreesClassifier`; sketch why averaging cannot beat the `rho*sigma^2` floor.

## Decisions resolved at review (2026-06-23)
1. Proximities: **cut to a one-liner** folded into the strengths frame (18). No dedicated frame.
2. Ensemble-size viz (frame 14): **Titanic** (canonical through-line).

## Figures (`py_src/make_figures.py` -> `fig/`, `ma` venv)
- `rf_instability` — bar of "how many test predictions flipped" when a tree is refit
  on a slightly perturbed Titanic sample (frame 1). (Preferred over two rendered trees.)
- `rf_n_estimators` — accuracy / OOB error vs `n_estimators` on Titanic, showing the
  plateau; annotate the plateau height `~ rho*sigma^2` (frame 14).
- `rf_importance` — default impurity importances bar chart on Titanic, value labels (frame 15).
- TikZ schematics (no matplotlib): bootstrap resample (frame 3), correlated-vs-
  independent errors (frame 6), the variance-floor color-coding (frame 8), OOB routing
  (frame 10), `rf_parallel_vs_sequential` bridge diagram (frame 19, reused in L11).
- Data: Titanic via `fetch_openml` (as in L09's `make_figures.py`) + the synthetic
  Yerevan rent-vs-area toy. Seed 509. Armenian-flag palette for multi-series plots.

## Sources (carried from prior research; re-verify library facts at build)
- **ml_old:** back half of `L06 Decision Tree + Random Forest.pdf` — bagging, averaging
  M models, why bagging helps, RF (Breiman: random feature subset per split),
  ensemble-size effect, proximities. Code: `0x_log_reg__trees__rf.ipynb`.
- **Upstream (raw):** `ml_old/slides-i2ml-301-598.pdf` (forests chapter).
- **Web:** Harvard CS109A Bagging & RF; mlcourse.ai Topic 5. Facts: bootstrap keeps
  ~63% in-bag / ~37% OOB per tree; OOB = built-in validation; importance = MDI (impurity).
- **VERIFIED at review (sklearn ensemble docs):** `RandomForestClassifier` default
  `max_features="sqrt"`; `RandomForestRegressor` default `max_features=1.0` (= ALL
  features) since v1.1 — NOT P/3. `ExtraTrees*` default `bootstrap=False` + random
  split thresholds. Re-confirm at build via Context7. Also: `oob_score`, `n_jobs`,
  `class_weight`.
- **Variance formula** `Var = rho*sigma^2 + (1-rho)/M*sigma^2`: ESL eq. 15.1 (mean of M
  identically distributed vars, pairwise correlation rho; `sigma^2` = per-tree
  variance). `Var -> rho*sigma^2` as `M -> inf`. Stated, not derived.
