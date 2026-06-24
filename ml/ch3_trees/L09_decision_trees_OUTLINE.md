# L09 Decision Trees — Deck Outline / Design

Design doc for rebuilding the **Decision Trees** lecture from scratch, in the
house style of `ml_new/02_main_concepts_continued/04_overfitting_cross_validation.tex`
(the validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> Revision history: v1 = first outline. v2 = self-review fixes + LightGBM
> forward-hook. v3 = pedagogical-review pass — KEPT the LightGBM single-tree frame
> (per instruction) and ACCEPTED L09 running long (no forced cuts). See changelogs.
>
> Lecture-numbering note: "L01d" callbacks = `04_overfitting_cross_validation.tex`;
> "L01c" = `03_data_preprocessing.tex`. The folder is migrating to numeric prefixes;
> short L01c/L01d labels kept below until the scheme is final.

## Locked decisions

- **Scope:** Decision Trees only (L09). Bagging / Random Forest / proximities go
  to a separate **L10** deck (the old combined deck `ml_old/Chapter 3 Trees/PDFs/
  L06 Decision Tree + Random Forest.pdf` splits across L09 + L10 in the new structure).
- **Style:** match `L01d` — hand-drawn TikZ / pgfplots (no external images),
  Armenian-flag palette (`armblue` theory/train, `armred` test/warnings,
  `armorange` validation/highlight, `paramgreen` good/goal), `\fcolorbox{...}{...!8}`
  callout boxes, `\pause`-driven predict-first frames, hook frames before the
  `\tableofcontents`, Recap + HW frames, provenance comment block at the bottom.
  Preamble via `\input{../preamble}`, code via the `listings` `\lstset` from L01d.
- **Pedagogy:** follow the old deck's worked-example spine —
  majority vote -> decision stump -> add features -> recursive splitting ->
  *feature order matters* -> *which tree is better?* (coverage) -> smaller trees
  generalize -> therefore pick the feature that increases purity most ->
  Gini / Entropy / Information Gain -> CART -> pruning -> adv/disadv -> bridge.
- **Datasets (three, each for what it is best at):**
  - **Play Tennis** (14 rows: Outlook / Temperature / Humidity / Wind -> Play)
    for the hand-worked impurity arc. Information gains are textbook-clean and
    verifiable: **Gain(Outlook)=0.247, Humidity=0.152, Wind=0.048, Temperature=0.029**
    -> Outlook is the root. Keep it recognizable as Play Tennis, add an Armenian
    framing ("go out this weekend?") alongside the canonical version. Keep the
    canonical column labels visible (Armenian alongside, not replacing).
  - **Titanic** for the real-world sklearn section: mixed types (sex categorical
    split, age/fare numeric threshold splits), the 2D axis-aligned partition
    picture (age x fare), and feature importance. Full set for sklearn.
  - **Synthetic Yerevan rent vs area** (1D toy, ~20 points) for the regression-tree
    frame: a step-function fit. Doubles as a continuity callback to L01c/L01d.
    (NOTE: added in v2 — neither Play Tennis nor Titanic is a regression target,
    so the regression-tree frame needs its own data. If we want to cut scope,
    drop frame 12 to a single "trees do regression too" line instead.)
- **Libraries (added v2):**
  - **sklearn `DecisionTreeClassifier` / `DecisionTreeRegressor` is THE default**
    teaching tool: exact depth-wise CART, interpretable, `plot_tree` / `export_text`.
    All worked frames use it.
  - **LightGBM** is introduced as a *forward-hook* only (frame 23): the
    gradient-boosting library students will use properly in L11/L12. Point: a
    single tree is the atom; `LGBMRegressor(n_estimators=1, learning_rate=1.0)`
    fits one tree (conceptually a decision tree); bagging many trees -> Random
    Forest (L10); boosting trees -> LightGBM / XGBoost (L11/L12).
  - Verified LightGBM facts (Context7, lightgbm.readthedocs.io sklearn API):
    defaults `boosting_type="gbdt"`, `num_leaves=31`, `max_depth=-1`,
    `learning_rate=0.1`, `n_estimators=100`, `min_child_samples=20`. LightGBM
    grows **leaf-wise** (complexity knob = `num_leaves`, not `max_depth`) and bins
    features into histograms -> a single LightGBM tree is NOT byte-identical to
    sklearn CART. Use `learning_rate=1.0` with `n_estimators=1` so the single
    tree's output isn't shrunk by 0.1.
- **File:** write the deck to `ml_new/ch3_trees/L09_decision_trees.tex`, overwriting
  the empty skeleton (ignored per instruction). Compile from inside `ch3_trees/`.

## Callbacks / cross-references to thread in

- **L01c** (data preprocessing, `03_data_preprocessing.tex`): trees need no scaling
  (scale-invariant); but in **sklearn** categoricals still need encoding (sklearn
  trees are NOT auto-categorical).
- **L01d** (validation & CV, `04_overfitting_cross_validation.tex`): `max_depth` is the "complexity knob" (its table
  literally lists "Decision trees -> maximum depth"); the depth train/test U-curve
  is the overfitting fingerprint; pick depth / `ccp_alpha` by cross-validation.
- **Forward ref -> L10/L11/L12**: single trees are unstable (high variance);
  ensembles fix it — bagging -> Random Forest (L10); boosting -> LightGBM/XGBoost (L11/L12).

## Frame-by-frame outline (~25 frames)

### Hook (before outline)
1. **The weekend decision** — Play Tennis table + a new day with `???`. "How would
   you decide?" (Armenian "go out this weekend?" framing alongside canonical Play Tennis.)
2. **Predict from one feature** — distribution table on a single feature (Outlook:
   Overcast is pure Yes), majority vote per value -> define the **decision stump**
   (a one-question tree = 1 decision node) DELIBERATELY (the term is reused for
   AdaBoost stumps in L11).

### Outline frame (`\tableofcontents`)

### Section 1 — What a tree is
3. Add a second feature -> a small tree built by NAIVE fixed order; show if/else
   rules <-> TikZ tree are equivalent. (Naive order on purpose, sets up frame 6.)
4. Anatomy: root / internal nodes (questions) / branches / leaves (predictions) /
   depth; prediction walks root -> leaf (trace one day).
5. Grow the full tree recursively (still fixed feature order); predict the test day.

### Section 2 — Why we need impurity
6. **Feature order matters** — two split orders -> two different trees -> can give
   different predictions for the same new day.
7. **Predict-first: which tree is better?** Coverage argument — a leaf backed by
   many rows beats a leaf backed by 1. Smaller / higher-coverage trees generalize.
   (Seed of overfitting; callback forward to S5.)
8. So: greedily pick the feature that increases purity the most -> we must be able
   to *measure* purity.

### Section 3 — Impurity measures
9. **Gini** impurity: `2p(1-p)` binary, `1 - sum p_i^2` multiclass; meaning
   (expected misclassification if leaf labelled randomly). pgfplots curve.
10. **Entropy + Information Gain**: entropy formula; IG = parent impurity − weighted
    children impurity; meaning (bits). Curve overlaid on Gini.
11. **Worked split selection** on Play Tennis: compute IG for Outlook / Humidity /
    Wind / Temperature -> Outlook wins (0.247) and becomes the root.
    **Gini-vs-IG note (v2):** the 0.247-style numbers are entropy/IG; sklearn defaults
    to **Gini** -> same winner here, near-identical trees. State this explicitly so
    the deck does not silently switch criteria between frame 11 and frame 19.
12. **Regression trees**: SSE / variance-reduction split, leaf predicts the mean;
    step-function visual on the **synthetic Yerevan rent-vs-area** toy.
    **Predict-first (added v3):** what does a regression tree predict for an x
    BETWEEN two training points? Reveal: the leaf mean — a flat step, not a smooth
    interpolation. Pre-empts the smoothness misconception and seeds L11's residual
    example. (L09 may run long — this gets proper space, not one cramped frame.)

### Section 4 — Tree growing (CART)
13. **CART algorithm**: root of all data -> best split (min impurity / empirical risk)
    -> recurse on children -> stop criterion. Pseudocode box.
    **Numeric threshold splitting (added v2):** for a numeric feature, sort values,
    try thresholds at midpoints between neighbors, score each by impurity decrease,
    keep the best (this is the "if area > 60" mechanic L01d assumes; Play Tennis is
    all-categorical so it is never shown otherwise). Small visual.
14. **Greedy heuristic**: finding the optimal tree is NP-hard; CART takes the best
    split now and never reconsiders; purity-increase is the proxy for final size.

### Section 5 — Regularization (overfitting + pruning)
15. **Predict-first: can a tree reach 0 training error?** Yes — one leaf per point ->
    memorizes -> overfits. Depth train/test U-curve (callback L01d knob table).
16. **Pre-pruning** (stopping rules): `max_depth`, `min_samples_split/leaf`,
    `max_leaf_nodes`, `min_impurity_decrease`. Param table.
17. **Post-pruning**: grow full then remove weak subtrees; reduced-error pruning on
    a hold-out; sklearn cost-complexity `ccp_alpha` (error-vs-alpha path).
18. **Pre vs post + choose by CV**: do both; pre is fast / needs no extra data, post
    is slower / needs data but can undo bad greedy splits. Pick depth / `ccp_alpha`
    via cross-validation (callback L01d).

### Section 6 — In practice & bridge
19. **sklearn DecisionTree = the default** (Titanic): `DecisionTreeClassifier` on
    sex/pclass/age/fare; `plot_tree` / `export_text`; tree picks sex then an age
    threshold. `[fragile]` code. No scaling needed (callback L01c); **categoricals
    still need encoding** (sklearn trees are not auto-categorical — corrected v2).
20. **Decision boundary = axis-aligned rectangles** (tree <-> boxes), Titanic
    age x fare partition. (v2: this frame is now ONLY the positive partition picture.)
21. **Strengths & weaknesses** (two-column).
    + interpretable, scale-invariant (no scaling needed), captures interactions /
      non-linearities, automatic feature selection, fast.
    − high variance / **instability** (small data change -> very different tree;
      breast-cancer label-flip example), greedy/suboptimal, **axis-aligned only**
      (linear dependencies need many splits — staircase), **never smooth / poor
      extrapolation** (step function beyond training range).
    **sklearn-accuracy corrections (v2):** do NOT claim "automatic handling of
    non-numerical features" (false in sklearn — must encode) or "missing values via
    surrogate splits" (sklearn has no surrogates; it gained NaN support in v1.4 by
    routing missing to the impurity-minimizing child). Frame as "trees in principle
    vs what sklearn does."
22. **Feature importance** (brief) bar chart (sex/fare dominant), with value labels;
    caveat: impurity-based importance biased to high-cardinality features, preview
    permutation / SHAP.
23. **The tree is the atom + LightGBM forward-hook (new v2):**
    - sklearn single tree = our default for *learning and interpreting*.
    - LightGBM (you'll use it in L11/L12) is a gradient-boosting *library*, but you
      can fit ONE tree: `LGBMRegressor(n_estimators=1, learning_rate=1.0)` ~ a
      decision tree. Caveat box: LightGBM grows **leaf-wise** (`num_leaves`, not
      `max_depth`) + histogram bins -> not identical to sklearn CART; default
      `learning_rate=0.1` would shrink the single tree, so set it to 1.0.
    - Atom view: 1 tree -> bagging many -> **Random Forest (L10)**; boosting trees
      sequentially -> **LightGBM / XGBoost (L11/L12)**. "Everything in this chapter
      is built from the tree you just learned."

### Wrap-up
24. **Recap** + paramgreen workflow box.
25. **HW** — by-hand: build a stump + compute IG for one Play Tennis split.
    sklearn: fit a Titanic tree, plot the depth U-curve, prune via CV `ccp_alpha`,
    visualize the tree, confirm scale-invariance. Bonus: feature importance,
    compare to a logistic-regression baseline, and refit the single tree as
    `LGBMRegressor(n_estimators=1, learning_rate=1.0)` to see it match. Further
    reading (r2d3 visual intro, etc.).

## Tightening options (if 25 is too long)
- Merge 7+8 (coverage -> purity motivation) into one frame.
- Merge 13+14 (CART algorithm + greedy/NP-hard) into one frame.
- Merge 17+18 (post-pruning + pre-vs-post/CV) into one frame.
- Merge 22 into 21 (feature importance as part of strengths) to keep the LightGBM
  frame without growing the count.

## Sources
- Play Tennis IG numbers: GeeksforGeeks (Decision Tree example); ID3 Play Tennis
  worked example (Studocu); CMU 10-401 Decision Trees lecture (Balcan).
- sklearn categorical/missing-value behavior: scikit-learn tree docs
  (scikit-learn.org/stable/modules/tree.html); Release Highlights 1.4
  (NaN support for DecisionTree, splitter='best').
- LightGBM defaults + leaf-wise growth: Context7 / lightgbm.readthedocs.io
  (sklearn API LGBMClassifier/LGBMRegressor; Parameters-Tuning).

## Changelog (v1 -> v2)
1. Advantages frame (21): removed/ corrected two R/rpart claims that are false for
   sklearn (auto-categorical handling; missing via surrogate splits).
2. Regression-tree frame (12): added an explicit third dataset (synthetic Yerevan
   rent-vs-area). Was silently missing in v1's "two datasets" plan.
3. Frame 11: added explicit Gini-vs-Information-Gain reconciliation note.
4. Frame 13: added the numeric threshold-splitting mechanic (sort + midpoints).
5. Frame 20/21: split overloaded frame — 20 is now only the partition picture;
   staircase + extrapolation moved into the disadvantages frame (no double-coverage).
6. New frame 23: LightGBM forward-hook ("tree is the atom", single-tree LightGBM,
   bagging/boosting preview, sklearn as default). Per user request, Context7-verified.

## Changelog (v2 -> v3, pedagogical review)
1. KEPT the LightGBM single-tree frame 23 (instructor override; the reviewer had
   suggested demoting the code to an HW bonus — NOT done).
2. L09 running long is ACCEPTED (instructor override); the "tightening options" above
   are now optional, not required.
3. Frame 2: define "decision stump" deliberately (reused in L11).
4. Frame 12: regression trees upgraded with a predict-first (flat-step vs smooth) to
   pre-empt the smoothness misconception and seed L11's residual example.
5. File references updated: L01d -> 04_overfitting_cross_validation.tex,
   L01c -> 03_data_preprocessing.tex.
6. HW stays consistent with the chapter running-project (single tree stage on Titanic).
