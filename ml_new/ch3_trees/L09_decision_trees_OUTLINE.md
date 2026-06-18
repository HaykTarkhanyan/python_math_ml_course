# L09 Decision Trees — Deck Outline / Design

Design doc for rebuilding the **Decision Trees** lecture from scratch, in the
house style of `ml_new/02_main_concepts_continued/L01d_validation_and_cv.tex`.

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
- **Datasets (two, each for what it is best at):**
  - **Play Tennis** (14 rows: Outlook / Temperature / Humidity / Wind -> Play)
    for the hand-worked impurity arc. Information gains are textbook-clean and
    verifiable: **Gain(Outlook)=0.248, Humidity=0.151, Wind=0.048, Temperature=0.029**
    -> Outlook is the root. Keep it recognizable as Play Tennis, add an Armenian
    framing ("go to Sevan / khorovats this weekend?") alongside the canonical version.
  - **Titanic** for the real-world sklearn section: mixed types (sex categorical
    split, age/fare numeric threshold splits), the 2D axis-aligned partition
    picture (age x fare), and feature importance. Curate a ~14-row subset for any
    by-hand part; full set for sklearn.
- **File:** write the deck to `ml_new/ch3_trees/L09_decision_trees.tex`, overwriting
  the empty skeleton (ignored per instruction). Compile from inside `ch3_trees/`.

## Callbacks / cross-references to thread in

- **L01c** (data preprocessing): trees need no scaling (scale-invariant), but
  categoricals still need encoding for sklearn; handles missing/outliers.
- **L01d** (validation & CV): `max_depth` is the "complexity knob" (its table
  literally lists "Decision trees -> maximum depth"); the depth train/test U-curve
  is the overfitting fingerprint; pick depth / `ccp_alpha` by cross-validation.
- **Forward ref -> L10**: single trees are unstable (high variance); ensembles
  (bagging -> Random Forest) fix it.

## Frame-by-frame outline (~24 frames)

### Hook (before outline)
1. **The weekend decision** — Play Tennis table + a new day with `???`. "How would
   you decide?" (Armenian "go to Sevan?" framing alongside canonical Play Tennis.)
2. **Predict from one feature** — distribution table on a single feature, majority
   vote per value -> introduce the **decision stump** (tree with 1 decision node).

### Outline frame (`\tableofcontents`)

### Section 1 — What a tree is
3. Add a second feature -> a small tree; show if/else rules <-> TikZ tree are equivalent.
4. Anatomy: root / internal nodes (questions) / branches / leaves (predictions) /
   depth; prediction walks root -> leaf (trace one day).
5. Grow the full tree recursively (fixed feature order); predict the test day.

### Section 2 — Why we need impurity
6. **Feature order matters** — two split orders -> two different trees -> can give
   different predictions for the same new day.
7. **Predict-first: which tree is better?** Coverage argument — a leaf backed by
   many rows beats a leaf backed by 1. Smaller / higher-coverage trees generalize.
8. So: greedily pick the feature that increases purity the most -> we must be able
   to *measure* purity.

### Section 3 — Impurity measures
9. **Gini** impurity: `2p(1-p)` binary, `1 - sum p_i^2` multiclass; meaning
   (expected misclassification if leaf labelled randomly). pgfplots curve.
10. **Entropy + Information Gain**: entropy formula; IG = parent impurity − weighted
    children impurity; meaning (bits). Curve overlaid on Gini.
11. **Worked split selection** on Play Tennis: compute IG for Outlook / Humidity /
    Wind / Temperature -> Outlook wins (0.248) and becomes the root. (Verifiable.)
12. **Regression trees (brief)**: SSE / variance-reduction split, leaf predicts the
    mean; step-function visual on a small numeric set. Course-continuity nod to regression.

### Section 4 — Tree growing (CART)
13. **CART algorithm**: root of all data -> best split (min impurity / empirical risk)
    -> recurse on children -> stop criterion. Pseudocode box. Handles numeric
    (threshold) + categorical.
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

### Section 6 — In practice & bridge (Titanic)
19. **sklearn on Titanic**: `DecisionTreeClassifier` on sex/pclass/age/fare;
    `plot_tree` / `export_text`; tree picks sex then an age threshold. `[fragile]`
    code. No scaling needed (callback L01c); categoricals still need encoding.
20. **Decision boundary = axis-aligned rectangles** (tree <-> boxes), Titanic
    age x fare partition. Note: linear dependencies need many splits (diagonal
    staircase); never smooth, poor extrapolation (step function).
21. **Strengths & weaknesses** (two-column): + interpretable, minimal preprocessing,
    scale-invariant, handles missing/outliers, captures interactions, auto feature
    selection; − high variance / instability (breast-cancer label-flip example),
    greedy/suboptimal, axis-aligned only, overfits, poor extrapolation.
22. **Feature importance** (brief) bar chart (sex/fare dominant), with value labels;
    caveat: impurity-based importance biased to high-cardinality features, preview
    permutation / SHAP. **Bridge:** single trees unstable -> ensembles (Random
    Forest, L10). Forward ref.

### Wrap-up
23. **Recap** + paramgreen workflow box.
24. **HW** — by-hand: build a stump + compute IG for one Play Tennis split.
    sklearn: fit a Titanic tree, plot the depth U-curve, prune via CV `ccp_alpha`,
    visualize the tree, confirm scale-invariance. Bonus: feature importance,
    compare to a logistic-regression baseline. Further reading (r2d3 visual intro, etc.).

## Tightening options (if 24 is too long)
- Merge 7+8 (coverage -> purity motivation) into one frame.
- Merge 17+18 (post-pruning + pre-vs-post/CV) into one frame.

## Sources for the Play Tennis IG numbers
- GeeksforGeeks — Decision Tree example
- ID3 Play Tennis worked example (Studocu)
- CMU 10-401 Decision Trees lecture (Balcan)
