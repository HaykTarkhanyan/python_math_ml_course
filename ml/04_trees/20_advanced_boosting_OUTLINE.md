# Advanced Boosting (L12) - Deck Outline (v2, 2026-06-23)

House style per `ml/SLIDE_STYLE.md`. This is the "deep library session" that
`L11_boosting.tex` explicitly promised ("Bridge: the deep library session (L12)").

## Scope (from two interview rounds)
- **XGBoost-centered + trio comparison.** Deep on XGBoost (full math + solid engineering),
  then a LightGBM/CatBoost comparison and a tuning/HP tour. **Stacking is the chapter closer**
  (v3: section 5 - concept + out-of-fold + voting/blending + ensemble-landscape recap - appended
  to this deck per the "finish the chapter" request).
- **LightGBM gets expanded coverage** (overview, leaf-wise, GOSS, EFB) + a fun "Me and LightGBM"
  frame; it is the **instructor's favorite** (per request).
- **Math depth: full derivation** - regularized objective -> 2nd-order Taylor -> closed-form
  optimal leaf weight (structure score) -> split-gain formula, with a small numeric anchor.
  L2 is the worked path; L1 mentioned, not derived.
- **Engineering: solid section** - growing/pruning, histogram split finding (global vs local),
  sparsity-aware splits, subsampling, parallel/GPU, **DART**.
- **Extras included (round 2):** DART, feature importance / SHAP, monotonic & interaction
  constraints.
- **Hands-on:** HP + tuning tour, and a **LightGBM + XGBoost side-by-side** code snippet.
  NOT chosen: by-hand worked frame, predict-first frames, live bake-off figure.
- **Figures use synthetic tuned data** (`make_classification`) so the percentile bins,
  early-stop point, and importances read clearly. "Deeper where useful" is fine (~25 content frames).

## Boundary with L11 (no repeats)
L11 covered: AdaBoost, fit-the-residuals, function-space gradient descent, any loss, GB for
classification, the three knobs (eta / n_trees / depth), stochastic GB, sklearn
`GradientBoosting`. This deck assumes all of that and starts at the libraries.

## Sources (cite on-slide where borrowed)
- LMU `lecture_sl`: `slides-boosting-xgboost.tex`, `-xgboost-deepdive.tex`, `-lgm-ctbm.tex` (CC BY 4.0).
- Chen & Guestrin (2016) *XGBoost*; Ke et al. (2017) *LightGBM*; Prokhorenkova et al. (2018) *CatBoost*.
- LightGBM docs (leaf-wise tuning, `num_leaves`); Lundberg & Lee (2017) *SHAP*.

---

## Frame-by-frame (~25 content + outline + 4 transitions)

### Hook + bridge
1. **Cold-open.** "You built gradient boosting by hand (L11). Now the version that wins on
   tabular data." Same function-space gradient descent - made fast, regularized, production-ready.
   Name the trio: **XGBoost / LightGBM / CatBoost**.

[Outline frame]

### Section 1 - XGBoost: the regularized objective and the structure score  [transition]
2. **What XGBoost is.** Scalable C++ tree-boosting system: regularized objective, approximate
   split finding, subsampling, sparsity-aware, GPU. "Top tabular performance - if tuned."
3. **A regularized objective.** `risk = sum_i L(y_i, f + b) + gamma*T + (1/2)*lambda*||c||^2 (+ alpha*||c||_1)`.
   Penalize **#leaves T** and **leaf magnitudes** - complexity baked into the loss (contrast L11:
   no penalty inside the objective). Software names: `gamma`, `lambda` (L2), `alpha` (L1).
4. **Second-order Taylor.** `L(y, f + b) ~ L(y,f) + g*b + (1/2)*h*b^2`, gradient `g = dL/df`,
   Hessian `h = d^2L/df^2`. **`g` is the negative pseudo-residual from L11**; XGBoost adds the
   curvature `h`. (The one real step beyond L11.)
5. **Per-leaf objective.** Drop the constant, regroup by leaf `t`:
   `risk ~ sum_t [ G_t*c_t + (1/2)(H_t + lambda)*c_t^2 ] + gamma*T`, with `G_t = sum g_i`,
   `H_t = sum h_i`. A clean quadratic in each `c_t`.
6. **Optimal leaf weight = structure score.** `c_t* = -G_t/(H_t + lambda)`; plug back ->
   `obj* = -(1/2) sum_t G_t^2/(H_t + lambda) + gamma*T` - the **structure score**, scoring a tree
   *shape*. **Numbers:** `G=-3, H=4, lambda=1 -> c* = 0.6`. (L1 adds a soft-threshold; mentioned.)
7. **The split-gain criterion.** `gain = (1/2)[ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ] - gamma`.
   Split only if `gain > 0`; **`gamma` = minimum gain to keep a split**. What XGBoost maximizes
   per node - not Gini/variance.
   - **Numbers (anchor):** parent `(G=-3,H=4)`, `lambda=1`, split `L(-3,2)`/`R(0,2)` ->
     `gain = (1/2)[9/3 + 0/3 - 9/5] - gamma = 0.6 - gamma`; keep only if `gamma < 0.6`.
   - **Figure (TikZ):** node `t (G,H)` -> `L`,`R` with the same numbers beside the formula.

### Section 2 - Engineering: why it is fast  [transition]
8. **Tree growing + pruning.** Grow **level-wise** to `max_depth`, then **prune** splits with
   `gain < 0` (the `gamma` threshold). Depth-limited + post-pruned.
9. **Approximate / histogram split finding.** Bucket each feature into ~`l` **percentile bins**,
   evaluate only bin edges. **Global** (once per tree) vs **local** (per node). "Histogram-based
   gradient boosting" (XGBoost `tree_method="hist"`; default in LightGBM and sklearn HistGB).
   - **Figure (real, synthetic):** a feature's histogram with percentile bin edges + chosen split.
     {cite Chen & Guestrin 2016}
10. **Sparsity-aware splits.** Each split learns a **default direction**; missing/zero entries
    follow it. Native missing-value + sparse one-hot handling, and faster.
11. **Subsampling + parallel/GPU.** Row subsampling (`subsample`) + **column** subsampling
    (`colsample_bytree/bylevel/bynode`, like RF `mtry`). Boosting is sequential, but the
    **split search within a tree parallelizes** via sorted feature blocks; big GPU speedups.
12. **DART: dropout for trees.** Borrow dropout from deep learning: when adding a new tree,
    compute the update against a **random subset of existing trees** (drop the rest), then rescale
    so the ensemble does not overshoot. `p_drop=0` -> ordinary GB; `p_drop=1` -> all trees
    independent ~ random forest. A smooth GB<->RF dial (XGBoost `booster="dart"`). {cite LMU}

### Section 3 - The modern trio (LightGBM gets the spotlight)  [transition]
13. **LightGBM - the instructor's favorite.** Why it is the go-to for tabular work: very fast,
    low memory, excellent defaults, native categoricals, scales to large data. `paramgreen` note:
    *"My favorite - what I reach for first on tabular data."* {LightGBM; Ke et al. 2017}
14. **"Me and LightGBM `\heartsuit`" (fun interlude).** Full-bleed `fig/12_me_and_lightgbm.png`
    (instructor charging into battle, LightGBM in hand), **no caption**, just the image + clip
    link `https://youtu.be/UqmqClOZRwg`. Red heart (`\textcolor{armred}{$\heartsuit$}`) after
    "LightGBM" in the title.
15. **Leaf-wise (best-first) growth.** XGBoost grows level-wise then prunes; LightGBM splits the
    **single highest-gain leaf** anywhere -> unbalanced, deeper, converges faster. Catch: can
    **overfit**, so the main knob is **`num_leaves`** (`< 2^max_depth`) + `min_data_in_leaf`.
    - **Figure (TikZ):** level-wise (XGBoost) vs leaf-wise (LightGBM), chosen leaf highlighted.
      {cite LightGBM docs}
16. **GOSS - gradient-based one-side sampling.** Keep the **`a*n` largest-|gradient|** rows,
    sample **`b*n`** of the rest, **up-weight** them by `(1-a)/b` to stay unbiased. Defaults
    `a=0.2, b=0.1`. Fewer rows per split, little accuracy lost. {cite Ke et al. 2017}
17. **EFB - exclusive feature bundling.** Bundle **mutually exclusive** sparse features (never
    nonzero together, e.g. one-hot) into one histogram, nearly lossless -> build drops `O(n*p)`
    -> `O(n*#bundles)`. Greedy (optimal is NP-hard). {cite Ke et al. 2017}
18. **CatBoost (the categorical specialist).** **Ordered target statistics** (leakage-free
    category encoding) + **ordered boosting** (fixes prediction-shift bias) + **symmetric/oblivious
    trees** (fast, regularizing). Reach for it with many categoricals. {cite Prokhorenkova et al. 2018}
19. **Which one, when.** Decision table - growth / observation sampling / feature handling /
    categoricals / speed-on-large-n / "reach for it when". All three are the same gradient
    boosting; the differences are engineering + categorical handling. LightGBM the strong default,
    CatBoost for heavy categoricals, XGBoost the battle-tested baseline.

### Section 4 - Using them well  [transition]
20. **Hyperparameters that matter.** Grouped table: **learning** (`eta`/`learning_rate`,
    `n_estimators` + early stopping), **complexity** (XGB `max_depth`/`min_child_weight`;
    **LGBM `num_leaves`/`min_data_in_leaf`** - the knob differs!), **regularization**
    (`gamma`/`min_gain_to_split`, `lambda`, `alpha`), **randomness** (`subsample`, `colsample`/
    `feature_fraction`). Defaults: XGB `eta` 0.3 / `max_depth` 6; LGBM `lr` 0.1 / `num_leaves` 31.
21. **The tuning order that matters.** (1) big `n_estimators` + **early stopping**; (2) learning
    rate vs `n_estimators`; (3) tree complexity (`max_depth` XGB / `num_leaves` LGBM);
    (4) regularization; (5) subsampling. Random / Bayesian search (callback to the tuning deck).
    - **Figure (real, xgboost on synthetic):** train vs validation logloss over rounds, early-stop
      point marked.
22. **Inject domain knowledge: constraints.** **Monotonic constraints** (`monotone_constraints`):
    force a feature's effect to be non-decreasing/non-increasing (e.g. price up -> risk up) - more
    trust, free regularization. **Interaction constraints**: restrict which features may share a
    tree path. Use when you *know* the structure.
23. **Are they black boxes? Feature importance.** Built-in importances (gain / cover / frequency)
    are quick but **split-count importance is biased to high-cardinality features**; prefer
    **permutation importance**. Name-drop **SHAP** for per-prediction attribution but say *"we'll
    cover it later"* - no SHAP figure here.
    - **Figure (real, xgboost on synthetic):** gain-based feature-importance bar chart (value
      labels on bars).
24. **In code (minimal): LightGBM + XGBoost side by side.** Two short snippets, near-identical
    API (`LGBMClassifier` / `XGBClassifier`, both with early stopping). One-line note: sklearn
    `HistGradientBoosting` mirrors them with no extra install. Lead with LightGBM.

### Wrap-up
25. **Recap.** XGBoost = gradient boosting + regularized objective (structure score) + engineered
    speed; **LightGBM = leaf-wise + GOSS + EFB, the fast favorite**; CatBoost = categoricals done
    right; constraints + importance make them trustworthy, not black boxes. (No forward pointer -
    stacking is deferred, to be discussed later.)

---

## Figures (`py_src/make_xgb_figures.py` -> `fig/`, `ma` venv; xgboost installed, install `lightgbm`)
All real figures use one **synthetic** `make_classification` dataset (seed 509), fit once and reused.
- `12_xgb_histogram_split.pdf` - a feature histogram with percentile bin edges + chosen split.
- `12_xgb_earlystop.pdf` - xgboost train/validation logloss vs rounds, early-stop point marked.
- `12_xgb_importance.pdf` - gain-based feature-importance bar chart (labels on bars; bar not pie).
- (No SHAP figure - SHAP is deferred to a later session.)
- **TikZ in-deck:** structure-score split (frame 7), level-wise vs leaf-wise (frame 15).
- **Cut per review:** the live XGBoost-vs-LightGBM bake-off (unreliable on small data); the
  "favorite/fast" claim stays qualitative + cited. `lightgbm` is installed only to verify the
  frame-24 snippet runs (one-shot), plus the side-by-side code.

## Build notes
- File: `ml/ch3_trees/L12_advanced_boosting.tex`. Match `L11_boosting.tex` boilerplate
  (`\input{../preamble}`, `\sectiontransition` macro, `listings` setup, `% Provenance:` footer).
- Title: **Advanced Boosting**; subtitle: *XGBoost - LightGBM - CatBoost*.
- Compile twice, `clean_latex.py`, eyeball overflow-prone frames as PNG: math (5-7), tables
  (19, 20), the side-by-side code (24), and the full-bleed image (14, 2.23:1 -> letterboxes).
