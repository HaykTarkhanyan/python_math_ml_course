# L12 XGBoost / LightGBM / CatBoost — Deck Outline / Design (DRAFT v3)

Planning doc for the **gradient-boosting libraries** deck, house style of
`ml/02_main_concepts/04_overfitting_cross_validation.tex` (the
validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> STATUS: DRAFT v3 (pedagogical review folded in). Spine stable; pending go-ahead.
> Lecture-numbering note: "L01d" = `04_overfitting_cross_validation.tex`;
> "L01c" = `03_data_preprocessing.tex`.

## Chapter-level conventions (shared across L09-L12, recorded in each file)
- **Canonical datasets:** Titanic (classification) + synthetic Yerevan rent-vs-area
  (regression). For the library bake-off, reuse **Titanic** so scores are comparable
  to L09/L10/L11 — do not introduce a fresh Kaggle set just for this deck.
- **Running-project HW:** the chapter's final stage — tuned LightGBM on Titanic,
  compared against the single tree (L09), RF (L10), and hand-built GBM (L11).
- **Math-depth policy:** practical/usage-focused; algorithm internals stated as
  ONE-LINERS framed by the problem they solve, not derivations.
- **Verification debt:** verify all library-specific facts at BUILD time via Context7.

## Scope
The production gradient-boosting libraries. Practical, comparison-driven. Assumes
L11 (gradient boosting concept). ONE signature idea per library taught deeply; the
rest demoted to the comparison table. Closes with a **stacking/ensembling capstone**
(the third ensemble strategy, capping the whole trees chapter). Follows L11.

## L11 / L12 boundary (fixed in v2, held in v3)
- **L11** owns: the algorithm + GENERAL regularization (M / depth / shrinkage).
- **L12** owns: each library's SIGNATURE innovation + library-specific knobs
  (`num_leaves` vs `max_depth`, early-stopping APIs, native categorical handling)
  + a real bake-off + a tuning-order recipe. DO NOT re-teach shrinkage — reference L11.

## Sources
- **ml_old**: `L07 Boosting.pdf` front-slide library links; conceptual base in L11.
- **Context7 (verified)**: LightGBM sklearn API defaults `boosting_type="gbdt"`,
  `num_leaves=31`, `max_depth=-1`, `learning_rate=0.1`, `n_estimators=100`,
  `min_child_samples=20`. LightGBM = leaf-wise (knob `num_leaves`); XGBoost =
  depth-wise (knob `max_depth`).
- **VERIFY at build (Context7)**: XGBoost 2nd-order/Newton objective + missing-value
  handling; CatBoost ordered boosting + ordered target statistics (leakage); current
  default params for each.
- **Web**: AnalyticsVidhya GB-vs-AdaBoost-vs-XGBoost-vs-CatBoost-vs-LightGBM;
  XGBoost-vs-LightGBM comparisons.

## Design note (the key call for this deck)
ONE signature idea per library, taught so it sticks; everything else (GOSS/EFB/
histogram/Newton/parallel) is a ONE-LINER demoted to the comparison table. A
feature-dump is trivia that dates fast and is the least retainable format.

## Proposed spine (~18-20 frames, L01d style — lighter/practical)

### Hook
1. Gradient boosting dominated tabular ML / Kaggle for a decade. Here are the
   libraries everyone actually ships.

### Section 1 — XGBoost (signature idea: regularization in the objective)
2. The idea that made it win Kaggle: explicit regularization baked into the training
   objective (penalize tree complexity directly). Depth-wise growth (knob `max_depth`).
   One-liners (-> comparison table): 2nd-order/Newton approximation, missing-value
   handling, parallel/cache-aware speed.

### Section 2 — LightGBM (signature idea: speed on big data)
3. Leaf-wise growth (callback L09 single-tree LightGBM) + histogram binning = fast and
   memory-light on large data. Knob = `num_leaves` (not `max_depth`). One-liners (->
   table): GOSS, EFB, native categorical handling.

### Section 3 — CatBoost (signature idea: no categorical leakage)
4. Ordered target statistics: encode a categorical using only PAST rows' targets so
   the label never leaks into its own feature. **Explicit callback to L01c/L01d**:
   "remember mean-target encoding leaks the label? this is the principled fix." Plus
   strong defaults. (If we cut scope: keep "handles categoricals safely, strong
   defaults, try it" without claiming to teach ordered boosting.)

### Section 4 — Comparison
5. depth-wise vs leaf-wise visual + side-by-side table (growth, categorical handling,
   speed, default quality, when to use which) — this table holds all the demoted
   one-liners.

### Section 5 — Practical tuning
6. **`num_leaves` vs `max_depth` footgun box (v3):** LightGBM's `max_depth=-1` by
   default; the real knob is `num_leaves`; setting a small `max_depth` out of XGBoost
   habit can cripple it. (House-style footgun box, cf. L01d's cross_val_score sign-flip.)
7. **Tuning order recipe (new v3) — the highest-transfer frame:** (1) set a low
   learning rate + early stopping to fix #trees; (2) tune `num_leaves`/`max_depth`;
   (3) `min_child_samples`; (4) regularization `reg_lambda`/`reg_alpha`; (5)
   `subsample`/`colsample_bytree`. Early stopping on a validation set; CV (callback
   L01d/L05). General shrinkage/M/depth theory lives in L11 — reference, don't repeat.

### Section 6 — Ensembling, the complete picture (stacking capstone, NEW v4)
8. **The three ways to ensemble** — bagging (parallel, cuts VARIANCE -> Random Forest,
   L10) / boosting (sequential, cuts BIAS -> GBM, L11/L12) / **stacking** (combine
   DIFFERENT model types via a meta-learner). The ensemble family tree; closes the
   whole chapter arc.
9. **Voting / stacking / blending, concretely** — voting (hard = majority vote, soft =
   average predicted probabilities); **stacking** = train diverse base models, feed
   their OUT-OF-FOLD predictions as features to a small meta-model (OOF keeps it
   leakage-safe, callback L01d); blending = the simpler held-out variant. When it
   helps: DIVERSE base learners (e.g. a tree model + a linear model + KNN); usually
   small gains, but it is how many Kaggle competitions were won. Caveat: extra
   complexity for marginal gain.

### Section 7 — Wrap-up
10. Capstone: **GBMs still beat neural nets on most tabular data** (true + motivating).
    Recap + HW (running project: train XGBoost/LightGBM/CatBoost on Titanic with sane
    defaults, bake-off vs L09/L10/L11 scores, then tune one with early stopping + CV;
    bonus: stack/vote your L09-L12 models with a meta-learner and see if it beats the
    best single model).

## Open decisions (pending)
1. CatBoost: the one signature frame above (recommended) or fuller coverage?
2. Internals depth: one signature idea per library + table one-liners (recommended)
   or actually show the 2nd-order objective / GOSS / ordered boosting?

## Changelog (v2 -> v3, pedagogical review)
1. Rebuilt around ONE signature idea per library (XGBoost = regularized objective;
   LightGBM = leaf-wise+histogram; CatBoost = ordered target stats vs leakage); the
   rest demoted to the comparison table. Was still a feature-dump in v2.
2. Added a dedicated tuning-order recipe frame (the highest-transfer practical skill).
3. Added the `num_leaves` vs `max_depth` footgun box.
4. CatBoost's signature now grounded in an explicit L01c/L01d target-leakage callback.
5. Naming/file references updated; chapter-level conventions recorded.

## Changelog (v3 -> v4)
1. Added a **stacking / ensembling capstone** (new Section 6, frames 8-9): the three
   ensemble strategies (bagging / boosting / stacking) + voting/stacking/blending
   concretely, OOF leakage-safety (callback L01d), and when it helps. Caps the whole
   trees chapter arc. (Wrap-up renumbered to Section 7 / frame 10.)
2. HW gains a stacking/voting bonus over the L09-L12 models.
