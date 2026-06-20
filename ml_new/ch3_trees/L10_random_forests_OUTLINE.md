# L10 Random Forests — Deck Outline / Design (DRAFT v3)

Planning doc for the **Random Forests** deck, house style of
`ml_new/02_main_concepts_continued/04_overfitting_cross_validation.tex` (the
validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> STATUS: DRAFT v3 (pedagogical review folded in). Spine stable; pending go-ahead.
> Lecture-numbering note: "L01d" = `04_overfitting_cross_validation.tex`;
> "L01c" = `03_data_preprocessing.tex`. Short labels kept until numbering is final.

## Chapter-level conventions (shared across L09-L12, recorded in each file)
- **Canonical datasets:** **Titanic** (classification through-line) + **synthetic
  Yerevan rent-vs-area** (regression / residuals). Any other dataset must EARN its
  place; do not introduce a new one just for variety.
- **Running-project HW:** one Titanic task carried across the chapter — single tree
  (L09) -> random forest (L10) -> hand-built GBM (L11) -> tuned LightGBM (L12),
  comparing the SAME score each step. Each deck's HW is one stage. State per stage
  whether it is classification (Titanic) or regression (Yerevan toy).
- **Math-depth policy:** intuition + worked example in the body; heavier formalism
  in a boxed / optional frame. Hold across all four decks.
- **RF-vs-boosting contrast diagram:** one recurring "parallel trees (RF) vs
  sequential trees (boosting)" picture, introduced in this deck's bridge and reused
  in L11's hook, so variance-vs-bias is SHOWN, not just asserted.
- **Verification debt:** library defaults/facts verified at BUILD time.

## Scope
Bagging -> OOB -> Random Forest -> feature importance -> bridge. (Proximities = one
optional frame.) Follows L09.

## Sources
- **ml_old**: back half of `L06 Decision Tree + Random Forest.pdf` — bagging,
  averaging M models, why bagging helps, RF (Breiman: random feature subset per
  split), ensemble-size effect, proximities (now optional). Code:
  `0x_log_reg__trees__rf.ipynb`.
- **Upstream (raw)**: `ml_old/slides-i2ml-301-598.pdf` (forests chapter).
- **Web**: Harvard CS109A Bagging & RF; mlcourse.ai Topic 5. Facts: bootstrap keeps
  ~63% in-bag / ~37% OOB per tree; OOB = built-in validation; importance = MDI or
  permutation.
- **VERIFY at build (sklearn defaults):** `RandomForestClassifier` default
  `max_features="sqrt"` (= sqrt(P)); `RandomForestRegressor` default
  `max_features=1.0` (= ALL features) since v1.1 — NOT P/3. (sqrt(P) and P/3 are the
  Breiman/ESL *recommendations*, not the regressor's sklearn default.)

## Proposed spine (~22-24 frames, L01d style)

### Hook
1. Single trees are unstable (callback L09) — QUANTIFY it: refit on a slightly
   perturbed sample, count how many predictions flip / how different the top split
   is. "We will kill this variance by averaging."

### Section 1 — Bagging
2. Bootstrap sampling — sample n rows WITH replacement; train one model per sample.
   Note: bagging is GENERAL (any model); RF = bagging specialized to trees + per-split
   feature subsampling (foreshadow).
3. Aggregate — average (regression) / majority vote (classification).
4. **Intuition first (the predict-first, v3):** "averaging helps only if the things
   you average don't all make the SAME mistake." Hand-drawn picture: correlated
   errors don't cancel; independent ones do. Callback L01d frame 11 (CV variance
   reduction by ~1/k) as the anchor students already have.
5. **The formula (boxed, v3):** averaging M identically-distributed but CORRELATED
   estimators gives `Var = rho*sigma^2 + (1-rho)/M * sigma^2`, color-coded: the
   `(1-rho)/M` term = "the part averaging kills", the `rho*sigma^2` term = "the floor
   RF will attack". (Bias ~unchanged.)
6. The ~63/37 split — each bootstrap leaves ~37% of rows out (the OOB rows).

### Section 2 — OOB error
7. OOB error — **chain it to the 37% (v3):** "remember the ~37% left out for each
   tree? those trees never saw that row, so they form a free held-out validation set
   for it." Score each row only with the trees that didn't see it. `oob_score=True`.
   (Roughly ~3-fold-CV-like; slightly pessimistic.)

### Section 3 — From bagging to Random Forest
8. Bagged trees are correlated — the same strong feature dominates the top split in
   every tree -> the `rho*sigma^2` floor (frame 5) stays high.
9. **Random Forest (Breiman) + full predict-first (v3):** before the reveal, ask
   "if each tree may only consider a RANDOM subset of features per split, are the
   trees better or worse?" Students predict "worse." Reveal: each tree is individually
   slightly worse, but they DISAGREE more -> lower rho -> lower the floor -> the
   average wins. Recommended F=sqrt(P) (clf), P/3 (reg); **sklearn regressor default
   differs** (see Verify note).

### Section 4 — Ensemble size
10. More trees -> lower variance, then plateaus; adding trees does NOT overfit (key
    property; contrast with boosting in L11). Viz on Titanic.

### Section 5 — Feature importance (RF is the home for the full treatment)
11. MDI (mean impurity decrease) + **permutation importance**; MDI caveat: biased to
    high-cardinality features (L09 only teased; RF gives the full version). Bar chart
    with value labels. SHAP previewed (interpretability chapter).

### Section 6 — Practice, strengths/weaknesses, bridge
12. sklearn `RandomForestClassifier/Regressor` — `n_estimators`, `max_features`,
    `max_depth`, `oob_score`, `n_jobs`. `[fragile]` code on Titanic.
13. Strengths/weaknesses — strong default, robust, parallel, OOB built-in / less
    interpretable than a single tree, memory & inference cost. **Frame the geometric
    limits as INHERITED from trees (v3):** "averaging boxes still gives boxes" — still
    axis-aligned, extrapolation still poor (not new facts, inherited from L09).
14. (OPTIONAL, one frame) Proximities — RF similarity; outlier detection; missing
    imputation. Cuttable.
15. Bridge -> boosting (L11): introduce the **parallel-vs-sequential trees diagram**.
    RF cuts VARIANCE by averaging INDEPENDENT trees; boosting cuts BIAS by SEQUENTIAL
    correction. Recap + HW (running project: RF stage on Titanic — OOB vs CV, compare
    to the L09 single tree, importances).

## Open decisions (pending)
1. Ensemble-size viz: Titanic (canonical) or keep Iris from the old deck?
2. Proximities: one optional frame (current plan) or cut entirely?

## Changelog (v2 -> v3, pedagogical review)
1. Variance frame split into intuition (predict-first, frame 4) + boxed color-coded
   formula (frame 5) — was one un-scaffolded frame; this is the deck's hardest math
   for a mixed-background audience.
2. OOB (frame 7) now explicitly chains off the 37% (why OOB is honest), not asserted.
3. "Fewer features per split" promoted to a full predict-first frame (frame 9) — it
   is the RF "aha" and students' naive prior is the opposite.
4. Strengths/weaknesses (13): axis-aligned/extrapolation framed as inherited from
   trees, not new bullets.
5. Bridge (15): added the recurring parallel-vs-sequential diagram (shared with L11).
6. Naming/file references updated; chapter-level conventions recorded.
