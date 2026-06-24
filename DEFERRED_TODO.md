# DEFERRED TOPICS — DO NOT FORGET

Things explicitly punted out of the main lecture flow. Park here so they don't get lost. Revisit when course pace allows or when student level is ready.

Last updated: 2026-06-19

---

## From bias-variance deck (`ml/upcoming_lectures/L01d2_bias_variance.tex`)

The full deck got cut. Status of the 3 surviving concepts (as of 2026-06-19):

1. **Irreducible error / Bayes risk** — adopted into L03 regularization (see `L03_OUTLINE.md`). Becomes the "noise floor" frame in the "Why regularize?" section. ✅
2. **Approximation vs Estimation error** — adopted into L03 regularization. Replaces the existing "model bias / estimation bias" frames using the canonical names. ✅
3. **Double descent** — **DEFERRED FURTHER.** Originally planned for L03, but user wants it pushed back. Natural home: the neural networks chapter, where over-parameterization is the rule and double descent stops feeling like a wrinkle and starts feeling like the regime. Frames stay drafted in `L01d2_bias_variance.tex` under `\section{Modern wrinkle: double descent}` — lift them when ch5 is built.

**Action:** after L03 lifts concepts 1 and 2, archive the rest of `L01d2_bias_variance.tex` — only the double descent section needs to survive for later.

---

## From `ml/deferred/` folder

Pre-built decks parked because they need more statistics background than the current cohort has. Each is mostly compile-ready.

### `deferred_glms.tex` — Generalized Linear Models (29 frames)

Sections: Why GLMs / The GLM Framework / OLS as GLM / Logistic as GLM / Poisson regression / Estimation & inference (IRLS, deviance) / Family tree / Choosing the right GLM / Practical Python.

**When to revisit:** after logistic regression is taught in the classification chapter. GLMs are the natural unification — "logistic and linear are two faces of the same thing." Needs MLE comfort.

### `deferred_causal_inference.tex` — Causal Inference (37 frames)

Sections: Correlation != causation / RCTs / Potential outcomes (ATE) / DAGs (chain, fork, collider) / Observational strategies (regression, matching, propensity score, IVs) / Doing vs seeing / Backdoor criterion / Ladder of causation / Practical checklist.

**When to revisit:** could be a standalone bonus lecture once the prediction chapter is solid. High value, very different mental model from prediction. Pair with a Simpson's-paradox demo.

### `deferred_regression_inference.tex` — Regression Inference / Coefficient Significance (31 frames)

Sections: Linear model assumptions / Sampling distribution of beta-hat / t-test for coefficients / Confidence intervals / F-test / Prediction intervals vs confidence intervals / Diagnostics for assumptions.

**When to revisit:** this is the *statistics* side of regression (p-values, CIs, hypothesis tests) that we're currently skipping. Ties to the broader stat lectures in `math/Lectures/stat/`. Could insert between L01c and L01d as an optional sidebar, OR fold into a dedicated "regression for inference vs regression for prediction" lecture.

---

## Other deferred ideas (collected over sessions)

- **1-SE rule** (Tibshirani): pick simplest model within 1 SE of best CV score. Removed from L01d. Reintroduce when regularization is taught.
- **LOOCV** (leave-one-out CV): removed from L01d. Mention as the k=n boundary case when discussing CV bias-variance.
- **Out-of-bag (OOB) error**: parallel to CV via bootstrap. Natural to introduce alongside random forests.
- **Bootstrap as practical bias/variance measurement.** Closes the loop between theory and lab.
- **Repeated K-fold motivation:** brief frame on "when CV scores are jumpy, repeat the CV."
- **Hyperband / successive halving** (`HalvingGridSearchCV`): visualize the bracket. For L01e.
- **Quantile / pinball loss** for non-mean prediction. For L01f.
- **MASE** for time-series metrics. For L01f.
- **Binning / discretization** as cheap nonlinearity for linear models. For L01g.
- **Mutual information filter** (`mutual_info_regression`). For L01h.
- **mRMR** feature selection. For L01h.
- **Selection vs PCA distinction** — interpretable vs not. For L01h.

---

## Cross-deck recurring idea

A **single worked example threaded across L01d / L01d2 / L01e / L01f / L01h** (rental prices? bike-share counts?). Each deck references back: "remember the bias-variance plot from L01d2 — this is what the validation curve from L01d looks like for that case." Heavy refactor, defer until decks are otherwise stable.

---

## Housekeeping deferred

- **Migrate or delete the auto-memory folder.** As of 2026-06-19 we switched to "all persistence in repo files, never write memory." But ~13 pre-existing memory files still live at `~/.claude/projects/C--Users-hayk--OneDrive-Desktop-01-python-math-ml-course/memory/` (user role, feedback rules, pedagogy notes, course completion status, etc.). Decide per file: copy still-useful content into `CLAUDE.md` / `CONVENTIONS.md` / `LEARNINGS.md`, then delete the memory folder. Until done, future sessions may still load those memories as context.

---

## Planned: Diagnostic plots section (for `[12]` metrics, or a standalone)

A "what to actually plot when debugging a classifier" section. Some plots already live in L12
(ROC, PR, lift, threshold-metric curves, confusion matrices) and calibration is in `[13]`; this
section would consolidate the visual toolkit and add the missing pieces. Proposed frames (each a
real figure via a `py_src/diagnostics_demo.py` on the cheese model):

1. **Score distribution by class** — histogram/KDE of predicted scores split by true label.
   Shows class separation/overlap and whether any threshold cuts cleanly. *(new figure)*
2. **Confusion-matrix heatmaps** — binary + the `K×K` `cm_multiclass.pdf` (already built). *(reuse)*
3. **Per-class metrics bar chart** — precision/recall/F1 per class with support; surfaces the
   weak class. *(new figure)*
4. **Error analysis** — the most confident *wrong* predictions / hardest examples (small table or
   annotated scatter): "look at what it gets wrong." *(new figure/table)*
5. **Learning curve** — train vs validation score vs training-set size → over/underfitting
   diagnosis (more data vs more model). *(new figure)*
6. Cross-reference the curves already in the deck (ROC / PR / lift / threshold) and reliability in `[13]`.

Scope ~5 frames + a transition. Decide whether it lives at the end of `[12]` or as its own short deck.

---

## How to use this file

- Top-level of repo means it appears in every `ls` and every `git status`. Visible by design.
- When a deferred topic gets adopted into a real lecture, DELETE that line (don't strike through — keeps the file short).
- Add new deferred items here the moment they get cut from a deck. The longer you wait, the more you forget the context.
