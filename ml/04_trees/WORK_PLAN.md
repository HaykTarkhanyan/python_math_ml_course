# Trees chapter (ml/04_trees) - full-finish implementation plan

> **Execution:** This is deck/figure work, not TDD code. The "test" for each task is:
> compile twice from the deck dir, `grep -cE '^!' FILE.log` == 0, run `beamer-overflow-check`
> on any new/edited frame, then `clean_latex.py`. Follow the `slide-style`, `make-figures`,
> and `compile-deck` skills. Steps use `- [ ]` for tracking.

**Goal:** Take the trees chapter from "strong drafts with known defects" (see `REVIEW.md`) to a
delivery-ready chapter: renumbered to the real video slots, defects fixed, every stated-in-words
concept given its figure, HW moved to the qmd, data pinned, project notebook built.

**Approach:** Work in the priority order from `REVIEW.md` §6, front-loading the mechanical
renumber (it touches every later task's cross-references) and deferring figure generation until
the frame structure is settled. Every essential visual is Python-generated into `fig/` via a
`py_src/` script (house rule), never TikZ except small schematics.

**Decisions locked (2026-07-10, instructor):**
- **Numbering: trees videos start at [17].** `L09 -> [17]`, `L10 -> [18]`, `L11 -> [19]`,
  `L12 -> [20]`. The ch3 bank project keeps `16_`. `ml/00_plan.md` is NOT to be edited (its
  [16]-[19] slotting is stale; leave it, flag only).
- **HW frames move into `04_trees.qmd`** (removed from decks; SLIDE_STYLE unchanged).
- **Scope: full chapter finish.**
- **Titanic story: proposed approach = REVIEW §2 options 1 + 2 combined** (confront the result
  in an L18 box AND retune the RF figure so its plateau is honest). Confirm on first read.

**Out of scope:** `L12b_classic_methods_survey_OUTLINE.md` (draft outline for a separate later
video; leave as-is). SHAP/permutation-importance internals (own chapters). The live bake-off.

---

## Phase 0 - Bug tier (DONE 2026-07-10, committed separately from this plan)

- [x] `-¿` arrows -> `$\to$` (L11 contrast box)
- [x] Recap mislabel "One tree (L10)" -> "(L09)" (L12) *(will be re-touched in Phase 1 sweep)*
- [x] "L06 logistic-regression" -> "chapter-3 logistic-regression" (L09)
- [x] GOSS/subsample "defaults" contradiction: row -> "Row sampling (option)" + corrected footnote (L12)
- [x] Provenance "chapter 3" -> "chapter 4" (L12)
- [x] Stale `ml/ch3_trees/` path + deck list (04_trees.qmd)
- [x] Stub `L12_xgboost_lgbm` confirmed already deleted

---

## Phase 1 - Renumber to [17]-[20] + reference sweep

**STATUS 2026-07-10:** Task 1.1 + 1.2 DONE (all four decks renamed, titles -> `[17]`-`[20]`,
cross-refs swept, all compile clean 0 errors: 17=34p, 18=27p, 19=31p, 20=38p). Only the HW frames
(17/18/19) and historical provenance comments still carry old L-codes - HW frames handled in
Phase 6, comments left as build history. **Task 1.3 descoped** (see below). Not yet committed.

**Why first:** 89 reference occurrences across the four decks; every later task edits these
files, so lock names/numbers now to avoid re-sweeping.

### Task 1.1: Rename the four decks (files + OUTLINEs)

**Files (git mv, preserves history):**
- `L09_decision_trees.{tex,pdf}` + `L09_decision_trees_OUTLINE.md` -> `17_decision_trees.*`
- `L10_random_forests.{tex,pdf}` + `_OUTLINE.md` -> `18_random_forests.*`
- `L11_boosting.{tex,pdf}` + `_OUTLINE.md` -> `19_boosting.*`
- `L12_advanced_boosting.{tex,pdf}` + `_OUTLINE.md` -> `20_advanced_boosting.*`

- [ ] **Step 1:** `git mv` each of the 12 files (4 decks x tex/pdf/outline).
- [ ] **Step 2:** Update `\title{}` in each deck to ch3 style: `\title{[17] Decision Trees}`,
  `[18] Random Forests`, `[19] Boosting`, `[20] Advanced Boosting`. Subtitles unchanged.
- [ ] **Step 3:** Recompile all four (twice each), log grep == 0, `clean_latex.py`.
- [ ] **Step 4:** Commit: `chore(ml/ch4): renumber trees decks L09-L12 -> [17]-[20]`.

### Task 1.2: Sweep in-slide cross-references

**Prefer descriptive wording over bare codes** (REVIEW §1.5) to reduce future breakage.

- [ ] **Step 1:** Fix broken callbacks to renamed ch2 decks:
  - `L01d` -> "the overfitting / cross-validation lecture" (was ch2 `06_overfitting_cross_validation`)
  - `L01c` -> "the data-preprocessing lecture" (was ch2 `03_data_preprocessing`)
  - Grep both across all four decks; replace each rendered occurrence.
- [ ] **Step 2:** Fix internal chapter cross-refs to the new scheme, descriptive where natural:
  - "L10's random forest" / "Contrast with L10" -> "the random-forests lecture" / "Contrast with random forests"
  - Recap bullets in [20]: "(L09)"->"(the single-tree lecture)", "(L10)"->"(random forests)",
    "(L11)"->"(boosting)", "(L12)"->"(this lecture)" OR the bare new numbers - pick one style, be consistent.
  - Bridge frames ("promised by L11's closing", etc.) -> descriptive.
- [ ] **Step 3:** HW frame internal refs ("HW09"/"HW10"/"HW11") - these move to the qmd in Phase 6;
  for now leave, Phase 6 relabels them HW1/HW2/HW3 within the chapter.
- [ ] **Step 4:** Grep `L0[19]|L1[0-2]|L01[cd]` across the four decks -> expect only legitimate
  hits (e.g. code like `LGBMRegressor`). Verify none are stale lecture codes.
- [ ] **Step 5:** Recompile all four, log grep == 0, overflow-check any reflowed frames, clean.
- [ ] **Step 6:** Commit: `fix(ml/ch4): sweep stale L01c/L01d + internal cross-references`.

### Task 1.3: De-duplicate the `\sectiontransition` macro - DESCOPED (deferred)

**Why deferred:** the macro is defined locally in FIVE decks across THREE chapters -
`ml/04_trees/{18,19,20}`, `ml/02_main_concepts/09_regression_metrics.tex`, and
`ml/04b_classic_methods/L12b_svm_and_classic_methods.tex`. Hoisting to `ml/preamble.tex` with
`\newcommand` clashes ("already defined") with every local def, so a safe hoist must atomically
remove all five locals + add one to preamble + recompile all five. That is a cross-chapter cleanup,
not a trees-chapter change - do it as its own focused task.

- [ ] **Interim (Phase 2):** copy the macro into deck [17] so its new section transitions work
  (keeps the trees chapter uniform with 18/19/20).
- [ ] **Deferred cleanup (own task, all 3 chapters):** hoist to preamble, remove all 5 locals,
  recompile [17]-[20] + `09_regression_metrics` + `L12b_svm_and_classic_methods`, verify each.

---

## Phase 2 - Low-risk content additions (no new figures)

**STATUS 2026-07-10:** DONE and committed. [17] +6 transitions +local macro, cardinality frame,
one-hot/CART-binary frame, missing-values line (Context7-verified). [18] OOB caveat. [20]
calibration ([14]) + imbalanced ([15]) callbacks. All overflow-checked, 0 errors.

Small, high-value, no figures. Each is one frame or a box/line. Verify overflow per new frame.

### Task 2.1: [17] Decision trees - conceptual gaps

- [ ] **Binary-vs-multiway split footnote/mini-frame** (REVIEW §3 L09, §4): CART is strictly
  binary; after one-hot the real tree asks `Outlook_Overcast <= 0.5`, NOT the 3-way hand-drawn
  split. One footnote on the "exactly the tree we drew by hand" frame OR a small frame
  "CART speaks binary: what one-hot does to Outlook".
- [ ] **Missing-values line** (REVIEW §4): one line on tree behaviour with NaN. **Verify current
  sklearn behaviour via Context7 before stating** (DecisionTree NaN support is version-dependent;
  HistGradientBoosting handles natively). State the verified version behaviour.
- [ ] **Add 5 section-transition slides** (`[plain]` popblue title + one motivation line) - [17]
  currently has bare `\section`s only; house style is transitions by default.
- [ ] **Cardinality-bias frame** (REVIEW §4, optional-but-high-value): the ID3 "Day column
  achieves IG = H(parent)" 1-frame by-hand stunner; motivates gain ratio + high-cardinality danger
  (ties to CatBoost in [20]).
- [ ] Verify (compile x2, log==0, overflow-check the new frames), clean, commit.

### Task 2.2: [18] Random forests - one honesty clause

- [ ] **OOB caveat** (REVIEW §3 L10): add clause to the OOB frame - "for a final report, still
  keep an untouched test set" (consistent with ch2 leakage messaging).
- [ ] Verify, clean, commit.

### Task 2.3: [20] Advanced boosting - cross-chapter callbacks

- [ ] **Calibration callback** (REVIEW §4): one sentence - boosted `predict_proba` is often
  miscalibrated; reliability-check it (cite the ch3 calibration lecture by name, NOT a bare code -
  it is video [14]).
- [ ] **Imbalanced callback** (REVIEW §4): in the "Beyond binary" box, cite the ch3 imbalanced
  lecture ([15]) by name instead of re-deriving advice.
- [ ] Verify, clean, commit.

---

## Phase 3 - Titanic score story (pedagogical, chapter-level)

**STATUS 2026-07-10:** DONE and committed (options 1+2). Outcome was more honest than the plan
assumed: on the same 70/30 split, pruned tree = **0.835**, untuned RF = **0.809**, CV-tuned RF =
**0.827** (max_features=0.8, min_samples_leaf=2). The retune did NOT surpass 0.835 - even tuned, the
forest stays just below. New `rf_vs_tree.pdf` (bar chart) + a "Reality check" frame in [18] own this:
the tree's 0.835 is test-picked while the RF's 0.809 is zero-tuning; RF wins with more data/features
and less tuning effort. No 83.5% chasing / cherry-picking. Overflow-checked, 0 errors.

**Approach (confirm): REVIEW §2 options 1 + 2.** The chapter's own numbers have a pruned single
tree (83.5%) beating the RF plateau (~81%); HW sends students to discover this, and no deck answers it.

### Task 3.1: Retune the RF figure so the plateau is honest (option 2)

**Files:** `py_src/make_rf_figures.py` (the `rf_n_estimators` figure), `fig/`.

- [ ] Add `max_features` / `min_samples_leaf` tuning to the RF in the figure script so the
  plateau lands at/above 83.5%; regenerate the PDF with the `ma` venv. Log the achieved number.
- [ ] Verify the figure renders and the new plateau value is shown on-axes.

### Task 3.2: Confront it in [18] (option 1)

**Files:** `18_random_forests.tex`.

- [ ] Add one frame OR an armorange box: on small tabular data with one dominant feature (sex), a
  well-pruned single tree is competitive; RF's edge shows with more rows/features, noisier signal,
  and *less tuning effort* ("RF is strong without the pruning search"). Connect the importance
  frame's fare/age-outrank-sex observation to the decorrelation knob (max_features hides sex +
  cardinality bias) - one sentence closes the loop (REVIEW §2 tail).
- [ ] Reframe HW (Phase 6) so the comparison is a feature, not a landmine.
- [ ] Verify (compile x2, overflow-check the new frame), clean, commit:
  `feat(ml/ch4): confront pruned-tree-vs-RF on Titanic + retune RF plateau`.

---

## Phase 4 - Figures (the visual gaps)

**Draft the exact figure list for instructor OK before generating** (per house workflow). Each
figure: real matplotlib into `fig/`, generated by a `py_src/` script, Armenian-flag palette when
3+ colours. Group commits per deck. Below are the specs from REVIEW §3-4.

### Task 4.1: [17] figures (`py_src/make_figures.py`)

- [ ] **Axis-aligned staircase** (3 panels): diagonal class boundary vs tree staircase at
  depth 3 / 6 / 10. The picture students remember.
- [ ] **Extrapolation mini-figure** (shared by [17]/[18]/[20]): rent-vs-area, linear fit vs tree
  fit, both extended 30% beyond training range - line trends, tree goes flat.
- [ ] **Greedy/XOR panel**: 2x2 checkerboard where no single split helps but two do.
- [ ] **Regenerate `titanic_tree.pdf`** readable on a projector: depth-2, `impurity=False,
  proportion=True`, larger fontsize.
- [ ] Wire each into its frame (replace the text-only versions); verify + overflow-check; commit.

### Task 4.2: [18] figures (`py_src/make_rf_figures.py`)

- [ ] **Test error vs number of trees for 3 `max_features`** (1.0 / sqrt / low) on one axes -
  shows the floor dropping with decorrelation (the section's whole argument; nothing currently
  shows max_features doing anything).
- [ ] **RF decision-boundary smoothing**: single tree's blocky boundary vs forest's softer
  boundary on the same 2D data (pairs with the [17] staircase).
- [ ] **Fix `rf_instability` legend** overlapping the tallest bar (cosmetic, on regen).
- [ ] Wire in; verify + overflow-check; commit.

### Task 4.3: [19] figures (`py_src/make_boost_figures.py`)

- [ ] **AdaBoost 3-panel** (keep the 1-frame budget): point size = weight, stump boundary per
  round - shows the reweighting dynamic (replaces the small static TikZ).
- [ ] **Classification GB 2-panel**: left F (log-odds) accumulating per round; right sigmoid(F)
  tightening on labels - the log-odds part is what students find abstract.
- [ ] **`boost_anim` round ~50 overlay**: one extra frame so diminishing returns are visible.
- [ ] Wire in; verify + overflow-check; commit.

### Task 4.4: [20] figures (`py_src/make_xgb_figures.py` + small TikZ)

- [ ] **Monotonic-constraint figure**: unconstrained (wiggly) vs constrained (clean) partial-effect
  curve on a noisy monotone feature. Reuse ch5 py_src PDP tooling. (Plan flagged constraints as
  required for credit/insurance/healthcare.)
- [ ] **Point `12_xgb_importance` at a real dataset** (bike or Titanic) so bars mean something,
  not synthetic f0..f11.
- [ ] **Schematics (small TikZ, ~15 min each):** sparsity default-arrow; EFB 3 columns merging to
  1; CatBoost ordered TS row-timeline; DART with 2/5 trees greyed.
- [ ] **Stacking receipt**: one real number line (e.g. bike: LGBM 0.87 / RF 0.87 / stack 0.88) OR
  an OOF 5-fold matrix diagram for the leakage trap - whichever the numbers support.
- [ ] Wire in; verify + overflow-check; commit.

---

## Phase 5 - [20] structural content (breaks the bullet-only stretch)

REVIEW §3 L12: sections 2-3 are 7 near-consecutive bullet-only frames; zero predict-first frames.

### Task 5.1: XGBoost-on-Yerevan worked frame (highest-value bridge)

- [ ] One frame connecting to [19]'s by-hand round: squared loss h=1, so
  `c* = sum(residuals)/(n_leaf + lambda)` = the leaf residual mean shrunk by lambda. Compute it for
  the round-1 stump (left leaf: `-17 -> -17*n/(n+lambda)`). Makes the structure score "[19]'s
  by-hand step + a built-in brake."

### Task 5.2: Add 1-2 predict-first frames

- [ ] Candidates: "leaf-wise vs level-wise, same #leaves - which overfits sooner?"; "GOSS keeps 30%
  of rows - how much accuracy lost?" (~none); "stacking meta-learner: strong or simple?" (simple).
- [ ] Verify + overflow-check; clean; commit: `feat(ml/ch4): [20] worked frame + predict-first`.

---

## Phase 6 - Chapter infrastructure (delivery blockers)

### Task 6.1: Move HW into `04_trees.qmd`; strip in-deck HW frames

**Files:** `04_trees.qmd`, `17/18/19` decks (remove HW frames), `20` (already none).

- [ ] Read the three in-deck HW frames (they are well-written; paste job); move content to the
  qmd `# 🏡 Տնային` section as HW1/HW2/HW3, reframing HW3 (Titanic) per Phase 3 so pruned-tree-
  vs-RF is a feature.
- [ ] Delete the HW frames from [17]/[18]/[19]; recompile the three, log==0, clean.
- [ ] Commit: `refactor(ml/ch4): move HW frames from decks into 04_trees.qmd`.

### Task 6.2: Build out `04_trees.qmd`

- [ ] Replace the TODO/TBD stub with the real page (follow the ch3 qmd + CONVENTIONS QMD template):
  callout (lectures recorded? - NO yet, keep an accurate "not delivered" note), `# 📚 Նյութը`
  bullets = the four deck clean PDFs (no notes/video links yet - not delivered), Google Form TBD,
  `# 🏡 Տնային` from Task 6.1.
- [ ] Commit: `feat(ml/ch4): build 04_trees.qmd page`.

### Task 6.3: Pin Titanic data + project notebook

**Files:** `ml/04_trees/data/`, a project notebook.

- [ ] Pin Titanic: add `data/titanic.csv` (or an explicit `fetch_openml` cell) so figure scripts
  and HW don't depend on a live OpenML fetch. Register in the qmd `resources:` if the page serves it.
- [ ] Build the chapter project notebook (revive the old L12 running-project idea): single tree vs
  RF vs hand-GBM vs tuned LightGBM on one dataset, one comparison table. Name per CONVENTIONS
  (proposed `trees_project.ipynb`; confirm number/prefix). Link from the qmd like ch3's bank project.
- [ ] Commit: `feat(ml/ch4): pin Titanic data + chapter project notebook`.

---

## Phase 7 - Final verification

- [ ] Full `beamer-overflow-check` pass on all four final PDFs (not just edited frames).
- [ ] Confirm all four compile clean from scratch; `clean_latex.py` repo-wide.
- [ ] Grep the four decks + qmd for any surviving `L09|L10|L11|L12|L01c|L01d|ch3_trees`.
- [ ] Update `PROGRESS.md` (via wrap-session) and note in `LEARNINGS.md` anything non-obvious.
- [ ] Flag to instructor: `ml/00_plan.md` still says trees [16]-[19] (left untouched per instruction);
  it now disagrees with the delivered [17]-[20].

---

## Self-review notes (gaps / confirmations needed)

- **Confirm Titanic approach** (Phase 3 = options 1+2) and the **figure list** (Phase 4) before
  generating - both per house "outline -> approve -> build".
- **Project notebook number/name** (Task 6.3) - undecided; `trees_project.ipynb` is a placeholder.
- **Execution mode:** inline (this session, `executing-plans`) is the default given the instructor's
  "ask before subagents" rule; subagent-driven would need explicit OK on agent/token cost.
- REVIEW §7 "do not touch" items (chapter arc, L10 variance spine, predict-first placement, by-hand
  numerics, residual animation, honest-defaults boxes, personal voice, provenance) are preserved -
  no task modifies them.
