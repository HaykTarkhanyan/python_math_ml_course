# Trees chapter (ml/04_trees) - review

Reviewed 2026-07-07 (Claude). Scope: L09_decision_trees, L10_random_forests, L11_boosting,
L12_advanced_boosting (+ the L12_xgboost_lgbm stub), the outlines, figures, and 04_trees.qmd.
Method: close read of the .tex sources + visual skim of all 4 compiled PDFs (rendered pages).
Lectures not yet delivered (qmd says so), so restructuring is still cheap.

**Verdict up front:** this is a strong chapter - the L09 -> L12 arc (tree -> bag -> boost ->
libraries) is coherent, the predict-first frames and by-hand numerics are exactly right, the
variance-formula spine of L10 is the best single pedagogical decision in the chapter, and the
L11 residual animation is a highlight. The weak points are: (1) a handful of real bugs and
internal contradictions, (2) L12 is text-heavy where every other deck is visual, (3) the
chapter's own numbers undermine its central claim (pruned tree 83.5% > RF ~81% on Titanic) and
no deck addresses it, (4) numbering/callback debt that will bite at delivery time, and (5) no
decision-boundary or extrapolation visuals anywhere - the two most instructive tree pictures.

---

## 1. Bugs and contradictions (fix before anything else)

1. **`-¿` typo on a rendered slide.** `L11_boosting.tex:230-231` uses text-mode `->` inside
   the armred "Contrast with L10" box; it renders as "-¿" on the slide (visible in the PDF,
   frame "Predict-first: does boosting overfit"). Use `$\to$` / `$\Rightarrow$`.

2. **L12 recap mislabels the whole chapter.** `L12_advanced_boosting.tex:634`: "One tree
   (L10)" should be L09, and "Bagging -> random forests (L10)" is then right. As rendered, the
   final chapter recap starts with a wrong pointer.

3. **L12's comparison table contradicts L11.** L11 (correctly, and in an armorange box) says
   subsampling is NOT on by default: XGBoost `subsample` and LightGBM `bagging_fraction`
   default to 1.0 (`L11_boosting.tex:353-356`). L12's "Which one, when?" table then lists
   "Row sampling: random / GOSS (by gradient) / random" under a footnote saying "the table
   shows defaults" (`L12_advanced_boosting.tex:384-403`). GOSS is opt-in in LightGBM
   (`data_sample_strategy="goss"`), not a default, and neither XGBoost nor CatBoost randomly
   row-samples by default. Reword the row to "Row sampling (option):" or split
   available-vs-default. Also verify the CatBoost cell: its default bootstrap is
   Bayesian/MVS depending on version and device - "random" is loose. (Chapter convention says
   verify library facts via Context7 at build time - do that here.)

4. **Stale cross-references students can't resolve.**
   - `L09_decision_trees.tex:837` HW bonus: "compare to your **L06** logistic-regression
     baseline" - logistic regression is L11 in ch3 (video [11]); L06 is overfitting/CV.
   - "callback L01c" / "callback L01d" appear on rendered slides in all three core decks
     (e.g. `L09:556`, `L09:624`, `L10:102`, `L10:189`, `L11:369`). Those file codes were
     renamed away when chapter 2 was renumbered (L01d = `06_overfitting_cross_validation`,
     L01c = `03_data_preprocessing`). A student watching video [16] has never heard of
     "L01d". Sweep and replace with the real lecture numbers/names once numbering is locked.

5. **Deck numbering will collide at delivery.** The plan (`ml/00_plan.md`) slots trees as
   videos [16]-[19], but the files and every in-slide reference say L09-L12 - which collide
   with ch2's 09 (regression metrics) and ch3's L11-L14 (logreg..imbalanced). There are
   currently two "L11" decks and two "L12" decks in the course. Same treatment as the ch2
   "renumber to match video numbers" commit will be needed, and it touches many in-slide
   references ("L10's random forest", "HW09", recap bullets, bridge frames). Suggestion:
   decide the video numbers now, renumber once, and prefer wording like "the random-forests
   lecture" over bare codes inside frames to reduce future breakage.

6. **Delete or park the stub.** `L12_xgboost_lgbm.tex` + its PDF are a 4-frame TODO skeleton
   superseded by `L12_advanced_boosting` (its own outline's scope decisions - Titanic
   bake-off, running-project HW - were explicitly dropped or moved during the advanced-deck
   interviews). Two L12 PDFs side by side is an accident waiting to happen. Move the pair to
   `ml/deferred/` or delete; keep the outline only if the bake-off idea might return.

7. **Small nits.** `L12_advanced_boosting.tex:653` provenance says "chapter 3 (trees)" - now
   chapter 4. `04_trees.qmd:15` points to `ml/ch3_trees/` - stale path. L09-L11 use `--`
   (en-dash) in prose while L12 deliberately uses plain hyphens per SLIDE_STYLE - pick one
   (style guide says hyphens).

---

## 2. The Titanic score story is upside down (pedagogical, chapter-level)

The chapter's central promise is "ensembles beat the single tree". Its own generated numbers
say otherwise:

- L09: unrestricted tree 65% CV, depth-1 stump 78%, **pruned tree 83.5%** (titanic_pruning).
- L10: RF test accuracy **plateaus at ~0.81** (rf_n_estimators, provenance confirms
  0.75 -> ~0.81).

So the best model in the chapter so far is L09's pruned single tree, and L10 never mentions
it - it only compares RF against the *unrestricted* tree. HW10 explicitly tells students to
compare against their L09 tree ("the chapter's running score"), so they will discover this
themselves and the deck has no answer ready.

Options, in order of preference:

1. **Confront it.** Add one frame or an armorange box in L10: on small tabular data with one
   dominant feature (sex), a well-pruned single tree is genuinely competitive; RF's edge shows
   up with more rows/features, noisier signals, and less tuning effort ("RF is strong
   *without* the pruning search"). This is honest and is itself a good lesson (callback: the
   "low-tuning default" strength). Then make HW10's comparison a feature, not a landmine.
- 2. **Also tune the RF** in the figure script (max_features, min_samples_leaf) so the plateau
   lands at/above 83.5%, and show that number. (1) and (2) combine well.
3. Alternatively run the chapter's running score on a second, larger dataset where the
   ordering is the textbook one (e.g. the ch5 bike-sharing data: single tree R^2 ~0.6x vs RF
   ~0.87 - those numbers already exist in ch5 and would make a tidy cross-chapter callback).

Related smaller issue: L10's importance frame says fare/age outrank sex "the opposite of
L09's single tree" - good observation, but *why* (max_features hides sex from many splits +
cardinality bias) is only half-said; one sentence connecting it to the decorrelation knob
would close the loop.

---

## 3. Per-deck review

### L09 decision trees - the strongest deck; mostly polish

Weak points / changes:

- **No section-transition slides.** L10/L11/L12 all have the `[plain]` popblue transitions;
  L09 has bare `\section`s only. House style says transitions by default - add them (5
  sections). Also: `\sectiontransition` is copy-pasted into L10, L11, and L12 - move the
  macro into `ml/preamble.tex` and reuse.
- **Binary vs multi-way splits is silently fudged.** The hand-built tree splits Outlook
  3 ways (ID3-style), then the deck says "CART makes it the root - exactly the tree we drew
  by hand" (`L09:391`). sklearn's CART is strictly binary and needs the categoricals encoded,
  so the real tree asks `Outlook_Overcast <= 0.5` etc. and is *not* the tree drawn by hand.
  Students who run HW step 1-2 will see a different-looking tree and wonder. One footnote or
  a small frame ("CART speaks binary: what happens to Outlook after one-hot") fixes it and
  reinforces the encoding lesson from the sklearn frame.
- **Missing values never mentioned.** The armred box handles the "no scaling / encode
  categoricals" folklore, but the equally common question "do I impute for trees?" is never
  answered in L09 (only L12's XGBoost sparsity frame). Add one line; verify the current
  sklearn behavior (recent sklearn DecisionTree accepts NaN for numeric splits;
  HistGradientBoosting handles natively) before stating it.
- **titanic_tree.pdf leaf text is near-unreadable** on a projector (bottom-row boxes). Regenerate
  with fewer displayed fields (`plot_tree(..., impurity=False, proportion=True)`), a depth-2
  tree, or larger fontsize - the frame's point is "readable", so the figure must be.
- **"Greedy, not optimal" is a text-only frame** and the concept is beggining for a picture: the
  classic XOR/checkerboard where no single split helps but two do. A 2x2 scatter TikZ or tiny
  matplotlib panel makes greedy-failure concrete in 10 seconds.
- The Gini-vs-economics box is a nice touch; keep.
- Impurity section ordering is good (need -> Gini -> entropy -> worked-by-hand), and the
  "criterion silently changes between slides" armred box preempts exactly the right confusion.

Illustrations to add (L09):

1. **Axis-aligned staircase figure** - a diagonal class boundary and the tree's staircase
   approximation at depth 3/6/10 (real matplotlib, 3 panels). Currently the weakness is only
   stated in words; this picture is the one students remember.
2. **Extrapolation mini-figure** - rent-vs-area with a linear fit and a tree fit, both
   extended 30% beyond the training range: line keeps trending, tree goes flat. Stated in
   words in L09, L10, and L12 - never shown. One small figure serves all three decks.
3. The greedy/XOR panel above.

### L10 random forests - conceptually excellent, visually mid-heavy

Weak points / changes:

- The middle stretch (frames "can averaging fix a biased model?" -> "how many features per
  split?") is 5 consecutive text/formula frames with a lot of white space. The content is
  right; add one real figure to anchor it - the natural one: **test error vs number of trees
  for three max_features values** (1.0 / sqrt / low) on one axes - it shows the floor moving
  down with decorrelation, which is the section's whole argument, and no current figure
  shows max_features doing anything (HW asks students to "watch it matter" but the deck
  never shows it).
- **RF decision-boundary smoothing figure missing** - single tree's blocky boundary vs the
  forest's averaged, softer boundary on the same 2D data. This is the canonical RF picture
  and pairs with the L09 staircase panel. (TikZ "averaging boxes gives boxes" line in
  weaknesses is correct and would sit nicely next to it.)
- **OOB overstatement (small).** "no separate train/test split needed" - fine for model
  *assessment inside training*, but if you tune on OOB you still need untouched test data.
  One clause ("for a final report, still keep a test set") keeps it honest and consistent
  with ch2's leakage messaging.
- The by-hand bootstrap frame's honesty ("predictions are given, not fit by hand") is good;
  keep the ~37% large-n caveat.
- rf_instability legend slightly overlaps the tallest bar - cosmetic, fix on regeneration.

### L11 boosting - the best-paced deck; two small holes

Weak points / changes:

- **AdaBoost's one frame undersells the weights idea.** The round-1 TikZ mini-picture is
  static and small. Within the same one-frame budget, a real 3-panel matplotlib figure
  (point size = weight, stump boundary per round) shows the reweighting *dynamic*. AdaBoost
  was deliberately scoped to one frame - keep that, just upgrade the visual.
- **Classification GB frame is text-only** and it's the one part students consistently find
  abstract (log-odds space). A two-panel figure - left: F (log-odds) accumulating per round;
  right: sigmoid(F) tightening on the labels - would earn its space. Even simpler: reuse the
  Titanic HW output as a "what you'll see" plot.
- The `-¿` bug (section 1) is on this deck's most important predict-first frame.
- The residual animation (boost_anim_1..6) is excellent - consider one extra overlay showing
  round ~50 so students see diminishing returns, not just rounds 1-6.
- "the method that ruled tabular ML for a decade" - arguably still rules; phrasing invites a
  "what replaced it?" question you may not want. Cheap fix: "that has ruled ... for a decade".

### L12 advanced boosting - right content, weakest visual/pedagogical execution

This deck breaks the chapter's own house pattern: L09-L11 average a figure or worked example
every ~2 frames; L12 sections 2-3 run seven nearly consecutive bullet-only frames (tree
growing/pruning, sparsity, subsampling/GPU, DART, EFB, CatBoost, and "beyond binary"). On the
rendered pages they are half-empty. Also: **zero predict-first frames** in the deck (L09 has
3, L10 has 4, L11 has 2).

Concrete fixes, highest value first:

1. **"XGBoost on the Yerevan toy" worked frame.** The G/H numbers boxes are good but
   abstract. One frame connecting to L11's by-hand round: for squared loss h=1, so
   `c* = sum(residuals)/(n_leaf + lambda)` - i.e. *the leaf residual mean, shrunk by lambda*.
   Compute it for the round-1 stump (left leaf: -17 -> -17*n/(n+lambda)). Suddenly the
   structure score is "L11's by-hand step + a built-in brake". This is the single best
   bridge the deck is missing.
2. **Predict-first candidates** (add 1-2): "leaf-wise vs level-wise: same number of leaves -
   which overfits sooner?"; "GOSS keeps 30% of rows - how much accuracy do you lose?"
   (answer: ~none, that's the point); "stacking meta-learner: strong model or simple one?"
   (answer: simple/regularized - counter-intuitive and true).
3. **Small schematics for the text-only frames** (TikZ, 15 min each): sparsity = a split
   node with a dashed "missing goes here" default arrow; EFB = 3 sparse columns visibly
   merging into 1; CatBoost ordered TS = a row timeline where row i's encoding uses only
   rows above it; DART = ensemble with 2 of 5 trees greyed out.
4. **Monotonic constraints deserve a figure** - unconstrained vs constrained partial-effect
   curve on a noisy monotone feature (two lines, one wiggly one clean). The plan explicitly
   flagged constraints as required for credit/insurance/healthcare; right now it's one
   text frame. The ch5 py_src PDP tooling can be reused nearly verbatim.
5. **Stacking has no evidence.** One real number line ("bike data: LGBM 0.87, RF 0.87,
   stack 0.88 - meta-learner buys the last point") - or, if it *doesn't* help, show that:
   "stacking usually wins but costs the most" is more credible with the receipt. An OOF
   diagram (5-fold matrix showing which fold produces which meta-feature) would also do more
   than the current prose for the leakage trap.
6. The 12_xgb_importance figure uses synthetic f0..f11 features - generic names teach
   nothing. Point it at a real dataset (bike or Titanic) so the bars mean something.

### Chapter infrastructure (blocking delivery, not the decks)

- **04_trees.qmd is a stub** (TODO warning, TBD everywhere, stale ch3_trees path). Per house
  rules HW lives here, not in slides - but L09/L10/L11 each carry an in-deck HW frame,
  violating SLIDE_STYLE ("No HW frame in the deck") while L12 (correctly) has none. Decide:
  either move HW09-HW11 content into the qmd (my recommendation - and the frames are already
  well-written, it's a paste job) or amend SLIDE_STYLE if you actually want in-deck HW for ml/.
- **No data/ folder and no project notebook.** The HWs lean on Titanic ("chapter project")
  and the Yerevan rent toy; figure scripts fetch Titanic from OpenML at run time. Students
  need a pinned CSV (or an explicit fetch_openml cell) and ideally a project notebook like
  ch3's bank_marketing_project.ipynb. The old L12 outline's "running-project HW" idea
  (single tree vs RF vs hand-GBM vs tuned LightGBM, one table) is genuinely good - consider
  reviving it as the chapter project notebook even though the in-deck bake-off was cut.
- **L12b classic-methods survey** exists only as a draft outline. The plan lists it as its
  own video [22]; fine - just note it is not part of this chapter's deliverable so it doesn't
  read as unfinished chapter work.

---

## 4. Missing topics - recommend adding

- **Binary-vs-multiway splits / what one-hot does to a tree** (L09; see above) - the only
  outright conceptual gap in the chapter.
- **Missing-value behavior of sklearn trees** (L09, one line; verify docs).
- **Split-selection cardinality bias** (distinct from *importance* bias, which is covered
  twice): a categorical/numeric feature with many distinct values gets more chances to win a
  split - the classic ID3 "Day column achieves IG = H(parent)" example is a 1-frame,
  by-hand-checkable stunner, and it motivates gain ratio + why high-cardinality categoricals
  are dangerous (ties into CatBoost later). Optional but high value per minute.
- **Calibration callback** (L12, one line): boosted-model scores are often miscalibrated;
  students just had a whole calibration lecture (ch3 L13) - one sentence connects the
  chapters ("before trusting predict_proba from a booster, reliability-check it - L13").
- **Imbalanced callback** (L12 "Beyond binary" box): cite the ch3 imbalanced deck by name
  instead of re-deriving the advice - it exists now.
- **Cost of ensembles at inference** is mentioned (L10 weaknesses) - fine as is.

## Missing topics - fine to skip (deliberate, agree with the cuts)

- Shapley/SHAP details (own chapter, exists), permutation importance mechanics (ch5 covers),
  AdaBoost math beyond one frame (HW bonus covers), oblique/rotation trees, isolation
  forests (different task), NGBoost/quantile detail (named in "any loss"), the live
  bake-off (cut for the right reason - noisy on small data), proximities (one-liner is enough).

---

## 5. What is working - do not touch

- The chapter arc and the L10 variance formula as the load-bearing spine (floor -> need
  decorrelation -> max_features) - textbook-grade sequencing.
- Predict-first placement in L09-L11 (esp. "more trees: RF safe vs boosting overfits"
  mirrored across L10/L11 - the strongest recurring contrast in the chapter).
- By-hand numerics that are real (Gini 0.459 -> gain 0.116; bootstrap OOB table; F0=304,
  eta=0.4 round; G/H numbers boxes) - all recomputed and correct.
- The L11 residual animation with the faint-previous-fit overlay, and the by-hand frame
  using the *same* numbers as round 1 of the animation - excellent design.
- The honest armorange boxes about sklearn defaults (RF regressor max_features=1.0;
  subsampling off by default) - keep this pattern; it's what separates these decks from
  every tutorial online.
- "Me and LightGBM" + instructor's-pick framing - personal voice, keep.
- Provenance blocks are consistently thorough - they made this review possible.

---

## 6. Suggested priority order

| # | Item | Effort | Deck |
|---|------|--------|------|
| 1 | Fix `-¿`, L10-recap mislabel, L06 ref, GOSS-defaults row | minutes | L11/L12/L09 |
| 2 | Decide video numbers; sweep L01c/L01d + L09-L12 refs | small | all |
| 3 | Titanic story: RF-vs-pruned-tree frame + retuned RF figure | medium | L10 |
| 4 | XGBoost-on-Yerevan worked frame | small | L12 |
| 5 | Decision-boundary staircase + RF smoothing + extrapolation figs | medium | L09/L10 |
| 6 | Binary-split/one-hot footnote + missing-values line | small | L09 |
| 7 | L12 schematics + 1-2 predict-first frames | medium | L12 |
| 8 | Move HW frames to qmd; build 04_trees.qmd; pin Titanic data | medium | chapter |
| 9 | Delete/park L12_xgboost_lgbm stub | minutes | chapter |
| 10 | Monotonic-constraint figure; stacking receipt; L09 transitions | medium | L12/L09 |
