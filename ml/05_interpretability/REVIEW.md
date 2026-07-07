# Interpretability chapter (ml/05_interpretability) - content & teaching review

Reviewed 2026-07-07 (Claude). Scope: the 3 decks (01_linear_and_trees, 02_pfi_and_effects,
03_shap_lime), their outlines, ADDITIONS_OUTLINE, and figures. Method: close read of the .tex
sources + visual pass over all 3 compiled PDFs (rendered pages). This review is about
**content and teaching only** - not file/nav/numbering infrastructure.

**Verdict up front:** the strongest-taught of the three chapters I've looked at. The 3-deck arc
is exactly right - read the model (glass box) -> explain any model (PFI / PDP / ICE) ->
attribute one prediction (SHAP / LIME / counterfactual). Two teaching decisions are genuinely
excellent: (1) the **correlation thread** that runs through all three decks - correlation
*deflates* impurity importance (deck 1), *inflates* PFI (deck 2), and *muddies* SHAP (deck 3),
each flagged as a deliberate "note the flip"; and (2) the **"Same data, four rankings"** payoff
that proves "importance is method-dependent" with real numbers on one dataset. Every data claim
is a real figure, third-party figures are attributed (LMU IML, CC BY 4.0), and predict-first
frames land on genuinely counter-intuitive moments.

The content weak points are concentrated in the two late-added sections (RuleFit,
counterfactuals) and a couple of figure/redundancy issues. Nothing here is a deep structural
problem - it's polish on an already-good chapter.

---

## 1. The two late additions are the weakest-taught part of the chapter

RuleFit (deck 1) and counterfactuals (deck 3) were folded in after the fact. Both are good
ideas, but as taught they stand out:

- **They are the only figure-less sections in a figure-rich chapter.** RuleFit is a LaTeX table
  + prose; counterfactuals is a LaTeX table + prose. Every *other* concept here earns a real
  figure. `interp_rulefit_figs.py` already fits an `imodels` RuleFit and logs the rules - plot
  them as a horizontal bar of rule weights (the "scorecard" made visual, reusing the PFI/SHAP
  bar style the students already know). For counterfactuals, a small before/after strip with the
  one changed cell highlighted teaches "smallest change that flips it" far faster than the
  current 8-row table.

- **RuleFit placement interrupts the spine toward the payoff.** It sits as its own
  `\section{Rule-based models}` between "Reading trees" and "So which features matter?" - i.e.
  between the two glass-box families and the importance-comparison punchline. It's defensible
  (rules = trees + Lasso, a synthesis of the two things just taught), but it delays the "do the
  two methods agree?" reveal. Decide: does RuleFit earn a full section here, or read better as a
  1-frame coda *after* the importance comparison? If it stays, the callback box tying it to "the
  two glass boxes from this deck" is the right framing - keep it.

- **Counterfactuals give the deck two endings.** The CF section (transition + 3 frames) lands
  right before the chapter closers ("Same data, four rankings" + "Which method when?"). CF is
  arguably a bigger idea than LIME (which gets 2 frames) yet comes *after* the "SHAP is the
  modern default" conclusion. The dataset switch to bank-marketing is flagged on-slide (good).
  Check the section order reads as one arc, not "SHAP wrap-up -> oh-also-counterfactuals ->
  chapter wrap-up." The "four rankings" table correctly omits CF (it isn't an importance method).

---

## 2. Per-deck content review

### Deck 1 - linear models & trees (glass box)

- **`tree_bike.pdf` is unreadable** on a projector - and the entire frame's teaching point is
  "a single tree is a glass box you can READ." The figure must actually be readable. Regenerate
  at depth 2, `impurity=False, proportion=True`, larger font. (Same defect as the trees-chapter
  Titanic tree - fix both the same way.)
- **"How a split is chosen: reduce impurity" re-teaches the trees lecture.** It re-derives
  information gain *and* variance reduction in full, with worked numbers - which L09 (decision
  trees) already taught in full (the provenance even says "ch4 trees ... assumed delivered"). If
  interpretability follows trees, this is redundant re-teaching; make it a one-line callback
  ("recall impurity reduction from the trees lecture") and shrink the two-column derivation to a
  single reminder. As built it's also slightly cramped - the regression column's three stacked
  equations crowd the `Var_parent` line.
- The **standardize-to-compare** misconception pre-empt (raw 5000-on-temp vs 2-on-hum is units,
  not importance) is exactly right - keep it.
- The **Lasso "keeps one of a correlated pair arbitrarily"** caveat is correct and seeds the
  correlation thread - keep.
- Hook forward-ref ("catch models that cheat - like deck 3's wolf detector") lands in deck 3.
  Good chapter-level setup.

### Deck 2 - PFI & feature effects (model-agnostic) - the cleanest deck, barely touch it

- The **PFI step-by-step animation** (perturb -> predict -> aggregate, 6 overlays) is excellent
  and earns its space.
- **Train-vs-test PFI**, the **correlation trap** (impossible temp/atemp rows), and
  **important != causal** (ice-cream/drownings DAG) are three distinct, well-chosen traps. The
  "note the flip" line (correlation deflated impurity, here inflates PFI) is the best single
  sentence in the chapter - it's what makes the three decks feel like one course.
- **ICE fanning out to reveal a hidden interaction** while the PDP is flat is a great
  predict-first, and it explicitly calls back to "the interaction a linear model couldn't
  represent (deck 1)." Strong connective tissue.
- **ALE is one frame + one figure** (a deliberate scope call - fine). But the whole argument
  lives in the figure ("PDP overshoots at extremes; ALE stays realistic"), and at slide scale
  the two curves barely separate. Make the divergence at the extremes visually obvious, or
  annotate the gap - otherwise the frame asserts a fix the student can't see.

### Deck 3 - SHAP & LIME (+ counterfactuals) - excellent SHAP core

- The **base value + sum of SHAP values** identity, then the **four plots** (waterfall / bar /
  beeswarm / dependence) each with a "read it" gloss, is the right way to teach SHAP
  intuitively without the Shapley axioms (correctly cut). The bar-≈-PFI and dependence-≈-PDP
  unification is the payoff that ties deck 3 back to deck 2 - keep it prominent.
- **TreeSHAP is the default; KernelSHAP is slow** - correct and the right depth.
- **husky/wolf -> snow** (clever-Hans) and **LIME instability** (two runs, two fits) are the
  two right things to show for LIME in its 2-frame budget.
- The **consistency guarantee** one-liner ("use a feature more => its SHAP can't fall") justifies
  "SHAP is the modern default" without hand-waving.
- Counterfactuals: see section 1. The **5 desiderata** (validity / proximity / sparsity /
  plausibility / actionability) are the right list, and the **actionability/recourse** angle
  (change a balance not an age; never a protected attribute) is the one place the chapter touches
  fairness - keep it, and consider promoting it (see missing topics).

---

## 3. Missing topics - recommend adding

- **Global surrogate models** (fit an interpretable model to *mimic* a black box). The chapter
  teaches the *local* surrogate (LIME) but never its global sibling - one frame completes the
  pair and it's a common, intuitive technique ("distill the forest into a readable tree/GAM").
- **Feature interaction as a named, measurable quantity** (H-statistic). The chapter *shows*
  interactions three times (ICE fans out; SHAP dependence colour; the flat-PDP toy) but never
  names "interaction strength" as something you can compute. One line + a pointer closes it.
- **Calibration callback** (deck 3, one line): SHAP explains the model's *score*; if the model
  isn't calibrated, the explanation faithfully explains a miscalibrated number. One sentence ties
  it to the calibration lecture.
- **Fairness beyond the CF bullet** (optional, one line): the course uses bank-marketing /
  credit-like decisions; "an explanation can explain-wash a biased model - interpretability is
  not fairness" would close an ethics loop the actionability bullet only half-opens.
- **A GAM / EBM name-drop** (optional): EBM ("glass-box boosting") is the natural modern bridge
  between "glass box" and "boosting" - one line in the recap.

## Missing topics - fine to skip (agree with the cuts)

- Shapley axioms / ordering math, KernelSHAP weighting, SAGE (intuition-first is the right call
  for this audience).
- Anchors (rule-based local explanations) - LIME + counterfactuals already cover the local space.
- SHAP interaction values - too deep for a first pass.
- CFI / conditional-PFI mechanics - correctly demoted to a one-line "one fix" mention.

---

## 4. What is working - do not touch

- The **three-deck arc** (read it -> explain it -> attribute it) and the **single bike-sharing
  anchor** threaded through all three, switching to bank-marketing *only* where a decision is
  needed (counterfactuals). Textbook structure.
- The **correlation thread** with explicit "note the flip" callbacks across decks - the
  strongest connective tissue in any of the three chapters I've reviewed.
- **"Same data, four rankings"** and **"Which method when?"** as the twin closers - concrete,
  honest, and exactly the takeaway ("say *which* importance, on *what* data").
- **Real figures everywhere**, hybrid with attributed LMU IML reuse. The PFI animation and the
  ice-cream/drownings DAG especially.
- **Predict-first placement** on genuinely counter-intuitive moments (which can you read; do the
  two methods agree; PDP flat but ICE fans; what did LIME reveal) - not sprinkled everywhere.

---

## 5. Suggested priority order (content only)

| # | Item | Effort | Where |
|---|------|--------|-------|
| 1 | Regenerate `tree_bike.pdf` readable (depth-2, big font) | small | deck 1 |
| 2 | Add a RuleFit scorecard figure; add a CF before/after visual | small | deck 1/3 |
| 3 | De-dupe the "how a split is chosen" frame vs the trees lecture (callback + shrink) | small | deck 1 |
| 4 | Make the ALE-vs-PDP divergence visible in the figure | small | deck 2 |
| 5 | Reconsider RuleFit placement / CF's double-ending for flow | small | deck 1/3 |
| 6 | Add global-surrogate frame; interaction-strength + calibration one-liners | small | deck 2/3 |

(Minor, content-adjacent: the deck-1 and deck-3 per-deck outlines list RuleFit and
counterfactuals as "out of scope" - both were later added, so those outlines are now stale;
reconcile them so they don't mislead later. And verify the in-frame cross-references point to
the lectures you actually mean.)
