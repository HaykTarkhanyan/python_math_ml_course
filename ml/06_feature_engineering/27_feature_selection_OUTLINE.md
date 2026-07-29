# [27] Feature Selection — Outline (v3)

Design doc for the **second deck of `ml/06_feature_engineering/`**. House style + `ml/SLIDE_STYLE.md`.
Replaces `L01h_feature_selection.tex` (written 2026-06-05, before the interpretability chapter existed).

> **v3** = self-review fixes applied (see *Changelog*). **v2** = instructor interview, 4 rounds.
> **Status: draft for approval.** Built **after** [26], because it consumes [26]'s candidate pool.

---

## Scope / thesis

**A ranking is not a decision.** Every tool from chapter 5 — PFI, CFI, LOCO, SAGE, SHAP, Lasso —
hands you an ordered list. None tells you **where to cut**. That gap is the deck, and it is why
wrapper methods exist at all.

This reframing is why the deck is being rewritten rather than polished. `L01h` spends 5 of its 20
frames re-teaching rankings students now know better than the old deck teaches them, and treats the
cutoff as a footnote. Inverting that is the whole change.

This deck is the second half of [26]'s thesis, *generate cheaply, select ruthlessly.*
**Regression-only**, matching [26] (instructor decision).

## Locked decisions (from the interview)

| Decision | Choice |
|---|---|
| Dataset | **Bike**, same 70/30 split and seed 509 as [22]-[24]. Baseline test MAE **455.1** |
| Chaining | **Chained** — opens on the ~200 candidates [26] generated, not the raw 11 |
| Depth | **Same as ch5** — Boruta and RFE get their algorithms, not just their names |
| Ch5 cut | **One recap frame, then the bridge.** Not absolute; students may be rusty |
| Multiple testing / FDR | **Keep as a real frame** (new scope) |
| "When not to select" | **Show both regimes** — measured on 11 raw features AND on 200 candidates |
| Null results | **Go on the slide** as measured |
| Leakage | **Distributed**, not a section — selection bias stays attached to its own frame |
| Classification FE | **Out of scope**, matching [26] |
| Length | **~35 frames is fine**, do not pre-emptively trim |
| Homework | **Out of scope for now** |

## Changelog — v2 → v3 (self-review)

| # | Issue | Fix |
|---|---|---|
| 1 | **I wrote predictions as if they were results** on frames 25 and 27 (Boruta confirms both `temp` and `atemp`; stability shows ~50/~50 with union ~100%). Both plausible, both possibly wrong in a *more interesting* way — if the pair splits credit as [22] measured, each could fall **below** max shadow importance and Boruta might reject **both** | Frames 26 and 28 now state **what will be measured**, with the competing outcomes and what each would mean. Same failure as the R² contradiction caught last session: measure, do not pick a side |
| 2 | Frames 19 and 30 were **nearly the same experiment** (noise columns → select → observe the lie) with the distinction left implicit | Frames 20 and 31 now name the distinction explicitly and cross-reference each other. 20 = *how many pass a threshold* (multiplicity). 31 = *how inflated the resulting score is* (selection bias) |
| 3 | Decision flowchart was **misfiled** under "Pitfalls" — it is a synthesis frame | Moved to Wrap-up (frame 33) |
| 4 | Section 2 had **no `[plain]` transition** while every other section did | Added (frame 9) |
| 5 | Frame 3 said [26] "ended by" generating candidates — [26] has six frames after that point | Corrected to "[26] section 5" |
| 6 | Header said ~30 frames; the list ran to 33 | Corrected — 35 frames (34 numbered + Outline) |
| 7 | Regression-only scope not stated | Stated |

## What is cut from `L01h` and why

| Old frame | Already taught, better | Disposition |
|---|---|---|
| L1 / Lasso for sparsity | [07]: 7 frames (path, geometry, soft thresholding, priors). [22]: "Lasso selects its own features" | → one line in the recap, one in the bridge |
| Tree impurity importance | [22]: "How a split is chosen", "Forests and boosting score features", "your library picks the definition" | → one line in the recap |
| Predict-first: impurity on correlated features | [22]: "Correlated features split the credit", with measured bootstrap numbers | **cut** — the [22] version is strictly better |
| Permutation importance | [23]: 8 frames — idea, step-by-step, on bike, train vs test, correlation trap, CFI, LOCO, SAGE | → one line in the recap, re-read in the bridge |
| SHAP | [24]: ~14 frames | → one line in the recap |
| Decision flowchart | 3 of its 5 outcomes point at [22]-[24] material | **rebuilt** around cutoff strategy, not ranking method |
| Pitfall #4 (correlated features) | compressed restatement of [22]/[23] | **cut**, folded into the bridge |
| HW01h frame | tasks 2-3 duplicate Parts 3-4 of the [25] project | **removed** |

Net: 20 frames → ~35, of which roughly 24 are new or substantially rebuilt.

## Callbacks

- **[23]** — `fig/pfi_cfi_loco_bike.pdf` **reused verbatim** in the cold open. Same picture, not a
  redraw, so the connection is unmissable.
- **[22]** — Lasso as built-in selection; correlated features split the credit (resurfaces in frame 28).
- **[07]** — Lasso, the L1 corner, solution paths. **[06]** — cross-validation; frame 31 is that
  lesson one level up.
- **[26]** — the ~200 candidates this deck prunes, and frame 20 of [26] (custom transformers), which
  is what makes "selection inside the CV loop" implementable at all.
- **math 30** — curse of dimensionality. **Stats module** — multiple comparisons, for frame 20.

## Citations — ALL need web-search verification

`L01h` cites exactly one paper. Candidates, all to verify:

- Guyon & Elisseeff, 2003 — the canonical survey, for the three-families frame.
- Guyon, Weston, Barnhill & Vapnik, 2002 — SVM-RFE.
- Kursa & Rudnicki, 2010 — Boruta.
- Peng, Long & Ding, 2005 — mRMR.
- Yamada et al., 2014 — HSIC-Lasso.
- Meinshausen & Bühlmann, 2010 — stability selection *(already correct in `L01h`)*.
- Nilsson et al., 2007 — minimal-optimal vs all-relevant, for frame 12.
- Benjamini & Hochberg, 1995 — false discovery rate, for frame 20.

Verify `Boruta` / `boruta_py` is alive on PyPI before sending students there.

## Acronym discipline

`L01h` fails the `SLIDE_STYLE.md` mechanical check on `GBM` and `MI`. This deck adds `RFE`, `RFECV`,
`SFS`, `mRMR`, `HSIC`, `FDR`, `FWER`. Plain phrase first, acronym in parentheses, never
letter-highlighting inside words. Before calling it done:

```bash
sed 's/%.*//' 27_feature_selection.tex | grep -oE '\b[A-Z]{2,6}\b' | sort -u
```

---

## Measured results (2026-07-29, `ml/06_feature_engineering/py_src/fe_measure.py`)

Every `[MEASURE]` frame now has a real number, and **two of my predictions were wrong** — see
frames 17 and 28.

| # | Measurement | Result |
|---|---|---|
| M7 | **Regime 1**, 11 raw features | keep-all **655.2**, RFE-CV kept **all 11**, selected **655.2**, delta **exactly 0.0** |
| M8 | **Regime 2**, 230 candidates | keep-all **573.7**, selected (k=**45**) **505.5**, delta **+68.2** |
| M9 | RFE-CV survivors | 45 of 230 — raw 4, difference 14, ratio 14, product 11, **domain 2** (`feels_gap_C`, `temp_C`; `temp_bin` dropped) |
| M10 | Boruta on the pair | **both `temp` and `atemp` CONFIRMED.** Also **rejects** `holiday`, `weekday`, `workingday` |
| M11 | Stability selection | `temp` **68%**, `atemp` **99%**, **union 100%**. `mnth` only **9%** |
| M12 | Filter disagreement | **`mnth`, not `temp`** — F rank 6 vs MI rank 3. `temp` is rank 2 on both |
| M13 | Multiple testing | of 10,000 noise columns: **528** pass at α=0.05, **104** at 0.01, **0** survive Bonferroni or Benjamini-Hochberg. Best noise p = **1.17e-04** |

**Regime 1 could not have gone better for frame 7:** RFE-CV kept every one of the 11 features and
the delta is *exactly* zero. At p=11 selection buys literally nothing.

**A method disagreement worth a frame (M10 vs M11):** Boruta **rejects** `weekday` and `workingday`
as no better than shuffled noise, while stability selection keeps `weekday` in **100%** of
bootstraps and `workingday` in 80%. Both ran on the same 11 features. They disagree because they ask
different questions — Boruta asks "does this beat a random column in a forest?", Lasso asks "does
this earn a non-zero coefficient in a *linear* model?" This belongs on frame 29 (four ways to pick
`k`) or as its own frame; it is the deck's thesis in miniature.

---

## Frame-by-frame (35 frames: 34 numbered + Outline)

### Cold open (before the Outline)

1. **"Deck [23] told you to drop `atemp`. Should you?"**
   The measured numbers back on screen: PFI ranks `atemp` **second** (413, behind `yr` at 766), but
   **LOCO(`atemp`) = −0.7 MAE** — remove it and the model gets slightly *better*. Predict-first:
   *do we drop it?* Earns a `\pause`; the tools genuinely disagree.
   Reuses `ml/05_interpretability/fig/pfi_cfi_loco_bike.pdf` unchanged.

2. **Neither number answers the question.**
   PFI says "the model leans on it". LOCO says "the model copes without it". Both true — `temp`
   carries the same signal ([22]). **A ranking tells you the order. It does not tell you where to
   cut.** That sentence is the deck.

3. **And now you have 200 of them.**
   [26] section 5 mechanically generated ~200 candidates. Put the count up. Ranking 200 features is
   easy; deciding which 15 ship is not. This is the deck's working problem, and the second half of
   [26]'s thesis: *generate cheaply, select ruthlessly.*

### Outline

### Section 1 — Why select, and when not to

4. **`[plain]` transition — "Fewer features, on purpose."**

5. **Five reasons to drop features.**
   Curse of dimensionality (math 30) / multicollinearity ([22]) / variance / compute and latency /
   interpretability and audit. Keep `L01h`'s frame, tighten the prose.

6. **Predict-first: does adding features always help?** **[MEASURE]**
   Training error falls monotonically; test error is U-shaped. **Generate the curve from a real
   run** — add [26]'s candidates one at a time in PFI order, record both errors.
   Figure: `fig/fs_add_features_ucurve.pdf`

7. **Regime 1: eleven honest features.** **[MEASURED — a clean null]**
   Bike's raw 11 + Ridge. RFE-CV **kept all 11**. Keep-all 655.2, selected 655.2, **delta exactly
   0.0**. At p=11 selection buys *literally nothing* — not "a little", zero. Put the number up
   unedited; it is the strongest possible version of this frame and it was not guaranteed.

8. **Regime 2: two hundred and thirty candidates.** **[MEASURED]**
   Same experiment on [26]'s pool: keep-all **573.7**, RFE-CV selected **k=45** → **505.5**,
   **delta +68.2**.
   **The lesson across both frames: selection pays in proportion to how much junk you generated.**
   Makes frame 32's "always compare to keep-all" a rule the deck actually follows.
   **Do not stop there — close the honest loop back to [26] frame 7:** a forest on the *raw 11* gets
   **455.1**. Engineering 219 features and selecting the best 45 got a linear model to 505.5 and
   still lost to the model that was handed the raw columns.
   Figure: `fig/fs_both_regimes.pdf`

### Section 2 — You already own half the toolkit

*Nothing new here; it re-files what students know under a new question.*

9. **`[plain]` transition — "You have done this before without calling it selection."**

10. **Recap: what each chapter-5 tool actually measures.**
    One dense frame, one line each — PFI, CFI, LOCO, SAGE, SHAP, Lasso — with a deck pointer per row.
    Memory jog only. *(Per the interview: not an absolute cut, so a rusty student can follow the
    bridge.)* **Overflow risk — six rows plus pointers. Check the render; split if tight.**

11. **Re-read as selection algorithms.**
    - **LOCO** = drop one, refit, measure. That is **one step of backward elimination.** They have
      already run it.
    - **Lasso path** ([07], [22]) = embedded selection; the coefficient hitting zero *is* the drop.
    - **Impurity / permutation importance** = the ranking wrapper methods consume.
    - **Boruta's shadows** (frame 25) = permutation importance with a null distribution bolted on.

12. **Minimal-optimal vs all-relevant.**
    Two genuinely different goals that students conflate:
    - **Minimal-optimal** — smallest subset predicting as well as the full set. Lasso, RFE, stability
      selection. Correlated duplicates get dropped arbitrarily.
    - **All-relevant** — *every* feature carrying signal, duplicates included. Boruta. The right goal
      when you are doing science, not deployment.
    **Bike makes it concrete:** minimal-optimal drops `atemp`; all-relevant keeps it. *Same data,
    both answers correct, different question* — which resolves frame 1. Cite Nilsson et al., 2007.

13. **So what is left to learn?**
    Three things a ranking cannot give you: **(a)** a cutoff, **(b)** a way to score a *subset*
    rather than a feature, **(c)** a guarantee the answer is stable. Sections 3-5, in order.

### Section 3 — Filter methods

14. **`[plain]` transition — "Score every feature once, before you fit anything."**

15. **The three families.**
    Filter / embedded / wrapper, cost-vs-accuracy ordering, and the 10,000 → 500 → 50 → 15 funnel.
    Keep `L01h`'s table. Cite Guyon & Elisseeff.

16. **Variance threshold and univariate filters.**
    Near-constants first. Then `f_regression` (linear) vs `mutual_info_regression` (any dependence);
    `f_classif` / `chi2` / `mutual_info_classif` named for classification, not developed.

17. **Where the two filters disagree.** **[MEASURED — my prediction was wrong, and the real answer
    is better]**
    I expected `temp` to be the disagreement. It is not: `temp` ranks **2nd on both** (F=473.5,
    MI=0.388). **The disagreement is `mnth`** — F rank **6** (F=62.0) but mutual-information rank
    **3** (MI=0.376), a three-place gap and the only one in the table.
    Why it is the better example: month is **cyclic and non-monotonic**, so a linear F-test sees
    almost nothing while mutual information sees a lot. That is exactly the case [03]'s cyclic sin/cos
    encoding exists for, so the frame calls back to preprocessing instead of re-treading [23].
    **Bonus for frame 28:** `mnth` is also the *least stable* feature under bootstrapping (9%), so
    the same column carries this frame and the stability frame.
    Figure: `fig/fs_filter_scores.pdf` (paired ranks, `mnth` highlighted)

18. **The independence trap.**
    `L01h`'s XOR frame, kept nearly intact — the best frame in the old deck. Four rows,
    `corr(x₁,y) = corr(x₂,y) = 0`, mutual information ≈ 0 per feature, jointly a perfect predictor.
    **Filter is a screen, not a decision.**
    Figure: `fig/fs_xor_trap.pdf` (generated: both marginals plus the joint)

19. **Multivariate filters.**
    mRMR (relevance minus redundancy), cluster-and-represent, HSIC-Lasso. Expand every acronym.
    Honest framing: rarely the main event, genuinely useful at large `p` with heavy correlation.

20. **Multiple testing: how many pass by chance?** *(new scope, kept per interview)*
    **This frame is about the *count* of survivors, not about score inflation — frame 31 covers that.
    Say so on the slide and cross-reference it, because the two demos look alike.**
    Score 10,000 features at α = 0.05 and ~500 clear the bar **by chance alone**. Family-wise error
    rate vs false discovery rate; Bonferroni vs Benjamini-Hochberg; `SelectFwe` / `SelectFdr`.
    **Directly relevant to [26]'s pool:** 200 mechanically generated candidates scored against one
    target *is* this problem, at smaller scale.
    **[MEASURED]** 10,000 pure-noise columns against bike `cnt`: **528** pass at α=0.05 (expected
    ~500), **104** at α=0.01, and the best noise column reaches **p = 1.17e-04** — a result most
    students would call significant without blinking. **Bonferroni: 0 survive. Benjamini-Hochberg:
    0 survive.** Correction works; the point is that nobody applies it to a feature-engineering pool.
    Figure: `fig/fs_multiple_testing.pdf`

### Section 4 — Wrapper methods

21. **`[plain]` transition — "Stop scoring features. Start scoring subsets."**

22. **Recursive Feature Elimination: the mechanism.**
    Fit, rank, drop the weakest, refit, repeat. Why refitting matters — importances shift once a
    correlated partner leaves, which is [22]'s split-the-credit result in motion. Cost: O(p) refits
    per fold. Cite Guyon et al., 2002.

23. **RFE-CV: where the cutoff finally comes from.** **[MEASURED]**
    Cross-validate at each subset size, take the best. **This is the frame that answers frame 2.**
    On [26]'s 230 candidates it chose **k = 45**, taking test MAE from 573.7 to **505.5**.
    **Survivors by origin:** difference 14, ratio 14, product 11, raw 4, **domain 2**.
    The two hand-designed survivors are `feels_gap_C` and `temp_C`; `temp_bin` was dropped.
    **The nuanced read, and it is worth saying out loud:** both surviving domain features are
    temperature-in-Celsius reconstructions — features mechanical expansion **could not have
    produced**, because they need the normalization constants from the dataset documentation. So
    domain knowledge earned its place (2 of 3 offered survived) but is numerically swamped
    (2 of 45). Neither "domain knowledge wins" nor "just generate everything" is the honest summary.
    Figure: `fig/fs_rfecv_curve.pdf` *(replaces `L01h`'s invented elbow at k=9)*

24. **Sequential forward and backward selection.**
    Forward when `k ≪ p`, backward when `k ≈ p`. Both greedy; both miss combinations that only pay
    off jointly — **explicit callback to the XOR frame**, where forward selection provably fails
    because neither feature helps alone.

25. **Boruta: the algorithm.**
    Shadow features = shuffled copies. Fit a forest on real + shadows, compare each real feature to
    the **max** shadow importance, repeat, and use a binomial test over iterations to confirm or
    reject. At ch5 depth this gets its mechanics. Cite Kursa & Rudnicki, 2010.

26. **Boruta on bike: what happens to the correlated pair?** **[MEASURED]**
    **Both `temp` and `atemp` CONFIRMED** — the all-relevant goal behaves exactly as frame 12
    advertises, and the contrast with RFE-CV (which kept only one of the pair) is clean. Frame 12's
    claim is now demonstrated rather than asserted, on the same 11 features.
    **The unplanned finding, which is the better half of the frame:** Boruta **rejects** `holiday`,
    `weekday` and `workingday` outright — no better than shuffled columns in a forest. But stability
    selection (frame 28) keeps `weekday` in **100%** of bootstraps and `workingday` in **80%**.
    Same data, same 11 features, flatly opposite verdicts. They disagree because they ask different
    questions: Boruta asks *"does this beat a random column inside a forest?"*, Lasso asks *"does
    this earn a non-zero coefficient in a linear model?"* — a weekday effect that a tree absorbs
    through `season` and `temp` can still be worth a coefficient of its own.
    **This is frame 2's thesis in miniature: the method you pick is a question you asked.**
    Figure: `fig/fs_boruta_shadow.pdf` *(replaces `L01h`'s invented bars)*

### Section 5 — How many, and is the answer stable?

27. **`[plain]` transition — "Run it twice. Same features?"**

28. **Stability selection** (Meinshausen & Bühlmann, 2010). **[MEASURED — my prediction was wrong]**
    Bootstrap 100 times, fit Lasso on each, keep features selected in > 50% of runs. Answers *how
    many* automatically.
    **The pair does not split 50/50.** Measured: `atemp` **99%**, `temp` **68%**, **union 100%**.
    The union result is exactly the split-the-credit prediction — one of the two is always in — but
    the split is **asymmetric**, not a coin flip. `atemp` wins the tie most of the time because it is
    the marginally better predictor (F 482.5 vs 473.5, MI 0.464 vs 0.388), so Lasso reaches for it
    first and `temp` only survives when the bootstrap happens to favour it.
    **That is a more useful lesson than the symmetric story I expected:** correlated features do not
    split credit *evenly*, they split it in proportion to a small edge, and the weaker one looks
    dispensable when it is not.
    **The real instability is `mnth` at 9%** — in nine bootstraps out of a hundred. Pair with frame
    17, where `mnth` is also the F-vs-MI disagreement: a feature that a linear model can barely use
    is exactly the one that flickers in and out of a linear model's selection.
    Figure: `fig/fs_stability.pdf`

29. **Four ways to pick `k`.**
    RFE-CV elbow / stability threshold / Boruta's shadow cutoff / a hard business constraint ("the
    credit model ships with 12 features because the regulator reads all of them"). The last is real
    and students never consider it.

### Section 6 — Pitfalls

30. **`[plain]` transition — "The mistake that makes everything look great."**

31. **Selection bias: a frame, not a bullet.** **[MEASURE]**
    **This frame is about *score inflation*, not about how many features pass a threshold — frame 20
    covers that. Cross-reference it explicitly; the two demos use the same trick for different ends.**
    Select on the full dataset, then cross-validate → inflated, because selection already saw the
    test rows. Select inside the CV loop → honest. **`L01h` gives this two lines in a list of six.**
    It is the most common way a student's model lies to them, and it is exactly [06]'s lesson one
    level up: *the whole procedure* goes inside the loop, not just the fit — which is implementable
    only because of [26] frame 20.
    Worst-case demo: top-`k` of pure-noise columns selected on full data, then CV. The score looks
    good on data with no signal at all.
    Figure: `fig/fs_selection_bias.pdf`

32. **Four more pitfalls.**
    Leakage via un-CV'd target encoding ([26] frames 18-19) / model-dependence (Lasso's picks ≠ the
    tree's; re-select if you switch models) / arbitrary thresholds ("importance > 0.01" is a number
    somebody made up) / **forgetting the baseline** — which frames 7-8 already demonstrated.

### Wrap-up

33. **Decision flowchart, rebuilt.** *(moved here from Pitfalls — it is a synthesis frame)*
    `L01h` branches on *which ranking tool*; three of its five outcomes are now chapter-5 material.
    Rebuild around this deck's questions: what is `p`? minimal-optimal or all-relevant? can you
    afford O(p) refits? do you need a stability guarantee? is there a hard cap on `k`?

34. **Recap** + close the loop on frame 1 (**so, do we drop `atemp`?** — it depends which question
    you are asking, per frame 12) + `paramgreen` **"Next: [28] classic methods."**

---

## Figures (`py_src/fs_figs.py` → `fig/`)

| File | Frame | Measured? |
|---|---|---|
| *(reuse `05_interpretability/fig/pfi_cfi_loco_bike.pdf`)* | 1 | already measured — **do not redraw** |
| `fs_add_features_ucurve.pdf` | 6 | **yes** |
| `fs_both_regimes.pdf` | 7-8 | **yes** |
| `fs_filter_scores.pdf` | 17 | **yes** |
| `fs_xor_trap.pdf` | 18 | generated |
| `fs_multiple_testing.pdf` | 20 | **yes** |
| `fs_rfecv_curve.pdf` | 23 | **yes — replaces invented elbow** |
| `fs_boruta_shadow.pdf` | 26 | **yes — replaces invented bars** |
| `fs_stability.pdf` | 28 | **yes** |
| `fs_selection_bias.pdf` | 31 | **yes — the deck's most important figure** |

All: `ma` venv, seed 509, same 70/30 split as [22]-[24].

**Hard dependency:** `py_src/fs_figs.py` **loads [26]'s candidate matrix from disk**, it does not
regenerate it. If [26]'s script does not persist the pool, the chaining is decorative and frames 3,
8, 20 and 23 lose their premise.

**Compute:** Boruta (100 iterations × 200 trees), stability selection (100 bootstrap LassoCV fits)
and RFE-CV (O(p) refits × 5 folds) on **200 features** are meaningfully heavier than on 11. Still
small in absolute terms (731 rows), but run **sequentially, one script, no parallel agents**, and
cache intermediates so a re-run does not redo Boruta.

## Deliverables beyond the deck

- `git rm` `L01h_feature_selection.{tex,pdf}` once this compiles.
- The chapter qmd links both decks (see [26]'s outline — it does not exist yet).

## Open questions

1. **Frame 23's payoff is unknown and it tests [26]'s thesis.** If RFE-CV's survivors from the 200
   are mostly mechanically-generated features rather than the hand-designed domain ones, that
   complicates [26] frame 7's "domain knowledge wins" bullet. Interesting either way — I will
   measure it early and report before either deck is written.
2. **Frame 20's noise-column count.** I picked 10,000 because it makes the arithmetic clean (500
   false positives at α = 0.05), but 10,000 columns × 731 rows is a p ≫ n regime disconnected from
   the deck's actual 200-feature problem. Say if you would rather it match the real pool size for
   continuity, at the cost of a less dramatic number.
