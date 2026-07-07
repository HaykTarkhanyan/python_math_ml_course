# L14 Imbalanced Learning — Deck Outline / Design (v4)

Design doc for a **[14] Imbalanced Learning** deck. House style of
`ml/03_classification/12_classification_metrics.tex` and the SLIDE_STYLE guide.
Continues the chapter-3 arc (L11 logreg → L12 metrics → L13 calibration → **L14**), which
L13's "Next:" box already promises.

> v4 (credibility self-review): demonstrate the punchline on **our** data (cheese/bank, not a
> borrowed table); **regenerate Tomek** (third-party image, not ours to reuse); **sharpen the
> cost-sensitive frame** vs L12; and state the **reuse-vs-regenerate animation tradeoff honestly**
> instead of leaning on a weak "precedent." See changelog.

## The thesis (the deck's spine)
**Fix imbalance at the operating point first (metric + threshold); reach for data-level tricks
only when the *fit itself* is the bottleneck — and know they decalibrate.** Threshold + metric
(L12/L13) handle most cases. Class weights and resampling (ROS/RUS, SMOTE) genuinely help when
imbalance corrupts the *fit*, but they distort probabilities and have leakage traps. Honest:
neither a cure nor useless.

## Locked decisions
- **Scope (intro-level):** taxonomy + **class weights** + a **light cost-sensitive-learning
  idea** (one frame, no math) + **data-level resampling in depth** (under/over/hybrid,
  RUS+Tomek/cleaning, ROS, **SMOTE** + animation) + **when it actually helps** + caveats
  (leakage, decalibration, **small gains shown on our data**). Defer metric/threshold *details*
  to L12/L13.
- **Explicitly NOT covered** (too deep for intro): MetaCost, CSOVO, instance-specific costs,
  cost curves, the full performance-measures section (→ L12). Cost-sensitive learning is named
  and motivated **as an idea only**. Imbalance-specific **ensembles** → one-line pointer to ch4.
- **Running example:** **cheese factory (~3% bad)**; the bank-marketing project (~11%, 9
  categoricals) is the HW vehicle + SMOTENC motivator + a benchmark dataset.
- **SMOTE animation (the honest tradeoff):** the source `smote_viz_1..12` is **CC-BY 4.0
  (verified) and reusable with credit** — *fast, but off-brand* (foreign palette, generic
  blobs, not cheese; the same compromise flagged as L12's weakest figures). The **on-brand
  alternative** (SLIDE_STYLE preference) is to **regenerate the animation on a 2D cheese blob**.
  Default to reuse per the instructor's steer, but this is an open decision (below), stated as a
  real choice — not as a clean precedent.
- **Tomek figure:** **regenerate in house style.** The source `tomek_link_plot.png` is credited
  to **Herrera (2013)** inside the CC-BY deck → third-party, *not* ours to relicense.
- **Style:** house style — cold-open hook before `\tableofcontents`, `[plain]` transitions,
  predict-first frames, a worked-numbers frame, `\fcolorbox` boxes, real figures, Recap +
  paramgreen "Next:" box. **No HW frame** (homework on the qmd).
- **Runtime (be honest):** ~19 frames + a ~12-step click-through animation ≈ a **long video** —
  a deliberate choice given the requested sampling depth.
- **File:** `ml/03_classification/L14_imbalanced_learning.tex`; figures `fig/`, generator
  `py_src/imbalanced_figs.py`.

## Source
- `lecture_appml/slides/09_imbalancy_corr/advml-imbalanced-learning/` (LMU Advanced ML,
  **CC-BY 4.0**): `-intro` (taxonomy), `-sampling-methods-1` (ROS/RUS, Tomek, NCL/CNN/OSS),
  `-sampling-methods-2` (SMOTE idea/formula, `(1,2)/(3,1)/λ=0.25` example, `smote_viz_*`
  animation, iris demo, comparison table), `-cost-sensitive-learning-1` (cost-matrix idea —
  high-level framing only).
- **Reusable (CC-BY, credit):** `figure_man/smote_viz_1..12.pdf`, `figure_man/coordinate_system.jpeg`,
  `figure/under_oversampling.png`. **NOT reusable:** `figure/tomek_link_plot.png` (Herrera 2013).
- VERIFY at build: `class_weight="balanced"` `= n/(K·n_k)`; `imblearn` samplers + `Pipeline`;
  Elkan (2001) prior-shift.

## Callbacks / cross-references
- **L11**: `class_weight`; scaling (SMOTE interpolation needs scaled features).
- **L12**: accuracy-lies, PR-AUC/recall/MCC/BAC, threshold tuning, **leakage**.
- **L13**: proper scoring, ECE, **reweighting/resampling decalibrates → recalibrate**, `c*`.
- **Bank project**: already shows `class_weight` blowing ECE 0.01 → 0.30.
- **Forward → ch4 trees/boosting**: native imbalance handling + imbalance-specific ensembles.

## Frame-by-frame outline (~19 frames + transitions)

### Hook
1. **From measuring to fixing** — you can *score* imbalance (L12) and *trust* probabilities
   (L13), but the cheese detector still misses most of the 3% bad. Can we change the **data or
   training** to catch more? Teaser: the taxonomy.
2. **Predict-first (precise reveal):** duplicate the bad batches to 50/50 — what happens to
   **(a) recall at 0.5, (b) ranking/AUC, (c) the probabilities?** Reveal (threaded): **(a)
   recall jumps**, **(b) AUC barely moves**, **(c) probabilities inflate/decalibrate**; resample
   *before* the split → leak. ⟶ "(a) was just a threshold move in disguise."

### Outline frame (`\tableofcontents`)

### Section 1 — What you already have
3. **Why imbalance hurts + your two tools** (merged) — loss dominated by the majority → minority
   ignored (recap L12). You already have (a) the right metric and (b) the cost-tuned threshold
   (L12/L13). Box: **data tricks only after these.**
4. **Taxonomy (honest about coverage)** — algorithm-level (reweight / cost-sensitive) ·
   data-level (resample) · cost-sensitive *thresholding* (**done L12/L13**) · ensemble (**→ ch4**).
   This deck: algorithm-level (briefly) + data-level (in depth).

### Section 2 — Algorithm-level: reweighting & costs
5. **Class weights in the loss** — up-weight minority errors; `class_weight="balanced"` `=
   n/(K·n_k)`; weighted ERM (callback L11).
6. **The catch (approximation, not identity)** — under a well-specified model reweighting is
   *roughly* a prior/threshold shift (Elkan 2001), but regularized **coefficients change** and it
   **decalibrates** (callback L13; bank ECE 0.01 → 0.30) → recalibrate. Changes the fit for
   tree/SVM criteria (bridge to "when it helps").
7. **Cost-sensitive learning — just the idea (train-time, not the threshold)** — *L12/L13 put
   costs at the **decision threshold**; you can instead bake a **cost matrix** `C(j,k)` into
   **training** so the model itself minimizes expected cost.* `class_weight` is the simplest
   case (one cost per class); a full matrix is the generalization. Deeper methods (MetaCost,
   instance costs) exist → *beyond this course.* No derivations — one frame. (Sharpened vs L12.)

### Section 3 — Data-level: resampling (the heart of this deck)
*(plain transition: "Rebalance the data itself")*
8. **Overview: under / over / hybrid** — change the data, not the model; classifier-independent.
   **Signpost (reconcile with the thesis):** "these are widely used, so we go deep — but keep
   frame 4's ordering: try metric + threshold first." Schematic (`under_oversampling.png`, credited).
9. **Undersampling (Tomek carries the frame)** — **RUS** (drop majority → may discard
   informative points), then **Tomek links** with its figure + intuition (a pair of opposite-class
   nearest neighbors → drop the *majority* one to clean the border). One line: "other cleaning
   rules exist (edited-NN/NCL, CNN, OSS)." Figure: **regenerated** house-style Tomek.
10. **Oversampling** — **ROS** (replicate minority → cheap but **overfits** exact copies) →
    motivates *synthetic* oversampling → SMOTE.
11. **SMOTE — the idea + by hand** — interpolate between neighbors: `x_new = x_i + λ(x_j − x_i)`,
    `λ∈[0,1]`. Worked: `x_i=(1,2)`, `x_j=(3,1)`, `λ=0.25` → `(1.5,1.75)` (`coordinate_system.jpeg`).
12. **SMOTE — the animation** — *build past* frame 11's single point: step through generating
    **multiple** synthetic points and the **K=2 vs K=3** effect (reuse `smote_viz_1..12` as a
    `\includegraphics<n>` overlay, with source credit — *or* the regenerated cheese version; see
    open decisions).
13. **SMOTE in practice** — **pros** (generalizes the minority region vs copies) / **cons**
    (ignores the majority → can bleed across the boundary; Borderline-SMOTE & 90+ variants);
    **numeric-only → SMOTENC for categoricals — needed for the bank HW (9 categorical columns).**

### Section 4 — Does it help, and when?
*(plain transition: "The catch, and an honest verdict")*
14. **When it genuinely beats thresholding** — helps when imbalance corrupts the **fit**, not the
    operating point: minority region **too sparse to carve a boundary**; **regularization shrinks**
    minority signal; **tree split criteria** change. Rule of thumb: *threshold fixes it → you
    didn't need it; model never learned the minority region → you might.*
15. **Two traps** — **leakage** (resample the **training fold only**; `imblearn.Pipeline`;
    callback L12) and **decalibration** (distorts base rate → **recalibrate** on untouched data;
    callback L13). Figure: reliability before/after, with ECE.
16. **Reality check — on OUR data** — show `imb_benchmark` on **cheese/bank**: ROS/RUS/SMOTE give
    only **small** gains over baseline + threshold tuning (one line: "matches the literature, e.g.
    Optdigits F1 0.92 → 0.95"). Trees/boosting (ch4) handle imbalance fairly natively. **Tool, not
    a cure.** (Punchline now *demonstrated*, not borrowed.)
17. **Decision guide** (paramgreen workflow) — (1) right metric + cost threshold (L12/L13); (2)
    `class_weight` if your learner uses it, then recalibrate; (3) resampling/SMOTE only if the
    minority is truly tiny **and the fit (not the threshold) is the bottleneck** — inside CV +
    recalibrate; (4) more minority data / reframe; (5) tabular → trees/boosting (ch4).

### Wrap-up
18. **Recap** + paramgreen **"Next:"** box (→ ch4 trees).

## Figures
**Generate (`py_src/imbalanced_figs.py`, Armenian palette, house style):**
- `imb_resampling_2d.pdf` — original vs ROS vs RUS vs SMOTE on one **overlapping** 2D blob
  (comparison; overlap so SMOTE's "bleed" con shows).
- `imb_tomek.pdf` — **regenerated** Tomek-links illustration (replaces the Herrera image).
- `imb_decalibration.pdf` — reliability before/after resampling, with ECE (frame 15).
- **`imb_benchmark.pdf` — NOW REQUIRED** — PR-AUC/F1 of baseline+threshold vs ROS/RUS/SMOTE on
  **cheese (and/or bank)**, so frame 16's "gains are small" is shown, not asserted.
- *(optional)* regenerated cheese SMOTE animation, if we go on-brand instead of reusing.

**Reuse from source (CC-BY, on-slide credit):** `smote_viz_1..12.pdf` (animation, if reused),
`coordinate_system.jpeg` (worked numbers), `under_oversampling.png` (overview).

**Dependency:** `imbalanced-learn` (`uv pip install --python ./ma/Scripts/python.exe
imbalanced-learn`) for the generated resampling/benchmark figures — or hand-roll (~10 lines each).

## Open decisions (pending)
1. **SMOTE animation: reuse `smote_viz_*` (fast, off-brand) or regenerate on cheese (on-brand,
   more work)?** Default = reuse per your steer; flag if you'd rather it match the deck.
2. **Placement/number:** `L14` in `03_classification/` → chapter 3 becomes 4 lectures. OK?
3. **`imblearn` dependency:** add to `ma`, or hand-roll the samplers?
4. **One sklearn code frame** (`class_weight` + `imblearn.Pipeline` with `SMOTE`) — lean *yes*, §3.
5. **HW** (qmd): "extend the bank project — try `class_weight` and SMOTE *inside* CV, recalibrate,
   and check whether PR-AUC actually beats your threshold-tuned baseline."
6. **G-mean?** optional one-liner in frame 3, or skip.

## Changelog
**v3 → v4 (credibility self-review):**
1. **Reuse tradeoff stated honestly** — `smote_viz` reuse is CC-BY-clean but off-brand (the same
   soft spot as L12's reused pages); on-brand alternative (regenerate on cheese) named; made an
   explicit open decision instead of a misleading "precedent."
2. **Tomek regenerated** — source image is third-party (Herrera 2013), not ours to reuse →
   `imb_tomek.pdf` in house style.
3. **Punchline demonstrated on our data** — `imb_benchmark` promoted optional → **required**
   (cheese/bank); Optdigits demoted to a corroborating one-liner (frame 16).
4. **Cost-sensitive frame 7 sharpened** — now explicitly *train-time costs* vs L12's *decision-time
   threshold* (removes the L12 overlap).
5. **§3 signpost** added (frame 8) to reconcile the sampling depth with the "threshold-first" thesis.
6. **Frame 9** refocused on Tomek; NCL/CNN/OSS demoted to one line.
7. **Frames 11/12** de-duplicated — the animation now builds *past* the single worked point
   (multiple points, K=2 vs K=3).
8. Runtime honesty note added (~19 frames + 12-step animation = long video).

**v2 → v3 (instructor steer):** expanded under/over-sampling to 6 frames; reuse the SMOTE
animation; add a light cost-sensitive idea frame.
**v1 → v2 (self-review):** de-overshot thesis + "when it helps" frame; class-weight = approximation;
precise 50/50 reveal; honest taxonomy; SMOTENC tied to bank HW; SMOTE worked-numbers.
