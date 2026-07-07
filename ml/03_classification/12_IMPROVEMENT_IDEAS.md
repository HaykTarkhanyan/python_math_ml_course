# L12 Classification Metrics - improvement ideas

Parked ideas beyond the 2026-07-07 edit batch (outline "64" image, accuracy grid, cropped LMU
images, PR-AUC rework, gain/lift split + decile lift, worked macro/micro/weighted example).
Ranked by value. None of these are done yet.

## High value
- **`\pause` on the predict-first frames.** Three frames spoil their own answer: "where accuracy
  breaks" (97%/0-recall), "P and R at extreme thresholds", "ROC's blind spot" (A vs B). ch5 now
  hides the answer with `\pause` until students commit. These are the deck's best gotchas and are
  currently given away. Cheap, high pedagogical payoff.
- **"Metrics are estimates" caveat.** Every number (AUC, AP, F1) is computed on a finite test set
  and has sampling noise. One short frame or box: report a bootstrap CI, and don't over-read a
  0.91-vs-0.90 AUC gap. Currently absent; students treat point estimates as exact.
- **Weighted != micro for precision/F1.** The new worked example uses recall, where weighted =
  micro = accuracy, so it can't show weighted as a distinct third thing. A one-line follow-up with
  per-class *precision* (or F1) would show weighted landing strictly between macro and micro.

## Medium value
- **Split the "metric zoo" frame.** Eight metrics (TPR/FNR/TNR/FPR | PPV/FDR/NPV/FOR) plus the
  overall box is the densest slide in the deck. Consider a 2-up reference card or moving the
  rarely-used rates (FOR, NPV) to a backup slide.
- **Tiny visual for ROC-AUC's probabilistic meaning.** "P(random positive scores above random
  negative)" is stated but abstract; a 6-dot pairwise-comparison sketch would make Mann-Whitney
  concrete.
- **Multiclass ROC AUC / one-vs-rest** gets only a one-liner. Half a frame on macro-OVR AUC and
  when to prefer OVO would round out the multiclass section now that averaging has a worked example.

## Low value / nice-to-have
- **Cohen's kappa** next to MCC in the zoo (agreement corrected for chance) - one row.
- **Lift *curve*** (continuous lift vs fraction) alongside the decile bars, for the reader who
  wants the smooth version. Deciles already carry the lesson, so optional.
- **A "which threshold-tuning method" mini-recap** at the end of the threshold section (cost vs
  Youden vs recall-floor) mirroring the metric decision flowchart.
- **Multi-label vs multi-class** aside: micro/macro averaging also applies to multi-label, where
  micro != accuracy. One sentence to prevent a common conflation.

## Known overflow (pre-existing, discovered 2026-07-07)
- **"Which metric for which problem?" table** overflows ~16pt: the "Default: report 3-4 metrics"
  footer sits tight against the page number (not clipped, but cramped). Trim a `Why` cell or drop
  a row to a backup slide. (The badly-broken sklearn frame - whole footgun box off-slide - and the
  precision/recall caption clip were fixed in the 2026-07-07 batch.)

## Housekeeping
- `py_src/cheese_metrics.py` header still says `ml/ch2_classification` in comments/paths prose
  (the code resolves correctly via `parents[1]`, but the docstring is stale). Trivial doc fix.
- Original uncropped `roc_build_*.png` are still in `fig/` next to the `*_crop.png` now used.
  Keep for provenance, or delete the originals if repo size matters.
