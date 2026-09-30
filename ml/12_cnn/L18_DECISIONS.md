# L18 Transfer Learning - build decisions (2026-07-14)

Decisions taken where the approved outline left room, plus open questions.
Interview-locked content was not re-litigated.

## Decisions

1. **transfer_curves.pdf is a SCHEMATIC, not measured data.** The outline has the
   training run in the HW3 Colab notebook with the local script only plotting shipped
   metrics - but that notebook does not exist yet. The figure therefore shows the three
   canonical curve shapes with deliberately unlabeled axes and a printed disclaimer
   "schematic - typical shapes, real curves come with HW3"; the slide text says the same.
   `py_src/transfer_curves.py` auto-switches to plotting measured curves the moment
   `py_src/data/hw3_metrics.csv` appears (columns: epoch, from_scratch,
   feature_extraction, fine_tune; fails loudly on a malformed file). When HW3 lands,
   rerun the script and recompile the deck - nothing else changes.
2. **19 frames instead of the outline's ~16.** Two splits, both for the one-idea-per-
   frame rule: (a) the 2x2 quadrant got its own frame instead of sharing the two-recipes
   frame with the mandatory unfreeze ANIM; (b) Grad-CAM became idea frame (with the
   3-click ANIM) + verdict frame (static figure, honest prediction, [05] callback).
   No new content was added; rejected extras stayed out.
3. **Grad-CAM overlay uses per-pixel alpha, not a constant 40% wash.** A constant-alpha
   jet blend over an already-red photo was unreadable (verified by rendering). The
   overlay bakes alpha = 0.5 * cam^1.5 into an RGBA image - ~40-50% at the hotspot,
   fading to 0 where evidence is low - so the heatmap reads as a spotlight. Note:
   matplotlib's `imshow(..., alpha=<array>)` was silently ignored in the PDF output;
   baking alpha into the RGBA array is the robust path.
4. **Real Grad-CAM run, honest numbers.** ImageNet resnet18 (IMAGENET1K_V1), one
   forward+backward on CPU, layer4 hooked. Actual top-1 on src_pomegranate.jpg:
   **pomegranate, p = 0.995** (then strawberry 0.004, trifle 0.001). The verdict frame's
   "p = 0.99" is the measured value. Param count on slides is 11.7M = the script-logged
   11,689,512 (outline said "11M"; the verified number is used).
5. **pretrained_filters.pdf embedded.** The L17 sibling build had generated
   `fig/pretrained_filters.pdf` by compile time, so the "why transfer works" frame embeds
   the same file (per the coordinator's rule: reference, don't duplicate). If L17
   regenerates it, this deck just recompiles.
6. **WEB-IMG choices.** Wolf-vs-husky = the actual Figure 11 of Ribeiro et al. 2016,
   cropped at 300 dpi from the arXiv PDF (captions "(a) Husky classified as wolf /
   (b) Explanation" preserved) -> `fig/borrowed/wolf_husky_lime.png`. Clever Hans =
   Wikimedia Commons "Osten und Hans.jpg" (von Osten, the horse, and the arithmetic
   blackboard) -> `fig/borrowed/clever_hans.jpg`.
7. **Quadrant cell advice** follows the canonical four-scenario guidance (cs231n
   transfer-learning notes): small+close = feature extraction; large+close = fine-tune;
   small+far = hardest case, extract from earlier layers + simple head, get more data;
   large+far = fine-tune everything (pretrained init still helps). The sorting line
   (2,000 photos, natural images) is marked in the small+close cell.
8. **Recipe facts web-verified 2026-07-14** against the official PyTorch
   transfer-learning tutorial: weights="IMAGENET1K_V1", replace fc via fc.in_features,
   freeze via requires_grad=False, optimizer over model.fc.parameters() only,
   SGD lr=0.001 momentum=0.9, ImageNet normalization (0.485,0.456,0.406) /
   (0.229,0.224,0.225). Grad-CAM steps verified against Selvaraju et al.,
   arXiv:1610.02391 (ICCV 2017).
9. **Cold open reveal** uses `\onslide<2->` instead of `\pause` so the pomegranate photo
   (which sits after the reveal in source order) is visible before the click.

## Open questions for the reviewer

1. The "[05]" / "[06]" module-callback notation appears on three frames (verdict, Clever
   Hans, cold open). It matches the outline's own notation, but if students do not know
   the bracket numbering, swap for "the interpretability module" / "the overfitting
   lecture" wording.
2. The Grad-CAM ANIM frames vary a few points in size between clicks (bbox_inches="tight"
   + different title lengths). Visually negligible at 0.62 linewidth; can be pinned with
   fixed axes rects if the click-jitter bothers.
3. transfer_curves: after HW3 ships real metrics, the payoff frame's armblue box text
   ("These are the typical shapes...") should flip to a measured-data caption.
