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

## 2026-10-13 pedagogy + gaps pass (instructor: "improve them pedagogically, find important gaps")

All entries status `active`. Deck went 26 -> 31 frames (32 -> 41 pages).

10. **The canonical snippet now calls `model.eval()`.** Why: the slide before it says "frozen
    trunk -> BN in eval mode", and the snippet then said "the usual training loop, unchanged",
    which runs `train()` and drifts the BN statistics - the deck taught the bug it warned about.
    The Gesture Snake practical already does it right (Part 7). Alternative rejected: freezing
    only the BN modules with a helper - more code than the one-snippet budget allows. Revisit
    if the snippet grows a head with dropout (then `eval()` on the whole model is wrong).
11. **New frame: per-stage parameter counts** (`py_src/trainable_params.py`, asserted). Why: the
    counter-intuitive fact (top block = 75.1% of ResNet-18, head = 1,539) is what makes "start
    by unfreezing the top block" a real decision, and "1,539 numbers on 2,000 photos" flips the
    cold open's overfitting argument with numbers. Alternative rejected: a table only - the bar
    lengths make the 75% visible in one glance. Revisit if the deck switches its backbone.
12. **New frame: why freeze first (LP-FT).** Why: the deck called "freeze first, unfreeze
    gradually" the one new move and never said why. Evidence = Kumar et al. ICLR 2022 abstract
    (+2% in-distribution / -7% shifted for plain fine-tuning vs linear probe; LP-FT +1% / +10%
    over plain fine-tuning). "Linear probe" is now defined here, not on the 2026 frame.
    Alternative rejected: Yosinski et al. 2014 layer-transferability evidence - supports the
    quadrant, not the ordering question, and [48]/[50] already show the feature hierarchy.
13. (Superseded by #19, 2026-10-14.) **New frame: two small-data traps, with the practical's measured number** (1-NN on pixels:
    random split 100.0% vs burst split 43.2%). Why: real local measurement, already in the
    executed notebook; no new run. The flip-changes-the-label trap is the practical's Part 3.
    Alternative rejected: re-measuring on fruit photos - no such dataset exists here.
14. **"When transfer fails" gained He, Girshick & Dollar 2019** (scratch matches pretraining on
    COCO given longer training) and an input-contract box (ImageNet stats, cv2 BGR,
    `weights.transforms()`, checked on torchvision 0.24). Why: students should know pretraining
    buys speed and small-data robustness, not a guaranteed higher ceiling.
15. **New frame: Grad-CAM is class-discriminative, and what it cannot say.** Figure = the
    paper's own Fig. 1 a/c/i (cropped, `fig/borrowed/gradcam_cat_dog.png`). Alternative
    rejected: our own run for a second ImageNet class on the pomegranate photo - one object, so
    it cannot show the cat-vs-dog split that defines the property.
16. **New frame: "Spending fewer labels"** - qualitative only, keeping the July interview's
    "zero invented numbers" rule for labeling economics.

Open for the instructor: the student review (Sonnet, rendered PNGs) was not run on this pass.

## 2026-10-14 instructor review of the 2026-10-13 pass

Five requests, all done. Deck went 31 -> 32 frames (41 -> 40 pages). All entries `active`.

17. **New frame "ResNet-18: a trunk and a head"**, placed after "Why transfer works"
    (`py_src/resnet18_arch.py`). Shapes are traced through torchvision's resnet18 with forward
    hooks and asserted, and so is the count of 18 weight layers. Why: the instructor said the
    deck froze "block 1-4" without ever showing what they are. [50] teaches the residual idea
    but never draws ResNet-18 stage by stage. The frame also says where the head's 512 inputs
    come from: block 4's 512 maps of 7x7, global-average-pooled. The parameter-count frame's
    `512 * 3 + 3` bullet now says the same in one line.
    Alternatives rejected:
    - TikZ: essential figures are matplotlib.
    - The paper's Table 1: it covers every ResNet depth and is unreadable at slide size.
    Revisit if the deck switches backbones.
18. **"Trunk" is kept and explained, not renamed to "backbone".** The new frame's box explains
    that head and trunk are body words, and that "backbone" is the more common term, which
    [52] uses. The "Why transfer works" box no longer uses "trunk" before the word is defined.
    Alternative rejected: renaming to "backbone" throughout [51]. Nobody asked for it, and it
    would touch about 25 lines plus the trunkrow diagram. Revisit if students mix up the two
    words; then unify on "backbone" in both decks.
19. **Removed the "Two small-data traps" frame (supersedes #13).** The instructor's call. The
    content is not lost: the Gesture Snake practical measures the burst split (Part 6) and
    swaps the label when it flips a crop (Part 3). The matching recap bullet went too; a
    trunk + head bullet replaced it.
20. **The three backbone frames (SSL, CLIP, 2026) are now their own section, "Bigger
    shoulders", with a transition slide.** Why: the instructor found the jump from "When
    transfer fails" to self-supervised learning too quick. The transition line names the
    change: ImageNet's 1.2M hand-labeled photos vs the 400M (CLIP) to 1.7B (DINOv3) photos
    that nobody labeled by hand. It adds one Outline entry.
21. **The Grad-CAM "another class" frame now says how to ask:** backprop the chosen class's
    score (`model(x)[0, c].backward()`, the same call `gradcam.py` makes with the top class).
    In ResNet-18, alpha_k then comes from row c of fc, consistent with the step-2 frame's
    alpha_k = w_k / 49. Why: the instructor asked "how do you ask for another class?"; the
    frame said that you can, not how. The figure shrank from 3.45 in to 3.1 in to fit.

Still open: the student review was not run on this pass either.
