# L19 Vision Tasks - build decisions log

Built 2026-07-14 from `52_vision_tasks_OUTLINE.md` (interview-locked 2026-07-13;
GANs/style-transfer showcase added 2026-07-14). This file logs only the decisions the
outline left open, plus open questions for the reviewer. Interview-locked content was
built as specified and is not re-argued here.

## Decisions taken where the outline left room

1. **Belt photo choice (WEB-IMG, resolves plan open question 7).** Picked Wikimedia
   Commons "Pomegranates at farmers market (43134440935).jpg" (downloaded at 1920x1280
   as `fig/borrowed/pomegranate_market.jpg`). Reasons: landscape (fits 16:9), ~8 whole
   fruit clearly separable for the overlays, one split-open fruit front-center that
   makes a natural localization target and echoes L18's `src_pomegranate.jpg`.
   Runners-up (Kolkata stall: bokeh background, fewer fruit; Kandahar shipment crates:
   portrait, fruit cropped at edges) rejected on framing.
2. **Task zoo layout: 2x2 grid**, not a 1x4 strip. At 16:9 frame width a 1x4 strip
   makes each panel ~2 cm tall - boxes and mask labels become unreadable. The 2x2 at
   0.72\textheight keeps all four readable; the classification -> segmentation
   progression still reads left-right, top-bottom.
3. **Frame splits (one-idea-per-frame, per SLIDE_STYLE):**
   - Detection II (outline: IoU + NMS + mAP on one frame) became three frames: IoU
     worked example (static fig), NMS (the mandatory ANIM), and a short "mAP in one
     line" frame carrying the L12 callback.
   - Separable convolutions split into "the idea" (depthwise/pointwise) + "the count"
     (worked numbers + bar figure).
   - Dilated convolutions split into the kernel-gap ANIM frame + a "stack dilations"
     frame (receptive_field_dilated.pdf: linear 3-5-7 vs exponential 3-7-15).
   - U-Net split into architecture (paper Fig 1) + "in action" (LMU road result,
     COPY-IMG frame).
   Total 29 unique frames (37 PDF pages with overlays) vs the outline's ~21 - all
   growth comes from these splits, no added topics.
4. **GANs/style-transfer showcase stayed ONE frame** (the outline allowed a split).
   Layout: a 3-image strip (Gatys content photo, Starry Night result with painting
   inset, StyleGAN face grid) + two half-width text blocks + the honest diffusion
   one-liner box + citations. Verified visually - no clipping.
5. **Showcase visuals cropped from the papers themselves** (300 dpi pdftoppm + PIL
   crops from arXiv PDFs, same mechanism as L18's LIME figure): YOLO Fig 2
   (arXiv:1506.02640), U-Net Fig 1 (1505.04597), Mask R-CNN Fig 1 (1703.06870,
   optional item - included since the crop is clean), Gatys Fig 2 panels A and C
   (1508.06576), StyleGAN Fig 3 face block (1812.04948).
6. **Dilated figure = new script `receptive_field_dilated.py`**, not an extension of
   `receptive_field.py`: running the old script would regenerate L17's embedded
   `receptive_field.pdf` (matplotlib is not byte-deterministic - commit-noise
   LEARNINGS entry). New files only: `receptive_field_dilated.pdf` +
   `dilation_anim_0..2.pdf`.
7. **Second (optional) ANIM built: dilation 1 -> 2 -> 4** (3 clicks, kernel taps
   widening on a fixed grid), as the outline's "ANIM candidate". The
   transposed-conv upsampling ANIM candidate was NOT built - a static TikZ stamp
   sketch carries that frame; a third animation felt like diminishing returns.
   COPY-SLIDE fallback (LMU transposed-conv matrix pages) was not needed.
8. **NMS demo data**: 9 hard-coded candidates around 3 schematic objects, greedy NMS
   at IoU threshold 0.45 -> 3 survivors in 3 iterations (5 anim frames). `nms_anim.py`
   imports boxes, NMS logic and the panel-drawing function from `iou_nms.py`, so the
   anim's final frame and the static right panel cannot drift apart. Score-label
   corners are hand-assigned per box to avoid chip collisions inside clusters.
9. **Detection I classes** use market-stall nouns (pomegranate / apple / quince)
   instead of LMU's cat/car/frog - running-example flavor, same label-vector math.
10. **mAP framing**: presented as "L12 machinery wrapped around boxes, swept over IoU
    thresholds" with the 0.5-0.95 COCO-style sweep in one line - no PR-curve figure,
    keeping it the one-liner the outline requested.
11. **Next box** points to HW4 (YOLO + NMS-threshold experiment, dilated-conv swap)
    and to L20 RNN Foundations ("networks with memory"), matching the built ch7 deck.
12. **1D/3D frame** kept text-only (no figure): the outline's "REDERIVE, no figure"
    note for the convolution-types material.

## Numbers verified (all script-asserted before typesetting)

- IoU example: inter 8, union 24, IoU exactly 1/3 (`iou_nms.py` asserts via
  `fractions.Fraction`).
- Separable count: 1,228,800 vs 4,800 + 49,152 = 53,952; ratio 22.8x, stated ~23x
  (`separable_count.py` asserts the tuple).
- Dilated RF: standard stack 3, 5, 7; d=1,2,4 stack 3, 7, 15
  (`receptive_field_dilated.py` asserts).
- YOLO v1: 7x7 grid, 2 boxes/cell, 98 boxes, 7x7x30 tensor (B*5+C, C=20 on VOC) -
  from Redmon et al. 2016; frame kept version-free beyond that ("industry default",
  no Ultralytics version named).

## Compliance notes

- Interview-rejected items honored: no real YOLO output images, no IoU compute-it,
  no SAM screenshot, no Dice/mask-metrics line, no transfer-learning re-teach.
- COPY-IMG whitelist: only `unet_road_result.png` (= LMU
  `cnn3/plots/outlook/31.png`, path verified) with the CC BY 4.0 attribution line on
  the frame. All other borrowed images are WEB-IMG (no credit lines per standing
  policy; provenance recorded in the .tex Provenance block).
- No training runs, no dataset downloads; only WEB-IMG image downloads.
- Compile: 2x pdflatex, 0 `!` lines, aux cleaned (own deck only). One residual
  overfull vbox of 0.85pt on the transposed-conv frame - sub-pixel, invisible in the
  rendered page (verified).

## Open questions for the reviewer

1. **Deck length**: 29 frames vs the outline's ~21 (all from sanctioned splits).
   SLIDE_STYLE says long decks are fine; flag if the 90-min slot wants the mAP frame
   or the "stack dilations" frame folded back in.
2. **Task-zoo detection panel** shows confidence numbers (0.62-0.96) on the boxes -
   they are invented (hand-placed overlays, no model). Fine as an illustration, or
   should the numbers come off to avoid "where did these come from"?
3. **Mask R-CNN framework image** (paper Fig 1) was the outline's optional item -
   included on the semantic-vs-instance frame. Drop if the frame feels busy.
4. **The paint frame's "None of these people exist"** sits in the GAN text block;
   if a spoken beat is preferred, the line can move to a \pause reveal instead.
5. **HW4 practical + cnn.qmd chapter page** remain to be written (out of this build's
   scope per the brief - no qmd, no _quarto.yml changes).

## 2026-10-13 pedagogy + gaps pass (instructor: "improve them pedagogically, find important gaps")

All entries status `active`. Deck went 33 -> 36 frames (41 -> 44 pages).

13. **Cold-open answer corrected.** The [51] classifier outputs a grade (export / market /
    juice), not "pomegranate"; the reveal now says it grades the whole stall as one fruit.
14. **New frame: training targets for the YOLO grid** (`py_src/yolo_targets.py`, counts
    asserted: 8 fruit -> 8 distinct responsible cells, 98 predictors, 90 trained toward
    confidence 0). Why: the deck stated the "any number of objects vs fixed output" mismatch
    but solved it only at inference (candidates + NMS); assignment is the training half, and
    it hands [12]'s class imbalance (lambda_noobj, focal loss) a second home. All YOLO facts
    from the paper's Sec. 2/2.2 text, not memory. The label-vector frame now says "c_o = 0 ->
    don't care" to set it up.
15. **New frame: building your own detector** (Ultralytics' >= 1,500 images and >= 10,000 boxes
    per class; "label every object"; YOLO-World; SAM 3; draft-then-correct). Why: [51] ends with
    "hold that thought - detection needs boxes" and [52] never paid it off. Not a reversal of
    #10's interview items: no YOLO output images, no SAM screenshot - text only.
16. **U-Net figure replaced by a generated shape diagram** (`py_src/unet_shapes.py`, shapes
    traced through torch layers and asserted). Why: the paper's Fig. 1 was unreadable at its
    slot size, and the frame's message is the shapes. `unet_paper.png` stays in `fig/borrowed/`.
17. **New frame: more output structures** (keypoints with MediaPipe's 21 hand points, depth,
    text). Why: generalizes the deck's own thesis and connects to Gesture Snake.
18. **Chapter strip badges L16-L19 -> [48], [50], [51], [52]** to match the numbering the
    slides cite.

Open for the instructor: (a) segmentation still has no metric/imbalance line - the July
interview said "no Dice line"; roads are a small share of pixels, so a pixel-accuracy trap
([12]'s "97% accuracy, 0% recall") would fit naturally on the road-result frame. Not added
without a yes. (b) Student review not run on this pass.

## 2026-10-14 instructor review (13 requests, plus "pls also add the yolos founders cv/resume")

All entries status `active`. Deck went 36 -> 49 frames (44 -> 64 pages); delivered the same day
together with transfer learning. Labels from here on use the playlist numbers of #31 ([49]
architectures, [50] transfer learning, [51] this deck); the entries above use the old ones.

19. **"Nothing here" gets its own frame**, before the candidate boxes: why a c_o = 0 target must
    exist, and confidence defined as the predicted c_o. Why: the instructor asked why empty
    cases are needed. Alternative rejected: a longer remark on the label-vector frame, which was
    already full.
20. **Three frames on where candidate boxes come from**, all from a real run
    (`py_src/candidate_boxes.py` -> `results/candidate_boxes.json`): crop and classify (ResNet-18
    on 120 sliding windows, 12 s on a laptop CPU; R-CNN's ~2,000 crops at 13 s/image on a GPU,
    arXiv 1311.2524), one pass with a small head per cell (SSDlite's 6 grids x 6 anchors = 3,234
    candidates), and anchors + offsets (a 160x160 anchor decoded to 111x163: box size is
    learned, not fixed). Why: "how are those boxes obtained? ... a fixed box size? spend more
    frames". Alternative rejected: one schematic frame - it would assert the answers instead of
    showing them.
21. **New section "Building a detector" with a transition frame**; "Two families" maps
    two-stage = idea 1 made fast, one-stage = idea 2. Why: the jump into the families was abrupt.
22. **YOLO explained from the cell's side.** Confidence is defined (the predicted c_o, trained
    toward the box's IoU with the truth). New frame "A cell is an address, not a window"
    (`yolo_cell_figs.py`): a cell's own patch next to the whole photo; YOLO v1 ends in two fully
    connected layers, so every cell's numbers come from every pixel (Redmon et al. 2016, Sec. 2).
    The decode frame is now a figure (`yolo_decode.pdf`, numbers asserted) and a short table.
    Why: "how can a cell holding the belly of a dog predict anything, and how is the box bigger
    than the cells". Alternative rejected: a receptive-field answer only - true for today's
    fully convolutional YOLOs, but v1, the one the deck decodes, ends in FC layers.
23. **New frame "Meet YOLO's author"**: Redmon's CV (pjreddie.com/static/resume.pdf rendered to
    `fig/borrowed/redmon_resume.png`), the YOLOv3 paper's tone, and his 2020 exit from computer
    vision over military and privacy uses. Instructor's request.
24. **New frame "The detection loss: regression and classification, added"**: YOLO v1 (sum-squared
    error throughout, sqrt(w) and sqrt(h), lambda_coord 5, lambda_noobj 0.5; paper Sec. 2.2) next
    to modern Ultralytics YOLO (BCE for classes, CIoU + DFL for boxes; YOLO26 drops DFL and NMS;
    docs checked 2026-10-14). Why: "clarify what loss is used, since there is both regression
    and classification".
25. **Separable convs moved next to their reason**: "Making it fast: separable convolutions"
    opens with why they appear here (the detector of #20 is SSDlite-MobileNetV3, built for
    phones), with a slab drawing (`separable_idea.py`, 19,200 vs 843 weights asserted). The
    count frame gained a line telling depthwise-separable from spatially separable apart.
26. **Transposed conv as a 5-click stamp animation** (`transposed_anim.py`: 2x2 input, 3x3
    kernel, stride 2; the 5x5 result checked against `torch.nn.functional.conv_transpose2d`).
    Why: "demo transposed conv better".
27. **U-Net roads re-laid out IN -> OUT -> truth -> check** (`road_panels.py` crops the LMU
    figure's four panels), with In and Out spelled out in the text. Why: the source's 2x2 grid
    put the input bottom-right and the instructor could not tell input from output. 2026-10-15:
    the credit line moved under the figure, because it touched the footer.
28. **New section "The 2026 toolbox"**: a SAM frame (self-drawn `sam_diagram.py`: heavy image
    encoder once per photo, light prompt encoder + mask decoder ~50 ms per prompt on CPU;
    SA-1B 11M images, 1.1B masks; SAM 2 video, SAM 3 concept prompts) and a toolbox table
    (YOLO26, RF-DETR, SAM 3, DINOv3, Gemini's box_2d; YOLO27 announced, not out). Reopens #10's
    "no SAM screenshot" only partly: the masks are drawn by hand and the slide says so. Why:
    "spend a couple of slides on industry standards nowadays like SAM".
29. **1D/3D frame: the movie-review figure removed** (instructor: confusing); one column now.
30. **Two frames on neural style transfer, idea level**: the optimization loop
    (`style_transfer_figs.py` -> `style_loop.pdf`: frozen VGG-19, content at conv4_2, style via
    Gram matrices at conv1_1-conv5_1, gradient descent on the pixels; Gatys et al. 2015) and a
    Gram matrix by hand (`gram_toy.pdf`: shuffling positions leaves G unchanged, asserted).
    Johnson et al. 2016 is named as the real-time version. Instructor's request.
31. **The chapter's deck numbers follow the playlist** (2026-10-15, after delivery; instructor:
    "i havent done practicals in between so last 2 are 50 and 51"). Files `50_*` -> `49_*`,
    `51_*` -> `50_*`, `52_*` -> `51_*`, the practical `53_` -> `52_`; slide labels follow in
    every deck that cites them. Details in the repo's `DECISIONS.md`.

Open: 81 frames in one session (with [50]) was the heaviest so far - check the notes PDF for
where the lecture actually stopped. Student review not run on this pass.
