# L19 Vision Tasks - build decisions log

Built 2026-07-14 from `L19_vision_tasks_OUTLINE.md` (interview-locked 2026-07-13;
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
