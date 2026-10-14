# L19 - Vision Tasks: Detection and Segmentation - outline (for approval)

Drafted 2026-07-13, split out of the old L18 in the 4-deck restructure; restructured
same day after the instructor interview (variants now taught just-in-time). Source: conv
variants from LMU `cnn2/{dilated-transposed, separable, convolution-types}`; task zoo +
localization from `cnn1/application`; U-Net from `cnn3/modern-cnn-2`; detection
vocabulary (IoU / NMS / YOLO) = own material. This deck closes the chapter (chapter-arc
recap, inductive-bias ledger, ViT hand-off). Subtitle idea: "One photo, four questions."

**Interview decisions (2026-07-13, instructor):**
- Conv variants taught JUST-IN-TIME, not as an up-front toolkit: separable lives inside
  the detection section (real-time/edge context); transposed + dilated live inside the
  segmentation section (right before U-Net uses them); 1D/3D moves to the closing
  section as the generalization beat.
- Detection gains a YOLO GRID MECHANICS frame (how one-stage actually predicts).
  REJECTED: embedding real YOLO output images in the deck; IoU as a compute-it moment
  (the worked example is presented, not student-computed).
- Chapter close gains a CHAPTER-ARC RECAP frame (the whole L16-L19 story). REJECTED:
  SAM demo screenshot, mask-metrics (Dice) one-liner.
- Multi-fruit belt photo: WEB-DOWNLOAD one (resolves plan open question 7) - a
  sorting-line or market-stall shot with several pomegranates, into `fig/borrowed/`.
- (Added 2026-07-14, instructor direction) GANs + NEURAL STYLE TRANSFER appear in the
  chapter close as SHOWCASE EXAMPLES ONLY - no math, no training details, no depth.
  One frame (builder may split into two if layout demands - log it).

Target: ~21 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - the belt sees ten fruit.** The L18 classifier grades ONE fruit per photo.
  The real belt camera frame holds ten. "Predict: what does the classifier output on
  this photo?" `\pause` Reveal: one label for the whole image - useless. We need models
  whose OUTPUT has structure: boxes, masks. That is today.
  `[predict-first]` `[running example]`

### Outline frame

### Section 1: One photo, four tasks

- `[plain]` transition: "What changes is the output, not the backbone."
- **The task zoo.** Classification -> classification+localization -> object detection ->
  semantic segmentation, one image shown four ways. What changes is the OUTPUT
  structure, not the backbone. `[real fig: task_zoo.pdf, on the web-downloaded
  multi-fruit photo]`

### Section 2: Object detection

- `[plain]` transition: "What and where."
- **Detection I: from "what" to "what and where".** Extend the label vector to
  [b_x, b_y, b_h, b_w, objectness, class one-hot] - classification plus box regression
  in one output (adapted from LMU cnn1/application). Then the real difficulty: a photo
  can hold ANY number of objects, but a network has a fixed output size - the whole
  field of detection is engineering around that mismatch.
- **Detection II: the working vocabulary.** IoU = intersection over union, the overlap
  score between two boxes. Worked example, use exactly: A = (0,0)-(4,4), B = (2,0)-(6,4)
  -> intersection 8, union 16+16-8 = 24, IoU = 1/3 (presented, not a compute-it - per
  interview). NMS (non-max suppression): the net proposes many overlapping boxes; keep
  the highest-scoring one, drop neighbors above an IoU threshold. mAP in one line
  (precision/recall machinery from L12, swept over IoU thresholds). `[worked-numbers]`
  `[real fig: iou_nms.pdf]` `[ANIM: nms_anim flip-book - boxes pruned click by click]`
  `[callback: L12 metrics]`
- **Detection III: two families + the industry standard.** Two-stage: R-CNN -> Fast ->
  Faster R-CNN (propose regions, then classify each; accurate, slower). One-stage: YOLO
  ("You Only Look Once", 2016) and SSD - one forward pass, real-time. YOLO became the
  industry default: a decade of iterations, and in practice you pip-install a pretrained
  YOLO and fine-tune it on your boxes - that IS the sorting line's counting camera.
  `[running example]` `[armblue key box: two-stage = accuracy, one-stage = speed; the
  gap has mostly closed]`
- **Detection IV: how one-stage actually works (YOLO grid mechanics).** Divide the image
  into an SxS grid; EACH cell predicts a few candidate boxes (x, y, w, h, confidence)
  plus class probabilities; all cells predict simultaneously in one forward pass - that
  is why it is real-time. The flood of overlapping candidates is exactly why NMS
  (Detection II) exists. Numbers from the original paper for concreteness: 7x7 grid,
  2 boxes per cell. `[WEB-IMG: YOLO grid figure from the paper]`
  `[callback: Detection II NMS]`
- **Separable convolutions (just-in-time: why detection runs on a belt camera).**
  Depthwise (one filter per channel) + pointwise (1x1 mix) - the efficiency trick behind
  MobileNet-class backbones that real-time, on-device detectors ride on. Worked
  multiplication count, use exactly: input 12x12x3, 5x5 filters, 256 output channels ->
  output 8x8; standard conv = 256*25*3*64 = 1,228,800 multiplications; depthwise
  3*25*64 = 4,800 + pointwise 256*3*64 = 49,152 -> 53,952 total, ~23x fewer.
  `[worked-numbers]` `[callback: L17 1x1 convs]` `[running example]`

### Section 3: Segmentation

- `[plain]` transition: "Every pixel gets an answer."
- **Segmentation I: semantic vs instance.** Per-pixel classification. Semantic: every
  pixel gets a class label - all pomegranates merge into one red mask. Instance: each
  fruit gets its OWN mask (Mask R-CNN = Faster R-CNN + a mask head). Panoptic = both at
  once, one line. SAM callback: segmentation is now also a promptable foundation-model
  capability (L17 epilogue). `[callback: L17 epilogue SAM]`
- **Transposed convolution (just-in-time: the output must be an image).** Going the
  other way: upsample a small feature map back toward pixel resolution - needed whenever
  the OUTPUT is an image, which per-pixel segmentation demands. Matrix view in one line;
  watch-out: checkerboard artifacts, fixed by kernel-divisible-by-stride or
  upsample-then-conv. `[armorange watch-out box]` `[ANIM candidate: upsampling filling
  in cell by cell]`
- **Dilated (atrous) convolution (just-in-time: context without losing resolution).**
  Insert gaps in the kernel: receptive field grows exponentially with depth, parameter
  count unchanged - segmentation's way to see the whole scene while keeping per-pixel
  detail. Also: WaveNet/TCN audio and time series. `[TikZ dilation-gap diagram]`
  `[real fig: receptive_field.pdf, dilated panel]` `[callback: L17 receptive field]`
  `[ANIM candidate: dilation 1 -> 2 -> 4 across 3 frames]`
- **Segmentation II: U-Net, the workhorse.** Now that both tools are on the table:
  downsample (convs/pooling) then upsample (transposed convs) back to full resolution,
  with skip connections carrying detail across - L17's skips again, used for resolution
  instead of optimization; dilated convs grow context mid-net. Medical imaging's
  favorite architecture. `[callback: L17 skip connections + the two just-in-time
  variants]` `[WEB-IMG: U-Net diagram (or TikZ U-shape if a clean one is not found)]`

### Section 4: The sorting line, completed - and the chapter close

- `[plain]` transition: "Cash in the running example, then zoom out."
- **The sorting line, completed (thread payoff).** One scenario, four output structures:
  classification grades each fruit (L16); transfer learning built it from 2,000 photos
  (L18); a fine-tuned YOLO counts and localizes fruit on the belt; a small U-Net
  outlines the bruised region for the cutter. Close with one-liners on other domains
  (roads, satellite, medical). Light, one frame. `[running example]`
- **Same idea, other shapes: 1D and 3D (the generalization beat).** 1D conv on time
  series and text (L16's opening strip, now for real: sensor data, char-level text); 3D
  conv on video and MRI volumes. Convolution is a structural prior, not an image trick -
  sets up the ledger. `[moved here per interview: just-in-time = right before the
  zoom-out]`
- **CNNs can also paint (showcase, not deep - added 2026-07-14).** Two examples, one
  intuition sentence each, big visuals: 1) NEURAL STYLE TRANSFER (Gatys et al. 2015) -
  repaint a photo in a painting's style by matching CNN features: content = deep-layer
  activations, style = feature correlations - and the features come from VGG (L17!).
  2) GANs (Goodfellow et al. 2014) - two networks in a duel: a generator forges
  images, a discriminator calls fakes, both improve until the fakes pass (StyleGAN's
  "this person does not exist" faces); the generator is built from THIS lecture's
  transposed convolutions. Honest one-liner: for image generation, diffusion has
  largely taken over (L17 epilogue) - GANs live on in upscaling and faces. No formulas,
  no training mechanics - the GenAI chapter owns generation.
  `[WEB-IMG: a classic style-transfer triptych (content photo / painting / result) +
  a StyleGAN face grid]` `[callback: L17 VGG + epilogue diffusion; this deck's
  transposed convs]` `[story frame]`
- **The inductive-bias ledger (chapter close I).** CNNs assume locality + translation
  equivariance. That is why they win on images with less data - and why they are NOT the
  answer for tabular (ch4: trees/boosting) and struggle when global context dominates.
  Honest-scope close, same spirit as L14's tabular frame. `[callback: ch4, L14]`
- **The chapter in one frame (chapter-arc recap, per interview).** Four beats, one
  strip: L16 built the layer (a neuron reused at every pixel) -> L17 evolved it into
  architectures (ImageNet story) -> L18 taught reuse and trust (transfer, Grad-CAM) ->
  L19 structured the output (boxes, masks). One visual band, four icons/mini-figures -
  the chapter's own timeline ribbon. `[real fig or TikZ: chapter_arc strip]`
- **ViT hand-off (chapter close II).** Vision Transformers drop the convolutional prior
  and buy it back with scale - attention over image patches (already met in the L17
  epilogue). One frame, forward pointer to the GenAI / DL4NLP material. No math.
  `[callback: L17 epilogue]` `[paramgreen pointer: GenAI chapter]`

### Recap + Next

- Recap: the output structure defines the task; IoU/NMS vocabulary; two detection
  families + YOLO grid mechanics; semantic vs instance; U-Net and its two just-in-time
  tools; convolution as a prior with a domain.
- `[paramgreen Next box]`: forward pointer to HW4 (run a YOLO, dilation experiment) and
  to the next chapter per the course plan.

---

## Figures (py_src -> fig, `ma` venv)

1. `task_zoo.pdf` - the web-downloaded multi-fruit photo, four task overlays
   (classification / localization / detection / segmentation) - matplotlib boxes/masks
   over the photo.
2. `iou_nms.pdf` - left panel: two boxes with the IoU computation annotated; right
   panel: many overlapping detections -> the NMS survivors. Pure matplotlib,
   hard-coded boxes, no model.
3. `receptive_field.pdf` - shared with L17; the dilated panel is used here.
4. `nms_anim_*.pdf` - ANIM flip-book for the Detection II frame: one script
   (`nms_anim.py`) emits N frames of overlapping boxes being pruned click by click.
5. `chapter_arc.pdf` - the four-beat chapter strip for the arc-recap frame (or TikZ).
6. (optional) `separable_count.pdf` - multiplication-count bars, standard vs separable.

## TikZ (small)

Dilation-gap diagram; transposed-conv upsampling sketch; U-shape fallback; chapter-arc
strip fallback.

## Homework hook (cnn.qmd)

HW4 - Part A: run a pretrained YOLO (ultralytics package, Colab) on your own photos;
inspect boxes and confidences; vary the NMS IoU threshold and report what appears and
disappears. Part B (moved from the old HW3): swap a dilated conv into the small L16 CNN
and observe the receptive-field / accuracy effect. Difficulty markers TBD.

## Build notes (for the implementing model)

- Consult read-only (REDERIVE):
  `ml/deep_learning/moodle_s26_course/slides_tex/slides/cnn2/`
  `{slides-dilated-transposed-convolutions, slides-separable-convolutions-flattening,
  slides-convolution-types}.tex`, `cnn1/slides-cnn-application.tex`,
  `cnn3/slides-modern-cnn-2.tex` (U-Net). Detection vocabulary (IoU/NMS/YOLO families,
  grid mechanics) has no LMU upstream - author from this outline.
- **Credit line** on the title/Outline frame, same wording as L16.
- **Visual style (match the built L16):** the MANDATORY animation is `nms_anim.py` (THE
  flip-book of the chapter: many overlapping detections, prune click by click, end state
  = iou_nms right panel). Optional extras: dilation 1 -> 2 -> 4 across 3 frames;
  transposed-conv upsampling filling in cell by cell. Diagrams over text walls
  throughout; WEB-IMG for U-Net / YOLO / Mask R-CNN diagrams, no licensing caveats.
- **Belt photo (interview decision, resolves plan open question 7):** web-download a
  photo with several pomegranates (sorting line or market stall) into `fig/borrowed/`;
  `task_zoo.py` reads it from there and must fail loudly if missing. Hand-placed overlay
  coordinates hard-coded in the script; no model inference.
- **YOLO grid facts (stable, from Redmon et al. 2016):** v1 used a 7x7 grid, 2 boxes per
  cell, per-cell class probabilities, one forward pass. Keep the frame version-free
  beyond that ("the YOLO family"); verify the current Ultralytics release only if you
  decide to name one.
- Detection/segmentation facts (stable, cite the papers): R-CNN 2014 (Girshick et al.),
  Faster R-CNN 2015 (Ren et al.), YOLO v1 2016 (Redmon et al., "You Only Look Once"),
  SSD 2016 (Liu et al.), Mask R-CNN 2017 (He et al.), U-Net 2015 (Ronneberger et al.).
- `iou_nms.py`: hard-coded boxes; the IoU worked example is A = (0,0)-(4,4),
  B = (2,0)-(6,4) -> intersection 8, union 24, IoU = 1/3 - figure and frame must match.
- Separable worked count is fixed in the outline (12x12x3 / 5x5 / 256 channels):
  1,228,800 vs 53,952 multiplications.
- **COPY-IMG whitelist:** LMU's own U-Net road-segmentation result (under
  `ml/deep_learning/_reference/lecture_i2dl/slides/cnn3/plots/outlook/`, e.g. `31.png` -
  verify the exact path exists). The original U-Net paper diagram: WEB-IMG is fine.
- **COPY-SLIDE option** (recipe in the L16 build notes): compiled sources
  `slides/week24_cnns_modern/{01_convolution-types, 02_dilated-transposed-convolutions,
  03_separable-convolutions-flattening}.pdf`. Candidate: the transposed-convolution
  matrix-multiplication walkthrough pages, if the redraw stalls.
- **Rejected in the interview (do not add):** real YOLO output images embedded in the
  deck; IoU as a compute-it moment; SAM demo screenshot; Dice/mask-IoU metrics line.
- Transfer-learning content lives in L18 - do not duplicate it here.
- **GANs / style transfer showcase (added 2026-07-14):** WEB-IMG both visuals (a
  classic Gatys-style triptych - the paper's own figure is fine - and a StyleGAN face
  grid); cite Gatys et al. 2015, Goodfellow et al. 2014, StyleGAN = Karras et al. on
  the frame as paper citations. NO local style-transfer optimization runs, no GAN
  training, no formulas (standing rule: no training, no dataset downloads; small
  WEB-IMG downloads are fine).
