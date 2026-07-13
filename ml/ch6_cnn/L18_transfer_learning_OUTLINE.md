# L18 - Transfer Learning and CNNs in Practice - outline (for approval)

Drafted 2026-07-13, restructured same day (4-deck split): conv variants and
detection/segmentation moved to the new L19; this deck gained the Grad-CAM / trust
frames, the BatchNorm fine-tuning gotcha, and the labeling-economics frame. Transfer
learning and Grad-CAM have NO LMU upstream - own material. Subtitle idea: "Standing on
ImageNet's shoulders."

**Interview decisions (2026-07-13, instructor, 2 rounds):**
- Grad-CAM: INTUITION ONLY - no formula frame, no saliency/occlusion comparison.
- HW3 Part A dataset: Oxford Pets CONFIRMED (resolves plan open question 4); the
  Armenian khachkar/landmark set stays a stretch goal only.
- Iconic originals (WEB-IMG): the LIME paper's actual wolf-vs-husky figure + the
  historic Clever Hans horse photo. REJECTED extras (do not add): live Colab demo
  moment, in-class labeling activity, domain-gap failure demo.
- Mandatory ANIM: the freeze/unfreeze trunk diagram unfreezing stage by stage; the
  Grad-CAM build-up animation is optional.
- Deck stays LEAN (~16 frames) - no case-study section, nothing pulled from L19;
  slower pace + discussion fills the session.
- BN gotcha: claim + rule only, no measured-evidence figure.
- Freeze-vs-fine-tune rule: a 2x2 QUADRANT DIAGRAM (data size x domain similarity,
  recipe per cell), not a table or flowchart.
- Grad-CAM subject on slides: the pomegranate photo (running-example continuity).

Target: ~16 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - the 2,000-image reality.** The chapter thread comes to a head: the pomegranate
  sorting line needs its variety/grade classifier, and you have 2,000 labeled photos.
  "Predict: what happens if you train the L17 ResNet from scratch on them?" `\pause`
  Reveal: memorizes the training set, poor test accuracy - 11M parameters vs 2,000
  examples (the L01d/[06] overfitting story at its bluntest). Hook: "the fix is to not
  start from scratch." `[predict-first]` `[callback: [06] overfitting]` `[running example]`

### Outline frame

### Section 1: Transfer learning

- `[plain]` transition: "Reuse features, not just architectures."
- **Why it works.** L16 showed early conv layers learn edges and textures, and L17 showed
  a pretrained net's conv1 filters ARE the kernel zoo - those features are the same for
  pomegranates, pets, and X-rays. Only the late, task-specific layers need your data.
  Feature hierarchy = the reusable asset. `[callback: L16 feature hierarchy, L17
  pretrained_filters figure]`
- **Two recipes.** 1) Feature extraction: freeze the pretrained trunk, train only a new
  head - tiny data suffices. 2) Fine-tuning: unfreeze some/all layers with a SMALL learning
  rate - more data, better ceiling. Rule of thumb as a 2x2 QUADRANT DIAGRAM (data size x
  domain similarity, recommended recipe in each cell). `[real fig or TikZ: quadrant]`
  `[ANIM (mandatory): the trunk diagram unfreezing stage by stage across clicks - TikZ
  \only overlays suffice, no script needed]` `[armblue key box]`
- **The fine-tune recipe in practice.** Small LR (the L15 LR intuition: you are near a good
  minimum already - do not jump away), augmentation on (L15), early stopping on val (L15).
  Everything is a callback; the only new move is freezing. `[callback: L15 LR, augmentation,
  early stopping]`
- **The BatchNorm gotcha.** Freezing the trunk's weights does NOT freeze BatchNorm's
  running statistics: fine-tuning with small batches lets the BN stats drift toward your
  tiny dataset and silently hurts accuracy. Practical rule: put BN layers in eval mode
  when the trunk is frozen; suspect this whenever a "frozen" model underperforms.
  No BN internals re-teach - L15 owns the formula. `[armorange watch-out box]`
  `[callback: L15 BatchNorm frame]`
- **The canonical snippet.** Load pretrained `resnet18`, replace `fc`, freeze trunk, train
  head. ~8 lines. `[minimal code]`
- **Payoff figure.** Same small dataset: from-scratch vs feature-extraction vs fine-tune
  accuracy curves. `[real fig: transfer_curves.pdf]`
- **When transfer fails.** Domain too far (spectrograms, medical volumes), or the
  pretrained bias mismatches; watch-out: forgetting to normalize inputs with the pretrained
  model's statistics. `[armorange watch-out box]`

### Section 2: Seeing what the net sees

- `[plain]` transition: "Trust, but verify."
- **Grad-CAM: where is the net looking?** The idea in three steps, one intuition sentence
  each, no full derivation (Selvaraju et al. 2017): take the last conv layer's feature
  maps; weight each map by the gradient of the class score with respect to it; sum, ReLU,
  upsample -> a coarse heatmap of the evidence FOR that class, overlaid on the photo.
  Show it on the classifier: heatmap over the pomegranate. Course callback: the
  interpretability module asked which FEATURES matter (importance, SHAP); Grad-CAM asks
  which PIXELS. `[real fig: gradcam.pdf]` `[ANIM candidate: photo -> coarse heatmap ->
  overlay, 3 clicks]` `[callback: 05_interpretability]`
- **Clever Hans, the vision edition.** The classic: a wolf-vs-husky classifier that was
  really a SNOW detector (Ribeiro et al. 2016) - high accuracy, wrong reason. The horse
  who "did arithmetic" by reading his trainer's face gave the effect its name. On the
  sorting line: is the model grading the fruit, or the conveyor belt behind it? Grad-CAM
  is the five-minute check before you trust a model. `[story frame]` `[running example]`
  `[armred trap box]` `[WEB-IMG: the LIME paper's actual wolf-vs-husky figure + the
  historic Clever Hans horse photo]`

### Section 3: The economics of small data

- `[plain]` transition: "The real budget is labels."
- **Somebody has to draw the boxes.** 2,000 labeled photos = hours of human annotation;
  per-pixel masks cost far more than class labels; real projects spend their budget on
  labeling and label-quality checks, not on GPUs. This is why pretraining + transfer
  matter economically: they trade YOUR labels for compute someone else already spent.
  Forward hook: detection labels (L19) are pricier still. Keep qualitative - no invented
  cost numbers. `[armblue key box]`

### Recap + Next

- Recap: freeze vs fine-tune (and when each); small LR + augmentation + early stopping;
  the BN gotcha; Grad-CAM = the trust check; labels are the budget.
- `[paramgreen Next box]`: "The classifier is built and audited. But the belt camera
  sees ten fruit at once - object detection and segmentation (L19)."

---

## Figures (py_src -> fig, `ma` venv)

1. `transfer_curves.pdf` - from-scratch vs feature-extraction vs fine-tune on small data
   (train on Colab, ship metrics; local script only plots).
2. `gradcam.pdf` - Grad-CAM heatmap over the pomegranate photo. ImageNet-pretrained
   resnet18 is enough (no training); single forward+backward, CPU-light.
3. (optional) nothing else - the labeling-economics frame stays qualitative.

## TikZ (small)

Frozen-vs-unfrozen trunk diagram; 2x2 recipe table if not a plain tabular.

## Homework hook (cnn.qmd)

HW3 - Part A: fine-tune pretrained ResNet on Oxford-IIIT Pets (default; Armenian
khachkar/landmark set only if a clean labeled set is cheap - do not gate on it).
Part B: Grad-CAM audit of the fine-tuned model - heatmaps on 5 correct and 5 wrong
predictions, plus three written sentences on what the model attends to.
(The dilation experiment moved to HW4 / L19.)

## Build notes (for the implementing model)

- No LMU upstream for this deck. Sanity-check the transfer recipe against the official
  PyTorch transfer-learning tutorial and Grad-CAM against Selvaraju et al. 2017 - not
  memory.
- **Visual style (match the built L16; interview round 1):** the MANDATORY animation is
  the freeze/unfreeze trunk diagram unfreezing stage by stage across clicks (TikZ
  `\only` overlays are fine - it is a diagram, not data). Optional extra:
  `gradcam_anim_*.pdf` (photo -> coarse 7x7 heatmap -> upsampled overlay, 3 clicks,
  emitted by gradcam.py alongside the static figure). Diagrams over text walls; WEB-IMG
  allowed for any illustrative image, no licensing caveats.
- **Iconic originals (interview round 1):** download the LIME paper's wolf-vs-husky
  figure (Ribeiro et al. 2016) and a historic Clever Hans photograph into
  `fig/borrowed/` for the Clever Hans frame.
- **Rejected extras (interview round 1 - do not add):** live Colab demo moment,
  in-class labeling activity, domain-gap failure-demo frame.
- **Credit line** on the title/Outline frame, same wording as L16.
- Facts (stable, cite the papers): Grad-CAM = Selvaraju et al. 2017; the wolf-vs-husky
  snow detector = Ribeiro et al. 2016 ("Why Should I Trust You?", the LIME paper);
  Clever Hans = the early-1900s horse, standard telling.
- `gradcam.py`: torchvision resnet18 with ImageNet weights (torchvision must be
  installed into `ma` first - see L17 build notes) + `fig/src_pomegranate.jpg`; hook the
  last conv block, one forward+backward on CPU, overlay the heatmap at ~40% alpha; fail
  loudly if the photo or weights are missing. If an HW3 fine-tuned checkpoint exists,
  prefer it; do not require it.
- `transfer_curves.py`: training runs in the HW3 notebook on Colab; the notebook saves
  metrics (csv/json) to `ml/ch6_cnn/py_src/data/`; the local script only plots and must
  fail loudly if the metrics file is missing.
- BN-gotcha frame: practical rule only (BN layers to eval mode when the trunk is
  frozen); L15 owns the BN formula.
