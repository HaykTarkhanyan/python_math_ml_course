# L18 - Transfer Learning and Vision Beyond Classification - outline (for approval)

Drafted 2026-07-13. Source: transfer learning = own material (not in the LMU CNN decks);
conv variants from LMU `cnn2/{dilated-transposed, separable, convolution-types}`; task zoo
from `cnn1/application` + `cnn3/modern-cnn-2` (U-Net). Subtitle idea: "Standing on
ImageNet's shoulders."

Target: ~19 frames, one ~90-min session.

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
  rate - more data, better ceiling. Rule of thumb by dataset size + similarity (2x2 table).
  `[armblue key box]`
- **The fine-tune recipe in practice.** Small LR (the L15 LR intuition: you are near a good
  minimum already - do not jump away), augmentation on (L15), early stopping on val (L15).
  Everything is a callback; the only new move is freezing. `[callback: L15 LR, augmentation,
  early stopping]`
- **The canonical snippet.** Load pretrained `resnet18`, replace `fc`, freeze trunk, train
  head. ~8 lines. `[minimal code]`
- **Payoff figure.** Same small dataset: from-scratch vs feature-extraction vs fine-tune
  accuracy curves. `[real fig: transfer_curves.pdf]`
- **When transfer fails.** Domain too far (spectrograms, medical volumes), or the
  pretrained bias mismatches; watch-out: forgetting to normalize inputs with the pretrained
  model's statistics. `[armorange watch-out box]`

### Section 2: Convolutions beyond the vanilla

- `[plain]` transition: "Three variants you will meet in the wild."
- **Dilated (atrous) convolution.** Insert gaps in the kernel: receptive field grows
  exponentially with depth, parameter count unchanged. Where: segmentation, WaveNet/TCN
  audio and time series. `[TikZ dilation-gap diagram]`
  `[real fig: receptive_field.pdf, dilated panel]` `[callback: L17 receptive field]`
- **Transposed convolution.** Going the other way: upsample a small feature map back
  toward pixel resolution (needed whenever the OUTPUT is an image). Matrix view in one
  line; watch-out: checkerboard artifacts, fixed by kernel-divisible-by-stride or
  upsample-then-conv. `[armorange watch-out box]`
- **Separable convolutions.** Depthwise (one filter per channel) + pointwise (1x1 mix):
  worked multiplication count vs a standard conv - the efficiency trick behind MobileNet /
  on-device vision. `[worked-numbers]` `[callback: L17 1x1 convs]`
- **Same idea, other shapes: 1D and 3D.** 1D conv on time series and text (the L16 opening
  strip, now for real: sensor data, char-level text); 3D conv on video and MRI volumes.
  One frame - the point is that "convolution" is a structural prior, not an image trick.

### Section 3: Beyond classification (conceptual only)

- `[plain]` transition: "One label per image is the easy case."
- **The task zoo.** Classification -> classification+localization -> object detection ->
  semantic segmentation, one image shown four ways. What changes is the OUTPUT structure,
  not the backbone. `[real fig or TikZ 4-panel]`
- **Detection in one frame.** Extend the label to [box coords, objectness, class one-hot];
  the multi-object problem; R-CNN family vs single-shot (YOLO/SSD) as one-line name-drops.
  (Adapted from LMU cnn1/application.)
- **Segmentation in one frame.** Per-pixel classification; U-Net shape: downsample
  (convs/pooling) then upsample (transposed convs) with skip connections carrying detail
  across - L17's skips again, used for resolution instead of optimization.
  `[callback: L17 skip connections]` `[TikZ U-shape sketch]`
- **The sorting line, completed (thread payoff).** One scenario, four output structures:
  classification grades each fruit (L16); transfer learning built it from 2,000 photos
  (this lecture); detection counts and localizes fruit on the belt; segmentation outlines
  the bruised region for the cutter. Close with one-liners on other domains (roads,
  satellite, medical). Light, one frame. `[running example]`

### Section 4: Wrap-up - what CNNs assume, and what comes after

- `[plain]` transition.
- **The inductive-bias ledger.** CNNs assume locality + translation equivariance. That is
  why they win on images with less data - and why they are NOT the answer for tabular
  (ch4: trees/boosting) and struggle when global context dominates. Honest-scope close,
  same spirit as L14's tabular frame. `[callback: ch4, L14]`
- **ViT teaser.** Vision Transformers drop the convolutional prior and buy it back with
  scale - attention over image patches. One frame, forward pointer to the GenAI/DL4NLP
  material. No math.

### Recap + Next

- Recap: transfer learning (freeze vs fine-tune, small LR); dilated / transposed /
  separable and where each lives; detection and segmentation = new output structures on the
  same backbone; CNN = a prior, priors have domains.
- `[paramgreen Next box]`: forward pointer to the practical (HW3 fine-tune) and to the next
  chapter (sequences / attention or GenAI, per course plan).

---

## Figures (py_src -> fig, `ma` venv)

1. `transfer_curves.pdf` - from-scratch vs feature-extraction vs fine-tune on small data
   (train on Colab, ship metrics/checkpoint; local script only plots).
2. `receptive_field.pdf` - shared with L17; add dilated panel.
3. `task_zoo.pdf` - one sorting-line photo with several pomegranates visible, four task
   overlays (classification / localization / detection / segmentation) - matplotlib
   boxes/masks over the photo.
4. (optional) `separable_count.pdf` - multiplication-count bars, standard vs separable.

## TikZ (small)

Dilated-kernel gaps; transposed-conv upsampling sketch; U-Net shape; frozen-vs-unfrozen
trunk diagram.

## Homework hook (cnn.qmd)

HW3 - Part A: fine-tune pretrained ResNet on Oxford-IIIT Pets (default; Armenian
khachkar/landmark set only if a clean labeled set is cheap - do not gate on it).
Part B: swap a dilated conv into the small L16 CNN and observe the receptive-field /
accuracy effect.
