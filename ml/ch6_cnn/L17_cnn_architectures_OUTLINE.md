# L17 - CNN Architectures: the ImageNet story - outline (for approval)

Drafted 2026-07-13. Source: LMU `cnn3/{modern-cnn-1, modern-cnn-2}` +
`cnn1/architecture` + own material (ImageNet-error chart, ResNet-boosting callback,
degradation predict-first). Subtitle idea: "Ten years that changed computer vision."

Target: ~19 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - one chart, one question.** ImageNet top-5 error by year, 2010-2017, bars
  labeled with the winning system. "Something happened in 2012. Predict: what?" `\pause`
  Reveal: AlexNet - the first deep CNN winner - cut error from 26% to 16% in one year;
  by 2015 machines passed the human baseline (~5%). Everything in this lecture is on that
  chart. `[predict-first]` `[real fig: imagenet_error.pdf]`

### Outline frame

### Section 1: The benchmark that made deep learning

- `[plain]` transition: "ImageNet" + one motivation line.
- **ImageNet + ILSVRC.** 1.2M labeled images, 1000 classes; why a shared benchmark
  accelerates a field (same reason we always compare on held-out data - [06]).
  Top-5 error defined. `[callback: [06] evaluation]`

### Section 2: The classics - LeNet, AlexNet, VGG

- `[plain]` transition: "Three architectures, three ideas."
- **LeNet (1998): the blueprint.** conv-pool-conv-pool-dense, digits on checks. Everything
  in L16 was already here - what was missing was compute and data. `[TikZ block strip]`
- **AlexNet (2012): the breakout.** Same blueprint, scaled: 8 layers, 60M params, trained
  on 2 GPUs. Its "new" tricks are all L15 material deployed at once: ReLU (vs saturating
  sigmoids), dropout, data augmentation. "You already know every ingredient."
  `[callback: L15 dropout, augmentation, activations; L14 GPU frame]`
- **What AlexNet learned (the L16 arc, paid off).** Conv1 filters of a pretrained network:
  crisp Gabor-like edge, blob and color detectors - the L16 kernel zoo, reinvented by
  gradient descent, designed by nobody. L16's tiny Fashion-MNIST kernels looked blurry; at
  ImageNet scale the learned kernels become textbook. Closes the hand-designed -> learned
  arc. `[real fig: pretrained_filters.pdf]` `[callback: L16 kernel zoo + punchline frame]`
- **VGG (2014): depth via small filters.** Only 3x3 convs, stacked. Worked count: two
  stacked 3x3 layers see a 5x5 receptive field with 2 x 9C^2 weights vs 25C^2 for one 5x5 -
  deeper AND cheaper, with an extra nonlinearity for free. `[worked-numbers]`
- **Receptive field.** How far one deep neuron "sees" into the input; grows with depth.
  Sets up dilated convolutions (L18). `[real fig: receptive_field.pdf]`

### Section 3: Going deeper - degradation and ResNet

- `[plain]` transition: "If depth is good, why not 100 layers?"
- **Predict-first: the degradation problem.** "A 56-layer plain net vs a 20-layer plain net,
  same recipe. Which has lower TRAINING error?" `\pause` Reveal: the 20-layer one. Not
  overfitting (training error!) - deep plain nets are just hard to optimize.
  `[predict-first]` `[real fig: degradation.pdf, re-created from He et al. 2015, cited]`
- **The residual idea.** Let a block learn the correction, not the whole mapping:
  H(x) = F(x) + x; the skip connection carries x for free, the layers learn the residual
  F(x). Learning "do nothing" is now trivial (push F to 0) - so extra layers can no longer
  hurt. Dimension-matching projection W_s x in one line. `[boxed math, short]`
- **You have seen residual learning before.** Gradient boosting (ch4): every new tree fits
  the residual of the ensemble so far. A ResNet block does the same thing in depth: each
  block fits a correction on top of what the network already computes. Boosting stacks
  residual learners in sequence; ResNet stacks them in depth. Caveat line ON the frame:
  analogy, not identity - boosting fits each learner to the FROZEN ensemble's residual,
  stage by stage; ResNet blocks train jointly, end to end. `[armblue key box]`
  `[callback: ch4 boosting]` `[armorange one-liner: limits of the analogy]`
- **Why skips also fix gradients.** The identity path gives the gradient an undamped route
  backwards - the vanishing-gradient problem from L15, relieved by wiring rather than by
  activation choice. Reference only, no re-derivation. `[callback: L15 vanishing gradients]`
- **The modern conv block.** conv -> BatchNorm -> ReLU, repeated, plus skips. BN is the
  L15 formula, unchanged - one callback line, no re-teach. `[callback: L15 BatchNorm frame]`

### Section 4: The wider zoo, briefly

- `[plain]` transition: "Other tricks that stuck."
- **1x1 convolutions + Inception.** 1x1 conv = per-pixel dense layer across channels
  (cheap depth mixing); Inception runs several filter sizes in parallel and concatenates.
  One frame, intuition only.
- **Global average pooling.** Replace the giant flatten-dense head by averaging each final
  feature map to one number (NiN idea) - kills most parameters of the classic head; used by
  everything modern. Worked count: 256 maps of 100x100 -> dense-10 head = 25.6M weights;
  GAP head = 2,560. `[worked-numbers]`
- **Scaling and searching.** EfficientNet compound scaling in one line; architectures are
  now found by search as much as designed by hand. DenseNet name-check (concatenate, don't
  add). No depth here - pointer to the LMU/d2l material for the curious.

### Section 5: Using architectures in practice

- `[plain]` transition: "Nobody types out a ResNet."
- **torchvision.models.** The canonical snippet: load `resnet18`, print parameter count,
  swap the head for your classes. ~6 lines. `[minimal code]`
- **Which architecture when (the sorting line decides).** Back to the pomegranate line:
  a Fashion-MNIST-sized toy -> the small custom CNN from L16; the real sorting camera ->
  start from a pretrained ResNet-family model - which is exactly transfer learning, next
  lecture. `[paramgreen takeaway box]` `[running example]`

### Recap + Next

- Recap: benchmark -> progress; LeNet blueprint; AlexNet = L15 tricks at scale; VGG = small
  filters, deep; degradation -> residual learning (= boosting in depth); BN blocks; GAP;
  load pretrained, don't retype.
- `[paramgreen Next box]`: "These models were trained on 1.2M images. You have 2,000.
  Next: transfer learning - stand on ImageNet's shoulders (L18)."

---

## Figures (py_src -> fig, `ma` venv)

1. `imagenet_error.pdf` - top-5 error by year, labeled bars, human line.
2. `receptive_field.pdf` - stacked 3x3 vs single large filter receptive-field growth.
3. `degradation.pdf` - 20 vs 56-layer plain-net training error (schematic re-creation,
   He et al. 2015 cited on the frame).
4. `pretrained_filters.pdf` - conv1 filters of a torchvision pretrained model (resnet18 is
   the light download; alexnet if the classic look is wanted). One-time weight download;
   plot-only script otherwise.
5. (optional) `resnet_block.pdf` - residual block diagram if TikZ feels too cramped.

## TikZ (small)

LeNet/AlexNet/VGG block strips; residual block with skip arrow; Inception parallel paths.

## Homework hook (cnn.qmd)

HW2 - compare a small custom CNN vs a ResNet-18-scale net on CIFAR-10; capped epochs
(< 15 min/run on free Colab), shipped checkpoint for the analysis part.
