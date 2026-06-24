# CNN Lecture Block: Design

**Date:** 2026-06-16
**Status:** Reviewed and revised; awaiting final approval before implementation plan
**Location of work:** `ml_new/ch5_neural_networks/`

## Problem

The neural-networks chapter (`ml_new/ch5_neural_networks/`) is all skeleton. CNNs in particular are a single overloaded deck, `L16_cnns.tex` — 11 frames, every content frame just a `% TODO: port from lecture_i2dl/cnn1+cnn2` comment, cramming 9 section headers (convolution basics, properties, components, pooling, classic architectures, modern architectures, conv variants, applications) into one "lecture." That is realistically 2-3 lectures of material with no hands-on path, no pedagogy, and no homework.

The instructor wants CNNs taught as a **deep, practical, 3-lecture block** with a coding component, following the course's established per-topic delivery pattern: **slide deck + practical homework + solution codebase**.

## Goal

Replace the single `L16_cnns.tex` skeleton with a planned **3-lecture CNN block**, each lecture a three-part unit:

1. a project-styled Beamer **slide deck**,
2. a **practical homework** (problems in the chapter `.qmd`),
3. a **solution codebase** (`HW{n}_solution.ipynb`, PyTorch, Colab-runnable).

This spec defines the structure, per-lecture content, homework arc, sourcing, and conventions. Authoring the decks/notebooks is the implementation phase (separate plan).

## Decisions (locked during brainstorming)

| Dimension | Decision |
|---|---|
| Lecture count / depth | **3 lectures**, deep + practical. NB: "deep" here means **breadth of topics + hands-on mastery**, *not* mathematical depth — see math rigor row |
| Framework | **PyTorch**, run on **Google Colab** free GPU (instructor machine has no CUDA) |
| Math rigor | **Mechanics + intuition, light proofs.** Rigorous about *what* convolution does (worked numeric examples, output-size formulas, weight-sharing/sparsity intuition); the conv layer's backward pass is "autograd handles it," not derived |
| Delivery per unit | slide deck + practical homework (`.qmd`) + solution codebase (`.ipynb`) |
| Numbering | **L16a / L16b / L16c** — splits the existing `L16_cnns.tex`; leaves L17 (RNN) and L18 (optimizers) unchanged. Consistent with existing letter-suffix companions (L01b, L05b, L13b) |

## Prerequisites & chapter dependencies

This block does **not** teach neural-network fundamentals. L16a assumes students arrive (from L14/L15, which must be delivered first) knowing:

- the MLP / fully-connected network and its forward pass,
- loss + gradient descent + backpropagation at a working level,
- how to read/write a **basic PyTorch training loop** (`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`, epochs, autograd).

If L14/L15 have not been delivered when L16a runs, L16a must include a 1-2 frame recap of the training loop, or HW1 Part B (training a CNN) inherits unbudgeted teaching debt.

**Cross-chapter overlap to avoid re-teaching or contradicting:**

- ResNet skip connections (L16b) relate to **vanishing gradients** — covered in L15 / L17. Reference, don't re-derive.
- **BatchNorm** (open decision below) ties to L18 (optimizers/init). Decide one home for it.

## Scope

### In scope

Three lecture units in `ml_new/ch5_neural_networks/`:

- **L16a — CNN Foundations** (convolution, pooling, first CNN)
- **L16b — CNN Architectures** (the ImageNet story: LeNet → ResNet)
- **L16c — Transfer Learning, Advanced Convolutions & Vision Tasks**

Plus: the practical homeworks (in a chapter `.qmd`, see below) and three `HW{n}_solution.ipynb` solution notebooks. Delete `L16_cnns.tex` once L16a/b/c compile and land.

**Homework `.qmd`:** this spec creates `ml_new/ch5_neural_networks/05_neural_networks__concepts.qmd` (or appends the CNN problems to the NN-chapter `.qmd` if one is created first), following the course homework conventions: `{data-difficulty="1|2|3"}` 🧀 levels and `{.content-visible when-profile="solution"}` for solution visibility.

### Out of scope (YAGNI)

- GANs / diffusion / Vision Transformer deep dives — these belong to the GenAI chapter (`ch6_genai`) and `misc/dl4nlp/`.
- Full *implementations* of object detection / semantic segmentation — covered conceptually only.
- Training from scratch on ImageNet.
- Deriving the backward pass through a conv layer.
- Renumbering L17/L18.

## Per-lecture deck outlines

Target ~18-22 frames per deck (one ~90-min session each).

### L16a — CNN Foundations

Port from `_reference/lecture_i2dl/slides/cnn1/{introduction, conv2d, math, properties, components, pooling}`.

1. **Why CNNs.** MLP-on-images parameter explosion. Predict-first `\pause`: how many weights does a single dense layer on a 224×224×3 image need? (reveal: ~150M for 1000 units) → motivates local connectivity + weight sharing.
2. **The convolution operation.** Start in **1D** (sliding window + dot product is easiest to see in one dimension), then generalize to a worked **2D** numeric example by hand. Frame the kernel as a **"feature detector."**
3. **The kernel zoo.** Classic image-processing kernels and their visual effect on a real photo: identity, box blur, Gaussian blur, sharpen, Sobel X/Y (edge), emboss, Laplacian. Predict-first `\pause`: "what will this kernel do to the image?" This frame sets up HW1 Part A and the "hand-designed vs learned filters" payoff.
4. **Properties of convolution.** Weight sharing, sparse connectivity, translation equivariance. Myth-buster predict-first `\pause`: "are CNNs translation-*invariant*?" (reveal: no — convolution is *equivariant*; pooling/downsampling buys only *limited* invariance. Belief in full invariance is a documented misconception.)
5. **Components.** Stride, padding, channels; the output-size formula.
6. **Pooling.** Max vs average; why downsample.
7. **Anatomy of a conv stack** (LeNet-style) → bridge to PyTorch `nn.Conv2d` / `nn.MaxPool2d`.

### L16b — CNN Architectures (the ImageNet story)

Port from `cnn1/architecture`. **Modern architectures (ResNet/Inception/EfficientNet) have NO local upstream — author as original content, referencing CS231n and the original papers.**

1. **ImageNet as the driver.** LeNet (1998) → AlexNet (2012: ReLU, dropout, GPU) → VGG (2014: depth via stacked 3×3).
2. **The degradation problem → ResNet (2015).** Skip connections. Predict-first `\pause`: does adding more layers always lower training error? (reveal: no — plain deep nets degrade; residual connections fix it). Reference vanishing gradients (L15/L17), don't re-derive.
3. **Inception / EfficientNet** — briefly (width, compound scaling).
4. Bridge: `torchvision.models` (sets up HW2 and L16c transfer learning).

### L16c — Transfer Learning, Advanced Convolutions & Vision Tasks

Port from `cnn1/application` + `cnn2/{convolution-types, dilated-transposed-convolutions, separable-convolutions-flattening}`; transfer-learning content is original. **Conv variants live here (moved from L16b) so they sit next to the dense-prediction tasks that use them.**

1. **Why transfer learning.** Small-data reality; feature extraction vs fine-tuning (freeze/unfreeze).
2. **Data augmentation** (`torchvision.transforms`); the fine-tune recipe.
3. **Advanced convolutions** (explicit treatment, not one-liners):
   - **Dilated (atrous) convolution** — TikZ diagram of the dilation gaps; grows receptive field *without* adding parameters; used in segmentation, WaveNet.
   - **Transposed convolution** — upsampling / "deconv"; used to go from features back to pixel resolution.
   - **Separable convolution** — depthwise + pointwise; the efficiency trick behind MobileNet.
4. **Beyond classification** (conceptual): object detection (bounding boxes), semantic segmentation (per-pixel) — where dilated/transposed convolutions show up.
5. **Wrap-up + ViT teaser** linking forward to the GenAI / dl4nlp chapter.

## Practical homeworks + solution codebases

All notebooks PyTorch and Colab-runnable on free GPU. Part A of HW1 is pure NumPy and runs locally without a GPU.

### HW1 (L16a) — "Build your own Photoshop", then let the network design the kernels

Deliberate two-part arc: hand-designed filters → learned filters.

- **Part A — Build your own Photoshop** (NumPy, local). Implement 2D convolution from scratch. Apply named kernels to a real image and visualize: identity, box blur, Gaussian blur, sharpen, Sobel X/Y, emboss, Laplacian.
- **Part B — Let the network design the kernels** (PyTorch, Colab). Train a small CNN on MNIST. Visualize the learned first-layer kernels and compare them to Part A's hand-designed ones.

Payoff: "Photoshop filters are hand-designed convolution kernels; a CNN *learns* its kernels." This is the central intuition of the whole block and directly targets the #1 documented CNN misconception.

### HW2 (L16b) — Compare architectures on CIFAR-10

Train and compare 2 architectures on CIFAR-10; observe depth / skip-connection effects.

**Feasibility (must scope in the notebook):** use *small* variants (e.g. a small custom CNN vs a ResNet-18-scale net), **cap epochs** (target **< 15 min/run** on Colab free GPU), and **ship a checkpoint** so students who hit Colab disconnects can still complete the analysis. Do not ask for training-to-convergence from scratch.

### HW3 (L16c) — Transfer learning (+ a short dilation experiment)

- **Part A — Transfer learning.** Fine-tune a pretrained ResNet on a small dataset. **Default dataset: Oxford-IIIT Pets or Flowers-102** (clean, built into torchvision, reliable). An Armenian landmark / khachkar set is a *nice-to-have* only if a clean labeled set can be cheaply assembled — do not gate the homework on data that may not materialize.
- **Part B — Dilation / receptive-field experiment** (short). Swap in a dilated convolution and observe the receptive-field / accuracy effect — reinforces the L16c advanced-convolutions material in code, not just on slides.

## Style & pedagogy conventions

- Project Beamer style: `\documentclass[aspectratio=169]{beamer}`, `\input{../preamble}`, dove theme, popblue / sampred / paramgreen / warnred palette.
- TikZ for the convolution animation and the dilated-kernel diagram — no external images for diagrams.
- Predict-first `\pause` frames on the counter-intuitive points (param explosion, "what will this kernel do?", "is it translation-invariant?", "does deeper always help?").
- Armenian local examples alongside the canonical Western ones.
- Target ~18-22 frames per deck so each fits a ~90-min session.
- Run `beamer-overflow-check` after each deck compiles.

## Open decisions (resolve before/at implementation)

1. **HW3 Armenian dataset** — pursue a khachkar/landmark set, or stay with the Oxford default? (Default is Oxford; Armenian only if a clean set is cheap to assemble.)
2. **BatchNorm in CNNs** — teach it inside this block (L16b, since AlexNet→ResNet rely on it) or defer to L18 (optimizers/init)? Pick one home; avoid duplication.
3. **Homework difficulty levels** — assign 🧀 / 🧀🧀 / 🧀🧀🧀 per problem.
4. **NN-chapter `.qmd`** — does CNN get its own homework `.qmd`, or share one NN-chapter `.qmd` across L14-L18? (Spec assumes a chapter `.qmd`.)
5. **Folder-naming inconsistency** — Chapter 1 lives in `01_regression_intro/`, the NN chapter in `ch5_neural_networks/`. Not blocking; flag only.

## Sourcing summary

| Lecture | Local upstream available? |
|---|---|
| L16a Foundations | Yes — `cnn1/{introduction, conv2d, math, properties, components, pooling}` |
| L16b Architectures | Partial — `cnn1/architecture` present; **modern archs (ResNet/Inception/EfficientNet) NOT present → original content (CS231n + papers)** |
| L16c Transfer / adv-conv / vision | Yes for conv variants — `cnn2/{convolution-types, dilated-transposed-convolutions, separable-convolutions-flattening}` + `cnn1/application`; transfer-learning content original |

## Pedagogy research findings (2026-06-16)

Web research into CNN-teaching best practices. Findings folded into the outlines above:

1. **Intuition → math → implementation** is the validated ordering — matches the block's structure (deck intuition → light math → PyTorch homework).
2. **#1 documented misconception: students think kernels are hand-engineered, not learned.** HW1's "Photoshop (hand-designed) → CNN learns its own kernels" arc is a direct intervention on exactly this error. Strongest single validation of the design.
3. **#2 misconception: CNNs are translation-*invariant*.** They aren't — convolution is *equivariant*; pooling/downsampling buys only limited invariance. Added as a predict-first myth-buster frame in L16a.
4. **Introduce 1D convolution before 2D** — the sliding dot-product is easiest to grasp in one dimension. Added to L16a's convolution frame.
5. **"Feature detector" terminology** is the recommended intuitive framing for a kernel. Adopted in L16a.
6. **Visualization makes convolution click** — supports the TikZ-animation and kernel-zoo approach.

Sources:

- [CS231n: Convolutional Networks (course notes)](https://cs231n.github.io/convolutional-networks/)
- [10-315 CNN Notes, Carnegie Mellon](https://www.cs.cmu.edu/~10315/notes/10315_S25_Notes_CNNs.pdf)
- [CNN for Skin Lesion Classification: hands-on learning (NCBI)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7969634/)
- [The Most Intuitive Guide for CNNs (TDS)](https://medium.com/data-science/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480)
- [All about convolutions, kernels, features in CNN](https://medium.com/@abhishekjainindore24/all-about-convolutions-kernels-features-in-cnn-c656616390a1)

## Status / next step

Design reviewed and revised. When picked up again, the next step is the `writing-plans` skill → a deck-by-deck + notebook-by-notebook implementation plan. No authoring has started.
