# CNN chapter - plan (for approval)

Drafted 2026-07-13. Supersedes the *sourcing and status* sections of
`ml/ch5_neural_networks/CNN_BLOCK_DESIGN.md` (2026-06-16); keeps its locked decisions.
Outlines live next to this file: `L16_cnn_foundations_OUTLINE.md`,
`L17_cnn_architectures_OUTLINE.md`, `L18_cnn_transfer_vision_OUTLINE.md`.

## What changed since the 2026-06-16 design

1. **Source material upgraded.** The full LMU Moodle DL course is now archived locally at
   `ml/deep_learning/moodle_s26_course/slides_tex/slides/` (ingested 2026-07-13). It includes
   `cnn3/` (LeNet/AlexNet/VGG/NiN + GoogLeNet/ResNet/DenseNet/U-Net) - the old design's
   "modern architectures have NO local upstream" gap is **closed**. Every deck now has a
   readable `.tex` upstream to borrow from.
2. **BatchNorm home resolved** (old open decision #2): L15 now teaches the BN formula
   (z-hat = (z - mu_B)/sqrt(sigma_B^2 + eps), out = gamma*z-hat + beta), added 2026-07-12.
   L17 only calls back.
3. **L14/L15 were rebuilt and enriched** - the CNN block gets stronger callback anchors:
   L15's "Reverse mode: one sweep, any architecture" frame literally promises conv layers;
   L14 has the why-GPUs frame and the XOR re-coordinatisation frame. The outlines lean on
   these hard.
4. **New chapter folder** `ml/ch6_cnn/` (instructor request) instead of L16a/b/c inside
   `ch5_neural_networks/`. Proposed numbering: plain **L16 / L17 / L18**. The old
   `L16_cnns.tex` / L17 / L18 skeletons no longer exist in ch5, so nothing collides.

## Locked decisions (carried over from CNN_BLOCK_DESIGN.md)

| Dimension | Decision |
|---|---|
| Lecture count | **3 decks**, each ~90 min |
| Framework | **PyTorch** on **Google Colab** free GPU (no CUDA locally) |
| Math rigor | mechanics + intuition; conv backward pass = "autograd handles it" (callback to L15 comp-graph frames), not derived |
| Per-lecture delivery | slide deck + practical homework (chapter `.qmd`) + `HW{n}_solution.ipynb` |
| Out of scope | GANs/diffusion/ViT deep dives (GenAI chapter), full detection/segmentation implementations, ImageNet-scale training |

## The three decks

| Deck | Working title | One-line pitch |
|---|---|---|
| L16 | CNN Foundations | dense nets can't afford pixels; convolution = a neuron with shared weights; kernels are learned, not hand-designed |
| L17 | CNN Architectures | the ImageNet story: LeNet -> AlexNet -> VGG -> ResNet; residual learning = boosting in depth |
| L18 | Transfer Learning and Vision Beyond Classification | you rarely train from scratch; conv variants (dilated/transposed/separable); detection + segmentation conceptually |

## Borrow vs add (the core of this plan)

### Borrow from LMU (`ml/deep_learning/moodle_s26_course/slides_tex/slides/`)

| LMU source | What we take | Lands in |
|---|---|---|
| `cnn1/slides-cnn-conv2d.tex` | Sobel bridge (classic CV filter -> learned filter), 2D conv worked cell-by-cell, averaging x differentiation decomposition | L16 |
| `cnn1/slides-cnn-properties-of-convolution.tex` | sparse interactions (16 vs 36 connections), parameter sharing (4 vs 36 weights), 75 vs 300M example, equivariance | L16 |
| `cnn1/slides-cnn-components.tex` + `pooling.tex` | padding (valid/same), stride, output-size formula o = floor((i-k+2p)/s)+1, max vs avg pooling toy example | L16 |
| `cnn1/slides-cnn-architecture.tex` | filters -> feature maps -> tensor -> pool -> flatten -> dense progression | L16 |
| `cnn3/slides-modern-cnn-1.tex` | LeNet, AlexNet, VGG blocks (3x3 stacking), NiN 1x1 convs, global average pooling | L17 |
| `cnn3/slides-modern-cnn-2.tex` | Inception block, ResNet residual math H(x) = F(x) + x with W_s projection, DenseNet one-liner, U-Net shape | L17, L18 |
| `cnn2/slides-dilated-transposed-convolutions.tex` | dilated conv (receptive field w/o params), transposed conv (matrix view, checkerboard artifacts + fixes) | L18 |
| `cnn2/slides-separable-convolutions-flattening.tex` | spatially separable (6 vs 9 params), depthwise+pointwise (MobileNet), multiplication counts | L18 |
| `cnn2/slides-convolution-types.tex` | 1D conv (time series, char-level text), 3D conv (video/MRI) - brief | L16 (1D first), L18 (variants) |
| `cnn1/slides-cnn-application.tex` | localization label-vector trick, task zoo, CNN-vs-dense inductive-bias framing | L18 |
| `cnn1/slides-cnn-math.tex` | **skip almost entirely** (convolution theorem/Fourier proof too deep); keep only the one-line "frameworks implement cross-correlation" footnote | L16 footnote |

Borrowing = re-derive in our own style with our own Python figures. Nothing is copied;
diagrams are regenerated (matplotlib in `py_src/`), TikZ only for small throwaway visuals.

### What we add ourselves (not in the LMU decks)

1. **The callback spine** (see table below) - this is what makes it our chapter.
2. **Predict-first frames** on the counter-intuitive points: parameter explosion,
   "what does this kernel do?", translation invariance myth, "deeper always better?",
   "2,000 images from scratch?".
3. **The Photoshop -> learned-kernels arc** (deck + HW1): targets the #1 documented CNN
   misconception (students think kernels are hand-engineered).
4. **ResNet = boosting-in-depth callback** - residual learning connects to ch4 gradient
   boosting (each tree fits the ensemble's residual; each ResNet block fits a residual on
   top of identity). Original framing, not in any upstream.
5. **Transfer learning** (all of L18 section 1) - not in the LMU CNN decks at all.
6. **CNN-vs-MLP on Fashion-MNIST** - re-uses the ch5 practical's data and MLP baseline so
   students see the same task jump in accuracy when the architecture matches the data.
7. **The pomegranate-sorting running example** (approved 2026-07-13), per the style
   guide's one-scenario-per-chapter convention: L16 cold open + kernel-zoo photo, L17
   which-architecture frame, L18 cold open + task-zoo figure + closing payoff frame.
   Doubles as the Armenian local flavor; the optional khachkar/landmark transfer set
   stays an open question.
8. **ImageNet error-over-years chart** (own matplotlib figure, labeled bars).

## Pedagogical review fixes folded in (2026-07-13)

1. **L16:** conv-layer parameter formula boxed + worked (params = k*k*C_in*C_out + C_out),
   plus a multi-channel worked example ("one filter spans the full depth -> ONE map");
   param column added to the anatomy shape table.
2. **L16:** image-as-numbers frame moved ahead of the kernel zoo; sections 2-3 run
   grayscale, color returns in section 4.
3. **L17:** ResNet-boosting framed as analogy-not-identity (joint vs staged training
   caveat on the frame).
4. **L16:** optional training-loop recap frame (include only if L15 is not fresh at
   delivery - the old design doc's prerequisite warning).
5. **All decks:** pomegranate-sorting running example threaded through.
6. **L17:** pretrained conv1-filters payoff frame closes the hand-designed -> learned arc;
   L18 "why transfer works" calls back to it.

## The course-callback spine

| New concept | Callback to | Framing |
|---|---|---|
| kernel = 9 weights + bias, affine + activation | L14 single neuron | "one neuron, reused at every pixel" |
| conv backward pass | L15 comp-graph / "reverse mode: one sweep, any architecture" | promise kept: autograd handles conv unchanged |
| weight sharing = fewer params = less overfitting | L15 weight decay, L01b Ridge | "regularization by architecture, not by penalty" |
| feature hierarchy (edges -> parts -> objects) | L14 XOR re-coordinatisation, L01g feature engineering | "conv layers re-coordinatise pixels" |
| AlexNet's tricks (ReLU, dropout, augmentation) | L15 | "everything you learned in L15, deployed at once" |
| ResNet skips help gradients | L15 vanishing gradients | reference, don't re-derive |
| ResNet residual learning | ch4 gradient boosting | "boosting in depth" |
| BatchNorm in conv blocks | L15 BN formula frame | callback only, one line |
| conv = big matrix multiply -> GPU | L14 "Why this runs on a GPU" | same frame, now with conv |
| trees win tabular / CNNs win images | ch4 + L14 honest-scope frame | inductive bias completes the story |
| tuning machinery | [08] HP tuning | not re-taught |

## Homework arc (chapter `.qmd`, PyTorch, Colab-runnable)

- **HW1 (L16) - "Build your own Photoshop, then let the network design the kernels."**
  Part A: NumPy 2D convolution from scratch + named kernel zoo on a real photo (local, no
  GPU). Part B: train a small CNN on Fashion-MNIST, visualize learned first-layer kernels,
  compare with Part A and with the ch5 MLP baseline accuracy.
- **HW2 (L17) - architecture comparison on CIFAR-10.** Small custom CNN vs ResNet-18-scale
  net; capped epochs (< 15 min/run on free Colab), checkpoint shipped.
- **HW3 (L18) - transfer learning + a short dilation experiment.** Fine-tune pretrained
  ResNet on Oxford-IIIT Pets (default; Armenian set only if cheap to assemble); Part B swaps
  in a dilated conv and observes the receptive-field effect.

Difficulty markers and solution-profile blocks per course homework conventions.

## Figure pipeline (scripts in `py_src/`, PDFs in `fig/`, run with `ma` venv)

| Script | Figure | Deck |
|---|---|---|
| `pixel_shuffle.py` | pomegranate photo: original vs fixed-permutation shuffle + zoomed number patch | L16 |
| `kernel_zoo.py` | the pomegranate photo (grayscale), 6 kernels applied: identity, box blur, Gaussian, sharpen, Sobel X/Y, emboss | L16 |
| `conv_arithmetic.py` | padding/stride/output-size diagrams (small grids) | L16 |
| `param_explosion.py` | bar chart: dense-on-image vs conv layer parameter counts (labeled bars, Armenian-flag palette) | L16 |
| `feature_maps.py` | learned first-layer kernels + activations of a small CNN trained on Fashion-MNIST | L16 |
| `cnn_vs_mlp.py` | accuracy curves: ch5 MLP baseline vs small CNN, same Fashion-MNIST subsample | L16 |
| `imagenet_error.py` | top-5 error by year 2010-2017 with architecture labels | L17 |
| `pretrained_filters.py` | conv1 filters of a torchvision pretrained model (resnet18 default) | L17 |
| `receptive_field.py` | receptive-field growth: stacked 3x3 vs one 7x7; dilated variant | L17, L18 |
| `degradation.py` | train error: 20-layer vs 56-layer plain net (schematic re-creation of He et al. Fig 1, cited) | L17 |
| `transfer_curves.py` | fine-tune vs from-scratch on small data | L18 |
| `task_zoo.py` | sorting-line photo, four task overlays (classify/localize/detect/segment) | L18 |

Heavy training runs happen on Colab; scripts that need a trained model load a shipped
checkpoint or train a tiny net on a subsample (this machine: CPU only, 16 GB - keep local
scripts light).

## Definition of done (per WORKFLOWS.md)

Each deck: 2x pdflatex, 0 `!` log lines, overflow-checked visually, aux cleaned,
`% Provenance:` block, figures generated via `ma` venv into `fig/`. Chapter `cnn.qmd`
registered in `_quarto.yml` (exact case). Commit without aux files.

## Open questions for the instructor

1. **Numbering:** plain L16/L17/L18 in `ml/ch6_cnn/` OK? (Old design said L16a/b/c inside
   ch5; the ch5 skeletons are gone, so plain numbers are free.)
2. **Folder name:** `ch6_cnn` OK? (Old design once referred to a future `ch6_genai`; GenAI
   would then become ch7.)
3. **HW1 Part B dataset:** old design locked MNIST; this plan proposes **Fashion-MNIST** so
   the CNN-vs-MLP comparison re-uses the ch5 practical baseline. Confirm the switch.
4. **Armenian dataset for HW3:** pursue a khachkar/landmark set, or ship Oxford Pets?
5. **The pomegranate photo:** thread approved (2026-07-13); still need the actual image.
   Your own photo avoids licensing; otherwise a permissive-license stock shot. For the
   L18 task-zoo figure a shot with several fruit visible works best.
6. **3 decks confirmed**, or compress L17+L18 into one architectures-and-practice deck?
