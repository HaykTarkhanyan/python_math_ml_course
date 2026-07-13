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
   `ch5_neural_networks/`. Numbering: plain **L16 / L17 / L18 / L19** (L19 added in the
   2026-07-13 restructure). The old `L16_cnns.tex` / L17 / L18 skeletons no longer exist
   in ch5, so nothing collides - but note L19 takes the number the old RNN design
   assumed was free; renumber the future RNN block accordingly.

## Locked decisions (carried over from CNN_BLOCK_DESIGN.md)

| Dimension | Decision |
|---|---|
| Lecture count | **4 decks**, each ~90 min (locked at 3 in the 2026-06-16 design; expanded 2026-07-13 when detection/segmentation grew - see "The four decks") |
| Framework | **PyTorch** on **Google Colab** free GPU (no CUDA locally) |
| Math rigor | mechanics + intuition; conv backward pass = "autograd handles it" (callback to L15 comp-graph frames), not derived |
| Per-lecture delivery | slide deck + practical homework (chapter `.qmd`) + `HW{n}_solution.ipynb` |
| Out of scope | GANs/diffusion/ViT deep dives (GenAI chapter), full detection/segmentation implementations, ImageNet-scale training |

## The four decks (restructured from 3 on 2026-07-13)

The old L18 had grown into three unrelated topics (transfer learning + conv variants +
detection/segmentation, ~22 frames of new vocabulary). Instructor approved the split:
vision tasks moved to a new L19, and the slimmed L18 gained the Grad-CAM / trust frames,
the BatchNorm fine-tuning gotcha, and the labeling-economics frame. L16 was already in
build when the split happened and is untouched by it. Outline files:
`L18_transfer_learning_OUTLINE.md` (replaces `L18_cnn_transfer_vision_OUTLINE.md`) and
`L19_vision_tasks_OUTLINE.md`.

**Build status (2026-07-13):** L16 is BUILT (v3, compiled) - its outline file is now
historical; the deck's `.tex` + Provenance block are the source of truth for L16
(instructor-driven v2/v3 changes: image-fundamentals front half, conv animation,
edges-as-derivatives arc, translation-invariance frame removed, demo image = skimage
astronaut, pomegranate = cold open only). L17-L19 outlines remain the working spec.

| Deck | Working title | One-line pitch |
|---|---|---|
| L16 | CNN Foundations | dense nets can't afford pixels; convolution = a neuron with shared weights; kernels are learned, not hand-designed |
| L17 | CNN Architectures | the ImageNet story: LeNet -> AlexNet -> VGG -> ResNet; residual learning = boosting in depth; the people (LeCun, Fei-Fei Li, Simonyan) + a vision-in-2026 epilogue (ViT, VLMs, diffusion, SAM) |
| L18 | Transfer Learning and CNNs in Practice | you rarely train from scratch; freeze vs fine-tune (+ the BatchNorm gotcha); Grad-CAM and Clever Hans (is it grading the fruit or the belt?); labels are the real budget |
| L19 | Vision Tasks: Detection and Segmentation | detection (label trick, IoU/NMS, families + YOLO grid mechanics); segmentation (semantic vs instance, U-Net) with the conv variants taught just-in-time; chapter close (arc recap + inductive-bias ledger + ViT hand-off) |

## Borrow vs add (the core of this plan)

### Borrow from LMU (`ml/deep_learning/moodle_s26_course/slides_tex/slides/`)

| LMU source | What we take | Lands in | How |
|---|---|---|---|
| `cnn1/slides-cnn-conv2d.tex` | Sobel bridge (classic CV filter -> learned filter), 2D conv worked cell-by-cell, averaging x differentiation decomposition | L16 | REDERIVE + REGEN-FIG |
| `cnn1/slides-cnn-properties-of-convolution.tex` | sparse interactions (16 vs 36 connections), parameter sharing (4 vs 36 weights), 75 vs 300M example, equivariance | L16 | REDERIVE + TIKZ + REGEN-FIG |
| `cnn1/slides-cnn-components.tex` + `pooling.tex` | padding (valid/same), stride, output-size formula o = floor((i-k+2p)/s)+1, max vs avg pooling toy example | L16 | REDERIVE |
| `cnn1/slides-cnn-architecture.tex` | filters -> feature maps -> tensor -> pool -> flatten -> dense progression | L16 | REDERIVE + TIKZ |
| `cnn1/slides-cnn-introduction.tex` | the application-tour idea (showcase montage) | L16 | COPY-IMG (see whitelist) |
| `cnn3/slides-modern-cnn-1.tex` | LeNet, AlexNet, VGG blocks (3x3 stacking), NiN 1x1 convs, global average pooling | L17 | REDERIVE + WEB-IMG (download architecture diagrams; supersedes the earlier "redraw, don't copy" note) |
| `cnn3/slides-modern-cnn-2.tex` | Inception block, ResNet residual math H(x) = F(x) + x with W_s projection, DenseNet one-liner, U-Net shape | L17, L19 | REDERIVE + TIKZ; U-Net showcase optionally COPY-IMG |
| `cnn2/slides-dilated-transposed-convolutions.tex` | dilated conv (receptive field w/o params), transposed conv (matrix view, checkerboard artifacts + fixes) | L19 | REDERIVE + TIKZ |
| `cnn2/slides-separable-convolutions-flattening.tex` | spatially separable (6 vs 9 params), depthwise+pointwise (MobileNet), multiplication counts | L19 | REDERIVE |
| `cnn2/slides-convolution-types.tex` | 1D conv (time series, char-level text), 3D conv (video/MRI) - brief | L16 (1D first), L19 (variants) | REDERIVE |
| `cnn1/slides-cnn-application.tex` | localization label-vector trick, task zoo, CNN-vs-dense inductive-bias framing | L19 | REDERIVE + REGEN-FIG (task zoo on our photo) |
| `cnn1/slides-cnn-math.tex` | **skip almost entirely** (convolution theorem/Fourier proof too deep); keep only the one-line "frameworks implement cross-correlation" footnote | L16 footnote | REDERIVE |

### Borrowing mechanics - how borrowed material physically enters our decks

License basis: `slds-lmu/lecture_i2dl` is **CC BY 4.0** (verified in the local clone's
LICENSE file, 2026-07-13), so copying its images with attribution is allowed. Caveat:
some figures inside the LMU decks are themselves third-party (taken from papers or
blogs) - LMU's CC BY does not cover those. For paper figures: cite the original paper,
or better, do not copy them at all.

Four mechanisms, in order of preference:

1. **REDERIVE** (default for concepts, math, worked examples, tables). Read the named
   LMU `.tex`, then write the frame from scratch in our own words, our notation
   (`ml/preamble.tex` macros) and house structure. No asset is copied; the LMU deck is a
   content checklist, not a template. Never paste their LaTeX - their `vbframe`
   environment, style files and macros do not exist in our preamble and will not compile.
2. **REGEN-FIG** (default for every essential figure). A matplotlib script in `py_src/`
   recreates the idea from scratch with our own data/photo. See the figure pipeline.
3. **COPY-IMG** (exception, needs listing). Copy a specific image file from
   `ml/deep_learning/_reference/lecture_i2dl/slides/<chapter>/figure/` into
   `ml/ch6_cnn/fig/borrowed/`, embed with `\includegraphics`, and put an attribution
   line on the frame: "Source: LMU I2DL (slds-lmu/lecture_i2dl), CC BY 4.0" - or the
   original paper's citation if the image is a paper figure. Reserved for showcase
   images we cannot generate (application photos, paper result collages). Every COPY-IMG
   use must appear in the deck outline's build notes whitelist - do not add new ones
   silently.
4. **TIKZ** (small throwaway visuals only). Boxes-and-arrows the frame could live
   without, drawn fresh - never traced pixel-for-pixel from an LMU diagram.
5. **COPY-SLIDE** (allowed per instructor decision 2026-07-13; use sparingly). Embed a
   full LMU slide as a page image from the compiled PDFs in
   `ml/deep_learning/moodle_s26_course/slides/`:
   `\includegraphics[page=N, height=0.92\textheight]{<deck>.pdf}` on a `[plain]` frame,
   with a small source line on that frame. LMU decks are 4:3, ours 16:9 - fit by height
   and accept side margins. Best for multi-overlay click-through sequences that would be
   expensive to recreate (they arrive as consecutive PDF pages, so several `page=N`
   frames in a row reproduce the animation). Each deck's build notes name the candidate
   pages; do not add others silently.
6. **WEB-IMG** (added 2026-07-13 after the L16 build; instructor direction). Illustrative
   images and architecture diagrams may be DOWNLOADED from the web into
   `ml/ch6_cnn/fig/borrowed/` whenever a good one exists and recreating it is costly -
   e.g. LeNet/AlexNet/VGG/Inception/ResNet architecture diagrams, the U-Net diagram, a
   YOLO grid figure. Per the instructor's standing instruction, do NOT add licensing
   caveats or per-image credit lines for assets used this way; the chapter-level LMU
   credit line stays as is. L16 already uses this (`multi_kernel.png`). This supersedes
   earlier "redraw, don't copy" notes for architecture diagrams.

### Visual style: animations and diagrams (2026-07-13, instructor direction)

The L16 build set the bar the remaining decks must match:

- **ANIM flip-books:** multi-frame matplotlib sequences (`name_0.pdf .. name_N.pdf`
  emitted by ONE `py_src/` script) shown with `\only<n>` so the lecturer advances them
  by clicking. L16's 9-frame convolution animation is the template. Every remaining deck
  gets at least one; candidates are listed in each outline's build notes.
- **Diagrams over text walls:** when a frame's core content is something visual (an
  architecture, a data flow), show a real diagram (WEB-IMG or generated figure), not
  bullets. TikZ stays for small schematics only.
- **`\pause` builds** remain the cheap animation for step-wise reveals inside one frame.

**Credit rule (instructor decision):** every deck carries a one-line acknowledgment on
its title or Outline frame - "Parts of this chapter adapt LMU I2DL (slds-lmu),
CC BY 4.0" - and `cnn.qmd` repeats it. License note for the record: the material is
CC BY 4.0 (verified in the clone's LICENSE), which requires attribution and permits any
use including commercial - so a free course with up-front credits is safely covered.

Still avoid: reusing the LMU beamer theme wholesale, or a deck that is mostly embedded
LMU pages. The default remains REDERIVE so the chapter keeps the house voice; COPY-SLIDE
is the escape hatch, not the plan.

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
6. **CNN-vs-MLP on Fashion-MNIST** - re-uses the ch5 practical's data and MLP baseline.
   Corrected during the L16 build (2026-07-13): at this scale the two nets TIE on
   accuracy (~86%), so the honest payoff is "equal accuracy, 12x fewer parameters" -
   parameter efficiency, not an accuracy jump. The accuracy gap appears on richer
   images (L17's story).
7. **The pomegranate-sorting running example** (approved 2026-07-13), per the style
   guide's one-scenario-per-chapter convention: L16 cold open + kernel-zoo photo, L17
   which-architecture frame, L18 cold open + task-zoo figure + closing payoff frame.
   Doubles as the Armenian local flavor; the optional khachkar/landmark transfer set
   stays an open question.
8. **ImageNet error-over-years chart** (own matplotlib figure, labeled bars).
9. **L18 practice frames + the 4-deck split** (2026-07-13): Grad-CAM ("where is the net
   looking?") with the wolf-vs-husky Clever Hans story, the BatchNorm fine-tuning
   gotcha, and the labeling-economics frame - all in the slimmed L18; detection and
   segmentation got their own L19, which now also carries the chapter close.
10. **L17 history + 2026 epilogue** (added 2026-07-13, instructor request): the LeCun arc
   (Bell Labs -> NCR check reading -> Turing Award -> AMI Labs world-models bet), the
   Fei-Fei Li data bet (WordNet, Mechanical Turk, CVPR 2009), AlexNet's two-GTX-580
   story, Karen Simonyan on the VGG frame (the Armenian touch: VGG -> DeepMind ->
   Inflection -> Microsoft AI), and a "vision in 2026" close (ViT, CLIP/VLMs, diffusion,
   SAM) ending on the honest "CNNs still run the edge" counterpoint. All facts
   web-verified 2026-07-13; the exact claims live in the L17 build notes.

## Examples ledger (matching the reference decks' example density)

The LMU decks teach through a steady stream of concrete examples (Sobel on an Einstein
photo, the CIFAR-10 frog, Tesla autopilot, Covid CT scans, U-Net road segmentation, a
char-CNN on a Yelp review of LMU itself). We keep that density but swap in our own
anchors:

| Example | Role | Deck | Mechanism |
|---|---|---|---|
| Pomegranate sorting line | chapter running example (cold opens, kernel zoo, which-architecture, task zoo, closing payoff) | all | REGEN-FIG (own photo) |
| "CNNs in the wild" montage: self-driving, medical CT, segmentation, colorization | motivation frame mirroring LMU cnn1-intro's application tour | L16 | COPY-IMG (3-4 images, attributed) |
| Sobel edge detection, dissected | the hand-designed-filter era | L16 | REGEN-FIG (our photo, not Einstein) |
| Fashion-MNIST | training dataset for all L16 figures + the ch5 MLP-baseline callback | L16 + HW1B | REGEN-FIG |
| ImageNet story + error-by-year chart | why architecture design matters | L17 | REGEN-FIG (chart) + REDERIVE (story) |
| Pretrained conv1 filters (resnet18) | learned-kernels payoff, closes the HW1 arc | L17 (recalled in L18) | REGEN-FIG |
| CIFAR-10 | HW2 architecture-comparison dataset | L17 HW | notebook |
| Oxford-IIIT Pets | HW3 fine-tune dataset (khachkar set = stretch) | L18 HW | notebook |
| Grad-CAM heatmap + the wolf-vs-husky snow detector (Clever Hans) | trust / interpretability frames | L18 | REGEN-FIG (gradcam.py) + REDERIVE |
| Pretrained YOLO on students' own photos | HW4 detection practical | L19 HW | notebook |
| HAR sensor data / char-level text (1D), MRI + video (3D) | "convolution is a prior, not an image trick" name-drops | L16, L19 | REDERIVE, no figure |
| U-Net road/satellite segmentation | segmentation showcase | L19 | COPY-IMG (LMU's own result image) or paper citation |

Standing preference: Armenian examples supplement the canonical Western ones, never
replace them.

## Implementation notes for the authoring model

This plan will be implemented by another model (Sonnet/Opus) and reviewed afterwards.
Non-negotiables, in build order:

1. **Read first:** `CLAUDE.md`, `WORKFLOWS.md` ("Create a new slide deck" row),
   `ml/SLIDE_STYLE.md` (single source of truth), this plan, then the deck's outline.
   The interview and outline stages are done and approved; you are at the build stage.
   Each outline ends with deck-specific build notes - follow them.
2. **File layout:** decks are `ml/ch6_cnn/L16_cnn_foundations.tex` (etc.),
   `\documentclass[aspectratio=169]{beamer}` + `\input{../preamble}` (ch6_cnn sits one
   level under `ml/`, exactly like ch5). Figure scripts in `ml/ch6_cnn/py_src/`, output
   PDFs in `ml/ch6_cnn/fig/`, borrowed images in `ml/ch6_cnn/fig/borrowed/`. Chapter
   page `ml/ch6_cnn/cnn.qmd`, registered in `_quarto.yml` with exact-case path.
3. **Outline tags translate as:** `[plain]` = plain transition frame (popblue bold title
   + one motivation line); `[predict-first]` = question, `\pause`, reveal on the same
   frame; `[worked-numbers]` = compute with the exact numbers given in the outline (do
   not invent different ones); callout boxes per the SLIDE_STYLE palette table;
   `[running example]` = the pomegranate thread. Every deck ends with a recap frame, a
   paramgreen "Next:" box, and a `% Provenance:` comment block (sources consulted,
   figures used, key decisions, forward pointer).
4. **Figure scripts:** run with `./ma/Scripts/python.exe` (never bare python, never uv
   ephemeral envs); use `logging` to console + `logs/` (create the dir first), never
   print; seed 509; fail loudly - no silent fallbacks, missing inputs must raise;
   labeled bars via `ax.bar_label`; Armenian-flag palette (#D90012, #0033A0, #F2A800)
   when a chart needs 3+ colors. Keep local runs CPU-light: Fashion-MNIST subsampled
   (<= 8k train / 2k test), tiny nets, <= 5 epochs, minutes not hours. Anything heavier
   trains in the HW notebooks on Colab; only its saved metrics get plotted locally.
5. **Compile loop per deck:** `pdflatex -interaction=nonstopmode -halt-on-error` twice
   -> zero `!` lines in the log -> `./ma/Scripts/python.exe clean_latex.py ml/ch6_cnn`
   -> visual overflow check (`beamer-overflow-check`; Beamer clips silently) -> open
   the PDF for instructor review.
6. **Prose rules:** no em-dashes (use -), no curly quotes; English slide body.
7. **Build one deck at a time** (L16 -> L17 -> L18 -> L19), each compiled,
   overflow-checked and approved before starting the next.

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
- **HW3 (L18) - transfer learning + a Grad-CAM audit.** Part A: fine-tune pretrained
  ResNet on Oxford-IIIT Pets (default; Armenian set only if cheap to assemble). Part B:
  Grad-CAM heatmaps on 5 correct + 5 wrong predictions of the fine-tuned model, with a
  short written read of what it attends to.
- **HW4 (L19) - detection hands-on + the dilation experiment.** Part A: run a pretrained
  YOLO (ultralytics, Colab) on your own photos; inspect boxes and confidences; vary the
  NMS IoU threshold and report what appears/disappears. Part B (moved from the old HW3):
  swap a dilated conv into the small L16 CNN and observe the receptive-field / accuracy
  effect.

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
| `receptive_field.py` | receptive-field growth: stacked 3x3 vs one 7x7; dilated variant | L17, L19 |
| `timeline_ribbon.py` | 1989-2026 timeline strip, one variant per L17 section transition ("you are here") | L17 |
| `cifar_grid.py` | CIFAR-10 class grid from keras.datasets (HW2 bridge) | L17 |
| ~~`degradation.py`~~ | superseded 2026-07-13: use He et al. 2015 Fig. 1 itself (WEB-IMG) | L17 |
| `transfer_curves.py` | fine-tune vs from-scratch on small data | L18 |
| `gradcam.py` | Grad-CAM heatmap over the pomegranate photo (pretrained resnet18, CPU-light) | L18 |
| `task_zoo.py` | sorting-line photo, four task overlays (classify/localize/detect/segment) | L19 |
| `iou_nms.py` | IoU worked-example boxes + NMS before/after panels (hard-coded, no model) | L19 |
| `nms_anim.py` | ANIM flip-book: overlapping detections pruned by NMS, click by click | L19 |
| `chapter_arc.py` | four-beat chapter strip (L16 layer -> L17 architectures -> L18 reuse -> L19 tasks) for the arc-recap frame | L19 |

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
4. ~~Armenian dataset for HW3~~ Resolved 2026-07-13 (L18 interview): **Oxford Pets**
   confirmed; the khachkar/landmark set stays a stretch goal only.
5. **The pomegranate photo:** thread approved (2026-07-13); still need the actual image.
   Your own photo avoids licensing; otherwise a permissive-license stock shot. For the
   L18 task-zoo figure a shot with several fruit visible works best.
6. ~~3 decks confirmed, or compress?~~ Resolved 2026-07-13: expanded to **4 decks**
   (L19 split out of L18; L16 unaffected). Calendar cost acknowledged: one extra
   session and a 4th homework.
7. ~~Multi-fruit photo for L19's task_zoo figure~~ Resolved 2026-07-13 (L19 interview):
   **web-download** a multi-pomegranate photo (sorting line / market stall) into
   `fig/borrowed/` under the WEB-IMG policy.
