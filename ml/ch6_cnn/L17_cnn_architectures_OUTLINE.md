# L17 - CNN Architectures: the ImageNet story - outline (for approval)

Drafted 2026-07-13; history + 2026-epilogue frames added same day (instructor request,
facts web-verified - see build notes). Source: LMU `cnn3/{modern-cnn-1, modern-cnn-2}` +
`cnn1/architecture` + own material (ImageNet-error chart, ResNet-boosting callback,
degradation predict-first, the LeCun / Fei-Fei Li stories, the vision-in-2026 close).
Subtitle idea: "Ten years that changed computer vision."

**Interview decisions (2026-07-13, instructor):**
- Story/depth balance stays ~60% technical / 40% story; keep all worked numbers AND all
  history frames.
- Wider zoo EXPANDED: 1x1 convs, Inception block, GoogLeNet full architecture, and GAP
  (2 frames) each get real treatment; DenseNet / EfficientNet stay one-liners.
- Epilogue stays 4 frames + a new "one paper, three fates" AlexNet-authors frame
  (placed in the epilogue, not after AlexNet).
- Clickable resource link-buttons sprinkled throughout (3-5; list in build notes).
- HW2 unchanged (CIFAR-10, small CNN vs ResNet-18-scale).
- (Round 3) Worked numbers run as a MIX: counter-intuitive ones predict-first, plus two
  "take a minute, compute it" moments (VGG count, GAP count).
- (Round 3) ResNet-boosting lands as a predict-first RECALL pause, not a statement.
- (Round 3) Timeline ribbon on every section-transition slide (no cheat-sheet table, no
  end mini-quiz).
- (Round 3) Illustrations: real dataset samples (ImageNet montage, CIFAR-10 grid) and
  ICONIC ORIGINAL figures (AlexNet two-GPU diagram, He et al.'s actual degradation
  curves) - explicitly NO portraits of the people.

Target: ~27 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - one chart, one question.** ImageNet top-5 error by year, 2010-2017, bars
  labeled with the winning system. "Something happened in 2012. Predict: what?" `\pause`
  Reveal: AlexNet - the first deep CNN winner - cut error from 26% to 16% in one year;
  by 2015 machines passed the human baseline (~5%). Everything in this lecture is on that
  chart. `[predict-first]` `[real fig: imagenet_error.pdf]`

### Outline frame

### Section 1: The benchmark that made deep learning

- `[plain]` transition: "ImageNet" + one motivation line.
- **The bet on data (the Fei-Fei Li story).** 2006: while most of the field tuned
  algorithms, Fei-Fei Li bet that what vision lacked was DATA. Built on WordNet's noun
  hierarchy (with WordNet's creator, Christiane Fellbaum); images scraped from the web,
  labeled by Amazon Mechanical Turk workers; presented at CVPR 2009 to little fanfare:
  ~14M labeled images, ~22k categories. Lesson that outlives every architecture on this
  deck: data bets age better than algorithm bets. `[story frame]`
- **ILSVRC + top-5 error.** The challenge subset: 1.2M training images, 1000 classes;
  top-5 error defined (fair scoring for ambiguous images). Why a shared benchmark
  accelerates a field - same reason we always compare on held-out data ([06]).
  `[callback: [06] evaluation]` `[WEB-IMG or montage: real ImageNet sample photos]`

### Section 2: The classics - LeNet, AlexNet, VGG

- `[plain]` transition: "Three architectures, three ideas."
- **LeNet (1998): the blueprint - and the LeCun story.** Yann LeCun, AT&T Bell Labs
  (joined 1988): backprop-trained convolutional nets reading handwritten digits.
  conv-pool-conv-pool-dense - everything in L16 was already here. Deployed commercially
  by NCR in check-reading machines; at its peak the system read an estimated 10-20% of
  all checks written in the US. Then the field moved on to SVMs and CNNs went quiet for
  a decade: what was missing was compute and data. Plant the hook: "keep an eye on
  LeCun - he returns at the end of this lecture." `[WEB-IMG: LeNet architecture diagram]`
  `[story frame]` `[link button: Adam Harley's interactive 3D CNN visualization + CNN
  Explainer - "play with a LeNet yourself"]`
- **AlexNet (2012): the breakout.** Same blueprint, scaled: 8 layers, 60M parameters -
  too big for one GPU, so Krizhevsky SPLIT the network across two consumer gaming cards
  (GTX 580, 3GB each) and trained 5-6 days (~90 epochs over 1.2M images) on a single PC.
  The GPU was the unlock: L14's matrix-multiply argument, made history. The "new" tricks
  are all L15 material deployed at once: ReLU (vs saturating sigmoids), dropout, data
  augmentation - "you already know every ingredient; they had the hardware to use them
  at scale." Authors: Krizhevsky, Sutskever, Hinton.
  `[callback: L15 dropout, augmentation, activations; L14 GPU frame]` `[story frame]`
  `[WEB-IMG: the paper's original two-GPU architecture diagram]`
- **What AlexNet learned (the L16 arc, paid off).** Conv1 filters of a pretrained network:
  crisp Gabor-like edge, blob and color detectors - the L16 kernel zoo, reinvented by
  gradient descent, designed by nobody. L16's tiny Fashion-MNIST kernels looked blurry; at
  ImageNet scale the learned kernels become textbook. Closes the hand-designed -> learned
  arc. `[real fig: pretrained_filters.pdf]` `[callback: L16 kernel zoo + punchline frame]`
- **VGG (2014): depth via small filters.** Only 3x3 convs, stacked. Worked count: two
  stacked 3x3 layers see a 5x5 receptive field with 2 x 9C^2 weights vs 25C^2 for one
  5x5 - plug in C = 64: 73,728 vs 102,400 (biases ignored) - deeper AND cheaper, with an
  extra nonlinearity for free. `[worked-numbers]` `[compute-it: students do the C=64
  count themselves, ~1 min, then reveal]`
- **Who built VGG (the Armenian touch).** Karen Simonyan and Andrew Zisserman, Oxford's
  Visual Geometry Group - and yes, Karen Simonyan is an Armenian name. His arc since:
  DeepMind 2014-2022 (contributor to AlphaZero, AlphaFold, WaveNet), co-founder of
  Inflection AI (2022, with Hoffman and Suleyman), now Chief Scientist at Microsoft AI.
  One frame, light touch - a VGG author's career IS the last decade of AI in miniature.
  `[story frame]` `[Armenian touch]`
- **Receptive field.** How far one deep neuron "sees" into the input; grows with depth.
  Sets up dilated convolutions (L18). `[real fig: receptive_field.pdf]`

### Section 3: Going deeper - degradation and ResNet

- `[plain]` transition: "If depth is good, why not 100 layers?"
- **Predict-first: the degradation problem.** "A 56-layer plain net vs a 20-layer plain net,
  same recipe. Which has lower TRAINING error?" `\pause` Reveal: the 20-layer one. Not
  overfitting (training error!) - deep plain nets are just hard to optimize.
  `[predict-first]` `[WEB-IMG: He et al. 2015 Figure 1 - the ACTUAL curves (round-3
  decision: iconic originals over re-creations; degradation.py superseded)]`
- **The residual idea.** Let a block learn the correction, not the whole mapping:
  H(x) = F(x) + x; the skip connection carries x for free, the layers learn the residual
  F(x). Learning "do nothing" is now trivial (push F to 0) - so extra layers can no longer
  hurt. Dimension-matching projection W_s x in one line. `[boxed math, short]`
  `[ANIM candidate: residual-block build-up]` `[WEB-IMG: ResNet block diagram]`
- **You have seen residual learning before (recall pause).** Leave H(x) = F(x) + x on
  screen and ask: "where in this course have you already met a model that only fits a
  CORRECTION to what already exists?" `\pause` Reveal: gradient boosting (ch4) - every
  new tree fits the residual of the ensemble so far. A ResNet block does the same thing
  in depth: boosting stacks residual learners in sequence, ResNet stacks them in depth.
  Caveat line ON the frame: analogy, not identity - boosting fits each learner to the
  FROZEN ensemble's residual, stage by stage; ResNet blocks train jointly, end to end.
  `[predict-first recall]` `[armblue key box]` `[callback: ch4 boosting]`
  `[armorange one-liner: limits of the analogy]`
- **Why skips also fix gradients.** The identity path gives the gradient an undamped route
  backwards - the vanishing-gradient problem from L15, relieved by wiring rather than by
  activation choice. Reference only, no re-derivation. `[callback: L15 vanishing gradients]`
- **The modern conv block.** conv -> BatchNorm -> ReLU, repeated, plus skips. BN is the
  L15 formula, unchanged - one callback line, no re-teach. `[callback: L15 BatchNorm frame]`

### Section 4: The wider zoo (expanded per interview)

- `[plain]` transition: "Other tricks that stuck - and one network built entirely from
  them."
- **1x1 convolutions.** A 1x1 conv = a per-pixel dense layer across channels: it mixes
  depth without touching space. Two jobs: cheap channel-count reduction before expensive
  convs, and extra nonlinearity for free. Worked mini-count: 1x1 from 256 -> 64 channels
  = 256*64 + 64 = 16,448 params vs 3x3 doing the same = 147,520. Forward pointer:
  pointwise convs return in L19 (separable). `[worked-numbers]`
- **The Inception block.** Why choose a kernel size at all? Run 1x1, 3x3, 5x5 and a pool
  path IN PARALLEL and concatenate the maps - let the network pick per layer. The 1x1
  bottlenecks keep it affordable. `[WEB-IMG: Inception block diagram]`
- **GoogLeNet: a network of Inception blocks.** The full 2014 winner: ~22 layers of
  stacked Inception blocks, only ~5M params (vs AlexNet's 60M) - the zoo tricks
  compound. One architecture tour on the full diagram. `[WEB-IMG: GoogLeNet full
  architecture diagram]` `[story beat: won ILSVRC 2014 at 6.7%]`
- **The dense-head problem.** Where do classic CNNs spend their parameters? The
  flatten -> dense head. Worked count: 256 maps of 100x100 -> dense-10 head = 25.6M
  weights - more than all conv layers combined. `[worked-numbers]` `[predict-first:
  "where are most of AlexNet's 60M params?" -> the dense head]`
- **Global average pooling.** The fix (NiN idea): average each final map to ONE number ->
  a 256-vector -> dense-10 = 2,560 weights, a 10,000x cut. Bonus: acts as a regularizer
  and accepts any input size. Used by everything modern - ResNet already ends this way.
  `[worked-numbers]` `[compute-it: students compute the GAP-head count, quick]`
  `[callback: the L16 pooling frame - same op, global window]`
- **Scaling and searching.** EfficientNet compound scaling in one line; architectures are
  now found by search as much as designed by hand. DenseNet name-check (concatenate, don't
  add). One-liners only - pointer to the LMU/d2l material for the curious.

### Section 5: Using architectures in practice

- `[plain]` transition: "Nobody types out a ResNet."
- **torchvision.models.** The canonical snippet: load `resnet18`, print parameter count,
  swap the head for your classes. ~6 lines. `[minimal code]`
- **Which architecture when (the sorting line decides).** Back to the pomegranate line:
  a Fashion-MNIST-sized toy -> the small custom CNN from L16; the real sorting camera ->
  start from a pretrained ResNet-family model - which is exactly transfer learning, next
  lecture. `[paramgreen takeaway box]` `[running example]`

### Section 6: Epilogue - the pioneers and the field, 2026

- `[plain]` transition: "Where did everyone go?" + one motivation line.
- **The pioneers today.** 2018 Turing Award to Hinton, LeCun and Bengio "for deep
  learning". And the LeCun hook from the LeNet frame resolves: after ~12 years as Meta's
  chief AI scientist he left (Nov 2025) and founded AMI Labs (Paris), raising about $1B
  within months - to bet AGAINST large language models and on "world models" (JEPA):
  systems that learn how the physical world works by predicting in representation space,
  not in pixels or tokens. The man who made CNNs work thinks the next leap is not a
  bigger chatbot. Discussion beat: the field's pioneers still disagree about its future -
  that is healthy, not alarming. `[story frame]` `[source line on frame: CNBC Nov 2025 /
  MIT Technology Review Jan 2026]`
- **One paper, three fates.** The three names on the AlexNet paper, thirteen years on:
  Geoffrey Hinton - 2018 Turing Award, then the 2024 Nobel Prize in Physics (with
  Hopfield), now the field's loudest safety voice; Ilya Sutskever - co-founded OpenAI,
  chief scientist through the GPT era, left in 2024 to found Safe Superintelligence
  (SSI); Alex Krizhevsky - the one who actually trained it, quietly stepped away from
  the field. One paper, three completely different lives. `[story frame]`
  `[facts: verify per build notes]`
- **Vision in 2026 (I): understanding.** Two shifts on one frame. 1) Attention ate the
  frontier: Vision Transformers treat an image as a sequence of patches - no convolution
  at all (L19 closes this thought against CNN inductive bias). 2) Image-text pretraining
  (CLIP, 2021) fused vision with language: today's assistants (GPT, Gemini, Claude) SEE -
  vision-language models answer questions about your photo. Classification stopped being
  a standalone task and became a capability inside larger systems.
  `[link button: a VLM chat - "show it a photo, ask it questions"]`
- **Vision in 2026 (II): generation.** Diffusion models flipped the task: classification
  asks "what is this?", diffusion answers "draw me one" (Stable Diffusion, DALL-E, Flux;
  video followed with Sora and Veo). Plus foundation models for the classic tasks: SAM
  segments anything, promptable, no task-specific training. One intuition sentence on
  diffusion (learn to undo noise, step by step) - the mechanics belong to the GenAI
  chapter. `[paramgreen pointer: GenAI chapter]`
  `[link button: SAM demo page - "click an object, get its mask"]`
- **And yet: CNNs did not die.** The honest close: on phones, cars, cameras and sorting
  lines (our pomegranate belt), CNNs and CNN hybrids remain the workhorse - cheap, fast,
  and the right inductive bias at the edge; ResNets are still the default baseline in
  papers. You learned the layer that still does most of the world's looking.
  `[running example]` `[paramgreen takeaway box]`

### Recap + Next

- Recap: benchmark -> progress; LeNet blueprint; AlexNet = L15 tricks at scale; VGG = small
  filters, deep; degradation -> residual learning (= boosting in depth); BN blocks; GAP;
  load pretrained, don't retype; the arc runs Bell Labs checks -> ImageNet -> ViT/VLMs/
  diffusion - and CNNs still run the edge.
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

## Build notes (for the implementing model)

- Consult read-only (REDERIVE):
  `ml/deep_learning/moodle_s26_course/slides_tex/slides/cnn3/{slides-modern-cnn-1,
  slides-modern-cnn-2}.tex`, `cnn1/slides-cnn-architecture.tex`.
- **Architecture diagrams = WEB-IMG** (instructor direction 2026-07-13, supersedes the
  earlier "redraw, don't copy" note): download good LeNet / AlexNet / VGG / Inception /
  ResNet / GAP diagrams into `fig/borrowed/` (d2l.ai's are fine); no licensing caveats
  or per-image credit lines. TikZ only for small block strips.
- **Visual style (match the built L16):** at least one ANIM flip-book in this deck.
  Candidates, in order of preference: (a) `imagenet_anim_*.pdf` - the error chart
  revealed year by year, one bar per click (one script emits all frames; the static
  `imagenet_error.pdf` stays as the final frame); (b) residual-block build-up: plain
  block -> degradation -> add the skip -> gradient flows through the identity (3-4
  frames); (c) receptive-field growth layer by layer. Pick at least (a) or (b); `\pause`
  builds elsewhere where a frame reveals stepwise.
- **Credit line** on the title/Outline frame, same wording as L16.
- **COPY-SLIDE option** (recipe in the L16 build notes): compiled sources are
  `slides/week24_cnns_modern/04_modern-cnn-1.pdf` and `05_modern-cnn-2.pdf`. Prefer
  REDERIVE throughout this deck; embed a page only if a diagram recreation stalls.
- `imagenet_error.py` data (ILSVRC top-5 error, winners): 2010 NEC 28.2, 2011 XRCE 25.8,
  2012 AlexNet 16.4, 2013 Clarifai/ZFNet 11.7, 2014 GoogLeNet 6.7 (VGG 7.3 runner-up),
  2015 ResNet 3.57; human baseline ~5.1. VERIFY these against Russakovsky et al. 2015 /
  the original papers before hard-coding; cite the source on the frame.
- `degradation.py`: schematic re-creation of He et al. 2015 Figure 1 (training error,
  20-layer curve below 56-layer); label it "re-created after He et al. 2015" - do not
  copy the paper figure itself.
- `pretrained_filters.py`: torchvision `resnet18` with pretrained weights (one-time
  ~45MB download); plot all 64 conv1 filters (7x7x3) as an 8x8 RGB grid, per-filter
  min-max normalized.
- Worked numbers are fixed in the outline: VGG C=64 count (73,728 vs 102,400); GAP head
  (256 maps of 100x100 -> dense-10 = 25.6M weights vs 2,560 with GAP).
- **Verified history facts (web-checked 2026-07-13; use these, do not re-invent):**
  - LeCun: joined AT&T Bell Labs 1988; LeNet check-reading deployed by NCR, at its peak
    read an estimated 10-20% of US checks (LeCun's own account - keep "an estimated").
    Turing Award 2018 with Hinton and Bengio. Left Meta Nov 2025 after ~12 years as
    chief AI scientist; co-founded AMI Labs (Advanced Machine Intelligence Labs, Paris,
    with Alexandre LeBrun); ~$1.03B raised by early 2026; thesis = world models / JEPA
    over LLMs. Sources for the frame: CNBC (2025-11), MIT Technology Review (2026-01).
    Dollar figures age fast - re-verify if built months from now.
  - ImageNet: started 2006 by Fei-Fei Li; built on WordNet with Christiane Fellbaum;
    labeled via Amazon Mechanical Turk; presented CVPR 2009; ~14M images / ~22k
    categories; ILSVRC subset 1.2M / 1000 classes.
  - AlexNet: 60M params; did not fit one GPU -> split across two GTX 580s (3GB each);
    5-6 days, ~90 epochs on 1.2M images; Krizhevsky, Sutskever, Hinton (2012) - cite the
    paper itself.
  - Karen Simonyan: VGG with Zisserman at Oxford (Visual Geometry Group); DeepMind
    2014-2022 (AlphaZero, AlphaFold, WaveNet contributions); co-founded Inflection AI
    2022 (Hoffman, Suleyman); now Chief Scientist, Microsoft AI. Keep the Armenian
    connection at the level of the name ("Karen Simonyan is an Armenian name") - do not
    invent birthplace details.
  - "One paper, three fates" frame - stable parts: Hinton won the 2018 Turing Award and
    the 2024 Nobel Prize in Physics (shared with Hopfield); Sutskever co-founded OpenAI,
    was chief scientist, left May 2024 and founded Safe Superintelligence (SSI).
    VERIFY before building: the Krizhevsky beat (left Google ~2017, later at Dessa,
    stepped away from research) - phrase softly ("quietly stepped away from the field")
    and web-check it; also re-check whether SSI has shipped/changed by build time.
  - GoogLeNet numbers: ~22 layers, won ILSVRC 2014 at 6.7%; the "~5M params vs
    AlexNet's 60M" claim is commonly cited - VERIFY the exact figure (sources vary
    5M-6.8M) or say "under 7M".
- **Link buttons** (style: `\href{...}{\beamergotobutton{...}}`, one per frame max;
  verify each URL resolves at build time):
  - LeNet frame: Adam Harley's 3D CNN visualization (adamharley.com/nn_vis - the old
    scs.ryerson.ca URL redirects/died, check) + CNN Explainer
    (poloclub.github.io/cnn-explainer).
  - torchvision practice frame: pytorch.org/vision models docs.
  - Epilogue (I): any public VLM chat the instructor prefers (chatgpt.com / gemini /
    claude.ai) - pick one, do not list all.
  - Epilogue (II): SAM demo (segment-anything.com/demo).
- **WEB-IMG downloads for the expanded zoo** (into `fig/borrowed/`, no credit lines):
  Inception block diagram, GoogLeNet full-architecture diagram, LeNet architecture
  diagram, ResNet block diagram. d2l.ai / paper figures are fine sources.
- 1x1 worked mini-count is fixed in the outline: 256->64 via 1x1 = 256*64 + 64 = 16,448
  vs via 3x3 = 9*256*64 + 64 = 147,520.
- **Timeline ribbon (round-3 decision, REQUIRED):** every `[plain]` section-transition
  slide carries a thin horizontal 1989-2026 timeline strip with the current story
  position highlighted (1989 LeNet -> 2012 AlexNet -> 2014 VGG/GoogLeNet -> 2015 ResNet
  -> 2017 transformer -> 2020 ViT -> 2021 CLIP -> 2022 Stable Diffusion -> 2023 SAM ->
  2025 world-model bet). One script `timeline_ribbon.py` emits one variant per section
  (`ribbon_1.pdf` ... `ribbon_N.pdf`). This absorbs the old optional vision_timeline
  idea - no separate timeline figure needed.
- **Dataset-sample figures (round-3 decision):** real ImageNet sample photos on the
  ILSVRC frame (WEB-IMG montage or a grid assembled from a handful of downloaded
  samples); `cifar_grid.py` - a CIFAR-10 class grid generated locally from
  `keras.datasets.cifar10` for the HW2 bridge (recap/Next area or the qmd).
- **Iconic originals (round-3 decision):** `degradation.py` is SUPERSEDED - use He et
  al. 2015 Figure 1 itself (WEB-IMG), and the AlexNet paper's two-GPU diagram on the
  AlexNet frame. NO portraits of the people (explicitly decided against).
