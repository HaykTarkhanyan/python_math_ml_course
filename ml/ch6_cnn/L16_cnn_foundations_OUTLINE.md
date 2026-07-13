# L16 - CNN Foundations - outline (for approval)

Drafted 2026-07-13. Source: LMU `cnn1/{conv2d, properties, components, pooling,
architecture}` + own material (callback spine, kernel-zoo arc, Fashion-MNIST payoff).
Subtitle idea: "One neuron, reused at every pixel."

Target: ~22 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - the sorting line (chapter running example).** A pomegranate-sorting line: a
  camera photographs each fruit, a model grades it (export / market / juice). This scenario
  runs through all three lectures. "In L14 we counted 109,386 weights for MNIST
  (784-128-64-10). The camera's photo is 224x224x3 = 150,528 inputs. Predict: how many
  weights does ONE dense layer with 1,000 hidden units need?" `\pause` Reveal: ~150
  million - for one layer, before learning anything. Hook: "today's fix costs a few
  hundred." `[predict-first]` `[callback: L14 param-count frame]` `[running example]`

### Outline frame

### Section 1: Why dense nets are the wrong tool for images

- `[plain]` transition: "Images are not feature vectors" + one motivation line.
- **An image is a grid of numbers.** Grayscale = one matrix, entries 0 (black) to 255
  (white); zoom into a patch of the pomegranate photo and show the raw numbers. Color = 3
  such grids stacked (R, G, B) - teaser only, the full channel treatment waits for
  section 4. Sections 2-3 work in grayscale so the mechanics stay 2D.
  `[real fig: photo + zoomed number patch]`
- **An MLP cannot see structure.** Flattening throws away geometry: show the photo and the
  same photo with pixels shuffled by a fixed permutation - to an MLP these are equally
  learnable, to us one is noise. Nearby pixels are related; far ones mostly are not.
  `[real fig: photo vs pixel-shuffled photo]`
- **Two wishes.** 1) Locality: detect small patterns from small neighborhoods.
  2) Reuse: an edge detector useful in the corner is useful in the center - "a cat in the
  corner is still a cat". These two wishes ARE the convolutional layer. `[armblue key box]`

### Section 2: The convolution operation

- `[plain]` transition: "Sliding a small window with a dot product."
- **1D first.** A length-3 kernel sliding over a sequence; each output = dot product with
  the window. Moving average as the familiar special case. (1D-before-2D per pedagogy
  research; also plants "convolutions work on time series too" for L18.)
  `[TikZ small strip diagram]`
- **2D by hand.** 2x2 kernel on a 3x3 input, all four output cells computed with real
  numbers, cell by cell. `[worked-numbers]` (adapted from LMU cnn1/conv2d)
- **The kernel zoo.** The pomegranate photo (grayscale), six classic kernels. For 2-3 of them:
  show the kernel numbers first, "predict what this does to the image" `\pause`, reveal the
  filtered photo. Identity, box blur, sharpen, Sobel X, Sobel Y, emboss.
  `[predict-first]` `[real fig: kernel_zoo.pdf]`
- **Sobel, briefly dissected.** Why those numbers detect vertical edges: a differentiation
  row smoothed by an averaging column. Filters like these ran computer vision for 30 years -
  hand-designed by experts. (Sets up the punchline and HW1 Part A.)
- **The punchline: a CNN designs its own kernels.** Kernel entries are just weights, tuned
  by gradient descent like every weight since L02. We stop hand-designing feature detectors
  and let the loss choose them. Misconception pre-empt: "CNN kernels are NOT
  hand-engineered - they are learned. That is the whole point." `[armblue key box]`
  `[callback: L01g feature engineering, L14 representation learning]`
- **A kernel is a neuron you already know.** 3x3 kernel = 9 weights + bias, affine +
  activation - exactly the L14 neuron, just connected to a patch instead of everything, and
  reused at every location. Nothing about training changes. `[callback: L14 single neuron]`
- **Footnote frame (short): convolution vs cross-correlation.** True convolution flips the
  kernel; deep-learning frameworks implement cross-correlation and call it convolution.
  Irrelevant in practice because the kernel is learned either way. One slide, no Fourier.

### Section 3: Why convolution is the right prior

- `[plain]` transition: "Counting connections and weights."
- **Sparse connectivity.** 3x3 input, 2x2 kernel -> 2x2 output: 16 connections. The dense
  equivalent: 36. Receptive field idea named here. `[TikZ two small connection diagrams]`
- **Parameter sharing.** Those 16 connections use only 4 distinct weights (the dense net:
  36). Scale it up: 100x100x3 image, one 5x5 filter with same padding = 75 parameters; a
  dense layer with the same output size = 300,000,000. `[worked-numbers]`
  `[real fig: param_explosion.pdf, labeled bars]`
- **Regularization by architecture.** Fewer parameters + the locality assumption = a
  restricted hypothesis space that matches image structure. Weight decay (L15) shrinks
  weights; convolution removes them entirely - the strongest regularizer is the right
  inductive bias. Callback to "trees win on tabular" (L14/ch4): inductive bias is also why
  CNNs win on images. `[callback: L15 weight decay, ch4/L14 tabular frame]`
  `[armblue key box]`
- **Myth-buster: translation invariance.** "A CNN gives the same output wherever the cat
  sits - true?" `\pause` Reveal: false as stated. Convolution is translation-EQUIVARIANT
  (shift input -> output shifts along); pooling and downsampling buy only limited
  invariance. `[predict-first]` `[armorange watch-out box]`

### Section 4: The components of a conv layer

- `[plain]` transition: "The four knobs: channels, padding, stride, pooling."
- **Channels, done properly.** Color returns: RGB input = depth-3 tensor. ONE filter spans
  the FULL input depth and sums across channels - it outputs ONE feature map, not three.
  Misconception pre-empt: "3 channels in does not mean 3 maps out." Tiny worked example:
  2x2x2 input patch, 2x2x2 filter, one output number with the cross-channel sum written
  out. N filters -> N maps -> the next layer's depth-N input. `[worked-numbers]`
  `[TikZ stacked-maps sketch]`
- **The conv-layer parameter formula.** Boxed: params = k*k*C_in*C_out + C_out (one bias
  per filter). Two cases by hand: 3x3 conv from RGB to 16 maps = 3*3*3*16 + 16 = 448; the
  section-3 5x5 filter on RGB = 76 with its bias. This formula plus the output-size formula
  are the two tools for reading ANY architecture - used throughout L17 and HW2.
  `[worked-numbers]` `[boxed formula]`
- **Padding.** Valid vs same; without padding the map shrinks every layer and depth is
  capped; with zero-padding depth is free. `[real fig: conv_arithmetic.pdf panel]`
- **Stride + the output-size formula.** o = floor((i - k + 2p)/s) + 1. Compute 2-3 cases by
  hand (the classic exam question). `[worked-numbers]`
- **Pooling.** Max vs average on a 4x4 toy grid, both computed. What pooling buys:
  downsampling, fewer params downstream, a little invariance. Max keeps the strongest
  response; average keeps everything (and blurs). `[worked-numbers]` (adapted from LMU)

### Section 5: Your first CNN

- `[plain]` transition: "Assemble the blocks."
- **(Contingency) The training loop in 60 seconds.** forward -> loss -> zero_grad ->
  backward -> step, one recap frame. Include ONLY if L15 is not fresh at delivery time
  (per CNN_BLOCK_DESIGN's prerequisite warning); drop it if the block runs right after
  L15. `[callback: L15 training loop]`
- **Anatomy of a conv stack.** [conv -> ReLU -> pool] x2 -> flatten -> dense -> softmax
  (LeNet-style). Track the tensor through every layer for a 28x28 input - one table with a
  SHAPE column and a PARAMS column (both section-4 formulas); the total lands near ~9k -
  hold that number for the payoff frame. Note on the frame: conv is affine, so without the
  ReLU between them stacked convs collapse into one big conv (L14's linearity lesson); and
  pooling has ZERO parameters. `[worked-numbers]` `[callback: L14 why-nonlinearity]`
  `[TikZ or real fig]`
- **What the layers learn.** First-layer kernels of a small CNN trained on Fashion-MNIST:
  edge/texture detectors nobody designed. Deeper layers: parts, then objects. This is L14's
  XOR lesson at scale - the network re-coordinatises pixels until a linear head suffices.
  `[real fig: feature_maps.pdf]` `[callback: L14 XOR re-coordinatisation]`
- **The canonical snippet.** `nn.Conv2d` / `nn.MaxPool2d` / `nn.Linear` - the section-5
  architecture in ~8 lines. Training loop unchanged from L15; autograd differentiates conv
  like anything else (the L15 "any architecture" promise, kept). `[minimal code]`
  `[callback: L15 training loop + comp-graph frames]`
- **Why this is fast.** Convolution unrolls into big matrix multiplies - exactly the shape
  GPUs eat (L14's GPU frame). Same reason a 60M-param CNN trains overnight.
  `[callback: L14 why-GPUs frame]`
- **Payoff: CNN vs MLP, same data.** Fashion-MNIST, same subsample as the ch5 practical:
  the L15 MLP baseline vs this small CNN - accuracy jump with FEWER parameters. The
  architecture, not capacity, did the work. `[real fig: cnn_vs_mlp.pdf]`
  `[callback: ch5 practical]`

### Recap + Next

- Recap: dense can't afford pixels; conv = local + shared neuron; one filter spans the
  full depth -> one map; the two formulas (output size, params = k*k*C_in*C_out + C_out);
  equivariance not invariance; pooling (zero params); kernels are learned.
- `[paramgreen Next box]`: "We can build a small CNN. Next: what 10 years of architecture
  design bought - the ImageNet story, from 26% error to superhuman (L17)."

---

## Figures (py_src -> fig, `ma` venv)

1. `pixel_shuffle.pdf` - pomegranate photo vs fixed-permutation shuffled photo; plus the
   zoomed number-patch panel for the image-is-numbers frame.
2. `kernel_zoo.pdf` - pomegranate photo (grayscale) + 6 filtered versions, kernels shown.
3. `param_explosion.pdf` - labeled bar chart, dense vs conv parameter counts.
4. `conv_arithmetic.pdf` - padding/stride small-grid diagrams.
5. `feature_maps.pdf` - learned kernels + activations, small Fashion-MNIST CNN.
6. `cnn_vs_mlp.pdf` - accuracy curves, MLP baseline vs small CNN.

## TikZ (small, throwaway only)

1D sliding-window strip; sparse-vs-dense connection diagrams; stacked feature maps; conv
stack block diagram (if not matplotlib).

## Homework hook (lives on cnn.qmd)

HW1 "Build your own Photoshop, then let the network design the kernels" - Part A NumPy
conv + kernel zoo (local; the pomegranate photo or the student's own), Part B small CNN on
Fashion-MNIST + learned-kernel visualization (Colab). Direct interventions on the "kernels
are hand-engineered" misconception.
