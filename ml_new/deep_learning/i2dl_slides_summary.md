# `lecture_i2dl` Slides — Topic Map and Summary

A comprehensive index of the **Introduction to Deep Learning (I2DL)** slide collection
that lives in `_reference/lecture_i2dl/`.

- **Source:** compstat-LMU `lecture_i2dl` (LMU Munich), built on the `latex-math` Beamer system.
- **Site:** https://compstat-lmu.github.io/lecture_i2dl/
- **Investigated:** 2026-06-22
- **Purpose of this file:** know what deep-learning topics the reference slides cover and exactly where each one lives, so material can be reused/adapted for our course.

> Note on "slides paths": every deck has a LaTeX source under `_reference/lecture_i2dl/slides/<folder>/`
> and a compiled PDF under `_reference/lecture_i2dl/slides-pdf/<folder>/` (same filename, `.pdf`).

---

## 1. At a glance

- **53 slide decks** across **13 active topic folders**.
- **6 empty stub folders** reserved for future topics (only a `Makefile`, no slides): `bnn`, `diff`, `flows`, `gan2`, `gnn`, `sparse` (Bayesian NNs, diffusion models, normalizing flows, GANs-part-2, graph NNs, sparse methods).
- Each deck carries a `\learninggoals{}` block (a hand-written 2-4 bullet abstract) and a `\lecturechapter{}` title. This summary is built from those plus the per-frame titles.
- "Frame blocks" counts below are `\begin{frame}` / `\begin{vbframe}` declarations. `vbframe` auto-splits overflowing content, so the actual page count of a PDF is usually higher than the frame-block count.

### Coverage map (deep-learning topic -> where it lives)

| Theme | Folder(s) | Decks |
|---|---|---|
| Intro, history, what is DL | `intro` | 2 |
| Perceptron, MLPs, universal approximation | `mlps` | 8 |
| Backprop, computational graphs, basic training, HW/SW | `opt1` | 5 |
| Regularization, early stopping, dropout, augmentation | `regu` | 3 |
| Optimization challenges, advanced optimizers, activations, init | `opt2` | 4 |
| CNN fundamentals (convolution, pooling, components, applications) | `cnn1` | 8 |
| CNN convolution variants (1D/2D/3D, dilated, transposed, separable) | `cnn2` | 3 |
| Modern CNN architectures (LeNet -> DenseNet, U-Net) | `cnn3` | 2 |
| RNNs, LSTM/GRU, attention and transformers | `rnn` | 5 |
| Autoencoders, unsupervised, manifold learning, VAE | `ae` | 6 |
| Generative models (overview) | `genmod` | 1 |
| GANs (intro, challenges, variants) | `gan1` | 3 |
| Adversarial examples and adversarial training | `adver` | 3 |

---

## 2. Intended course arc (from the repo README)

The README lays out a 14-block lecture sequence. The folders above feed it as noted in brackets.

1. Introduction, Overview, Brief History of Deep Learning  -> `intro`
2. Deep Feed-Forward NNs, Gradient Descent, Backpropagation, Hardware/Software  -> `mlps` + `opt1`
3. Regularization of Neural Networks, Early Stopping  -> `regu`
4. Dropout and Challenges in Optimization  -> `regu` (dropout) + `opt2` (challenges)
5. Advances in Optimization  -> `opt2` (advanced-optim)
6. Activation Functions and Initialization  -> `opt2` (activations, initialization)
7. Convolutional Neural Networks, Variants of CNNs, Applications  -> `cnn1` + `cnn2`
8. Modern CNNs and Overview of some Applications  -> `cnn3`
9. Recurrent Neural Networks  -> `rnn` (introduction, backprop)
10. Modern RNNs and Applications  -> `rnn` (modernrnn, applications, attention)
11. Deep Unsupervised Learning  -> `ae` (unsupervised-learning)
12. Autoencoders, AE Regularization and Variants  -> `ae` (autoencoders, regularized, specific)
13. Manifold Learning  -> `ae` (manifold-learning)
14. Deep Generative Models, VAE, GANs  -> `genmod` + `ae` (vae) + `gan1`

Plus an extra block not in the numbered list: **Adversarial examples / training** -> `adver`.

The `slides-pdf/topicN/` folders are an older, partial render that confirms the early-lecture bundling:
`topic1` = intro + all MLP decks, `topic2` = `opt1`, `topic3` = `regu`, `topic5` = `cnn1`, `topic8` = adversarials (single combined PDF), `topic9` = autoencoders. Topics 4, 6, 7 and everything past 9 are not present as bundles, so treat the `slides/` source folders as the authoritative current set.

---

## 3. Deck-by-deck breakdown

Format: `filename` (frame blocks) - **chapter title**. Goals = the deck's own `\learninggoals`. Key frames = representative slide titles.

### `intro/` - Introduction and history (Topic 1)

- `slides-intro-introduction.tex` (12) - **Introduction**
  - Goals: relationship of DL and ML; concept of representation / feature learning; use-cases and data types for DL.
  - Key frames: What is Deep Learning; DL and Neural Networks; Image Classification with NNs; use-cases for images / text / text classification; speech applications.
- `slides-intro-brief-history.tex` (2) - **Brief History**
  - Goals: predecessors of modern (deep) neural networks; history of DL as a field.
  - Key frames: a brief history of neural networks; what the field looks like today.

### `mlps/` - From a single neuron to deep feed-forward nets (Topic 2)

- `slides-mlps-single-neuron.tex` (6) - **Single Neuron / Perceptron**
  - Goals: graphical representation of a single neuron; affine transformation + non-linear activation; hypothesis space of a single neuron; typical loss functions.
  - Key frames: A Single Neuron; Hypothesis Space; Optimization.
- `slides-mlps-single-hidden-layer-networks.tex` (9) - **Single hidden layer neural networks**
  - Goals: architecture of single hidden layer nets; representation learning / why hidden layers help; typical non-linear activations.
  - Key frames: Motivation; Representation Learning; Single Hidden Layer Networks (+ Example); Hidden Layer Activation Function.
- `slides-mlps-mlps-as-predictor.tex` (9) - **Single hidden layer neural networks** *(duplicate of the deck above - same chapter, same frames; likely an older filename kept around)*
- `slides-mlps-matrix-notation.tex` (2) - **MLP - Matrix Notation**
  - Goals: compact representation of NN equations; vector notation for neuron layers; vector/matrix notation of bias and weight parameters.
- `slides-mlps-multilayer-FNNs.tex` (4) - **MLP - Multi-Layer Feedforward Neural Networks**
  - Goals: architectures of deep neural networks; deep nets as chained functions; why depth.
  - Key frames: Feedforward neural networks (+ Example); Why add more layers?; Deep neural networks.
- `slides-mlps-multiclass-classification.tex` (6) - **Single Hidden Layer Networks for Multi-Class Classification**
  - Goals: NN architectures for multi-class classification; softmax activation; softmax loss.
- `slides-mlp-univ-approx-theorem.tex` (4) - **Universal Approximation**
  - Goals: universal approximation theorem for one-hidden-layer nets; pros/cons of low approximation error.
  - Key frames: regression and classification examples with training-iteration snapshots.
- `slides-mlps-xor.tex` (2) - **XOR-Problem**
  - Goals: an example a single neuron cannot solve but a single hidden layer net can.

### `opt1/` - Backprop, training, hardware/software (Topic 2)

- `slides-opt1-comp-graphs.tex` (5) - **Chain Rule and Computational Graphs**
  - Goals: chain rule of calculus; computational graphs.
  - Key frames: chain rule examples; computational graph of a neural net.
- `slides-opt1-basic-backpropagation1.tex` (5) - **Basic Backpropagation 1**
  - Goals: forward and backward passes; chain rule; details of backprop.
  - Key frames: Basic Idea; XOR example; Forward pass; Backward pass; Result.
- `slides-opt1-basic-backpropagation2.tex` (8) - **Basic Backpropagation 2**
  - Goals: backprop formalism and recursion.
  - Key frames: Backward Computation and Caching; Backprop Recursion.
- `slides-opt1-basic-training.tex` (8) - **Basic Training**
  - Goals: empirical risk minimization; gradient descent; SGD; minibatch GD; learning rates and (S)GD specifics.
  - Key frames: Training Neural Networks; Gradient Descent and Optimality; Learning rate; Weight Initialization; Stochastic gradient descent.
- `slides-opt1-hardware-and-software.tex` (7) - **Hardware and Software**
  - Goals: GPU training for accelerated learning; software for hardware support; DL software platforms.
  - Key frames: GPUs; TPUs; software for deep learning.

### `regu/` - Regularization (Topics 3-4)

- `slides-regu-basic-regularization.tex` (7) - **Basic Regularization**
  - Goals: regularized cost functions; norm penalties; weight decay; equivalence with constrained optimization.
  - Key frames: Regularized Risk Minimization; L2 / weight decay; equivalence to constrained optimization; TensorFlow Playground exercise.
- `slides-regu-early-stopping.tex` (3) - **Early Stopping**
  - Goals: how early stopping works; how it acts as a regularizer; how it imitates L2 in some cases.
- `slides-regu-ensemble-dropout-augmentation.tex` (7) - **Dropout and Augmentation**
  - Goals: recap of ensemble methods; dropout; data augmentation.
  - Key frames: Dropout (algorithm, weight scaling, example); dropout vs weight decay; Dataset Augmentation.

### `opt2/` - Optimization challenges, advanced optimizers, activations, init (Topics 4-6)

- `slides-optim-challenges.tex` (15) - **Challenges in Optimization**
  - Goals: ill-conditioning; local minima; saddle points; cliffs and exploding gradients.
  - Key frames: effects of curvature; ill-conditioned Hessian; unimodal vs multimodal loss surfaces; saddle points; cliffs / exploding gradients.
- `slides-advanced-optim.tex` (27) - **Advanced Optimization** *(largest deck)*
  - Goals: SGD with momentum; learning-rate schedules; adaptive learning rates; batch normalization.
  - Key frames: Momentum; Nesterov momentum; LR schedules; cyclical LR; Adagrad; RMSProp; Adam; Batch Normalization (+ illustration, prediction).
- `slides-activations.tex` (13) - **Modern Activation Functions**
  - Goals: optimization challenges related to activations; activations for hidden units; activations for output units.
  - Key frames: sigmoidal activations; ReLU and generalizations; output activations.
- `slides-initialization.tex` (4) - **Network Initializations**
  - Goals: why initialization matters; weight initializations; bias initialization.

### `cnn1/` - CNN fundamentals (Topic 7)

- `slides-cnn-introduction.tex` (3) - **CNN: Introduction**
  - Goals: what are CNNs; when to apply them; a glimpse into CNN architectures.
- `slides-cnn-conv2d.tex` (7) - **Convolutional Operation**
  - Goals: what are filters; the convolution operation; 2D convolution.
  - Key frames: filters to extract features; horizontal vs vertical edges; working with images; the 2D convolution.
- `slides-cnn-math.tex` (11) - **Convolutions - Mathematical Perspective**
  - Goals: convolution vs cross-correlation.
  - Key frames: properties of the convolution; convolution theorem (+ proof); cross-correlation vs convolution.
- `slides-cnn-properties-of-convolution.tex` (3) - **Properties of Convolution**
  - Goals: sparse interactions; parameter sharing; equivariance to translation.
- `slides-cnn-components.tex` (7) - **CNN Components**
  - Goals: input channel; padding; stride; pooling.
  - Key frames: valid vs same padding; padding and network depth; max / average pooling.
- `slides-cnn-pooling.tex` (3) - **CNN: Pooling** *(overlaps with the pooling content in `cnn-components`)*
  - Key frames: Max Pooling; Average Pooling; comparison.
- `slides-cnn-architecture.tex` (3) - **CNN: Architecture**
  - Goals: architecture (three perspectives on the CNN structure).
- `slides-cnn-application.tex` (8) - **CNN Applications**
  - Goals: application of CNNs in visual recognition.
  - Key frames: image classification; CNN vs fully connected on CIFAR-10; image colorization; object localization; semantic segmentation; image captioning; visual question answering.

### `cnn2/` - Convolution variants (Topic 7)

- `slides-convolution-types.tex` (11) - **1D / 2D / 3D Convolutions**
  - Goals: 1D, 2D, 3D convolutions.
  - Key frames: 1D convolutions (operation, sensor data, text mining, advantages); 2D; 3D (+ data).
- `slides-dilated-transposed-convolutions.tex` (5) - **Important Types of Convolutions**
  - Goals: dilated convolutions; transposed convolutions.
  - Key frames: dilated convolutions; transposed convolutions (+ drawbacks).
- `slides-separable-convolutions-flattening.tex` (6) - **Separable Convolutions and Flattening**
  - Goals: separable convolutions; flattening.
  - Key frames: spatially separable; depthwise separable; depthwise / pointwise convolution; flattening.

### `cnn3/` - Modern CNN architectures (Topic 8)

- `slides-modern-cnn-1.tex` (7) - **Modern Architectures - I**
  - Goals: LeNet; AlexNet; VGG; Network in Network.
  - Key frames: LeNet; AlexNet; VGG blocks/network; NiN blocks; global average pooling.
- `slides-modern-cnn-2.tex` (7) - **Modern Architectures - II**
  - Goals: GoogLeNet; ResNet; DenseNet; U-Net.
  - Key frames: Inception modules; GoogLeNet; residual block / skip connections; ResNet; DenseNet; U-Net.

### `rnn/` - Recurrent nets, LSTM/GRU, attention (Topics 9-10)

- `slides-introduction.tex` (15) - **Recurrent Neural Networks - Introduction**
  - Goals: why we need them; how they work; computational graph of recurrent networks.
  - Key frames: motivation; sentiment-analysis example; parameter sharing; use-case-specific architectures; bidirectional RNNs.
- `slides-backprop.tex` (2) - **Recurrent Neural Networks - Backpropagation**
  - Goals: how backprop works for RNNs (BPTT); exploding and vanishing gradients.
  - Key frames: character-level language model; long-term dependencies.
- `slides-modernrnn.tex` (5) - **Modern Recurrent Neural Networks**
  - Goals: LSTM cell; GRU cell.
  - Key frames: Gated Recurrent Units; GRU vs LSTM.
- `slides-applications.tex` (4) - **Applications of RNNs**
  - Goals: language modelling; encoder-decoder architectures; further RNN applications.
  - Key frames: language modelling; word embeddings; encoder-decoder network.
- `slides-attention.tex` (3) - **Attention and Transformers**
  - Goals: attention mechanism; transformers; the CNN alternative to RNNs.
  - Key frames: Attention; Transformers; Transformer Components.

### `ae/` - Unsupervised, autoencoders, manifold learning, VAE (Topics 11-14)

- `slides-unsupervised-learning.tex` (3) - **Unsupervised Learning**
  - Goals: unsupervised learning tasks; unsupervised deep learning.
- `slides-autoencoders.tex` (8) - **Autoencoders - Basic Principle**
  - Goals: task and structure of an AE; undercomplete AEs; relation of AEs and PCA.
  - Key frames: AE task and structure; computational graph; undercomplete AEs; encode MNIST experiment; increasing capacity; AEs as PCA.
- `slides-autoencoders-regularized.tex` (5) - **Regularized Autoencoders**
  - Goals: overcomplete AEs; sparse AEs; denoising AEs; contractive AEs.
  - Key frames: overcomplete-AE problem; sparse AE; denoising AE; encode MNIST with a DAE.
- `slides-autoencoders-specific.tex` (2) - **Specific Autoencoders and Applications**
  - Goals: convolutional AEs; applications of AEs.
- `slides-manifold-learning.tex` (1) - **Manifold learning**
  - Goals: manifold hypothesis; manifold learning with AEs.
- `slides-vae.tex` (n/a) - **Variational Autoencoder (VAE)**
  - Goals: introduction and intuition of VAE; VAE parameter fitting; reparameterization trick.
  - Note: this deck's body is built from included sub-files, so frame titles do not show up via a simple scan; check the PDF directly.

### `genmod/` - Generative models overview (Topic 14)

- `slides-introduction.tex` (9) - **Introduction to Generative Models**
  - Goals: learning a generative model; examples of generative models.
  - Key frames: which face is fake; density fitting / learning a generative model; why generative models; application examples (image generation, neural style transfer, inpainting, semantic-labels-to-images, text-to-image).

### `gan1/` - Generative Adversarial Networks (Topic 14)

- `slides-GAN-intro.tex` (13) - **Introduction to Generative Adversarial Networks (GANs)**
  - Goals: architecture of a GAN; minimax loss; training a GAN.
  - Key frames: what is a GAN; fake-currency illustration; minimax loss; training pseudocode + illustration; divergence measures; optimal discriminator.
- `slides-GAN-challenges.tex` (10) - **Challenges for GAN Optimization**
  - Goals: (no) convergence to a fixed point; problems of the adversarial setting.
  - Key frames: convergence vs chaotic behaviour vs cycles; non-stationary loss surface; illustration of convergence; challenges for GAN training.
- `slides-GAN-variants.tex` (9) - **GAN variants**
  - Goals: non-saturating loss; conditional GANs.
  - Key frames: non-saturating loss; other loss functions; architecture-variant GANs; conditional GANs (motivation, architecture, examples).

### `adver/` - Adversarial examples and training (extra topic)

- `slides-adversarials-examples.tex` (10) - **Adversarial Examples**
  - Goals: adversarial robustness; adversarial examples; targeted attacks.
  - Key frames: adversarial robustness; creation of adversarial examples; ResNet50 example; targeted attacks.
- `slides-adversarials-training-basics.tex` (3) - **Adversarial Training Basics**
  - Goals: basics of adversarial training; adversarial training for linear models.
  - Key frames: adversarial training; linear models; MNIST example.
- `slides-adversarials-training-advances.tex` (2) - **Adversarial Training Advances**
  - Goals: advanced adversarial training; projected gradient descent; fast gradient sign method.

---

## 4. Empty / planned-but-not-yet-written folders

These exist under `slides/` with only a `Makefile` (no decks yet). They signal the topics the authors planned to add:

- `bnn` - Bayesian neural networks
- `diff` - diffusion models
- `flows` - normalizing flows
- `gan2` - GANs, part 2
- `gnn` - graph neural networks
- `sparse` - sparse methods / sparsity

---

## 5. Companion material in the repo (not slides)

Beyond the decks, `_reference/lecture_i2dl/` ships supporting material worth knowing about:

- **`exercises/R/`** - 12+ R/Keras lab templates (`R-lab-01` ... `R-lab-12`, plus an LSTM lab) matching the lecture order, with a dog-breed-identification transfer-learning dataset.
- **`cheetsheets-pdf/`** - one-page cheatsheets (`cheatsheet_01`..`06`, plus `7-1/7-2`, and a notation cheatsheet).
- **`code-demos/`** - code demo stubs.
- **`exam/`** - exam question assignment tooling.
- **`attic/`** - retired decks (graphical models, maximum likelihood).
- The README lists curated external resources: Goodfellow et al. *Deep Learning*, d2l.ai, Karpathy's blog, the CS231n course, distill.pub articles, and topic-specific reading on optimization, regularization, CNNs, LSTMs, and VAEs.

---

## 6. Things to watch when reusing this material

- **Duplicate deck:** `mlps/slides-mlps-mlps-as-predictor.tex` and `mlps/slides-mlps-single-hidden-layer-networks.tex` are the same content under two filenames.
- **Pooling appears twice:** in `cnn1/slides-cnn-components.tex` and again in `cnn1/slides-cnn-pooling.tex`.
- **Convolution properties / math overlap:** `cnn1/slides-cnn-math.tex` and `cnn1/slides-cnn-properties-of-convolution.tex` share material on sparse interactions and parameter sharing.
- **Template "bleed" in some `\learninggoals`:** a few GAN / generative / adversarial decks have boilerplate goal lines copy-pasted from sibling decks (e.g. "projected gradient descent", "fast gradient sign method", "principal component analysis"). Trust the deck's own leading goals over those trailing lines. Cleaned-up goals are what is listed in section 3.
- **`slides-pdf/topicN/` is stale/partial:** use the `slides/` source folders as the source of truth.
- Slides are built with the compstat-LMU `latex-math` macros; compiling a deck standalone requires that the `latex-math/` repo is cloned into the project root (per the repo README).
