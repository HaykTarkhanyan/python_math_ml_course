# Neural Networks chapter - outline (for approval)

Drafted 2026-07-07. Anchored on `ml_old/Chapter 5 Neural Networks` (L09 Neural Networks,
L10 Backprop+Dropout+HPO), re-cut to the current `ml/` house style and course sequence.

## Locked decisions (from instructor)

- **Scope: MLP core, 2 decks.** Perceptron -> MLP -> forward pass -> backprop -> training
  and regularization. CNNs, RNNs/LSTM, optimizers/init/batchnorm, transformers are
  **deferred** to a later deep-learning chapter (the current `L16`-`L18` skeletons become
  that chapter's stubs; they are not part of this outline).
- **Framework: PyTorch** for the one canonical code snippet per topic and for the practical.
  Old course used Keras; we modernize.
- **Backprop depth: by-hand numbers + vectorized form.** Keep the ml_old worked XOR
  forward+backward with real numbers, then generalize to the matrix/delta recursion.

## What we do NOT re-teach (callback only, already delivered)

Old L10 spent ~20 slides on CV, train/val/test split, and grid/random search. Those are
already taught in **[06] model evaluation** and **[08] hyperparameter tuning**. Here they
are one-line callbacks. This chapter owns only NN-*specific* training.

## The course-callback spine (what makes this our chapter, not a generic intro)

| New concept | Callback to | Framing |
|---|---|---|
| sigmoid neuron | L11 logistic regression | "you already trained one neuron" |
| softmax output layer | L11 multiclass logreg | "same softmax, now on top of hidden features" |
| cross-entropy loss | L11 log-loss | not new, reuse |
| gradient descent | regression chapter | same optimizer, harder function |
| L2 weight decay | L01b Ridge | "Ridge penalty, new name" |
| dropout = model averaging | ch4 bagging / random forest | ensemble intuition |
| representation learning | L01g feature engineering | "the net learns the features you used to hand-craft" |
| grid / random / Optuna | [08] HP tuning | same machinery |
| overfitting / bias-variance / CV | [06] / L01d | don't re-derive |
| "trees still win on tabular" | ch4 trees + boosting | honest scope limit (`TODO_nn_vs_boosting_on_tabular.md`) |

## Proposed file names (numbering to confirm at build)

- Deck 1: `L14_neural_networks.tex` (rebuild the current thin stub; the instructor said the
  current L14 can be ignored).
- Deck 2: `L15_training_neural_networks.tex` (supersedes/merges the current
  `L15_backprop_dropout_hpo` stub).

Note: the former `L14` collision with the classification imbalanced-learning deck is resolved
--- it was renumbered to `15_imbalanced_learning` (2026-07-08), so `L14`/`L15` here are free.
The plan lists NN as video **[23]**. Numbering can be reconciled at build time; not blocking.

---

# DECK 1 - Neural network fundamentals

*From a single neuron to a deep feedforward net. Source: old L09.*
Subtitle idea: "A neuron you already know, stacked."

### Cold open (before Outline)
- **Frame - the wall linear models hit.** Show a two-moons / XOR dataset. "Logistic
  regression (L11) draws a straight boundary. Predict: what's its best accuracy here?"
  Reveal: stuck near chance. Hook: "we need a model that bends the boundary." `[predict-first]`
  `[real fig: two-moons + logreg boundary]`

### Outline frame

### Section 1: Why neural networks?
- `[plain]` transition: "Neural networks" + one line.
- **What is deep learning?** AI superset ML superset DL. DL = ML with learned
  representations. `[TikZ nested sets]`
- **When NNs win, and when they don't.** Win: high-dimensional raw signals (images, audio,
  text), each feature weakly informative, lots of labelled data. Lose: tabular - trees and
  boosting usually beat NNs without heavy tuning. `[callback: ch4]` `[armorange watch-out box]`
- **Representation learning.** Before DL, features were hand-designed (feature engineering,
  L01g). A deep net learns them. This is the whole point. `[callback: L01g]`

### Section 2: The single neuron
- `[plain]` transition.
- **A neuron = affine + activation.** f(x) = phi(w^T x + b). Two steps: weighted sum, then
  nonlinearity. `[TikZ neuron schematic]`
- **You already know one neuron.** A sigmoid neuron IS logistic regression; a linear neuron
  is linear regression. `[callback: L11]` `[armblue key box]`
- **Activation functions.** sigmoid, tanh, ReLU - shapes, ranges, one-line pros/cons. Plant
  the seed: ReLU matters for deep nets (why: deck 2, vanishing gradients).
  `[real fig: 3 activations]`
- **Losses are not new.** L2 for regression, cross-entropy for classification (= log-loss).
  `[callback: L11]`
- **Motivation: one neuron can't do XOR.** A single neuron = linear boundary. Show it fail
  on the cold-open data. Sets up hidden layers. `[predict-first tie-back]`

### Section 3: From one neuron to a network
- `[plain]` transition.
- **The fix: transform the feature space.** XOR is separable in the right coordinates
  (Cartesian -> polar). A hidden layer learns such a transform instead of us picking it.
- **Single hidden-layer architecture.** input layer -> hidden layer (affine+activation) ->
  output. `[TikZ 3-in, 4-hidden, 1-out net]`
- **Worked forward pass, by hand.** Concrete weights and inputs; compute each hidden z, apply
  sigmoid, combine at output. Real numbers (reuse ml_old L09 example). `[worked-numbers]`
- **Multiclass: softmax output.** Add one output neuron per class, softmax over them = a
  probability distribution. Same softmax as multiclass logreg, now on learned features.
  `[callback: L11]`

### Section 4: Deep feedforward networks
- `[plain]` transition.
- **Stacking layers = the MLP.** "A series of affine transforms, each wrapped in a
  nonlinearity." Vectorized layer: z = W a + b, a = phi(z). Sets up backprop notation.
  `[TikZ deep MLP]`
- **Why depth?** Hierarchy of representations (pixels -> edges -> parts -> objects); depth is
  exponentially more expressive than width for some functions (Booleans-over-the-reals /
  Montufar intuition, stated, not proved). `[TikZ or embedded schematic w/ attribution]`
- **Universal approximation, honestly.** One hidden layer can approximate any continuous
  function - but might need absurd width; depth is the practical win. Misconception pre-empt:
  "universal approx does not mean easy to train or generalize."
- **Count the parameters.** MNIST 784-128-64-10 -> ~109k weights. "Gradient descent needs a
  derivative for every one of these. How?" -> bridge to Deck 2. `[hook to backprop]`

### Recap + Next
- Recap bullets (neuron, activation, hidden layers, depth = learned representations).
- `[paramgreen Next box]`: "We can build the network. Next: how do we *learn* 109k weights -
  backpropagation."

---

# DECK 2 - Training neural networks

*Backprop, regularization, and practical tuning. Source: old L10 (CV/HPO stripped to callbacks).*
Subtitle idea: "Backprop, and how to stop it from overfitting."

**Interview decisions (2026-07-07):**
- **Backprop:** full worked XOR (forward + backward, real numbers, weight update, loss-dropped
  check) + the vectorized delta recursion. This is the deck's centerpiece.
- **Optimizers:** teach mini-batch SGD as the base; for momentum / Adam / schedules, *point to
  the course Optimization module* rather than re-teach - "10 Optim: Momentum + First Order
  Methods" (and "09 Optim: Prerequisites and Gradient Descent") on the site. No Adam derivation
  in-deck.
- **Regularization:** cover all four in depth - dropout, weight decay (= Ridge), data
  augmentation, and init (He/Xavier) + BatchNorm. (BatchNorm/init are no longer deferred.)
- **Code:** minimal in the deck - one short training-loop snippet only. The full PyTorch /
  MNIST walkthrough goes in the practical notebook, not the slides.

**Interview round 2 (2026-07-07):**
- **Worked backprop example:** reuse Deck 1's exact 2-2-1 sigmoid net (x=(2,-1), same
  weights, f=0.69) and backprop it with **cross-entropy** (target y=1). Bonus: sigmoid +
  cross-entropy gives output gradient = f - y, the same "residual" as linear/logistic
  regression (callback to L11). Continue to the same net's hidden weights, then the update +
  loss-dropped check.
- **Figures dataset: Fashion-MNIST** (harder than MNIST -> over/underfit, dropout, BN effects
  are visible). Subsample + small nets to keep compute light.
- **Init + BatchNorm:** intuition + rule of thumb only (He for ReLU / Xavier for tanh; BN =
  standardize activations per mini-batch). One frame + one real figure each, no derivations.
- **Momentum/Adam:** one frame - momentum = velocity, Adam = per-parameter adaptive step,
  "default to Adam" - then link to "10 Optim: Momentum + First Order Methods". No math in-deck.

**Interview round 3 (2026-07-07):**
- **CE gradient:** show the 2-line derivation that sigmoid + cross-entropy gives
  dL/df_in = f - y; call back to the same "residual" gradient in linear/logistic regression
  (L11).
- **Gradient checking:** add one short frame - numerical gradient via finite differences,
  (L(w+eps) - L(w-eps)) / 2eps, as a backprop sanity check.
- **Practical/homework:** deferred ("worry about practical later"). The deck's Next box points
  generically to "the practical" - no MNIST-vs-Fashion commitment yet. Figures still use
  Fashion-MNIST.

### Cold open
- **Frame - the 109k-gradient problem.** Recall gradient descent from regression: it needs
  the gradient. Our loss is a deep composition of 109k weights. Numerical differencing =
  one forward pass *per weight* = hopeless. "Backprop computes all of them in ~one extra
  pass." `[hook]`

### Outline frame

### Section 1: Backpropagation
- `[plain]` transition.
- **The idea: chain rule + reuse.** Differentiate the composition from the output backward,
  caching shared intermediates. `[TikZ forward/backward arrows]`
- **Forward pass, store the intermediates.** Recap the small XOR net; list what we cache
  (z_in, z_out, f_in, f_out). `[worked-numbers, sets up backward]`
- **Backward pass, by hand (the gem).** Walk dL/du and dL/dW via the chain rule with real
  numbers; do the weight update at lr; show the loss dropped after one step. 2-3 frames.
  `[worked-numbers]` (directly from ml_old L10)
- **Vectorized backprop.** Generalize to the delta recursion:
  delta_L = grad_a L (*) phi'(z_L); delta_l = (W_{l+1}^T delta_{l+1}) (*) phi'(z_l);
  dL/dW_l = delta_l a_{l-1}^T. `[full derivation, boxed]`
- **This is autograd.** PyTorch builds the computational graph and runs this for you;
  `loss.backward()` is exactly the delta recursion. `[light, 1 frame]`

### Section 2: The training loop
- `[plain]` transition.
- **Batch, stochastic, mini-batch GD + epochs.** Why mini-batches (noise + speed). Callback
  to GD from regression. `[callback: regression]`
- **The training loop, minimally.** One short snippet: forward -> loss -> `zero_grad` ->
  `backward` -> `step`. Just enough to name the pieces; the full run lives in the practical.
  `[minimal code, ~6 lines]`
- **Which optimizer? Reuse what you already have.** SGD is the base. Momentum, Adam, and LR
  schedules are a whole topic you have already covered - point to the course Optimization
  module ("10 Optim: Momentum + First Order Methods"). Default recipe: Adam to start.
  `[reference frame, paramgreen box with the link; NO Adam derivation]`
- **Learning rate intuition.** Too high diverges, too low crawls; the one hyperparameter you
  always touch. `[predict-first: 3 LR curves, which is which]` `[real fig]`
- **Loss landscapes are non-convex** (brief). Many minima/saddles, yet SGD works; why is an
  open research area. `[1 short frame, optional embedded landscape w/ attribution]`

### Section 3: Regularizing neural networks
- `[plain]` transition.
- **Nets overfit hard.** Huge capacity. Callback to bias-variance/overfitting; don't
  re-derive. `[callback: [06]/L01d]`
- **Dropout.** Randomly zero neurons each step; forces redundancy; = training an ensemble of
  sub-networks that share weights (model averaging). Weight scaling at test time.
  `[predict-first: "does deleting neurons at random really help?" -> yes]`
  `[real fig: test error vs dropout rate]` `[callback: ch4 bagging]`
- **L2 weight decay = Ridge.** The same penalty, new name. `[callback: L01b]`
- **Data augmentation.** Cheap label-preserving transforms multiply the data (flips,
  crops, noise). The 6-vs-9 rotation caveat. `[TikZ or embedded examples]`
- **Init (He/Xavier): give gradients a fighting start.** Bad init -> activations blow up or
  vanish before learning begins. He/Xavier scale the initial weights by fan-in so signal
  variance is preserved layer to layer. Directly addresses the vanishing gradients teased in
  Deck 1. `[1 frame, small real fig: activation-variance with/without proper init]`
- **BatchNorm: normalize activations mid-network.** Standardize each layer's pre-activations
  per mini-batch -> faster, more stable training, mild regularization. `[1 frame, real fig:
  train curve with vs without BN]`
- **Early stopping + reading training curves.** The four cases: val > train (fine),
  val >> train (overfit), val ~ train (underfit), val < train (leak/bug). Stop at the val
  minimum. `[real fig: 4 training-curve cases]`

### Section 4: Practical training and tuning
- `[plain]` transition.
- **NN hyperparameters at a glance.** Architecture (depth/width), lr, batch size, dropout
  rate, epochs, optimizer, init, activation - condensed from the ml_old practical table
  (symptom -> fix). `[1 dense reference frame]`
- **Tuning machinery is not new.** Grid / random / Optuna from [08], early stopping as the
  cheap budget guard. `[callback: [08]]`
- **The honest close.** You can now train an MLP end to end. But on tabular data, boosting
  still wins (Deck 1) - NNs earn their keep on raw high-dim signal. `[short callback, no
  repeat of the full Deck 1 tabular frame]`

### Recap + Next
- Recap (backprop = chain rule + reuse; the training loop; the four regularizers - dropout /
  weight decay / augmentation / init+BN; read your curves; optimizers -> optim module).
- `[paramgreen Next box]`: forward pointer to the practical (train an MLP on MNIST in PyTorch)
  and to the future deep-learning chapter (CNNs, RNNs, transformers).

### Figures to generate for Deck 2 (real matplotlib -> `fig/`, scripts in `py_src/`)
1. `dropout_test_error.pdf` - test error vs epochs for a few dropout rates.
2. `lr_curves.pdf` - loss vs epoch for too-high / good / too-low lr.
3. `training_curves_4cases.pdf` - the four val-vs-train patterns.
4. `init_variance.pdf` - activation variance across layers, bad init vs He/Xavier.
5. `batchnorm_curves.pdf` - training loss with vs without BatchNorm.
6. (optional) `loss_landscape.pdf` - a 2D non-convex surface (or a credited external image).

---

## Figures to generate (real matplotlib -> sibling `fig/`, scripts in `py_src/`, `ma` venv)

1. `two_moons_logreg.pdf` - two-moons data + failing linear boundary (cold open, Deck 1).
2. `two_moons_mlp.pdf` - same data, MLP boundary bends around it (payoff, Deck 1 or recap).
3. `activations.pdf` - sigmoid, tanh, ReLU on one axis.
4. `dropout_test_error.pdf` - test error vs epochs for a few dropout rates (Deck 2).
5. `lr_curves.pdf` - loss vs epoch for too-high / good / too-low lr (Deck 2 predict-first).
6. `training_curves_4cases.pdf` - the four val-vs-train patterns.
7. (optional) `loss_landscape.pdf` - a 2D non-convex surface, or embed a credited external one.

## TikZ schematics (conceptual, hand-drawn)

Neuron; single-hidden-layer net; deep MLP; forward/backward arrow diagram; computational
graph; dropout mask; AI/ML/DL nested sets.

## Homework (lives on the `.qmd`, not in the decks - per style guide)

Port both ml_old exercises to PyTorch:
- **Regression:** predict a rectangle's area from its sides (the `nn_area_sol` toy - great
  for "a net learns x1*x2 that a linear model can't").
- **Classification:** MNIST digits MLP (the `mnist_nn_tf` practical), rewritten in PyTorch;
  add a dropout ablation and a training-curve read.

## Deferred to a future deep-learning chapter (out of scope here)

CNNs (convolution/pooling/architectures), RNNs/LSTM/GRU, attention/transformers, modern
optimizers (Adam, schedules), initialization (He/Xavier), batchnorm/layernorm. The current
`L16`-`L18` skeletons + `CNN_BLOCK_DESIGN.md` / `RNN_BLOCK_DESIGN.md` are that chapter's
starting point.

## Open questions for the instructor

1. Deck numbering: keep `L14`/`L15` (folder-local) or renumber to the video `[23]` scheme?
2. Running example: keep MNIST as the anchor, or introduce an Armenian-flavored dataset
   (per the local-examples preference)?
3. Loss-landscape frame: generate a simple one, or embed a credited external visualization?
