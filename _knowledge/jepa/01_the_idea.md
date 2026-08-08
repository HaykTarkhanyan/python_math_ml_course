# 01 - The idea

## The question the whole field is answering

Self-supervised learning needs a task that produces labels for free. Everything else - the
architecture, the optimiser, the scale - is downstream of the choice of task. The choice is
usually framed as "what do we hide, and what do we ask the model to recover", and the second half
of that question is the one JEPA answers differently from everything before it.

Three answers have been tried, and LeCun's 2022 position paper names them as three
**architectures** rather than three losses, which is the useful framing.

### 1. Joint-embedding architecture (JEA)

Take two views `x` and `y` of the same thing - two crops, two augmentations, an image and its
caption. Encode both. Train so that compatible pairs land close together.

```
x --> Enc --> s_x  \
                     ---> D(s_x, s_y) small if compatible
y --> Enc --> s_y  /
```

This is SimCLR, MoCo, BYOL, DINO, CLIP. It works, and chapter 12 already taught it via CLIP.

Two problems. First, **collapse**: mapping every input to the same vector makes `D` zero for all
pairs, and nothing in the objective forbids it. Second, and less often said, a JEA can only
answer *"are these the same thing?"* It has no way to express *how* `y` relates to `x`. If `y` is
the right half of an image and `x` the left half, a JEA can score their compatibility but cannot
say what the right half looks like.

### 2. Generative architecture

Hide part of the input, reconstruct it in input space, conditioned on what is left.

```
x --> Enc --> Dec --> y_hat,   loss = || y_hat - y ||
```

This is the denoising autoencoder, MAE, BEiT, and - in a different modality - next-token
prediction. It answers the "how" question directly, and it cannot collapse, because the target is
fixed by the data.

Its problem is subtler and it is the reason JEPA exists. **`y` is not a deterministic function of
`x`.** Given the left half of a photograph, the exact pixel values of the right half are
massively underdetermined: the leaf positions, the grain, the exact shade of a shadow. A model
trained with a squared-error pixel loss must nevertheless put probability mass somewhere, and
what minimises expected squared error is the *conditional mean* - which is why reconstruction
models produce blurry output. Worse, from a representation-learning point of view, the model
spends capacity modelling detail that no downstream task will ever ask about. The paper's phrase
for what you want instead is a target for which "irrelevant pixel-level details are eliminated".

### 3. JEPA

Encode both sides, then predict **`s_y` from `s_x` through a predictor**, optionally conditioned
on a latent `z` that soaks up what is genuinely unpredictable.

```
x --> Enc_ctx --> s_x --> Pred(., z) --> s_y_hat
                                          |
                                          v   loss = || s_y_hat - s_y ||
y --> Enc_tgt --> s_y --------------------+
```

Two things distinguish this from both predecessors:

- **The loss lives in representation space**, so the model is not charged for detail it decided
  to discard.
- **The prediction target is learned.** `Enc_tgt` is not a fixed feature extractor and not the
  identity. It is being trained at the same time as everything else, which means the model is
  choosing what "getting it right" means.

That second point is the entire idea, and it is worth stating to students as a single sentence:

> **JEPA gives the model permission to throw information away, and the rest of the design exists
> to stop it throwing everything away.**

## Why "the target is learned" is dangerous

Write out the degenerate solution. Let `Enc_tgt` map every input to the constant vector `c`, and
let `Pred` output `c` regardless of its input. The loss is exactly zero. The representation is
worthless. This is **collapse**, and unlike in the generative case, nothing in the objective
rules it out - because there is no longer any fixed thing the representation has to be able to
reconstruct.

So every JEPA needs an answer to "why doesn't it collapse?". There are three families, and they
are worth teaching as a taxonomy because they recur far beyond JEPA.

### One loss, three choices

The JEPA tutorial (Monemi et al., 2025) gives the useful unification: **every** JEPA objective has
the same two-term shape.

```
L_total  =  E[ d( Pred(s_x, z), s_y ) ]   +   lambda * R( s_1, ..., s_B )
            \_________________________/       \_____________________/
             latent predictive invariance      anti-collapse regulariser
```

The first term is what everyone agrees on. **The families differ only in what `R` is** - and in
one case, in whether it is a term in the loss at all rather than a property of the architecture.
Teaching it this way beats presenting three unrelated tricks, and it makes the LeJEPA argument
legible: LeJEPA is a claim about what `R` *should* be.

| Family (tutorial's name) | `R` is... | Examples | Cost |
|---|---|---|---|
| **Non-parametric estimators** (contrastive) | not a regulariser but a second data term - pull positives together, push negatives apart | SimCLR, MoCo, CLIP | Needs negatives; degrades unless the batch or queue is large; the number of negatives needed grows badly with dimension |
| **Teacher-student schemes** (architectural asymmetry) | absent from the loss. Collapse is blocked *structurally* - stop-gradient on one branch, a predictor on only one branch, a teacher that is a slow EMA of the student | BYOL, SimSiam, DINO, **I-JEPA**, V-JEPA, CNN-JEPA | Works empirically, poorly understood theoretically, and adds hyperparameters (the EMA momentum schedule) whose failure mode is silent |
| **Moment-matching objectives** (explicit regularisation) | an explicit penalty on batch statistics - keep every dimension's variance up, decorrelate the dimensions, or match a whole target distribution | VICReg, Barlow Twins, W-MSE, **C-JEPA**, **LeJEPA / SIGReg** (2025) | Adds a term and a weight; historically weaker, but LeJEPA claims this is the only family with a proof behind it |

I-JEPA and V-JEPA are in the middle row. Two consequences worth stating in class:

- **C-JEPA is literally I-JEPA plus VICReg's three terms** - *variance* (stop any embedding
  dimension going constant across the batch), *covariance* (decorrelate the dimensions),
  *invariance* (views of the same image agree). Same ViT backbone, so its reported gains come from
  the regularisation alone. It exists because its authors argue **I-JEPA's EMA does not fully
  prevent collapse**.
- **LeJEPA** (Nov 2025, Balestriero and LeCun) argues the middle row should be deleted in favour of
  the bottom row - see `03_the_model_line.md`.

## The energy-based picture

LeCun frames all of this as **energy-based modelling**, and the picture is worth one slide
because it makes collapse visible.

Define an energy `F(x, y)` that is low when `x` and `y` are compatible and high otherwise. For a
JEPA:

```
F(x, y) = min_z || Pred(Enc_ctx(x), z) - Enc_tgt(y) ||
```

Training pushes `F` down on observed pairs. The failure mode is not "energy is low on the data" -
that is the goal. The failure mode is **energy is low everywhere**, a flat landscape that
distinguishes nothing. So the real objective is always *low energy on the data plus some device
that keeps energy high off it*, and the three families in the table are three devices.

The latent `z` is the part students find slippery, and it is worth being blunt about: `z` exists
because the future is not unique. Given a video of a ball rolling toward a table edge, several
continuations are physically consistent. `min_z` says "score this prediction by the best
explanation available", which is exactly the multimodality problem chapter 15 (VLA) already met
when averaging two valid trajectories produced a third invalid one. If `z` has too much capacity
it becomes another route to collapse: the model can explain any `y` by choosing `z`, and the
energy goes flat again. Regularising the information content of `z` is a standing requirement.

## Why the target encoder must see the whole input

A detail that is easy to get wrong and that matters: in I-JEPA the target blocks are masked at
the **output** of the target encoder, not at its input. The target encoder sees the entire,
unmasked image, and only afterwards do we select the patch embeddings belonging to the target
blocks.

The paper is explicit that this "distinction is crucial to ensure target representations of a
high semantic level". If the target encoder only saw the target block in isolation, the target
embedding would describe a small patch out of context, and the model would be predicting local
texture again. Because it sees everything, the embedding of a patch already reflects the whole
scene, which is what makes it a semantic target.

It is worth flagging to students that this is a slightly uncomfortable asymmetry - the target
branch has access to information the context branch does not - and it is one of the reasons the
JEPA loss cannot be interpreted as a clean likelihood.

## What this buys, in one number

The clearest evidence for the whole thesis is I-JEPA's Table 7. Identical setup, identical
architecture, only the target changes:

| Target | Arch | Epochs | 1% ImageNet top-1 |
|---|---|---|---|
| Target-encoder output | ViT-L/16 | 500 | **66.9** |
| Pixels | ViT-L/16 | 800 | **40.7** |

Predicting pixels loses 26 points *while getting 60% more training*. Whatever else is true about
JEPA, "the choice of prediction target dominates" is not a marketing claim.
