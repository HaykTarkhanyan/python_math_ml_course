# 02 - The mechanics, with the paper's own numbers

Everything in this file marked `[read]` was taken from the I-JEPA paper itself
(arXiv 2301.08243v3, CVPR 2023), downloaded and read with `pdftotext`. Nothing here is from a
blog summary.

## The three networks

I-JEPA has exactly three trainable-or-tracked components. All are Vision Transformers.

| Component | What it is | Detail `[read]` |
|---|---|---|
| **Context encoder** `f` | Standard ViT. Sees only the visible context patches | Trained by gradient descent |
| **Target encoder** `f_bar` | Standard ViT, same architecture | **Not** trained by gradient descent. Weights are an exponential moving average of the context encoder's |
| **Predictor** `g` | A deliberately *narrow* ViT | Embedding dimension fixed at **384** regardless of backbone size; same number of heads as the backbone; depth **6** for a ViT-B/16 backbone, **12** for ViT-L/16, ViT-H/16 and ViT-H/14, **16** for ViT-G/16 |

Two conventions worth noting because they trip people up:

- **There is no `[CLS]` token.** Evaluation uses the average-pooled patch representations from the
  *target* encoder, either from the last layer or the concatenation of the last four.
- **The target encoder is what you keep.** After pretraining, the thing you ship and probe is
  `f_bar`, not `f`.

## The EMA update

```
theta_target  <-  m * theta_target  +  (1 - m) * theta_context
```

`[read]`: the target encoder is initialised **identical** to the context encoder, momentum starts
at **m = 0.996** and is increased linearly to **1.0** over pretraining. By the end of training the
target encoder has stopped moving at all.

Three things to say about this in class:

1. It is the same trick as BYOL and DINO. Nothing about it is JEPA-specific.
2. It is the anti-collapse device. The target is a slowly-moving version of yourself, so "output a
   constant" is not a fixed point you can reach in one step - you would have to drag the teacher
   there too, and it resists.
3. **It is a heuristic, not a theorem.** There is no general proof that EMA prevents collapse.
   By 2025-2026 there is published work arguing it is not even necessary
   (see `05_what_is_contested.md`).

## The masking strategy - and why it is not a detail

`[read]`, from Section 4 and Appendix:

- Sample **M = 4** target blocks, each with scale in **(0.15, 0.2)** of the image and aspect ratio
  in **(0.75, 1.5)**. Blocks may overlap each other.
- Sample **one** context block with scale in **(0.85, 1.0)** and **unit aspect ratio**.
- **Remove from the context any region overlapping any target block.** Without this the task is
  trivial.
- Result: the context is on average **25%** of the patches - informative but sparse, so cheap to
  encode.
- Masks are sampled per image, but sizes are constrained to be equal across images co-located on
  the same GPU so the batch stays rectangular. Implemented in the data loader's collate function.

The obvious reaction is that this is a pile of arbitrary hyperparameters. The ablation says
otherwise. Table 6, ViT-B/16, 300 epochs, linear probe on 1% ImageNet `[read]`:

| Masking strategy | Targets | Context | Avg context ratio | Top-1 |
|---|---|---|---|---|
| **multi-block** (proposed) | 4 blocks, scale (0.15, 0.2) | block (0.85, 1.0) minus overlap | 0.25 | **54.2** |
| rasterized | 3 of 4 quadrants | the remaining quadrant | 0.25 | 15.5 |
| block | 1 block, scale 0.6 | complement | 0.4 | 20.2 |
| random | random patches, 60% | complement | 0.4 | 17.6 |

A **34 to 39 point** spread from the masking strategy alone, at fixed architecture and fixed
compute. This is a bigger effect than most of what the course has taught about architecture
choice, and it is the number to use if a student asks why anyone would fuss over how you cut up
an image.

The two design rules the paper extracts, and the reason each matters:

- **Targets must be large enough to be semantic.** A small target is a texture patch; predicting
  it teaches you texture. Appendix Table 8 sweeps target scale and larger is better up to a point.
- **Context must be spatially distributed and informative.** A single quadrant (the "rasterized"
  row) is 25% of the image, the same budget as the proposed strategy, and scores 15.5 against
  54.2. Same amount of information by patch count, catastrophically worse by *placement*.

## The target-space ablation - the load-bearing number

Table 7 `[read]`. Linear evaluation on 1% ImageNet:

| Target | Arch | Epochs | Top-1 |
|---|---|---|---|
| Target-encoder output | ViT-L/16 | 500 | **66.9** |
| Pixels | ViT-L/16 | 800 | **40.7** |

Same architecture, same masking, only the target changes; the pixel run gets 60% more epochs and
still loses 26.2 points. If the chapter has one figure, it should be this one.

## Where I-JEPA actually stands - the honest table

Table 1 `[read]`. Linear evaluation on full ImageNet-1k:

| Method | Arch | Epochs | Top-1 |
|---|---|---|---|
| *No view augmentations* | | | |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| MAE | ViT-B/16 | 1600 | 68.0 |
| MAE | ViT-L/16 | 1600 | 76.0 |
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| **I-JEPA** | ViT-B/16 | 600 | 72.9 |
| **I-JEPA** | ViT-L/16 | 600 | 77.5 |
| **I-JEPA** | ViT-H/14 | 300 | **79.3** |
| **I-JEPA** | ViT-H/16 (448px) | 300 | **81.1** |
| *With view augmentations* | | | |
| SimCLR v2 | RN152 (2x) | 800 | 79.1 |
| DINO | ViT-B/8 | 300 | 80.1 |
| iBOT | ViT-L/16 | 250 | 81.0 |

Read this carefully before putting it on a slide. I-JEPA **clearly beats** everything in its own
category (no hand-crafted augmentations) while using **a quarter to a half the epochs**. It does
**not** beat DINO (80.1) or iBOT (81.0) at 224px; only the 448-resolution variant reaches 81.1.
The paper's own wording is that it "decreases the gap". A slide claiming I-JEPA beat the
augmentation-based methods would be wrong.

Table 2, 1% ImageNet `[read]`:

| Method | Arch | Epochs | Top-1 |
|---|---|---|---|
| data2vec | ViT-L/16 | 1600 | 73.3 |
| MAE | ViT-L/16 | 1600 | 67.1 |
| MAE | ViT-H/14 | 1600 | 71.5 |
| **I-JEPA** | ViT-L/16 | 600 | 69.4 |
| **I-JEPA** | ViT-H/14 | 300 | **73.3** |
| **I-JEPA** | ViT-H/16 (448px) | 300 | **77.3** |
| iBOT | ViT-B/16 | 400 | 69.7 |
| DINO | ViT-B/8 | 300 | 70.0 |

**This is the table a widely-circulated secondary source gets wrong**, claiming 72.4 for I-JEPA
against 59.8 for MAE. The real comparison at matched architecture is 73.3 against 71.5, and MAE
got 1600 epochs to I-JEPA's 300. The real story is efficiency, not a blowout.

## The efficiency claim

`[read]`, from the abstract and Section 7: a **ViT-Huge/14 trained on 16 A100 GPUs in under 72
hours**. The paper's sharper phrasing: a ViT-Huge/14 trained with I-JEPA "requires less
computational effort than a ViT-Small/16 trained with iBOT".

The reason is structural and worth teaching, because it explains why augmentation-free matters
beyond ideology:

- Augmentation-based joint-embedding methods (DINO, iBOT, MSN) process **many views** of every
  image per step - multi-crop is typically 2 global plus 8-10 local crops.
- I-JEPA processes **one** view. The target encoder does one forward pass over the full image; the
  context encoder sees only ~25% of the patches.

So the saving is not a clever kernel, it is that the objective does not require you to look at the
same image ten times.

## What the predictor actually learns

The paper decodes predictor outputs through a separately-trained generative decoder purely for
visualisation (Figure 6). The qualitative finding is the one that makes the idea click:

> across samples, the *pose and object part* are consistent (the back of a bird, the top of a
> car), while precise low-level detail and background vary.

Read that as: **the predictor is representing exactly what is predictable and is silent about the
rest.** That is the behaviour the architecture was designed to produce, and it is the cleanest
available answer to "what does predicting in latent space actually buy you".

Do not over-claim it - the decoder is not part of I-JEPA, and these are hand-picked
visualisations in a paper.

## The training recipe, for completeness

`[read]`: AdamW; batch size 2048; learning rate warmed linearly from 1e-4 to 1e-3 over the first
15 epochs then cosine-decayed to 1e-6; weight decay increased linearly from 0.04 to 0.4 over
training. Loss is the average L2 distance between predicted and target patch embeddings over the
target blocks.

Nothing exotic. The interesting part of I-JEPA is entirely in the objective and the masking, not
in the optimisation - which is itself a useful thing to point out.
