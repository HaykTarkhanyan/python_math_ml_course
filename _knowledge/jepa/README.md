# JEPA - Joint-Embedding Predictive Architectures

Reference material for an `ml/` chapter. Gathered **2026-08-08**, from primary sources where the
paper was readable (the I-JEPA paper was pulled down and read in full with `pdftotext`) and from
web search elsewhere. Every number is labelled with where it came from in `sources.md`.

## In three sentences

A **JEPA** is trained to predict, not the missing part of an input, but the *representation* of
the missing part - the target is produced by a second encoder that is itself being learned, so
the model is free to throw away whatever it cannot predict. That freedom is the whole point and
the whole danger: if the encoder may discard information to make prediction easier, the cheapest
solution is to discard all of it and output a constant, which is called **collapse**, and most of
the engineering in this field is machinery to permit the first without permitting the second.
Yann LeCun's claim, from a 2022 position paper onwards, is that this is not one more
self-supervised trick but the right shape for a *world model* - something an agent can run
forward in its head to plan - and in 2026 he is running a $1.03B startup on that claim.

## Why this is worth a chapter here

Three reasons, in order of how much they matter pedagogically.

**1. It is the cleanest instance of a question the course keeps circling: what should a model be
asked to predict?** Chapter 8 (autoencoders) predicts the input. Chapter 8b (GANs) and ch10
(diffusion) predict the data distribution. Chapter 9 / the LLM material predicts the next token.
JEPA is the first thing in the course that predicts *a representation the model itself invented*,
and the single ablation that shows why - I-JEPA scores **66.9** with a representation-space target
and **40.7** with a pixel target, the pixel run getting 800 epochs against 500 - is one of the
most convincing single numbers available anywhere in self-supervised learning.

**2. It carries a real, live intellectual dispute that students can watch being adjudicated.**
The most prominent AI researcher alive says the dominant paradigm is a dead end and left the
largest AI lab in the world to prove it. That is a genuinely open question, the evidence is
mixed, and it is far more interesting to teach than another architecture. Crucially, the
evidence cuts both ways and the honest answer today is "not settled" - see `05_what_is_contested.md`.

**3. It makes collapse concrete.** Collapse is usually taught as a footnote in the contrastive
learning slide. Here it is the central design constraint, it has a clean energy-based picture,
and there are three genuinely different families of fixes, one of which (LeJEPA, Nov 2025) claims
a proof rather than a heuristic. This is a good place in the course to show that "the loss went
down" and "the model learned something" are different statements.

## Files

| File | Contents |
|---|---|
| `01_the_idea.md` | The three architectures (joint-embedding / generative / JEPA), why pixel targets waste capacity, the energy-based picture, collapse, and the three families of anti-collapse machinery |
| `02_the_mechanics.md` | I-JEPA in full detail with the paper's own numbers: context/target encoders, EMA, the predictor, multi-block masking, and the four ablations that carry the argument |
| `03_the_model_line.md` | I-JEPA to V-JEPA to V-JEPA 2 / 2-AC to V-JEPA 2.1 to VL-JEPA, plus LeJEPA, LLM-JEPA and the long tail of modality variants. Dates, sizes, numbers, open-weight status |
| `04_world_models_and_planning.md` | LeCun's six-module architecture, H-JEPA, action-conditioned prediction, planning by minimising latent distance, and the two incompatible definitions of "world model" |
| `05_what_is_contested.md` | The honest part. What is disputed, what has already been walked back, where JEPA loses, and one secondary-source number that is simply wrong |
| `06_teaching_notes.md` | Prerequisites already in this course, the misconceptions to pre-empt, figure ideas, and exercises |
| `sources.md` | Every URL, with read/search status and what it was good for |

## Reading order

`01` is the conceptual core and is most of one lecture. `02` is the mechanism and the evidence.
`03` and `04` are the second lecture. `05` is what keeps the chapter honest and should be read
before writing a single slide - several claims that circulate widely do not survive contact with
the papers.

## One warning about this literature

JEPA has an unusual amount of **advocacy** attached to it, in both directions. LeCun promotes it
in public as the successor to LLMs; his critics dismiss it because he does. Neither posture
matches the papers, which describe a strong self-supervised vision encoder with a genuinely
interesting objective, a promising and clearly incomplete video world model, and an open question
about language.

Concretely: while writing these notes, a search-result summary confidently reported that I-JEPA
gets **72.4%** on 1% ImageNet against MAE's **59.8%**. The paper's own Table 2 says I-JEPA
ViT-H/14 gets **73.3** and MAE ViT-H/14 gets **71.5**. The gap was inflated roughly sixfold by a
secondary source. Every number in these notes that is marked `[read]` was taken out of the paper
itself for exactly this reason.
