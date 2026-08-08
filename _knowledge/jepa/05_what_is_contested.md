# 05 - What is contested

Read this before writing slides. Several claims that circulate widely about JEPA do not survive
contact with the papers, and the chapter's credibility depends on getting these right.

## 1. The training loss does not tell you whether it is working

In an EMA-based JEPA the loss measures how well the predictor matches the target encoder - and the
target encoder is itself drifting. A low loss is consistent with a degenerate representation.
There is no threshold that means "trained".

Consequences, and they are practical rather than philosophical:

- **Model selection needs a probe.** Every candidate has to be evaluated end to end on a
  downstream task, which is expensive, which means hyperparameter search is expensive, which
  means fewer configurations get tried than in a supervised setting.
- You cannot early-stop on the loss.
- Two runs with the same loss can differ substantially downstream.

LeJEPA's most useful claim is not its accuracy number, it is that **its loss does correlate with
downstream accuracy**. If that holds up it removes a real operational cost.

## 2. The EMA teacher may not be necessary at all

The EMA teacher is presented in every JEPA paper as the thing that prevents collapse. Two 2025
results push back.

**SALT** (arXiv 2509.24317, Sep 2025) `[read, abstract]` replaces the EMA teacher with a **frozen**
one: first train a target encoder with plain pixel reconstruction under V-JEPA masking, freeze it,
then train a student to predict its latents. Reported findings:

- Higher probing accuracy **at matched pretraining FLOPs**.
- Students outperform V-JEPA 2 encoders under frozen-backbone evaluation across benchmarks.
- Its scaling curves "dominate V-JEPA's accuracy-FLOPs Pareto frontier".
- **Student quality is remarkably robust to teacher quality** - good students emerge from small,
  sub-optimal teachers. Their conclusion: spend the compute on the student.

**LeJEPA** (Nov 2025) removes the teacher, the stop-gradient, and the predictor entirely,
replacing the whole apparatus with one regulariser.

Neither has displaced the standard recipe yet. But a chapter that presents the EMA teacher as
*the* solution to collapse would be teaching something the field is actively unlearning. Present
it as "the device that worked first, and is now contested".

## 3. The dense-feature problem was real and lasted nine months

V-JEPA 2 shipped in June 2025 as a state-of-the-art video world model. In March 2026, V-JEPA 2.1's
own paper describes its predecessor's feature maps as "noisy" with "only fragmented local spatial
structure", and fixes it:

| Task | V-JEPA 2 | V-JEPA 2.1 |
|---|---|---|
| NYUv2 depth RMSE | 0.642 | 0.350 |
| ADE20K mIoU | 24.4 | 47.8 |

An ADE20K mIoU of 24.4 is not a small weakness, it is close to unusable for segmentation. So for
nine months the flagship "world model" had a representation that was strong globally and close to
uninformative about *where things are* - in a system whose entire proposed use is robot
manipulation. This is a good, concrete lesson: **"the representation is good" is not one number**,
and a model can be simultaneously state of the art and missing something basic.

## 4. The physical-reasoning result is a failure, and Meta published it

Covered in `04_world_models_and_planning.md`. Humans 85-95%, models at or near chance on IntPhys 2.
The thesis of the whole programme is that predicting in latent space from video yields intuitive
physics. The benchmark built to test exactly that says it has not happened yet.

Do not soften this. It is the single most important honest fact in the chapter, and the fact that
Meta shipped the benchmark anyway is a point in the field's favour.

## 5. The JEPA image line did not win the image-encoder race

This is the one most likely to be missed, because papers do not advertise it.

- I-JEPA (Jan 2023) does **not** beat DINO (80.1) or iBOT (81.0) at 224px linear probe - it gets
  79.3, and only reaches 81.1 by going to 448px `[read]`. The paper says it "decreases the gap".
- The default general-purpose image encoder in 2026 is **DINOv2 / DINOv3** - which is
  self-distillation with hand-crafted augmentations, i.e. exactly the family I-JEPA was positioned
  against. Meta shipped DINOv3 in 2025 and it, not I-JEPA, is what people build on.
- On video, a direct comparison (arXiv 2509.21595) `[read]` finds **neither dominates**: DINOv3
  clusters better (silhouette 0.31 vs 0.21, 6.16x class separation) and is better on static pose;
  V-JEPA 2 is far more *consistent* across action types (performance variance 0.094 vs 0.288) and
  better on motion-dependent actions. Their conclusion is explicitly that the choice depends on the
  task.

The defensible claim is therefore: **JEPA's advantage is efficiency and motion, not raw accuracy on
static images.** V-JEPA 2.1 narrowing the depth gap to DINOv3 ViT-7B (0.307 vs 0.309) is a 2026
development, not the 2023 story.

## 6. The rhetoric runs well ahead of the evidence

LeCun's public position is that autoregressive LLMs are a dead end and world models are the way
forward. He left Meta over it in November 2025 and raised **$1.03B at a $3.5B pre-money valuation**
in March 2026 for AMI Labs, with no revenue plans and a stated horizon of years.

What the evidence actually supports as of Aug 2026:

- JEPA produces **good, efficient self-supervised encoders**. Well supported.
- JEPA produces **a video model you can plan with, slowly, at 65-80% on pick-and-place**. Supported,
  with the 16 s/action caveat.
- JEPA produces **intuitive physics**. Not supported - see IntPhys 2.
- JEPA **replaces next-token prediction for language**. Not demonstrated. LLM-JEPA is a finetuning
  improvement on paired data; VL-JEPA is the strongest evidence so far and is a 1.6B model beating
  GPT-4o on **one** benchmark it was built for.

A student should leave able to state both the strongest case for the thesis and the strongest case
against it. Anyone who leaves certain either way has been taught badly.

## 7. It may be less different from contrastive learning than advertised

*Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning*
(arXiv 2410.19560) `[search]` argues the two families are more closely related than the framing
suggests, and identifies two specific weaknesses of I-JEPA: EMA is not sufficient to prevent
*complete* collapse, and the I-JEPA prediction does not accurately learn the mean of the patch
representations.

Worth one line, not a section. But it is a corrective to "JEPA is a new third thing".

## 8. There is a structural asymmetry in the objective

Noted in `01_the_idea.md` and repeated here because it is a legitimate criticism: the **target
encoder sees the whole image** while the context encoder sees ~25% of it. The paper is explicit
that this is deliberate and necessary for semantic targets. It also means the two branches are not
symmetric views of the same evidence, the loss is not a likelihood of anything, and comparisons to
"predicting the missing part" are looser than they sound.

## 9. A worked example of secondary sources being wrong

While researching this chapter a search summary stated that I-JEPA achieves **72.4%** semi-supervised
accuracy on 1% ImageNet against MAE's **59.8%**. The paper's Table 2 `[read]`:

- I-JEPA ViT-H/14, 300 epochs: **73.3**
- MAE ViT-H/14, 1600 epochs: **71.5**

The reported gap of 12.6 points is actually 1.8 points - and the real story is that I-JEPA got
there in **300 epochs against MAE's 1600**, which is a *better* argument than the false one.

This is worth an actual slide. The lesson generalises past JEPA: secondary sources inflate gaps,
and a paper's real result is often more interesting than the version that circulates.

## 10. It cannot reconstruct, and that is not a bug you can patch

A scoping fact rather than a criticism, but students will ask and the answer needs to be crisp.

**Standard JEPA is not designed for precise pixel-, waveform-, or token-level reconstruction.** It
deliberately discards the information a decoder would need. So it cannot be used for high-fidelity
synthesis or exact data recovery without bolting on a decoder, an autoregressive generator, or
other generative components.

This is the correct answer to "why not just use JEPA instead of diffusion (ch10)?" - they are not
competing for the same job. JEPA is an *understanding* architecture. The reason it is good at
understanding is precisely the reason it cannot generate.

It also bounds the world-model story: a latent digital twin can tell you what will happen, but not
show you. Anything needing physical-state estimation or visualisation needs a hybrid.

## 11. Two specific technical criticisms of I-JEPA worth knowing

Beyond the general disputes, two follow-up papers name concrete defects:

- **C-JEPA**: I-JEPA's **EMA does not fully prevent collapse**, and its predictor fails to
  accurately model the mean patch representation. Their fix is to add VICReg's variance,
  covariance and invariance terms on the same ViT backbone - so the reported speedup and accuracy
  gain on ImageNet-1K come from the regularisation, not the architecture.
- **StoP-JEPA**: I-JEPA uses **fixed positional embeddings**, so it predicts a masked patch as if
  its exact location were known. Their example is worth stealing for a slide: *given only part of a
  dog, you cannot locate its tail precisely.* MAE and I-JEPA both predict masked tokens
  deterministically and neither represents that uncertainty. StoP models each masked position as a
  Gaussian random variable whose mean is the original position and whose covariance is learned, tying
  the noise projection to the context projection so it cannot collapse back to fixed positions.
  Reported as a few extra lines of code, no added compute, better linear probing.

The second one is pedagogically valuable well beyond JEPA: it is the same "the target is genuinely
uncertain, so stop pretending it isn't" argument as ch15's action multimodality and as the
blurry-mean argument in `01_the_idea.md` - now applied to *position* rather than to content.

## 12. The philosophical objection

*Sora and V-JEPA Have Not Learned The Complete Real World Model* (arXiv 2407.10311) `[search]`
argues from Kant's productive imagination that video models of either kind cannot in principle
acquire a complete world model from observation.

Include it only if the room enjoys that kind of argument. It is not an empirical result and should
not be presented as one. But it does frame the honest question: passive video contains no
counterfactuals - you never see what *would* have happened if the hand had moved left - and
learning causal structure from purely observational data has known limits that chapter 5
(interpretability) and the causal material already gesture at.
