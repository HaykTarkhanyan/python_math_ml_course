# 03 - The model line

Dates are the arXiv v1 date unless stated. Status key as in `sources.md`: `[read]` means the
claim came out of a page or paper I fetched and read; `[search]` means it came from a
search-result synthesis and should be treated as reliable for headline figures, less so for fine
detail.

## The main sequence

| Model | Date | Modality | Size | Pretraining data | Headline result |
|---|---|---|---|---|---|
| **I-JEPA** | Jan 2023 (CVPR 2023) | Image | ViT-B/16 to ViT-G/16 | ImageNet-1k / ImageNet-22k | 79.3 linear ImageNet at 300 ep; ViT-H/14 in <72h on 16 A100s `[read]` |
| **V-JEPA** | Feb 2024 | Video | ViT-L/H | ~2M public videos | 82.1 K400, 71.2 SSv2, frozen evaluation `[search]` |
| **V-JEPA 2** | Jun 2025 | Video | 1.2B (ViT-g) | >1M hours video + 1M images | 77.3 SSv2; 39.7 recall@5 EK100 anticipation; 84.0 PerceptionTest, 76.9 TempCompass with an 8B LLM `[read]` |
| **V-JEPA 2-AC** | Jun 2025 | Video + action | V-JEPA 2 + action head | <62h unlabelled DROID robot video | Zero-shot pick-and-place on Franka arms in two unseen labs, 65-80% `[read]` |
| **LeJEPA** | Nov 2025 | Image (general) | up to 1.8B | various | ViT-H/14 79% ImageNet linear in 100 epochs; beats DINOv2/v3 on in-domain pretraining `[read, secondary]` |
| **VL-JEPA** | Dec 2025 | Vision + language | ~1.6B | image-text and video-text | 65.7% WorldPrediction-WM, above GPT-4o at 53.3 and Gemini-2.0 at 55.6; 50% fewer trainable params than comparable VLMs `[read]` |
| **V-JEPA 2.1** | Mar 2026 | Video + image | ViT-g 1B, ViT-G 2B, distilled ViT-B 80M / ViT-L 300M | VisionMix-163M | NYUv2 depth RMSE 0.350 vs V-JEPA 2's 0.642; ADE20K mIoU 47.8 vs 24.4 `[read]` |

### I-JEPA (Jan 2023)

The proof of concept. Fully covered in `02_the_mechanics.md`. Code and weights are open at
`facebookresearch/ijepa`.

### V-JEPA (Feb 2024)

The same recipe applied to video: mask spatio-temporal regions, predict their representations. Two
things changed conceptually.

First, **the masking is now over space *and* time**, so the model is forced to represent motion,
not just layout. Second, the evaluation was deliberately **frozen** - no fine-tuning at all, just a
probe on top of fixed features - which is a much harder and much more honest test than the
fine-tuned numbers video papers usually report.

`[search]`: 82.1 on Kinetics-400 and 71.2 on Something-Something-v2 under frozen evaluation,
reported as +4 and +10 points over the previous best video models. Something-Something-v2 is the
interesting one because its classes are defined by *motion* ("pushing something from left to
right"), so a model that only sees appearance cannot do it.

Released via `facebookresearch/jepa` with checkpoints and the training recipe.

### V-JEPA 2 and V-JEPA 2-AC (Jun 2025)

The jump from "good video encoder" to "world model you can plan with". `[read]` from the paper and
Meta's blog:

- **Stage 1, actionless.** 1.2B parameters, over **1 million hours** of internet video plus 1
  million images. No actions, no labels, no rewards - pure observation.
- **Stage 2, action-conditioned.** Freeze what you have, and train a predictor that takes the
  current latent state **plus a robot action** and predicts the next latent state. This used
  **under 62 hours** of unlabelled robot video from the public DROID dataset - four orders of
  magnitude less than the stage-1 corpus.

The result is a model that can answer "if I command this action, what will the world look like?"
entirely in latent space. Planning then falls out: given a **goal image**, encode it, and search
over action sequences for the one whose imagined final latent is closest to the goal latent. The
search is model-predictive control with the cross-entropy method, replanned each step.

Deployed **zero-shot** on Franka arms in two labs Meta had collected no data from, reaching
**65-80%** on pick-and-place with novel objects `[read]`. `[search]` puts the per-action planning
time at around **16 seconds**, which is the number to quote when a student asks whether this could
drive a real robot today.

Meta released three physical-reasoning benchmarks alongside it - **IntPhys 2**, **MVPBench**,
**CausalVQA** - and reported that humans score **85-95%** while models, including their own, sit
at or near chance `[read]`. Publishing a benchmark your own flagship model fails is unusual and
the chapter should say so.

### V-JEPA 2.1 (Mar 2026)

The one that quietly admits a real problem. `[read]`: V-JEPA 2's feature maps were "noisy and
show only fragmented local spatial structure" - fine for global classification, bad for anything
needing *where*. Four changes: a dense predictive loss applied to visible tokens as well as
masked ones, deep self-supervision at intermediate layers, separate 2D/3D patch embeddings, and
scale (VisionMix-163M, ViT-G at 2B).

| Task | V-JEPA 2 (ViT-g) | V-JEPA 2.1 |
|---|---|---|
| NYUv2 depth, RMSE (lower better) | 0.642 | **0.350** |
| ADE20K segmentation, mIoU | 24.4 | **47.8** |
| Something-Something v2 | 77.3 | 76.9 |

Note the last row: the global-motion number did not improve, it moved 0.4 points the wrong way.
The gains are entirely in dense, spatial tasks - which is a clean demonstration that "the
representation is good" is not a single scalar. Against DINOv3 ViT-7B, V-JEPA 2.1 ViT-G reports
0.307 vs 0.309 RMSE on NYUv2 depth and 7.71 vs 5.68 mAP on Ego4D short-term anticipation `[read]`.

### VL-JEPA (Dec 2025)

JEPA applied to vision-language. Instead of autoregressively generating the answer text, it
**predicts the continuous embedding of the target text in one shot**, and only runs a lightweight
text decoder when a human actually needs words.

`[read]`: ~1.6B parameters against 7B-13B for InstructBLIP and Qwen-VL; **65.7%** on
WorldPrediction-WM, a new state of the art over GPT-4o (53.3) and Gemini-2.0 (55.6); selective
decoding cuts decoding operations by **2.85x**. Beats CLIP, SigLIP2 and Perception Encoder on
zero-shot video classification (46.4 vs 44.6 average over eight datasets for the base model).
Stated weakness: slightly worse on appearance-centric tasks, attributed to having seen fewer
image-text pairs.

Yann LeCun and Pascale Fung are both authors, and both are now at AMI Labs. This is the paper to
use if a student asks "does the JEPA idea say anything about language", because it is the first
one where a JEPA-style objective produces a system that competes with frontier VLMs on a
reasoning benchmark rather than just a probe.

## The theory branch: LeJEPA (Nov 2025)

Balestriero and LeCun, arXiv 2511.08544 `[read, via secondary overview]`.

The argument in three steps:

1. **Prove what the embedding distribution should be.** They show the **isotropic Gaussian** is
   the unique distribution minimising downstream prediction risk. So there is now a stated target,
   rather than a folk belief that "spread out is good".
2. **Enforce it cheaply.** Matching a distribution in high dimensions is hard, so **SIGReg**
   (Sketched Isotropic Gaussian Regularization) projects embeddings onto a handful of *random
   1-D directions* and runs a univariate goodness-of-fit test on each, using the Epps-Pulley
   statistic (bounded loss, bounded gradients, O(N)). Resample directions each step and you
   enforce isotropy in expectation.
3. **Delete the heuristics.** With an explicit anti-collapse term you no longer need
   stop-gradient, the EMA teacher, the predictor, register tokens, or asymmetric augmentation.
   The loss reduces to `L = (1 - lambda) * L_pred + lambda * SIGReg` with **one** hyperparameter.

Reported: 50+ architectures across 8 families, 10+ datasets, up to 1.8B parameters; ViT-H/14 hits
**79%** ImageNet linear in **100 epochs**; beats DINOv2/v3 when pretrained directly on
specialised data (Galaxy10, Food101, Flowers102).

The claim that matters most for teaching is a side effect: **the training loss correlates with
downstream accuracy.** In every EMA-based JEPA it does not (the target is moving), which is why
model selection needs an expensive probe. If LeJEPA holds up, that is a bigger practical change
than the accuracy numbers.

Treat it as promising and young. It is one paper, roughly nine months old as of Aug 2026.

## The language branch: LLM-JEPA (Sep 2025)

`[search]`, arXiv 2509.14252. Applies the JEPA objective to LLM training by treating naturally
paired data as two views - a natural-language description and the corresponding code, for example
- and requiring each to predict the other's embedding. Uses a `[PRED]` special token so the LLM's
own weights serve as the predictor, avoiding a separate network.

Reported to beat standard finetuning objectives across Llama-3 and Gemma-2 on NL-RX, GSM8K and
Spider, and to resist overfitting. Two stated limitations: it needs **naturally paired multi-view
data**, which most text is not, and training costs roughly **3x** because each view needs its own
forward pass.

The honest summary for a slide: this is evidence that the objective transfers, not evidence that
JEPA replaces next-token prediction.

## The I-JEPA descendants worth knowing

Each of these exists because it identified one specific defect in I-JEPA, which makes them useful
teaching devices rather than just a list `[read, from the tutorial]`:

| Model | The defect it names | The fix |
|---|---|---|
| **C-JEPA** | EMA does not fully prevent collapse; the predictor does not model the mean patch representation well | Add VICReg's variance / covariance / invariance terms. Same ViT backbone, so the gain is attributable to the regularisation alone. Converges faster and scores higher on ImageNet-1K |
| **StoP-JEPA** | Fixed positional embeddings make the model assume it knows exactly where a masked patch is - but "given part of a dog, you cannot locate its tail precisely" | Model each masked position as a Gaussian random variable with learned covariance, tying the noise projection to the context projection so it cannot collapse back to fixed positions. A few lines of code, no extra compute |
| **CNN-JEPA** | The recipe is entangled with ViTs | Run the same objective on a fully convolutional backbone |
| **MIM-JEPA** | - | Hybrid CNN-ViT |
| **IWM** (Image World Model) | I-JEPA predicts *masked content*, never how a representation changes under an action | Condition the predictor on known transformations (rotation, translation, colour) - latent equivariant dynamics. The conceptual bridge from I-JEPA to V-JEPA 2-AC |
| **D-JEPA** | - | Casts generative modelling as denoising in embedding space |
| **MC-JEPA** | - | Learns optical flow and content features in one shared encoder |

**LeWM (LeWorldModel, 2026)** belongs with these but is covered in `04_world_models_and_planning.md`
- it is the compact, single-GPU JEPA world model, and the only model in this whole chapter a
student could plausibly reproduce.

## The long tail

There are JEPA variants in almost every modality by 2026 `[read, from the awesome-jepa list]`.
Not chapter material, but useful for one "the idea generalises" slide and for students picking
projects:

- **Audio/speech**: A-JEPA (2023), Stem-JEPA (2024), Audio-JEPA (2025), WavJEPA (2025, raw
  waveform)
- **3D**: Point-JEPA (2024), 3D-JEPA (2024), CrossJEPA (2025, predicts 3D from 2D)
- **Graphs/molecules**: Graph-level JEPA (2023), Polymer-JEPA (2025)
- **Time series and tabular**: T-JEPA (tabular, 2024), T-JEPA (trajectory, 2024), LaT-PFN (2024,
  JEPA crossed with prior-fitted networks - a direct link to **ch14**), MTS-JEPA (2026)
- **Biosignals and medical**: S-JEPA (EEG, 2024), Brain-JEPA (2024), ECG-JEPA (2024), RadJEPA
  (2026), EchoJEPA (2026), JEPA-DNA (2026), US-JEPA (2026)
- **Remote sensing**: SAR-JEPA (2023), AnySat (2024), REJEPA (2025), X-JEPA (2026)
- **Robotics and world models**: ACT-JEPA (2025), VLA-JEPA (2026, couples a **ch15** VLA to a
  latent world model), Causal-JEPA (2026), hierarchical latent planning (2026)
- **Generative**: D-JEPA (2024), Diffusion-JEPA (2025), JEPA-T (2025)

The medical and biosignal cluster is the most interesting one for this course's students: those
are exactly the domains with small labelled datasets and large unlabelled ones, which is the
regime where "pretrain without labels or augmentations" pays.

## Who is building this now

`[read]`, TechCrunch and corroborating coverage:

- LeCun **announced his departure from Meta on 19 Nov 2025**, leaving at the end of 2025 after 12
  years as Chief AI Scientist `[search, multiple outlets]`.
- He co-founded **AMI Labs**, headquartered in **Paris** with offices in New York, Montreal and
  Singapore. LeCun is Chairman; CEO **Alexandre LeBrun**; Chief Science Officer **Saining Xie**;
  Chief Research and Innovation Officer **Pascale Fung**; VP of World Models **Michael Rabbat**.
- **March 2026: raised $1.03B at a $3.5B pre-money valuation** - reported as the largest seed
  round ever raised by a European startup. Co-led by Cathay Innovation, Greycroft, Hiro Capital,
  HV Capital and Bezos Expeditions; also NVIDIA, Samsung, Temasek, Toyota Ventures, Mark Cuban,
  Eric Schmidt.
- **No revenue plans.** The CEO's framing: "not your typical applied AI startup" - it "starts with
  fundamental research", commercial applications "could take years". They intend to publish and
  open-source. First disclosed partner: Nabla, a digital health startup.

Meta continues the V-JEPA line independently - V-JEPA 2.1 shipped in March 2026, after LeCun left.
