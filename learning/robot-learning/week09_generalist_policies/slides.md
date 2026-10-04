# Lecture 9: Generalist Robot Policies - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 27.04.2026
- Source: `slides.pdf` in this folder (64 frames, 1280x720, no text layer), re-extracted on 2026-10-04 from the 720p YouTube recording with the slide repo's own `lectures/extract_slides.py` (github.com/idanL1212000/RobotLerning-2026-ETH-Zurich). That repo's `week09_slides.pdf` is only 640x360, so its frame numbering differs from this one.
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/dtofzDY9zuo (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_quan_vuong.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 10: Generalist Robot Policies

Oier Mees - ETH zurich, Microsoft - 27.04.2026

[Note: the slide says "Lecture 10", but the course page and the YouTube title list this as Lecture 9 (week 9, Apr 27).]

[Note: slides 1-3 do not appear in the captured frames; the numbering starts at 4.]

## Slide 4: Foundation Models in NLP (frame 2)

[Figure: **Large Dataset** -> **Foundation Model** -> arrows to Code completion, Question Answering, Translation, ...]

## Slide 5: Foundation Models > Specialist Models (frame 3)

Left, specialist models:

- Translation: Seq2Seq, GNMT... [Figure: unrolled recurrent network]
- Object Detection: R-CNN, Fast-RCNN... [Figure: R-CNN pipeline: 1. Input image, 2. Extract region proposals (~2k), 3. Compute CNN features, 4. Classify regions]
- Image Captioning: DenseCap, Show & Tell.. [Figure: CNN -> region features -> LSTM captioning pipeline]

-> [arrow] **Transformer** [Figure: one block with a row of **Images** tokens and a row of **Language** tokens] Vision-Language Models: ViLBERT, Flamingo, BLIP, GPT-4V...

## Slide 6: Ingredients for Generalist Robot Policies (frames 4-7)

| Large Datasets | Large Models | Scalable Evaluation |
|---|---|---|
| Robot Data is **Scarce** | Robot Data is **Multimodal & Heterogeneous** | Robot Evals are **Tedious & Expensive** |
| Collecting Data requires **Human Supervision** | Robots need **High Frequency Control** | **Difficult to reproduce** |

[Figure: pipeline along the bottom: a dataset cylinder -> **Robot Foundation Model** -> collage of robots deployed in many environments. The three columns appear one per build; same slide as week 1, slide 35.]

[Note: slide 7 does not appear in the captured frames; the numbering jumps from 6 to 8.]

## Slide 8: Key Idea: Leverage Existing Robot Datasets (frames 8-9)

[Figure: a cluster of circles of different sizes, one per existing robot dataset; the large ones are labelled Fractal, Bridge V2, RoboTurk and RoboNet, the small ones are unlabelled.]

[Second build (frame 9) draws one large ellipse around all the circles and adds:] **Aggregate** existing data into **common format**

## Slide 9: Large Robot Data: Open X-Embodiment (frame 10)

[Figure: a banner of the logos of the contributing institutions, e.g. ASU, Carnegie Mellon, Columbia, DLR, ETH zürich, Flexiv, Google DeepMind, Google Research, Intrinsic, Illinois, NYU, Imperial College London, IIT, KAIST, Stanford, Berkeley, Universität Freiburg, UTN, UT Austin, Max Planck Institute for Intelligent Systems, QUT, Shanghai Jiao Tong, UC San Diego, USC, Toyota Research Institute.]

- [robot emoji] **1M+** Real Robot Episodes
- [arm emoji] **22** Robot Embodiments
- [building emoji] **34** Research Labs
- [picture emoji] **300+** Scenes

Reference: *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, Open X-Embodiment Collaboration, ..., **Mees** et al., ICRA, 2024. **Best Conference Paper Award (out of 1765 papers)**

## Slide 10: Sparks of a Generalist Robot Policy (frame 11)

[Figure: a video still of a robot arm at a desk with a drawer and buttons, captioned "4. 'Place the purple block inside the drawer'" (2x), labelled "My PhD robot setup" with an arrow to the AiS column of the chart below.]

[Figure: grouped bar chart of success rate (0-100) per evaluation, comparing the Original Method, RT-1 and RT-1-X:]

| Evaluation (lab) | Original Method | RT-1 | RT-1-X |
|---|---|---|---|
| Kitchen Manipulation (CLVR; original: MVP BC-RNN) | 43 | 48 | 63 |
| Cable Routing (RAIL; original: Resnet + MLP) | 24 | 18 | 56 |
| NYU Door Opening (CILVR; original: VINN) | 53 | 65 | 80 |
| Autolab UR5 (AUTOLAB) | 53 | 25 | 45 |
| Task-Agnostic Play (AiS; original: TACORL, HULC2) | 33 | 68 | 72 |
| Mean ("Multi-robot & Multi-dataset") | 41 | 44 | 63 |

[Note: the colour legend has four original methods (MVP BC-RNN, Resnet + MLP, VINN, TACORL/HULC2) for five tasks; the Autolab UR5 original method is not named.]

Reference: *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, Open X-Embodiment Collaboration, ..., **Mees** et al., ICRA, 2024. **Best Conference Paper Award (out of 1765 papers)**

[Note: slide 11 does not appear in the captured frames; the numbering jumps from 10 to 12.]

## Slide 12: Ingredients for Generalist Robot Policies [Large Models] (frame 12)

[The slide 6 table, with the first column now reading "Leverage **Existing Robot Data**" over a thumbnail of the Open X-Embodiment summary figure (1M Episodes from 311 Scenes, 34 Research Labs across 21 Institutions, 22 Embodiments, 527 Skills, 60 Datasets). The **Large Models** column is outlined: Robot Data is **Multimodal & Heterogeneous**; Robots need **High Frequency Control**.]

## [untitled: Generalist Robot Policy - plug-in inputs and outputs] (frames 13-16)

[Figure: a jigsaw-puzzle diagram. A central piece, **Generalist Robot Policy**, has input slots on the left and an output slot on the right. Build 1 (frame 13): inputs **Language Instruction** (a person speaking) and **Wrist & 3rd Person Camera**; output **End-Effector Control** (a gripper with arrows); a heading being typed in, "N". Build 2 (frame 14), heading "**New Observation**": the language piece is swapped for a **Goal Image**. Build 3 (frame 15), heading "**New Action Space**": the camera piece is swapped for **Proprio** and the output for **Joint Control** (a robot arm with arrows at the joints). Build 4 (frame 16): a box overlays the diagram:]

- Robot Data is **Heterogeneous**
- **Diverse** Sensors
- **Diverse** Actuators
- **Diverse** Control Frequencies

[These four frames show no slide number.]

## Slide 14: [untitled: policy inputs and outputs] (frame 17)

[Figure: a camera image of a tabletop (towel, spoon, toy watermelon slice) and the instruction "Pick up the spoon" both feed a **Policy** box, which outputs ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$ (picture of a robot arm). Same figure as week 1, slide 30.]

[Note: slides 13 and 15 do not appear in the captured frames; the numbering jumps from 12 to 14 and from 14 to 16.]

## Slide 16: [untitled: vision-language model inputs and outputs] (frame 18)

[Figure: a photo of the Statue of Liberty and the New York skyline plus the prompt "Caption the scene" feed a **Vision-Language Model** box, which outputs "The picture shows the Statue of Liberty in NY". Same figure as week 1, slide 32.]

## Slide 17: Key: Robotics as Multimodal Sequence Modeling (frame 19)

[Figure: three groups of tokens, **Language** (green), **Image** (blue), **Action** (purple), above the instruction "Pick up the spoon", the tabletop camera image, and "ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$". Same figure as week 1, slide 33.]

[Note: slide 18 does not appear in the captured frames; the numbering jumps from 17 to 19.]

## Slide 19: [octopus emoji] Octo Architecture (frames 20-21)

[Figure: two tokenizer panels from the Octo paper. **Task Tokens**: the instruction "Put the knife on the plate" -> **Language Encoder** -> tokens, plus position embeddings $p$ -> task tokens $\mathcal{T}_T$. **Observation Tokens**: a camera image cut into patches -> **CNN** -> tokens, plus position embeddings $p$ -> observation tokens $\mathcal{T}_o$. Frame 21 moves the panels to the left edge to make room for the next build.]

Reference: *Octo: An Open-Source Generalist Policy*, Octo Model Team, Ghosh\*, Walke\*, Pertsch\*, Black\*, **Mees\*** et al., RSS, 2024.

## Slide 20: [octopus emoji] Octo Architecture [transformer] (frame 22)

[Figure: the task and observation tokenizer panels on the left; the task tokens (green) and observation tokens (blue) enter the **Octo Transformer**, "~ (ViT)".]

Reference: *Octo: An Open-Source Generalist Policy*, Octo Model Team, Ghosh\*, Walke\*, Pertsch\*, Black\*, **Mees\*** et al., RSS, 2024.

## Slide 21: [octopus emoji] Octo Architecture [readout and action head] (frame 23)

[Figure: the Octo Transformer now takes **Task**, **Observation** and a **Readout** token; the readout output goes into an **Action Head** -> a.]

$$\pi_\theta(a_t \mid s_t, s_g)$$

- [arrow at $s_g$] goal image, language
- Goal state conditioned BC

Reference: *Octo: An Open-Source Generalist Policy*, Octo Model Team, Ghosh\*, Walke\*, Pertsch\*, Black\*, **Mees\*** et al., RSS, 2024.

## Slide 22: [octopus emoji] Octo Architecture [diffusion action head] (frame 24)

[Figure: the same transformer, with further faded blocks of observation and readout tokens to the right (later time steps); the readout token now feeds a **Diffusion Action Head** -> a.]

Reference: *Octo: An Open-Source Generalist Policy*, Octo Model Team, Ghosh\*, Walke\*, Pertsch\*, Black\*, **Mees\*** et al., RSS, 2024.

## Slide 23: [octopus emoji] Octo Design Decisions (frame 25)

- Train everything from scratch, most params in Transformer
- Align gripper action across OXE datasets

[Figure: "Gripper action conventions in OXE - Typical motion (open -> closed to grasp -> open to release), four dataset conventions", each with a small step plot:]

| Convention | Datasets |
|---|---|
| Absolute, +1 open / 0 closed | Taco, Austin Sailor, Austin Sirius, NYU Franka Play |
| Absolute, 0 open / 1 closed (inverted) | Roboturk, Viola, Stanford Hydra, BC-Z |
| Relative deltas, +1 opening / -1 closing | RT-1, Kuka, Jaco Play, Berkeley AutoLab UR5 |
| Continuous with transition ramps | Bridge - intermediate values during opening/closing (orange points: analog values during transitions) |

Octo aligns all four to a single convention. Target: absolute, +1 open / 0 closed

[Note: slide 24 does not appear in the captured frames; the numbering jumps from 23 to 25.]

## Slide 25: The Eureka Moment (frames 26-28)

[Figure: an embedded phone video of a laptop showing a video call ("Octo on Meta Hacking"): one participant in a lab, a robot arm at a table with fruit (the robot running Octo), and Oier Mees with two colleagues reacting and giving a thumbs-up.]

## Slide 26: [untitled: transition question] (frame 29)

**How To Adapt To New Obs/Action Spaces?**

## Slide 27: [octopus emoji] Octo Finetuning (frame 30)

**Adaptable** pre-trained representation for **finetuning**

[Figure: the task and observation tokenizers on the left. Top, "Pre-Training": the Octo Transformer over Task, Observation, Readout, Observation, Readout, Observation tokens, each readout feeding an Action Head -> a. Bottom, "Finetuning": the same transformer with new observation token blocks ("New Observation") and a new readout ("New Action Space") feeding a "New Action Head" -> a; the original task tokens are faded.]

Reference: *Octo: An Open-Source Generalist Policy*, Octo Model Team, Ghosh\*, Walke\*, Pertsch\*, Black\*, **Mees\*** et al., RSS, 2024.

## Slide 28: Adoption of Octo (frame 31)

- Key: RFMs learn **better representations** for **transfer**
- Researchers finetune to their robot with 50 demos

```python
from octo import OctoModel

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
```

[Figure: two community examples. Left, a Spot quadruped robot with an arm next to a vacuum cleaner, "Source: Peter Mitrano via Twitter". Right, a humanoid upper body placing a green bell pepper on a plate ("Both videos are playing at 4x"), "Source: Tokyo Robotics".]

## Slide 29: Cross-Embodiment Learning (frame 32)

- How "generalist" is a single arm manipulation policy?
- Can we **learn from more robot embodiments**?
- **Transfer** knowledge across embodiments

[Figure: jigsaw diagram: **Large Robot Dataset** -> **Generalist Robot Policy** -> output pieces Quadrupeds, Single Arms, Navigation, Quadcopters, each with small robot icons.]

## Slide 30: One Policy for Manipulation, Navigation, Locomotion & Aviation (frames 33-34)

[Figure: a collage of robot-data thumbnails under a box: [robot emoji] 900k Robot Trajectories - Navigation, Locomotion, Manipulation, Bimanual. Second build (frame 34): an arrow into the **Cross-Embodied Transformer** box - Flexible Observation Spaces; Flexible Action Spaces $\in [2, \dots, 1400]$; Control Freq. $\in [5\text{Hz}, \dots, 50\text{Hz}]$ - which fans out to Quadrupeds, Single Arms, Navigation, Bimanual Arms and Quadcopters (robot icons).]

Reference: *Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation*, Doshi\*, Walke\*, **Mees**, Dasari and Levine. CoRL 2024. **Oral, top %4 of 670 papers**.

## Slide 31: CrossFormer Model (frame 35)

- No action space alignment required
- Can consume data from *any* robot embodiment
- Maximize parameter sharing across embodiments

[Figure: left, "Observation Image Tokenization": the instruction "Sweep the objects into the dustpan" goes through a Language Encoder and conditions (FiLM Conditioning) a **ResNet** that encodes the Current Image and Goal Image into tokens $\mathcal{T}_o$ with position embeddings $p$. Right, the **Cross-Embodied Transformer** over Observation Tokens (Workspace Image, Navigation Image, Wrist Image, Quadruped Proprio, Bimanual Proprio) and Readout Tokens; separate readouts feed a Quadruped Action Head, Single Arm Action Head, Navigation Action Head and Bimanual Action Head, each with its robots.]

[Note: slide 32 does not appear in the captured frames; the numbering jumps from 31 to 33.]

## Slide 33: Quantitative Results (frame 36)

Key: matches and **outperforms specialist policies**

[Figure: grouped bar chart of success rate (0-1) for the Best Prior Method, a Single-Robot Dataset policy and CrossFormer. Values read from the bars, approximate:]

| | Best Prior Method | Single-Robot Dataset | CrossFormer |
|---|---|---|---|
| Average (highlighted) | ~0.52 | ~0.62 | ~0.70 |
| WidowX | ~0.21 | ~0.21 | ~0.25 |
| Franka | ~0.52 | ~0.41 | ~0.41 |
| ALOHA | ~0.70 | ~0.50 | ~0.80 |
| LoCoBot | ~0.48 | ~0.92 | ~0.93 |
| Go1 | [no bar] | ~1.0 | ~1.0 |
| TELLO | ~0.68 | ~0.68 | ~0.82 |

[Note: slide 34 does not appear in the captured frames; the numbering jumps from 33 to 35.]

## Slide 35: [untitled: transition question] (frame 37)

Everything so far trains on robot data **from scratch**

Do **internet-scale priors like VLMs transfer to control?**

## Slide 36: Recap Llava (frame 38)

- Prepends image tokens to the text sequence, LLM processes everything in single unified self-attention pass

[Figure: the LLaVA pipeline from week 7, slide 23: image input -> **CLIP vision encoder** (always frozen; patch embeddings) -> **MLP connector** (projection layer; trainable) -> 196 image tokens prepended to the text tokens from the text prompt ("describe this") -> **LLM (LLaMA)**: self-attention over all tokens jointly, standard autoregressive generation, no new attention mechanisms -> text output "a cat sitting on a mat".]

Reference: *Visual Instruction Tuning*, Liu et al. (2023)

## Slide 37: [untitled: transition question] (frame 39)

LLaVA treats images as another token sequence the LLM can attend to

**Can we treat robot actions the same way?**

## Slide 38: VLM -> Vision-Language-Action Model (VLA) (frame 40)

| Pretrained VLM | -> Fine-tune on robot data -> | Fine-tuned into VLA |
|---|---|---|
| Image (RGB pixels) -> Vision (ViT / CLIP) | | Image (Robot camera) -> Vision (Same encoder) |
| Text (Instruction) -> Tokenizer (Subword IDs) | | Text (Task command) -> Tokenizer (Same vocab) |
| LLM transformer (Autoregressive decoder) | | LLM transformer (Same weights, fine-tuned) |
| Language tokens (Vocab of ~32k IDs) | | Action tokens (256 reused vocab IDs) |
| Output: natural language answer, e.g. "The cat is on the mat." | | "1 128 91 241 5 101 127" de-tokenize -> $[\Delta x, \Delta y, \Delta z, \Delta\theta, \text{grip}]$ |

[Note: the example token string has 7 numbers but the de-tokenized action lists 5 values; a typical 7-DoF action would add $\Delta$ roll/pitch or similar.]

## Slide 39: Recap: Robot Action Tokenization for VLAs (frame 41)

[Same content as week 7, slide 29 (per-dimension, per-timestep binning): quantile normalization per dimension for robot A ($Q_1 = -0.02$, $Q_{99} = +0.04$) and robot B ($Q_1 = -1.50$, $Q_{99} = +2.30$) to a shared $[-1, +1]$ range; 256 equal bins of width $2/256 \approx 0.0078$; a 7-dim action $a_t \in \mathbb{R}^7$ becomes 7 tokens (183, 107, 240, 018, 155, 072, 201); the 256 action tokens are injected into the language-model vocabulary and trained with next-token prediction.]

References: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, Brohan et al. (2023); *OpenVLA: An Open-Source Vision-Language-Action Model*, Kim et al. (2024)

## Slide 40: Tips & Tricks: Convergence (frame 42)

- How do I know my VLA training has converged?

[Figure: three training curves against training step (0-100k+). **Action token accuracy** rises from about 0.05 and flattens near the dashed "95% target" after about 75k steps. **L1 action error** falls from about 0.5 to about 0.09. **L2 action error** falls from about 0.85 to about 0.1.]

- [arrow at the accuracy plot] Did we pick the exact action bin?
- [brace under the two error plots] How far off are the predicted continuous actions (after detokenization) from ground-truth actions?

## Slide 41: Tips & Tricks: Batching Heterogeneous Datasets [padding] (frame 43)

- Datasets in OXE have different observation spaces
- Padding & Masking:
  - Easiest to implement
  - Half your batch will be full of zeros, wasted FLOPS

[Figure: Dataset A: 1 third-person camera; Dataset B: 1 third-person + 2 wrist cameras. A batch table with columns 3rd-person Cam, Wrist L Cam, Wrist R Cam, Lang + actions: samples 1-2 (dataset A) have "padding" in both wrist columns; samples 3-4 (dataset B) fill every column.]

## Slide 42: Tips & Tricks: Batching Heterogeneous Datasets [sequence packing] (frame 44)

- Datasets in OXE have different observation spaces
- Sequence Packing:
  - Maximize GPU Throughput
  - Impl Complexity: reset positional encodings, block-diagonal att.

[Figure: samples 1-2 (A) are "img, lang+act"; samples 3-4 (B) are "img, wrist L, wrist R, lang+act". They are concatenated into one "Packed view (one sequence)": S1, S2, S3, S4 back to back, with `cu_seqlens = [0, 8, 16, 32, 48]` "<- prefix sums of sample lengths".]

## Slide 43: Tips & Tricks: Batching Heterogeneous Datasets [masking and positions] (frame 45)

1. Attention must be block-diagonal
   - Tokens in sample i can only attend to other tokens in sample i. Without this mask, sample 1 would attend to sample 3's images.
   - [Figure: a queries x keys grid over S1-S4 where only the diagonal blocks S1↔S1, S2↔S2, S3↔S3, S4↔S4 are "Attention allowed"; everything else is "Masked out".]
2. Positional encodings reset at each boundary
   - Token 0 of every sample gets position 0 - not its absolute offset in the packed sequence. Otherwise sample 4 would see "position 24+" as out-of-distribution input.
   - [Figure: position ids 0 1 | 0 1 | 0 1 2 3 | 0 1 2 3, "positions reset per sample".]

## Slide 44: Tips & Tricks: Dataloading (frame 46)

[Figure: two loaders. "Sequential reads + shuffle buffer" (Octo, OpenVLA): shards 1-3 are streamed into a shuffle buffer that batches are drawn from; "too small -> correlated batches". "True random reads" (index-based samplers): random seeks into a flat index $[0 \dots N)$ to build a batch.]

| | Sequential + buffer | True random reads |
|---|---|---|
| I/O pattern | sequential reads (advantage) | random seeks (disadvantage) - seek-bound |
| throughput | very high (advantage) | lower (disadvantage) - bottleneck at scale |
| randomness | approximate (disadvantage) | exact (advantage) - no buffer bias |
| memory | large buffer in RAM (disadvantage) | index only (advantage) - lightweight |
| obs. history | free - adjacent (advantage) | extra seeks (disadvantage) - cost x window size |

## Slide 45: Tips & Tricks: Cross-Embodiment Heads (frame 47)

| CrossFormer-style | $\pi_0$-style |
|---|---|
| Observations (images + language + proprio) -> Shared transformer backbone -> separate heads: Single-arm, Bimanual, Quadruped, Drone, each with its own action length | Observations (images + language + proprio) -> Shared transformer backbone -> one **Action expert** with a **padded action space**: single-arm, bimanual, quadruped and drone actions all fill the front of one fixed-length vector, the rest padded |

## Slide 46: [untitled: transition question] (frame 48)

**How does the VLA know which embodiment to produce actions for at test time?**

## Slide 47: Cross-Embodiment Heads (frame 49)

| CrossFormer-style | $\pi_0$-style |
|---|---|
| User specifies head in prompt | Model infers embodiment from obs, proprio & task |
| [Figure: the CrossFormer-style diagram from slide 45: shared backbone, one head per embodiment] | [Figure: the $\pi_0$-style diagram from slide 45: shared backbone, one action expert with a padded action space] |

## Slide 48: Current VLA Recipe (frame 50)

- Next token prediction for VLM: FAST robot actions + Web
- Single action expert: flow matching + stop gradient to VLM

[Figure: **VLM backbone** over image tokens (Images: cameras + web), language tokens (NTP loss) and FAST action tokens (Robot actions (FAST), NTP loss). Its output goes, across a "stop gradient" barrier, into an **Action expert** trained with a **flow matching loss**, which outputs continuous actions (e.g. -1.7, 1.25, 3.14, 1.42). Legend: "stop gradient - action expert cannot update backbone".]

References: *$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization Model*, Physical Intelligence (2025); *Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*, Driess et al., Physical Intelligence (2025)

## Slide 49: Ingredients for Generalist Robot Policies [Scalable Evaluation] (frame 51)

[The slide 12 table, with the **Large Models** column now reading "**Cross-embodied** Policies" over the Octo emoji and a robot collage. The **Scalable Evaluation** column still reads: Robot Evals are **Tedious & Expensive**; **Difficult to reproduce**.]

## Slide 50: Evaluating Policies is Expensive (frames 52-53)

- Sim evals **democratize research**
- Sim evals are **more reproducable**

[Figure: three video clips captioned "... people bump into cameras," (a person's arm knocking a camera next to a robot), "grippers get stuck," (a gripper stuck on a toy eggplant in a toy sink) and "and real eval is slow and tedious." (a robot at a toy sink, played at 100x). Frame 52 is an earlier build with only the first clip and no bullets.]

## Slide 51: Evaluating Real-World Policies in Simulation (frames 54-55)

- Sim evals are **more reproducable**

... and make them **SIMPLER** (real -> sim)

[Figure: simulated replicas of real evaluation scenes. Google Robot: a counter with a can, an apple and a sponge; a coke can; a cabinet with an open drawer; a cabinet with an apple on top. BridgeData V2: a carrot and a plate; a towel and a spoon; two cubes; a toy sink with an eggplant. Second build (frame 55): a large grid of simulated rollouts with the caption "SIMPLER enables effortless and reproducible eval at scale."]

Reference: *Evaluating Real-World Robot Manipulation Policies in Simulation Policy*, Li\*, Hsu\*, Gu\*, Pertsch†, **Mees†** et al., CoRL, 2024.

## Slide 52: Evaluating Real-World Policies in Simulation [real vs sim correlation] (frame 56)

**OK, but how meaningful are SIMPLER results?**

[Figure: scatter plot of SIMPLER success rate against real success rate for RT-1, RT-1-X, RT-2-X and Octo; the points lie close to a dashed diagonal, with Octo points scattered a little more widely.]

How correlated are real and SIMPLER performance measures?

**Very!**

SIMPLER is a reliable proxy for real robot evaluation.

Reference: *Evaluating Real-World Robot Manipulation Policies in Simulation Policy*, Li\*, Hsu\*, Gu\*, Pertsch†, **Mees†** et al., CoRL, 2024.

[Note: the paper's title is "Evaluating Real-World Robot Manipulation Policies in Simulation"; the trailing word "Policy" in the slide's citation looks like a slip.]

## Slide 53: Evaluating Real-World Policies in Simulation [robustness and what matters] (frames 57-59)

**OK, but how meaningful are SIMPLER results?**

SIMPLER can accurately characterize an RT-1 policy's robustness to various distribution shifts.

| Distribution shift | $\Delta$ Real success rate | $\Delta$ SIMPLER success rate |
|---|---|---|
| Camera pose | -0.38 | -0.39 |
| Table texture | -0.17 | -0.19 |
| Background | -0.17 | -0.12 |
| Distractors | -0.08 | -0.06 |
| Lighting | -0.04 | -0.07 |

[Frames 58-59 replace the chart:] **So, what matters for good simulated evaluation?**

1. Accurate control dynamics via system identification (SysID)
   - Same open-loop action sequence: [Figure: three stills of a robot arm and a coke can, captioned Real, Sim w/o SysID (the can ends up somewhere else), Sim after SysID (ours) (matches the real outcome).]

Reference: *Evaluating Real-World Robot Manipulation Policies in Simulation Policy*, Li\*, Hsu\*, Gu\*, Pertsch†, **Mees†** et al., CoRL, 2024.

## Slide 54: Evaluating Real-World Policies in Simulation [visual matching] (frame 60)

**So, what matters for good simulated evaluation?**

2. Mitigating visual distribution shifts via "Visual Matching"

[Figure: two stills of a robot arm over a counter with an orange and two cans, captioned Real and Sim + Green Screen + Texture Matching; the two look nearly identical.]

Reference: *Evaluating Real-World Robot Manipulation Policies in Simulation Policy*, Li\*, Hsu\*, Gu\*, Pertsch†, **Mees†** et al., CoRL, 2024.

## Slide 55: Follow Up Works on Real-to-Sim Evals (frame 61)

[Figure: overview figures from three papers.]

- *PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies*, Jain et al., 2025. [Figure: 1. Tools for Scalable Real-to-Sim Environment Generation (a short video of a real scene -> PolaRiS Scene Builder -> simulated evaluation environment); 2. Simulation Dataset for Bridging Real-to-Sim Gap (short sim data cotraining turns a generalist policy into a PolaRiS-ready policy); 3. Evaluation Environments with Strong Real-to-Sim Correlation (scatter of sim against real performance); 4. Hub for Environment Sharing.]
- *RobotArena: Scalable Robot Benchmarking via Real-to-Sim Translation*, Jangir et al., 2025. [Figure: robotic datasets (DROID, Bridge V2, RH20T) and user-recorded videos are turned into "Large Scale Simulated Environments - Automated Environment Creation from Real Video for VLA Evaluation" (RobotArena ∞), with automatic policy evaluation by VLM scores and human preference evaluation by pairwise comparison and policy ranking.]
- *RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies*, Yang et al., 2026. [Figure: "RoboLab Benchmarking Framework": policies $\pi_1 \dots \pi_N$; Scene Generation, Task Generation (e.g. "Put the __ in the blue bin") and Environment Generation, each done by a user or an LLM.]

## Slide 56: Ingredients for Generalist Robot Policies [summary] (frames 62-63)

| Large Datasets | Large Models | Scalable Evaluation |
|---|---|---|
| Leverage **Existing Robot Data** | **Cross-embodied** Policies with **Internet-scale Priors** | Leverage **Real2Sim Evals** |
| [the Open X-Embodiment summary figure] | [octopus emoji, a robot collage, and a $\pi$ symbol] | [the SIMPLER Google Robot / BridgeData V2 scenes; frame 63 swaps in the grid "SIMPLER enables effortless and reproducible eval at scale."] |

## End (frame 64)

Thank you for your attention
