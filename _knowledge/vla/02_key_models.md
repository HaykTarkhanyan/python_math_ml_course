# 02 - The models that matter, with dates

Everything below was checked by web search on 2026-08-07. Parameter counts and dates are from the
primary paper or the releasing lab's own page wherever possible; where only a secondary source
was available it is flagged.

## The table

| Model | Who | When | Size | Action head | Open weights | What was new |
|---|---|---|---|---|---|---|
| **RT-1** | Google Brain / Everyday Robots | Dec 2022 | 35M | discrete tokens | yes | Proved one transformer could do 700+ instructions on real hardware |
| **RT-2** | Google DeepMind | Jul 2023 | 12B (PaLM-E) / 55B (PaLI-X) | discrete tokens in the LM vocabulary | no | Coined the VLA paradigm: fine-tune a *web-pretrained* VLM to emit actions |
| **Open X-Embodiment / RT-X** | 21 institutions, DeepMind-led | Oct 2023 | - (dataset + RT-1-X, RT-2-X) | - | dataset yes | Pooled 60 datasets into 1M+ trajectories, 22 embodiments; showed positive cross-robot transfer |
| **Octo** | UC Berkeley, Stanford, CMU, DeepMind | May 2024 | 27M / 93M | diffusion | yes, fully (data + ckpt + pipeline) | First fully open generalist policy; small, flexible observation/action spaces |
| **OpenVLA** | Stanford, Berkeley, DeepMind, TRI | Jun 2024 | 7B | discrete tokens | yes, MIT | Beat RT-2-X (55B) by 16.5 points absolute with 7x fewer params; became the field's default baseline |
| **pi-0** | Physical Intelligence | Oct/Nov 2024 | 3.3B (3B PaliGemma + 300M action expert) | **flow matching** | yes, Apache-2.0 via openpi | Continuous 50 Hz action chunks; first credible commercial generalist policy |
| **Helix** | Figure AI | Feb 2025 | 7B (S2) + 80M (S1) | continuous, dual-system | no | 200 Hz full upper-body humanoid control off one latent vector |
| **Gemini Robotics 1.0 / -ER** | Google DeepMind | Mar 2025 | undisclosed (Gemini 2.0 based) | undisclosed | no | Frontier VLM used as the backbone; -ER split off as a reasoning-only model |
| **GR00T N1** | NVIDIA | Mar 2025 | 2.2B (1.34B VLM) | diffusion transformer | yes, Apache-2.0 | Open humanoid foundation model, dual-system, Eagle-2 backbone |
| **pi-0-FAST / FAST** | Physical Intelligence | Jan 2025 | as pi-0 | DCT+BPE tokens | yes | Made autoregressive VLAs viable at high frequency (~10x compression) |
| **pi-0.5** | Physical Intelligence | Apr 2025 | 3B-class + 300M expert | hierarchical: discrete subtask then flow | yes, Apache-2.0 | Trained across ~100 homes; the open-world generalization claim |
| **OpenVLA-OFT** | Stanford et al. | Feb 2025 | 7B | parallel decoding + L1 regression | yes | 26x faster than OpenVLA, 97.1% LIBERO; showed diffusion may not be necessary |
| **SmolVLA** | Hugging Face / LeRobot | Jun 2025 | 450M | flow matching | yes | Consumer-hardware VLA trained on 487 *community* datasets |
| **Gemini Robotics 1.5 / -ER 1.5** | Google DeepMind | Sep 2025 | undisclosed | undisclosed | ER via API only | "Thinking before acting"; Motion Transfer across embodiments |
| **pi\*0.6** | Physical Intelligence | Nov 2025 | 5B backbone | flow | no (as of Aug 2026) | RECAP: RL from the robot's own experience plus expert corrections |
| **Dream-VLA** | HKU + Huawei | Dec 2025 | 7B | **diffusion LLM backbone** | yes, Apache-2.0 | Whole backbone is a diffusion language model; one denoising step, 27x speedup |
| **Gemini Robotics-ER 1.6** | Google DeepMind | Apr 2026 | undisclosed | reasoning only | API | Gauge/sight-glass reading, built with Boston Dynamics |
| **LingBot-VLA 2.0** | Robbyant (Ant Group) | Jul 2026 | 6B (Qwen3-VL-4B backbone) | MoE action expert | yes, Apache-2.0 | 55-dim whole-body canonical action space; 60k hours incl. 10k h human video |
| **Gemini Robotics 2 / ER 2 / On-Device 2** | Google DeepMind | Jul 2026 | undisclosed | undisclosed | no (ER 2 via API) | Whole-body humanoid control, 22-DoF five-fingered hands, multi-robot collaboration |

## The prose

### RT-1 (December 2022) - the proof that scale works on real robots

[RT-1](https://arxiv.org/abs/2212.06817) is pre-VLA in the strict sense: no web-pretrained
backbone, just a FiLM-conditioned EfficientNet with TokenLearner, 35M parameters, running at 3 Hz.
What it established was the data story. 130k episodes covering 700+ instructions, collected by
**13 robots over 17 months** of teleoperation - that "17 months" is the number to put on a slide,
because it is the honest price of a robot dataset. Reported 97% success on the training
instructions and 76% on new tasks, 24 points above the best baseline.

Everything after RT-1 is an attempt to avoid ever having to spend 17 months again.

### RT-2 (July 2023) - the actual invention

[RT-2](https://arxiv.org/abs/2307.15818) is where the paradigm starts. Take a web-scale VLM
(PaLI-X at 55B, or PaLM-E at 12B), and fine-tune it on robot trajectories where the *action is
written as a string of numbers in the model's own vocabulary*. That is the whole trick. The model
does not know it is doing robotics; it is still doing next-token prediction.

The payoff was emergent semantic generalization: the robot could act on concepts that never
appeared in robot data but did appear on the web. This is the demo-versus-measurement moment to
be careful about - the compelling evidence was qualitative ("pick up the extinct animal") and the
quantitative claim was roughly 3x improvement on emergent-skill evaluations over RT-1.

Never open. A 55B model at 3 Hz was also never deployable, which is precisely the gap OpenVLA
walked into.

### Open X-Embodiment (October 2023) - the field's ImageNet moment, sort of

[Open X-Embodiment](https://arxiv.org/abs/2310.08864) pooled **60 existing datasets from 34 labs
across 21 institutions** into 1M+ real robot trajectories, 22 embodiments, 527 skills. RT-1-X and
RT-2-X trained on it showed positive transfer: RT-1-X about 50% higher mean success than the
single-robot baselines it was compared against; RT-2-X roughly 3x on emergent skills.

OXE is the reason the open field exists at all. It is also, as `03` argues, much smaller and much
messier than the ImageNet analogy suggests.

### Octo (May 2024) - fully open, and small

[Octo](https://octo-models.github.io/) is 27M or 93M parameters, trained on 800k OXE trajectories,
with a diffusion head for continuous multimodal actions. Berkeley/Stanford/CMU/DeepMind. Its
contribution was *completeness of release* - data, checkpoints, and training pipeline all open -
plus flexible conditioning: language instruction or goal image, arbitrary camera configurations,
and cheap fine-tuning to new action spaces.

Strictly, Octo is not a VLA under the narrow definition: the backbone was not internet-pretrained.
It is the strongest "trained from scratch on robot data" baseline, which makes it exactly the
right control experiment for the question of whether VLM pretraining actually pays.

### OpenVLA (June 2024) - the default baseline

[OpenVLA](https://arxiv.org/abs/2406.09246) is the model most academic papers still compare
against. 7B parameters: a Llama-2 language model with a *fused* visual encoder combining
**DINOv2 ViT-L/14 and SigLIP ViT-So400M/14** (the fusion matters - DINOv2 contributes spatial
features that SigLIP alone lacks). Trained on 970k episodes from OXE on 64 A100s for 15 days.
Output is a 7-DoF end-effector delta, tokenized into 256 bins that overwrite the least-used Llama
tokens.

Headline result: **+16.5 points absolute** over RT-2-X (55B) across 29 tasks and multiple
embodiments, with 7x fewer parameters. MIT licensed. It also demonstrated that LoRA fine-tuning
and 4-bit quantization put a generalist VLA on a single consumer GPU, which is what made VLAs a
university-lab topic rather than a frontier-lab topic.

[OpenVLA-OFT](https://arxiv.org/abs/2502.19645) (Feb 2025) is the same model with a better recipe:
parallel decoding, action chunking, continuous actions, L1 regression. 26x faster generation, 3x
lower latency, 97.1% average on LIBERO. Note the implication - the improvement came from the
*decoding scheme*, not the backbone.

### pi-0 and the Physical Intelligence line (Oct 2024 onwards)

[pi-0](https://arxiv.org/abs/2410.24164) is the architecture most 2025-2026 work descends from.
PaliGemma 3B backbone plus a **300M "action expert"** trained with flow matching, 3.3B total, 10
integration steps at inference, action chunk H=50, up to 50 Hz. Training data: roughly **10,000
hours**, of which 903M timesteps came from Physical Intelligence's own fleet (68 tasks, 7 robot
configurations), the rest from OXE, Bridge v2 and DROID.

The design insight worth teaching: the VLM is frozen-ish and generic, the action expert is small
and specialized, and they are trained jointly but with the action expert carrying the burden of
continuous high-frequency output. This is the "big slow brain, small fast hands" pattern that
recurs everywhere.

[pi-0.5](https://www.pi.website/blog/pi05) (April 2025) added the hierarchy: predict a
natural-language subtask autoregressively, then flow-match the motor chunk. Trained across
**roughly 100 distinct homes**, plus web VQA/captioning/detection data and verbal coaching data.
The company-reported numbers on unseen homes are a 94% out-of-distribution follow rate and
success rate against 83% in-distribution success. Treat those as company-reported: they come from
a blog post, the evaluation protocol is not independently reproducible, and the "OOD higher than
ID" pattern should make you want to see the task breakdown. The ablation is the more interesting
claim - **web data mattered most for generalizing to novel objects**, which is the cleanest
published support for the entire VLA thesis.

[pi\*0.6](https://www.pi.website/blog/pistar06) (November 2025) is the RL turn. **RECAP** = RL with
Experience and Corrections via Advantage-conditioned Policies: demonstrations, then real-time
expert corrections when the robot errs, then RL on the robot's own autonomous trials using a
learned value function. Company-reported: espresso throughput from ~15 to >30 successes per hour,
failure rates cut 2x or more, >90% success per stage. The demos (18 hours of continuous espresso
service, 50 novel laundry items in an unseen home, 59 factory boxes) are demos - impressive, and
not a benchmark. But the underlying claim is the important one for a course: **imitation learning
alone plateaus, because the model never sees its own mistakes**, and this is the standard argument
for RL that ch11 already sets up.

openpi ships pi-0, pi-0-FAST and pi-0.5 under Apache-2.0. Inference needs >8 GB (an RTX 4090),
LoRA fine-tuning >22.5 GB, full fine-tuning >70 GB
([openpi](https://github.com/Physical-Intelligence/openpi)). pi\*0.6 is not released.

### Helix (February 2025) - the dual-system reference design

[Figure's Helix](https://www.figure.ai/news/helix) is the cleanest statement of the two-timescale
architecture. **S2** is a 7B open-weight VLM running at 7-9 Hz that compresses everything
semantic into a *single continuous latent vector*. **S1** is an 80M visuomotor policy at 200 Hz
that takes that vector plus fresh images and outputs full upper-body control including individual
fingers, wrist, torso and head. Trained on roughly 500 hours of teleoperation - small, which is
the point: the semantics come from S2's pretraining, the dexterity from S1's speed.

Closed. Everything public about Helix is a company blog post and videos; there are no benchmark
numbers to cite. Teach the architecture, not the claims.

### NVIDIA GR00T N1 (March 2025) and after

[GR00T N1](https://arxiv.org/abs/2503.14734) is the open dual-system model: 2.2B total with a
1.34B Eagle-2 VLM as System 2 and a diffusion transformer as System 1, Apache-2.0, checkpoints
plus training data plus simulation benchmarks released. 63.9 ms to sample a 16-action chunk on an
L40 in bf16. Trained on a mix of real robot data, Isaac Sim synthetic data, and internet video -
NVIDIA's whole strategic bet is that simulation closes the data gap, so their models are the ones
to look at when discussing sim-to-real.

N1.5 followed as an update. In June 2026 NVIDIA announced an [open **reference humanoid**
hardware design](https://nvidianews.nvidia.com/news/nvidia-open-humanoid-robot-reference-design)
(Unitree H2 Plus body, Sharpa hands, Jetson Thor compute, GR00T software) aimed at academic labs
including Ai2, ETH Zurich, Stanford Robotics Center and UCSD. That is a notable move: the
bottleneck is data, and standardizing the hardware is how you pool it.

### SmolVLA (June 2025) - the one you can actually run in a class

[SmolVLA](https://huggingface.co/blog/smolvla) is 450M parameters, flow-matching head, trained on
10M frames curated from **487 community datasets** tagged `lerobot` on the Hub. Hugging Face
reports it beating much larger VLAs and ACT on LIBERO, Meta-World, and real SO-100/SO-101 arms,
with asynchronous inference giving ~30% faster response and 2x throughput.

Two reasons it matters here. First, it is the only serious VLA that runs on consumer hardware, so
it is the realistic choice for a practical. Second, its training set is *community-contributed* -
this is the one existence proof that the robot-data bottleneck might be attacked the way
open-source attacks everything else. The SO-101 arm is a few hundred dollars.

### Gemini Robotics (March 2025 - July 2026) - the frontier, behind glass

Four releases in seventeen months:

- **Gemini Robotics 1.0 + Gemini Robotics-ER** (March 12, 2025). Gemini 2.0 adapted to output
  actions, trained primarily on the bimanual ALOHA 2 platform and shown transferring to Franka
  bi-arm. The **-ER** split is the durable idea: separate the *embodied reasoning* model (spatial
  understanding, planning, success detection, tool calls) from the *action* model.
- **Gemini Robotics On-Device** (June 2025), a local-execution variant.
- **[Gemini Robotics 1.5 / -ER 1.5](https://arxiv.org/abs/2510.03342)** (September 25, 2025). Two
  things new: the VLA "thinks before acting", generating an internal natural-language reasoning
  trace for multi-step tasks; and **Motion Transfer**, a mechanism for learning across
  heterogeneous embodiments so a skill learned on ALOHA 2 shows up on Apptronik Apollo or a
  Franka bi-arm. ER 1.5 was reported state of the art on 15 academic embodied-reasoning
  benchmarks, aggregate score above 60. ER 1.5 shipped in the Gemini API via AI Studio; the VLA
  went to selected partners only.
- **Gemini Robotics-ER 1.6** (April 2026), with gauge and sight-glass reading developed with
  Boston Dynamics.
- **[Gemini Robotics 2, ER 2, On-Device 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)**
  (July 30, 2026). Whole-body humanoid control - walking, crouching and manipulating in one
  policy - 22-DoF five-fingered hands, multi-robot collaboration, and On-Device 2 adaptable to a
  new robot with "a few hours of data". Robots covered include Apptronik Apollo 2, Franka Duo,
  Dexmate, SO-101, Trossen. DeepMind published bar charts: general manipulation 45.7-76.3%,
  multi-finger dexterity 32-92% (unscrewing a bulb highest, screwing one in lowest at 36%),
  gripper tasks 74.2-89.6%. Those are company-reported numbers on tasks the company chose, but at
  least they are numbers, and the 36% on screwing in a light bulb is a usefully honest data point
  to show students next to the marketing video.

None of the VLA models are open. ER 2 is available through AI Studio and the Gemini Enterprise
Agent Platform.

### Dream-VLA (December 2025) - the architecture bet worth watching

[Dream-VLA](https://arxiv.org/abs/2512.22615), from the University of Hong Kong and Huawei, swaps
the autoregressive LLM backbone for a **diffusion language model** and continues pretraining it on
open robot datasets. 7B, Apache-2.0, checkpoints and codebase released.

The claimed payoff is the one that matters for control: **one diffusion step suffices** for
competitive action prediction, giving a reported **27x speedup** over autoregressive generation.
Reported 97.2% average on LIBERO, 71.4% on SimplerEnv-Bridge, 60.5% on SimplerEnv-Fractal,
above GR00T-N1 and OpenVLA-OFT.

This is one instance of a broader ICLR 2026 trend: [four concurrent papers](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
proposed discrete-diffusion VLAs that generate action tokens in parallel rather than
sequentially. Note the confusing terminology - "diffusion" here is over *tokens in the language
backbone*, not the continuous diffusion action head of Octo or GR00T. Worth disambiguating
explicitly on a slide, because a student who has just done ch10 will conflate them.

### LingBot-VLA 2.0 (July 2026) - the current open frontier

[LingBot-VLA 2.0](https://www.marktechpost.com/2026/07/08/lingbot-vla-2-0/) from Robbyant (Ant
Group), Apache-2.0, 6B on a Qwen3-VL-4B backbone. Notable for three things: a **55-dimensional
canonical action space** covering arms, end effectors, grippers, dexterous hands, waist, head and
mobile base (i.e. one action vocabulary for whole-body robots); a token-level MoE action expert;
and a training mix of ~50,000 hours of robot trajectories across 20 configurations plus **10,000
hours of egocentric human video**. Reported to beat pi-0.5 on the GM-100 generalist benchmark.

Cited here from a secondary source (MarkTechPost) plus the release announcement; the technical
report was not read directly, so treat the numbers as unverified.

## What to notice about the table as a whole

- **The open-closed gap is about data, not architecture.** Open models (OpenVLA, GR00T N1,
  pi-0.5, Dream-VLA) publish architectures that frontier labs are plausibly also using. What they
  cannot publish is 10,000 hours from a proprietary fleet in 100 homes.
- **Parameter counts stopped growing.** RT-2 was 55B in 2023. Almost everything since is 0.45B to
  7B. Cause: the model has to run on the robot, in real time, and the data does not support more.
  This is the opposite of the LLM trajectory and is worth a slide.
- **Chinese labs are now a large share of open releases** (Dream-VLA, LingBot, AgiBot's GO-1,
  VLA-Adapter, InternVLA-M1, XPeng, Unitree's UnifoLM-VLA-0). A 2026 model list that only has US
  labs on it is out of date.
