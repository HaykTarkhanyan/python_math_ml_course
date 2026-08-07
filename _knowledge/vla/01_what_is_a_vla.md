# 01 - What a VLA is, and how a VLM becomes an action model

## Robotics vocabulary you need first

Five terms, then we can talk normally.

- **End effector.** The business end of a robot arm - a gripper, a suction cup, a five-fingered
  hand. Most manipulation policies output the *pose* of the end effector rather than individual
  joint angles, because pose transfers between arms with different link lengths and joint counts.
- **Degrees of freedom (DoF).** The number of independently controllable numbers. A standard
  arm-plus-gripper action is **7-DoF**: three translation (x, y, z), three rotation (roll, pitch,
  yaw), one gripper open/close. A humanoid with two five-fingered hands is 40+ DoF. This number
  is the single biggest driver of how hard the action-representation problem is.
- **Proprioception / proprioceptive state.** The robot's own sense of where its body is: current
  joint angles, gripper width. Cheap, exact, and fed to the policy alongside the camera images.
- **Teleoperation.** A human driving the robot in real time - via a VR controller, a leader arm
  that the follower arm mirrors, or a glove. This is how essentially all robot training data is
  produced, and it is why robot data costs roughly one human-hour per robot-hour.
- **Embodiment.** A specific robot body: a Franka Panda 7-DoF arm, a bimanual ALOHA rig, a
  Unitree humanoid. Two embodiments have different cameras, different action space dimensions,
  different reachable workspaces. "Cross-embodiment" means one model driving several of these.

## The definition, and what it excludes

The narrow definition, and the useful one, is the one used in the
[ICLR 2026 survey of VLA submissions](https://mbreuss.github.io/blog_post_iclr_26_vla.html): a VLA
is a model built on **backbones pretrained on large-scale vision-language data**, then fine-tuned
to emit control commands. The pretraining is the load-bearing part of the definition. A
transformer trained from scratch on robot demonstrations is a *visuomotor policy*, not a VLA, and
the distinction matters because the entire argument for VLAs is that internet pretraining buys
generalization that robot data alone cannot.

Wikipedia's [VLA article](https://en.wikipedia.org/wiki/Vision-language-action_model) states it
operationally: given an image or video of the surroundings plus a text instruction, a VLA
"directly outputs low-level robot actions that can be executed to accomplish the requested task".

The word **directly** is doing the work. The classical robotics stack is a pipeline: perception
module produces object poses, a symbolic planner produces a sequence of subgoals, a motion
planner produces a collision-free trajectory, a controller tracks it. Each stage has its own
representation and its own failure mode, and the interfaces between them are hand-designed. A VLA
replaces the whole pipeline with one network trained end to end on (observation, instruction,
action) triples. That is the idea in one sentence, and everything else is consequences of it.

## The observation-to-action loop

At run time the loop is:

1. Capture images from 1-3 cameras. Typically one third-person/scene camera and one **wrist
   camera** mounted on the end effector. The wrist camera matters more than people expect: it
   gives a close, viewpoint-stable view of the contact, and policies lean on it heavily.
2. Concatenate: image tokens + tokenized instruction + (optionally) the proprioceptive state.
3. Forward pass. Out comes either a sequence of discrete tokens that decode to actions, or a
   continuous action vector produced by a diffusion or flow head.
4. Send actions to the robot's low-level controller, which does the actual joint servoing at
   several hundred Hz. **The VLA does not close the torque loop** - a conventional controller
   underneath still handles that.
5. Repeat.

Two things about this loop are non-obvious to someone coming from language modelling.

**The actions are usually deltas, not absolute positions.** OpenVLA's output is a 7-vector of
end-effector *deltas* - how much to move from here, not where to go
([OpenVLA](https://arxiv.org/abs/2406.09246)). This makes the same action mean the same thing
regardless of where in the workspace the robot currently is, which is a form of built-in
translation invariance.

**There is no ground truth to condition on.** In language modelling, teacher forcing means a
mistake at step t does not corrupt step t+1 during training. On a robot, an error at step t moves
the arm to a state that was not in the demonstration data, and step t+1 is now an out-of
distribution prediction. This is the **compounding error** problem, and it is the reason for
action chunking (below). Behaviour cloning without something to counteract it degrades in a way
that has no analogue in offline supervised learning.

## Action chunking: the single most important trick

Instead of predicting one action, the policy predicts the next **H** actions in one shot and
executes them open-loop before re-querying. This is **action chunking**, introduced with ACT in
[ALOHA (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705).

Why it works:

- It cuts the effective decision horizon by a factor of H, so compounding error accumulates over
  T/H decisions instead of T.
- It handles **non-Markovian** human demonstrations. A teleoperator pauses, hesitates, and moves
  in ways that depend on their intent, not just the current image. Predicting a whole chunk lets
  the model commit to one intent instead of averaging over several at every step.
- It cuts the query rate, which is what makes a 3B-parameter model usable for control at all.

The cost is reactivity: during a chunk the robot is blind to changes. ACT's answer was **temporal
ensembling** - re-query every step anyway, so overlapping chunks give several predictions per
timestep, and take an exponentially weighted average. That smooths the trajectory but costs a
forward pass per control step.

Typical H is 8 to 50. pi-0 uses **H = 50** ([pi-0](https://arxiv.org/abs/2410.24164)).

## Making a VLM emit actions: three families

This is the design decision that separates the model families, and it is worth teaching as such,
because it is a clean instance of "discretize versus model the continuous distribution directly".

### 1. Discrete action tokens (RT-2, OpenVLA)

Bin each action dimension into 256 bins, treat each bin as a token, and let the VLM autoregress
over them exactly as it does over text. RT-2 and OpenVLA both do this; OpenVLA overwrites the 256
least-used tokens in the Llama-2 vocabulary rather than extending it.

- **Pro:** zero architectural change. The action head *is* the language head. You keep the VLM's
  pretrained machinery intact, and you can co-train on text and vision-language data for free.
- **Con:** quantization error, and cost. A 7-DoF chunk of length 50 is 350 tokens generated
  autoregressively, which is far too slow. [FAST](https://arxiv.org/abs/2501.09747) reports that
  naive per-timestep binning "fails completely" on high-frequency dexterous tasks - not degrades,
  fails - because at 50 Hz consecutive actions are nearly identical and the token sequence
  carries almost no information per token.

### 2. Compressed discrete tokens (FAST)

[FAST (Physical Intelligence, January 2025)](https://arxiv.org/abs/2501.09747) fixes the above
with a trick borrowed from JPEG and MP3: apply a **discrete cosine transform** to each action
dimension across the chunk, round away the small high-frequency coefficients, then BPE the sparse
coefficient matrix. Reported at roughly **10x compression** over naive binning and **5x faster
VLA training**. FAST+ is shipped as a universal tokenizer trained on 1M real robot trajectories,
usable black-box on new robots.

This is a genuinely elegant idea and a good lecture beat: robot action sequences are smooth
signals, so treat them like audio.

### 3. Continuous heads: diffusion and flow matching

Attach a separate head that outputs the chunk as a continuous vector, trained as a conditional
diffusion or flow-matching model conditioned on the VLM's features.

- **Octo** uses a diffusion head on a small transformer ([Octo](https://octo-models.github.io/)).
- **pi-0** attaches a 300M-parameter "action expert" to a 3B PaliGemma backbone, trained with
  **flow matching**, 3.3B total, 10 integration steps at inference
  ([pi-0](https://arxiv.org/abs/2410.24164)).
- **GR00T N1** uses a diffusion transformer conditioned on an Eagle-2 VLM
  ([GR00T N1](https://arxiv.org/abs/2503.14734)).

Why diffusion rather than regression: demonstration data is **multimodal** in the probabilistic
sense. Two humans reaching for a mug take different valid paths; an L2-regression policy averages
them and produces a path that goes through the mug. A diffusion or flow head samples one mode.
This is the same argument as for image generation, and the reader already has it from ch10.

Wikipedia's summary of the tradeoff matches what the papers say: discrete tokens "can limit
spatial accuracy or temporal resolution", continuous heads "scale better to robots with many
degrees of freedom, where discretization for every DoF would be impractical".

### The families are converging

pi-0.5 is the interesting case, because it uses **both**, hierarchically
([pi-0.5, April 2025](https://www.pi.website/blog/pi05)): a discrete autoregressive stage first
predicts a natural-language subtask ("pick up the pillow"), then a flow-matching stage decodes
that into a 50-step motor command chunk. Gemini Robotics 1.5 does something similar with an
explicit natural-language "thinking" trace before acting
([DeepMind, September 2025](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)).

Note also the counter-trend: [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) shows you can drop
diffusion entirely, use **parallel decoding + action chunking + continuous actions + plain L1
regression**, and get 26x faster action generation with 97.1% average on LIBERO. So "you need
diffusion for multimodality" is not settled - with chunking, plain regression is competitive on
current benchmarks. Report this as an open disagreement, not a resolved question.

## Control frequency, latency, and why it is the hard engineering constraint

Rough numbers for orientation:

| System | Rate | Source |
|---|---|---|
| Low-level joint controller (not the VLA) | 100-1000 Hz | standard robotics |
| RT-1 policy queries | 3 Hz | [RT-1](https://arxiv.org/abs/2212.06817) |
| pi-0 action output | up to 50 Hz | [pi-0](https://arxiv.org/abs/2410.24164) |
| Helix System 2 (VLM planner) | 7-9 Hz | [Figure](https://www.figure.ai/news/helix) |
| Helix System 1 (visuomotor) | 200 Hz | [Figure](https://www.figure.ai/news/helix) |
| GR00T N1 chunk of 16 actions | 63.9 ms on an L40 | [GR00T N1](https://arxiv.org/abs/2503.14734) |

The tension: contact-rich manipulation wants tens to hundreds of Hz, and a 3B-7B transformer
forward pass is 50-200 ms. Chunking buys you a factor of H, but it costs reactivity, and there is
still a discontinuity every time a new chunk starts - the robot jerks.

Two structural answers, both worth a slide:

**Dual-system / asynchronous architectures.** Split into a slow semantic system and a fast
reactive one. Figure's **Helix** is the clearest instance: a 7B VLM at 7-9 Hz compresses the
scene and instruction into a *single continuous latent vector*, which conditions an 80M
visuomotor policy running at 200 Hz. GR00T N1 uses the same split. The cost, per Wikipedia's
summary, is "increased computational complexity" and a harder training problem, since the latent
interface has to be learned.

**Real-time chunking (RTC).** [Black et al., NeurIPS 2025](https://arxiv.org/abs/2506.07339)
compute the next chunk *while executing the current one*, freeze the actions that are guaranteed
to have executed by the time the new chunk lands, and "inpaint" the rest - literally using the
diffusion inpainting trick on an action sequence. It is training-free and applies to any
diffusion or flow VLA including pi-0.5. This is a nice pedagogical payoff for a course that has
already covered diffusion inpainting on images.

## What "generalist robot policy" actually means

Three distinct claims travel under this label, and they are not equally supported. Keep them
separate in the lecture.

1. **Multi-task.** One set of weights does many tasks in one environment, selected by the
   instruction. Well established since RT-1 in 2022 (700+ instructions, one model).
2. **Cross-embodiment.** One set of weights drives *different robots*. Established at the level
   of "it works and often helps" - RT-1-X trained on Open X-Embodiment achieved about 50% higher
   mean success than the single-robot baselines
   ([Open X-Embodiment](https://arxiv.org/abs/2310.08864)) - but with real caveats, see `03`.
3. **Open-world generalization.** New objects, new scenes, new instructions, zero-shot. This is
   the claim that justifies the whole enterprise, and it is the weakest. The
   [ICLR 2026 review](https://mbreuss.github.io/blog_post_iclr_26_vla.html) is blunt: open
   academic models match frontier models on simulation benchmarks but "dramatically lag" in
   zero-shot open-world behaviour after pretraining, and the gap is attributed to data quality
   and evaluation scope rather than architecture.

The honest summary for students: claim 1 is solved, claim 2 mostly works, claim 3 is the research
frontier and is where the marketing lives.
