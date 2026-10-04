# Lecture 8: World Models - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 13.04.2026
- Source: `slides.pdf` in this folder (56 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week08_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/cTTmUZlOF2s (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_scott_reed.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 8: World Models

Oier Mees - ETH zurich, Microsoft - 13.04.2026

## Slide 1: Mid-term feedback (frame 2)

- Well-received: main lectures, guest lectures & homeworks
- Things to improve: paper presentations, more time for main lecture & a break, more ECTS credits for workload

[Figure: two screenshots of anonymous survey answers. Question: "What is your favorite aspect of the course so far? (What should we keep doing?)" - the answer praises the speakers and the homeworks and calls the course practical. A second answer says the paper presentations are useful as a list of papers everybody in the field knows about, but doubts the presentations themselves because they are very short and hard to follow.]

## Slide 2: Changes (frame 3)

- We will move one additional paper presentation from Monday's to Thursdays
  - More time for main lecture & a break

## Slide 3: Projects: Key Information (frame 4)

- 5 Projects & competitions, **Demo day on May 21th**
  - **Best team for per project will have potential chance to demo to Yann LeCun, Shuran Song, Jitendra Malik & others on May 29th (https://vreai.ch)**
- NVIDIA supports the course with 12500 H100 hours, ~277 H100 hours per team

## Slide 4: Project 1: Reasoning Pick&Place - VLA (frame 5)

- Eval 1(50pts): Pick up a plastic banana and place it into one of 3 bowls in front of the arm
- Eval 2 (50pts): Perform the same task, but the language instruction this time requires your VLA to reason about which bowl to put the banana into (i.e. like "put the banana into the 2 minus 1 bowl from the left")
- Eval 3 (50pts): Place a coke can on a certain image (you won't know the images beforehand)
- Bonus (50pts): We give bonus points for the smallest model in number of parameters

[Figure: a robot arm above a table with a coke can and three printed portrait photos, captioned "Move Coke Can near Taylor Swift".]

## Slide 5: Project 2: Pushing - World Model (frame 6)

- Eval 1 (50pts): Push a given smaller object (TBD which one) in a straight line
- Eval 2 (50pts): Push a given smaller object around an obstacle
- Eval 3 (50pts): Push a new unknown object in a straight line (25pts) and around an obstacle (25pts again)
- Bonus (50pts): Least amount of pushes needed in each of the three challenges (10 pts for least pushes for the first, 10 for the second, 30 for the last, will be divided amont the 10 groups competing).

[Figure: a robot pusher above a felt-covered table with three small coloured cubes, captioned "Time Step = 5".]

## Slide 6: Project 3: Singulation - Reinforcement Learning (frame 7)

- Eval 1 (50pts): Learn a pick an place policy that picks up one wooden block and placing them in one of three bowls laid out in front of the arm (can be just BC).
- Eval 2 (50pts): Singulate the combined four wooden blocks
- Eval 3 (50pts): Singulate more combined blocks.
- Bonus (50pts): Fastest singulating policy

[Figure: a robot arm next to a cluster of colourful wooden blocks on a table.]

## Slide 7: Project 4: Keyboard Typing - Code as Policies (frames 8-13)

- Design a system (VLM -> keypoints + some classical keypoint grasping policy) that is able to work in unseen eval environment.
- Eval 1 (50pts): Move to four points drawned on a paper sheet.
- Eval 2 (50pts): Pressing defined keys on a laptop keyboard.
- Eval 3 (50pts): Typing words on an unseen laptop keyboard.
- Bonus (50pts): Fastest spelling of words on the keyboard.

[Figure: a small yellow 3D-printed robot arm with an orange fingertip over a laptop keyboard, with lab power supplies and an oscilloscope behind it.]

[Frames 9-13 are pixel-identical repeats of frame 8; the slide stayed on screen.]

## Slide 8: Project 5: Cloth Folding - Diffusion Policy (frame 14)

- Milestone (50pts): Grab one edge.
- Milestone (50pts): Make one fold.
- Milestone (50pts): Second fold.
- Bonus (50pts): Fold the full cloth & fastest policy.

[Figure: a two-armed robot folding a cloth on a table next to a laundry basket, overlay "autonomous, 2x speed".]

[Note: slide 9 does not appear in the captured frames; the numbering jumps from 8 to 10.]

## Title [shown again] (frame 15)

[Same title slide as frame 1, shown when the lecture proper starts.]

## Slide 10: World Models (frame 16)

- Batters can hit 160km/h balls by anticipating the location
- Our world model allows to decide batting in shorter time than visuals reach our brain

[Figure: illustration of a baseball batter swinging, labelled "Subconscious world model" (brain icon), "Vision delay (100ms)", "Fastball arrival (400ms)", "Anticipated location" and "Reflexive action".]

## Slide 11: World Models vs Policies (frame 17)

- Policies/VLAs are blind to physical causality & temporal dynamics

| Policy / VLA | World model |
|---|---|
| Given $s_t$ and goal $g$, what action should I take? | Given $s_t$ and action $a_t$, what happens next? |
| inputs: $s_t$ (state), $g$ (language goal) | inputs: $s_t$ (state), $a_t$ (action) |
| $\pi(a_t \mid s_t, g)$ - policy / VLA | $p(s_{t+1} \mid s_t, a_t)$ - world model |
| $a_t$ - action to execute | $s_{t+1}$ - next state |

## Slide 12: Why World Models? (frame 18)

- What if we could predict how different sequence of actions would affect the environment?
  - Unlock optimizing for the optimal trajectory

[Figure: **robot** (obs $x_t$) -> **world model** $p(s_{t+1} \mid s_t, a_t)$, learned from data -> "N imagined rollouts", one marked "best" -> **execute best plan** (real world · 1 trial).]

## Slide 13: World Models: Data-driven Simulators (frame 19)

- Function that predicts the consequences of actions, i.e. a learned simulator

| Traditional simulator | World model |
|---|---|
| **hand-engineered physics engine**: rigid bodies · contact models · friction · collision geometry | **real interaction data**: robot trajectories · human video · internet-scale video |
| **simulator**: $s_{t+1} = f(s_t, a_t)$ [deterministic, hand-coded] | **world model**: $p(s_{t+1} \mid s_t, a_t)$ [learned from data] |
| **predicted next state** $s_{t+1}$: accurate only for modelled phenomena | **predicted next state** $s_{t+1}$: generalises to anything seen in data |
| ● fast, repeatable simulation; ✗ requires hand-engineered physics models | ● models anything capturable on video; ● no manual engineering required |

## Slide 14: World Model Definition (frame 20)

- Given the current state, an action, and a memory of the past - what happens next?

$$P(s_{t+1} \mid s_t, a_t, h_t)$$

- Strictly, *video models* $P(\text{video} \mid \text{text})$ *are not world models*

## Slide 15: Pixel AC-WMs (frame 21)

- Instead of full video prediction, predict action conditioned flow field that warps $\hat{F}_{t+1 \leftarrow t}$ current image to next frame

[Figure: the CDNA video-prediction network from the paper. RGB input $\hat{I}_{t-1}$ (64x64x3) -> 5x5 conv 1 (stride 2, 32x32) -> seven 5x5 conv LSTM layers (32 to 128 channels, down to 8x8 then deconvolved back to 32x32); the robot action and state (5 values each) are tiled to 8x8 and concatenated at the bottleneck. A fully connected branch outputs 10 5x5 CDNA kernels $\hat{m}^{(i)}$ that are convolved with the input image to give 10 transformed images $\hat{J}_t^{(i)}$ (64x64x3); a 1x1 conv with channel softmax gives compositing masks $\Xi$ (64x64), and masked compositing produces the 64x64 RGB prediction $\hat{I}_t$.]

Reference: *Unsupervised Learning for Physical Interaction through Video Prediction*, Finn, Goodfellow, Levine (2016)

[Note: slide 16 does not appear in the captured frames; the numbering jumps from 15 to 17.]

## Slide 17: Pixel AC-WMs [training and Visual MPC] (frame 22)

[Figure: two panels. **Training**: robot (autonomous interaction) -> $(o_t, a_t, o_{t+1})$ (large-scale robot interaction dataset) -> **pixel pred. model** (conv-LSTM, $\ell_2$ loss, pixel warp). "key insight: predict pixel motion, not pixel values: $\hat{o}_{t+1} = \hat{F}_{t+1 \leftarrow t} \circ o_t$ - warp frame by predicted flow field". **Test time - Visual MPC**: $o_t$ (current obs) -> sample N action seqs $\{a_{t:t+H}\}$ -> roll out pixel preds $\hat{o}_{t:t+H}$ -> "cost function - choose one": pixel distance $\|d_t - d_g\|_2$, goal image (image registration), or classifier (few-shot success) -> best $a^*$ -> execute + replan. "optimiser: Cross-Entropy Method (CEM) - gradient-free; iteratively refit Gaussian to top-k sequences · replan every step".]

- ✅ Self-supervised training on robot videos
- ✅ Generalizes to unseen objects due to motion objective
- ❌ Blurry predictions with L2 loss
- ❌ Planning in pixel space computationally expensive
- ❌ Prediction error accumulates

## Slide 18: [untitled: transition question] (frame 23)

**What if we compressed observations into a latent space and trained a policy entirely inside that learned world?**

## Slide 19: Latent AC-WMs (frame 24)

- By learning a compact, latent world model a policy can be trained entirely in imagination

[Figure: "Latent AC-WMs - general structure". environment ($o_t$) -> **encoder** ($z_t = \text{enc}_\phi(o_t)$, $z_t \in \mathbb{R}^d$, compact latent code) -> $z_t$ -> **dynamics model** ($h_{t+1} = f(h_t, z_t, a_t)$; $p(z_{t+1} \mid z_t, a_t, h_t)$; $p(\hat{r}_t \mid h_t, z_t)$) -> $z_t, h_t$ -> **policy** ($a_t = \pi(z_t, h_t)$, optimised on imagined rewards) -> $a_t$ -> environment ($o_{t+1}, r_t$); $o_{t+1}$ fed back to encoder. Box below: "training the policy in imagination: 1. fix encoder + dynamics model 2. roll out imagined $z_1, \dots, z_H$ 3. policy acts on $z_t, h_t$ 4. optimise policy on imagined $\hat{r}_t$ => policy improves without real environment interaction".]

## Slide 20: The OG Latent AC-WM (frame 25)

[Figure: the "World Models" agent diagram, unrolled over three time steps: each observation (a first-person game frame) goes through the Vision Model **V** to a latent z; the Memory RNN **M** carries h forward in time; the Controller **C** takes z and h and outputs an action a. Side text from the paper: "At each time step, our agent receives an observation from the environment. World Model: The Vision Model (V) encodes the high-dimensional observation into a low-dimensional latent vector. The Memory RNN (M) integrates the historical codes to create a representation that can predict future states. A small Controller (C) uses the representations from both V and M to select good actions. The agent performs actions that go back and affect the environment."]

Reference: *World Models*, Ha and Schmidhuber (2018)

## Slide 21: The OG Latent AC-WM: Visual Model (frame 26)

- Learn abstract, compressed representation of each input frame

[Figure: Original Observed Frame (a first-person corridor with a fireball) -> **Encoder** -> z -> **Decoder** -> Reconstructed Frame (a blurrier version of the same frame).]

Reference: *World Models*, Ha and Schmidhuber (2018)

## Slide 22: The OG Latent AC-WM: Memory Model (frame 27)

- Compress what happens over time $P(z_{t+1} \mid a_t, z_t, h_t)$
  - The MDN output models $P(z_{t+1})$ as a mixture of Gaussians

[Figure: an RNN unrolled over three steps; each RNN cell takes $a$, $z$ and the previous $h$, passes $h$ forward, and feeds an **MDN** head (with temperature $\tau$) that outputs the predicted next latent.]

Reference: *World Models*, Ha and Schmidhuber (2018)

## Slide 23: The OG Latent AC-WM: Controller Model (frame 28)

- Policy conditions on both the current latent $z_t$ and the memory $h_t$, single FC layer $a_t = \pi(z_t, h_t)$
  - Trained with black-box evolutionary optimisation algorithm (CMA-ES)

1. Sample N controllers from $W_c(i) \sim \mathcal{N}(\mu, \Sigma)$
2. For each candidate run dream rollout and collect imagined reward
3. Select top rollouts
4. Update $\mu$ and $\Sigma$ toward the top rollouts

Reference: *World Models*, Ha and Schmidhuber (2018)

[Note: slides 24 and 25 do not appear in the captured frames; the numbering jumps from 23 to 26.]

## Slide 26: Why Imagination Without Priors Drifts (frame 29)

- RNN is trained on real $z_t$ from VAE, not own predictions
- No principled way to generate $z_t$ from $h_t$ alone, $P(z_t \mid h_t)$

[Figure: two pipelines. "during training - real observation available": $o_t$ -> **VAE** encoder -> $z_t$ -> **MDN-RNN** ($h_t = f(h_{t-1}, z_t, a_t)$) -> **MDN** ($p(z_{t+1} \mid h_t)$) -> $z_{t+1}$ -> **controller** ($a_t = \pi(z_t, h_t)$). "in imagination - no real observation": the VAE box is crossed out with a "?"; the MDN outputs a hallucinated $\tilde{z}_{t+1}$ that is fed back into the RNN at the next step and also drives the controller ($a_t = \pi(\tilde{z}_t, h_t)$).]

**problem**

- VAE is trained separately from the RNN - it encodes $o_t$, not the model's own belief
- There is no prior $p(z_t \mid h_t)$ - without $o_t$ the model has no way to form a belief
- **Sampling errors in $\tilde{z}_t$ compound across steps => imagination drifts from reality**

## Slide 27: Training a Recurrent State-Space Model (RSSM) (frame 30)

[Figure: an RSSM unrolled over three steps. Top, "deterministic path - $h_t$ carries memory (never sampled)": RNN cells $h_t, h_{t+1}, h_{t+2}$ with $h_{t+1} = f(h_t, z_t, a_t)$, each taking the action. Under each RNN cell a box: "training - posterior: $q(z_t \mid h_t, o_t)$; $L_{KL} = D_{KL}(q \| p)$; imagination - prior: $p(z_t \mid h_t)$". "stochastic path - $z_t$ captures uncertainty about current state": the sampled $z_t$ feeds the next RNN step and decodes to $\hat{o}_t$ reconstruction and $\hat{r}_t$ reward (train only), with the observation $o_t$ entering the posterior.]

- **fix 1 - exposure bias**: $h_t$ is deterministic - never sampled, never corrupted. Even if $z_t$ drifts, $h_t$ remains a clean memory anchor. Errors cannot compound through $h_t$ - reset at every step. $\hat{o}_t$ and $\hat{r}_t$ decoded from $(h_t, z_t)$: Forces $z_t$ to encode everything needed to reconstruct the world
- **fix 2 - missing prior**: Prior $p(z_t \mid h_t)$ is learned - no observation needed. In imagination, sample $z_t$ from prior conditioned on $h_t$. The model has an internal compass - grounded in its own memory. $D_{KL}(q \| p)$ forces prior to match posterior. Imagination stays close to what training taught the model

Reference: *Learning Latent Dynamics for Planning from Pixels*, Hafner et al. (2019)

## Slide 28: RSSM Closed-Loop Inference Imagination (frame 31)

[Figure: the same three-step RSSM, now with no observations. Deterministic path: RNN $h_t$ (current state), $h_{t+1} = f(h_t, z_t, a_t)$, ... Under each RNN cell, "prior - imagination: $p(z_t \mid h_t)$"; "stochastic path - $z_t \sim p(z_t \mid h_t)$, no observation needed". A **policy** $\pi$ box at each step reads $h_t$ and $z_t$ and outputs $a_t = \pi(h_t, z_t)$, which feeds the next RNN step; each step also decodes $\hat{o}_t$ reconstruction and $\hat{r}_t$ reward.]

**policy + world model - closed-loop imagination**

- At each step: $\pi$ reads $(h_t, z_t)$ and outputs $a_t$ - no real robot needed
- World model advances: $h_{t+1} = f(h_t, z_t, a_t)$, then $z_{t+1} \sim p(z_{t+1} \mid h_{t+1})$
- $\hat{r}_t$ accumulated over the rollout -> policy optimised on imagined reward

Reference: *Learning Latent Dynamics for Planning from Pixels*, Hafner et al. (2019)

## Slide 29: Dreamer V1 (frame 32)

- Why plan from scratch with CEM at every step when you could amortise that planning into a learned policy?

| PlaNet (Hafner et al., 2019) - World model only, plan at test time with CEM | Dreamer V1 (Hafner et al., 2020) - World model + actor-critic, policy trained by backprop |
|---|---|
| **RSSM (world model)**: $h_{t+1} = f(h_t, z_t, a_t) \cdot p(z_{t+1} \mid h_{t+1}) \cdot p(\hat{r}_t \mid h_t, z_t)$; trained on real data · ELBO loss · frozen during policy training | **RSSM (world model)**: same; trained on real data · ELBO loss · frozen during policy training |
| [learned policy box crossed out: actions chosen at test time] **CEM planning (test time)**: 1. sample N action sequences 2. roll out in RSSM imagination 3. keep top-k, refit Gaussian 4. execute best first action, replan | **imagination rollout**: roll out $(z_t, a_t, \hat{r}_t)_{t=1}^{H}$ in latent space; no real environment needed; $\lambda$-returns for policy gradient; world model frozen during this phase. **actor-critic**: actor $a_t = \pi_\psi(h_t, z_t)$; critic $V_\xi(h_t, z_t)$; trained by backprop through frozen RSSM |
| training: collect real data -> train RSSM -> repeat | 1. collect real data -> train RSSM (ELBO) 2. freeze RSSM -> train actor-critic in imagination (backprop); alternate between phases |
| ✅ Actions chosen without policy bias; ❌ Planning from scratch at every step CEM | ✅ Backprop through differentiable dynamics enables long-horizon planning; ❌ Policy can exploit world model errors |

Reference: *Dream to Control: Learning Behaviors By Latent Imagination*, Hafner et al. (2020)

[Note: slides 30 and 31 do not appear in the captured frames; the numbering jumps from 29 to 32.]

## Slide 32: Dreamer Lineage (frame 33)

| | World Models (2018) | PlaNet (2019) | Dreamer V1 (2020) | Dreamer V2 (2021) | Dreamer V3 (2023) | Dreamer V4 (2025) |
|---|---|---|---|---|---|---|
| Key innovation | V-M-C decomposition (VAE + MDN-RNN + CMA-ES) | RSSM (deterministic $h_t$ + stochastic $z_t$) | Backprop through dynamics (actor-critic replaces CMA-ES) | Categorical latents (+ KL balancing) | Symlog + value normalisation (+ fixed hyperparams across all domains) | Flow matching + transformer KV (+ unlabeled video pretraining) |
| Headline result | First agent trained purely in imagination | 200x more sample-efficient than model-free methods | Outperforms model-free RL on DMControl | First world model to match DQN on Atari | First to obtain diamonds in Minecraft from pixels | Diamonds from offline data only |
| Detail | Car racing solved from pixels | CEM planning in latent space | Policy learned entirely in imagination | Human-level on 45 of 55 Atari games | ~20,000 actions from raw pixels. Same hyperparams: Atari, robotics, DMControl | 20,000+ actions - no env. interaction. Real-time on single GPU · learns from video |

## Slide 33: Dreamer V4: Decoupling Videos & Actions (frames 34-35)

- Pretrain dynamics model on *unlabeled videos* via shortcut forcing/flow matching $p(z_{t+1} \mid z_{\le t})$, finetune $p(z_{t+1} \mid z_{\le t}, a_{\le t})$
- [arrow at $z_{\le t}$] Memory: Replace RNN hidden states with Transformer KV cache

Learns MineCraft Diamonds, 20000 decisions, **purely offline**

[Figure: embedded Minecraft gameplay video (crafting-table and furnace menus, timestamps "1 min" and "3 min"); right, "Works on offline dataset from my Soar paper": three frames of a robot arm putting objects (a red block, a green sponge) into a bowl on a table.]

Reference: *Training Agents Inside of Scalable World Models*, Hafner et al. (2025)

## Slide 34: [untitled: transition question] (frame 36)

Dreamer's latents are domain specific. What if we want to scale the videos to all of Internet?

-> **Generative Video Models**

## Slide 35: Naïve Video Tokenization (frame 37)

- Per-frame ViT produces too many tokens

[Figure: pipeline. One frame $o_t$ (256x256 pixels) -> **ViT Encoder** (per-frame, no temporal; 16x16 patches, each -> d-dim vector) -> tokens per frame: $\frac{256}{16} \times \frac{256}{16} = 256$ tokens per frame -> x number of frames: 20 FPS x 10 seconds = 200 frames (short clip!) -> **context explosion**: total context length $256 \times 200 = 51{,}200$ tokens for a 10-second clip. "GPT-4: 128K limit total!"]

## Slide 36: Video Tokenization Compression (frame 38)

| Axis 1 - Spatial Compression | Axis 2 - Temporal Compression | Axis 3 - Adaptive Compression |
|---|---|---|
| deeper encoder -> fewer tokens per frame | merge frames -> fewer time steps | variable budget -> complex frames get more tokens |
| [naive: 16 patches -> 256 tokens/frame; compressed: 4 patches -> 4 tokens/frame] | [8 frames $o_{t+0} \dots o_{t+3}$ merged into 1 token $z_t$] | [four frames with token budgets that grow with frame complexity: "token budget allocated proportionally to frame complexity"] |
| **mechanism**: VQ-VAE / VQGAN encoder with stride; 8x spatial: 256x256 -> 4x4 tokens. **examples**: VQGAN · SD-VAE · Cosmos CV8x8x8. causal? yes - no temporal dependency | **Tubelets**: 3D patch (t,h,w); full tube at once; non-causal (TimeSformer · ViViT; Cosmos (analysis)). **Causal Aggregation**: attend past only; frame-by-frame decode; causal (Dreamer V4 tokenizer; Cosmos world model) | **mechanism**: tail-drop masking during training; tokens ordered by information content. **examples**: ElasticTok (Yan et al., ICLR 2025). causal? yes - works per-frame with context |
| 256 -> 4 tokens / frame: 64x spatial reduction | 200 -> 25 time steps: 8x temporal reduction | variable tokens per frame: 2-5x reduction on average |

**Combined (Cosmos)**: spatial 8x + temporal 8x = 512x compression -> 51,200 tokens becomes 100 tokens for a 10-sec clip. *Axes are independent and composable · causality requirement determines which temporal method to use*

[Note: as written, the numbers do not agree. The column footers give 64x spatial (256 -> 4 tokens per frame) and 8x temporal; the combined line says 8x spatial x 8x temporal = 512x. 51,200 / 512 = 100 tokens is consistent only with the combined line.]

## Slide 37: Video World Model: Where Do Actions Live? (frame 39)

| Action-Conditioned World Models (AC-WMs) | vs | Video-Based Action Prediction Models |
|---|---|---|
| *actions in · future states out* | | *pretrain on video · actions predicted from generated frames* |
| [observations + future actions -> **World Model** $p(s_{t+1} \mid s_t, a_t)$ -> future states; "[observations + actions] -> [$s_{t+1}$]"] | | **WAM - World Action Model**: *single model - joint video + action prediction*. [image + text -> **Joint Model** $p(o_{t:t+H}, a_{t:t+H} \mid \text{text}, o_{0:t}, q_t)$ -> video and actions] example: DreamZero |
| **what it does**: given current obs + planned actions, simulate what the world will look like; $a_t$ is an input - conditions future prediction. examples: Dreamer V1-V4 · DreamDojo · V-JEPA2 | | **VAM - Video Action Model**: *frozen video backbone · lightweight policy head*. [image + text -> **Video Backbone** (frozen, pretrained on internet video) -> video features -> **IDM** (robot data only) -> actions] example: mimic-video |

## Slide 38: [untitled: transition question] (frame 40)

Didn't we say video is expensive? **Why use video backbones?**

## Slide 39: VLAs: Poor Sample Efficiency (frame 41)

- VLMs are often trained on static images & text, blind to physical causality and temporal dynamics
  - Larg-scale teleop data necessary to convert a VLM to VLA

[Figure: Image-Text Pairs (a photo, "Person cutting carrots") -> **VLA** box: **VLM** (Semantics) -> Large Scale Robotics Data -> robot arm; "❌ Expensive Post-Training"; "Learn: Dynamics+Control".]

Figure designed by Oier Mees for: mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

## Slide 40: Video Backbones (frame 42)

- Offload dynamics learning to scalable action-free video data

[Figure: Video-Text Pairs (a stack of video frames, "Person cutting carrots") -> **Video Model** (Semantics + Visual Dynamics) -> Small Scale Robotics Data -> robot arm; "✅ Efficient Post-Training"; "Learn: Control".]

Figure designed by Oier Mees for: mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

## Slide 41: Video-Action Models (frame 43)

- Key: leverage **generative video models** as backbone

[Figure: the two pipelines from slides 39 and 40 stacked: **VLA** (VLM, large-scale robotics data, expensive post-training, learn dynamics + control) above **Video-Action Model (VAM)** (video model, small-scale robotics data, efficient post-training, learn control) -> "Dexterous & Generalizable Manipulation" (robot photos). A plot of success rate against robot data quantity (2%, 10%, 50%, 100%): the Video-Action Model (ours) starts near 0.75 at 2% and reaches about 0.9; the Vision-Language-Action Model (VLA) rises from about 0.3 at 2% to about 0.85 at 100%; an arrow marks "10x Sample-Efficiency".]

Reference: *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs*, Pai\*, Achenbach\*, Montesinos, Forrai, **Mees\***, Nava\*. arxiv, 2025.

## Slide 42: Policy Performance Scales with Video Model (frames 44-45)

[Figure: bar chart of Success Rate (0-1.0) by Action Decoder Input, for a Pretrained Video Model (grey) and a Finetuned Video Model (orange). With Predicted Video: pretrained about 0.03, finetuned about 0.48. With Expert Video: both 1.0. An arrow runs from the pretrained predicted-video bar through the finetuned one up to the expert-video bars.]

- ✅ Policy achieves perfect performance when conditioned on ground truth video
- ✅ Finetuning a video model on robot data improves video prediction quality and thus policy performance
- ✅ Policy performance scales with better video models

[Frame 44 shows only the Expert Video bars and the first point; frame 45 is the full build.]

## Slide 43: mimic-video (frames 46-47)

[Figure: the mimic-video architecture. A **Language Model** (with a Language Encoder for the instruction "put the package on the conveyor belt") and a **Video Model** (inputs: Video Input, a camera frame of two robot arms at a conveyor belt; Partial Video Prediction; "Video Noise $\tau_v$" schedule; "Repeat"). In the second build an **Action Decoder** ("Action Noise $\tau_a$", "Repeat") is added, outputting a row of robot actions.]

- mimic-video first partially denoises future video via flow matching
- [second build, frame 47:] Then, video model activations condition a light-weight flow matching action decoder

[Note: slides 44 and 45 do not appear in the captured frames; the numbering jumps from 43 to 46.]

## Slide 46: Joint Video-Action Sampling (frame 48)

| Autonomous Policy Execution | Video Generation (skipped in regular policy inference) |
|---|---|
| [Figure: two robot arms with humanoid hands handing a package over a conveyor belt; overlay "real autonomous"] | [Figure: the model's generated video of the same scene, blurrier, with a person walking past; overlay "video generation"] |

Training data: 512 episodes (2x speed)

## Slide 47: World Action Models: DreamZero (frame 49)

- Joint video-action prediction, no explicit IDM
- Autoregressive video chunk generation, faster KV cache inference with 14B model

[Figure: two panels around a **Joint Video-Action DiT** made of Causal DiT Blocks. **Training: Joint Video-Action Flow Matching**: video frames go through a VAE Encoder to latents and actions through an Action Encoder, both noised; proprioception and language ("Pack up the fruits into the bag") go through a State/Text Encoder; the DiT is trained with joint flow matching (teacher forcing). **Inference: Closed-Loop Real World Execution**: past frames (VAE Encoder), proprioception and language feed the DiT, which uses a KV Cache and autoregressive flow sampling to output future frames (VAE Decoder) and a future action chunk (Action Decoder) for async real-world execution; the scene is updated with real observations.]

Reference: *World Action Models are Zero-shot Policies*, Ye et al. arxiv, 2026.

## Slide 49 [shown early]: WAMs vs VAMs (frame 50)

[Same slide as frame 52; the lecturer showed it before going back to slide 48.]

## Slide 48: World Action Models: DreamZero [results] (frame 51)

- Strong generalization and few-shot adaptation capabilities

[Figure: four video stills of a humanoid robot (autonomous, 4x or 8x speed), each with its instruction and a success mark:]

- Pick up the red apple on the table and place it on the gray towel. ✅
- Fold the bottom of the green short sleeve to the middle. Then, pull the shirt toward the edge of the table. Next, fold the top of the shirt down to the middle. Finally, grasp the collar and folds it down. ⚠️
- Untie the knot of the shoelace. ✅
- Pick up the marker and draw a circle on the book. ⚠️

Reference: *World Action Models are Zero-shot Policies*, Ye et al. arxiv, 2026.

## Slide 49: WAMs vs VAMs (frame 52)

| Dimension | WAMs (DreamZero, mimic-video) | AC-WMs (Dreamer, DreamDojo) |
|---|---|---|
| Data | − instruction-labeled only - hard to use play data or failure trajectories at scale | + all robot data: play, failures, policy rollouts - easier to scale |
| Cross-embodiment | + easy - inputs/outputs embodiment-agnostic; only action decoder needs robot data | − open challenge - each robot has its own action space |
| Pre-training preservation | + small gap from pre-trained video model; preserves general visual capabilities | − action conditioning changes input distribution; may destroy pre-trained abilities |
| Beyond imitation | − BC paradigm only - instructions -> actions; no RL or counterfactual simulation | + RL in imagination + fine-grained gradient-based planning |
| Planning | + best-of-N via diverse text prompts; easy action proposal generation | + fine-grained gradient-based optimization of action sequences at inference time |

[Note: the title says "WAMs vs VAMs", but the table compares WAMs (with mimic-video, a VAM, listed under them) against AC-WMs.]

## Slide 50: [untitled: transition question] (frame 53)

All approaches so far had a decoder to reconstruct or generate pixels at some point

**Do you even need to predict pixels at all?**

## Slide 51: Joint-Embedding Predictive Architecture (JEPA) (frame 54)

- What if we don't predict pixels or tokens at all?

[Figure: "JEPA - Joint-Embedding Predictive Architecture: predict in latent space - no decoder, no pixel reconstruction". $o_t$ -> **Encoder** $z_t = \text{enc}(o_t)$ -> **Predictor** $\hat{z}_{t+1} = \text{pred}(z_t, a_t)$ -> $\hat{z}_{t+1}$ (predicted). $o_{t+1}$ -> **Same encoder** $z_{t+1} = \text{enc}(o_{t+1})$ -> $z_{t+1}$ (target). Loss $\|\hat{z}_{t+1} - z_{t+1}\|^2$. Side boxes: "No Decoder: no pixel reconstruction"; "Collapse risk: enc maps all frames to same vector: MSE = 0 trivially".]

**How to prevent collapse - three strategies**

| EMA Target Encoder | Frozen Pretrained Encoder | Gaussian Regularisation |
|---|---|---|
| Target enc = slow-moving copy (momentum update). Breaks gradient symmetry | Encoder fixed (e.g. DINO). Collapse impossible - encoder never changes | Force latent distribution to be Gaussian. Provably prevents collapse |
| e.g. I-JEPA, V-JEPA 2 | e.g. DINO-WM | e.g. LeWorldModel |
| not fully end-to-end / requires tuning | no end-to-end learning | fully end-to-end / one hyperparameter |

## Slide 52: Conclusion (frame 55)

One master formula $P(s_{t+1} \mid s_t, a_t, h_t)$, five different approaches

| Family | $s_t$ (state repr.) | $a_t$ role | Planning | Data | Example |
|---|---|---|---|---|---|
| Pixel AC-WMs | pixels $o_t$ | input - conditions future frame | visual MPC (CEM) | robot data + action labels | Finn 2016, Visual Foresight |
| Latent AC-WMs | compact latent $z_t$ | input - conditions latent dynamics | RL in imagination, Dreamer-style | robot data + action labels | Dreamer V1-V4, DayDreamer |
| WAMs | video tokens (full frames) | output - jointly predicted | generate video + extract actions | internet video + robot fine-tune | DreamZero |
| VAMs | video tokens (frozen backbone) | output - IDM reads from video | generate latent video + IDM | internet video (backbone) + robot data (IDM) | mimic-video |
| JEPA | latent $z_t$ (no decoder ever) | input - conditions latent predictor | latent MPC, no pixels needed | robot data + action labels | LeWorldModel, V-JEPA 2 |

**Open questions**

- **Will flexible conditioning win?** one model on text OR actions - best of both worlds
- **Pixels or latents?** does pixel prediction help cross-embodiment, or is it unnecessary cost?
- **Can JEPA scale?** V-JEPA 2 shows promise - can latent WMs match generative ones at scale?

## End (frame 56)

Thank you for your attention
