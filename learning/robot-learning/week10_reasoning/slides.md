# Lecture 10: Embodied Reasoning & Test-time Scaling - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 04.05.2026
- Source: `slides.pdf` in this folder (56 frames, 1280x720, no text layer), re-extracted on 2026-10-04 from the 720p YouTube recording with the slide repo's own `lectures/extract_slides.py` (github.com/idanL1212000/RobotLerning-2026-ETH-Zurich). That repo's `week10_slides.pdf` is only 640x360, so its frame numbering differs from this one.
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/CxhrjQuGEuE (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_archit_sharma.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 11: Embodied Reasoning and Test-time Scaling

Oier Mees - ETH zurich, Microsoft - 04.05.2026

[Note: the slide says "Lecture 11", but the course page and the YouTube title list this as Lecture 10 (week 10, May 4).]

## Slide 1: [untitled: transition question] (frame 2)

**Is Scaling Robot Data All We Need?**

## Slide 2: What is Missing? Intelligent Reasoning (frames 3-4)

- Training LLMs on all of Internet not enough
- LLMs train specifically for **reasoning**

[Figure: left, a grid of many robot-task video stills, captioned "Robots capable of **many tasks**". Right, the ChatGPT and DeepSeek logos, captioned "**Intelligent Reasoning**: Generalization, Interpretability, Interaction, Controllability". The two bullets are added in the second build (frame 4).]

## Slide 3: Robotics as Multimodal Sequence Modeling (frame 5)

- Can we **transfer** algorithmic improvements like **reasoning**?
- Can a robot **think harder** to **act better**?

[Figure: the Language / Image / Action token rows over "Pick up the spoon", the tabletop image and "ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$" (same figure as week 1, slide 33).]

## Slide 4: Embodied Chain-of-Thought Reasoning (frames 6-7)

Break down hard problems into intermediate reasonings

[Figure: left, a camera image of a tabletop with a towel (blue box), a toy watermelon (red box) and the gripper position (red dot), with a green arrow from the gripper to the watermelon. Right, a chat: the user asks "Put the watermelon on the towel"; the robot answers with "Plan: 1. Move to watermelon, 2. Grasp it, ...", "Subtask: Move to watermelon", "Move: Move backwards", "Bboxes, Gripper Pos:", then "ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$". Text beside it:]

- Standard policies directly map images to actions
- ❌ They thus often fail on novel tasks
- We thus propose to add *embodied reasoning* to the policy
- *Embodied reasoning* causes policies to "**look**" and "**think**" carefully
- ✅ This yields better performance, interpretability, and generalization!

[Frame 7 is an earlier build without the last two lines.]

Reference: *Robotic Control via Embodied Chain-of-Thought Reasoning*, Zawalski\*, Chen\*, Pertsch, **Mees**, Finn, Levine. CoRL, 2024.

## Slide 5: Creating Multimodal Reasonings at Scale (frames 8-9)

Maximally use existing data by **distilling foundation models**

[Figure: "How do we create robot reasoning data at scale? We use a suite of foundation models to generate reasoning data for training". A robot trajectory from the dataset (instruction "Put the watermelon on the towel", image frames, Proprio 1: $[\Delta x, \Delta\theta, \Delta\text{Grip}]$) is processed by Gemini, OWLv2 + SAM, Prismatic VLM and Grounding DINO (-> Bounding Boxes) to produce a REASONING record:]

- TASK: Place the watermelon on the towel
- PLAN: 1. Move to watermelon 2. Firmly grasp it 3. ...
- SUBTASK: The watermelon is the first object the robot needs to interact with -> Move to the watermelon
- MOVE: The watermelon is behind the robot, so it needs to move backward -> Move backward.
- GRIPPER POS: [156, 55]

Reference: *Robotic Control via Embodied Chain-of-Thought Reasoning*, Zawalski\*, Chen\*, Pertsch, **Mees**, Finn, Levine. CoRL, 2024.

[Second build of slide 5 (frame 9): the trajectory and the reasoning record are now boxed together as the "Embodied Chain-of-Thought Dataset", and the record gains "VISIBLE OBJS: Watermelon [126, 146, 141, 125], Towel [20, 59, 218, 198], ...".]

## Slide 6: Embodied Chain-of-Thought Architecture (frames 10-11)

"**Think carefully**" and "**look carefully**" before acting

[Figure: a timeline. **Input**: the tabletop image and "USER: 'Put the watermelon on the towel'". **Generated Embodied Chain-of-Thought** (added in frame 11), in order:]

- Semantic Reasoning:
  - TASK: Place the watermelon on the towel
  - PLAN: 1. Move to watermelon 2. Firmly grasp it 3. Move to towel 4. Place watermelon on towel
  - SUBTASK REASONING: The watermelon is the first object the robot needs to interact with. The robot is not yet close to the watermelon, so the robot needs to move closer
  - SUBTASK: Move to the watermelon
- Visual Reasoning:
  - MOVE REASONING: The watermelon is behind the robot, so it needs to move backward
  - MOVE: Move backward
  - GRIPPER POS: [156, 55]
  - VISIBLE OBJECTS: Watermelon [126, 146, 141, 125], Towel [20, 59, 218, 198], Spoon [114, 93, 141, 125] ...

[**Robot Action**: ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$ (robot arm). Frame 10 shows only Input -> Robot Action.]

Reference: *Robotic Control via Embodied Chain-of-Thought Reasoning*, Zawalski\*, Chen\*, Pertsch, **Mees**, Finn, Levine. CoRL, 2024.

## Slide 7: Generalization Evaluations (frame 12)

Boosts performance 30% **without extra robot data**

[Figure: horizontal bar chart, "Our Embodied Chain-of-Thought VLA is more performant than other generalist policies, especially on novel tasks". Success rate, approximate: ECoT (Ours) ~65, RT-2-X ~47, OpenVLA ~37, Octo ~19. A red arrow labelled "30%" runs from the end of the OpenVLA bar to the ECoT level.]

[Note: the arrow spans about 28 points of absolute success rate (37 -> 65); "30%" is the absolute gap, not a relative improvement.]

## Slide 8: Human-Robot Interaction (frames 13-15)

- ECoT makes policies more **interpretable**
- Allows human to **correct** behaviors

[Figure: a robot over a table with a yellow plush chick, a purple plush eggplant, a yellow cloth and a yellow sponge. "Lastly, human operators can correct robot behaviors by editing reasoning chains". The user asks "Pick up any object that is **not yellow**"; the robot's reasoning reads "Plan: ... Subtask: Move to the yellow cloth. Move: Move left" (marked wrong): "Robot misbehaves due to **incorrect reasoning**". The user adds "The eggplant near the arm is purple!", and in the last build (frame 15) the corrected reasoning reads "Subtask: The eggplant is a non-yellow object and should be picked up. -> Move to the eggplant. ..." (marked correct): "ChatGPT's corrected reasoning fixes robot behavior!"]

## Slide 9: Reasonings Generalize to Unseen Robots (frame 16)

[Figure: four scenes from robots not seen in training, each with labelled bounding boxes, a predicted gripper path, and a reasoning panel:]

1. [white plate, black bowl, orange, blue sink] TASK: Put the black bowl in the blue sink. PLAN: Move to the black bowl. Pick up the black bowl. Move to the sink. Drop the black bowl in the blue sink. SUBTASK: The black bowl is the first object to interact with -> Move to the black bowl. MOVE: The black bowl is to the right of the robot -> Move right.
2. [a robot, red bottle, yellow banana] TASK: Move to the banana. PLAN: Move to the banana, grasp the banana, move the banana to the plate, release the banana. SUBTASK: The banana is on the left side of the plate, so the robot needs to move left -> Move to banana. MOVE: The robot needs to move left to get closer to the banana -> Move left.
3. [black robot arm, carrot, red bowl] TASK: The task is to put the carrot in the red bowl. PLAN: 1. Move to the carrot. 2. Grasp the carrot. 3. Move to the red bowl. 4. Release the carrot. SUBTASK: The robot has grasped the carrot and needs to move it to the red bowl -> Move to the red bowl. MOVE: The red bowl is to the right of the robot, so the robot needs to move right to reach it -> Move right
4. [yellow towel, grey sponge; 4x] TASK: Put the grey sponge on the yellow towel. PLAN: Move to the sponge, pick up the sponge, move to the towel, put down the sponge. SUBTASK: The sponge is on the table, and it needs to be reached first -> Move to the sponge. MOVE: The sponge is below the robot, so move down -> Move down.

## Slide 10: Impact of Embodied Chain-of-Thought (frames 17-18)

- Key: new standard to train VLAs with reasoning
- Improves generalization, interpretability & interactiveness
- Adoption by industry and academia

[Figure: embedded video, captioned "Gemini Robotics VLA 1.5": a two-armed robot over a table with green, orange and yellow plates and fruit, overlay "THINKING WHILE ACTING" (Autonomous 1x); then a researcher talking at the same table.]

## Slide 11: Follow Up Works on Embodied Reasoning (frame 19)

[Figure: overview figures from four papers.]

- *MolmoAct: Action Reasoning Models that can Reason in Space*, Lee et al., 2025. [Figure: a Pre-Trained Action Reasoning Model takes camera images plus an instruction ("Pour water into the cup") or a trajectory input, and outputs depth perception tokens, a visual reasoning trace (2D waypoints) and action tokens.]
- *Action-Free Reasoning for Policy Generalization*, Clark et al., 2025. [Figure: "RAD learns broad **reasoning** from human + robot data and **action** generation from just robot data!": for "Pick up the controller" the model emits Task, Plan, Subtask Reason, Subtask, Move Reason, Move, Visible Objects and Gripper Pos before the robot action; action-free human videos supply reasoning only, robot data supplies reasoning + action.]
- *CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models*, Zhao et al., 2025. [Figure: pre-training data (visual Q&A, captioning, text to image/video), robot demonstrations and action-less videos train a model with causal & full attention that first predicts a future image (visual chain of thought), then actions $a_1 \dots a_n$, in a closed-loop control cycle.]
- *Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning*, Ganai et al., 2026. [Figure: reasoning primitives (Plan, Vis.Ob., Subtask, Ego) generated by a foundation model, combined with reasoning dropout, a prior VLA and a posterior VLA trained jointly; candidate reasonings are sampled, weighted by importance weights and used to train the reasoning VLA.]

## Slide 12: Inference Cost of Reasoning (frame 20)

| | |
|---|---|
| **Autoregressive VLA Inference Speed** *OpenVLA* | 4 Actions/Second |
| **Embodied Chain-of-Thought Reasoning Inference Speed** | 4 **Seconds/Action** [crying-face emoji] |

## Slide 13: [untitled: transition question] (frame 21)

Can we get the benefits of reasoning **without the inference costs**?

## Slide 14: Why Does Embodied Reasoning Help Robot Policies? (frame 22)

1. Better representation learning
2. Improved learning "curriculum"
3. Increased policy expressivity

[Slide footer, in square brackets on the slide:] ECoT-Lite Slides adapted from William Chen

## Slide 15: Why Does Embodied Reasoning Help Robot Policies? [1. representation learning] (frame 23)

**1. Better representation learning**

[Figure: INSTRUCTION + OBSERVATION (robot arm icon) -> network -> two outputs: REASONING (dashed) and ACTION.]

- [brace under the outputs] Train on both, but only choose *actions* at test time

## Slide 16: Why Does Embodied Reasoning Help Robot Policies? [2. curriculum] (frame 24)

**2. Improved learning "curriculum"**

Predicting continuous robot actions is very OOD for a VLM, is ECoT a better curriculum?

[Figure: INSTRUCTION + OBSERVATION + REASONING (dashed, as input) -> network -> ACTION.]

- [brace at the reasoning input] In-context reasoning *scaffolds* policy -> Akin to teacher-student learning!

## Slide 17: Why Does Embodied Reasoning Help Robot Policies? [3. expressivity] (frame 25)

**3. Increased policy expressivity**

[Figure: INSTRUCTION + OBSERVATION + <THINKING> (filler tokens, as input) -> network -> ACTION.]

- [brace at <THINKING>] More tokens in-context improves expressivity (even when meaningless)

References: *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models*, Pfau et al, 2024; *Think before you speak: Training Language Models With Pause Tokens*, Goyal et al, 2024.

## Slide 18: Results in LIBERO-90 (frame 26)

[Figure: left, two simulated kitchen scenes and the LIBERO benchmark overview figure. Right, horizontal bar chart "LIBERO-90 Success Rate", axis 70-90, values read from the bars, approximate: Embodied CoT ~91, Standard VLA ~82, Reasoning Pre-training ~87, Reasoning Dropout ~89, Reasoning "Scaffolding" ~84, Thinking Tokens ~79. Reasoning Pre-training and Reasoning Dropout are boxed together.]

Reference: *Training Strategies for Efficient Embodied Reasoning*, Chen, Belkhale, Mirchandani, **Mees**, Driess, Pertsch, Levine. CoRL, 2025.

[Note: slides 19-23 do not appear in the captured frames; the numbering jumps from 18 to 24.]

## Slide 24: Embodied Chain-of-Thought Lite [reasoning pre-training] (frames 27-28)

**Reasoning Pre-training** (top right, smaller: Reasoning Dropout)

[Figure: a simulated kitchen scene with three bowls and a plate, instruction "Put middle bowl on plate" -> **Policy** -> "Reasoning: Plan, Subtask, Move, Bounding Boxes, Gripper" (faded); an arrow labelled "Transfer" points down from the policy. Frame 28 (numbered slide 25) adds a second **Policy** below that receives the transferred weights and takes the same input; its output is not yet shown.]

Reference: *Training Strategies for Efficient Embodied Reasoning*, Chen, Belkhale, Mirchandani, **Mees**, Driess, Pertsch, Levine. CoRL, 2025.

[Note: frames 27 and 28 carry slide numbers 24 and 25; slides 26-29 do not appear in the captured frames.]

## Slide 30: Embodied Chain-of-Thought Lite [pre-training completed] (frame 29)

**Reasoning Dropout** (top left, smaller: Reasoning Pre-training)

[Figure: the slide 24-25 diagram completed: the pre-trained policy (faded, now also labelled "Action: $[\Delta x, \Delta\theta, \Delta\text{Grip}]$") transfers to a second policy that maps the scene directly to "Action: $[\Delta x, \Delta\theta, \Delta\text{Grip}]$", with no reasoning at test time.]

[Note: the heading emphasises Reasoning Dropout while the diagram still shows the completed reasoning pre-training pipeline; slide 31 does not appear in the captured frames.]

Reference: *Training Strategies for Efficient Embodied Reasoning*, Chen, Belkhale, Mirchandani, **Mees**, Driess, Pertsch, Levine. CoRL, 2025.

## Slide 32: Embodied Chain-of-Thought Lite [reasoning dropout] (frame 30)

**Reasoning Dropout**

[Figure: scene + instruction "Put middle bowl on plate" -> **Policy** -> "Reasoning: Plan, Subtask, Move, Bounding Boxes, Gripper" (dashed box, tagged "Randomly Dropped") -> "Action: $[\Delta x, \Delta\theta, \Delta\text{Grip}]$".]

Reference: *Training Strategies for Efficient Embodied Reasoning*, Chen, Belkhale, Mirchandani, **Mees**, Driess, Pertsch, Levine. CoRL, 2025.

## Slide 33: Results in Real-World Bridge WidowX Setup (frame 31)

[Figure: left, a sketch of Policy Performance against Inference Freq. (Hz): Embodied CoT sits top left (strong but slow), Standard VLA bottom right (fast but weaker); arrows from both lead to "ECoT-Lite Variants (Ours)" at top right. Caption: "*ECoT-lite* is faster than reasoning policies and stronger than non-reasoning policies". Right, two bar charts, values read from the bars, approximate:]

| | Bridge Success Rate | Inference Rate (Hz) |
|---|---|---|
| Embodied CoT | ~78 | ~0.4 (hatched extension to ~1.2) |
| Standard VLA | ~51 | ~3.5 |
| Reasoning Pre-training | ~70 | ~3.5 |
| Reasoning Dropout | ~61 | ~3.5 |

More Performant than Standard VLAs *without* sacrificing speed!

Reference: *Training Strategies for Efficient Embodied Reasoning*, Chen, Belkhale, Mirchandani, **Mees**, Driess, Pertsch, Levine. CoRL, 2025.

[Note: slide 34 does not appear in the captured frames; the numbering jumps from 33 to 35.]

## Slide 35: [untitled: transition question] (frame 32)

Embodied Reasoning so far has used a **fixed compute budget**

Can we do better **by letting it think longer at test time**?

-> **Test-Time Compute Scaling**

## Slide 36: Isn't LLM Test-Time Compute Scaling just Planning? (frame 33)

| | Classical planning (A\*, STRIPS, PDDL) | Learned planning (Chess, Go, Atari) | LLM reasoning (CoT, GRPO, R1) |
|---|---|---|---|
| World model | Hand-specified - Transition function | Learned - Policy + value network | Implicit in weights - No explicit model |
| Search / thinking | Explicit tree search - A\*, BFS, DFS | MCTS rollouts - Many sims per move | Token generation - Sequential reasoning |
| Verifier | Goal test - Exact, hand-specified | Win / loss - Exact, from game rules | Rule-based reward - Math grader, unit test |
| Scope | Known domains - Structured, discrete | Discrete games - Bounded action space | Open-ended tasks - Language, code, robots |

[A "<" sign sits between neighbouring cells of each row and an arrow runs left to right under the table: each column generalises the one before.]

## Slide 37: Scaling Test-Time Compute in Go (frame 34)

Rule of Thumb, increasing 120 Elo points requires either:

- 2x model size
- 2x test-time search

**Improving the raw policy from 3000 to 5200 Elo points would require scaling the model by ~100,000x**

[Figure: bar chart of Elo rating: Raw Network ~3000 (arrow: "No test-time search"), AlphaGo Zero ~5200 (arrow: "Full AlphaGo Zero"), AlphaGo Master ~4850, AlphaGo Lee ~3700, AlphaGo Fan ~3150, Crazy Stone ~1900, Pachi ~1300, GnuGo ~400. A dotted line at about 3650 marks "Superhuman Performance".]

[Slide footer, in square brackets on the slide:] Slides adapted from Noam Brown

Reference: *Mastering the Game of Go without Human Knowledge*, Silver, et al. Nature, 2017.

## Slide 38: [untitled: transition question] (frame 35)

Does this hold too in **language**?

## Slide 39: Scaling Test-Time Compute in Language (frame 36)

- Compute-optimal test-time scaling beats a ~14x larger model on easy/medium problems at equal FLOPs

[Figure: "Comparing Test-time and Pretraining Compute in a FLOPs Matched Evauation": relative improvement in accuracy from test-time compute (%) against the ratio of inference tokens to pretraining tokens:]

| Ratio | Easy Questions | Medium Questions | Hard Questions |
|---|---|---|---|
| <<1 | +21.6% | +27.8% | +11.8% |
| ~=1 | +16.7% | +3.5% | -11.9% |
| >>1 | +5.4% | -24.3% | -37.2% |

- SFT on own failed attempts + learned verifier - model learns to produce better answer given previous wrong ones, verifier picks the best one
- Generate multiple answers at test time, select best with verifier

Reference: *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, Snell et al, 2024.

## Slide 40: Why Do Reasoning Tokens Help? (frame 37)

- Any problem solvable in T computational steps can be solved by a **constant-size transformer** generating $O(T)$ reasoning tokens
- Generating $O(T)$ intermediate tokens before the answer gives the model $O(T)$ effective computation steps
- Without reasoning tokens: needs a much deeper/larger model or can't solve the problem at all

Reference: *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*, Li et al, 2024.

## Slide 41: Chain-of-Thought with LLMs (frame 38)

More tokens = more computation steps

[Figure: the well-known side-by-side example from the CoT paper. Standard Prompting: a one-shot arithmetic word problem with a bare answer, then a new word problem; the model output gives a wrong number (red cross). Chain-of-Thought Prompting: the same prompt, but the example answer spells out the intermediate arithmetic before the answer (highlighted); the model output also writes out its steps and reaches the correct answer (green tick).]

Reference: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, Wei, Wang et al, NeurIPS 2023.

[Note: the CoT paper appeared at NeurIPS 2022.]

## Slide 42: CoT Emerges in Large Models (frame 39)

[Figure: a 3x3 grid from the CoT paper: solve rate (%) on GSM8K, SVAMP and MAWPS against model scale (# parameters in billions) for LaMDA (0.4-137B), GPT (0.4-175B) and PaLM (8-540B), comparing standard prompting, chain-of-thought prompting and the prior supervised best (dashed). Chain-of-thought helps little for small models and pulls clearly ahead only at the largest scales; on GSM8K, PaLM 540B with CoT (~57%) slightly beats the prior supervised best (~55%).]

Reference: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, Wei, Wang et al, NeurIPS 2023.

## Slide 43: Original CoT Prompting = Few-shot/In-Context Learning (frame 40)

```text
Q: [problem 1]
A: [step 1... step 2... therefore answer] ✓
Q: [problem 2]
A: [step 1... step 2... therefore answer] ✓
Q: [new problem]
A: ???
```

- ⚠️ Prompt-sensitive, brittle
- ⚠️ Requires Human input
- ⚠️ Doesn't generalize beyond the examples

## Slide 44: [untitled: transition question] (frame 41)

Instead of eliciting reasoning with human prompts, can we **train the model to reason intrinsically**?

## Slide 45: [untitled: LLM vs reasoning-model response] (frame 42)

[Figure from the cited paper, summarised rather than copied: one arithmetic word problem (pages written per year), answered two ways. Top, "Large Language Model (LLM) GPT-4o's Response", labelled "Single Chain of Thought": a short numbered step-by-step calculation ending in the answer (624 pages, boxed). Bottom, "Large Reasoning Model (LRM) DeepSeek-R1's Response": a long `<think> ... </think>` block labelled "Thinking Process", full of self-checking phrases (underlined: "Wait", "Let me check that", "Alternatively", "Is that right?", "I need to double-check", a digression about leap years), followed by a short "Answer" that reaches the same 624 pages (boxed).]

Reference: *DeepSeek-R1 Thoughtology: Let's think about LLM reasoning*, Marjanović et al, 2026.

[Note: this frame shows no slide number; it sits between slides 44 and 46.]

## Slide 46: Learning to Reason with RL - GRPO (frame 43)

- Sample answers, compare with each other, update toward good ones
- No reward model, no value function, no human labels

[Figure: left to right. Prompt $q$ (a train-speed word problem) -> Policy $\pi_\theta$ (policy model) -> $G$ rollouts (chain-of-thought + answer): three reach the correct answer, two do not -> Reward $r_i$ (binary; correctness, format): 1, 1, 1, 0, 0 -> Advantage $\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$, green bars for the correct rollouts and red for the wrong ones; "No critic model - rewards relative to other samples in the group" -> GRPO objective -> Updated $\pi_{\theta'}$ (reinforce the good, suppress the bad).]

$$\mathcal{L} = \frac{1}{G} \sum_{i=1}^{G} \min\big(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1 \pm \varepsilon)\, \hat{A}_i\big) - \beta\, \text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

- $\rho_i = \pi_\theta / \pi_{\text{old}}$: importance ratio
- $\text{clip}(\rho_i, 1 \pm \varepsilon)$: trust region ($\varepsilon \approx 0.2$)
- $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$: drift penalty from $\pi_{\text{ref}}$
- $\hat{A}_i$: group-relative advantage

Reference: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, Shao et al, 2024.

[Note: the objective is written to be maximised; as a loss to minimise, the sign flips.]

## Slide 47: [untitled: transition question] (frames 44-45)

**Why didn't this work before?**

[Second build (frame 45):] **Base LLM needs to be good enough to produce any correct rollouts to reinforce!**

## Slide 48: Reasoning Behaviors Emerge from RL (frame 46)

[Figure: two plots against training steps (0 to ~10,500). (a) The frequency of reflection words climbs from about 1,000-2,000 early on to 6,000-10,000 after about 6,000 steps, peaking near 14,300. (b) The frequency of the word "wait" stays near zero until about 4,000 steps, then rises, spiking to about 1,500 near 8,600 steps. Caption: "Figure 9 | Evolution of reasoning behaviors during training. (a) Frequency of representative reflective words during the training process; (b) Specific occurrence patterns of the word 'wait' throughout the training process."]

Reference: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, DeepSeek-AI, 2026.

[Note: DeepSeek-R1 appeared on arXiv in January 2025 and in Nature in September 2025; the slide's 2026 matches neither.]

## Slide 49: [untitled: transition question] (frame 47)

Does this RL reasoning apply too in **vision**?

## Slide 50: Visual Reinforcement Fine-Tuning (frame 48)

- Applies GRPO to enhance visual grounding of VLMs
- The verifier is IoU (geometric), not a math grader

[Figure: a policy model answers visual questions ("the part of the vehicle that can be opened?" about a caravan photo; "What spices is the flower in the image?" about a flower photo) with a `<think>` reasoning block and an answer (a bounding box on the caravan door; "columbine"). Several candidate boxes are scored by a verifiable reward and the policy is updated by policy-gradient optimization against a frozen reference model. Rewards:]

$$R_{IoU} = \begin{cases} f(IoU), & \text{if match} \\ 0, & \text{otherwise} \end{cases} \qquad R_{cls} = \begin{cases} 1, & \text{if } P_{cate} = GT_{cate} \\ 0, & \text{otherwise} \end{cases}$$

Reference: *Visual-RFT: Visual Reinforcement Fine-Tuning*, Liu et al., 2025.

## Slide 51: [untitled: failure mode of answer-only rewards] (frame 49)

**Failure mode**: Standard GRPO rewards the final answer only, the **reasoning trace is invisible to the verifier**

In vision, hallucinated reasoning & correct answers aren't mutually exclusive

## Slide 52: ARGOS - Agentic Reasoning with Multimodal RL (frame 50)

- **Key:** aggregate multiple reward signal from intermediate reasoning steps, ie spatial & temporal reasoning in videos

[Figure: an image question (distance between a lamp and a sofa; response "1.8 metres") and a video question (which action the person failed to complete; response "the bottle cap is on ... option C") go to an **Adaptive Verifier**. For the image it scores a Spatial 2D Point (object: white lamp, pixel coordinate x=346, y=126), the reasoning quality (image + question + response) and the final answer (predicted 1.8, ground-truth 2.1). For the video it scores a Spatial 2D Point (dark bottle, x=195, y=199, frame 22, time 23.01 s), a Temporal Segment (event: the person is handling what appears to be an oil bottle, frames 22-23, 23.01-24.11 s) and the final answer (predicted C, ground-truth B). Teacher models and scoring functions: Grounding DINO, SAM-2, MOLMO-7B, GLM-4.5V, Pointing Hand Metric, String Match, Relative Accuracy, Language Model Score.]

Reference: *Multimodal Reinforcement Learning with Agentic Verifier for AI Agents*, Tan, Peng, Yang, Cheng, **Mees** et al. arxiv, 2025.

## Slide 53: ARGOS - Embodied AI Benchmarks (frames 51-54)

- Test-time reasoning helps for high-level planning
- 3-4x better than supervised CoT on the hardest tasks
- For continuous control, internalize reasoning during VLA training and drop it at inference - ECoT-Lite style

[Figure: three embedded simulation videos playing across four frames. A robot arm with the instruction "Stack the maroon triangular prism and the olive triangular prism in sequence." ("Environment Step 5: Task successfully completed!"); an indoor navigation agent with "Navigate to the Toaster in the room and be as close as possible to it." (a planning panel, then "Environment Step 14-15: Task successfully completed!" in front of a toaster); a robot arm at a stove with a pan and a moka pot.]

Reference: *Multimodal Reinforcement Learning with Agentic Verifier for AI Agents*, Tan, Peng, Yang, Cheng, **Mees** et al. arxiv, 2025.

## Slide 54: Conclusion (frame 55)

- Scaling test-time compute is often more effective than scaling pre-training
- Reasoning SFT in VLAs improves representations for control - drop reasoning at inference
- RL can induce emergent reasoning in LLMs and VLMs
- For multimodal agents, verifying the reasoning trace, not just the final answer, is essential

## End (frame 56)

Thank you for your attention
