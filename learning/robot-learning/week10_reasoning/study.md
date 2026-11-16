# Week 10 study guide: Embodied Reasoning & Test-time Scaling

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Week 5's PPO objective (the clipped ratio) and its baselines, week 7's autoregressive VLAs and their tokens-per-second cost, and week 1's AlphaGo story. The slides call this "Lecture 11"; it is the week 10 lecture.

### Extra comments

**Where ECoT's reasoning labels come from.** No human wrote them. A pipeline of existing models labels existing robot data: a large language model writes the plan and sub-task text, open-vocabulary detectors and segmenters give the object boxes, and the gripper's position is tracked in the image. The robot data stays the same; the labels are distilled from foundation models. That is why the gain comes "without extra robot data". ([06:01](https://youtu.be/CxhrjQuGEuE?t=361), [slide 5](slides.md#slide-5))

**Read "30%" carefully.** On slide 7 the arrow spans about 28 points of absolute success rate (OpenVLA about 37, ECoT about 65), as the transcriber note says. Relative to OpenVLA that is about a 75% improvement. Papers quote whichever sounds best; always ask "points or percent?". ([09:01](https://youtu.be/CxhrjQuGEuE?t=541), [slide 7](slides.md#slide-7))

**Editable reasoning is a control channel.** Because the plan is written in language before the action, a person (or another model) can read it, spot "move to the yellow cloth" as wrong, and correct it, and the action follows the corrected plan. A policy that maps pixels straight to actions offers no such handle. ([10:02](https://youtu.be/CxhrjQuGEuE?t=602), [slide 8](slides.md#slide-8))

**Where the slowdown comes from.** An autoregressive VLA pays one forward pass of a 7B model per generated token. A plain OpenVLA step is 7 action tokens; an ECoT step first writes a plan, sub-task, move, boxes and gripper position, often hundreds of tokens. Hence 4 actions per second against about 4 seconds per action, roughly a 16-fold slowdown, at every step. ([14:00](https://youtu.be/CxhrjQuGEuE?t=840), [slide 12](slides.md#slide-12))

**What ECoT-Lite's experiments tell you.** Three hypotheses for why reasoning helps: better representations, an easier curriculum, or simply more computation from more tokens. Filler "thinking" tokens did not help (they scored below the plain VLA on LIBERO-90), which argues against the pure-compute explanation for this setting. Training with reasoning, then acting without it (reasoning pre-training, or randomly dropping the reasoning during training), recovered much of the gain at full speed. So the reasoning mostly improves what the network learns, the same point Scott Reed made about video prediction in week 8. ([20:00](https://youtu.be/CxhrjQuGEuE?t=1200), [22:01](https://youtu.be/CxhrjQuGEuE?t=1321), [slides 14](slides.md#slide-14) to [33](slides.md#slide-33))

**Reasoning dropout gives you a speed dial.** Trained to act both with and without the reasoning, one model can be run either way at test time: think when there is time or the situation is unusual, act directly when speed matters. ([24:00](https://youtu.be/CxhrjQuGEuE?t=1440), [slide 32](slides.md#slide-32))

**The Go rule of thumb, as arithmetic.** If doubling the model or doubling the search each buys about 120 Elo, going from 3000 to 5200 Elo with model size alone needs $2200/120 \approx 18$ doublings: $2^{18} \approx 260{,}000$ times the model. The slide's "~100,000x" is the same order of magnitude; it is a rule of thumb. To calibrate Elo: a 100-point gap means the stronger player wins about 64% of the time, since $1/(1 + 10^{-100/400}) \approx 0.64$. ([29:00](https://youtu.be/CxhrjQuGEuE?t=1740), [slide 37](slides.md#slide-37))

**When test-time compute wins, and when it does not.** In Snell et al.'s compute-matched comparison, spending compute at inference beats a roughly 14-times-larger model on easy and medium problems when inference is light relative to pre-training. When a model will answer huge numbers of queries, or the questions are hard, compute spent once in pre-training pays off better. Inference compute is paid per query; pre-training compute is paid once. ([31:01](https://youtu.be/CxhrjQuGEuE?t=1861), [slide 39](slides.md#slide-39))

**Why more tokens means more computation.** A transformer has a fixed number of layers, so one forward pass is a fixed number of serial steps. Each generated token is another full pass whose output becomes part of the next input, so writing $T$ tokens buys $T$ serial steps, like unrolling a recurrent network. Problems that are inherently step-by-step need that, or a much deeper model. ([34:01](https://youtu.be/CxhrjQuGEuE?t=2041), [slide 40](slides.md#slide-40))

**GRPO is week 5 with the critic removed.** It is PPO's clipped objective, with the advantage of each sampled answer computed against the other answers to the same prompt: subtract the group's mean reward, divide by its standard deviation. That is slide 13's "mean reward" baseline from week 5. The consequence for "why didn't this work before": if every sample in a group fails (or every one succeeds), all advantages are zero and nothing is learned. The base model must already succeed sometimes, which is Wagenmaker's exploration argument from week 5 in a new setting. ([41:01](https://youtu.be/CxhrjQuGEuE?t=2461), [43:00](https://youtu.be/CxhrjQuGEuE?t=2580), [slides 46](slides.md#slide-46) and [47](slides.md#slide-47))

**"Wait" is move 37 again.** Nobody showed the model how to pause and re-check; the word "wait" appears and becomes frequent only after thousands of RL steps, because reconsidering raised the reward. RL finding a behaviour that is absent from the data is the same story as AlphaGo's move 37 in week 1, and the thing imitation alone cannot do. ([44:01](https://youtu.be/CxhrjQuGEuE?t=2641), [slide 48](slides.md#slide-48))

**Outcome rewards versus checking the steps.** In math, a wrong derivation rarely lands on the right answer. In vision it easily can: a model can describe objects that are not there and still pick the right option. Rewarding only the final answer then reinforces hallucinated reasoning. ARGOS also checks intermediate claims (points, time segments, reasoning quality) with a panel of verifier models. For robots, the hard part is the same as in week 2: building a verifier that tells success from failure. ([49:01](https://youtu.be/CxhrjQuGEuE?t=2941), [slides 51](slides.md#slide-51) and [52](slides.md#slide-52))

### Going further

- Zawalski et al., [Robotic Control via Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693) (ECoT), and Chen et al., [Training Strategies for Efficient Embodied Reasoning](https://arxiv.org/abs/2505.08243) (ECoT-Lite).
- Snell et al., [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314).
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948): the RL-only "R1-Zero" sections show reasoning emerging.
- Li et al., [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875), for the theory on slide 40.
- Follow-ups from slide 11: [MolmoAct](https://arxiv.org/abs/2508.07917) and [CoT-VLA](https://arxiv.org/abs/2503.22020).

## Lecture quiz

### Q1

Where did the reasoning labels used to train ECoT come from?

- A) Human annotators wrote them for every episode.
- B) A pipeline of foundation models generated them from existing robot data: plans and sub-tasks from a language model, object boxes from open-vocabulary detectors, and the gripper position from the images.
- C) They were learned with reinforcement learning.
- D) They came from a simulator.

<details><summary>Answer</summary>

**B.** That is what "distilling foundation models" on slide 5 means: the robot data is unchanged and the labels are free, apart from compute. ([slide 5](slides.md#slide-5), [06:01](https://youtu.be/CxhrjQuGEuE?t=361))

</details>

### Q2

Slide 7 says ECoT "boosts performance 30%". What does the chart actually show?

- A) A 30% relative improvement.
- B) 30% faster inference.
- C) 30% more training data.
- D) About 28 points of absolute success rate over OpenVLA (about 37 to 65), with no extra robot data.

<details><summary>Answer</summary>

**D.** The arrow measures an absolute gap. Relative to OpenVLA's 37 the improvement is about 75%. ([slide 7](slides.md#slide-7), [09:01](https://youtu.be/CxhrjQuGEuE?t=541))

</details>

### Q3

Why does ECoT run at about 4 seconds per action when OpenVLA runs at about 4 actions per second?

- A) It generates a long reasoning chain token by token, each token a full forward pass of the 7B model, before the action tokens, at every step.
- B) It uses larger images.
- C) It uses more cameras.
- D) It uses a diffusion head.

<details><summary>Answer</summary>

**A.** Autoregressive generation costs one forward pass per token, and the reasoning is many times longer than the 7 action tokens. Same model, same images: the extra tokens are the whole difference. ([slide 12](slides.md#slide-12), [14:00](https://youtu.be/CxhrjQuGEuE?t=840))

</details>

### Q4

What did the ECoT-Lite experiments find?

- A) Meaningless filler "thinking" tokens recovered all of the gain.
- B) Nothing helps unless the reasoning is generated at test time.
- C) Training with reasoning (as pre-training, or with random reasoning dropout) recovered much of the gain while acting at standard VLA speed; filler tokens did not help, so the benefit is mainly better learned representations.
- D) Reasoning makes policies worse.

<details><summary>Answer</summary>

**C.** In LIBERO-90, filler tokens even scored below the plain VLA, while reasoning pre-training and dropout came close to full ECoT. On the real WidowX they kept standard-VLA speed with clearly better success. ([slides 18](slides.md#slide-18) and [33](slides.md#slide-33), [25:00](https://youtu.be/CxhrjQuGEuE?t=1500))

</details>

### Q5

What does training with reasoning dropout let you do at test time?

- A) Choose per situation whether to generate the reasoning, trading speed for performance with a single model.
- B) Skip the camera images.
- C) Switch to a different robot.
- D) Train faster.

<details><summary>Answer</summary>

**A.** Because the model learned to predict actions both with and without the reasoning in front of them, both modes work at inference. ([slide 32](slides.md#slide-32), [24:00](https://youtu.be/CxhrjQuGEuE?t=1440))

</details>

### Q6

In Go, about 120 Elo can be bought either by doubling the model or by doubling the search. Roughly how many model-size doublings would it take to go from 3000 to 5200 Elo without search?

- A) 2
- B) 5
- C) 100
- D) About 18, a model several hundred thousand times larger

<details><summary>Answer</summary>

**D.** $2200 / 120 \approx 18$, and $2^{18} \approx 260{,}000$. The slide's "~100,000x" is the same order of magnitude. Search at test time bought the same improvement for a few seconds per move. ([slide 37](slides.md#slide-37), [29:00](https://youtu.be/CxhrjQuGEuE?t=1740))

</details>

### Q7

Why do intermediate reasoning tokens give a transformer more computation?

- A) They enlarge the vocabulary.
- B) Each generated token is another full forward pass whose output feeds the next one, so $T$ tokens give $T$ serial computation steps, while answering directly is limited to one pass of fixed depth.
- C) Because attention is quadratic in length.
- D) They do not; the computation is the same.

<details><summary>Answer</summary>

**B.** That is slide 40's result: a constant-size transformer that writes $O(T)$ reasoning tokens can solve problems needing $T$ sequential steps. Quadratic attention (C) is a cost, not the source of extra serial depth. ([slide 40](slides.md#slide-40), [34:01](https://youtu.be/CxhrjQuGEuE?t=2041))

</details>

### Q8

In GRPO, a prompt's 8 sampled answers all receive reward 0. What does that prompt contribute to the update?

- A) A large negative push.
- B) A large positive push.
- C) Nothing: each advantage is (reward minus the group mean) over the group's standard deviation, and with identical rewards all advantages are zero.
- D) A random push.

<details><summary>Answer</summary>

**C.** Advantages are relative to the group, so a group with no variation in reward carries no signal. That is why the base model must already be good enough to succeed sometimes, the answer to slide 47's question. ([slides 46](slides.md#slide-46) and [47](slides.md#slide-47), [43:00](https://youtu.be/CxhrjQuGEuE?t=2580))

</details>

### Q9

What problem does ARGOS address?

- A) Answers that are too long.
- B) Slow training.
- C) Models that cannot read images.
- D) In vision, a model can reach the right answer with hallucinated reasoning; rewarding only the final answer cannot catch that, so ARGOS also scores intermediate steps such as points, time segments and reasoning quality.

<details><summary>Answer</summary>

**D.** Slide 51: the reasoning trace is invisible to an answer-only verifier, and in vision wrong reasoning and right answers often coexist. ([slides 51](slides.md#slide-51) and [52](slides.md#slide-52), [49:01](https://youtu.be/CxhrjQuGEuE?t=2941))

</details>

### Q10

What two-level picture for robots does the lecture end with?

- A) Explicit, RL-trained reasoning for high-level planning and task decomposition; for low-level control, reasoning used during training to improve the VLA and dropped at inference for speed.
- B) No reasoning anywhere.
- C) Reasoning only at the low level.
- D) Replace VLAs with language models.

<details><summary>Answer</summary>

**A.** Slide 53's bullets say exactly this: test-time reasoning helps high-level planning, while continuous control internalises the reasoning ECoT-Lite style. ([slide 53](slides.md#slide-53), [53:01](https://youtu.be/CxhrjQuGEuE?t=3181))

</details>

## Guest talk

### Context

Archit Sharma is a research scientist at Google DeepMind working on post-training and reasoning. He is one of the creators of DPO (direct preference optimisation), a standard way to fine-tune language models on human preferences, and he led the development of Gemini Deep Think, which reached gold-medal standard on International Mathematical Olympiad problems. This talk is the language-model side of the lecture: what test-time compute has done for LLMs, and what is still missing.

### Extra comments

**Two axes of compute.** Until about 2024, model improvements came mostly from more pre-training compute; since then, increasingly from letting the model think longer at inference. Accuracy rises predictably with thinking budget: on AIME 2025 (a high-school math competition) it rose by more than 20 points in the Gemini 2.5 report he shows. ([02:01](https://youtu.be/oBEkY6NeE_o?t=121), [04:02](https://youtu.be/oBEkY6NeE_o?t=242))

**What counts as test-time compute.** Anything that spends more than one forward pass on an answer: search (AlphaGo), a chain of thought, many attempts in parallel with a selector, revising a previous answer, tool calls, checking its own work. Coding agents and deep-research agents are test-time compute wrapped in a harness. ([06:01](https://youtu.be/oBEkY6NeE_o?t=361), [12:00](https://youtu.be/oBEkY6NeE_o?t=720))

**Reasoning that grows by itself.** In RL with verifiable rewards (math answers you can check, code with unit tests), the average response length grows during training while accuracy rises: the model discovers that longer thinking earns more reward. Nobody sets a target length. ([10:00](https://youtu.be/oBEkY6NeE_o?t=600), [11:02](https://youtu.be/oBEkY6NeE_o?t=662))

**Deep Think's curve.** On hard proof problems, proof accuracy (judged by experts) rose from about 40% to about 90% as inference compute grew on a log scale, with the model working for hours per problem. He also shows gains on tasks that cannot be verified automatically, like generating a detailed 3D web scene, so the benefit is not limited to math. ([14:01](https://youtu.be/oBEkY6NeE_o?t=841), [16:01](https://youtu.be/oBEkY6NeE_o?t=961))

**Thinking as depth.** His view: generating more tokens before answering is a way to increase the depth of computation, like adding layers, but at inference. An open question is whether tokens are the best way to buy depth, or whether reasoning in continuous latent space would be more efficient; in the Q&A he says latent reasoning is harder to get working but people are trying. ([19:01](https://youtu.be/oBEkY6NeE_o?t=1141), [24:00](https://youtu.be/oBEkY6NeE_o?t=1440))

**Why scaling RL on verifiable problems is hard.** Training must stay stable for a long time (small mismatches between how the training and inference systems compute probabilities can grow until training diverges). And it is data-limited: once the model reliably solves easy problems, they give no signal (the all-correct group in GRPO has zero advantage), while hard problems like proofs are harder to verify and easier to game. ([28:00](https://youtu.be/oBEkY6NeE_o?t=1680), [29:00](https://youtu.be/oBEkY6NeE_o?t=1740))

**What he thinks is still missing.** Knowledge discovery: deciding which problems matter, not only solving well-posed ones (he mentions the proposed "Einstein test": rediscover general relativity from data up to the early 20th century). And learning from experience: agents that build intuition and shortcuts instead of reasoning from first principles every time. Robots face both, and have the extra problem of the lecture: verifiers for physical success are scarce. ([22:01](https://youtu.be/oBEkY6NeE_o?t=1321), [23:01](https://youtu.be/oBEkY6NeE_o?t=1381))

### Going further

- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), the tennis-ball example he walks through.
- Rafailov, Sharma, Mitchell et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), his best-known paper.
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948), for the response-length curve.

## Guest talk quiz

### Q1

Which of these counts as test-time compute in Sharma's sense?

- A) Only training a bigger model.
- B) Only longer pre-training.
- C) Any procedure that spends more than a single forward pass before answering: search, chains of thought, parallel attempts with selection, iterative refinement, tool use.
- D) Quantising the model.

<details><summary>Answer</summary>

**C.** A and B scale training compute. Test-time compute is spent per query, at inference. ([06:01](https://youtu.be/oBEkY6NeE_o?t=361), [12:00](https://youtu.be/oBEkY6NeE_o?t=720))

</details>

### Q2

Which AlphaGo result does he use to introduce test-time compute?

- A) The same network rose from about 3000 Elo without search to about 5000 with test-time search capped at a few seconds per move.
- B) A bigger network beat the champion without search.
- C) Training on more human games was the key.
- D) RL alone, without any search, reached superhuman play first.

<details><summary>Answer</summary>

**A.** Same weights, very different strength, purely from search at inference. He notes that by about 3500 Elo the system is already superhuman. ([07:00](https://youtu.be/oBEkY6NeE_o?t=420))

</details>

### Q3

What happens to the average response length during RL training with verifiable rewards, in the DeepSeek plot he shows?

- A) It shrinks.
- B) It stays constant.
- C) It fluctuates randomly.
- D) It grows: the model discovers that thinking longer earns more reward, so reasoning emerges.

<details><summary>Answer</summary>

**D.** Accuracy and response length rise together over training, without any instruction to think longer. ([11:02](https://youtu.be/oBEkY6NeE_o?t=662))

</details>

### Q4

Why does he call RL on verifiable problems "data-limited"?

- A) There are not enough GPUs.
- B) The supply of verifiable problems is finite: once the model solves the easy ones they stop giving signal, and harder domains such as proofs are harder to verify and easier to game.
- C) It needs human labels for every answer.
- D) Tokens are too expensive.

<details><summary>Answer</summary>

**B.** Problems the model always solves carry no learning signal, so training needs a steady supply of problems at the edge of its ability, with trustworthy verifiers. ([29:00](https://youtu.be/oBEkY6NeE_o?t=1740), [30:00](https://youtu.be/oBEkY6NeE_o?t=1800))

</details>

### Q5

What does Sharma see as still missing from today's reasoning systems?

- A) More pre-training data.
- B) Longer context windows.
- C) Knowledge discovery (working out which problems matter) and agents that improve with experience instead of reasoning from scratch every time.
- D) Nothing; scaling test-time compute will solve everything.

<details><summary>Answer</summary>

**C.** He is explicit that scaling inference compute will drive a lot of progress but will not solve every problem; these two are his picks for what is missing. ([22:01](https://youtu.be/oBEkY6NeE_o?t=1321), [23:01](https://youtu.be/oBEkY6NeE_o?t=1381))

</details>
