# Week 09 study guide: Generalist Robot Policies

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Week 1's "three ingredients" slide is the outline of this lecture. You also need week 6's diffusion and flow-matching action heads, and week 7's VLMs, action tokenization and FAST. The slides call this "Lecture 10"; it is the week 9 lecture (see the transcriber note on the title slide).

### Extra comments

**Labs versus institutions: the Open X-Embodiment numbers reconciled.** Week 1 quoted the paper's abstract (21 institutions), Ted Xiao said "more than 30 labs", and slide 9 says 34 research labs. The summary figure on slide 12 settles it: 34 research labs across 21 institutions, 22 embodiments, 60 datasets, over a million episodes. One university can host several labs. ([09:00](https://youtu.be/dtofzDY9zuo?t=540), [slides 9](slides.md#slide-9) and [12](slides.md#slide-12))

**What the RT-1-X chart does and does not show.** Averaged over five labs, one model trained on everyone's data beat each lab's own method (41 to 63), and it beat RT-1 trained on that lab's data alone. That is positive transfer. But not everywhere: on the Autolab UR5 it did worse than the original method. The paper's broader finding was that the small RT-1-X helped most on small datasets, while on the large ones a bigger model (RT-2-X) was needed to benefit: capacity matters once there is enough data. ([10:00](https://youtu.be/dtofzDY9zuo?t=600), [slide 10](slides.md#slide-10))

**Octo's readout tokens are passive by design.** Octo's attention is block-wise causal: task tokens see only task tokens, observation tokens see the task and earlier observations, and readout tokens see everything before them, but nothing attends to the readouts. So adding a new readout and a new action head for finetuning does not change how any existing token is computed: the pre-trained representation stays intact and the new head just reads from it. It is the same role as BERT's [CLS] token. ([16:00](https://youtu.be/dtofzDY9zuo?t=960), [slides 21](slides.md#slide-21) and [27](slides.md#slide-27))

**The gripper lesson: normalise meaning, not just scale.** In some datasets 1 means open, in others 1 means closed, and some record open/close changes rather than states. Mixed together, the same number commands opposite actions, and the lecture says many early Octo failures were premature grasps. Normalising each dimension's range (week 7's quantile trick) does nothing about this; you have to read each dataset's documentation. ([18:00](https://youtu.be/dtofzDY9zuo?t=1080), [slide 23](slides.md#slide-23))

**Teacher-forced token accuracy is a convergence check, not a success rate.** "Did the model pick the exact bin for the next action token, given the true previous tokens?" reaching about 95% means training has converged enough to be worth a robot evaluation. Because it is measured with teacher forcing, it says nothing about closed-loop success, where errors compound (week 3, week 7). ([38:00](https://youtu.be/dtofzDY9zuo?t=2280), [slide 40](slides.md#slide-40))

**Padding versus packing, with numbers.** If dataset A has one camera and dataset B has three, a padded batch with half of each is about one third zeros in the image columns: compute spent on nothing. Packing concatenates samples into one long sequence and keeps a list of where each starts (`cu_seqlens`, the format FlashAttention's variable-length kernels take). Two things then must change: the attention mask becomes block-diagonal so samples cannot see each other, and position ids restart at 0 for every sample. ([40:01](https://youtu.be/dtofzDY9zuo?t=2401), [slides 41](slides.md#slide-41) to [43](slides.md#slide-43))

**The shuffle buffer is week 4's replay buffer argument again.** Robot data is stored as trajectories, so reading it sequentially gives batches of nearly identical consecutive frames. A shuffle buffer mixes many trajectories before batching; if it is too small, batches stay correlated and SGD's i.i.d. assumption fails, which is exactly why DQN needed a replay buffer. The lecture's trick for growing it to 500,000 samples was to shuffle before decoding images, so the buffer holds compressed frames. ([43:01](https://youtu.be/dtofzDY9zuo?t=2581), [44:01](https://youtu.be/dtofzDY9zuo?t=2641), [slide 44](slides.md#slide-44))

**Two answers to "which robot am I?"** CrossFormer gives each embodiment its own action head and the user picks the head; nothing is ambiguous, but every new robot needs a head. $\pi_0$ pads every robot's action into one fixed-length vector and lets the model infer the embodiment from what it sees, its proprioception and the instruction; one head serves all, and the lecture's aside applies: you cannot ask a Roomba to cut sushi, so tasks themselves carry embodiment information. ([46:01](https://youtu.be/dtofzDY9zuo?t=2761), [50:01](https://youtu.be/dtofzDY9zuo?t=3001), [slides 45](slides.md#slide-45) and [47](slides.md#slide-47))

**Why the current recipe has a stop-gradient.** Training the VLM directly with the action expert's flow-matching loss tends to erode what it knew about language and the web (Scott Reed made the same point in week 8). The fix on slide 48: the VLM learns about actions only through discrete FAST tokens with ordinary next-token prediction, mixed with web data, and the flow-matching expert reads the VLM's features but cannot send gradients back. Physical Intelligence calls this "knowledge insulation". ([52:00](https://youtu.be/dtofzDY9zuo?t=3120), [slide 48](slides.md#slide-48))

**What SIMPLER can and cannot tell you.** Two things make sim results track real ones: system identification, so the simulated arm moves like the real one under the same commands, and visual matching, pasting the simulated objects over a green-screened real background. Even then, the lecture stresses, the absolute success rate means little; what transfers is the ranking of policies and how much each degrades under a distribution shift (camera pose: $-0.38$ real against $-0.39$ simulated). Gupta's week 2 talk made the same point with Pearson correlation and MMRV. ([59:01](https://youtu.be/dtofzDY9zuo?t=3541), [01:01:00](https://youtu.be/dtofzDY9zuo?t=3660), [slides 52](slides.md#slide-52) to [54](slides.md#slide-54))

### Going further

- Octo Model Team, [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213), and Doshi et al., [Scaling Cross-Embodied Learning](https://arxiv.org/abs/2408.11812) (CrossFormer).
- Physical Intelligence, [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054), and Driess et al., [Knowledge Insulating Vision-Language-Action Models](https://arxiv.org/abs/2505.23705), for the current recipe on slide 48.
- Li et al., [Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941) (SIMPLER).
- The open-source $\pi_0$ and $\pi_{0.5}$ code and weights: [openpi](https://github.com/Physical-Intelligence/openpi).

## Lecture quiz

### Q1

According to the lecture, what made the first generalist robot policies possible?

- A) Aggregating existing robot datasets from many labs into one common format.
- B) Collecting a huge new dataset on a single robot.
- C) Training only in simulation.
- D) Learning only from internet video.

<details><summary>Answer</summary>

**A.** Robot data was scarce, but scattered across labs; Open X-Embodiment's contribution was gathering it and converting it to one format, giving over a million episodes from 22 embodiments. ([slide 8](slides.md#slide-8), [08:00](https://youtu.be/dtofzDY9zuo?t=480))

</details>

### Q2

In the RT-1-X comparison, the mean success rose from 41 (each lab's original method) to 63. What does the chart support?

- A) The original methods were broken.
- B) Only the largest datasets benefited.
- C) Positive transfer: training on many robots' data helped on average, though not on every evaluation (the Autolab UR5 got worse).
- D) RT-1 trained on a single lab's data is enough.

<details><summary>Answer</summary>

**C.** RT-1-X beat both the original methods and single-dataset RT-1 on average. The UR5 drop is a reminder that "on average" hides losers, and the paper found larger models were needed to benefit on the larger datasets. ([slide 10](slides.md#slide-10), [10:00](https://youtu.be/dtofzDY9zuo?t=600))

</details>

### Q3

Why did Octo convert all gripper actions to one convention?

- A) To save memory.
- B) To speed up training.
- C) To support new robots.
- D) Datasets disagreed on what the numbers meant (1 = open in some, 1 = closed in others, or open/close changes), so the same value commanded opposite actions; many early failures were premature grasps.

<details><summary>Answer</summary>

**D.** Scaling each dimension to a common range cannot fix a dimension whose meaning is flipped. Semantic alignment has to be done per dataset. ([slide 23](slides.md#slide-23), [19:01](https://youtu.be/dtofzDY9zuo?t=1141))

</details>

### Q4

How do Octo's readout tokens work, and why does it matter for finetuning?

- A) Every token attends to them, so they shape the whole representation.
- B) They attend to the task and observation tokens, but nothing attends to them, so new readouts and action heads can be added without changing how the pre-trained tokens are computed.
- C) They store the action history.
- D) They are language tokens.

<details><summary>Answer</summary>

**B.** Readouts only read. That is what lets Octo add a new action head (slide 27) for a new robot while leaving the pre-trained representation untouched. ([slides 21](slides.md#slide-21) and [27](slides.md#slide-27), [16:00](https://youtu.be/dtofzDY9zuo?t=960))

</details>

### Q5

A VLA's training curve shows action-token accuracy near 95%. What does that mean?

- A) The robot will succeed 95% of the time.
- B) The model is overfitting.
- C) Under teacher forcing, the model picks the exact action bin 95% of the time; it signals convergence and readiness for real-robot evaluation, not closed-loop success.
- D) Nothing useful.

<details><summary>Answer</summary>

**C.** The lecture's rule of thumb: do not spend robot time before it is near 95%. But teacher-forced accuracy ignores compounding errors, so success has to be measured on the robot. ([slide 40](slides.md#slide-40), [38:00](https://youtu.be/dtofzDY9zuo?t=2280))

</details>

### Q6

What must change when you pack several samples into one long sequence instead of padding them?

- A) The attention mask must be block-diagonal (samples cannot attend to each other), and position ids must restart at 0 for each sample.
- B) You need more padding.
- C) Only a causal mask is needed.
- D) The batch size must double.

<details><summary>Answer</summary>

**A.** Without the mask, sample 1 would attend to sample 3's images; without the position reset, later samples would see positions they were never trained on. ([slide 43](slides.md#slide-43), [42:00](https://youtu.be/dtofzDY9zuo?t=2520))

</details>

### Q7

What goes wrong if the shuffle buffer in a sequential dataloader is too small?

- A) Training becomes slower.
- B) It uses too much memory.
- C) Images cannot be decoded.
- D) Batches are full of consecutive, highly correlated frames from a few trajectories, so samples are far from i.i.d. and gradient estimates are biased and noisy.

<details><summary>Answer</summary>

**D.** It is the problem DQN's replay buffer solved in week 4. A large buffer costs memory (B is the price of the fix, not the problem). ([slide 44](slides.md#slide-44), [43:01](https://youtu.be/dtofzDY9zuo?t=2581))

</details>

### Q8

How does a $\pi_0$-style model know which robot it is controlling at test time?

- A) The user selects an action head in the prompt.
- B) It infers the embodiment from the observations, proprioception and task; all robots share one padded action vector.
- C) It picks at random.
- D) There is a separate model for each robot.

<details><summary>Answer</summary>

**B.** A is the CrossFormer-style answer. The $\pi_0$ style keeps one action expert and lets the inputs reveal the body. ([slide 47](slides.md#slide-47), [50:01](https://youtu.be/dtofzDY9zuo?t=3001))

</details>

### Q9

In the current VLA recipe on slide 48, why is there a stop-gradient between the action expert and the VLM?

- A) So the flow-matching gradients cannot overwrite the VLM's pre-trained knowledge; the VLM learns about actions through FAST tokens and next-token prediction instead.
- B) To make inference faster.
- C) To save GPU memory.
- D) To stop the action expert from learning.

<details><summary>Answer</summary>

**A.** The expert still learns (D is wrong); it just cannot push gradients into the backbone. That keeps the VLM's language and web knowledge intact while it still learns robot actions in its own discrete form. ([slide 48](slides.md#slide-48), [53:01](https://youtu.be/dtofzDY9zuo?t=3181))

</details>

### Q10

What do SIMPLER evaluations measure reliably?

- A) Absolute real-world success rates.
- B) How fast policies run.
- C) Relative performance: policy rankings, and how much policies degrade under distribution shifts, track the real-world results.
- D) Sample efficiency.

<details><summary>Answer</summary>

**C.** The lecture is explicit that the absolute rate is not the point; pairwise comparisons and sensitivities are. Getting even that requires system identification and visual matching. ([slides 52](slides.md#slide-52) and [53](slides.md#slide-53), [01:01:00](https://youtu.be/dtofzDY9zuo?t=3660))

</details>

## Guest talk

### Context

Quan Vuong is a co-founder of Physical Intelligence (the captions call him "Quan Pham" in the introduction). He did his PhD at UC San Diego with Henrik Christensen and Hao Su, and at Google DeepMind started and co-led RT-X and Open X-Embodiment, this lecture's dataset. The talk presents $\pi_{0.7}$ (the captions write it as "PIO7", "PILE-7", "Pyro 7" and even "PyTorch 7"): one checkpoint that matches task-specific specialists.

### Extra comments

**A puzzle to start with.** In Physical Intelligence's earlier RL work, a policy trained by imitation on teleoperation data ran slower than the teleoperators themselves, and RL made it faster than them. A plausible reading, mine rather than his: imitation copies the average of hesitant and confident demonstrations, so pauses are learned too. The question he sets up is how one generalist can be as fast and reliable as specialists tuned with RL. ([04:02](https://youtu.be/pzolgvyWEFY?t=242), [05:02](https://youtu.be/pzolgvyWEFY?t=302))

**Metadata conditioning: use bad data instead of discarding it.** Each training episode comes with labels for its speed and quality, given to the model as inputs. Without them, adding the lowest-quality 20% of data made the policy worse; with them, the same data made it much better. At test time you simply ask for fast, high-quality behaviour. It is the same family of ideas as classifier-free guidance (week 6), return-conditioned policies such as the Decision Transformer, and the advantage-conditioned $\pi^*_{0.6}$ that came up in Wagenmaker's week 5 Q&A. ([06:00](https://youtu.be/pzolgvyWEFY?t=360), [11:00](https://youtu.be/pzolgvyWEFY?t=660))

**"Every evaluation is a generalization evaluation."** His explanation for why the generalist sometimes beats specialists on their own tasks: in the real world the lighting, the objects and their placement always change a little between evaluations, and the generalist is more robust to that. Read the comparisons with this in mind; they are not run under identical conditions, however hard one tries. ([09:03](https://youtu.be/pzolgvyWEFY?t=543), [10:02](https://youtu.be/pzolgvyWEFY?t=602))

**The cross-embodiment result is the most surprising one.** Shirt-folding data came from one kind of bimanual robot only, yet the model folded shirts on a bimanual UR5 station that had no shirt-folding data at all: a hard task on an unseen task-robot combination. Compare Scott Reed's "data islands" scepticism in week 8. ([12:00](https://youtu.be/pzolgvyWEFY?t=720), [13:00](https://youtu.be/pzolgvyWEFY?t=780))

**Low level and high level.** The low-level model takes the task, an optional subtask, the metadata and optionally a goal; a high level (a person, a high-level policy or a world model) proposes subtasks and goals but cannot produce motor actions itself. Coaching the robot step by step in language let it do tasks and handle objects that were not in its data. ([06:00](https://youtu.be/pzolgvyWEFY?t=360), [14:00](https://youtu.be/pzolgvyWEFY?t=840))

**What deployment really requires.** With partner companies the model folds unseen clothes in a real laundromat and packs customer orders into pouches from morning to night. The key requirements he names: recovering from mistakes, remote human intervention when it fails, and a low enough intervention rate that the operation breaks even financially. He also says the open-source $\pi_0$ and $\pi_{0.5}$ checkpoints are the same ones researchers use internally. ([17:00](https://youtu.be/pzolgvyWEFY?t=1020), [20:02](https://youtu.be/pzolgvyWEFY?t=1202))

**Three Q&A answers.** Multi-finger hands: for studying intelligence you need scale, so hardware must be cheap, reliable and easy to teleoperate, and today's hands give two of the three; few of their thousands of tasks needed more than two fingers. Home robots: the bottleneck is model capability, plus very different preferences across households. VLAs versus video models: undecided; model sizes will grow from single-digit billions to tens of billions, and both families need to be scaled before anyone knows. ([22:02](https://youtu.be/pzolgvyWEFY?t=1322), [31:03](https://youtu.be/pzolgvyWEFY?t=1863), [35:00](https://youtu.be/pzolgvyWEFY?t=2100))

### Going further

- Physical Intelligence, [$\pi^*_{0.6}$: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759), the RL and advantage-conditioning predecessor.
- Chen et al., [Decision Transformer](https://arxiv.org/abs/2106.01345), the classic "condition on the outcome you want" policy.
- Physical Intelligence's site, [pi.website](https://www.pi.website), where the $\pi_{0.7}$ write-up and the deployment blog post he shows are published.

## Guest talk quiz

### Q1

What is the main claim Vuong makes for $\pi_{0.7}$?

- A) It is the largest robot model so far.
- B) A single checkpoint matches or beats task-specific specialists (fine-tuned or RL-trained) across many hard tasks.
- C) It is the first home robot.
- D) It works without language input.

<details><summary>Answer</summary>

**B.** Previously, top performance meant post-training a separate specialist per task. One model for all tasks is far easier to manage and scale. ([01:01](https://youtu.be/pzolgvyWEFY?t=61), [08:00](https://youtu.be/pzolgvyWEFY?t=480))

</details>

### Q2

Why can a generalist outperform specialists on the specialists' own tasks, according to Vuong?

- A) It uses more compute at test time.
- B) The comparison is unfair.
- C) It runs on bigger GPUs.
- D) Every real-world evaluation is partly a generalization test, since lighting, objects and placement always shift a little, and the generalist is more robust to such shifts.

<details><summary>Answer</summary>

**D.** He stresses they tried to keep the data comparable; the advantage comes from robustness to the small changes that are unavoidable in physical evaluation. ([09:03](https://youtu.be/pzolgvyWEFY?t=543))

</details>

### Q3

What is the "metadata" in $\pi_{0.7}$, and what did it change?

- A) Labels for each episode's speed and quality, given as input; with them, adding the lowest-quality data improved the policy, while without them it hurt.
- B) Robot serial numbers.
- C) Timestamps of each frame.
- D) Camera calibration parameters.

<details><summary>Answer</summary>

**A.** Telling the model what kind of data it is looking at lets it learn from bad examples without imitating them; at test time it is asked for the good kind. ([11:00](https://youtu.be/pzolgvyWEFY?t=660), [12:00](https://youtu.be/pzolgvyWEFY?t=720))

</details>

### Q4

What was the cross-embodiment result he highlighted?

- A) The policy only worked on the robot it was trained on.
- B) Each new robot needed new data for every task.
- C) Trained with shirt-folding data from one bimanual robot only, the model folded shirts on a bimanual UR5 station that had no shirt-folding data.
- D) It worked only in simulation.

<details><summary>Answer</summary>

**C.** A hard task on an unseen task-robot combination, which surprised the team. ([13:00](https://youtu.be/pzolgvyWEFY?t=780))

</details>

### Q5

Why does Physical Intelligence mostly avoid multi-finger hands for now?

- A) Hands are useless for manipulation.
- B) Studying intelligence needs scale, so hardware must be cheap, reliable and easy to teleoperate, and current hands offer at most two of those three.
- C) Hands are too slow.
- D) Hands are not allowed in the lab.

<details><summary>Answer</summary>

**B.** For the price of one dexterous hand you can build several two-finger systems. He adds that few of their many tasks have needed more than a two-finger gripper, and that they will revisit hands when the hardware improves. ([22:02](https://youtu.be/pzolgvyWEFY?t=1322))

</details>
