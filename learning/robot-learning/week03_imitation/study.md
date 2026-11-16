# Week 03 study guide: Imitation Learning

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Supervised learning with a squared-error loss, what a probability distribution over a continuous variable is, and last week's notation ($s_t$, $o_t$, $a_t$, $\pi_\theta$). For the last third, it helps to have seen a variational autoencoder once: an encoder that outputs a Gaussian, a sampled latent $z$, a decoder, and a loss with a reconstruction term and a KL term. Homework 3 is on this lecture.

### Extra comments

**Squared error secretly assumes one Gaussian.** Minimising $\|a - \hat a\|^2$ is the same as maximising the likelihood of a Gaussian with fixed variance centred on $\hat a$, because $-\log \mathcal{N}(a; \hat a, \sigma^2 I) = \|a - \hat a\|^2 / 2\sigma^2 + \text{const}$. One Gaussian has one peak, which is the whole reason for the multimodality problem at the end of the lecture. Many robot policies use the L1 loss instead, which is the same story with a Laplace distribution. ([04:00](https://youtu.be/Ef4R5s1LqoQ?t=240), [slide 4](slides.md#slide-4))

**Where $\epsilon T^2$ comes from, in two lines.** Suppose that once the policy leaves the expert's states, it can be wrong on every remaining step. The first mistake happens at step $t$ with probability about $\epsilon$, and then costs up to $T - t$. Summing over $t$ gives about $\epsilon \sum_t (T - t) \approx \epsilon T^2 / 2$. If the policy instead is trained on its own states (DAgger), each step costs about $\epsilon$ on its own, so the total is $\epsilon T$. The bound is a worst case; real policies often recover. The original analysis is Ross and Bagnell (2010); the slide cites the 2011 DAgger paper, which restates it. ([15:01](https://youtu.be/Ef4R5s1LqoQ?t=901), [slide 11](slides.md#slide-11))

**The side-camera trick is older than NVIDIA.** Pomerleau's ALVINN, a neural-network driver from around 1990, already created shifted views of the road by transforming the camera image and labelling them with the steering that would bring the car back. NVIDIA's 2016 version used real side cameras. The forest-trail drone (Giusti et al., 2016, IDSIA in Lugano with the University of Zurich) applied it to a hiker wearing a head rig of three cameras: images from the side cameras were labelled "turn back toward the trail". ([12:01](https://youtu.be/Ef4R5s1LqoQ?t=721), [slides 9](slides.md#slide-9) and [10](slides.md#slide-10))

**The paradox on slide 13, put to use.** "BC works better if the data has more mistakes and recoveries." DART (Laskey et al., 2017) does this on purpose: inject noise into the expert's controls while recording, so the demonstrations visit slightly-off states and show how to come back, without the expert ever labelling the robot's own rollouts. ([20:00](https://youtu.be/Ef4R5s1LqoQ?t=1200), [slide 13](slides.md#slide-13))

**Why hindsight labelling is awkward for humans.** In DAgger the expert labels states the robot visited, after the fact, often by watching a video. People are poor at saying "steer 3 degrees left" without feeling the result of their action, and if the robot's action space differs from what the human controls, it is worse. Human-gated DAgger (Kelly et al., 2018) keeps the human in control of the timing: they take over when things go wrong, so the corrections are natural demonstrations from states the policy actually reached. The open problem the slide names, "when to intervene", leads to policies that estimate their own uncertainty and ask for help. ([21:01](https://youtu.be/Ef4R5s1LqoQ?t=1261), [slide 14](slides.md#slide-14))

**Reading the ranking loss on slide 16.** $S(o \mid r)$ is how well object $o$ matches expression $r$. Each hinge term says: the correct pair should score at least a margin $m_1$ higher than a wrong pair, where wrong means the same object with another object's description, or another object with this description. At test time that margin becomes the ambiguity test: if two objects score within $m$ of each other, the robot generates a description of each and asks which one you meant. It is the "detect when to ask" idea from slide 14, applied to language. ([23:00](https://youtu.be/Ef4R5s1LqoQ?t=1380), [slide 16](slides.md#slide-16))

**The "copycat" version of causal confusion.** Give a policy its past actions and it can often predict the next action just by repeating the last one, because consecutive actions are almost identical. The training loss looks great; at test time the policy ignores the camera and keeps doing whatever it did before. Wen et al. (2020) call this the copycat problem. The lecture's gripper-force example is the same failure with a sensor instead of past actions. ([30:00](https://youtu.be/Ef4R5s1LqoQ?t=1800), [slide 21](slides.md#slide-21))

**Why modes average, and the four fixes.** Half the demos go left of the tree and half go right; the mean of "left" and "right" is "straight". Any loss that predicts one point per input does this. The four fixes on slides 24 to 27 all predict a distribution instead:

- a mixture of Gaussians (you pick the number of modes in advance),
- discretisation, which turns regression into classification over bins,
- diffusion, which generates a sample by removing noise step by step,
- a latent variable that selects the mode before decoding.

Behavior Transformers (Shafiullah et al., 2022) is a well-known discretise-plus-offset version, and the ACT policy that runs the ALOHA robots is a conditional VAE. ([34:00](https://youtu.be/Ef4R5s1LqoQ?t=2040), [slide 23](slides.md#slide-23))

**The numbers behind "exponential" on slide 25.** A 7-D action with 256 bins per dimension has $256^7 \approx 7 \times 10^{16}$ joint cells: no softmax can have that many outputs. Predicting one dimension at a time, each conditioned on the ones before, needs 7 softmaxes of 256. That factorisation is the chain rule of probability, exactly how a language model predicts words, and it is how the first VLAs (RT-2) emit actions as text tokens. ([36:03](https://youtu.be/Ef4R5s1LqoQ?t=2163), [slide 25](slides.md#slide-25))

**Posterior at training time, prior at test time.** In the play-data model, the posterior $q_\phi(z \mid \tau, c)$ sees the whole demonstration and encodes "which way this demo went". The prior $p_\theta(z \mid c)$ sees only the current and goal states. At test time there is no demonstration, so $z$ is sampled from the prior; the KL term in the loss is what trains the prior to propose plans the posterior would have recognised. ACT simplifies further and just sets $z$ to the prior's mean at test time. ([45:00](https://youtu.be/Ef4R5s1LqoQ?t=2700), [slides 36](slides.md#slide-36) and [40](slides.md#slide-40))

**Why play data plus 1% language works.** Play is cheap: an operator does whatever they like for hours with no resets and no task list, and any window of it is a valid "reach this goal state" example (hindsight relabelling: the goal is just where the window ended). Language labels are expensive, so only a small slice gets them, and image goals and language goals are mapped into one shared goal space (Lynch and Sermanet, 2021). That is the line of work the lecturer's CALVIN benchmark is built on. ([44:00](https://youtu.be/Ef4R5s1LqoQ?t=2640), [slide 38](slides.md#slide-38))

### Going further

- Ross, Gordon and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (DAgger). Section 3 has the $\epsilon T^2$ versus $\epsilon T$ argument.
- de Haan et al., [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979), and Wen et al., [Fighting Copycat Agents in Behavioral Cloning from Observation Histories](https://arxiv.org/abs/2010.14876).
- Lynch et al., [Learning Latent Plans from Play](https://arxiv.org/abs/1903.01973), the model on slides 36 to 40.
- Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (ALOHA and ACT): a conditional VAE policy that works on real hardware, with action chunking, which returns in week 6.
- Levine's CS 285, lecture 2 (imitation learning), covers the same ground with more derivation: [course page](https://rail.eecs.berkeley.edu/deeprlcourse/).

## Lecture quiz

### Q1

Training a behaviour-cloning policy with the squared error $\|a - \hat a\|^2$ is the same as maximum likelihood when the policy is:

- A) A mixture of Gaussians.
- B) A single Gaussian with fixed variance, centred on $\hat a$.
- C) A categorical distribution over action bins.
- D) A diffusion model.

<details><summary>Answer</summary>

**B.** $-\log \mathcal{N}(a; \hat a, \sigma^2 I) = \|a - \hat a\|^2 / 2\sigma^2 + \text{const}$. A single Gaussian has one peak, so this policy can only predict the average of the demonstrated actions: the mode-averaging failure on slide 23. ([slide 4](slides.md#slide-4), [slide 23](slides.md#slide-23))

</details>

### Q2

Why did NVIDIA's 2016 end-to-end driving policy work despite distribution shift?

- A) The network was much larger than earlier ones.
- B) It was trained with DAgger.
- C) Driving has no distribution shift.
- D) Side cameras provided images of "drifted" views labelled with the steering that brings the car back.

<details><summary>Answer</summary>

**D.** The left camera sees what the centre camera would see if the car had drifted left, so its image is labelled "steer right" (and vice versa). That adds recovery examples for states the expert never visited, without any extra driving. ([slide 9](slides.md#slide-9), [12:01](https://youtu.be/Ef4R5s1LqoQ?t=721))

</details>

### Q3

A policy makes a mistake with probability $\epsilon = 0.001$ per step on states like its training data. Over a horizon of $T = 100$ steps, how do the worst-case bounds for behaviour cloning and DAgger compare?

- A) BC: about $\epsilon T^2 = 10$ mistakes; DAgger: about $\epsilon T = 0.1$. BC's bound is $T = 100$ times larger.
- B) Both about 0.1.
- C) BC: 0.1; DAgger: 10.
- D) BC: 100; DAgger: 1.

<details><summary>Answer</summary>

**A.** $\epsilon T^2 = 0.001 \times 10^4 = 10$ and $\epsilon T = 0.001 \times 100 = 0.1$. The ratio of the two bounds is always $T$, so the longer the task, the more training on the policy's own states matters. ([slide 11](slides.md#slide-11), [slide 13](slides.md#slide-13))

</details>

### Q4

Which step of DAgger is the hard one in practice?

- A) Rolling out the current policy.
- B) Adding the new pairs to the dataset.
- C) Getting the expert to label the states the policy visited, in hindsight.
- D) Retraining the policy.

<details><summary>Answer</summary>

**C.** The other steps are ordinary engineering. Asking a human "what would you have done here?" for states they did not drive into is slow, tiring and error-prone, which is why human-gated DAgger lets the person take over in real time instead. ([20:00](https://youtu.be/Ef4R5s1LqoQ?t=1200), [slide 13](slides.md#slide-13))

</details>

### Q5

What does human-gated DAgger change compared with DAgger?

- A) It needs no human at all.
- B) It uses only the original demonstrations.
- C) It trains a reward function.
- D) The human takes over when the policy starts to fail, and the partial demonstration from that point on is added to the data.

<details><summary>Answer</summary>

**D.** Corrections arrive exactly at the states where the policy fails, and the human acts naturally instead of labelling afterwards. The cost moves elsewhere: someone has to watch the robot and decide when to step in. ([slide 14](slides.md#slide-14), [21:01](https://youtu.be/Ef4R5s1LqoQ?t=1261))

</details>

### Q6

Why can adding observation history to a BC policy make it worse?

- A) The policy can latch onto spurious correlations in the history (like a past force reading or its own previous action) instead of the cause of the expert's action.
- B) History makes the network too small.
- C) History removes the Markov property from the environment.
- D) History cannot be fed to a neural network.

<details><summary>Answer</summary>

**A.** That is causal confusion. In the drawer example, a 10 N grip reading always preceded pulling in the data, so the policy learned "10 N means pull"; when the gripper slipped and read 2 N, it never pulled. History helps when the expert really used past information, and hurts when it offers an easier shortcut. ([slide 21](slides.md#slide-21), [30:00](https://youtu.be/Ef4R5s1LqoQ?t=1800))

</details>

### Q7

Half the demonstrations pass a tree on the left, half on the right. What does a deterministic policy trained with squared error do?

- A) Passes on the left, the first mode it saw.
- B) Goes straight into the tree, the average of the two modes.
- C) Picks left or right at random each time.
- D) Stops in front of the tree.

<details><summary>Answer</summary>

**B.** The squared-error minimiser is the mean of the demonstrated actions for that input, and the mean of "left" and "right" is "straight". Choosing randomly between modes (C) is what a multimodal policy, such as a mixture, a diffusion model or a latent-variable model, can do and a single-output one cannot. ([slide 23](slides.md#slide-23), [34:00](https://youtu.be/Ef4R5s1LqoQ?t=2040))

</details>

### Q8

A 7-dimensional action has each dimension cut into 256 bins. Why predict the bins one dimension at a time with an autoregressive model instead of all at once?

- A) Autoregressive models are always more accurate.
- B) It avoids using a transformer.
- C) A single classifier over all combinations would need $256^7 \approx 7 \times 10^{16}$ outputs; one dimension at a time needs only 7 predictions over 256 bins.
- D) It makes the policy deterministic.

<details><summary>Answer</summary>

**C.** The chain rule $p(a_0, \dots, a_6) = \prod_i p(a_i \mid a_{<i}, s)$ keeps the full joint distribution, including the correlation between dimensions, while the output size grows linearly. The price is 7 sequential steps per action, which costs speed. The model on slide 25 is itself a transformer, so B is backwards. ([slide 25](slides.md#slide-25), [36:03](https://youtu.be/Ef4R5s1LqoQ?t=2163))

</details>

### Q9

In the play-data latent-plan model, where does the latent plan $z$ come from at test time?

- A) From the prior (plan proposal), which sees only the current state and the goal.
- B) From the posterior (plan recognition), which sees the whole demonstration.
- C) From the action decoder.
- D) From a fixed lookup table of tasks.

<details><summary>Answer</summary>

**A.** At test time there is no demonstration to feed the posterior. Training pulls the prior toward the posterior with the KL term, exactly so that the prior can propose plausible plans on its own when it has to. ([slide 37](slides.md#slide-37), [slide 40](slides.md#slide-40), [46:00](https://youtu.be/Ef4R5s1LqoQ?t=2760))

</details>

## Guest talk

### Context

Danfei Xu runs the Robot Learning and Reasoning Lab at Georgia Tech and is also a researcher at NVIDIA; he did his PhD at Stanford with Fei-Fei Li and Silvio Savarese. He calls the talk an alternative view of imitation learning: instead of collecting demonstrations by driving a robot, learn from people doing tasks naturally with their own hands, recorded from the eye level with smart glasses.

### Extra comments

**The "triangle" is a useful lens.** Data source, learning algorithm and physical system have to be designed together. Teleoperation couples them tightly: data exists only when a person acts through one specific robot, and the policy works only on that robot. More data means more robots and more operator hours. ([02:01](https://youtu.be/qvTP6T5oq1w?t=121), [04:00](https://youtu.be/qvTP6T5oq1w?t=240))

**Week 2's transforms, applied.** With glasses on, the camera moves with your head, so hand positions measured from the camera are in a moving frame. The glasses (Meta's Project Aria; the captions say "Aura") run SLAM, which gives the head's pose in a fixed world frame at every moment, and chaining $T^{\text{World}}_{\text{Hand}} = T^{\text{World}}_{\text{Head}}(t) \, T^{\text{Head}}_{\text{Hand}}(t)$ turns jittery head-relative hand tracks into stable action trajectories. They also mounted the same glasses on the robot so robot and human see the world through the same camera. ([08:02](https://youtu.be/qvTP6T5oq1w?t=482), [09:00](https://youtu.be/qvTP6T5oq1w?t=540))

**What co-training buys, and what it does not.** EgoMimic trains one behaviour-cloning policy on human and robot data together, treating the human as "another robot". Adding human data improves the robot a lot, partly because people demonstrate up to about ten times faster than teleoperation. But the robot only improves on tasks that also appear in its own data; a task shown only by humans does not transfer. ([11:00](https://youtu.be/qvTP6T5oq1w?t=660), [13:01](https://youtu.be/qvTP6T5oq1w?t=781))

**EgoBridge and optimal transport.** When they looked at the policy's internal representation, human and robot samples formed two separate clusters, so there was nothing to transfer through. Optimal transport finds the cheapest way to move one distribution onto another, and in its regularised (Sinkhorn) form it is differentiable, so it can be used as an extra training loss that pulls the two clusters together. The cost they use compares the shape of the action trajectories: a human and a robot doing the same thing should move similarly. ([15:01](https://youtu.be/qvTP6T5oq1w?t=901), [16:01](https://youtu.be/qvTP6T5oq1w?t=961))

**Flipping the data pyramid.** The usual picture puts cheap, plentiful data (human video) at the bottom and scarce robot data on top. His argument from a collaboration with Physical Intelligence: a stronger robot foundation model is what lets you learn from human data, because its representation already aligns the two. Better robots make human data more useful, not just the other way round. ([27:02](https://youtu.be/qvTP6T5oq1w?t=1622))

**What first-person video still misses.** Touch and force are not in the video (in the Q&A he says plainly that they do not estimate forces). And the reasons for decisions often lie minutes in the past or in years of experience: the chef turns to the other counter because of something he prepared earlier. That is the lecture's non-Markovian expert (slide 18) in a real setting. ([24:02](https://youtu.be/qvTP6T5oq1w?t=1442), [34:00](https://youtu.be/qvTP6T5oq1w?t=2040))

**EgoVerse.** A shared human-data effort across universities and companies, about 1,300 hours at the time of the talk and growing, validated by running the same learning pipeline on three different robots. ([29:01](https://youtu.be/qvTP6T5oq1w?t=1741))

### Going further

- Kareer et al., [EgoMimic: Scaling Imitation Learning via Egocentric Video](https://arxiv.org/abs/2410.24221), the first paper in the talk.
- Fu et al., [Mobile ALOHA](https://arxiv.org/abs/2401.02117), the teleoperated mobile-manipulation baseline he compares against.
- Chi et al., [Universal Manipulation Interface](https://arxiv.org/abs/2402.10329) (UMI), a different way around teleoperation: a hand-held gripper with a camera. Cheng Chi is the week 6 guest.

## Guest talk quiz

### Q1

Why does Xu say teleoperation creates a bottleneck?

- A) Teleoperated data is low quality.
- B) VR controllers are expensive.
- C) Data exists only when a person acts through a specific robot, and the resulting policy works on that robot; more data needs more robots and operator time.
- D) Teleoperation cannot record images.

<details><summary>Answer</summary>

**C.** Teleoperated data is in fact high quality (A is wrong), which is why it dominates today. The problem is scale and coupling: human sensorimotor knowledge only becomes data when it is routed through a robot. ([04:00](https://youtu.be/qvTP6T5oq1w?t=240))

</details>

### Q2

Why do they need SLAM from the glasses to use human hand motion as robot actions?

- A) The head moves, so hand positions measured from the head camera are in a moving frame; SLAM gives the head's pose in a fixed world frame, so the hand trajectories can be expressed there.
- B) To estimate the forces the hand applies.
- C) To transcribe what the person says.
- D) To train the policy faster.

<details><summary>Answer</summary>

**A.** Without it, looking around would show up as hand motion. With the head pose from SLAM, chaining transforms gives the hand in a stable frame, the same index-cancelling rule as week 2. ([09:00](https://youtu.be/qvTP6T5oq1w?t=540))

</details>

### Q3

What did plain co-training on human and robot data fail to achieve, which motivated EgoBridge?

- A) Any improvement from human data.
- B) Running on a real robot.
- C) Working with smart glasses.
- D) Transfer to tasks that appear only in human data: human and robot samples sat in two separate clusters in the policy's representation.

<details><summary>Answer</summary>

**D.** Co-training clearly helped when the task was in both datasets (so A is wrong). Zero-shot transfer needs the two kinds of data to share a representation, and they did not. ([13:01](https://youtu.be/qvTP6T5oq1w?t=781), [15:01](https://youtu.be/qvTP6T5oq1w?t=901))

</details>

### Q4

What is optimal transport used for in EgoBridge?

- A) Compressing the egocentric video.
- B) An extra loss that pulls the human and robot (observation, action) distributions together, using the similarity of action trajectories as the cost.
- C) Translating human videos into robot videos.
- D) Estimating contact forces.

<details><summary>Answer</summary>

**B.** Optimal transport measures how far apart two distributions are by the cheapest way to move one onto the other; its smoothed version is differentiable, so it becomes a training loss. Matching on action shape means "human and robot doing the same motion" end up close. ([16:01](https://youtu.be/qvTP6T5oq1w?t=961))

</details>

### Q5

According to Xu, what does first-person human video still lack for robot learning?

- A) Force and touch signals, and the longer-term context behind decisions (memory, experience).
- B) Colour information.
- C) The positions of the hands.
- D) Enough image resolution.

<details><summary>Answer</summary>

**A.** Hands are tracked well (C) and video quality is fine. What is missing is sensing that cameras cannot see, since contact is hidden behind the fingers, and the context that made the person act: the non-Markovian expert from the lecture. ([24:02](https://youtu.be/qvTP6T5oq1w?t=1442), [34:00](https://youtu.be/qvTP6T5oq1w?t=2040))

</details>
