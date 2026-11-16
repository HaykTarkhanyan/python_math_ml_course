# Week 04 study guide: Reinforcement Learning I

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Week 2's MDP notation ($\mathcal{S}, \mathcal{A}, P, r, \gamma$, return, $J(\pi)$) is used without re-deriving it. You also need expectations of a function of a random variable, and what a fixed point of an update rule is. Homework 4 builds on this lecture and the next.

### Extra comments

**Why the expectation is smooth.** The reward is a step, but the policy chooses "fall" with a probability $p_\theta(s)$ that changes smoothly with $\theta$, so the expected reward $-p + (1 - p) = 1 - 2p$ is smooth too. Same reason a logistic-regression probability is smooth though the label is 0 or 1. This is what makes gradient-based RL possible with sparse rewards, as long as the policy is stochastic and sometimes sees the +1 at all. ([06:02](https://youtu.be/90raNpc11tQ?t=362), [slides 5](slides.md#slide-5) and [6](slides.md#slide-6))

**Bootstrapping means learning a guess from a guess.** $V(s) = \mathbb{E}[r + \gamma V(s')]$ lets you update a value from one step of experience plus your current estimate of the next state's value, without waiting for the episode to end. You no longer need to fall off the cliff to learn the cliff is bad; it is enough that the state just before the cliff already has a low value. The price: if the estimate of $V(s')$ is wrong, the error flows backwards. ([11:01](https://youtu.be/90raNpc11tQ?t=661), [slide 9](slides.md#slide-9))

**What "contraction" buys you, in numbers.** One Bellman update shrinks the largest error over all states by at least a factor $\gamma$. So after $k$ sweeps the error is at most $\gamma^k$ times the starting error, whatever you started with, and there is exactly one fixed point. With $\gamma = 0.9$, cutting the error a thousandfold takes at most $\ln 1000 / \ln(1/0.9) \approx 66$ sweeps; the demo stopped sooner because its test is the change between sweeps, not the error. ([15:03](https://youtu.be/90raNpc11tQ?t=903), [slide 11](slides.md#slide-11))

**Value iteration vs policy iteration.** Value iteration does many cheap sweeps; policy iteration does few expensive rounds (a full evaluation, then a greedy improvement). In the demo policy iteration needed only 4 rounds. The "policy improvement theorem" guarantees each greedy round is at least as good as the last, which is why the return plot on slide 16 is a staircase that only goes up. ([21:02](https://youtu.be/90raNpc11tQ?t=1262), [slide 16](slides.md#slide-16))

**V needs a model to act; Q does not.** To act from $V$ you must ask "where would each action take me?", which needs $P$. $Q(s, a)$ already folds that question in, so acting is just $\arg\max_a Q(s, a)$. The cost: one number per state-action pair instead of per state. Learning $Q$ from sampled transitions (Q-learning) then removes the model from training too. ([22:00](https://youtu.be/90raNpc11tQ?t=1320), [slides 17](slides.md#slide-17) to [20](slides.md#slide-20))

**When tabular Q-learning provably works.** Every state-action pair must keep being tried (that is what $\epsilon$-greedy is for), and the learning rate must shrink, but not too fast: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. The demo's fixed $\alpha = 0.5$ is fine for a demo, not for the proof. ([26:01](https://youtu.be/90raNpc11tQ?t=1561), [slide 20](slides.md#slide-20))

**The cliff, read carefully.** This is Example 6.6 in Sutton and Barto. SARSA learns the value of the $\epsilon$-greedy policy it actually follows. That policy sometimes takes a random step, and next to the cliff a random step is a disaster, so SARSA learns to keep away. Q-learning learns the value of the greedy policy, so it learns the edge path, but while it still explores it falls off often, so its returns during training are worse. Which one "scores higher" depends on whether exploration is still on, which is what the demo's caption is getting at. ([30:01](https://youtu.be/90raNpc11tQ?t=1801), [slide 24](slides.md#slide-24))

**The "deadly triad".** Function approximation (a network instead of a table), bootstrapping and off-policy data: any two are usually fine, all three together can diverge (Sutton and Barto, chapter 11). DQN uses all three, and its two tricks are stabilisers. The replay buffer makes minibatches look i.i.d.; the target network stops the regression target from moving with every step. ([35:00](https://youtu.be/90raNpc11tQ?t=2100), [slides 29](slides.md#slide-29) and [30](slides.md#slide-30))

**Overestimation, in one experiment.** Give 10 actions a true value of 0, and estimates with independent standard-normal noise. Each estimate is right on average, but the largest of 10 standard normals averages about $+1.54$, so $\max_a \hat Q$ is biased upwards, and bootstrapping copies that bias into earlier states. Double DQN picks the action with one network and scores it with the other, so the noise that made an action look best is not the noise that values it. TD3 (Fujimoto et al., 2018) brings the same fix to DDPG with two critics and the smaller of their estimates. ([38:01](https://youtu.be/90raNpc11tQ?t=2281), [slide 32](slides.md#slide-32))

**Atari history, straightened out.** The 2013 workshop paper on slide 27 played 7 games. "One network, one set of hyperparameters, 49 games" is the 2015 Nature paper, as the transcriber note says. The observation count $256^{84 \times 84 \times 4}$ comes from stacking 4 greyscale $84 \times 84$ frames, so the network can see motion: the week 2 point that one frame is not a Markov state. ([33:01](https://youtu.be/90raNpc11tQ?t=1981), [slides 26](slides.md#slide-26) and [27](slides.md#slide-27))

**Three ways to take $\arg\max_a$ over a continuous action, compared.** Discretise (exact, but the grid grows exponentially with action dimensions, unless the action has structure like the pixel grid in the push-grasp work). Sample and refine with CEM (no gradients and works on any $Q$, but costs $N \times K$ network calls per control step and settles on one peak). Learn an actor that outputs the argmax (DDPG: one call per step, but training is fragile). The actor update is the chain rule: nudge the action in the direction that raises $Q$ ($\nabla_a Q$), then nudge the actor's weights so it outputs that action ($\nabla_\phi \mu_\phi$). ([40:01](https://youtu.be/90raNpc11tQ?t=2401), [slides 33](slides.md#slide-33) to [40](slides.md#slide-40))

**QT-Opt's reward is a neat trick.** Grasp success is detected without a human: the robot photographs the bin, lifts, drops whatever it holds, photographs again, and the difference between the images says whether something was grasped. Self-labelled rewards like this are what let 580k real grasps be collected. ([46:00](https://youtu.be/90raNpc11tQ?t=2760), [slide 38](slides.md#slide-38))

### Going further

- Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), chapters 4 (dynamic programming) and 6 (Q-learning, SARSA, the cliff). Chapter 11 for the deadly triad.
- Mnih et al., [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), short and readable.
- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), and Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3), for overestimation in discrete and continuous actions.
- Kalashnikov et al., [QT-Opt](https://arxiv.org/abs/1806.10293), the real-robot case study.

## Lecture quiz

### Q1

The reward is $+1$ on the road and $-1$ off the cliff, a step function with zero gradient almost everywhere. Why is the RL objective still smooth in the policy parameters?

- A) The environment smooths the reward.
- B) Because $\gamma < 1$.
- C) Because the reward is actually dense.
- D) The objective is an expectation over a stochastic policy, and the action probabilities change smoothly with $\theta$.

<details><summary>Answer</summary>

**D.** With $p_\theta$ the probability of falling, $J = -p_\theta + (1 - p_\theta) = 1 - 2p_\theta$, which is as smooth as $p_\theta$. Discounting (B) has nothing to do with it, and the reward really is sparse (C). ([slide 5](slides.md#slide-5), [06:02](https://youtu.be/90raNpc11tQ?t=362))

</details>

### Q2

What does the Bellman expectation equation $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$ let you do?

- A) Compute $V^\pi$ without any data or model.
- B) Estimate a state's value from the immediate reward and the current estimate of the next state's value, without rolling out to the end of the episode.
- C) Avoid choosing a discount factor.
- D) Read off the optimal policy directly.

<details><summary>Answer</summary>

**B.** That is bootstrapping: a one-step look-ahead plus an estimate replaces the full sum of future rewards. You still need either a model or samples to compute the expectation (A), and this is the equation for a given $\pi$, not the optimal one (D). ([slide 9](slides.md#slide-9), [11:01](https://youtu.be/90raNpc11tQ?t=661))

</details>

### Q3

What is the only difference between the value-iteration update and the policy-evaluation update?

- A) Value iteration takes a max over actions; policy evaluation averages over the policy's actions.
- B) Value iteration uses a discount factor; policy evaluation does not.
- C) Value iteration needs no dynamics model.
- D) Policy evaluation stores a Q-table.

<details><summary>Answer</summary>

**A.** $\max_a$ finds the best possible value; $\mathbb{E}_{a \sim \pi}$ measures the value of one fixed policy. Both use the same discount and both need $P$ (so B and C are wrong). Slide 14 highlights just that one symbol. ([slide 14](slides.md#slide-14), [19:00](https://youtu.be/90raNpc11tQ?t=1140))

</details>

### Q4

Why is a Q-function more convenient than a value function for choosing actions at test time?

- A) A Q-table is smaller than a V-table.
- B) Q-functions need no training.
- C) $\arg\max_a Q(s, a)$ picks the action directly; acting from $V$ requires predicting the next state of every action, which needs a dynamics model.
- D) Value functions cannot be learned from data.

<details><summary>Answer</summary>

**C.** The Q-function already contains "what happens if I do $a$ here". It is larger, not smaller, by a factor of $|\mathcal{A}|$ (A is backwards). ([slides 17](slides.md#slide-17) and [19](slides.md#slide-19), [25:00](https://youtu.be/90raNpc11tQ?t=1500))

</details>

### Q5

What makes Q-learning off-policy?

- A) Its learning rate $\alpha$.
- B) Its target uses $\max_{a'} Q(s', a')$, the value of the best next action, whatever the behaviour policy actually does next, so transitions from any policy are valid training data.
- C) It always uses a replay buffer.
- D) It explores with $\epsilon$-greedy.

<details><summary>Answer</summary>

**B.** Because the target assumes optimal behaviour from $s'$ onward, it does not matter who produced $(s, a, r, s')$: an old policy, a random one or a human. SARSA's target uses the action actually taken next, which ties it to the current policy. The replay buffer (C) is what off-policy-ness makes possible, not its cause. ([slides 22](slides.md#slide-22) and [23](slides.md#slide-23), [29:01](https://youtu.be/90raNpc11tQ?t=1741))

</details>

### Q6

In cliff walking, why does SARSA learn the path far from the cliff while Q-learning learns the edge path?

- A) SARSA uses a larger learning rate.
- B) SARSA ignores the rewards.
- C) SARSA uses a model of the cliff.
- D) SARSA learns the value of the $\epsilon$-greedy policy it actually follows; its occasional random steps are disastrous next to the cliff, so those states get low values.

<details><summary>Answer</summary>

**D.** Q-learning values the greedy policy, which never steps off the cliff on purpose, so the edge looks fine. SARSA's values include its own exploration, so it pays to keep a safety margin. Both are model-free (C). ([slide 24](slides.md#slide-24), [31:01](https://youtu.be/90raNpc11tQ?t=1861))

</details>

### Q7

Ten actions all have true value 0. Your estimates of them are unbiased but noisy: each has independent standard-normal noise. On average, what is the largest of the ten estimates?

- A) About $+1.5$
- B) 0
- C) About $-1.5$
- D) 10

<details><summary>Answer</summary>

**A.** The expected maximum of 10 standard normals is about 1.54. Each estimate is unbiased, but the maximum of them is not: that is the overestimation bias, and bootstrapping passes it on to earlier states. Double DQN fixes it by using one network to choose the action and the other to evaluate it. ([slide 32](slides.md#slide-32), [38:01](https://youtu.be/90raNpc11tQ?t=2281))

</details>

### Q8

What is the delayed target network in DQN for?

- A) Making training faster.
- B) Exploration.
- C) Keeping the regression target fixed for a while, so the network is not chasing a target that moves every time its own weights change.
- D) Saving memory.

<details><summary>Answer</summary>

**C.** Without it, every gradient step changes both the prediction and the target, and training can oscillate or diverge. With $\theta^-$ frozen for a while, each stretch of training is ordinary supervised regression. It costs memory (a second copy of the network), so D is backwards. ([slide 30](slides.md#slide-30), [37:00](https://youtu.be/90raNpc11tQ?t=2220))

</details>

### Q9

How does QT-Opt choose an action at test time?

- A) A separate actor network outputs it.
- B) With the cross-entropy method on $Q(s, a)$: sample actions, keep the best ones, refit a Gaussian to them, repeat.
- C) By discretising the image into pixels.
- D) By running value iteration.

<details><summary>Answer</summary>

**B.** QT-Opt has only a critic; CEM searches its action space at every step, which works but costs many critic evaluations. An actor (A) is DDPG's alternative, which learns the argmax during training instead. Pixel-wise discretisation (C) is the push-grasp work on slide 34. ([slides 37](slides.md#slide-37) and [38](slides.md#slide-38), [46:00](https://youtu.be/90raNpc11tQ?t=2760))

</details>

## Guest talk

### Context

Aviral Kumar is an assistant professor at Carnegie Mellon and a researcher at Google DeepMind, where he works on RL for the Gemini models. He did his PhD at Berkeley with Sergey Levine and is best known for CQL, an offline RL algorithm that comes up in week 5. The talk asks which parts of the LLM training recipe robot learning has not copied yet, and offers one paper for "mid-training" and one for "post-training".

### Extra comments

**The three phases, as he uses them.** Pre-training models a broad distribution (all text on the internet). Mid-training adds behaviours that later help a model improve by itself, like the long chains of thought with backtracking that reasoning models learn. Post-training then improves the model with RL on its own attempts. His critique: robot "pre-training" on expert demonstrations is really closer to instruction tuning, and robots have no mid-training and rarely any RL post-training. ([03:02](https://youtu.be/fHHLmTu9sFk?t=182), [04:01](https://youtu.be/fHHLmTu9sFk?t=241))

**Recovery and correction (RaC), in one picture.** A good task execution is a funnel: many possible starting states narrow down to the few that lead to success. When the policy drifts out, the human first backs the robot up to a state that is still inside the funnel (recovery), then shows the right continuation from there (correction). Training on this data teaches the policy to retry, the robot's version of an LLM backtracking. ([13:01](https://youtu.be/fHHLmTu9sFk?t=781))

**Why recovery is easier than correction: the volume argument.** From a bad state, there are many earlier states you can back up to, but only a narrow band of states that lead forward to success. Imitating "go somewhere roomy" tolerates sloppiness; imitating "hit this narrow target" does not. That is his answer in the Q&A to why RaC beats human-gated DAgger, where the human just finishes the task from the bad state (compare week 3, slide 14). ([30:00](https://youtu.be/fHHLmTu9sFk?t=1800), [34:00](https://youtu.be/fHHLmTu9sFk?t=2040))

**The "$p^k$" model, worked.** Say a task has $k = 5$ subtasks and each succeeds with probability $p = 0.8$. Without retries the task succeeds with probability $0.8^5 \approx 0.33$. If the policy can recover and retry each subtask up to 3 times, each subtask fails only with probability $0.2^3 = 0.008$, and the task succeeds with probability $0.992^5 \approx 0.96$. Retrying turns compounding errors into something close to the per-step reliability. ([14:00](https://youtu.be/fHHLmTu9sFk?t=840), [15:01](https://youtu.be/fHHLmTu9sFk?t=901))

**Test-time scaling for robots.** For reasoning LLMs, accuracy rises with the length of the chain of thought. In his plot, success rate rises with the number of recoveries the policy makes within an episode. His headline comparison: about 5 hours of recovery-and-correction data matching a shirt-hanging result that used about 89 hours of expert data. ([18:00](https://youtu.be/fHHLmTu9sFk?t=1080))

**Policy-agnostic RL connects straight to this lecture.** Fine-tuning a big policy with RL usually needs $\nabla_\theta$ of the policy's action, which is awkward for diffusion policies (backpropagate through every denoising step) and impossible for policies that sample discrete tokens. PA-RL avoids it: sample several actions from the policy, keep those the critic scores highest (the sampling argmax of slides 36 and 37), push them a little uphill on $Q$ with gradient steps on the action itself (DDPG's $\nabla_a Q$), then train the policy to imitate these improved actions with its normal maximum-likelihood loss. He reports fine-tuning OpenVLA on a real robot in about 40 minutes, with no demonstrations. ([24:02](https://youtu.be/fHHLmTu9sFk?t=1442), [35:00](https://youtu.be/fHHLmTu9sFk?t=2100))

### Going further

- Sobol Mark et al., [Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone](https://arxiv.org/abs/2412.06685), the post-training paper in the talk.
- Kumar et al., [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) (CQL), his best-known algorithm, useful for week 5.
- Kim et al., [OpenVLA](https://arxiv.org/abs/2406.09246), the policy he fine-tunes; it returns in week 9.
- The RaC paper is cited on his slides at [08:00](https://youtu.be/fHHLmTu9sFk?t=480); search arXiv for "RaC recovery correction".

## Guest talk quiz

### Q1

What does Kumar mean by "mid-training", and what is its robot version in his work?

- A) A phase between pre-training and post-training that adds behaviours useful for later improvement (for LLMs, chains of thought with backtracking); for robots, training on recovery-and-correction data.
- B) Checkpoints saved halfway through pre-training.
- C) RL fine-tuning on the target task.
- D) Distilling a large model into a small one.

<details><summary>Answer</summary>

**A.** RL fine-tuning (C) is his post-training phase. Mid-training is about putting the right behaviours into the model, such as retrying and backtracking, before post-training. ([04:01](https://youtu.be/fHHLmTu9sFk?t=241), [08:00](https://youtu.be/fHHLmTu9sFk?t=480))

</details>

### Q2

How do RaC's human interventions differ from human-gated DAgger's?

- A) RaC uses no humans.
- B) In human-gated DAgger humans never intervene.
- C) In RaC the human first backs the robot up to an earlier, in-distribution state and then shows the correct continuation; in HG-DAgger the human simply completes the task from where the policy went wrong.
- D) There is no difference; only the policy architecture differs.

<details><summary>Answer</summary>

**C.** Both use interventions; the difference is backing up first. That teaches the policy to retreat and retry, which HG-DAgger's data does not show. ([30:00](https://youtu.be/fHHLmTu9sFk?t=1800))

</details>

### Q3

Why is recovering easier to learn than correcting, according to his answer in the Q&A?

- A) Recovery clips are shorter.
- B) There are many valid earlier states to back up to, but only a narrow set of states that lead forward to success, so imitating recovery tolerates more error.
- C) Recovery is learned with RL and correction with imitation.
- D) Human operators are better at recovering.

<details><summary>Answer</summary>

**B.** He calls it a volume argument: the funnel of good states is wide behind you and narrow ahead of you. ([34:00](https://youtu.be/fHHLmTu9sFk?t=2040))

</details>

### Q4

What did he show as the robot version of LLM test-time scaling?

- A) Larger models doing better.
- B) Longer chains of thought in language.
- C) More cameras on the robot.
- D) Success rate rising with the number of recoveries (retries) the policy makes within an episode.

<details><summary>Answer</summary>

**D.** For reasoning LLMs, accuracy rises with thinking length; here success rises with the number of recoveries, which is effort spent in action space rather than in text. ([18:00](https://youtu.be/fHHLmTu9sFk?t=1080))

</details>

### Q5

What is the core idea of policy-agnostic RL (PA-RL)?

- A) Backpropagate through all diffusion steps.
- B) Apply REINFORCE to every token of the VLA.
- C) Improve sampled actions directly against the critic (keep the best samples, then take gradient steps on the actions), then train the policy to imitate the improved actions with its usual maximum-likelihood loss.
- D) Replace the generalist policy with a small Gaussian policy.

<details><summary>Answer</summary>

**C.** All the RL happens on action vectors, and the policy only ever sees a supervised imitation loss, so any policy class that can be trained by maximum likelihood works, the same hyperparameters included. A and B are exactly the fragile routes he wants to avoid. ([25:00](https://youtu.be/fHHLmTu9sFk?t=1500))

</details>

### Q6

A task has 5 subtasks, each succeeding with probability 0.8. Roughly how likely is the whole task to succeed without retries, and with up to 3 attempts per subtask?

- A) About 0.33 without retries, about 0.96 with them.
- B) About 0.8 in both cases.
- C) About 0.33 in both cases.
- D) About 0.96 without retries, about 0.99 with them.

<details><summary>Answer</summary>

**A.** Without retries: $0.8^5 \approx 0.33$. With three attempts a subtask fails only if all three do, $0.2^3 = 0.008$, so the task succeeds with probability $0.992^5 \approx 0.96$. This is his "$p^k$" argument for why learning to recover matters on long tasks. ([14:00](https://youtu.be/fHHLmTu9sFk?t=840))

</details>
