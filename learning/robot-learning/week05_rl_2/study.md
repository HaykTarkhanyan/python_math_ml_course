# Week 05 study guide: Reinforcement Learning II

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Last week's value functions ($V$, $Q$, TD targets, replay buffers, target networks), week 2's trajectory probability, and from probability: expectations as integrals, $\nabla \log x = \nabla x / x$, and the KL divergence. The derivations are short; doing them yourself once is the best preparation for the second half of Homework 4.

### Extra comments

**Two ways to differentiate an expectation.** This lecture uses both, and seeing them side by side clears up a lot. The score-function (log-derivative) estimator, $\nabla_\theta \mathbb{E}[f] = \mathbb{E}[f \, \nabla_\theta \log p_\theta]$, needs no gradient of $f$: the reward can be a step function and the dynamics unknown. Its price is high variance. The reparameterisation (pathwise) estimator, $a = \mu_\theta + \sigma_\theta \epsilon$ (slide 31), needs a differentiable $f$ (SAC uses the critic $Q$), and in return has much lower variance. REINFORCE and PPO use the first, SAC and DDPG the second. ([08:02](https://youtu.be/AdTGz8YnnlE?t=482), [slides 7](slides.md#slide-7) and [31](slides.md#slide-31))

**Why no dynamics model is needed.** In $\log p_\theta(\tau)$ the initial-state and transition terms do not depend on $\theta$, so their gradient is zero and they drop out (slide 8). You still need to run the real system to get samples, but you never need to write down its equations. Slide 8's indices are off by one, as the transcriber note says; the final line is right. ([12:01](https://youtu.be/AdTGz8YnnlE?t=721), [slide 8](slides.md#slide-8))

**Policy gradient is reward-weighted behaviour cloning.** Slide 10 is the most useful way to remember it. For a Gaussian policy, $\nabla_\mu \log \pi(a \mid s) = (a - \mu)/\sigma^2$: each sampled action pulls the mean toward itself, and the return decides how hard (or, with a baseline, whether it pushes away). Behaviour cloning is the special case where every sample is the expert's and every weight is 1. ([15:01](https://youtu.be/AdTGz8YnnlE?t=901), [slide 10](slides.md#slide-10))

**What reward-to-go and baselines remove.** Rewards before step $t$ have zero expected effect on $\nabla \log \pi(a_t \mid s_t)$, but they still add noise to each sample, so dropping them (causality) lowers variance at no cost. A baseline does the same for the part of the return that is the same for every action. In the 99/101 example, the raw weights are about 100, and only the difference of 2 carries information; after subtracting 100 the weights are $\pm 1$, all signal. ([17:00](https://youtu.be/AdTGz8YnnlE?t=1020), [slides 11](slides.md#slide-11) to [13](slides.md#slide-13))

**Where advantages come from in practice.** You never have the true $A^\pi$. With a learned $V$, the one-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is an advantage estimate: low variance, but biased if $V$ is wrong. The full Monte Carlo return minus $V(s_t)$ is unbiased but noisy. Generalized Advantage Estimation (GAE) mixes the two with a parameter $\lambda$, and is what most PPO implementations use. ([23:00](https://youtu.be/AdTGz8YnnlE?t=1380), [slide 15](slides.md#slide-15))

**How fast trajectory importance weights blow up.** If the new policy makes each action just 10% more likely than the old one did, the trajectory weight after 100 steps is $1.1^{100} \approx 14{,}000$; at 0.9 per step it is $0.9^{100} \approx 3 \times 10^{-5}$. One or two trajectories dominate the batch and the estimate is useless. That is why slide 20 switches to per-step ratios and drops the state-distribution ratio, which is only safe if the policy changes little: exactly what TRPO and PPO enforce. ([31:00](https://youtu.be/AdTGz8YnnlE?t=1860), [slides 19](slides.md#slide-19) and [20](slides.md#slide-20))

**What the clip in PPO actually does.** Take $\epsilon = 0.2$. If an action had positive advantage, the objective stops rewarding you once the ratio passes 1.2: $\min(r A, 1.2 A)$ is flat in $\theta'$ beyond that, so the gradient is zero. If the advantage was negative, it stops rewarding you once the ratio drops below 0.8. The $\min$ makes it a pessimistic bound: clipping can only remove an incentive, never add one. This is what lets PPO take several epochs of minibatch steps on one batch without running away. ([40:01](https://youtu.be/AdTGz8YnnlE?t=2401), [slide 26](slides.md#slide-26))

**PPO's family tree reaches LLMs.** RLHF for chat models was PPO with a learned reward model. GRPO (from the DeepSeekMath paper) drops the critic altogether: sample a group of answers to the same prompt, and use the group's mean reward as the baseline. That is slide 13's "mean reward" baseline, now in production. ([41:02](https://youtu.be/AdTGz8YnnlE?t=2462), [slide 26](slides.md#slide-26))

**The two robot case studies, with context.** The ANYmal locomotion work (Lee et al., 2020, ETH) trained a "teacher" policy in simulation with privileged terrain information, then distilled it into a "student" that only uses the robot's own sensor history: Gupta's teacher-student pattern from week 2. OpenAI's Rubik's cube hand used PPO with automatic domain randomization, a vision network that estimated the cube's pose, and a classical solver to choose the face turns; what was learned was the dexterous manipulation, not the puzzle. ([39:01](https://youtu.be/AdTGz8YnnlE?t=2341), [42:01](https://youtu.be/AdTGz8YnnlE?t=2521), [slides 25](slides.md#slide-25) and [27](slides.md#slide-27))

**Asymmetric actor-critic.** The critic is only needed during training, so in simulation it can see the true state (object poses, velocities) while the actor sees only camera images, since the actor is what gets deployed. Slide 29 mentions this; the paper is Pinto et al. (2017). ([46:02](https://youtu.be/AdTGz8YnnlE?t=2762), [slide 29](slides.md#slide-29))

**SAC in one paragraph.** Maximise reward plus $\alpha$ times the policy's entropy, so the agent prefers to keep several ways of succeeding open. The critic learns from a replay buffer (off-policy, like DQN), the actor is updated through the critic with the reparameterisation trick (like DDPG, but stochastic), and in the standard version $\alpha$ is tuned automatically to hit a target entropy. The SAC follow-up paper learned quadruped walking on a real robot in about two hours of experience, which is why SAC remains a default for real-robot RL. ([47:00](https://youtu.be/AdTGz8YnnlE?t=2820), [slide 30](slides.md#slide-30))

### Going further

- OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/): the clearest short derivations of policy gradients, PPO and SAC, with reference code.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (short), and [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), with automatic temperature tuning and the real-robot experiments.
- Lee et al., [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251), the ETH case study.
- Pinto et al., [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542).

## Lecture quiz

### Q1

Why can REINFORCE compute $\nabla_\theta J$ without a model of the dynamics?

- A) Robot dynamics are linear.
- B) REINFORCE learns a dynamics model on the side.
- C) In $\nabla_\theta \log p_\theta(\tau)$ the initial-state and transition terms do not depend on $\theta$, so only the $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ terms remain.
- D) It estimates the gradient by finite differences.

<details><summary>Answer</summary>

**C.** The log turns the product into a sum, and the terms without $\theta$ have zero gradient. You still need samples from the real system; you just never differentiate through it. ([slide 8](slides.md#slide-8), [13:01](https://youtu.be/AdTGz8YnnlE?t=781))

</details>

### Q2

How does the policy-gradient estimate relate to the behaviour-cloning gradient?

- A) It is the BC gradient on the policy's own samples, weighted by return: actions from good trajectories become more likely, actions from bad ones less.
- B) The two are identical.
- C) It is the BC gradient with the opposite sign.
- D) It needs expert demonstrations like BC.

<details><summary>Answer</summary>

**A.** Both are sums of $\nabla \log \pi(a_t \mid s_t)$; policy gradients multiply each by how well that trajectory went. No expert is needed (D): the policy's own rollouts provide the actions. ([slide 10](slides.md#slide-10), [15:01](https://youtu.be/AdTGz8YnnlE?t=901))

</details>

### Q3

With the causality fix, what weights the gradient of the action taken at step $t$?

- A) The total return of the trajectory.
- B) The rewards received before step $t$.
- C) Only the reward at step $t$.
- D) The rewards from step $t$ onward (the reward-to-go).

<details><summary>Answer</summary>

**D.** An action cannot change rewards that already happened, so those terms only add noise. Using only $r_t$ (C) would ignore that the action also affects later rewards. ([slide 11](slides.md#slide-11), [17:00](https://youtu.be/AdTGz8YnnlE?t=1020))

</details>

### Q4

Two reaches get returns 99 (overshoots) and 101 (perfect). With the mean as baseline, $b = 100$, what weights do their gradients get?

- A) 99 and 101
- B) $-1$ and $+1$: the worse reach becomes less likely, the better one more likely.
- C) 0 and 0
- D) 0.99 and 1.01

<details><summary>Answer</summary>

**B.** Without a baseline both reaches are pushed up strongly and only 2% of the signal tells them apart. Centring keeps exactly the informative part. ([slide 13](slides.md#slide-13), [20:02](https://youtu.be/AdTGz8YnnlE?t=1202))

</details>

### Q5

Why does subtracting a constant baseline $b$ leave the policy gradient unbiased?

- A) Because $b$ is small compared to the returns.
- B) Because $b$ is learned.
- C) Because $\mathbb{E}[\nabla_\theta \log p_\theta(\tau)] = \nabla_\theta \int p_\theta(\tau) \, d\tau = \nabla_\theta 1 = 0$, so the extra term $b \, \mathbb{E}[\nabla_\theta \log p_\theta(\tau)]$ is zero.
- D) It is biased, but the lower variance is worth it.

<details><summary>Answer</summary>

**C.** Probabilities always sum to one, so the expected score is zero, and any constant times zero is zero. The same argument works for a baseline that depends on the state but not on the action, such as $V(s)$. ([slide 13](slides.md#slide-13), [21:01](https://youtu.be/AdTGz8YnnlE?t=1261))

</details>

### Q6

The new policy makes each action 10% more likely than the old policy did. Over a 100-step trajectory, what is the trajectory importance weight $\prod_t \pi_{\theta'}/\pi_\theta$?

- A) 1.1
- B) About 10
- C) About 110
- D) About 14,000

<details><summary>Answer</summary>

**D.** $1.1^{100} \approx 13{,}780$. Per-step ratios multiply, so trajectory weights explode or vanish with the horizon, which is why practical algorithms use per-step ratios and keep the policy close to the old one. ([slide 19](slides.md#slide-19), [31:00](https://youtu.be/AdTGz8YnnlE?t=1860))

</details>

### Q7

In PPO with $\epsilon = 0.2$, an action has positive advantage and its probability ratio is already 1.5. What does the clipped objective do about increasing the ratio further?

- A) Nothing: $\min(1.5A, 1.2A) = 1.2A$ does not depend on $\theta'$ there, so the gradient is zero.
- B) It rewards it even more strongly.
- C) It pushes the ratio back down.
- D) The same as the unclipped objective.

<details><summary>Answer</summary>

**A.** Clipping removes the incentive to move further, it does not actively push back (C). That is enough to stop one batch from dragging the policy far away from the policy that collected it. ([slide 26](slides.md#slide-26), [40:01](https://youtu.be/AdTGz8YnnlE?t=2401))

</details>

### Q8

What makes SAC more sample-efficient than PPO?

- A) It has no critic.
- B) Its critic learns off-policy from a replay buffer, so each transition is reused many times.
- C) It uses larger networks.
- D) It has no entropy term.

<details><summary>Answer</summary>

**B.** PPO is near-on-policy: data from old policies is soon discarded. SAC keeps everything in the buffer, like DQN. It has both a critic and an entropy term (A and D are backwards). ([slide 30](slides.md#slide-30), [47:00](https://youtu.be/AdTGz8YnnlE?t=2820))

</details>

### Q9

What does the reparameterisation trick do in SAC?

- A) It makes the policy deterministic.
- B) It removes the noise from the actions.
- C) It writes $a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ drawn independently of $\theta$, so the gradient of $Q(s, a)$ can flow through $\mu$ and $\sigma$ into the policy.
- D) It replaces $Q$ with $V$.

<details><summary>Answer</summary>

**C.** The randomness is moved into a fixed input, so the action becomes a differentiable function of the parameters. The policy stays stochastic (A and B are wrong): the noise is still there, it just no longer blocks the gradient. ([slide 31](slides.md#slide-31), [49:02](https://youtu.be/AdTGz8YnnlE?t=2942))

</details>

## Guest talk

### Context

Andrew Wagenmaker is a postdoc at UC Berkeley with Sergey Levine; he did his PhD at the University of Washington with Kevin Jamieson and spent time at Microsoft working with Dylan Foster. His work is on exploration, and the talk argues that the reason RL is rare on real robots is not the update rule but how robots explore, then shows how a behaviour-cloning policy can be turned into a good explorer.

### Extra comments

**"Collect the experience that distinguishes the correct behaviour."** His definition of the exploration problem. A policy that never picks up the spray bottle can run a million episodes and never learn that the bottle matters. Random actions in a 7-D action space almost never produce a coherent attempt at a multi-step task, which is why conventional exploration does not transfer to the real world. ([05:01](https://youtu.be/CPmTpXA5azw?t=301), [08:01](https://youtu.be/CPmTpXA5azw?t=481))

**A diffusion policy is a decoder from noise to plausible actions.** Every noise sample comes out as something that looks like a demonstration: reach for the bottle, the iron or the shirt. DSRL (diffusion steering with RL) keeps the trained BC policy frozen and learns a small "noise policy" that chooses the input noise, with ordinary actor-critic RL in the noise space. Exploration in noise space is automatically structured, because the decoder only produces sensible actions. This is week 3's latent variable that "picks the mode" (slide 27), now chosen by RL instead of by sampling. ([14:00](https://youtu.be/CPmTpXA5azw?t=840), [16:00](https://youtu.be/CPmTpXA5azw?t=960))

**Why it sidesteps a real technical problem.** PPO and REINFORCE need $\log \pi(a \mid s)$, and a diffusion policy has no tractable likelihood. DSRL never needs it, because the policy being trained is the small noise policy. Compare Kumar's PA-RL from week 4: both avoid differentiating through the diffusion model, one by improving actions and distilling them back, the other by steering the frozen model's input. ([19:00](https://youtu.be/CPmTpXA5azw?t=1140))

**The numbers he reports.** Policies that succeeded less than half the time reached 90 to 100% after 30 to 90 minutes of online training, and on the mushroom task the policy had locked onto the right behaviour after about 25 episodes. A ByteDance team applied it to a high-precision bimanual task. These are his claims from the talk; the paper has the details. ([18:00](https://youtu.be/CPmTpXA5azw?t=1080), [20:03](https://youtu.be/CPmTpXA5azw?t=1203))

**BC is a poor starting point for RL when demonstrations are thin.** If no demonstration shows the spray bottle, BC cannot produce it, so RL has nothing to reinforce. Adding random noise to the BC policy creates a trade-off: enough noise to explore also ruins the steps the demonstrations got right. His fix is to estimate how uncertain we are about what the demonstrator would do in each state, and make the pre-trained policy explore in proportion: confident where demonstrations agree, exploratory where they are missing. In his example both starting policies look equally bad, but only the uncertainty-aware one improves past 50% with RL. ([22:01](https://youtu.be/CPmTpXA5azw?t=1321), [25:00](https://youtu.be/CPmTpXA5azw?t=1500))

**Two good Q&A points.** Steering through noise only works if the BC policy actually responds to its input noise, and how much it does depends on how it was trained. And on Physical Intelligence's RL work: he describes their approach as training one policy conditioned on advantages, versus his approach of steering a fixed policy, and says honestly that nobody yet knows which works better when. ([28:01](https://youtu.be/CPmTpXA5azw?t=1681), [30:00](https://youtu.be/CPmTpXA5azw?t=1800))

### Going further

- Wagenmaker et al., [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (DSRL), the first part of the talk.
- Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137), to see what is being steered; it is the main topic of week 6.

## Guest talk quiz

### Q1

Why, according to Wagenmaker, does conventional RL exploration not scale to real robots?

- A) Robots move too slowly.
- B) Random actions almost never produce the behaviour that reveals the correct solution, such as picking up the spray bottle, however many times you try.
- C) Rewards cannot be defined for real tasks.
- D) Robots lack the compute for RL.

<details><summary>Answer</summary>

**B.** Exploration has to collect the experience that distinguishes the right behaviour. Undirected random actions mostly flail in free space, so the useful experience never shows up. ([08:01](https://youtu.be/CPmTpXA5azw?t=481))

</details>

### Q2

What is the key move in DSRL?

- A) Retrain the diffusion policy from scratch with RL.
- B) Add Gaussian noise to the policy's output actions.
- C) Backpropagate through all the denoising steps.
- D) Keep the BC diffusion policy frozen and learn a noise policy that picks its input noise, doing RL in the noise space.

<details><summary>Answer</summary>

**D.** The pre-trained policy stays as it is; RL only learns which noise to feed it in each state. That avoids both the likelihood problem and backpropagating through the denoising chain. ([16:00](https://youtu.be/CPmTpXA5azw?t=960))

</details>

### Q3

Why does exploring in noise space look "human-like"?

- A) The frozen BC policy maps any noise to a plausible, demonstration-like action, so each exploratory attempt is a reasonable attempt at the task.
- B) The noise is smaller than action noise.
- C) A human chooses the noise.
- D) It needs no reward.

<details><summary>Answer</summary>

**A.** The decoder does the heavy lifting: it only produces actions of the kind people demonstrated. RL then just has to choose among sensible options, like a person deciding between the iron, the bottle and the shirt. ([17:01](https://youtu.be/CPmTpXA5azw?t=1021))

</details>

### Q4

What trade-off appears when you explore by adding random noise to a BC policy's actions?

- A) None.
- B) Slower inference.
- C) More noise explores more but damages the steps the demonstrations already got right; less noise keeps performance but stops exploring where it is needed.
- D) It makes the policy deterministic.

<details><summary>Answer</summary>

**C.** Uniform noise cannot tell "I know what to do here" from "the demonstrations never covered this". His uncertainty-aware pre-training targets exactly that distinction. ([24:01](https://youtu.be/CPmTpXA5azw?t=1441))

</details>

### Q5

How does he propose to pre-train a policy that will later be improved with RL?

- A) Estimate the uncertainty about what the demonstrator would do, and explore in proportion to it: confident where demonstrations agree, exploratory where they are missing.
- B) Collect many more demonstrations.
- C) Use a larger model.
- D) Skip pre-training and run RL from scratch.

<details><summary>Answer</summary>

**A.** He frames pre-training in Bayesian terms. In his example the two starting policies perform about the same, but only the uncertainty-aware one improves well with online RL (to about 75% versus below 50%). ([25:00](https://youtu.be/CPmTpXA5azw?t=1500), [26:00](https://youtu.be/CPmTpXA5azw?t=1560))

</details>
