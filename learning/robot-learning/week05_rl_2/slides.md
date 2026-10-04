# Lecture 5: Reinforcement Learning II - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 16.03.2026
- Source: `slides.pdf` in this folder (48 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week05_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://www.youtube.com/watch?v=AdTGz8YnnlE (transcript: `transcript_lecture.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 5: Reinforcement Learning II

Oier Mees - ETH zurich, Microsoft - 16.03.2026

## Slide 1: Recap: RL Algorithm Taxonomy & Trade-offs (frame 2)

| | Exact Methods | Tabular RL | Deep RL (Discrete) | Deep RL (Continuous) |
|---|---|---|---|---|
| Algorithms | Value Iter., Policy Iter., Q-Val. Iter. | Q-Learning, SARSA | DQN, Double DQN | DDPG, QT-Opt (CEM) |
| Needs Dynamics Model | ✓ Train+Test | ✗ Model-free | ✗ Model-free | ✗ Model-free |
| State Space | Discrete, small | Discrete, small | High-dim images | High-dim images |
| Action Space | Discrete | Discrete | Discrete | Continuous |
| Policy (train time) | Tabular | Tabular | Neural Net | Actor + Critic NNs |
| Off-policy | N/A | Q-Learn ✓, SARSA ✗ | ✓ (Replay Buffer) | ✓ (Replay Buffer) |
| Convergence | ✓ Guaranteed | ✓ Tabular+explore | ✗ No guarantee | ✗ Brittle |
| Key Innovation | Bellman contraction | TD updates from experience | Replay buffer + Target network | Actor learns $\arg\max_a Q$ |

## Slide 2: [untitled: value-based vs. policy gradients] (frames 3-4)

**Value-Based Methods:** learn what every possible actions is worth & hope argmax will extract the best policy

**Policy Gradients:** what if we optimize our policy directly to make good actions more likely?

## Slide 3: Recap: The Learning Objective (frame 5)

- Accumulated reward an agent receives: $G = \sum_{t \ge 0} \gamma^t r_t$
- Expected Return: $J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t \ge 0} \gamma^t r_t\right]$
- Optimal Policy: $\pi^* = \arg\max_\pi J(\pi)$

[Figure: several trajectories fanning out from a start state $s_0$ to end states $s_{T_1}, \dots, s_{T_4}$; one highlighted as a solid path.]

## Slide 4: Recap: Trajectory Probability (frame 6)

- Probability of a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ under policy $\pi$:
- $p_\pi(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$
  - [braces under the three factors] **Initial State Distribution** ($\rho(s_0)$), **Policy** ($\pi_\theta$), **Transition Function** ($P$)

[Figure: same trajectory fan from $s_0$ to $s_{T_1} \dots s_{T_4}$.]

## Slide 5: How Good is the Policy? (frame 7)

- Let $\pi_\theta(a \mid s)$ be the policy we optimize
- $\theta^* = \arg\max_\theta \underbrace{\mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} R(s_t, a_t)\right]}_{J(\theta)}$
- $J(\theta) = \mathbb{E}_{\tau \sim p_\pi(\tau)}[R(\tau)] \approx \frac{1}{N}\sum_i R(\tau_i) = \frac{1}{N}\sum_i \sum_{t=0}^{T-1} r_{i,t}$
  - [arrow at $\sum_i$] **Monte Carlo Sampling**: Sum over samples from $\pi_\theta$

[Figure: trajectories from $s_0$ ending in states labelled good (green check), fair (yellow "="), bad, bad (red crosses).]

## Slide 6: How to get the Gradient? (frame 8)

- Let $\pi_\theta(a \mid s)$ be the policy we optimize
- $\theta^* = \arg\max_\theta \underbrace{\mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} R(s_t, a_t)\right]}_{J(\theta)}$
- $R(\tau) = \sum_{t=0}^{T-1} R(s_t, a_t)$
- $J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$

$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau)\, d\tau = \int \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau$$

[Figure: small version of the good / fair / bad trajectory fan.]

## Slide 7: How to get the Gradient? [log-derivative trick] (frames 9-10)

Chain Rule:

$$\nabla_\theta f(\mathbf{x}(\theta)) = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \nabla_\theta x_i(\theta)$$

$$\nabla_\theta \log(p_\theta(\tau)) = \frac{1}{p_\theta(\tau)} \nabla_\theta p_\theta(\tau)$$

$$p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)$$

- [brace label] **Log Derivative Trick** $\nabla \log x = \frac{\nabla x}{x}$

**Policy Gradient Theorem:**

$$\begin{aligned}
\nabla_\theta J(\theta) &= \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \\
&= \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\right]
\end{aligned}$$

- [arrow at $\nabla_\theta p_\theta(\tau)$ in the first line] "Can't sample this"
- [arrow at the expectation] "We can sample this!"
- [arrow at $\nabla_\theta \log p_\theta(\tau)$] "Sampling still non-differentiable, but we can differentiate wrt to $\theta$!"

## Slide 8: How to get the Gradient? [expanding the trajectory log-probability] (frame 11)

Trajectory Distribution: $p_\theta(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_t) + \log p(s_{t+1} \mid s_t, a_t)$$

**Policy Gradient Theorem:** $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\right]$

- [brace under $\nabla_\theta \log p_\theta(\tau)$] $\nabla_\theta\left[\log p(s_0) + \sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_t) + \log p(s_{t+1} \mid s_t, a_t)\right]$, with $\log p(s_0)$ and $\log p(s_{t+1} \mid s_t, a_t)$ crossed out in red (they do not depend on $\theta$)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right)\left(\sum_{t=0}^{T-1} R(s_t, a_t)\right)\right]$$

[Note: as written on the slide, the log-sum runs $t = 1 \dots T$ and the transition term sits outside the sum; the consistent form is $\sum_{t=0}^{T-1}\left[\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t)\right]$, which matches the final line.]

## Slide 9: REINFORCE (frame 12)

1. Initialize $\pi_\theta$
2. While not converged do
   1. Collect samples $\{\tau^i\}$ by rolling out policy $\pi_\theta$
   2. Compute gradient $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=0}^{N-1}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\right)\left(\sum_{t=0}^{T-1} R(s_{i,t}, a_{i,t})\right)\right]$
   3. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
3. Return $\pi_\theta$

Reference: *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Williams, R. J. (1992)

## Slide 10: Intuition Behind Gradient (frames 13-14)

- Policy Gradient:

$$\nabla_\theta J_{PG}(\theta) = \frac{1}{N} \sum_{i=0}^{N-1}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\right)\left(\sum_{t=0}^{T-1} R(s_{i,t}, a_{i,t})\right)\right]$$

  - [brace under the log-prob sum] "Imitation gradient, weighted by reward" -> ✅ Upweight good trajectories, ❌ Downweight bad ones

- BC Gradient:

$$\nabla_\theta J_{BC}(\theta) = \frac{1}{N} \sum_{i=0}^{N-1}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\right)\right]$$

## Slide 11: Causality (frames 15-16)

arm moves correctly toward cup, but overshoots. What will REINFORCE do?

[Figure: a robot arm moving along a yellow arrow toward a cup, overshooting it; a coin with a minus sign marks the negative reward.]

✅ first half of the trajectory was good! but REINFORCE **penalizes all the actions** in the trajectory

**Causality**: Policy behavior at $t'$ does not affect reward at time $t$ when $t < t'$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'})\right)\right]$$

- [arrow at the inner sum] "sum of future rewards"; brace: **Reward-To-Go**

## Slide 12: High Variance (frames 17-18)

- $\tau^1$: arm moves vaguely toward cup but overshoots - $R = 99$
- $\tau^2$: arm reaches cup perfectly - $R = 101$

[Figure: two robot-arm reaches; the first overshoots the cup (minus-sign coin), the second stops at the cup (green arrow).]

REINFORCE: Both have great return, lets make both behaviors more likely!

$$\Delta\theta \approx \nabla_\theta \log \pi_\theta(a \mid s) \cdot R(\tau)$$

$$\Delta\theta \approx \nabla_\theta \log \pi_\theta(a \mid s) \cdot 99 \quad \leftarrow \text{Huge gradient for your optimizer!}$$

Effective signal is $101 - 99 = 2$ -> Signal-to-Noise: $2/100$ = **2% of the gradient is informative**

## Slide 13: Reducing Variances with Baselines (frame 19)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) - b\right)\right]$$

- [arrow at $-b$] substract a constant baseline, often mean reward: $b = \frac{1}{N}\sum_{i=0}^{N-1} R(\tau_i)$

[Figure: the same two reaches, now scored $R = 99 - 100 = -1$ and $R = 101 - 100 = 1$.]

by substracting average reward, we get negative gradients for below average behavior ✅

Why can we do this?

$$\mathbb{E}[\nabla_\theta \log p_\theta(\tau)\, b] = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)\, b\, d\tau = \int \nabla_\theta p_\theta(\tau)\, b\, d\tau = b \nabla_\theta \int p_\theta(\tau)\, d\tau = b \nabla_\theta 1 = 0$$

✅ Subtracting a baseline does not change the gradient in expectation, it is unbiased.

## Slide 14: Advantage Function [the ideal baseline] (frame 20)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) - b\right)\right]$$

- [brace under the inner sum] **Reward-To-Go** $\;Q^\pi(s_t, a_t) = \mathbb{E}_{\pi_\theta}\left[\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) \mid s_t, a_t\right]$

What could be the ideal baseline?

$$V^\pi(s) = \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\left[Q^\pi(s, a)\right]$$

Accounts for the specific difficulty of each state, ensuring the robot is only rewarded for actions that outperform its own average expectation in that situation

## Slide 15: Advantage Function (frame 21)

How much better it is to take a action $a$ in state $s$?

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^\pi(s_t, a_t)\right]$$

- $Q^\pi(s_t, a_t) = \mathbb{E}_{\pi_\theta}\left[\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) \mid s_t, a_t\right]$ - expected return by taking action $a$ from state $s$
- $V^\pi(s) = \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\left[Q^\pi(s, a)\right]$ - expected return from state $s$

## Slide 16: Distribution Shift & Sample Efficiency (frame 22)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p(\tau)}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) - b\right)\right]$$

- [arrow at $\tau \sim p(\tau)$] assumes samples from current policy

1. Collect samples $\{\tau^i\}$ by rolling out policy $\pi_\theta$
2. Compute gradient $\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=0}^{N-1}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\right)\left(\sum_{t=0}^{T-1} R(s_{i,t}, a_{i,t})\right)\right]$
3. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ <- We need to **throw away old data & collect new data every gradient step!** [scream emoji]

**Vanilla policy gradient is On-Policy, unlike Off-Policy algorithms like DQN that can reuse past data**

## Slide 17: Policy Gradients Off-Policy Version [importance sampling] (frame 23)

- Can we use data from previous policy $\bar{p}(\tau)$?

[Figure: "Importance Sampling" plot over Trajectory Space ($\tau$), x from -4 to 4, y = Probability Density. Solid blue bimodal curve: Target $p(x)$ - new policy $p_\theta(\tau)$ (peaks near $\pm 1.5$, density 0.4). Dashed orange unimodal curve: Proposal $q(x)$ - old policy $\bar{p}(\tau)$ (peak at 0, density ~0.33). Orange dots: samples from $q(x)$. Green dots, sized by weight $w = p(x)/q(x)$: large near the target's peaks, tiny near 0.]

$$\begin{aligned}
\mathbb{E}_{x \sim p(x)}[f(x)] &= \int p(x) f(x)\, dx \\
&= \int \frac{q(x)}{q(x)} p(x) f(x)\, dx \\
&= \int q(x) \frac{p(x)}{q(x)} f(x)\, dx \\
&= \mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right]
\end{aligned}$$

$q$ needs to have **non-zero support** where $p(x) > 0$ to remain unbiased

## Slide 18: Policy Gradients Off-Policy Version [trajectory importance weight] (frames 24-26)

RL objective with importance sampling

$$J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)}\left[\frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau)\right]$$

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)}{p(s_1) \prod_{t=1}^{T} \bar{\pi}(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)} = \prod_{t=1}^{T} \frac{\pi_\theta(a_t \mid s_t)}{\bar{\pi}(a_t \mid s_t)}$$

- $p(s_1)$ and the transition terms $p(s_{t+1} \mid s_t, a_t)$ are crossed out: they cancel

How do derive the gradient to update our new policy $\pi_{\theta'}$ with samples from old $\pi_\theta$?

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\left(\sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t \mid s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

- [arrow at $p_{\theta'}(\tau)$] new policy

## Slide 19: Policy Gradients Off-Policy Version [importance weights] (frames 27-29)

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}\left(\sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t \mid s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

- [arrows] $p_\theta(\tau)$ = old policy; $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}$ = importance weights

$$= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\left(\prod_{t=1}^{T} \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}\right)\left(\sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t \mid s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

- [arrow at the product] substitute from previous slide. this product can lead to **exploding/vanishing gradients** for large T!

## Slide 20: Policy Gradients Off-Policy Version [per-timestep form] (frame 30)

In practice, we often consider expectations over timesteps instead of trajectories

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(s_{i,t}, a_{i,t})}{\pi_\theta(s_{i,t}, a_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

- [arrow at the ratio] more stable to compute, but hard to measure

$$\frac{\pi_{\theta'}(s, a)}{\pi_\theta(s, a)} = \frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)} \cdot \frac{\pi_{\theta'}(s)}{\pi_\theta(s)} \approx \frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)}$$

- "Often approximate with 1" (the state-marginal ratio $\pi_{\theta'}(s)/\pi_\theta(s)$)

**Final form:**

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

## Slide 21: Off-Policy Policy Gradients Algorithm (frame 31)

Take **multiple gradient steps** on the same batch

1. Collect samples $\{\tau^i\}$ by rolling out policy $\pi_\theta$
2. Compute gradient using $\{\tau^i\}$ samples $\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$
3. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

[Figure: a red outer loop arrow from step 3 back to step 1 (new rollouts), and a blue inner loop arrow from step 3 back to step 2 (several gradient steps on the same batch).]

## Slide 22: Entropy Regularization (frames 32-34)

- What if the support set is zero, $p(x) > 0$ but $q(x) = 0$?
  - New policy $\pi_{\theta'}$ assigns probability to action-state pairs that $\pi_\theta$ never took.
    - ❌ Importance weight explodes, estimator gets biased!
- **Solution**: Add **entropy bonus** to avoid policy becoming too deterministic

$$L(\theta') = L_{PG}(\theta') + \beta\, \mathbb{H}(\pi_{\theta'}(\cdot \mid s))$$

- [arrow at $\beta$] Temperature parameter

| Discrete Shannon Entropy | Continuous Differential Entropy |
|---|---|
| $\mathbb{H}(\pi_{\theta'}) = -\sum_{a \in A} \pi_{\theta'}(a \mid s) \log \pi_{\theta'}(a \mid s)$ | $\mathbb{H}(\pi_{\theta'}) = \frac{d}{2}(1 + \log(2\pi)) + \sum_{j=1}^{d} \log \sigma_j$ |

[Note: the continuous formula is the entropy of a $d$-dimensional Gaussian policy with diagonal covariance $\sigma_1^2, \dots, \sigma_d^2$.]

## Slide 23: Policy Gradients with Constraints (frames 35-36)

- What if our policy changes a lot before sampling new data?

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t} \mid s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

- [arrow at the ratio] Unlikely action under $\pi_\theta$ gets high reward, $\pi_{\theta'}$ makes it much more likely and gradient explodes! $w = \frac{0.10\ (\text{new})}{0.005\ (\text{old})} = 20$
- **Solution**: constrain the policy to not change too much during gradient updates

$$\mathbb{E}_{s \sim \pi_\theta}\left[D_{KL}(\pi_\theta(\cdot \mid s)\, \|\, \pi_{\theta'}(\cdot \mid s))\right] \le \epsilon$$

## Slide 24: Trust Region Policy Optimization (TRPO) (frames 37-38)

- Maximize advantage from old policy while not deviating too much

$$\text{maximize}_{\theta'} \quad L(\theta') = \mathbb{E}_{s, a \sim \pi_\theta}\left[\frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s, a)\right]$$

$$\text{subject to} \quad \mathbb{E}_{s \sim \pi_\theta}\left[D_{KL}(\pi_\theta(\cdot \mid s)\, \|\, \pi_{\theta'}(\cdot \mid s))\right] \le \epsilon$$

- ✅ Monotonic improvement guarantee
- ❌ Requires the Fisher Information Matrix to solve the constraint optimization, is $O(N^3)$ number of trainable parameters
- ⚠️ Conjugate gradient approximation reduces this to $O(kn) \approx O(n)$, but still ~20x more expensive than a standard gradient step
- ❌ Scaling to deep networks requires additional approximations like K-FAC (Kronecker-Factored Approximate Curvature)

Reference: *Trust region policy optimization*, Schulman, Levine, et al., (2015)

## Slide 25: Case Study: Quadruped Locomotion with TRPO (frame 39)

[Figure: embedded video, "DARPA Subterranean Challenge Urban Circuit", camera label BETA_ENTRANCE_CAM52: a legged quadruped robot walking through an underground concrete facility, with a person in a safety vest and a wheeled robot nearby.]

Reference: *Learning Quadrupedal Locomotion over Challenging Terrain*, Lee, Hwangbo, et al., (2020)

## Slide 26: Proximal Policy Optimization (PPO) (frames 40-41)

- What if just clip importance ration instead of expensive constraints to avoid getting too large?

$$L(\theta') = \mathbb{E}_{s, a \sim \pi_\theta}\left[\min\left(\frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s, a),\ \text{clip}\left(\frac{\pi_{\theta'}(a \mid s)}{\pi_\theta(a \mid s)}, 1 - \epsilon, 1 + \epsilon\right) A^{\pi_\theta}(s, a)\right)\right]$$

- [brace] Force the importance sampling ratio to stay within a safe interval

Reference: *Proximal Policy Optimization Algorithms*, Schulman, et al., (2017)

[Pros and cons, second build (frame 41):]

- ✅ Scales linearly $O(N)$ with the number of trainable parameters
- ✅ Significantly easier to implement, works with standard optimizers like Adam
- ✅ Foundation to train LLMs with RLHF or GRPO
- ❌ No monotonic improvement guarantee
- ❌ Near-on-policy, needs many samples

## Slide 27: Case Study: OpenAI Dexterous Hand (frame 42)

- PPO sim2real achieves human-level dexterity on 24 DoF

[Figure: a robot hand wearing a black rubber glove (labelled "Rubber Glove") holding a Rubik's cube.]

Reference: *Solving Rubik's Cube With a Robot Hand*, OpenAI et al., (2019)

## Slide 28: Full PPO Objective (frame 43)

$$L^{PPO}(\theta') = \mathbb{E}_{s, a \sim \pi_\theta}\left[L^{CLIP}(\theta') - c_1\left(V_{\theta'}(s) - V^{targ}\right)^2 + c_2 H(\pi_{\theta'})(s)\right]$$

- [braces] $L^{CLIP}$ = **Clipped Surrogate Objective** ("Actor"); $(V_{\theta'}(s) - V^{targ})^2$ = **For Advantage** ("Critic"); $H$ = **Entropy Bonus**
- ✅ Combines the best of value-based methods and policy gradients!

## Slide 29: Actor Critic Methods (frame 44)

- **Actor**: uses Policy Gradients to predict continuous actions
- **Critic**: uses Value learning to evaluate goodness of a state
- **Actor-Critic**: ✅ Low variance from critic, continuous actions from actor, direct policy optimization
  - Learns from every step via TD updates, at the cost of small bootstrapping bias
  - Optimal baseline to lower variance via advantage
  - Robotics: often asymmetric information via privileged info for critic
  - ❌ Actor and Critic still near-on-policy, needs many samples

## Slide 30: Soft Actor-Critic (frame 45)

- Critic trained fully off-policy on a replay buffer!

$$L^{Actor}(\theta') = \mathbb{E}_{s \sim \mathbb{D}_{replay},\, a \sim \pi_{\theta'}}\left[\alpha H(\pi_{\theta'})(s) - Q_\phi(s, a)\right]$$

- [arrow at $\alpha H$] **Exploration**: Entropy Bonus
- [arrow at $Q_\phi(s, a)$] **Exploitation**: if I were to do this action today, how many points would I get?

$$L^{Critic}(\phi) = \mathbb{E}_{\tau \sim \mathbb{D}_{replay}}\left[\left(Q_\phi(s, a) - \left(r + \gamma\left(Q_{\bar{\phi}}(s', a') + \alpha H(\pi_{\theta'})(s')\right)\right)\right)^2\right]$$

- [arrow at $Q_\phi(s, a)$] How many points did I think this past action $a$ in this past situation $s$ was worth?
- [arrow at $r$] actual reward received
- [arrow at $Q_{\bar{\phi}}$] target network
- [arrow at $\alpha H(\pi_{\theta'})(s')$] this state leads to high reward AND gives you many different ways to succeed

[Note: the standard SAC actor loss, minimized, is $\mathbb{E}\left[\alpha \log \pi_{\theta'}(a \mid s) - Q_\phi(s, a)\right]$, which in expectation is $-\alpha H - Q$. The slide's $+\alpha H - Q$ has the entropy sign flipped relative to that if read as a loss to minimize.]

## Slide 31: Reparameterization Trick (frame 46)

- Problem: Update Actor based on the Critic's score.

$$a \sim \mathcal{N}(\mu_{\theta'}(s), \sigma_{\theta'}(s)) \quad \rightarrow \quad \text{Stochastic \& Non-differentiable}$$

- [arrows] $\mu_{\theta'}(s)$ = Mean Action Dist.; $\sigma_{\theta'}(s)$ = Std Dev Action Dist.

$$a = f_{\theta'}(s, \epsilon) = \mu_{\theta'}(s) + \sigma_{\theta'}(s) \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$

- [arrow at $\epsilon$] sampled from a fixed distribution without weights, so no gradients needed!

$$L^{Actor}(\theta') = \mathbb{E}_{s \sim \mathbb{D}_{replay},\, \epsilon \sim \mathcal{N}(0,1)}\left[\alpha H(\pi_{\theta'})(s) - Q_\phi(s, f_{\theta'}(s, \epsilon))\right]$$

[Note: slide 32 does not appear in the captured frames; the numbering jumps from 31 to 33.]

## Slide 33: Conclusion (frame 47)

- REINFORCE, optimize policy directly
  - Causality + Baseline: reduce variance
  - Advantage Function: V(s) as optimal baseline
- Importance Sampling: multiple gradient steps on batch
- Entropy Regularization: maintain support
- TRPO: bound how much policy changes via KL
- PPO: Clip ratio instead, still near-on-policy
- SAC: fully off-policy critic via replay buffer, most sample efficient

## End (frame 48)

Thank you for your attention
