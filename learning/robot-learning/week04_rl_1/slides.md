# Lecture 4: Reinforcement Learning I - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 09.03.2026
- Source: `slides.pdf` in this folder (77 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week04_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/90raNpc11tQ (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_aviral_kumar.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 4: Reinforcement Learning I

Oier Mees - ETH zurich, Microsoft - 09.03.2026

## Slide 1: Recap: Imitation Learning (frame 2)

- **Data**: Given trajectories collected by an "expert" "demonstrations" $\mathcal{D} = \{(s_1, a_1, \dots, s_t)\}$ [Figure: a Dataset cylinder holding $(s_1, a_1, \dots, s_t)$]
- **Goal**: Learn a policy $\pi_\theta$ that "imitates" the expert's behavior -> ❌ **we can't get better than the expert!**

[Figure: road image -> **Policy** -> the same road image with a forked arrow.]

## Slide 2: Recap: Markov Decision Process (frame 3)

- Define by $M = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$

[Figure: loop between the environment (globe) and the robot (arm): "state $s_t$, reward $r_t$" to the robot, "action $a_t$" back to the environment.]

- $\mathcal{S}$: State space, $s_t \in \mathcal{S}$
- $\mathcal{A}$: Action space, $a_t \in \mathcal{A}$
- $\mathcal{P}$: Transition probability, $s_{t+1} \sim \mathcal{P}(\cdot|, s_t, a_t)$
- $\mathcal{R}$: Reward function, $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$, $r_t = R(s_t, a_t, s_{t+1})$

## Slide 3: Recap: The Learning Objective (frame 4)

- Accumulated reward an agent receives: $G = \sum_{t=0}^{T-1} r_t$ or $G = \sum_{t \ge 0} \gamma^t r_t$
- Expected Return: $J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t=0}^{T-1} r_t\right]$ or $J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t \ge 0} \gamma^t r_t\right]$
- Optimal Policy: $\pi^* = \arg\max_\pi J(\pi)$

[Figure: trajectories fanning out from $s_0$ to end states $s_{T_1}, \dots, s_{T_4}$.]

## Slide 4: Aren't Sparse Rewards Non-Smooth? (frames 5-6)

- $r(s,a) = \begin{cases} +1 & \text{if } s \in S_{\text{success}} \\ -1 & \text{otherwise} \end{cases}$

[Figure: a narrow road cut into a cliff above a misty drop; the road is marked +1 (green) and the drop -1 (red).]

[Second build (frame 6) adds a plot, "The Gradient Problem (Rewards: -1 or +1)": Reward Value against State / Action Space from -4 to 4. The reward is a step: -1 for negative inputs, +1 for positive ones. Both flat parts are labelled "Gradient $\nabla r = 0$"; the jump at 0 is labelled "Discontinuity: Gradient Undefined".]

## Slide 5: Aren't Sparse Rewards Non-Smooth? [the expectation is smooth] (frame 7)

- $r(s,a) = \begin{cases} +1 & \text{if } s \in S_{\text{success}} \\ -1 & \text{otherwise} \end{cases}$

[Figure: the same cliff-road photo.]

$$\pi_\theta(a = \text{fall} \mid s) = p_\theta(s)$$

$$a \mid s \sim \text{Bernoulli}(p_\theta(s))$$

$$J(\theta) = \underbrace{\mathbb{E}_{(a \sim \pi_{(\theta)})}[r(a)]}_{\text{smooth!}} = p_{(\theta)}(s) \cdot (-1) + \big(1 - p_{(\theta)}(s)\big) \cdot (+1)$$

## Slide 6: Aren't Sparse Rewards Non-Smooth? [landscape] (frame 8)

[Figure: the cliff-road photo and a plot, "Differentiable Landscape in Parameter Space": Value against Policy Parameter $\theta$ from -4 to 4. Red solid line, $r(s,a)$ (Non-Smooth): a step from -1 to +1 at $\theta = 0$. Blue dashed line, $\mathbb{E}_{a \sim \pi_\theta}[r(s,a)]$ (Smooth!): an S-curve rising smoothly from -1 to +1 through 0.]

## Slide 7: Don't Wait for the Cliff (frames 9-10)

$$J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t \ge 0} \gamma^t r_t\right] = \mathbb{E}_{s_0, a_0, s_1 \dots}\left[r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots\right]$$

[Figure: the cliff-road photo (+1 on the road, -1 off the cliff).]

[Second build (frame 10) adds a brace under the sum:] Can we estimate $J(\pi)$ **without unrolling** each episode to the end?

## Slide 8: Value Function (frame 11)

- Expected discounted return starting from state $s$ and following policy $\pi$ thereafter

$$V^\pi(s) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t \sim \pi(\cdot \mid s_t), s_{t+1} \sim P(\cdot \mid s_t, a_t)\right]$$

- RL objective is $V$ averaged over starting states

$$J(\pi) = \mathbb{E}_{s_0 \sim p(s_0)}\left[V^\pi(s_0)\right]$$

## Slide 9: Bootstrapping: Recursive Shortcut (frame 12)

Decompose infinite sums of rewards:

$$\sum_{t \ge 0} \gamma^t R(s_t, a_t) = R(s_0, a_0) + \gamma \underbrace{\sum_{t \ge 1} \gamma^{t-1} R(s_t, a_t)}_{\text{this is } V\text{!}}$$

- Bellman Expectation Equation

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s),\, s' \sim P(\cdot \mid s, a)}\left[r(s, a) + \gamma V^\pi(s')\right]$$

- [brace] value now = immediate reward + discounted value of next state; **allows value estimation without full rollouts**

## Slide 10: Optimal Policy (frame 13)

[Figure: small environment / robot loop diagram (state $s_t$, reward $r_t$ / action $a_t$).]

- Optimal Value:

$$V^*(s) = \max_\pi V^\pi(s)$$

- Optimal Policy:

$$\pi^* = \arg\max_\pi J(\pi) = \arg\max_\pi \mathbb{E}_{s_0 \sim p(s_0)}\left[V^\pi(s_0)\right]$$

$$\pi^*(s) = \arg\max_a \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[r(s, a) + \gamma V^*(s')\right]$$

## Slide 11: Value Iteration (frames 14-15)

- How can we compute the optimal policy in practice?

1. Initialize $V_0(s_t) = 0 \quad \forall s_t \in \mathcal{S}$
2. For $k = 0, 1, 2 \dots$ until $\max_{s_t} |V_{k+1}(s_t) - V_k(s_t)| < \epsilon$:
   Update $V_{k+1}(s_t) \leftarrow \max_a \left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)} V_k(s_{t+1})\right]$
3. Extract policy $\pi^*(s_t) = \arg\max_a \left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)} V^*(s_{t+1})\right]$

[Second build (frame 15) adds:]

- ✅ Convergence proof via Bellman contraction
- ❌ Assumes dynamics model
- ❌ Needs to sweep through all states $O(|\mathcal{S}|^2 |\mathcal{A}|)$
- ❌ State space must be discrete & small

## Slide 12: Value Iteration - Gridworld (frames 16-26)

[Figure: an interactive web demo, "Value Iteration - Gridworld - watching $V^*(s)$ emerge", run live across eleven frames (the lecturer plays it, resets it and plays it again). Update rule shown at the top: $V_{k+1}(s) \leftarrow \max_a \mathbb{E}_{s' \sim P(\cdot \mid s, a)}[r(s, a) + \gamma \cdot V_k(s')]$. A 5x5 grid with a Goal (+1) cell top right, a Danger (-1) cell below it and four wall cells. Status panel: $\gamma$ (discount) 0.90, $\epsilon$ (threshold) 0.001. Actions up / down / left / right, "Stochastic: 80% intended, 10% each perpendicular". Every cell starts at 0.000 (iteration 0); one mid-run frame shows iteration 19 with max $|\Delta V|$ = 0.00348.]

[Converged state shown (iteration $k$ = 22, max $|\Delta V|$ = 0.00059, "Converged! max $|\Delta V| < \epsilon$"); each cell gives $V^*(s)$ and the greedy action:]

| | col 1 | col 2 | col 3 | col 4 | col 5 |
|---|---|---|---|---|---|
| row 1 | 1.186 → | 1.384 → | 1.591 → | 1.851 → | Goal +1 |
| row 2 | 1.029 ↑ | wall | 1.384 ↑ | wall | Danger -1 |
| row 3 | 0.892 ↑ | wall | 1.186 ↑ | 1.029 ← | 0.590 ← |
| row 4 | 0.780 ↑ | 0.865 → | 1.013 ↑ | wall | 0.536 ↓ |
| row 5 | 0.680 ↑ | 0.751 ↑ | 0.853 ↑ | 0.737 ← | 0.625 ← |

## Slide 13: Policy Evaluation (frame 27)

- How good is our policy?

1. Initialize $V_0(s_t) = 0 \quad \forall s_t \in \mathcal{S}$
2. For $k = 0, 1, 2 \dots$ until $\max_{s_t} |V_{k+1}(s_t) - V_k(s_t)| < \epsilon$:
   Update $V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi(\cdot \mid s_t)}\left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)}[V_k(s_{t+1})]\right]$

- ✅ Convergence proof via Bellman contraction
- ⚠️ Cheaper than value iteration $O(|\mathcal{S}|^2)$, still sweep through all states
- ❌ Assumes dynamics model
- ❌ State space must be discrete & small

## Slide 14: Value Iteration vs Policy Evaluation (frame 28)

- Value Iteration:

$$V_{k+1}(s_t) \leftarrow \max_a \left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)} V_k(s_{t+1})\right]$$

- Policy Evaluation:

$$V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi(\cdot \mid s_t)}\left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)}[V_k(s_{t+1})]\right]$$

[The only difference, $\max_a$ versus $\mathbb{E}_{a \sim \pi(\cdot \mid s_t)}$, is highlighted in red.]

## Slide 15: Policy Iteration (frame 29)

- Alternates between finding out how good we are and trying to be even better

1. Initialize $\pi_0(s_t)$ arbitrarily $\forall s_t \in \mathcal{S}$
2. For $j = 0, 1, 2, \dots$ until $\pi_{j+1} = \pi_j$
   - **Policy Evaluation** $V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi_j(\cdot \mid s_t)}\left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)}[V_k(s_{t+1})]\right]$
   - **Policy Improvement** $\pi_{j+1}(s_t) = \arg\max_a \left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)} V^{\pi_j}(s_{t+1})\right]$

- [arrow at Policy Evaluation] this just one step of eval iteration until convergence
- [brace under Policy Improvement] find best action according to one-step look-ahead

## Slide 16: Policy Iteration - Gridworld (frames 30-32)

[Figure: an interactive web demo, "Policy Iteration - Gridworld - evaluate $\pi_j$, improve greedily, repeat", run live across three frames, on the same 5x5 grid as slide 12 ($\gamma$ 0.90, $\epsilon$ 0.001). Header formulas: Eval: $V_{k+1}(s) \leftarrow \mathbb{E}_{a \sim \pi_j}[r(s,a) + \gamma \cdot \mathbb{E}_{s' \sim P}[V_k(s')]]$; Improve: $\pi_{j+1}(s) = \arg\max_a [r(s,a) + \gamma \cdot \mathbb{E}_{s' \sim P}[V^{\pi_j}(s')]]$. Status shows outer step $j$, inner evaluation step $k$, phase (Policy Evaluation / Policy Improvement) and the number of policy changes. A side plot, "$J(\pi_j) = V^{\pi_j}(s_0)$ - Monotonically ↑ by improvement theorem", rises as a staircase from about -0.1 at $j=0$ to about 0.65 by $j=2$.]

- [Frame 30: $j$ = 0, $k$ = 29, Evaluation of the initial arbitrary policy: most values around -0.095, a few positive cells near the goal.]
- [Frame 31: $j$ = 1, $k$ = 31, 15 policy changes; the top rows already match the optimal values.]
- [Frame 32: $j$ = 4, 0 policy changes, "Converged! $\pi_{j+1} = \pi_j$". The values and arrows are identical to the value-iteration result on slide 12 (1.186, 1.384, 1.591, 1.851 along the top row, ...).]

## Slide 17: Limitations of $V(s)$ for Robotics (frame 33)

- Acting with V(s), requires **querying Dynamic Models at test time,** for every state-action combination

$$\pi^*(s_t) = \arg\max_a \left[r(s_t, a) + \underbrace{\gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)} V^*(s_{t+1})}_{\text{need dynamics model at test time!}}\right]$$

[Figure: illustration of a robot arm in state $S_t$ fanning out many candidate motions into a thought cloud, "Simulate all actions ($P$) (requires dynamics model) to find highest value ($V$)", then "Execute best action ($a^*_t$)". A timeline below: Sense -> Planning (long) -> Act, from Start to End.]

## Slide 18: Q-Values (frame 34)

- $V^*(s_t)$: Expected discounted return starting from state $s_t$, acting optimally afterwards

$$V^*(s_t) = \max_a \left[r(s_t, a) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)}\left[V^*(s_{t+1})\right]\right]$$

- $Q^*(s_t, a)$: Expected discounted return from state $s_t$, committing to action $a$ first, acting optimally afterwards

$$Q^*(s_t, a) = r(s_t, a) + \underbrace{\mathbb{E}_{s_{t+1} \sim P(\cdot \mid s_t, a)}}_{\text{still needs dynamics model during training}}\left[\gamma \max_{a_{t+1}} Q^*(s_{t+1}, a_{t+1})\right]$$

[thinking-face emoji next to "**training**"]

$$V^*(s_t) = \max_a Q^*(s_t, a)$$

## Slide 19: Q-Value Iteration Test Time (frame 35)

- No dynamics model needed during **test time**!

$$\pi^*(s_t) = \arg\max_a Q^*(s_t, a)$$

[Figure: the same robot-arm illustration, now with a single arrow to a box "Pre-computed Q-function ($Q(s,a)$ direct lookup) - Find highest value ($V$)" and "Execute best action ($a^*_t$)"; the timeline shrinks to Sense -> Act with almost no planning.]

## Slide 20: Q-Learning - Removing the Dynamics Model (frame 36)

- Dynamics model might be unknown, too complex or slow
- The Solution? Replace Expectation with Experience

1. Execute action and observe transition $(s_t, a_t, r_t, s_{t+1})$
2. TD target $= r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a)$, $r(s_t, a_t)$ if $s_{t+1}$ terminal
3. $Q(s_t, a_t) \leftarrow (1 - \alpha) \underbrace{Q(s_t, a_t)}_{\text{old estimate}} + \alpha \underbrace{[\text{TD target}]}_{\text{new experience sample}}$

- [arrow at $\alpha$] learning rate

## Slide 21: Q-Learning - Gridworld (frames 37-39)

[Figure: an interactive web demo, "Q-Learning - Gridworld - learn $Q(s,a)$ from experience, no dynamics model", run live across three frames. Update rule: $Q(s_t, a_t) \leftarrow (1-\alpha) \cdot Q(s_t, a_t) + \alpha \cdot [r(s_t, a_t) + \gamma \cdot \max_a Q(s_{t+1}, a)]$. Same 5x5 grid; each cell is split into four triangles holding the Q-value of each action, with an arrow for the best one and a white dot for the agent. Status: episode, step, total steps, TD error $|\delta|$, $\alpha$ (lr) 0.50, $\gamma$ 0.90, $\epsilon$-greedy. Hyperparameter sliders: $\alpha$ = 0.50, $\epsilon$-decay = 0.97. Buttons: Play / Pause, Step, Fast x50, Reset. A side plot, "$J(\pi) = Q(s_0, \text{best } a)$ - Improves as Q converges", shows max $Q(s_0, a)$ per episode.]

- [Frames 37-38: episodes 166 and 188 (about 3,400-3,700 total steps), $\epsilon$-greedy 0.050; the Q-values have spread from the goal (about 0.8-0.97 next to it) and the $J(\pi)$ curve has risen and plateaued around 0.3.]
- [Frame 39: after a reset, episode 4, $\epsilon$-greedy 0.885: almost all Q-values are still near -0.02, and only the cell next to the goal has learned a large value (0.50).]

## Slide 22: Off-Policy vs On-Policy (frames 40-41)

- Does it matter **which policy** collects the transition $(s_t, a_t, r_t, s_{t+1})$ used to update our TD target?

$$\text{TD target} = \underbrace{r(s_t, a_t)}_{\text{what happened}} + \gamma \underbrace{\max_a Q(s_{t+1}, a)}_{\text{max assumes future optimal action, regardless of past behavior}}$$

[Second build (frame 41) adds:]

- Off-Policy learns from **"any"** data, even sub-optimal behavior -> reuse experience (Experience Replay Buffer)

## Slide 23: Off-Policy vs On-Policy [Q-Learning vs SARSA] (frame 42)

- Q-Learning Off Policy target:

$$\text{TD target} = r(s_t, a_t) + \gamma \underbrace{\max_a Q(s_{t+1}, a)}_{\text{What is the best I could do next?}}$$

- SARSA On-Policy target:

$$\text{TD target} = r(s_t, a_t) + \gamma \underbrace{Q(s_{t+1}, a_{t+1})}_{\text{What am I actually going to do next?}}$$

## Slide 24: Q-Learning vs SARSA - Gridworld (frames 43-49)

- SARSA learns "safer" behaviors

[Figure: an interactive "cliff walking" web demo run live across seven frames (training, convergence, reset). Two side-by-side 4x12 grids: **Q-Learning (Off-Policy)**, "TD target = $r + \gamma \cdot \max_a Q(s', a)$ - best I *could* do", and **SARSA (On-Policy)**, "TD target = $r + \gamma \cdot Q(s', a_{t+1})$ - what I *will* do". Start S bottom left, goal bottom right, and the whole bottom row between them is a cliff of skull cells. Above the grids, a plot "Episode return - Q-learning vs SARSA during training" with the caption: "During training: SARSA is more cautious (avoids cliff-edge states). After convergence: Q-Learning scores higher (shorter path), SARSA scores lower (safer detour)." Over about 150 episodes both returns climb from around -120 toward about -15, with deep dips whenever an agent falls off the cliff.]

- [Converged Q-Learning: "13 steps (risky)", a path along the row directly above the cliff]
- [Converged SARSA: "17 steps (safe)", a detour along the top row, far from the cliff]

## Slide 25: RL Taxonomy (frame 50)

| Exact Solution Methods | Value-based Methods |
|---|---|
| Value Iteration, Policy Iteration, Q-Value Iteration | Q-Learning (off-policy), SARSA (on-policy) |
| ✅ Convergence proof via Bellman contraction | ✅ Converges to $Q^*$ (tabular, sufficient exploration) |
| ❌ Sweep through all states | ✅ No dynamics model needed (model-free) |
| ❌ Needs dynamics model | ✅ Learns from interaction |
| ❌ State space must be discrete & small | ❌ State space must be discrete & small |
| ❌ Tabular representation | ❌ Tabular representation |

## Slide 26: The Curse of Dimensionality: Scaling to Image Observations (frame 51)

**The Curse of Dimensionality: Scaling to Image Observations**

Atari game observation space $= 256^{84 \times 84 \times 4}$

## Slide 27: Case Study: Deep Q-Learning (DQN) (frame 52)

- First time a single algorithm learned to play 49 different games from raw pixels
- Launched Deep Reinforcement Learning in 2013

[Figure: screenshot of the title, author list and abstract of *Playing Atari with Deep Reinforcement Learning* (Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra, Riedmiller; DeepMind Technologies). The abstract describes a convolutional network trained with a variant of Q-learning on raw pixels, applied to seven Atari 2600 games.]

- [arrow at Martin Riedmiller] Personal note: Martin just had started his sabbatical at a small startup called DeepMind, when I joined Uni Freiburg

[Note: the 49-game result is from the later Nature paper (Mnih et al., 2015); the 2013 paper in the screenshot covers seven games.]

## Slide 28: DQN Algorithm (frame 53)

**Algorithm 1: Deep Q-learning with Experience Replay** (pseudocode from the paper, transcribed as a list)

- Initialize replay memory $\mathcal{D}$ to capacity $N$
- Initialize action-value function $Q$ with random weights
- for episode $= 1, M$:
  - Initialise sequence $s_1 = \{x_1\}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
  - for $t = 1, T$:
    - With probability $\epsilon$ select a random action $a_t$, otherwise select $a_t = \max_a Q^*(\phi(s_t), a; \theta)$
    - Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
    - Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
    - Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $\mathcal{D}$
    - Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $\mathcal{D}$
    - Set $y_j = r_j$ for terminal $\phi_{j+1}$; $y_j = r_j + \gamma \max_{a'} Q(\phi_{j+1}, a'; \theta)$ for non-terminal $\phi_{j+1}$
    - Perform a gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$

[Note: the greedy action should be $\arg\max_a$, not $\max_a$; this slip is in the original paper's pseudocode.]

## Slide 29: Experience Replay Buffer (frame 54)

- Store transitions in buffer $D_{\text{replay}} = \{(s_t, a_t, r_t, s_{t+1})\}_{t=1}^{N}$ instead of discarding them
- Sample random minibatch each update -> breaks temporal correlations (creates i.i.d. samples)
- Each transition **reused many times** -> more sample efficient

[Figure: $D_{\text{replay}}$ cylinder holding $(s_1, a_1, r_1, \dots, s_t)$ -> "Sample Minibatch" -> **Update $Q_\theta$ (gradient step)** -> the environment (globe); a two-way arrow "Interact with Environment" links the environment back to the buffer.]

## Slide 30: Delayed Target Network (frame 55)

$$\mathcal{L}(\theta) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathbb{D}_{\text{replay}}}\left[\Big(r(s_t, a_t) + \gamma \underbrace{\max_a Q(s_{t+1}, a; \theta)}_{\text{target moves every step}} - \underbrace{Q(s_t, a_t; \theta)}_{\text{prediction moves every step}}\Big)^2\right]$$

- [arrow] **optimizer unstable & diverges**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathbb{D}_{\text{replay}}}\left[\Big(r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a; \boldsymbol{\theta}^-) - Q(s_t, a_t; \theta)\Big)^2\right]$$

- [arrow at $\theta^-$] copy of $\theta$ frozen for some steps, then $\theta^- \leftarrow \theta$
- [arrow at $\theta$] updated every step
- Freezing $\theta^-$ makes the target fixed -> stable supervised regression

## Slide 31: DQN (frame 56)

- Discrete action space, image state space, sparse & dense rewards

[Figure: screenshots of five Atari games.]

| | B. Rider | Breakout | Enduro | Pong | Q\*bert | Seaquest | S. Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | -20.4 | 157 | 110 | 179 |
| Sarsa [3] | 996 | 5.2 | 129 | -19 | 614 | 665 | 271 |
| Contingency [4] | 1743 | 6 | 159 | -17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | -3 | 18900 | 28010 | 3690 |

## Slide 32: The Overestimation Bias (frame 57)

- Max is not symmetric, it amplifies positive noise leading to systematic overestimation

$$y_j = r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a; \theta^-)$$

- [arrow at $\theta^-$] In DQN, same network selects and evaluates action

- Double DQN: same two networks as DQN, just rerouted

$$y_j = r(s_t, a_t) + \gamma\, Q\Big(s_{t+1}, \underbrace{\arg\max_a Q(s_{t+1}, a; \theta)}_{\text{online network selects action}}; \underbrace{\theta^-}_{\text{target network evaluates it}}\Big)$$

Reference: *Deep Reinforcement Learning with Double Q-learning*, Van Hasselt et al., 2015

## Slide 33: DQN - Scaling to Continuous Action Spaces (frame 58)

- Robots have often continuous action spaces $\mathbf{a} = [\tau_1, \tau_2, \tau_3, \tau_4, \tau_5, \tau_6, \tau_7]^\top \in \mathbb{R}^7$ [Figure: a 7-joint robot arm]
- How do you compute $\arg\max_a Q(s_t, a, \theta)$ over an infinite action space?
  - Discretization
  - Sampling + CEM
  - Learn arg max with second network (DDPG)

## Slide 34: Case Study: Spatial Discretization (frame 59)

[Figure: system diagram. Left, a photo of the setup with labels RGB-D camera, robot arm, gripper, objects (a pile of wooden blocks) and workspace. The camera view becomes a top-down heightmap $s_t$, rotated into 16 orientations (x16). Each rotation goes through two fully convolutional networks, $\text{FCN}_{\text{pushing}}\ \phi_p$ and $\text{FCN}_{\text{grasping}}\ \phi_g$ (both DenseNet-121), which output pixel-wise Q-maps $Q_{\text{push}}(s_t, a_t)$ and $Q_{\text{grasp}}(s_t, a_t)$. Taking the max over every pixel and rotation gives the "best push" and the "best grasp", drawn on the heightmap. Every pixel is a discrete action.]

Reference: *Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning*, Zeng, Song et al., 2018

## Slide 35: Case Study: Spatial Discretization [video] (frame 60)

[Figure: video titled "Learning Pushing and Grasping": a robot arm above a table with a cluster of colourful wooden blocks.]

Reference: *Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning*, Zeng, Song et al., 2018

## Slide 36: Monte Carlo Sampling (frames 61-62)

- How to compute $\arg\max_a Q(s_t, a, \theta)$ over an infinite action space?

$$\arg\max_{a \in \mathbb{R}^7} Q(s, a; \theta) \approx \arg\max_{a^{(i)} \sim \text{Uniform}(\mathcal{A}),\ i = 1 \dots N} Q(s, a^{(i)}; \theta)$$

[Figure: interactive demo, "Continuous action spaces - why sampling approximates arg max and how CEM refines it", in Monte Carlo mode. A bumpy multimodal curve $Q(s, a, \theta)$ over action $a \in [0, 1]$, true argmax at 0.515 (dashed line). With only N = 1 sample the estimate lands on a low bump: estimated argmax 0.251 (error 26.4%) in one frame and 0.065 (error 44.9%) in the other, with the warning "Too few: N=1 is not enough. Increase samples."]

[Second build (frame 62) adds:]

- ✅ Works because $Q(s_t, a, \theta)$ is smooth in $a$, dense enough sampling always finds a point near the true peak
- ❌ Requires $N$ forward passes per action selection, scales poorly
- ⚠️ Quality depends entirely on $N$ and luck, no guarantee of finding the true argmax

## Slide 37: Cross-Entropy Method (CEM) (frames 63-64)

- Sample **smarter** by iteratively concentrating on high $Q$

1. Sample $N$ actions from $a^{(i)} \sim \mathcal{N}(\mu_k, \sigma_k)$, $i = 1 \dots N$
2. Keep top $M$ with highest $Q$, $\mathbb{E}_{\text{set}} = \{a^{(j)} \mid Q(s, a^{(j)}) \in \text{top } M\}$
3. Refit $\mu_{k+1} = \text{mean}(\mathbb{E}_{\text{set}})$, $\sigma_{k+1} = \text{std}(\mathbb{E}_{\text{set}})$ to elite set & repeat

Return $a^* = \mu_K$

[Figure: the same demo in CEM mode (samples N = 20, elite size M = 5). In one frame the sampling Gaussian has collapsed onto the highest peak and the estimate equals the true argmax, 0.515 ("Converged"); in the other, step 1, a wide Gaussian covers the middle of the range, its samples are dotted on the curve, the elite ones in green, and the current estimate is 0.613.]

- ✅ No gradients needed - works as a black-box optimizer on any $Q_\theta$
- ❌ Finds only one mode, if $Q_\theta$ is multimodal, converges to whichever peak it lands first
- ⚠️ Optimization at test time, pays $N \times K$ forward passes every single control step

## Slide 38: Case Study: QT-Opt (frames 65-67)

- First time Q-Learning + CEM scales to real robots with image obs -> 96% grasp success on unseen objects
  - Off-policy Q-learning: replay buffer stores 580k real grasps

[Figure: embedded video playing across three frames: a robot arm grasping objects from a bin, with a "Camera view" panel (overlay "Input: 472x472 Image and gripper aperture - Output: Gripper aperture and displacement"). The middle frame shows the system diagram. **Training time**: reward = grasp success, determined by subtracting pre and post-drop images; state, action, reward go to **Distributed RL**, which sends back learned weights. **Inference time**: the camera image (State: 472x472 image and gripper aperture) goes to the **Critic Function** $Q(\text{State}, \text{Action})$; the **Cross-Entropy Method** $\arg\max_{\text{Action}} Q(\text{State}, \text{Action})$ sends action proposals to the critic and gets Q-values back; the chosen Action (gripper displacement and aperture) goes to the robot.]

Reference: *QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation*, Kalashnikov et al., 2018

## Slide 39: Deep Deterministic Policy Gradient (DDPG) (frame 68)

- CEM solves $\arg\max$ at test time, DDPG learns it during **training**

Instead of:

$$a^* = \arg\max_{a^{(i)} \sim \mathcal{N}(\mu_k, \sigma_k)} Q(s_t, a^{(i)}; \theta) \qquad [N \times K \text{ passes}]$$

Learn a network $\mu_\phi$ such that:

$$\mu_\phi(s_t) \approx \arg\max_a Q(s_t, a; \theta) \qquad [1 \text{ forward pass}]$$

- Two Networks:
  - **Critic** $Q(s_t, a, \theta)$ -> same as DQN, trained with TD loss
  - **Actor** $\mu_\phi(s_t)$ -> new, trained to maximize $Q$

## Slide 40: DDPG [updates and exploration] (frame 69)

- **Actor update**: ask critic which action is better, move $\mu_\phi$ in that direction

$$\nabla_\phi J(\phi) \approx \mathbb{E}_s\left[\nabla_a Q(s, a; \theta)\Big|_{a = \mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s)\right]$$

- **Exploration**: A deterministic actor never explores, but unlike DQN we can't use $\epsilon$-greedy, we add gaussian noise

$$a_t = \mu_\phi(s_t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

Reference: *Continuous control with deep reinforcement learning*, Lillicrap et al., 2015

## Slide 41: DDPG [real-robot door opening] (frames 70-75)

[Figure: embedded video playing across six frames. Robot arms learning to open a door handle: "Two Workers - 2.5 hours - Ep1 1.5x" (then Ep2), and "Single Worker - 2.5 hours 1.5x". In between, a learning-curve plot: test reward (-0.3 to 0.4) against hours (0.5 to 4.0) for "1 worker" (blue) and "2 workers" (green). The 2-worker curve leaves the 1-worker curve at about 1.1 hours and reaches about 0.3 by 2.5 hours; the 1-worker curve stays near 0 until about 3 hours and ends near 0.1. Red circles mark the 1-worker curve at 2.5 hours and the point where the 2-worker curve takes off.]

[Added from frame 71 on:]

- ✅ Test time = 1 forward pass, runs at any control frequency
- ✅ Handles high-dimensional continuous actions, no CEM needed at test time
- ❌ Notoriously brittle, very sensitive to hyperparameters, often fails to converge
- ❌ Noise $\sigma$ must be tuned manually, too small: no exploration, too large: unstable

Reference: *Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates*, Gu et al., 2016

## Slide 42: RL Algorithm Taxonomy & Trade-offs (frame 76)

| | Exact Methods | Tabular RL | Deep RL (Discrete) | Deep RL (Continuous) |
|---|---|---|---|---|
| Algorithms | Value Iter., Policy Iter., Q-Val. Iter. | Q-Learning, SARSA | DQN, Double DQN | DDPG, QT-Opt (CEM) |
| Needs Dynamics Model | ✓ Train+Test | ✗ Model-free | ✗ Model-free | ✗ Model-free |
| State Space | Discrete, small | Discrete, small | High-dim images | High-dim images |
| Action Space | Discrete | Discrete | Discrete | Continuous |
| Policy (train time) | Tabular | Tabular | Neural Net | Actor + Critic NNs |
| Off-policy | N/A | Q-Learn ✓, SARSA ✗ | ✓ (Replay Buffer) | ✓ (Replay Buffer) |
| Convergence | ✓ Guaranteed | ✓ Tabular+explore | ✗ No guarantee | ✗ Brittle |
| Key Innovation | Bellman contraction | TD updates from experience | Replay buffer + Target network | Actor learns arg max_a Q |

[The same table opens Lecture 5 as its recap slide.]

## End (frame 77)

Thank you for your attention
