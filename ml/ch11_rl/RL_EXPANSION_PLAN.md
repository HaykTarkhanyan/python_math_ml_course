# RL chapter expansion - proposed outline

**Status:** BUILT 2026-08-12. All seven decks compile with 0 errors, **203 pages total**.
**Date:** 2026-08-11 (plan), 2026-08-12 (build).
**Supersedes nothing.** `RL_CHAPTER_PLAN.md` stays as the build record of L32 as it shipped.

## As built

| Deck | File | Pages |
|---|---|---|
| 1 | `L32_rl_problem.tex` | 39 |
| 2 | `L32b_tabular_learning.tex` | 31 |
| 3 | `L32c_deep_value.tex` | 21 |
| 4 | `L32d_policy_gradients.tex` | 27 |
| 5 | `L32e_planning_practice.tex` | 30 |
| 6 | `L32f_llm_alignment.tex` | 25 |
| 7 | `L32g_llm_reasoning.tex` | 30 |

**The original `L32_reinforcement_learning.tex/pdf` (44pp) is left in place, untouched**, and the
qmd labels it an archive. It was not deleted: it may have been delivered from, and its GitHub
link is live. Everything in it has been redistributed across the seven decks.

**Figures.** 6 new (`bandit_regret`, `bandit_estimates`, `gpi_sweeps`, `gpi_diagram`,
`mc_vs_td`, `cliff_walking`, `deadly_triad`, `dqn_loop`, `pg_family`, `mcts_phases` -- 10, of
which only the first four are measured; the rest are teaching diagrams, per the instructor's
2026-08-11 direction to stop reproducing results). 9 borrowed from `llm_training/slides/*/fig/`
and copied into `ch11_rl/fig/`. 10 reused from the original L32 build.

**Deviations from the plan, and why:**

- `\usepackage{cancel}` added to the shared `ml/preamble.tex` (purely additive, no existing deck
  changes) so the importance-sampling and DPO derivations can strike out the terms that cancel.
- The bandit regret figure was rebuilt at a **20,000-pull** horizon after its own fail-loud guard
  caught the assumed ordering being wrong: at 1,500 pulls $\epsilon$-greedy (86) still beats
  UCB1 (166). The crossover at pull **11,261** became a frame in its own right about asymptotic
  guarantees.
- The `gpi_sweeps` figure was redesigned after the first version buried its own lesson under
  policy iteration's flat curve. It now shows the surprising result directly: value iteration
  has the optimal policy at sweep 7 and then spends 17 more sweeps polishing values that change
  no decision.
- The acronym check was run per `SLIDE_STYLE.md` and 14 expansions were added. The remaining
  flags are **signposted forward references** ("GRPO in lecture 7") which the style guide's
  cross-deck clause permits -- **do not "fix" these**, and do not re-expand a term in every deck.

## Still open

- No homework or Google Form for the new decks; the tic-tac-toe project still serves the chapter
  and is now referenced from decks 5 (self-play, reward shaping) using its own measured numbers.
- No videos recorded.
- `ml/00_plan.md` still allots RL **2-3 sessions (Oct 21/23)**. Seven decks is seven sessions;
  that schedule needs updating and the projected course finish moves out by ~2 weeks.
- Practical candidates identified but not built: tabular Q-learning on FrozenLake (deck 2),
  REINFORCE from scratch on CartPole (deck 4), PPO CleanRL-style (deck 4), GRPO with TRL on
  Colab (deck 7, needs a GPU).

## Instructor decisions (2026-08-11)

1. **Seven decks.** Decks 5 and 6 below are merged into one: MCTS / AlphaZero / MuZero / self-play,
   then offline RL, imitation learning and reward design. Dreamer and Decision Transformers drop to
   single frames.
2. **Derive the load-bearing six**, and only those: Bellman optimality, the policy gradient theorem,
   importance sampling, the baseline theorem, DPO from the KL-regularised objective, and GRPO from
   PPO. Everything else is stated with intuition. **This reverses instructor decision 2 in
   `RL_CHAPTER_PLAN.md`** ("intuition-first, minimal derivation") for those six results only.

## The ask

Turn `ch11_rl` from one 44-page deck into a proper multi-lecture chapter that "properly explains
Q-learning, PPO, GRPO", covering both general RL and the ML/LLM-focused material a person needs to
feel comfortable in the subject.

## What exists today

`L32_reinforcement_learning.tex` - 44 pages, 10 Python figures, one real training run
(`py_src/q_learning_demo.py`). Good deck. Its coverage per topic:

| Topic | Frames today |
|---|---|
| MDP, discounting, gridworld | 6 |
| V, Q, Bellman, value iteration | 5 |
| Q-learning (incl. by-hand, demo, max bias) | 6 |
| Exploration, on/off-policy | 2 |
| DQN | 2 |
| Policy gradient, REINFORCE, actor-critic | 4 |
| **PPO** | **1** |
| Text-as-MDP, RLHF, reward hacking | 4 |
| **DPO, GRPO, RLVR, DAPO, GSPO** | **0** |

## Reference courses consulted

- **HF Deep RL Course** (Simonini). Units 1-8 + bonus. Excellent teaching order for classic deep RL.
  Gives MC-vs-TD, the PG theorem, REINFORCE variance, and the clipped surrogate objective each their
  own section - all of which L32 compresses to one frame. Skips bandits, DP/policy iteration,
  importance sampling, TRPO, SAC. Its modern material (RLHF, offline RL, Decision Transformers) is
  one page each in Bonus Unit 3. Predates GRPO entirely.
- **HF LLM Course ch10-12.** Where HF actually teaches GRPO (in TRL) and the R1 paper.
- **Sutton & Barto 2e** for the classical spine (bandits -> DP -> MC -> TD -> function approximation).
- `llm_training/slides/` in this repo: `01_grpo`, `02_dpo`, `10_instructgpt`, `11_deepseek_r1`,
  `14_chain_of_thought` (ORM/PRM/best-of-N figures), `08_qwen3`. These are **paper seminars**, not
  teaching decks - reuse their figures, not their structure.

---

## Proposed split - 8 decks

Deck 1 keeps the existing filename slot; the rest use letter suffixes, matching the
`L23b`/`L23c`/`L13b` precedent already in `ml/`.

| # | File | Title | New/reused |
|---|---|---|---|
| 1 | `L32_rl_problem.tex` | The RL problem | ~50% reused from L32 |
| 2 | `L32b_tabular_learning.tex` | Learning from experience | ~40% reused |
| 3 | `L32c_deep_value.tex` | Deep RL I: value-based | ~15% reused |
| 4 | `L32d_policy_gradients.tex` | Deep RL II: policy gradients | ~20% reused |
| 5 | `L32e_planning_search.tex` | Planning, search and self-play | new |
| 6 | `L32f_rl_in_practice.tex` | RL in the real world | ~15% reused |
| 7 | `L32g_llm_alignment.tex` | RL for LLMs I: alignment | ~10% reused |
| 8 | `L32h_llm_reasoning.tex` | RL for LLMs II: reasoning | new |

**Collapse options** if 8 sessions is too many:

- **7 decks:** merge 5 + 6 into one "planning, search, and what actually happens in production".
- **6 decks:** as above, and fold deck 3 into deck 4 (one "deep RL" deck), keeping DQN tight.
- **4 decks:** foundations / model-free-through-PPO / LLM alignment / LLM reasoning. Everything in
  decks 5 and 6 becomes name-drop survey frames.

---

## Deck 1 - The RL problem

*Nothing learns yet. This deck defines the problem and solves it exactly when you are allowed to.*

**Cold open** (reuse): AlphaGo move 37, Seoul 2016. Nobody taught it that move.

### Section 1 - A different kind of problem
- Why every model in chapters 1-10 cannot do this: supervised learning needs (input, correct answer).
- **Reward is not a loss** (reuse): delayed, sparse, non-differentiable, and the data depends on the
  policy so i.i.d. is gone.
- Episodic vs continuing tasks.

### Section 2 - The simplest RL problem: bandits *(new)*
- One state, k actions. Strip away everything except the exploration dilemma.
- Regret as the thing you actually minimise.
- epsilon-greedy vs UCB (Auer, 2002) vs Thompson sampling (Thompson, 1933).
- *Figure:* cumulative regret of the three on a 10-armed testbed.
- Where this is the whole job: A/B tests, recommendations, ad auctions, clinical trials.
- Contextual bandits in one frame, and the line where they stop being enough (no state transitions).

### Section 3 - The MDP
- The loop (reuse), the tuple (reuse), trajectories and returns.
- Why discount (reuse figure), and gamma changes the plan (reuse corridor figure).
- Deterministic vs stochastic policies, and when stochastic is strictly better.

### Section 4 - Value
- V and Q side by side (reuse).
- **Bellman expectation equation**, derived - one step of conditioning, nothing more.
- **Bellman optimality equation**, derived from it. The max is what makes it nonlinear.
- Predict-first frame (reuse): which square has the highest value?

### Section 5 - Solving it exactly *(mostly new)*
- Policy evaluation as a fixed-point iteration.
- Policy improvement, and the policy improvement theorem stated.
- Policy iteration vs value iteration; **generalised policy iteration** as the picture every later
  algorithm is an instance of.
- *Figure:* the GPI two-line diagram, plus policy iteration vs value iteration sweep counts on the
  gridworld (measured, not asserted).
- Gridworld solved exactly (reuse both figures).
- **What we just did was cheating** (reuse): you needed P and R.

**Next:** you do not get P.

---

## Deck 2 - Learning from experience

*Tabular model-free control. The deck where an agent finally improves.*

### Section 1 - Model-free vs model-based
- The split, and what each buys you.

### Section 2 - Monte Carlo *(new)*
- Wait for the episode, average the returns. First-visit vs every-visit.
- Unbiased, high variance, needs episodes to terminate.

### Section 3 - Temporal difference *(new)*
- TD(0): bootstrap off your own estimate. The TD error as the unit of learning.
- **Bias-variance, again.** MC is unbiased/high-variance, TD is biased/low-variance. This is the same
  axis as chapters 2 and 5, which is worth saying out loud.
- *Figure:* MC vs TD learning curves on the gridworld, same budget, showing the variance gap.
- n-step returns; TD(lambda) and eligibility traces at intuition level (one frame, the lambda knob
  returns in GAE two decks later).

### Section 4 - Control
- ε-greedy (reuse the measured cost figure), GLIE conditions.
- **SARSA** (on-policy) vs **Q-learning** (off-policy), expected SARSA.
- *Figure:* cliff-walking - the canonical picture of on- vs off-policy behaviour. SARSA walks the safe
  path, Q-learning walks the edge. (New; this is the example the current deck's SARSA frame lacks.)

### Section 5 - Importance sampling *(new, load-bearing)*
- How to evaluate policy A using data from policy B. The ratio, derived.
- Why the variance explodes over long trajectories.
- **Flag it forward:** this ratio is exactly what PPO clips and what GRPO reweights. Two decks from
  now, that equation should look familiar rather than arbitrary.

### Section 6 - Q-learning in practice
- The update, each term labelled (reuse).
- One update by hand (reuse).
- The training demo (reuse): matches pi* by episode 135, 12/13 squares at the end.
- **Maximisation bias** (reuse): 13/13 states overestimated, mean +0.079. Double Q-learning.
- Convergence conditions (Robbins-Monro), and why the constant-alpha run drifted.

---

## Deck 3 - Deep RL I: value-based

### Section 1 - The wall
- 256^33600 states (reuse). Generalisation, not memorisation.
- Function approximation: Q(s,a; theta). What breaks.
- **The deadly triad** (new): bootstrapping + off-policy + function approximation. Any two are fine;
  all three can diverge. *Figure:* Baird's counterexample divergence.

### Section 2 - DQN
- The network, the loss, the semi-gradient.
- **Experience replay** - why correlated samples break SGD.
- **Target network** - why chasing your own tail diverges.
- The full algorithm.
- *Figure:* the same gridworld Q-learning run with and without a target network (measured).

### Section 3 - The DQN family
- Double DQN (the maximisation bias from deck 2, fixed).
- Dueling architecture; prioritised replay; Rainbow (Hessel, 2018) and which component earned its keep.
- *Figure:* Rainbow ablation, redrawn.

### Section 4 - Honest limits
- 50M frames, ~38 days of play per game (reuse).
- Reproducibility: "Deep RL that Matters" (Henderson, 2018) - seed variance large enough to reverse
  conclusions. *Figure:* same algorithm, different seeds.
- Tooling: Gymnasium, Stable-Baselines3, CleanRL. One frame, no API teaching.

---

## Deck 4 - Deep RL II: policy gradients

*The deck PPO deserves.*

### Section 1 - Why parameterise the policy
- Value-based vs policy-based (reuse).
- Continuous actions, stochastic optima, huge action spaces.

### Section 2 - The policy gradient theorem *(derived)*
- The log-derivative trick, in full. Six lines.
- What the result says in words: push up the log-probability of actions that led to good returns.
- REINFORCE, and its variance (reuse figure: wrong direction 33.9% of the time without a baseline).

### Section 3 - Variance reduction
- Baselines, and the proof that a state-dependent baseline does not bias the gradient.
- Advantage A = Q - V.
- Actor-critic; A2C and A3C.
- **GAE** - the bias-variance knob, and the lambda from TD(lambda) returning.

### Section 4 - Trust regions and PPO *(the core, ~6 frames)*
- Why one bad step is unrecoverable in RL but not in supervised learning: the policy generates the
  next batch of data.
- *Figure:* a policy collapse - performance falling off a cliff after too large a step (measured on a
  toy problem).
- The surrogate objective, and where the importance ratio comes from (deck 2 cashes in here).
- TRPO's KL constraint, and why nobody wants to compute a Hessian-vector product.
- **PPO's clip**, both signs of the advantage (reuse and extend `ppo_clip.pdf`).
- The full PPO objective: clip + value loss + entropy bonus. Every term explained.
- Practical failure modes: entropy collapse, KL blowup, too many epochs per batch.

### Section 5 - Continuous control
- DDPG -> TD3 (twin critics, delayed updates, target smoothing) -> SAC (maximum entropy).
- One frame each, plus a "which one when" decision table.

---

## Deck 5 - Planning, search and self-play

*Finally explains the cold open from deck 1.*

- Model-based RL: learn P, then plan. Dyna as the bridge.
- Why model error compounds, and what that costs you.
- **MCTS**: selection, expansion, simulation, backup. UCT and the bandit connection from deck 1.
- *Figure:* an MCTS tree growing over iterations.
- **AlphaGo -> AlphaZero -> MuZero**: what each removed. AlphaGo used human games; AlphaZero used
  none; MuZero learned the rules too.
- **Self-play and multi-agent RL**: non-stationarity, Nash equilibria at intuition level.
  Hooks directly onto the existing tic-tac-toe project (100% draws vs perfect play).
- World models and Dreamer; learning in imagination.
- Exploration beyond epsilon: curiosity, count-based methods, RND, Go-Explore.
- **Bridge frame:** search at decision time = test-time compute. This is the frame that makes
  reasoning LLMs (deck 8) feel inevitable rather than magical.

---

## Deck 6 - RL in the real world

*The honest deck. Mostly the material that stops a student from applying RL where it does not belong.*

- **Reward design.** Potential-based shaping (Ng, Harada & Russell, 1999) and why it is the only
  provably safe kind. *Reuse the measured collapse from the tic-tac-toe project: bonus 0.3 changes
  nothing, bonus 3.0 loses 50.7% as X against perfect play.* That number is already in the repo.
- Reward hacking / Goodhart (reuse), with the boat-race example.
- **Offline RL**: you have logs, not a simulator. Distribution shift and extrapolation error.
  BCQ / CQL / IQL at one frame each.
- **Off-policy evaluation**: how you estimate a new policy's value from old logs without deploying it.
  Importance sampling, third appearance.
- **Imitation learning**: behaviour cloning, its compounding-error problem, DAgger. Inverse RL.
  **Bridge: behaviour cloning is SFT.**
- Decision Transformers - RL as sequence modelling, which connects straight back to ch9.
- Sim-to-real and domain randomisation.
- **When not to use RL** (reuse the decision table) and the sample-cost price.
- **The bitter lesson** (reuse both frames).

---

## Deck 7 - RL for LLMs I: alignment

### Section 1 - A language model is an agent
- Text generation as an MDP (reuse).
- What is weird about this MDP: ~50k actions, deterministic transitions, reward only at the end,
  horizon in the thousands, and a policy that already speaks the language before training starts.

### Section 2 - Learning a reward
- Where the reward comes from when nobody can write it down.
- Preference data, and the **Bradley-Terry model** derived into the reward-model loss.
- Reward model quality, and its ceiling (reuse `rm_vs_human.pdf` from `10_instructgpt`).

### Section 3 - RLHF with PPO
- The three stages (reuse, expanded).
- **The KL-to-reference penalty**: what it is, why it exists, what happens without it.
- The alignment tax (reuse `alignment_tax.pdf`).
- InstructGPT: 1.3B beating 175B on human preference (reuse `win_rate.pdf`).
- The infrastructure cost: four models in memory at once.

### Section 4 - DPO *(derived)*
- The closed-form optimal policy of the KL-regularised RLHF objective.
- Invert it: the reward is an implicit function of the policy. Substitute into Bradley-Terry.
- The DPO loss falls out. **No reward model, no sampling, no RL loop.**
- *Figure:* the loss shape (reuse `loss_shape.pdf` from `02_dpo`).
- DPO's limits: offline, bounded by the preference pairs, length bias, needs a reference model.

### Section 5 - The DPO family
- IPO (fixes the overfitting), KTO (binary thumbs up/down instead of pairs),
  SimPO (drops the reference model; +6.4 AlpacaEval 2 over DPO),
  ORPO (merges SFT and alignment into one objective).
- *Figure:* the "progressive simplification" ladder - each method removes one requirement.
- **When DPO, when PPO.** DPO cannot exceed its data; online RL can.

---

## Deck 8 - RL for LLMs II: reasoning

### Section 1 - The verifier replaces the human
- **RLVR.** Math answers, unit tests, proof checkers. Binary, cheap, consistent, unbribable-ish.
- Why this was the unlock: no annotator bottleneck, no reward model to hack.

### Section 2 - GRPO *(derived from PPO)*
- Start from the PPO objective from deck 4. Delete the critic. Replace the advantage with a
  group-relative z-score over G sampled answers.
- The objective, term by term (reuse `ppo_vs_grpo.pdf`, `group_advantage.pdf`).
- Why the group mean is a valid baseline - the baseline theorem from deck 4 applies.
- What it saves: no value network, roughly half the memory.
- The KL estimator (reuse `kl_estimator.pdf`).

### Section 3 - R1 and the aha moment
- DeepSeek-R1-Zero: pure RL from a base model, no SFT. Emergent self-verification.
- Response length growing on its own (reuse `response_length.pdf`, `aime_emergence.pdf`).
- R1's full four-stage pipeline, and why R1-Zero alone was not shippable.
- Distillation vs RL for small models (reuse `distill_vs_rl.pdf`).

### Section 4 - GRPO breaks, and the fixes *(the frontier)*
One frame per failure mode, each paired with its fix:
- **Length bias** -> Dr. GRPO (normalise by a fixed length, not the sampled one).
- **Entropy collapse** -> DAPO's clip-higher.
- **Uninformative groups** (all-correct or all-wrong give zero gradient) -> DAPO's dynamic sampling.
- **Vanishing gradients on long CoT** -> DAPO's token-level loss.
- **Token/sequence mismatch, MoE instability** -> GSPO's sequence-level ratio. Used for Qwen3;
  removed the routing-replay hack GRPO needed.
- DAPO's headline: 50 points on AIME 2024 with Qwen2.5-32B, at half the training steps of
  R1-Zero-Qwen-32B.

### Section 5 - Where the reward comes from next
- **ORM vs PRM.** Outcome vs process supervision (reuse `orm_prm_bestofn.pdf`, `step_scoring.pdf`
  from `14_chain_of_thought`; the Lightman et al. paper is already in `materials_md/`).
- Best-of-N and verifier-guided search - deck 5's test-time compute bridge cashed in.
- Self-play and synthetic curricula: SPIN, SPICE.

### Section 6 - Agentic RL
- Multi-turn tool use as a POMDP. Credit assignment across a trajectory of tool calls.
- Rollouts as the bottleneck; verl, NeMo Gym.
- **Forward pointer to ch18 agents.**

### Section 7 - Honesty and recap
- Reward hacking in LLMs: sycophancy, verifier gaming, formatting exploits, the KL budget.
- What RL does and does not add: sharpening what pretraining already contains, vs genuinely new
  capability. State the open disagreement rather than resolving it.
- Recap across all eight decks + `Next:` to `llm_training/`.

---

## Figure budget

Reused from `ch11_rl/fig/`: 10 existing.
Reused from `llm_training/slides/*/fig/`: ~12 (GRPO, DPO, InstructGPT, R1, CoT decks).
**New Python figures needed: ~22.** All matplotlib into `ch11_rl/fig/` via `ch11_rl/py_src/`, per
`SLIDE_STYLE.md`. All CPU-cheap; the heaviest are the bandit testbed, the MC-vs-TD curves, the
cliff-walking comparison, and the seed-variance plot. Nothing needs a GPU.

## Practicals available

- **Existing:** the tic-tac-toe self-play project (already written, already measured).
- **Candidates**, adapted from HF Deep RL Course Colabs, all CPU-feasible:
  1. Tabular Q-learning on FrozenLake/Taxi (deck 2).
  2. REINFORCE from scratch on CartPole (deck 4).
  3. PPO from scratch, CleanRL-style, on CartPole or LunarLander (deck 4).
  4. GRPO on a small model with TRL, following HF LLM Course ch12 (deck 8) - **needs a GPU, so Colab
     via the `colab` CLI, not local.**

## Open decisions for the instructor

1. **How many decks** - 8, 7, 6 or 4 (see collapse options above). This is a schedule decision:
   `ml/00_plan.md` currently allots RL 2-3 sessions on Oct 21/23. Eight decks makes RL the largest
   chapter in the course and pushes the projected finish from ~Nov 11 into late November. Open
   question #2 in that file already asks whether RL is really two blocks.
2. **Math depth.** `RL_CHAPTER_PLAN.md` records a decision: *"intuition-first, minimal derivation,
   a deliberate deviation from SLIDE_STYLE.md."* This plan reverses it for six results
   (Bellman, the policy gradient theorem, importance sampling, the baseline theorem, DPO, GRPO) and
   keeps intuition-only for the rest. Confirm the reversal.
3. **Classic breadth.** Decks 5 and 6 (planning/search/self-play, offline RL/imitation) are not
   needed for the LLM material. Keep them, survey them, or cut them.
4. **Overlap with `llm_training/`.** Decks 7 and 8 teach algorithms that have paper-seminar decks
   next door. Proposed: ch11 teaches the method, `llm_training` stays as "what this paper found",
   reusing figures across. Alternative: strip background frames out of the `llm_training` decks.
5. **Naming.** Proposed `L32` + `L32b..L32h`, matching `L23b`/`L23c`. Note `CONVENTIONS.md` says
   L-prefixes are legacy, but all of `ml/ch9`-`ch18` still uses them, so switching here alone would
   be inconsistent. Deck 1 would be renamed `L32_rl_problem.tex`, which changes one link in `rl.qmd`.

## Verify before building

Every year, arXiv ID and headline number in this plan gets web-checked at build time per the house
rule. The ones I am least sure of and have flagged: Dr. GRPO's arXiv ID and exact title, the
SimPO/KTO/ORPO/IPO years, and the SimPO AlpacaEval 2 delta.
