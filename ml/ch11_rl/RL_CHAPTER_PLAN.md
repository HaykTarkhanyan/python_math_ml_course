# Chapter plan — Reinforcement Learning (L32)

**Status:** BUILT 2026-08-06. `L32_reinforcement_learning.pdf`, **40 pages**, 9 Python-generated
figures, compiles clean (3 residual overfull vboxes, largest 5.97pt / ~2mm, all verified
non-clipping by rendering every page). Registered as `ml/ch11_rl/rl.qmd` in `_quarto.yml`,
positioned immediately before `ml/llm_training/`.

## Changes from this outline during the build

- **The Bitter Lesson was added** (instructor request, mid-build). Sutton 2019, web-verified.
  Two frames: the essay's claim with the four canonical domains, then "where this lecture is
  Exhibit A -- and where it is not", which pushes back honestly (convolution and attention are
  human-designed inductive biases; every method here is scaffolding someone designed; compute is
  not free). Foreshadowed in frame 3 by the AlphaGo-vs-AlphaGo-Zero clarification, which was
  already there. Deck went 37 -> 40 pages.
- **`tcolorbox` and `booktabs` added to the shared `ml/preamble.tex`** (instructor approved).
  Purely additive; no existing deck changes appearance.
- **Two figures were redesigned after their own measurements refuted them** (see below).
- Frame 21 (SARSA) kept - instructor is fine with long decks.
- Cold open uses a **schematic** Go board, not the real game-2 position. Reproducing 37 real
  stones needs a verified game record; the teachable content is the 3rd/4th vs 5th line
  convention. Labelled as a schematic on the slide itself.

## Measurements that changed the design

1. **The REINFORCE baseline figure was wrong on the first build.** Holding the score function
   fixed, subtracting a constant baseline shifted the mean and left the variance *identical*
   (sd 0.681 both ways) while the title claimed it "shrinks the scatter". Fixed by making the
   score function action-dependent, which is the real reason a baseline works. Now:
   **sd 1.126 -> 0.110**, and the gradient points the **wrong way 33.9% of the time without a
   baseline, 0.0% with**. That number is now quoted on the slide.
2. **The gamma-changes-the-policy figure did not.** On the 4x4 gridworld the optimal policy
   differed in only 1-3 of 12 squares across gamma, and *non-monotonically* (gamma=0.9 gave
   `left` at (2,3) while both 0.5 and 0.95 gave `down`). Replaced with a purpose-built corridor
   MDP - small reward near, large reward far - where the decision genuinely flips, at
   gamma ~ 0.68. Kept the gridworld for everything else.

## Figures actually built (9, all in `py_src/` -> `fig/`)

`gridworld_layout`, `gridworld_values`, `gridworld_policy` (value iteration, 23 sweeps),
`discount_horizon`, `gamma_policy` (corridor), `epsilon_cost` (exact policy evaluation),
`reinforce_variance`, `ppo_clip`, `go_lines` (schematic).

Numbers quoted on slides, all from the scripts' logs: `V*(start)=0.2075`,
`V*(beside goal)=0.7954`, eps-greedy `0.208 / 0.152 / -0.418`, corridor crossover
`gamma ~ 0.68`, REINFORCE `33.9% -> 0.0%`, `sd 1.13 -> 0.11`.

## Still open

- No homework (deferred). The natural one is a tabular Q-learning notebook on this gridworld -
  which is exactly the training demo the instructor deferred at the interview.
- No video recorded.

---

**Original outline below, as approved.**

**Source:** adapt `misc/dl4nlp/18_reinforcement_learning.tex` (24pp, 20 content frames) into the
`ml/` course, per instructor decision 2026-08-06.

---

## Instructor decisions (2026-08-06)

1. **Adapt the existing deck**, do not write from scratch.
2. **Intuition-first, minimal derivation.** Equations are shown and explained, not derived.
   *This is a deliberate deviation from `ml/SLIDE_STYLE.md`, which asks for full step-by-step
   derivations.* Recorded here so the next person does not "fix" it.
3. **No training demo.** Figures are Python-generated from **hand-specified** MDPs - no agent is
   trained. The instructor may add a real training run later; nothing in this design blocks it.

## Why this lecture exists

`ml/llm_training/` is a 12-paper seminar track built on **PPO, GRPO, DPO, InstructGPT and R1** -
all reinforcement learning - and RL is taught nowhere in the `ml/` course. Students read those
papers with no foundation to read them from. This deck closes that gap.

The foundations *do* exist today, compressed into ~7 frames of a deck in `misc/dl4nlp/`, which is
a different course and is not registered in `_quarto.yml`. So the material is invisible here.

## Relationship to `misc/dl4nlp/18`

`dl4nlp/18` **stays exactly as it is** - it serves that course and its LLM framing is correct there.
This is a fork, not a move. The two differ deliberately:

| | `dl4nlp/18` | `ml/ch11_rl/L32` |
|---|---|---|
| Framing | "RL for language models" | "RL as a subject", LLMs as the payoff |
| Foundations | ~7 dense frames | ~18 frames, room to breathe |
| Figures | TikZ boxes only | Python-generated where they carry meaning |
| Structure | Part I / Part II title frames | house: cold open, Outline, `[plain]` transitions, recap |

## Placement

- New chapter folder `ml/ch11_rl/`, deck **L32** (the L-sequence currently ends at L31, diffusion).
- Sidebar position: **immediately before `ml/llm_training/`**, which is what it feeds.
- Does not touch `ml/ch10_diffusion/` - safe to build while another session holds diffusion.
- Registering it touches `_quarto.yml`, the one shared file. Do that in a single small commit at the
  end to minimise conflict surface.

---

## Deck outline (~36 frames)

### Cold open
1. **A machine that was never told the rules.** One striking result (AlphaGo move 37, or an agent
   learning to walk) framed as: no labels, no correct answers, only a score at the end.
2. **Why every model so far cannot do this.** Supervised learning needs (input, correct answer).
   Here there is no correct answer - only consequences, often delayed. This is a different problem,
   not a harder version of the same one.
3. Outline.

### Section 1 - The setting
4. `[plain]` transition: *Agent, environment, reward.*
5. **The loop.** Agent / environment / action / state / reward. (adapted from dl4nlp frame 2)
6. **Reward is not loss.** Delayed, sparse, and not differentiable - the three properties that break
   everything from chapters 1-10. **New frame**, this is the conceptual crux.
7. **The MDP tuple** $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, each element in words first.
8. **Why discount?** $\gamma$ made concrete: the same reward 1, 10 and 100 steps away.
   *Python figure: discounted value vs horizon for several $\gamma$.*
9. **Worked example: a 4x4 gridworld.** The running example for the whole deck.
   *Python figure: the grid, rewards, terminal states.*

### Section 2 - Value
10. `[plain]` transition: *How good is this position?*
11. **$V^\pi(s)$ and $Q^\pi(s,a)$** side by side, in words. (dl4nlp frame 3, split)
12. **The Bellman equation**, stated and read aloud in English, not derived.
13. **Predict-first:** which square of the gridworld has the highest value - the one nearest the
    goal, or the one with most escape routes? Then reveal.
    *Python figure: value heatmap over the grid (computed by value iteration on the specified MDP).*
14. **Optimal policy from values:** $\pi^*(s) = \arg\max_a Q^*(s,a)$.
    *Python figure: policy arrows over the grid.*

### Section 3 - Learning without a model
15. `[plain]` transition: *You do not get a map.*
16. **Model-free vs model-based** - the split that matters in practice.
17. **Q-learning**, the update rule with each term labelled. (dl4nlp frame 4)
18. **The algorithm**, six numbered steps.
19. **By hand:** one Q-update computed with real numbers on the gridworld. **New frame** - the house
    style wants a worked-numbers frame where mechanics are computable.
20. **Exploration vs exploitation**, $\epsilon$-greedy. **New frame** - dl4nlp mentions it inside the
    algorithm box but never explains the dilemma.
    *Python figure: return vs $\epsilon$, showing both extremes are bad.*
21. **On-policy vs off-policy** (Q-learning vs SARSA), one frame, conceptual.

### Section 4 - When the table will not fit
22. `[plain]` transition: *Atari has more states than atoms.*
23. **The tabular wall.** $|\mathcal{S}| \approx 256^{33600}$. (dl4nlp frame 5)
24. **DQN:** the network, plus experience replay and target network, and *why each was needed*.
25. **What DQN achieved**, and the honest limits.

### Section 5 - Optimising the policy directly
26. `[plain]` transition: *Skip the values. Learn the behaviour.*
27. **Value-based vs policy-based**, and when each is the right tool. (dl4nlp frame 6)
28. **The policy gradient**, stated with the intuition: *push up the probability of actions that
    led to good returns.* Not derived (instructor decision 2).
29. **REINFORCE**, and its variance problem.
    *Python figure: gradient-estimate spread across episodes, from a specified toy distribution.*
30. **Baselines and advantage** $A(s,a) = Q(s,a) - V(s)$; actor-critic in one frame.
31. **PPO:** why unconstrained steps collapse, and what clipping does.
    *Python figure: the clipped objective as a function of the probability ratio, both signs of A.*

### Section 6 - The payoff
32. `[plain]` transition: *A language model is just another agent.*
33. **Text generation as an MDP:** state = tokens so far, action = next token, reward = at the end.
34. **RLHF in three stages**, and where PPO sits. Explicit pointer to `llm_training` [10] InstructGPT.
35. **Reward hacking / Goodhart**, with a concrete example. The honesty frame.
36. **Recap + `Next:` box** pointing at `ml/llm_training/` - specifically [01] GRPO, [02] DPO,
    [10] InstructGPT, [11] R1, which students can now actually read.

---

## Figure budget (Python, `py_src/` -> `fig/`, no training)

| Figure | Frame | What it shows |
|---|---|---|
| `discount_horizon.pdf` | 8 | discounted reward vs steps, several gamma |
| `gridworld_layout.pdf` | 9 | the running example |
| `gridworld_values.pdf` | 13 | value heatmap (value iteration on a specified MDP) |
| `gridworld_policy.pdf` | 14 | greedy policy arrows |
| `epsilon_tradeoff.pdf` | 20 | return vs epsilon |
| `reinforce_variance.pdf` | 29 | spread of the gradient estimate |
| `ppo_clip.pdf` | 31 | clipped objective vs probability ratio |

Seven figures. Value iteration on a 4x4 grid is deterministic arithmetic, not training - it runs in
milliseconds and gives exact numbers the slides can quote.

## Open questions

1. **Cold open choice:** AlphaGo move 37 (dramatic, needs an image) or an agent learning to walk
   (visual, easy to source a still)? Or the gridworld itself, kept abstract?
2. **SARSA (frame 21):** keep, or cut as the first thing to go if the deck runs long?
3. **Does this need a homework?** Every other built chapter currently has none, so the honest
   default is no - but a tabular Q-learning notebook would be a natural later addition, and is
   exactly the "demo" the instructor deferred.
