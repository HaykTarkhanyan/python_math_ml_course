# 04 - World models and planning

## LeCun's proposal, in the shape he proposed it

The 2022 position paper *A Path Towards Autonomous Machine Intelligence* is not an architecture
paper - it is a sketch of a whole agent, with JEPA occupying one box in it. `[search]`, the six
modules:

| Module | Job |
|---|---|
| **Perception** | Estimate the current state of the world from sensors |
| **World model** | Predict how the state will evolve, including under the agent's own actions. **This is where JEPA goes** |
| **Cost** | Score a state. Split into an immutable *intrinsic cost* (hardwired drives, pain, hunger) and a trainable *critic* that predicts future intrinsic cost |
| **Actor** | Propose action sequences and optimise them against the predicted cost |
| **Short-term memory** | Hold state, predictions and costs over the current episode |
| **Configurator** | Reconfigure all of the above for the task at hand - the executive that decides what the agent is currently trying to do |

The paper distinguishes two ways of acting:

- **Mode 1** - a reactive policy. Perception in, action out, one pass, no simulation. Fast, cheap,
  and this is what almost all deployed robot policies do, including every VLA in chapter 15.
- **Mode 2** - deliberative. Roll the world model forward over candidate action sequences, score
  the imagined outcomes, pick the best, execute the first step, repeat. Slow, but able to handle
  situations no policy was trained on.

The Kahneman System 1 / System 2 analogy is explicit in the literature and is a fair way to
introduce it, provided you say clearly that this is an engineering design borrowing a name, not a
claim about human cognition.

**H-JEPA** is the hierarchical version: a stack of JEPAs, each predicting at a coarser level of
abstraction and a longer time scale than the one below. The motivation is that a plan for "make
coffee" should not be represented as a sequence of joint torques. Note for honesty: H-JEPA is
largely still a proposal. The shipped systems are single-level.

## What "world model" means - and the fact that it means two things

This is the single most confusing point in the area, and the chapter should settle it early.

**Definition A - generative.** A world model is something that, given a state and an action,
produces the *observation* you would see next. Sora, Genie, NVIDIA Cosmos. You can watch its
predictions. It is evaluated on whether the video looks right.

**Definition B - predictive/latent.** A world model is something that, given a state and an
action, produces the *next state in its own internal representation*. You cannot watch its
predictions. It is evaluated on whether you can plan with it.

JEPA is firmly definition B, and LeCun's argument for B over A is the argument from `01_the_idea.md`
scaled up: rendering the pixels of an imagined future means committing to detail you cannot know
and do not need. If you are deciding whether to reach left or right, the exact texture of the
tablecloth in the imagined future is wasted computation, and getting it wrong is not an error you
should be penalised for.

The counter-argument, which deserves a fair hearing on the slide: a generative world model can be
**inspected**. You can look at what Sora thinks happens next and see that the spoon passed through
the bowl. A latent world model that has learned something wrong gives you no such handle, and
debugging it is genuinely harder.

## JEPA is not the first latent world model - and the difference is instructive

Chapter 11 (RL) already contains the ancestors:

- **World Models** (Ha and Schmidhuber, 2018): VAE encoder plus a recurrent dynamics model,
  trained with reconstruction.
- **PlaNet / Dreamer** (Hafner et al.): learn latent dynamics and train the policy entirely
  "in imagination". Dreamer still carries a decoder and a reconstruction term.
- **TD-MPC / TD-MPC2**: latent dynamics with **no** reconstruction at all.

So "predict in latent space" is not new. What is new is *what stops the latent from collapsing*,
and this is the comparison worth drawing explicitly:

> In model-based RL, the latent cannot collapse because it is also required to predict **reward**
> and **value**. Those are external, grounded signals. JEPA has no reward. It is trying to learn a
> world model from passive video with nothing external to anchor the representation - which is
> exactly why it needs the EMA teacher, or SIGReg, or a contrastive term.

That reframes all the anti-collapse machinery in `01_the_idea.md` as the price of dropping reward,
and it connects two chapters that otherwise look unrelated.

## The lineage from "encoder" to "world model"

V-JEPA 2-AC did not appear from nowhere. The tutorial traces a clean four-step progression, and it
is a good structure for a slide because each step adds exactly one thing.

| Step | Model | What it adds | What it still lacks |
|---|---|---|---|
| 1 | **I-JEPA** | Predict a representation from a representation | No actions, no time |
| 2 | **IWM** (Image World Model) | Condition the predictor on a **known transformation** (rotation, translation, colour), so it learns how representations move under action-like changes | The transformations are *sampled during view generation*, not chosen as control actions. A transformation-conditioned internal model, not a planner |
| 3 | **Seq-JEPA**, **PLDM** | Sequences: predict the next latent conditioned on an action. PLDM trains from **reward-free offline trajectories** with VICReg-style regularisation and does goal-conditioned planning at test time | PLDM needs a multi-term objective with **six** tunable loss weights |
| 4 | **V-JEPA 2-AC**, **LeWM** | Full goal-conditioned planning. LeWM trains end to end from raw pixels with only **two** loss terms - next-embedding prediction plus a Gaussian latent regulariser - cutting PLDM's six hyperparameters to one | Still goal-conditioned; see the limitation below |

**Primary sources for the last row**, both obtained 2026-08-08 from the Welch Labs series (the
first was recommended by Randall Balestriero on camera; the second appears as an on-screen credit):

- **LeWorldModel (LeWM)** - arXiv **2603.19312**
- **Hierarchical Planning with Latent World Models** - Zhang, Wancong et al., arXiv **2604.03208**
  (2026). Two levels of hierarchy take PushT from a **5-step** to a **15-step** planning horizon;
  the high-level model's predictions become **subgoals** for the low level. This is the concrete
  answer to the goal-conditioning limitation below.

Neither paper has been read directly yet - the IDs come from the video. Read before quoting
anything beyond what is recorded here.

**LeWM is the one worth showing students who want to build something.** It is a compact JEPA world
model trained on a **single GPU in a few hours**, reports planning up to **48x faster** than
foundation-model-based world models, stays competitive across 2D and 3D control tasks, and uses a
lightweight cross-entropy-method search rather than a trained planner. Everything else in this file
is a model nobody in the course can reproduce; this one is a plausible student project.

## Action-conditioned prediction: V-JEPA 2-AC

The concrete instance. `[read]`:

1. Pretrain V-JEPA 2 on over 1M hours of internet video. **No actions.** The model learns how the
   world moves, from watching.
2. Freeze that, and train a predictor `s_{t+1} = P(s_t, a_t)` on **under 62 hours** of unlabelled
   robot video from the public DROID dataset, with the recorded actions and end-effector poses.

The asymmetry is the headline: essentially all of the knowledge comes from passive observation,
and a rounding error's worth of robot data is enough to attach a control interface to it. This is
the direct answer to chapter 15's central problem - that robot data does not exist at internet
scale. JEPA's claim is that it does not need to, because *physics can be learned from video and
only the mapping from commands to consequences needs robot data*.

## Planning as search in embedding space

Given a **goal image** `g`:

```
s_goal = Enc(g)
at each step:
    sample K candidate action sequences a_{1..H}
    for each, roll the predictor forward:  s_1 = P(s_t, a_1), s_2 = P(s_1, a_2), ...
    score by  || s_H - s_goal ||          <- the energy from 01_the_idea.md, now a cost
    keep the best, resample around them   <- cross-entropy method
    execute only the first action, then replan            <- model-predictive control
```

Three things to make sure students see:

- **The cost function is a distance between embeddings.** No reward was ever specified, no reward
  model was trained. "What you want" is expressed as a picture.
- **Nothing is ever rendered.** The whole rollout happens in the representation, which is why it
  is fast enough to be considered at all.
- **Execute one step and replan.** This is standard MPC and it is what makes an imperfect model
  survivable - errors do not compound over the full horizon because the horizon keeps resetting.
  Contrast this directly with chapter 15's compounding-error figure: MPC is the other answer to
  the same problem that action chunking addresses.

Results `[read]`: deployed **zero-shot** on Franka arms in two labs whose data the model had never
seen, no task-specific training and no reward, **65-80%** success on pick-and-place with novel
objects. `[search]`: roughly **16 seconds** of planning per action.

That last number is the honest counterweight. This is not a controller. It is a demonstration
that the representation contains enough physics to plan in, running about three orders of
magnitude too slow for real use.

## The limitation hiding inside "you say what you want with a picture"

The goal-image framing is elegant and it is also the method's boundary. Stated plainly by the
tutorial, and this deserves a slide of its own:

> JEPA planning formulations are **goal-conditioned**. They assume a target representation exists
> and search for actions whose latent rollout approaches it. That is effective when **the goal is
> known and the path is unknown**. It is much less direct for open-ended decision-making where the
> optimal target state is unspecified.

Which is to say: "put the cup on the shelf" works, because you can photograph the outcome. "Tidy
the kitchen", "make this safe", "explore" do not, because there is no single image to aim at. Those
need exactly the things JEPA planning was advertised as avoiding - reward design, goal proposal, or
a mechanism that generates intermediate subgoals.

Two further honest points from the same section:

- **Latent errors compound over multi-step rollouts.** The predictor is imperfect, and rolling it
  forward feeds its own errors back in. This is ch15's compounding-error figure again, in a third
  setting, and it is the reason the horizon has to stay short and MPC has to replan.
- **The encoders and the predictor have to evolve at compatible rates.** Too fast and the predictor
  chases a drifting target; too slow and it learns from a stale one. Another silent failure mode
  with no signal in the loss.

## The benchmarks Meta published against itself

Released with V-JEPA 2 `[read]`:

| Benchmark | What it tests | Human | Models |
|---|---|---|---|
| **IntPhys 2** | Physically plausible vs implausible scenario, in video | near-perfect (85-95% band reported) | at or near **chance** |
| **MVPBench** | Video-language understanding on minimally-different video pairs, built to kill shortcut solutions | 85-95% | large gap |
| **CausalVQA** | Counterfactual, anticipation and planning questions - "what could happen", not "what happened" | 85-95% | large gap |

The design of MVPBench is worth a slide on its own: pairs of videos that differ minimally but
have different correct answers, so a model that has learned a shortcut - answer from the caption
prior, or from a single frame - scores at chance by construction. That is a general technique for
benchmark design and it applies far beyond video.

The finding is the important part: **a model that is state of the art at recognising and
anticipating actions is at chance at telling whether a scene is physically possible.** Whatever
V-JEPA 2 has learned, it is not the intuitive physics the world-model thesis promises. Meta says
so in its own launch material, which is the strongest available evidence that this is a real
result and not a rival's framing.
