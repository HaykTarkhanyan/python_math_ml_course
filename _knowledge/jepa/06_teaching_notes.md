# 06 - Teaching notes

## What this course already taught that the chapter can lean on

JEPA is unusually well-prepared-for by this syllabus. Almost nothing has to be introduced cold.

| Prerequisite | Where | How it is used |
|---|---|---|
| Autoencoders, reconstruct-the-input | `ch8_autoencoders` | The thing JEPA is defined against. "You already built the generative architecture" |
| Contrastive learning, two towers, shared space, temperature | `ch12_vlm` (CLIP) | CLIP **is** the joint-embedding architecture. Introduce JEA by pointing at it, not by defining it |
| Patchify, ViT, position embeddings | `ch12_vlm` | I-JEPA is a ViT. Zero new architecture |
| Masked prediction | `ch13_audio` (masked latent prediction shows up in speech SSL) and the LLM material | The masking half of the recipe is familiar |
| Generative modelling of a distribution | `ch10_diffusion`, `ch8b_gans` | The "predicting the mean gives you blur" argument |
| Model-based RL, planning, imagination | `ch11_rl` | JEPA's planning loop is MPC. Dreamer/TD-MPC are the ancestors |
| Compounding error, action multimodality, robot data scarcity | `ch15_vla` | V-JEPA 2-AC is a direct answer to ch15's central problem. Replanning is the other fix for compounding error |
| Learned prior over synthetic data, in-context adaptation | `ch14_tabular_fm` | LaT-PFN literally crosses JEPA with PFNs |

The strongest single hook is **ch12 CLIP**. The chapter can open with "you built a model that
knows whether two things match, but not what the second one *is*" and the whole JEPA idea follows
from repairing that.

## Misconceptions to pre-empt

Ordered by how likely a student is to leave with them.

1. **"JEPA is just contrastive learning."** No negatives anywhere in the objective. The anti-collapse
   device is architectural, not contrastive. (Caveat honestly - see contested item 7.)
2. **"JEPA is an autoencoder in feature space."** In an autoencoder the target is fixed. Here the
   target encoder is *learned and moving*. That single difference creates both the benefit and
   collapse.
3. **"It predicts the future."** I-JEPA has no time axis at all. It predicts one part of a *still
   image* from another. Prediction over time only arrives with V-JEPA. Students conflate "predictive
   architecture" with "predicts the future" constantly.
4. **"Latent space means lower-dimensional."** The target is per-patch embeddings at full model
   width. The compression is **semantic** - detail dropped - not dimensional. Nothing is smaller.
5. **"Collapse is mode collapse."** Same word, different failure. Mode collapse (ch8b) is a
   generator covering one mode of the data. Representation collapse is an encoder mapping
   everything to one point. Say this out loud, because the course has already used the word.
6. **"A world model generates video."** Two incompatible definitions in circulation. Settle it
   explicitly (see `04_world_models_and_planning.md`).
7. **"LeCun showed LLMs don't work."** He argued a position and funded it. The evidence is open.
8. **"The predictor is the world model."** In **I-JEPA the predictor is thrown away** - you keep the
   target encoder. In **V-JEPA 2-AC the predictor is the entire product** and the encoder is
   infrastructure. Same architecture, opposite half is the deliverable. This is one of the best
   single slides available in the chapter.

## Figures

Per `ml/SLIDE_STYLE.md` every essential figure must be Python-generated in `py_src/` writing to
`fig/`. Proposed list, in rough priority order:

1. **The three architectures.** JEA / generative / JEPA as three box-and-arrow panels with the loss
   marked in each. The one diagram the whole chapter refers back to.
2. **Why the pixel target is the wrong target.** A 1-D toy: several plausible continuations of a
   signal, plus the L2-optimal prediction, which is their mean and matches none of them. Reuses the
   ch15 action-multimodality argument in a new place, which is good - students should notice the
   pattern recurring.
3. **The masking strategy, on a real photo.** Draw I-JEPA's actual sampling - 4 target blocks at
   scale (0.15, 0.2), the context block at (0.85, 1.0) with overlaps removed - over
   `ml/ch12_vlm/fig/img/yerevan_market.jpg`, which is already in the repo. Reusing the ch12 image
   is a deliberate callback.
4. **The two ablations, as bars.** Masking strategy (54.2 / 20.2 / 17.6 / 15.5) and target space
   (66.9 vs 40.7). Bar labels on the bars. These two charts carry the empirical argument.
5. **Collapse, measured.** A genuinely runnable toy: small MLP encoders on 2-D synthetic data, train
   the JEPA objective with and without the EMA teacher, plot **embedding variance vs step**. The
   no-EMA run should visibly go to zero. Cheap - seconds of CPU, no GPU, no large arrays - and it
   turns collapse from an assertion into a measurement. This is the chapter's one real experiment.
6. **Energy landscape.** Two surfaces: sculpted (low on data, high off it) vs flat (collapsed).
7. **The efficiency argument.** Accuracy against pretraining epochs for I-JEPA / MAE / data2vec /
   iBOT, showing I-JEPA reaching comparable accuracy at a quarter of the epochs. Honest axis: do
   not crop it to hide that iBOT ends up higher.
8. **The data asymmetry.** 1,000,000 hours of internet video against 62 hours of robot video, log
   scale, labelled. One glance explains the whole V-JEPA 2-AC design.
9. **Planning in latent space.** Candidate action sequences fanning out from the current latent,
   each rolled forward, scored by distance to the goal embedding; the winner highlighted; a note
   that only the first action is executed.
10. **The honest scoreboard.** Human vs model on IntPhys 2 / MVPBench / CausalVQA. This is the
    figure that keeps the chapter from being an advertisement.
11. **Timeline** of the model line, 2022 position paper to V-JEPA 2.1 and AMI Labs.

Figures 1-5 are the ones the chapter cannot do without.

## Exercises

- **Predict-first.** Before showing Table 7: "we train the same model twice, once predicting pixels
  and once predicting representations. How big is the gap?" Almost nobody guesses 26 points.
- **Find the collapse.** Give students the toy JEPA from figure 5 with the EMA disabled and ask them
  to diagnose why the loss is beautiful and the representation is useless. Then ask what they would
  have monitored to catch it.
- **Design a masking strategy.** Give them the four rows of the masking ablation with the accuracies
  hidden and ask them to rank the strategies and justify it. The rasterized row (same context
  budget, 39 points worse) is the interesting one.
- **Which definition?** Give five short descriptions of systems (Sora, Dreamer, V-JEPA 2, a
  next-token LLM, a Kalman filter) and ask which are world models. There is no clean answer, which
  is the point.
- **Read the scoreboard honestly.** Hand them Table 1 and ask "did I-JEPA beat DINO?" The correct
  answer is no, at matched resolution, and it is the paper's own position.

9. **"So JEPA replaces diffusion / GANs."** No. JEPA **cannot reconstruct** - it deliberately
   discards what a decoder would need. High-fidelity synthesis needs a decoder or a generative
   model bolted on. They are not applying for the same job, and the reason JEPA is good at
   understanding is the reason it cannot generate. Students who just finished ch10 will ask this.

## Project ideas for students

> **Deferred out of ch16 (2026-08-08).** The chapter ships explanatory slides only - no practical,
> no project - matching the ch14 decision. These stay here as research, not course content. The
> LeWorldModel one is parked in `DEFERRED_TODO.md` with the reasoning.


- **Train a JEPA world model from scratch - genuinely feasible.** **LeWM** learns a compact latent
  world model end to end from raw pixel transitions with only two loss terms and **one**
  hyperparameter, on **a single GPU in a few hours**, then plans with a cross-entropy-method search
  over latent rollouts. This is the only model in the whole chapter a student can actually
  reproduce, and it exercises everything the chapter teaches - latent prediction, anti-collapse
  regularisation, action conditioning, and goal-conditioned planning - in one artifact. Strongest
  project on this list by some distance.
- Fine-tune a released V-JEPA 2 or I-JEPA checkpoint on a small domain dataset and compare against
  a supervised baseline and against DINOv2, on a dataset with **few labels**. The low-label regime
  is where the SSL argument actually pays and where students will see a real difference.
- Reproduce the target-space ablation at toy scale: same tiny model, pixel target vs
  representation target, on CIFAR-10. The gap should reproduce in direction if not magnitude.
- Take one of the domain JEPAs (ECG-JEPA, S-JEPA for EEG) and evaluate whether the pretraining
  actually helps on a public dataset. These are exactly the small-labelled-data domains the method
  is supposed to serve.

## Tone

Two failure modes to avoid.

**Advertisement.** The material comes wrapped in a lot of advocacy and a billion-dollar funding
round. If the chapter reads as "here is the future of AI", it is wrong and students will discover
it is wrong.

**Dismissal.** The opposite failure is just as bad - "LeCun's pet theory that never worked". The
target-space ablation is real, the efficiency result is real, and predicting-in-latent-space is now
a standard tool that shows up in a dozen modalities.

The chapter should land on: **the objective is a genuine contribution and is here to stay; the
world-model thesis is an open bet with one measured failure and several promising results, and it
is being tested right now with a lot of money.**
