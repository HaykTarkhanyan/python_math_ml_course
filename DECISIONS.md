# Decisions

Significant design choices for this repo, newest first. Each entry records what was decided, when,
why, what was rejected, and what would justify revisiting it. Superseded entries stay - the fact
that we changed our mind, and why, is the valuable part.

Deep supporting research lives in the relevant chapter's `_reference_*/` or `research/` folder;
this file holds the choice and a pointer.

---

## #9 - ch12 (vision-language models) ships as two decks with figures only, no trained model

**Date:** 2026-08-07 · **Status:** active

**Decision.** New chapter `ml/ch12_vlm/`, decks **L33** (how a model sees) and **L34** (how a
model draws), registered between `ch11_rl` and `llm_training`. **Intuition-first**, not full
derivations. **No neural network is trained anywhere in the chapter.**

**Why.** The course could explain transformers (ch9), autoencoders (ch8) and diffusion (ch10)
but not how a chat model reads a pasted photo - the single most visible AI capability to a
non-specialist, and the natural convergence point of four earlier chapters. Two decks because
"seeing" is a settled engineering recipe while "drawing" is an open architectural argument;
that is a real conceptual boundary, not an arbitrary split at 47 frames.

Intuition-first is a **deliberate deviation from `ml/SLIDE_STYLE.md`** (which asks for full
step-by-step derivations), matching the precedent set by `ch11_rl`. It shows in exactly two
places: the contrastive loss and the VQ straight-through estimator, both described in words.

**The chapter still measures something.** `fig/vq_quantization.pdf` fits a k-means codebook on
ch10's 4,481 letters, which is clustering rather than network training (~8 s):

| Codebook K | 8 | 32 | 128 | 512 |
|---|---|---|---|---|
| Reconstruction MSE | 0.01819 | 0.01064 | 0.00844 | **0.00654** |

Two results are taught from it: sharply diminishing returns (128 -> 512 buys 22% for 4x the
vocabulary), and visible stroke breakup even at K=512 because each patch is quantized with no
knowledge of its neighbours. The second is **#8's finding again** - 1-2 px strokes are what
every compression scheme destroys first.

**Alternatives rejected.**
- *Train a VQ-VAE plus an autoregressive generator on the ch10 letters and race it against the
  diffusion model* - the strongest idea in the plan, and cut by the instructor as too much
  build. The chapter now cites published comparisons instead of running its own.
- *A real CLIP zero-shot run on the Armenian letters* - cut. It needs a ~350 MB download and a
  new dependency (`open_clip` or `transformers`), which is a dependency choice that was not on
  the table. Parked in `DEFERRED_TODO.md`.
- *One long deck* (the ch11 shape) and *three decks* - rejected for the boundary reason above.

**What would change this.** If the chapter gets a homework slot, the cut AR-vs-diffusion
project is the obvious candidate and would give the chapter a project matching ch10's and
ch11's. If GPT-4o's architecture is ever published, the L34 "known vs inferred" frame needs
rewriting - it is currently the one frame in the chapter that could teach something false.

---

## #8 - The ՊԱՆԻՐ denoiser is a ONE-level UNet at ch=96; #7's two-level design was the bug

**Date:** 2026-08-07 · **Status:** active · **supersedes #7**

**Decision.** `LEVELS = 1` (24 -> 12 -> 24, a single halving) at `ch=96`, **1.50M params**,
10000 steps. Trained on a rented T4 via the Colab CLI, not locally.

**Why.** #7 assumed capacity was the constraint and went from 266k to 7.03M params. It was
wrong, and the measurement is unambiguous:

| arch | params | steps | final loss | samples |
|---|---|---|---|---|
| 2 levels, 24x24 | 7.03M | 20000 | 0.038 | fragments |
| 2 levels, 32x32 | 7.03M | 10000 | **0.9994** | diverged |
| **1 level, 24x24** | **1.50M** | **10000** | **0.0269** | **legible letters** |

A **4.7x smaller** model produced the best loss of any run and the first readable ՊԱՆԻՐ.
The cause is the same property that made crop-to-ink mandatory in #6: **the strokes are 1-2 px
wide.** Halving twice (24 -> 12 -> 6) leaves them sub-pixel in the deep layers, so the extra
capacity models a representation from which the letter has already been erased. Halving once
keeps them. The 20000-step run also plateaued by ~step 4000, ruling out training length.

**Alternatives rejected.**
- *More capacity* (#7's answer). Falsified above.
- *More steps.* The progression figure shows no change from step 4000 to 20000.
- *32x32.* Not rejected - **untested**. That run diverged (loss 0.9994 = predicting zero)
  because `lr=2e-3` is too hot for 7.03M params; the same run at 24x24 had already shown a
  27.32 loss spike over its first 250 steps. Retest with a lower LR before concluding anything.

**Cost accepted.** The UNet is now built from `ModuleList`s with `levels` as a parameter, so
**state-dict keys changed** and every checkpoint predating this entry is unloadable. Given all of
them produced unusable samples, nothing of value was lost.

**What would change this.** If a 32x32 run at a lower LR beats this, revisit - more pixels is the
other way to stop downsampling from destroying strokes. `pack_mashtots.py` takes a size argument
and `TAG` keeps experiment artifacts apart, so that test is ~10 min on a T4.

---

## #7 - The ՊԱՆԻՐ denoiser is a two-level UNet at ch=64, not digits_ddpm's TinyUNet

**Date:** 2026-08-06 · **Status:** active, **outcome pending** (6000-step run in flight)

**Decision.** `train_panir_ddpm.py` uses its own **two-level** conditional UNet
(24 -> 12 -> 6 -> 12 -> 24, skips at both scales, conditioning injected at all three),
**ch=64, 3.13M params**, rather than reusing `digits_ddpm.py`'s TinyUNet.

**Why.** The first full run *did* converge - loss 1.2 -> 0.0344 - but the samples were
malformed and **Ի effectively failed to render** (per-class ink 0.059 against 0.099-0.123 for
the others; 0.024 in the generated word). Loss went flat at **step ~800** and the remaining
5,200 steps bought 0.005. Flat loss plus bad samples is a capacity limit, not undertraining,
and TinyUNet is 266k params with a single down/up level - built for 8x8 digits, not 24x24
cursive across 5 classes. Notably the *thinnest* input class (ink 0.131 vs 0.16-0.21) became
the failed output class.

**Why ch=64 specifically.** Measured at 4 threads: **ch=48 -> 1274 ms/step, ch=64 -> 1288,
ch=96 -> 3245.** ch=64 buys 1.8x the parameters of ch=48 for ~1% more time - the step is
memory-bound at this size, so the capacity is nearly free - while ch=96 costs 2.5x.

**Alternatives rejected.**
- *More steps on TinyUNet.* The loss curve was flat for 5,200 steps. Nothing there to gain.
- *Drop to 16x16*, which is what #6 prescribed for trouble. Rejected because resolution was not
  what bound the first run; the same architecture would simply fail faster.
- *ch=96.* 5.4 h per run for capacity this dataset almost certainly does not need.

**Cost accepted.** The step-timing probe (a tight loop over one cached batch) predicted 1288 ms;
the real loop runs at **3.86 s/step**, so a 6000-step run is ~6.4 h rather than ~2 h. The probe
did not model per-step data indexing or memory pressure and should not be trusted for future
estimates without a real-loop check. The run was left at BelowNormal priority regardless, per the
freeze-safety rule in `diffusion_lib.py:23`.

**What would change this.** If the letters are still malformed after this run, capacity is *not*
the binding constraint and the next suspects are the data volume (~900 images/class) and the
per-glyph size normalization from #6 - not a still-larger model.

---

## #6 - The diffusion homework trains on five Armenian letters at 24x24, vendored as one .npz

**Date:** 2026-08-05 · **Status:** active

**Decision.** `ml/ch10_diffusion` gets a homework after all (reversing the "lectures only" call in
`DIFFUSION_CHAPTER_PLAN.md`), built on **five** classes of the Kaggle *Mashtots Dataset v2* -
**Պ Ա Ն Ի Ր**, which spell **ՊԱՆԻՐ** - preprocessed to **24x24** and committed as a single
**1.25 MB `.npz`** (`data/mashtots_panir_24.npz`). Students never touch Kaggle.

**Why.**
- *Five letters, not 78.* The word is the payoff: generate each letter class-conditionally, paste
  them side by side, and the result is visibly wrong because every letter comes from a different
  hand. That failure *is* the lesson about global coherence. ՊԱՆԻՐ also happens to be this course's
  difficulty unit. Five classes give ~4,481 images, against `digits_ddpm.py`'s 1,797.
- *24x24.* Measured, throttled to 4 threads on a loaded machine: **16x16 = 22.5 min/run,
  24x24 = 35.2 min, 32x32 = 109.1 min** for 6,000 steps. 32x32 is 3.1x the time of 24x24 for 1.8x
  the pixels - superlinear, so it is disqualified. 16 -> 24 costs only 1.56x and the glyphs are
  visibly better (`mashtots_letters.html` shows both).
- *Crop to the ink box before resizing.* Not an optimization - required. The glyph fills only
  ~34-40 px of the 64 px frame, so a naive resize applies a 4x reduction to 1-2 px strokes:
  ink fraction **0.133 vs 0.258** at 16x16, peak brightness 134 vs 154. A font-rendered probe missed
  this entirely because font strokes are 5-8x thicker than this handwriting.
- *Vendored `.npz`.* The source is a **competition**, so raw access needs an account, an API token
  and accepting the rules. Every other dataset in this course is a one-liner.

**Alternatives rejected.**
- *All 78 classes.* ~900 images/class either way, but 78-way conditioning on a CPU budget buys
  nothing the word demo needs.
- *64x64 native.* Hours per run. The chapter's own `digits_ddpm.py` docstring already made this call
  for MNIST, though note its "hours" figure is for 60,000 images, not our 4,481.
- *Pretrained Stable Diffusion via `diffusers`.* Teaches none of L27-L30 and is minutes per image on
  an Iris Xe. `diffusers` is not even installed.
- *Font-rendered letters (Sylfaen + augmentation).* Zero download and fully reproducible, but real
  handwriting is the better story and makes the per-writer inconsistency genuine. Kept as a fallback.

**Cost accepted.** Per-glyph cropping normalizes every letter to the same size, discarding the
natural ~3x size spread (19-57 px), so the model cannot generate size variation. Stroke weight and
slant survive, which is enough for the inconsistency lesson.

**What would change this.** If a training run at 24x24 fails to converge in ~35 minutes, drop to
16x16 rather than adding steps. If the letters Ի and Ր turn out to be confusable at 24x24 (they are
near-twins in cursive), swap one and re-pack - `extract_mashtots.py` and `pack_mashtots.py` are
parameterized by a single `LETTERS` list and the raw zip is kept.

---

## #5 - GANs get two decks in the generative thread, not a chapter after diffusion

**Date:** 2026-08-03 · **Status:** active

**Decision.** New chapter `ml/ch8b_gans/` with **L23b** (the adversarial game) and **L23c**
(applications and evaluation), delivered between L23 (VAE) and L24 (attention). The generative
thread now runs **L22 -> L23 -> L23b/L23c -> L27-L31**.

**Why.** GANs were referenced by three delivered decks and taught by none: L19 shows StyleGAN faces,
L23's comparison table calls GANs "unstable", and L28 had to teach mode collapse from scratch so its
"diffusion is just an MSE" argument would land. Verified by grep: "generative adversarial" appeared
in **zero** built decks. Placing the material *before* diffusion converts L28's improvised teaching
into a genuine callback, which has now been done.

**Alternatives rejected.**
- *L32, a chapter after diffusion.* Cleanest numbering, no suffixes. Rejected because the diffusion
  chapter spends five lectures comparing against a model students would not yet have met.
- *Fold into `ch8_autoencoders` as L23b.* Least churn, and the VAE-vs-GAN table already lives there.
  Rejected because the folder name would stop describing its contents.

**Cost accepted.** A `b`/`c` suffix in the lecture numbering, following the existing `L13b`
precedent. Renumbering L24 onward was never considered - it would break every cross-reference in
four chapters.

**What would change this.** If the generative material is ever reorganized into one large chapter,
these two decks and `ch10_diffusion` should merge rather than stay adjacent.

---

## #4 - L30 builds cross-attention itself; the merge-to-4-decks fallback is withdrawn

**Date:** 2026-08-03 · **Status:** active

**Decision.** (a) The diffusion chapter does **not** depend on `ch9_attention`'s unwritten decks:
L30 builds cross-attention in one frame from the Q/K/V material L24 already teaches. (b) The
"merge L27 into L28 to get back to four decks" fallback recorded in #3 is withdrawn.

**Why.** An adversarial review of the chapter plan checked both claims against the repo and both
failed. `L24_attention.tex` contains zero occurrences of "cross-atten", and **L25 and L26 do not
exist** - `ml/ch9_attention/` holds one deck of a planned three. So the original "the only place the
transformer chapter is load-bearing" note pointed at material nobody has written. Building the
concept locally costs one frame and removes the ordering constraint between two chapters entirely.

On (b): the fallback claimed a merged deck would run "~32 frames". The actual arithmetic is
17 + 20 = 37 numbered frames, and once the mandatory `[plain]` section-transition and Outline frames
are counted (`SLIDE_STYLE.md:63`) it is ~56 pages - larger than any deck in the course
(measured: L24 = 53, L17 = 47, L22 = 43). It was not a compression, it was two lectures relabelled.

**Alternatives rejected.**
- *Ship L26 first, then L30.* Correct dependency order, but it blocks a chapter the instructor asked
  for on a backlog item with no date.
- *Keep the merge option "just in case".* Rejected because the number in it was wrong; an escape
  hatch nobody has checked is worse than none.

**What would change this.** If L26 ships before L30 is built, the cross-attention frame becomes a
recap instead of new teaching. If the calendar forces a cut, cut scope (end the chapter at latent
diffusion, drop flow matching and video) rather than merging decks.

---

## #3 - Diffusion chapter scope: full derivation, through latent diffusion and video

**Date:** 2026-08-03 · **Status:** active

**Decision.** `ml/ch10_diffusion/` covers diffusion from the forward process through to video
models, deriving the DDPM loss in full rather than asserting it. Planned as **five decks
(L27-L31)**, not four.

**Why.** The instructor chose "full derivation" and "add latent diffusion + video" when asked. The
five-deck count follows from that pair: the ELBO -> L2 -> epsilon-prediction chain is a deck on its
own, and latent diffusion + flow matching + video is another. Four decks would have meant either
compressing the derivation (contradicting the first choice) or dropping the video material
(contradicting the second). The instructor's stated fallback was "build the full one, and maybe
later make it smaller," so the plan is built at full size with the merge points marked.

**Alternatives rejected.**
- *Intuition-only (no ELBO).* Fastest and most visual, but the loss function arrives unexplained -
  and students have already met the ELBO in L23 (VAE), so the machinery is not new to them.
- *DDPM only, stop at MNIST.* Would leave out conditioning and guidance, i.e. the part that makes
  text-to-image actually work, and would not pay off the L23 "diffusion is today's SOTA" claim.
- *Four decks.* Rejected as dishonest packaging of the chosen scope rather than a real reduction.

**What would change this.** If L27-L31 overruns the calendar, merge L27 (forward process) into L28
(the loss) - the merge point is marked in `ml/ch10_diffusion/DIFFUSION_CHAPTER_PLAN.md`. If students
stall on the ELBO derivation in delivery, demote it to an appendix deck and teach the vector-field
route as the main line.

---

## #2 - Two reference videos, deliberately chosen to be complementary

**Date:** 2026-08-03 · **Status:** active

**Decision.** The chapter is sourced from **two** videos, not one: Welch Labs / 3Blue1Brown
"But how do AI images and videos actually work?" (37:20) for intuition, and Deepia
"Diffusion Models: DDPM" (32:05) for the derivation. Both fetched at 1080p into
`ml/ch10_diffusion/_reference_*/`.

**Why.** Neither covers the chapter alone, and the gap is structural rather than a matter of taste:
Welch Labs never writes down the ELBO, and Deepia never covers conditioning or guidance (he defers
score-based/SDE to a later video). Welch Labs supplies *why* DDPM adds noise during sampling - shown
geometrically on a 2D spiral - which Deepia only gets to algebraically. Deepia's closing frame puts
FFHQ diffusion samples beside VAE samples, which closes the L23 VAE "blurry" cliffhanger with
evidence rather than assertion.

**Alternatives rejected.**
- *Welch Labs alone.* Was the original single-video request; leaves the loss unexplained, which is
  incompatible with decision #3.
- *Lilian Weng's blog / the DDPM paper directly.* Both are better references but neither yields
  slide-ready visuals, and the chapter's figure budget is the binding constraint.

**What would change this.** Deepia's promised score-based/SDE follow-up, if it lands, would be a
better source for the DDIM material than the Welch Labs treatment currently planned for L29.

---

## #1 - Diffusion gets its own chapter; `ch9_genai` narrowed to `ch9_attention`

**Date:** 2026-08-03 · **Status:** active

**Decision.** Created `ml/ch10_diffusion/` as a standalone chapter and renamed
`ml/ch9_genai/` -> `ml/ch9_attention/`. `ch9` keeps attention, transformers, and the LLM-training /
RLHF track; generative models move to `ch10`.

**Why.** Four planning documents (`ch6_cnn/L17`, `ch6_cnn/L19`, `ch6_cnn/CNN_CHAPTER_PLAN.md`,
`ch5_neural_networks/CNN_BLOCK_DESIGN.md`) handed diffusion off to "the GenAI chapter," and the ch9
plan itself said "later GenAI parts (generative models) extend the same chapter." That would have
made ch9 carry attention + transformers + LLM training + diffusion + GANs in one folder. Diffusion's
prerequisite spine is the VAE (L23) and CLIP, not the transformer stack, so it does not depend on
most of ch9. Renaming ch9 at the same time stops the folder name from implying it owns all of GenAI,
which is what caused the ambiguity in the first place.

**Alternatives rejected.**
- *Diffusion inside `ch9_genai`.* What the existing plans literally said. Rejected because the
  folder was already the largest in the course and the two topics share few prerequisites.
- *New `ch10` but leave `ch9_genai` named as-is.* Least churn, but preserves the misleading name -
  the next person adding GAN or RLHF material faces the same ambiguity again.

**Cost accepted.** The rename touched 76 tracked files (`git mv`, history preserved) plus five
files carrying the literal string. Decision #1 in `ml/ch9_attention/ATTENTION_CHAPTER_PLAN.md` is
marked superseded rather than rewritten. `_quarto.yml` did not reference ch9, so the site build is
unaffected.

**What would change this.** If the LLM-training track grows large enough to want its own chapter,
`ch9_attention` should split again rather than absorb it.
