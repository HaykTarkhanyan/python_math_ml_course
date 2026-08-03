# Decisions

Significant design choices for this repo, newest first. Each entry records what was decided, when,
why, what was rejected, and what would justify revisiting it. Superseded entries stay - the fact
that we changed our mind, and why, is the valuable part.

Deep supporting research lives in the relevant chapter's `_reference_*/` or `research/` folder;
this file holds the choice and a pointer.

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
