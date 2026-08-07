# Chapter plan — Vision-Language Models (L33, L34)

**Status:** APPROVED 2026-08-07, building.

**Source:** written from scratch, from a web sweep on 2026-08-07. No existing deck in this repo
covers it (`grep` confirms: CLIP, ViT and VQ-VAE appear nowhere in `ml/`).

## Instructor decisions (2026-08-07)

1. **Two decks** - L33 "How a model sees", L34 "How a model draws".
2. **Figures only, no training.** No VQ-VAE and no autoregressive generator are trained.
   The AR-vs-diffusion head-to-head is therefore **cut**; the chapter cites published
   comparisons instead of running its own.
3. **Intuition-first**, like `ch11_rl`. Equations are shown and explained, not derived.
   *This is a deliberate deviation from `ml/SLIDE_STYLE.md`, which asks for full step-by-step
   derivations. Recorded here so the next person does not "fix" it.* It bites in exactly two
   places: the contrastive loss and the VQ straight-through estimator.

**Consequences of decision 2**, applied during the build:

- `clip_zeroshot_letters` is **cut**. It needs a ~350 MB CLIP download and a new dependency
  (`open_clip` or `transformers`), which is a dependency choice that was not on the table.
  Open question 5 is therefore moot. Parked in `DEFERRED_TODO.md`.
- `ar_letters` and `ar_vs_diffusion_letters` are **cut** - both needed a trained AR model.
- `vq_quantization` is **kept and is real**, using **k-means on raw 4x4 patches** rather than a
  learned VQ-VAE. k-means is clustering, not network training, and it runs in ~8 s. It is
  labelled as a stand-in on the slide. A learned VQ-VAE encoder sees more context than one
  patch, so this figure is a **lower bound** on achievable quality - the safe direction for
  the claim the slide makes.

## What the VQ measurement actually found (2026-08-07)

Real numbers, over all 4,481 letters, `logs/l34_drawing_figs.log`:

| Codebook K | Reconstruction MSE | Bits per image |
|---|---|---|
| 8 | 0.01819 | 108 |
| 32 | 0.01064 | 180 |
| 128 | 0.00844 | 252 |
| 512 | 0.00654 | 324 |

Raw image = 4,608 bits, so K=512 is **14.2x compression**.

Two things worth teaching, both visible in `fig/vq_quantization.pdf`:

1. **Sharply diminishing returns.** 8 -> 32 nearly halves the error; 128 -> 512 buys only 22%
   for 4x the vocabulary. That is exactly why the field went to lookup-free quantization
   (MAGVIT-2) rather than simply growing K.
2. **Even at K=512 the strokes visibly break up.** Each patch is quantized with no knowledge of
   its neighbours, so stroke continuity across a patch boundary is not preserved. This is the
   same 1-2 px stroke fragility that drove ch10's depth finding, and it is the honest concrete
   answer to "what does discretizing an image cost".

---

## Why this chapter exists

The course can now explain how a transformer works (ch9), how an autoencoder compresses (ch8),
and how diffusion generates (ch10). A student who has finished ch10 can build a model that draws
Armenian letters — but cannot explain how ChatGPT looks at a photo they paste, and cannot explain
how the *same* model that writes text also draws pictures.

That is a real gap, and it is the single most visible thing about modern AI to a non-specialist.
It is also the natural place where four earlier chapters converge:

| Prerequisite | What this chapter uses it for |
|---|---|
| ch6 CNN | patches, receptive fields, why convolution was replaced here |
| ch8 autoencoders | VQ-VAE is an autoencoder with a quantized bottleneck |
| ch9 attention | ViT is a transformer; cross-attention is one of the three connector designs |
| ch10 diffusion | the generation half; the AR-vs-diffusion comparison is the payoff |

## Placement

- New chapter folder `ml/ch12_vlm/`, decks **L33** and **L34** (the L-sequence ends at L32, RL).
- Sidebar: after `ml/ch11_rl/`. **Open question 1** below asks whether it should instead sit
  before `ml/llm_training/`.
- Registering it touches `_quarto.yml`, the one shared file — do that in a single small commit
  at the end.

---

## The through-line: the same letters, drawn two ways

ch10 trained a diffusion model to write **ՊԱՆԻՐ**. This chapter re-tokenizes those *same 4,481
letters* with a VQ-VAE and generates them **autoregressively, one token at a time**, then puts the
two side by side.

That is not a contrived exercise — it is exactly the architectural fork the whole field is arguing
about (Chameleon vs Transfusion), reduced to a dataset the students already know, on a machine
without a GPU. It also reuses `data/mashtots_panir_24.npz` with no new download.

It gives the chapter a measurement of its own, in the house style: **does discretizing the letters
into a codebook destroy them?** For 1-2 px strokes this is a genuinely open question, and it is the
same failure mode as ch10's depth finding. If the codebook blurs the strokes, that *is* the
lesson about why discrete tokenization trades fidelity for LLM-compatibility.

---

## L33 — How a model sees (~30 frames)

### Cold open
1. **A photo, and a correct answer about it.** Paste an image into a chat model, ask something that
   needs real looking (how many people, what does the sign say). It answers. How?
2. **Why nothing so far can do this.** A transformer consumes a sequence of vectors. An image is a
   grid of pixels. There is no sentence to tokenize. The whole chapter is about that bridge.
3. Outline.

### Section 1 — Pixels into tokens
4. `[plain]` transition: *An image is not a sequence.*
5. **Patchify.** Cut the image into fixed squares, flatten each, one linear layer. That is it —
   each patch is now a "word". *Figure: `patchify_grid`.*
6. **The arithmetic that governs everything downstream.** 224×224 with 14×14 patches = **256**
   patches; at 336×336 = **576**. Quoted because it comes back as a cost.
7. **ViT.** Positional embeddings (without them the image is a bag of patches), CLS token,
   then a plain transformer encoder. Nothing new after ch9.
8. **Predict-first:** does a ViT need convolution to know that neighbouring patches are related?
   Reveal: no — it learns it from position embeddings, given enough data.

### Section 2 — Teaching vision and language the same space
9. `[plain]` transition: *Two encoders, one space.*
10. **CLIP.** Two encoders, **400M image-text pairs**, contrastive loss: match the right caption
    against every wrong one in the batch. *Figure: `clip_contrastive` — the N×N similarity matrix.*
11. **Why the diagonal is the whole trick.** The negatives are free; the batch supplies them.
12. **Zero-shot classification falls out.** No classifier head — write the classes as sentences.
13. **Worked measurement: CLIP on our Armenian letters.** Real run, real numbers.
    *Figure: `clip_zeroshot_letters`.* Honest framing: CLIP has barely seen Armenian script, so this
    is expected to be **bad**, and the failure is the point — it shows what "zero-shot" is bounded by.
14. **SigLIP.** Swap softmax for a sigmoid per pair; no batch-wide normalisation, so batches can be
    smaller. Now the default encoder in most open VLMs.

### Section 3 — Bolting eyes onto a language model
15. `[plain]` transition: *Three ways to plug vision into an LLM.*
16. **Design A — a projector (LLaVA).** Two-layer MLP maps 576 visual tokens into the LLM's
    embedding space; they are simply prepended to the text tokens. The LLM is not modified at all.
17. **Design B — a resampler (Flamingo, BLIP-2).** A fixed set of learned queries cross-attends to
    the image and compresses it to a constant token count regardless of resolution. Flamingo's
    gated cross-attention is initialised so **α = 0**, so at the start of training the model is
    exactly the original LLM and vision fades in.
18. **Design C — early fusion (Chameleon and after).** No bridge. One transformer, both modalities
    from the first layer.
19. **The comparison table**, and which won: the field moved projector → early fusion as data grew.

### Section 4 — What it costs, and what breaks
20. `[plain]` transition: *An image is expensive.*
21. **The token budget.** One 336px image = 576 tokens ≈ a page of text — for a thumbnail.
    *Figure: `token_budget`.* High resolution is quadratic in attention, so this is the binding
    constraint on every VLM.
22. **Dynamic resolution (Qwen2-VL).** Stop resizing everything to a square; emit a variable number
    of tokens. Why that fixes documents and charts.
23. **Token pruning and merging**, in one frame — the active research direction.
24. **Where VLMs still fail, and why:** counting, precise spatial relations, reading clocks and
    dense text. The honest frame. The cause is mostly that a patch grid at 576 tokens simply does
    not resolve it, plus contrastive pretraining that never rewarded counting.
25. **Hallucination in VLMs** — describing objects that are not there, and why the language prior
    is strong enough to override the image.
26. Recap + `Next:` box → L34.

## L34 — How a model draws (~30 frames)

### Cold open
27. **The same model wrote the caption and drew the picture.** Why that is architecturally strange:
    ch10's diffusion model works in *continuous* space, an LLM emits *discrete* tokens.
28. Outline.

### Section 5 — Images as discrete tokens
29. `[plain]` transition: *What if a picture were made of words?*
30. **VQ-VAE.** An autoencoder (ch8) with one change: snap each latent vector to the nearest entry
    of a learned codebook. The image becomes a grid of **integers**. VQ-VAE: 128×128 → 32×32 latents.
31. **The codebook is a vocabulary.** Chameleon: a 512×512 image → **1024 tokens** from a codebook
    of **8192**, next to a text vocabulary of 65,536. Same sequence, same softmax.
32. **What quantization costs.** *Figure: `vq_reconstruction` — our letters through codebooks of
    increasing size.* **This is the chapter's own measurement** — see the through-line above.
33. **Codebook collapse**, and why codebook size is not free (ViT-VQGAN 1024 → 8192;
    MAGVIT-2 pushes to 2^18 with lookup-free quantization).

### Section 6 — Generating an image left to right
34. `[plain]` transition: *Next-token prediction, on a picture.*
35. **Raster order.** Flatten the token grid top-to-bottom, left-to-right, and predict the next one.
    *Figure: `ar_raster_order`.* Exactly GPT, with a different vocabulary.
36. **Worked: our letters, autoregressively.** *Figure: `ar_letters`.* Trained on the ch10 dataset.
37. **The head-to-head.** *Figure: `ar_vs_diffusion_letters` — same letters, both methods.*
    Honest comparison on sample quality, sample count, and wall-clock.
38. **Why AR for images was long thought a dead end** — 1024 sequential forward passes vs a
    50-step diffusion sampler — and what changed.

### Section 7 — One model for both
39. `[plain]` transition: *Understanding and generation in one network.*
40. **Paradigm 1 — fully discrete AR (Chameleon).** One vocabulary, one loss. Elegant; pays in
    fidelity at the tokenizer.
41. **Paradigm 2 — hybrid (Transfusion).** Autoregressive on text, diffusion on continuous image
    latents, **one shared transformer**, a summed LM + DDPM loss. Avoids quantization entirely.
42. **Paradigm 3 — decoupled encoders (Janus).** SigLIP to understand, VQ to generate, shared LLM
    trunk — because the two tasks want different representations.
43. **GPT-4o's native image generation.** What is public vs inferred: continuous VAE patch latents
    interleaved with text, `<BOI>`/`<EOI>` markers, hybrid AR + DDPM loss. **Flagged on the slide
    as reconstruction from the Transfusion line of work, not an official architecture.**
44. **Why this made text-in-images work**, and why editing an image by chatting became possible.

### Section 8 — Limits
45. **The cost.** Tokens per image, and what that means for video.
46. **What is still unsolved**, in one honest frame.
47. Recap + `Next:` box → `ml/llm_training/`.

---

## Figure budget (all Python, `py_src/` → `fig/`)

| Figure | Deck | What it shows | Cost |
|---|---|---|---|
| `patchify_grid` | L33/5 | image → patch grid → bag of vectors | free |
| `clip_contrastive` | L33/10 | N×N similarity matrix, diagonal positive | free |
| `clip_zeroshot_letters` | L33/13 | **real CLIP run** on Mashtots letters | needs CLIP download (~350 MB), CPU inference |
| `token_budget` | L33/21 | bar chart, image vs text token cost | free |
| `vq_reconstruction` | L34/32 | **real VQ-VAE** on our letters, several codebook sizes | small training |
| `ar_raster_order` | L34/35 | scan order over the token grid | free |
| `ar_letters` | L34/36 | **real AR samples** of ՊԱՆԻՐ | small training |
| `ar_vs_diffusion_letters` | L34/37 | side by side with ch10's diffusion output | free (reuses ch10 weights) |

Bar charts get `ax.bar_label`; 3+ colours use the Armenian flag palette, per `CLAUDE.md`.

**Compute:** the VQ-VAE and the AR transformer are both tiny (24×24 inputs, ~6×6 token grids).
Expected minutes on CPU, and Colab/T4 is available if not. CLIP is inference-only. Nothing here
approaches the ch10 diffusion runs. Every training script follows the `TAG` convention in
`CONVENTIONS.md`.

---

## Open questions for the instructor

1. **Placement** — after `ch11_rl`, or immediately before `ml/llm_training/`? The RL deck was put
   before `llm_training` because it feeds it. VLM does not feed it as directly.
2. **One deck or two?** The outline is ~47 frames, which is two decks at ch10 granularity
   (L27-L31 are 20-30 frames each) or one long one at ch11 granularity (L32 is 44 pages).
3. **Derivation depth.** `ml/SLIDE_STYLE.md` asks for full derivations; ch11_rl deliberately went
   intuition-first. Contrastive loss and the straight-through estimator in VQ-VAE are the two
   places where this choice actually shows.
4. **Is the AR-vs-diffusion project worth the build?** It is the strongest idea here but the most
   work. The chapter stands without it, using figures only.
5. **CLIP on Armenian letters** — I expect this to fail. Keep it as an honest negative result, or
   cut it and use a normal photo?

---

## Sources swept (2026-08-07)

- [Unified Multimodal Understanding and Generation Models: Advances, Challenges, Opportunities](https://arxiv.org/abs/2505.02567) — the taxonomy this chapter's Section 7 follows
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/pdf/2405.09818) — discrete-token unified model; the 1024-token / 8192-codebook numbers
- [Autoregressive Models in Vision: A Survey](https://arxiv.org/pdf/2411.05902) — VQ tokenizer lineage, codebook sizes
- [Aman's AI Journal — GPT-4o Native Image Generation](https://aman.ai/primers/ai/gpt4o-native-image-generation/) — the reconstruction of GPT-4o's design; explicitly labelled inference
- [Aman's AI Journal — Vision Language Models](https://aman.ai/primers/ai/VLM/) — connector taxonomy
- [Vision Encoders in VLMs: A Survey (Jina AI)](https://jina.ai/vision-encoder-survey.pdf) — encoder landscape
- [Design choices for Vision Language Models in 2024 (HuggingFace)](https://huggingface.co/blog/gigant/vlm-design) — Q-Former vs Perceiver Resampler
- [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://static.hliu.cc/files/llava/improved_llava.pdf) — 576 tokens, MLP projector

**To verify at build time** (cited from secondary sources so far, not the primary papers):
BLIP-2's exact query count (32), Parti's token count, and MAGVIT-2's 2^18 codebook.
