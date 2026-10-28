# Outline — [LLM-16] RoPE (Rotary Position Embeddings)

**Paper:** Su, Lu, Pan, Murtadha, Wen, Liu, "RoFormer: Enhanced Transformer with Rotary
Position Embedding" (2021), arXiv:2104.09864.
Source text: `materials_md/papers/22_rope_roformer_2104.09864.txt`.

**Title:** `[LLM-16] RoPE: Rotary Position Embeddings`
**Subtitle:** `Encode position by rotating queries and keys -- relative distance falls out of the dot product (Su et al., 2021)`

## Core idea (one paragraph)
Self-attention is **permutation-invariant**, so position must be injected. Absolute position
embeddings (added to the input) don't naturally encode *relative* distance. **RoPE rotates** each
query and key vector by an angle **proportional to its position**. Because a dot product of two
rotated vectors depends only on the **difference** of their rotation angles, `qₘ · kₙ` ends up a
function of the **relative position `m - n`** — so relative-position encoding falls out for free,
inside the attention dot product, touching only Q and K (not V). RoPE also has **long-term decay**
(far tokens attend less) and **sequence-length flexibility** (it's defined at *any* position, not
capped at a trained length L). **IMPORTANT (attribution):** the `θ = 500 000` long-context trick
your Llama3/Qwen3 decks cite is **downstream work** (Llama-3, NTK-aware scaling, YaRN, 2023+),
**NOT** in the RoFormer paper — the paper fixes base θ = 10000 throughout. The deck must present
base-θ rescaling only as an explicitly-labeled forward pointer.

## Chapter fit
Fills the **position-encoding gap** and *explains a symbol the chapter already uses*: Llama3 (07)
cites `RoPE θ=500,000` and Qwen3 (08) cites `RoPE base 1e4 -> 1e6` with no explanation. Also
supports the long-context thread (with FlashAttention 15). dl4nlp teaches attention generally;
per instructor we did not check whether it already covers RoPE specifically — note this in the
provenance block.

## Section + frame plan

**Hook — "Shuffle the words, same answer"**
Self-attention with no position signal is order-blind: permute the tokens and the outputs just
permute with them — "the cat sat" == "sat cat the". Every LLM needs to break this symmetry.
RoPE's answer is unusually elegant: *rotate*, don't *add*. TikZ: tiny "permuted input -> permuted
output" sketch.

**§1 The problem: attention has no sense of order**
- *Permutation invariance*, shown concretely. Fig: `permutation_invariance.pdf` (schematic).
- *Absolute position embeddings.* Sinusoidal / learned vectors **added** to token embeddings.
  They encode an absolute index, not relative distance; and **learned** absolute embeddings have
  no vector for positions > L (the trained length), while **sinusoidal** ones are defined for all
  positions but still extrapolate poorly in practice. (Keep the two variants distinct — don't say
  "absolute embeddings have no vector past L" as if it covers sinusoidal.)
- *What we actually want.* Attention between positions m and n should depend on the **distance
  `m - n`**, and taper at long range. Keybox states this target — RoPE is built to hit it.

**§2 The idea: rotate, don't add (intuition before the algebra)**
- *Intuition frame (analogy):* treat each 2-D slice of a vector as a **clock hand**. Position =
  how far you rotate the hand. Two hands' dot product depends on the **angle between them** =
  difference of rotations = relative position. Fig: `rotation_2d.pdf` (clock-hand rotation).
- *The 2-D case (derivation, boxed).* Rotate q by angle `mθ`, k by `nθ` (a 2x2 rotation matrix).
  Then `R(mθ)q · R(nθ)k = qᵀ R((n-m)θ) k` — depends only on `n - m`. (Paper Eq. 13-16.) Full
  step-by-step.
- *Generalize to d dimensions.* Pair up the d coordinates; rotate pair i at its own frequency
  `θ_i = 10000^{-2i/d}` (verbatim from the paper §3.3): high-frequency pairs capture local, low-
  frequency global — same spirit as sinusoidal frequencies. Fig: `frequencies.pdf`.
- *Worked-numbers frame.* Tiny concrete 2-D example: pick q, k, positions m=2, n=5; rotate;
  show the resulting score depends on m-n. (Instructor likes a by-hand frame.)

**§3 Why it works (properties)**
- *Relative in absolute clothing.* Applied elementwise to Q and K only; V and the rest of the
  block are untouched — cheap, per-layer.
- *Long-term decay.* The multi-frequency sum makes far-apart tokens attend less, **proved in the
  paper** (§3.4.3, Abel transformation; Fig. 2). Fig: `decay.pdf` (score vs relative distance).
- *Sequence-length flexibility vs context extension (attribution-critical).* The RoFormer paper's
  own claim is that RoPE is defined at **any** position, so unlike learned absolute embeddings it
  isn't capped at length L. The **base-θ rescaling** trick (10000 -> 500000) that actually
  *extends* usable context is **later work** (Llama-3, NTK-aware scaling, YaRN, 2023+) and MUST be
  labeled a forward pointer, not a RoFormer result (the paper shows no zero-shot length
  extrapolation). Fig: `theta_scaling.pdf`, captioned "downstream (Llama/NTK/YaRN), NOT RoFormer."
  Watchbox: "this is where Llama3's `θ=500k` comes from — after this paper."

**§4 Adoption (short results section)**
- RoFormer's own benchmark gains are **modest** (e.g. WMT14 BLEU 27.3 -> 27.5; better on 3 of 6
  GLUE tasks). The real story is **adoption**: RoPE became the default position encoding —
  GPT-NeoX, PaLM, LLaMA/Llama-2/3, Qwen, Mistral, DeepSeek. Fig: `adoption.pdf` (curated list,
  clearly labeled as not-from-paper).

**§5 Recap + Next**
- Recap: permutation invariance -> need position; absolute adds don't give relative distance;
  RoPE rotates q,k so the score depends on m-n; proven long-term decay; defined at any position.
  (Base-θ scaling for long context is downstream, not this paper.)
- Next box: this is the position encoding inside Llama3 (07) / Qwen3 (08); long context also
  needs the efficient attention of **FlashAttention (15)**.

## Figures (Python/matplotlib)
- `permutation_invariance.pdf` — schematic (small; could be TikZ).
- `rotation_2d.pdf` — clock-hand / 2-D rotation by position (centerpiece intuition).
- `frequencies.pdf` — per-dimension-pair rotation frequency (like sinusoidal bands).
- `decay.pdf` — attention score vs relative distance, decaying (from paper Fig. 2 shape).
- `theta_scaling.pdf` — usable context vs base θ (schematic; caption: DOWNSTREAM Llama/NTK/YaRN,
  NOT RoFormer).
- `adoption.pdf` — models adopting RoPE (curated list, labeled).

## Pedagogical notes
- Intuition-before-math (clock hands) then the boxed 2-D derivation then the by-hand example.
- Minimal code: none, or one line noting `apply_rotary_emb`.
- Keep the d-dimensional generalization conceptual; the 2-D rotation is the real math.

## Build notes / cautions
- **Attribution is the top priority here:** base-θ rescaling / θ=500k / NTK / YaRN is NOT in this
  paper. Present it only as a forward pointer. The paper's claim is "sequence-length flexibility,"
  which is not the same as demonstrated length extrapolation.
- The frequency convention `θ_i = 10000^{-2i/d}` is verbatim in the paper (§3.3) — safe to state.
- Keep §4 about adoption + the modest real BLEU/GLUE gains; don't overclaim RoFormer benchmarks.
- Conventions: `\input{../../../preamble}`, local callout macros (copy from `13_moe.tex`),
  `[plain]` transitions, `% Provenance:` block. Compile twice, clean, eyeball for overflow.
