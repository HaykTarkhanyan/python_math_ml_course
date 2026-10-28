# Outline — [LLM-17] DeepSeek-V3

**Paper:** DeepSeek-AI, "DeepSeek-V3 Technical Report" (Dec 2024), arXiv:2412.19437.
Source text: `materials_md/papers/23_deepseek_v3_2412.19437.txt` (53 pages — abstract it).

**Title:** `[LLM-17] DeepSeek-V3: A Frontier MoE Trained Cheaply`
**Subtitle:** `671B params, 37B active -- MLA, auxiliary-loss-free MoE, FP8 training, multi-token prediction (DeepSeek-AI, 2024)`

## Core idea (one paragraph)
DeepSeek-V3 is a **671B-parameter MoE (37B active/token)** that matches frontier closed models,
yet was pre-trained on 14.8T tokens for only **~2.79M H800 GPU-hours (~$5.6M)** — extraordinarily
cheap. It gets there through a stack of **efficiency techniques**, not brute scale. **Two are
inherited from DeepSeek-V2 and re-validated here** — **Multi-head Latent Attention (MLA)** (compress
the KV cache into a low-rank latent) and **DeepSeekMoE** (fine-grained experts). **Three are new
in / first-at-this-scale in V3**: (1) **auxiliary-loss-free load balancing** (balance experts with
a per-expert routing *bias* nudged online, no aux-loss quality tax); (2) **FP8 mixed-precision
training** at extreme scale (8-bit training with fine-grained tile/block scaling + FP32
accumulation); (3) **Multi-Token Prediction (MTP)** (predict several future tokens for a denser
signal; also enables speculative decoding). A hardware/algorithm **co-design (DualPipe,
computation-communication overlap)** is a big part of *why* it was cheap. This is the base model
**DeepSeek-R1 (LLM-11)** is built on.

## Chapter fit
Pairs directly with **R1 (11)** (its base), **MoE (13)** (extends the load-balancing story it
teaches), and **QLoRA (12)** (quantization intuition, now for *training* not storage). Reinforces
the **efficiency** theme opened by FlashAttention (15). Abstract the 53-page report to a handful of
ideas, exactly as the Llama3 deck (07) abstracts to "three levers" — do NOT drown in tables.

## Section + frame plan

**Hook — "A frontier model for the price of a house"**
671B MoE, on par with GPT-4o / Claude 3.5, trained for ~$5.6M / 2.79M H800-hours — an order of
magnitude cheaper than you'd expect. And it's the model **R1's reasoning was trained on top of.**
The trick isn't scale, it's a stack of **efficiency techniques** — two carried over from
DeepSeek-V2, three new in V3. TikZ: tiny "brute force vs clever levers" 2-box.

**§1 Setup: total vs active, inherited vs new**
- Recall MoE (LLM-13): total vs active params. V3 = **671B total / 37B active** per token.
- Name the efficiency components up front, **clearly split**: **inherited from V2** (MLA,
  DeepSeekMoE) vs **V3's own** (aux-loss-free balancing, large-scale FP8, MTP), plus the DualPipe
  co-design. Fig: `four_levers.pdf` (labeled boxes, marking inherited vs new).

**§2 MLA — shrink the KV cache** (inherited from DeepSeek-V2, re-validated in V3)
- *The KV-cache problem.* At long context, caching K and V for every past token dominates
  memory (grows with sequence length x layers x heads). Intuition frame.
- *MLA idea.* Project K,V down to a shared **low-rank latent** `c`; cache only `c`, then
  up-project per head at use. Big memory cut, near-lossless. Fig: `mla_diagram.pdf` (KV -> latent
  -> per-head). Keep the math light; the *why* is the point. State plainly it's from V2.
- Connect to FlashAttention (15): FA cut the *attention-matrix* memory; MLA cuts the *KV-cache*
  memory — the two halves of attention's memory bill.

**§3 DeepSeekMoE (from V2) + auxiliary-loss-free balancing (V3's contribution)**
- *Recall the problem (LLM-13):* without balancing, experts collapse; but the aux balancing loss
  fights the main loss and costs quality (a real tension the MoE deck raised).
- *V3's fix (the new part).* Add a **per-expert bias** to the routing affinity **for routing only**
  (the gate value still uses the original affinity); **nudge the bias online** each step (−γ if the
  expert is overloaded, +γ if underloaded) to equalize load — **no gradient, no aux-loss tax.**
  Fig: `aux_free_balancing.pdf` (bias term shifting routing; load evens out — schematic).
- Note fine-grained experts + shared expert(s); a light complementary sequence-wise balance loss.

**§4 FP8 training** (new at this scale in V3)
- *Why 8-bit training is hard.* FP8's tiny dynamic range makes activation/gradient outliers
  overflow — naive FP8 training diverges. (Callback to QLoRA's quantization intuition, LLM-12,
  but here for the *forward+backward*, not frozen storage.)
- *V3's recipe.* **Fine-grained scaling** — tile-wise (1x128) for activations, block-wise (128x128)
  for weights — **FP32 accumulation** (promote to CUDA cores; Tensor-Core FP8 accumulate is only
  ~14-bit), and keeping sensitive components (embedding, output head, MoE gate, norms, attention)
  in BF16/FP32. Result: **~2x compute throughput** and **reduced memory**, at BF16-level quality
  (loss error < 0.25%). Fig: `fp8_map.pdf` (which ops FP8 vs higher precision — schematic).

**§5 Multi-Token Prediction (MTP)** (V3's contribution)
- Standard training predicts only the **next** token. MTP adds **sequential** lightweight modules
  that keep the **full causal chain** to predict the **next few** tokens — a **denser learning
  signal** per position (better sample efficiency), and it **enables speculative decoding** at
  inference. (Distinct from Gloeckle et al.'s parallel independent heads.) Fig: `mtp_diagram.pdf`
  (next-1 vs sequential next-k modules). Predict-first: "does predicting extra tokens help the
  main model?" (yes).

**§6 Does it work? + the R1 connection**
- Benchmarks vs open (DeepSeek-V2.5, Qwen2.5-72B, Llama-3.1-405B) and closed (GPT-4o-0513,
  Claude-3.5-Sonnet). Cost: **full training 2.788M H800-hours / $5.576M**; **pre-training alone
  2.664M** (Table 1); 14.8T tokens. Fig: `benchmarks.pdf` (real), `cost.pdf` (GPU-hours/$ vs
  peers, real).
- Close on: **R1 (LLM-11)** took *this* base model and added reasoning RL. Efficiency here is
  what made that affordable.

**§7 Recap + Next**
- Recap: inherited (MLA, MoE) vs V3-new (aux-loss-free, FP8, MTP) + DualPipe co-design + the cost
  headline + the R1 link.
- Next box: R1 (11) reasoning on top; MoE (13) / QLoRA (12) callbacks; the frontier is now
  "efficient training," not just "bigger."

## Figures (Python/matplotlib)
- `four_levers.pdf` — labeled overview marking inherited (MLA, MoE) vs V3-new (aux-free, FP8, MTP).
- `mla_diagram.pdf` — KV -> low-rank latent -> per-head (schematic).
- `aux_free_balancing.pdf` — per-expert bias equalizing load (schematic, labeled).
- `fp8_map.pdf` — FP8 vs high-precision components (schematic).
- `mtp_diagram.pdf` — next-token vs sequential multi-token modules (schematic).
- `benchmarks.pdf` — REAL, V3 vs peers (verify numbers from paper Table).
- `cost.pdf` — REAL training cost / GPU-hours (2.788M H800 full; 2.664M pre-train; verify).

## Pedagogical notes
- This is the **densest** paper — discipline is the main risk. One idea per component; resist
  adding benchmark tables beyond the two in §6.
- Callbacks are the glue: MoE (13) for §3, QLoRA (12) for §4, FlashAttention (15) for §2, R1 (11)
  throughout. Make them explicit (`[LLM-13]` style refs).
- Optional predict-first in §5 (MTP helping the base model).

## Build notes / cautions
- **Attribution matters (review-flagged):** present MLA & DeepSeekMoE as *inherited from V2,
  re-validated*, and only aux-loss-free balancing / large-scale FP8 / MTP as V3's own. Do not call
  MLA/MoE "V3 innovations."
- **Numbers locked by review:** 671B/37B, 14.8T tokens, full training 2.788M H800 / $5.576M,
  pre-training 2.664M, FP8 loss error < 0.25%. Verify benchmark scores from the paper before
  hardcoding.
- Keep MLA and FP8 math *conceptual* — intuition, not a reproduction of the report.
- Conventions: `\input{../../../preamble}`, local callout macros (copy from `13_moe.tex`),
  `[plain]` transitions, `% Provenance:` block. Compile twice, clean, eyeball for overflow.
