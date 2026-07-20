# Outline — [LLM-15] FlashAttention

**Paper:** Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" (NeurIPS 2022), arXiv:2205.14135.
Source text: `materials_md/papers/21_flash_attention_2205.14135.txt`.

**Title:** `[LLM-15] FlashAttention: Fast, Memory-Efficient Exact Attention`
**Subtitle:** `Make attention IO-aware -- tile it, and never write the N x N matrix to HBM (Dao et al., 2022)`

## Core idea (one paragraph)
Standard attention is bottlenecked by **memory traffic**, not FLOPs: it materializes the
full `N x N` attention matrix `S = QKᵀ` and `P = softmax(S)` in slow GPU HBM, reading and
writing it several times. FlashAttention computes the **exact** same attention **without ever
materializing** the `N x N` matrix, by **tiling** Q, K, V into blocks that fit in fast on-chip
SRAM, using an **online (running) softmax** to combine blocks, and **recomputing** the matrix
in the backward pass instead of storing it. Net: **up to ~3x end-to-end** wall-clock (**7.6x on
the attention operation alone**), **up to 20x** less memory (linear in N, not quadratic), exact
result. This is what makes long-context training practical.

## Chapter fit
Fills the chapter's **systems/efficiency gap** — the first deck on *how* attention is trained
fast. Forward-links to long context in Llama3 (07) / Qwen3 (08) and to DeepSeek-V3 (17) which
relies on efficient attention (MLA). No overlap with dl4nlp's attention decks (those teach
*what* attention is; this is the *kernel/systems* view).

## Section + frame plan

**Hook — "Attention got the FLOPs right and the memory wrong"**
Attention is O(N²) in both time and memory. As context grows the N x N matrix dominates —
but profiling shows the bottleneck is **HBM memory bandwidth**, not arithmetic. FlashAttention
makes attention *exact and* much faster by attacking memory, not math. TikZ: small "compute-
bound vs memory-bound" 2-box.

**§1 The problem: attention is memory-bound**
- *GPU memory hierarchy.* SRAM (tiny ~20 MB, ~19 TB/s; 192 KB per SM on A100) vs HBM (large
  40-80 GB, ~1.5-2.0 TB/s). Fast memory is small; big memory is slow. Fig:
  `gpu_memory_hierarchy.pdf` (bandwidth vs size, real A100 numbers).
- *Standard attention writes N x N to HBM.* Algorithm: `S=QKᵀ` (write N x N), `P=softmax(S)`
  (read+write N x N), `O=PV` (read N x N). Several full N x N round-trips to HBM. Note the
  softmax/dropout/mask steps are cheap in FLOPs but **memory-bound** — that's the point. Fig:
  `standard_attention.pdf` (the HBM round trips).
- *Predict-first frame:* "attention is O(N²) FLOPs and O(N²) memory — which is the real
  bottleneck?" Answer: for these sizes it's **memory IO**.

**§2 The idea: tiling + online softmax (intuition before math)**
- *Intuition frame (analogy):* compute the attention output **block by block**, carrying a
  small running summary — like computing a running average/max over a stream without storing
  the whole list. You never hold the whole N x N at once.
- *Online softmax.* The one technical hurdle: softmax needs the max and sum over a whole row,
  but we only see one block at a time. Keep a **running max `m`** and **running normalizer `ℓ`**
  and **rescale** the partial output when a new block shifts the max. Show the rescaling update.
  Fig: `online_softmax.pdf` (running max/sum across blocks).
- *The tiling algorithm (Algorithm 1).* Outer loop over K,V blocks, inner over Q blocks: load
  blocks into SRAM, compute the block's `S_ij`, update `m, ℓ, O_i` with rescaling, write only
  the final `O` (N x d) back to HBM. Never write `S`/`P`. Fig: `tiling_diagram.pdf`.

**§3 The backward pass: recompute, don't store**
- Backprop normally needs `S`/`P` (the N x N) — storing it defeats the purpose. FlashAttention
  **stores only the softmax stats** `(m, ℓ)` (size N) and **recomputes** the blocks of `S`,`P`
  on the fly in the backward pass. Extra FLOPs, far less memory IO → still a net win because
  attention is memory-bound. Keybox: "recomputation is usually a bad trade — here it wins
  because we're bound by memory, not compute."

**§4 Does it work?**
- *Speed + memory (exact numbers).* BERT-large **~15%** faster than the MLPerf 1.1 record
  (17.4 vs 20.0 min); GPT-2 **up to 3x** vs HuggingFace (3.5x small / 3.0x medium), 1.7-1.8x vs
  Megatron; LRA **2.4x**. The **7.6x** figure is the **attention kernel alone**, not end-to-end
  — keep that distinction. Memory **up to 20x**, linear in N. Fig: `speedup_bar.pdf` (real),
  `memory_scaling.pdf` (linear vs quadratic).
- *Longer context, better models.* Linear memory unlocks long sequences: **FlashAttention** is
  first to beat chance on **Path-X (16K, 61.4%)**; **block-sparse FlashAttention** reaches
  **Path-256 (64K, 63.1%)** — dense FA fails Path-256, so keep these two distinct. Frames the
  "long context" thread the report decks (Llama3/Qwen3) live in.

**§5 Recap + Next**
- Recap bullets: memory-bound not compute-bound; tiling + online softmax = never materialize
  N x N; recompute in backward; exact, up to ~3x faster end-to-end, up to 20x less memory.
- Next box (paramgreen): long context everywhere (Llama3 07, Qwen3 08 to 128k/32k); and the
  efficiency mindset continues in **DeepSeek-V3 (17)** — MLA shrinks the *KV cache*, the other
  half of the attention-memory story.

## Figures (Python/matplotlib, sibling `py_src/` -> `fig/`)
- `gpu_memory_hierarchy.pdf` — SRAM (~20 MB / 19 TB/s) vs HBM (40-80 GB / 1.5-2.0 TB/s), labeled.
- `standard_attention.pdf` — schematic of N x N HBM round-trips (small; could be TikZ instead).
- `online_softmax.pdf` — running max/normalizer combining two blocks (schematic).
- `tiling_diagram.pdf` — Q/K/V split into blocks, SRAM tile (schematic centerpiece).
- `speedup_bar.pdf` — REAL: BERT 17.4 vs 20.0 min (15%); GPT-2 small 2.7 vs 9.5 days (3.5x),
  medium 6.9 vs 21.0 days (3.0x); LRA 2.4x. (Label the 7.6x kernel-only speedup separately.)
- `memory_scaling.pdf` — memory vs sequence length, quadratic (standard) vs linear (flash).

## Pedagogical notes
- Predict-first: "math or memory the bottleneck?" (§1).
- No heavy code; at most one line naming `torch.nn.functional.scaled_dot_product_attention`
  (which dispatches to FlashAttention) in the recap as the "you already use this" note.
- Keep the online-softmax rescaling as the one real derivation; everything else conceptual.

## Build notes / cautions
- **Numbers are locked by the review** — use the exact values above; the 7.6x is kernel-only,
  end-to-end is up to ~3x. Keep **Path-X (dense FA)** and **Path-256 (block-sparse FA)** distinct.
- One idea per frame; the online-softmax frame is the densest — give it room, don't crowd.
- Follow chapter conventions: `\input{../../../preamble}`, local `\keybox/\warnbox/\nextbox/
  \watchbox` macros (copy from `13_moe.tex`), `[plain]` section transitions, `% Provenance:`
  block. Compile twice, clean aux, eyeball every page for overflow.
