# Outline — [LLM-19] Making Training Fast

**Source:** Andrej Karpathy, "Let's reproduce GPT-2 (124M)" (2024-06-09), **section 2**
(`01:22:18 - 02:14:55`): *"Let's make it fast. GPUs, mixed precision, 1000ms"*.
Reference folder: `misc/dl4nlp/_yt_videos/karpathy_reproduce_gpt2/`
(transcript `transcript.txt`, frames `frames/s2_*.jpg`).
Companion repo: `github.com/karpathy/build-nanogpt`.

**Title:** `[LLM-19] Making Training Fast: 1000 ms -> 93 ms`
**Subtitle:** `Precision, kernel fusion, and the memory wall - five rungs that never touch the model (Karpathy, 2024)`

## Core idea (one paragraph)

Training GPT-2 124M on one A100 starts at **1000 ms/iteration**. Five changes, none of
which alter the model's architecture or its mathematics, bring it to **93 ms** - about
**11x**. Every rung is one or two lines of code. Underneath all five sits a single thesis:
**arithmetic is cheap, moving bytes is expensive.** Tensor cores can multiply far faster
than HBM can feed them, so the wins come from spending fewer bits per number (TF32, BF16),
making fewer round trips between HBM and the chip (kernel fusion via `torch.compile`,
FlashAttention), and letting CUDA's power-of-two block tiles divide the work evenly (vocab
padding). The deck teaches the memory wall through the ladder rather than in the abstract.

## Chapter fit

Deck 19 sits with the chapter's efficiency decks. **15 (FlashAttention)** already derives
the algorithm - this deck treats it as *one rung* and points there for internals rather than
re-deriving. Complements **17 (DeepSeek-V3)**, which does FP8 training at frontier scale, and
**12 (QLoRA)**, which is the same "fewer bits" idea applied to fine-tuning memory. It is the
only deck in the folder sourced from a video rather than a paper.

## Verified numbers (A100 80GB SXM, B=16, T=1024, so 16,384 tokens/iteration)

| Rung | Change | ms | tok/s | cumulative |
|---|---|---|---|---|
| 0 | FP32 baseline | 1000 | 16,384 | 1.00x |
| 1 | TF32 (`set_float32_matmul_precision('high')`) | 333 | 49,201 | 3.00x |
| 2 | BF16 autocast | 300 | 54,613 | 3.33x |
| 3 | `torch.compile` | 130 | 126,031 | 7.69x |
| 4 | FlashAttention (`scaled_dot_product_attention`) | 96 | 170,667 | 10.42x |
| 5 | vocab 50257 -> 50304 | 93 | 176,172 | 10.75x |

Derived tok/s match the video's spoken values (49k, 55k, 125k) exactly. **The transcript's
"163,000 tokens per second" at baseline is an ASR error for 16,300** - do not use it.

Other verified facts:
- A100 80GB SXM: FP64 9.7, FP32 19.5, TF32 156, BF16/FP16 312, INT8 624 TFLOPS. Datasheet
  asterisk numbers are *with sparsity* - not used here.
- Bit layouts: FP32 `1+8+23`, TF32 `1+8+10` (19 used), FP16 `1+5+10`, BF16 `1+8+7`.
- `50257 = 29 x 1733` (odd, no power of two). `50304 = 2^7 x 393`, divisible by 128. Delta 47.
- GPT-2 vocab = 50,000 BPE merges + 256 byte tokens + 1 `<|endoftext|>` = 50,257.
- Online softmax: Milakov & Gimelshein (NVIDIA), 2018 - four years before FlashAttention
  (Dao et al., 2022).
- A100 has 108 active SMs; the full GA100 die has 128. (Video says 120; use 108/128.)

> **Second pass (same session).** The built deck is **48 pages**, not the ~33 planned here.
> Seven frames were added after a review of what section 2 offered that the first cut left
> unused. They are listed in the `% Provenance:` block of the `.tex` and marked **[+]**
> below. The most important: the deck had **no by-hand worked frame** (a SLIDE_STYLE
> requirement) and **no roofline**, so the memory wall was asserted six times and never
> demonstrated once.

## Section + frame plan (~33 frames as first drafted; [+] = added in the second pass)

**Hook — "Same model, same math, 11x faster"**
The ladder table with the rungs shown but the mechanism hidden. Ask: which of these five
changed what the network computes? Answer: none of them. Small TikZ before/after.

**Outline** (`\tableofcontents`)

### §1 Know your hardware
*Transition: "Before you optimize anything, ask what the machine can actually do."*

1. The three questions that start every optimization: what hardware, what does it offer,
   are you using it? (`nvidia-smi`, A100 80GB SXM.)
2. The datasheet: TFLOPS by precision. Fig `a100_tflops.pdf` (**real**). Sparsity asterisk
   explained and set aside.
3. Why INT8 is for inference, not training: uniform spacing vs the roughly-normal spread of
   weights and activations. Fig `int8_vs_float.pdf` (**schematic, labeled**).
4. **The memory wall.** Tensor cores idle waiting on data; 60% utilization is doing very
   well. Lower precision pays twice - faster math *and* less traffic. Key box.
5. Measuring honestly: `torch.cuda.synchronize()`. The CPU only queues work, so without it
   you time the queueing, not the compute. Misconception pre-empt. Baseline 1000 ms.
   Also: the first iteration is always slower (allocation, and later, compilation).

### §2 Rungs 1-2: spend fewer bits
*Transition: "The cheapest speedup is to stop moving 32 bits when 16 will do."*

6. Anatomy of a float: exponent sets **range**, mantissa sets **precision**. Fig
   `float_formats.pdf` (**real bit layouts**, centerpiece - redraw of A100 whitepaper Fig 9).
7. **TF32**: crop mantissa 23 -> 10 bits *inside the tensor-core instruction only*. Inputs,
   outputs and accumulation stay FP32; your tensors never change dtype. One line.
8. **Predict-first:** datasheet promises 8x. What do you get? `\pause` -> **3x**. Because the
   multiply got faster and the traffic did not. First real encounter with the memory wall.
9. **BF16 vs FP16** - the distinction worth the whole frame. Same 8 exponent bits as FP32 =
   same range = **no gradient scaler**. FP16's 5 exponent bits shrink the range, which is the
   entire reason gradient scalers exist. History: FP16 on Volta first, BF16 arrived with
   Ampere and removed the complexity. Fig `bf16_vs_fp16_range.pdf` (**real**, log scale).
10. **Mixed precision in practice:** `torch.autocast` wraps forward + loss **only** - never
    backward, never the optimizer step. What actually changes: logits become BF16, parameters
    stay FP32; matmuls get cast, layernorm/softmax/loss stay FP32. Fig
    `autocast_what_changes.pdf` (**schematic, labeled**). Watch-out box: PyTorch's exact cast
    list is under-documented; do not guess.
11. Result: 333 -> 300 ms. Honest note - precision alone is spent; the bottleneck moved.

### §3 Rung 3: stop making round trips
*Transition: "Arithmetic is cheap. Moving bytes is expensive."*

12. The memory hierarchy properly: on-chip SRAM (~20 MB, ~19 TB/s), HBM (40-80 GB,
    ~1.5-2.0 TB/s), CPU DRAM (>1 TB, 12.8 GB/s). Fig `memory_hierarchy.pdf` (**real**;
    deliberately different from deck 15's - adds the DRAM tier and annotates the round trip).
13. **Full-bleed still:** GA100 die, 128 SMs (`s2_01-56-40.jpg`). Where the SRAM physically is.
14. **Worked example - GELU op by op.** `x**3` streams the tensor HBM -> chip -> HBM.
    `* 0.044715` again. `+ x` again. Count the round trips for one elementwise chain. Fig
    `kernel_fusion.pdf` (**schematic, labeled**): unfused vs fused.
15. **torch.compile**, one line. Two mechanisms: (a) it removes the Python interpreter from
    the forward pass - eager mode dispatches op by op with no lookahead; (b) it sees the whole
    graph, so it fuses elementwise chains and keeps the chunk on-chip. 300 -> 130 ms, 2.3x.
16. The cost: compilation time up front, and awkward interaction with debugging.

### §4 Rungs 4-5: what the compiler cannot find
*Transition: "Two wins torch.compile will not hand you."*

17. **FlashAttention**: a kernel fusion that needs an *algorithmic* rewrite, so no compiler
    discovers it. Never materializes the T x T matrix. Built on the online-softmax trick
    (Milakov & Gimelshein, 2018), four years before Dao et al. (2022). Pointer to deck 15.
18. **Full-bleed still:** FlashAttention Fig 1 (`s2_02-01-10.jpg`).
19. **Predict-first:** it performs **more** FLOPs. Faster or slower? `\pause` -> **27% faster**
    (130 -> 96 ms). "FLOPs don't matter; the memory access pattern does."
20. The change: four lines become `F.scaled_dot_product_attention(q, k, v, is_causal=True)`.
21. **Nice and ugly numbers.** CUDA tiles in powers of two; an ugly remainder spins up
    inefficient boundary kernels. Scan the config: 1024 fine, 768 fine, **50257 bad**, and
    GPT-2 XL's **25 heads** bad.
22. **Predict-first:** pad vocab 50257 -> 50304, which is strictly *more* compute. Faster or
    slower? `\pause` -> **4% faster** (96.5 -> 93 ms), and ~30% on PyTorch <= 2.3.1.
23. Why it is safe: the added `wte` rows are never indexed, and the added logits get driven to
    negative infinity by the optimizer exactly like every other unused token. Where 50257
    comes from: 50,000 merges + 256 bytes + 1 `<|endoftext|>`.

### §5 The ladder, and what it means for you
*Transition: "11x, and the model never changed."*

24. The full ladder. Fig `speedup_ladder.pdf` (**real**: ms per rung + cumulative speedup).
25. Exact vs approximate: `torch.compile`, FlashAttention and padding are **bit-exact in
    intent**; TF32 and BF16 trade precision deliberately. Table, honest framing.
26. **Reality check for this course.** These are A100 numbers. On a Colab T4 (Turing) there is
    **no TF32 and no BF16** - FP16 with a gradient scaler is the option, and `torch.compile`
    plus FlashAttention still apply. What transfers and what does not. Fig
    `hardware_reality.pdf` or a table.
27. The transferable method (key box): measure before optimizing, synchronize before timing,
    and remember the memory wall explains four of the five rungs.
28. **Recap + `Next:` box** -> section 3 of the same video (gradient accumulation, AdamW,
    LR schedule, DDP) as a candidate deck 20; back-links to **15 (FlashAttention)** for the
    algorithm and **17 (DeepSeek-V3)** for FP8 training at scale.

## Figures

Python-generated (`py_src/make_figures.py`, matplotlib, SEED=509, output to `fig/`):

| File | Kind | Content |
|---|---|---|
| `a100_tflops.pdf` | real | TFLOPS by precision, A100 80GB SXM datasheet |
| `float_formats.pdf` | real | FP32 / TF32 / FP16 / BF16 bit layouts (centerpiece) |
| `bf16_vs_fp16_range.pdf` | real | representable magnitude range, log scale |
| `int8_vs_float.pdf` | schematic | uniform INT8 ticks vs float ticks over a Gaussian |
| `memory_hierarchy.pdf` | real | bandwidth vs capacity, SRAM / HBM / DRAM |
| `kernel_fusion.pdf` | schematic | GELU round trips, unfused vs fused |
| `autocast_what_changes.pdf` | schematic | what autocast casts and what stays FP32 |
| `speedup_ladder.pdf` | real | the five rungs, ms and cumulative speedup |
| `hardware_reality.pdf` | real | which rungs are available on T4 / A100 / H100 |

Borrowed stills into `fig/borrowed/` (attribution line on each frame):

| File | From | Use |
|---|---|---|
| `karpathy_a100_datasheet.jpg` | `s2_01-25-20.jpg` | what a real spec sheet looks like |
| `karpathy_ga100_die.jpg` | `s2_01-56-40.jpg` | full-bleed, GA100 128 SMs |
| `karpathy_flash_fig1.jpg` | `s2_02-01-10.jpg` | full-bleed, FlashAttention Fig 1 |

## Pedagogical decisions

- **Three predict-first frames**, all on genuinely counter-intuitive results: TF32's 8x that
  becomes 3x, FlashAttention doing more FLOPs and running faster, and padding the vocab to a
  larger size to go faster. This is the deck's spine.
- **The memory wall is taught through the ladder**, not before it. Students meet it as the
  explanation for a disappointing measurement (rung 1), which is when it lands.
- **Minimal code** per the style guide: the one-line change per rung, plus the
  `synchronize`-before-timing snippet, and nothing more.
- **No re-derivation of FlashAttention** - deck 15 owns that.
- **Frame 26 is the addition to the source material**, not in the video: Karpathy is on an
  8xA100 box, the students are on Colab. Without it the deck is a tour of hardware nobody in
  the room has.
