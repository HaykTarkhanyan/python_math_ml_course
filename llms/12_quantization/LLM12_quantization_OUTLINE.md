# LLM-12 - Quantization: an LLM on your laptop - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-12) and the queue (instructor:
"do all queue items"; D4 answered: "just overall tell about inference engines and why they exist
at all" - DECISIONS #74). Session 12. One question: **an 8B model is 16 GB in bf16 and the laptop
has 16 GB - how do you make it fit, and what does it cost?** Ideas first (bytes, formats, scales,
blocks, outliers, error correction, training-aware); inference engines in general, no runtime.

Deck file: `LLM12_quantization.tex`. Figures: `py_src/quant_figs.py` - the measured ladder is
GPT-2's own weights quantized in NumPy, perplexity measured with the cached model on cached
wikitext-2 (no downloads; queue option (a)); the by-hand numbers asserted.

**Decisions:** D4 -> no runtime (#74); the ladder (a); homework: no recommendation in the queue,
skipped (none in the slides; decide at delivery).

**Measured (2026-10-06):** GPT-2 perplexity on 8,192 wikitext tokens - fp32 32.65; int8 one scale
per matrix 37.60; int4 one scale per matrix 7,946; int8 blocks of 32 32.65; int4 blocks of 32 35.92;
int3 / int2 blocks of 32 (see results). Memory-copy bandwidth of this laptop (results).

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| Llama 3 8B: 32 layers, model dimension 4,096, 32 heads, 8 KV heads | Llama 3 paper Table 3 via `sources/llm_training/07_llama3` |
| MXFP4: blocks of 32, E8M0 scale (0.25 bits overhead per weight); NVFP4: blocks of 16, FP8 E4M3 scale + a per-tensor FP32 scale | search results incl. NVIDIA TransformerEngine issue and NVFP4 pretraining paper |
| gpt-oss: MoE weights post-trained in MXFP4; gpt-oss-20b (21B total, 3.6B active) "run within 16GB of memory"; 120b on one 80 GB GPU | Hugging Face model card (openai/gpt-oss-20b) |
| LLM.int8(): outlier feature dimensions isolated into a 16-bit matmul while "more than 99.9% of values are multiplied in 8-bit" | Dettmers et al. 2022 abstract |
| FP32 / FP16 / BF16 / FP8 E4M3 / E5M2 / FP4 E2M1 bit layouts | IEEE 754 and the OCP microscaling spec (standard) |

---

## Frames

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Quantization" / "An LLM on your laptop" | - |
| 1-2 | Where we are | Quantize lit | `roadmap_quantization_*` |
| 3 | Cold open | 8B model, bf16: 16 GB; the laptop: 16 GB. **Predict:** at 4 bits per weight, how much quality is lost? | - |
| 4 | GPT-2 at 4 bits, two ways | The ladder: one scale per matrix -> perplexity 7,946; blocks of 32 -> 35.9 (fp32 32.7). Same bits, different scales. | `quant_ladder.pdf` |
| 5 | Outline | | |
| 6 | [plain] The bill | | |
| 7 | Memory by hand | Weights: params x bytes (GPT-2 498 MB fp32; 8B: 16 GB bf16, 4.5 GB at 4.5 bits). | - |
| 8 | The KV cache | 2 x layers x kv_heads x head_dim x bytes per token: GPT-2 36,864 B (37.7 MB at 1,024); Llama 3 8B 128 KiB (1 GiB at 8K, 16 GiB at 128K); GQA's 8 KV heads cut it 4x (design: LLM-14). | `quant_kv.pdf` |
| 9 | Fewer bytes is faster | Decoding reads every weight once per token: tokens/s <= bandwidth / bytes per token. This laptop's measured copy bandwidth; an 8B model at 16 GB vs 4.5 GB. | - |
| 10 | [plain] Fewer bits | | |
| 11 | Number formats | FP32, FP16, BF16 (range vs precision), FP8, FP4. | `quant_formats.pdf` |
| 12 | Absmax by hand | [0.5, -1.1, 0.3, 2.4] to int8 and int4: scale, integers, back, errors. | - |
| 13 | One outlier | One scale for 8 numbers vs blocks of 4. | `quant_outlier.pdf` |
| 14 | Blocks, and their price | Bits per weight = bits + scale bits / block size (4 + 16/32 = 4.5); MXFP4 / NVFP4. | - |
| 15 | Outliers in real models | LLM.int8(): a few huge activation dimensions; weight-only quantization is the common path. | - |
| 16 | [plain] Smarter than rounding | | |
| 17 | Correct the error | GPTQ (fix the rounding error of one weight with the others, layer by layer), AWQ (protect the channels with large activations). Idea level. | - |
| 18 | Train for it | QAT: fake quantization during training; gpt-oss shipped in MXFP4 (20B in 16 GB); BitNet's ternary weights (log2 3 = 1.58 bits), trained from scratch. | - |
| 19 | [plain] Running it | | |
| 20 | What an inference engine does | Why they exist: load quantized weights, manage the KV cache, batch, stream, serve an API; laptop engines vs server engines. Names as of today, one dated line; no recommendation (#74). | - |
| 21 | Recap, Next (LLM-13), Sources | | |
