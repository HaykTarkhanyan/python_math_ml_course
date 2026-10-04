# LLM chapter - plan (draft for approval)

Drafted 2026-10-02. **Not approved yet** - the open decisions in section 1 come first.
Follows the house new-chapter workflow (interview -> outline -> approval -> build), like
`14_llms/ATTENTION_CHAPTER_PLAN.md` and `ch17_rag/RAG_CHAPTER_PLAN.md`.

This file sits at the `ml/` root because the chapter folder does not exist yet. It moves into the
chapter folder when that is created (decision D7).

**Scope.** Everything a student needs between "you know what the T in GPT is" (end of L26) and the
already-built application chapters (RAG, agents). Sources are the instructor's 18 `misc/dl4nlp/`
decks, the 19 `ml/llm_training/slides/` paper decks, and recent material verified on 2026-10-02
(section 9).

---

## 1. Open decisions

| # | Decision | Recommendation | What would flip it |
|---|---|---|---|
| D1 | Sessions for the block (not counting RAG/agents) | **9** (LLM-1 to LLM-9); LLM-10 if time allows | A hard course end date. See section 8 for the 6-session cut |
| D2 | Where GANs go | **Resolved 2026-10-03:** AE -> VAE -> GAN -> diffusion, after this block (DECISIONS #63) | - |
| D3 | How deep alignment goes here | **Idea level in LLM-5**; the math stays in L32f/L32g, which depend on PPO from L32d | Moving the whole RL chapter right after this block (+7 lectures before RAG) |
| D4 | Local runtime for the hands-on parts | **Ollama** for the first contact (one installer, Windows-friendly); **llama.cpp tools** for the quantization measurements (`llama-quantize`, `llama-perplexity`, `llama-bench` have no Ollama equivalent) | Most students on Macs -> LM Studio or MLX deserve the demo slot |
| D5 | Fine-tuning library for the practical | Two real options: **PEFT + TRL** (the standard HF stack, every step visible) or **Unsloth** (fits a free T4 more easily, `save_pretrained_gguf` exports straight to GGUF). Lean: Unsloth for the practical, PEFT shown on a slide | If the goal is "see the mechanics" over "ship a model", PEFT + TRL |
| D6 | Base model for practicals | **Decide by measurement**, not by name: tokens per Armenian word and Armenian perplexity for 3-4 small open models (e.g. Gemma 4 E2B, a small Qwen3.5) | - |
| D7 | Chapter folder and number | New folder, per DECISIONS #1 ("split `ch9_attention` rather than absorb"). Number follows the new order (RNN 13, attention 14 -> LLMs 15?) | Doing the full `chNN_` -> `NN_` renumbering pass first |

---

## 2. Where the block sits

**Course order (instructor decision 2026-10-03, DECISIONS #63, superseding #62's order of
2026-10-02):**
CNN (L16-L19) -> **RNN** (L20, L21) -> **Attention** (L24-L26) -> **this block** -> RAG (`ch17`,
L41-L43) -> Agents (`ch18`, L44) -> **Autoencoders** (L22, L23) -> **GANs** (L23b, L23c) ->
**Diffusion** (L27-L31) -> the rest (RL, VLM, ...).

Two threads, each unbroken: text (RNN -> attention -> LLMs) and generative (AE -> VAE -> GAN ->
diffusion).

**Nothing in or right after this block needs autoencoders** (checked 2026-10-03, grep for
autoencod / VAE / L22 / L23 over the decks): `14_llms` 0, `ch17_rag` 0, `ch18_agents` 0. The
LLM source decks say "denoising" only about T5/BART span corruption
(`misc/dl4nlp/06_early_notable_models.tex:588, 640`) - one sentence, no chapter needed.

**Chapters that DO need autoencoders** - schedule them after L22/L23:

- Mech interp (`ch19`): `xx_features.tex:261` "Back to the autoencoder chapter" and sparse
  autoencoders throughout; also `xx_opening_the_box.tex:908`, `xx_personas_and_diffing.tex:312`.
  It is LLM-adjacent, but it **cannot** follow this block directly.
- VLM (`ch12`): `L34_vlm_drawing.tex:84-86` builds VQ-VAE from "the autoencoder of ch8".
- JEPA (`ch16`): `L39_jepa_objective.tex:76, 148` compares against the autoencoders.
- Diffusion (`ch10`): latent diffusion runs inside the VAE.

**Edits this order needs:**

- `ch8_autoencoders/L23_vae.tex:473` - the closing "Next" box points at attention via L21; repoint
  it to GANs, then diffusion.
- `ch8b_gans/L23c_gan_applications.tex:427` - drop "Before it, the attention chapter (L24) picks up
  the other thread".
- `_quarto.yml` order, qmd title numbers, `ml/00_plan.md`.
- Chapter numbers written inside decks ("chapter 8" in `xx_features.tex:263`, "ch8" in `L34`,
  `L39`) - part of the renumbering pass (D7).
- **No edit needed:** L22's reference to the RNN (RNN now comes first), `HW1_sae_rnn` stays the
  autoencoder homework, `L23c:147`'s king-queen callback to L21, L21 -> L24.
- Small gain: L22's masked-modeling line (`:83`) and its "LLM interpretability" teaser (`:85`)
  turn from forward references into callbacks to this block.

---

## 3. What students already have when the block starts

Do not re-teach these; call back to them.

| Already taught | Where |
|---|---|
| One-hot encoding and why its geometry is meaningless (no tokenization, no embeddings - DECISIONS #64) | L20 |
| LM factorization, the generation loop (sample, feed back, repeat - no temperature), a GRU inventing surnames | L21 |
| The seq2seq bottleneck, and the alignment grid: every decoder step gets the same C, each needs a different part of the input | L21 |
| Bengio 2003 neural LM (the surname inventor practical) | ch11, `47_name_inventor_solution.ipynb` |
| Embeddings as directions, king - man + woman | L24 |
| Self-attention, multi-head, positional encoding, the block, causal mask, residual stream | L24, L25 |
| Encoder / decoder / enc-dec, CLM vs MLM, logits -> softmax -> sample, "autocomplete is not a chatbot", the O(n^2) bill and the 16,384-token FLOP crossover | L26 |
| Log-loss, temperature scaling, calibration | [11], [14] |

**Not yet taught when this block runs:**

- **Reinforcement learning.** The RL chapter (ch11, L32-L32g, which owns RLHF/DPO math, GRPO and
  R1) comes *after* this block. That is why D3 matters.
- **Autoencoders** (DECISIONS #63). No lecture may lean on them. Where an autoencoder callback
  would be natural, state the idea directly or call back to PCA (ch10, taught): masked LM and span
  corruption are "corrupt the input, reconstruct it" (LLM-3); LoRA's update is low rank (LLM-9);
  MLA squeezes the KV cache through a learned low-rank bottleneck (LLM-10).

---

## 4. Design principles

1. **Outside in.** What goes in (tokens) -> what comes out (decoding) -> where the weights come
   from (pretraining, scale) -> how a base model becomes an assistant -> how you use it -> how you
   know it is wrong -> how you run and adapt it yourself -> what changed since 2017.
2. **One home per topic.** The repo holds four overlapping sources; today RLHF/DPO/GRPO each appear
   in 4 decks, FlashAttention in 3-4, speculative decoding and the emergence "mirage" debate in 2.
   Section 7 assigns every source deck a destination.
3. **Port and adapt, not copy.** The dl4nlp decks are survey decks (definitions + "further
   reading") on their own `misc/dl4nlp/preamble.tex`. Each lecture gets the `ml/SLIDE_STYLE.md`
   treatment: cold open, transition slides, predict-first, Python figures, measured where cheap.
   Same mechanism as L24-L26 were built from `02_transformers.tex`.
4. **Recent material is perishable.** Every 2025-26 fact carries its verification date and source
   (section 9). The landscape frames in LLM-10 get re-verified right before delivery.
5. **Armenian thread.** Tokens per Armenian word, Armenian perplexity, an Armenian fine-tune. It is
   the course's local hook and it measures something real (non-English text is where tokenizers,
   quantization and small models break first - a hypothesis to measure, not to assert).

---

## 5. The lectures

Frame counts are for the source decks as they exist on 2026-10-02.

### Part A - How an LLM is built

#### LLM-1 Tokenization: from text to token IDs

L21 showed why subwords; now build the tokenizer every LLM actually uses.

- Unicode code points -> UTF-8 bytes; why an Armenian letter is 2 bytes.
- BPE: the idea, a worked example, encoding new text, byte-level BPE (GPT-2 onward),
  pre-tokenization regex, decoding IDs back - including the invalid-UTF-8-mid-character trap
  (`_learnings/2026-08-21-0140_bpe-merge-halves-are-invalid-utf8-alone.md` is a real instance).
- WordPiece and Unigram/SentencePiece, one frame each.
- Pitfalls: numbers split arbitrarily, spelling ("count the r's"), SolidGoldMagikarp and glitch
  tokens - plus our own `ml/claude_projects/armenian_glitch_token_hunt/`.
- Special tokens; the vocabulary-size trade-off; vocabularies grew from GPT-2's 50,257 to 200k+
  (`o200k`) - check per model at build time.
- **Source:** dl4nlp `03_tokenization.tex` (38 frames), now taught **from scratch** here: L21's
  three tokenization frames (trilemma, subwords on a real tokenizer, the Armenian tax) moved to
  this lecture on 2026-10-03 (DECISIONS #64). Reuse their measured figures
  (`ml/13_rnns/py_src/tokenizer_demo.py` -> `ml/13_rnns/fig/tokenizer_panel1-3.pdf`, cl100k counts)
  and frame text (`git show 6013bcc:ml/ch7_rnn/L21_road_to_attention.tex`). Chat tokens move to
  LLM-5. Optional 1 frame from llm_training 05 (BPE-dropout).
- **Hands-on:** train byte-level BPE on Armenian text (~50 lines), compare tokens per word against
  `o200k` and one open-model tokenizer.

#### LLM-2 Decoding: from logits to text (+ your first local model)

L26 ended with a distribution over the vocabulary. Every word a chatbot writes is a choice from it.

- The loop and its stop conditions (EOS, max tokens, stop sequences).
- Greedy and the repetition trap; log-probs; beam search in 2-3 frames (length normalization, the
  probability trap, why chat models do not use it).
- Temperature - first taught here, L21 no longer covers it (callback: the same division by T as
  temperature scaling in [14]), sampling, top-k, top-p, min-p; combining the knobs; typical API
  parameters; repetition penalties.
- Constrained / structured decoding: JSON schema, grammars (llama.cpp GBNF, Ollama structured
  outputs).
- **Source:** dl4nlp `04_decoding_strategies.tex` (33 frames -> ~25). Speculative decoding (3
  frames) moves to LLM-10. Contrastive search: cut to one frame or drop.
- **Hands-on:** install the local runtime chosen in D4 (recommended: Ollama), run a ~1B model on
  the laptop, sweep temperature / top-p / min-p on one prompt. First contact with the local stack;
  the theory is LLM-8.

#### LLM-3 Pretraining: from the internet to a base model

"Predict the next token" on trillions of tokens. What comes out is not an assistant - it is a
document simulator.

- The lineage of "predict the missing word": n-gram counts (1-2 frames) -> Bengio 2003 (callback:
  the surname inventor *is* this model) -> word2vec (2 frames; taught nowhere in `ml/` today) ->
  ELMo (1 frame) -> CLM vs MLM (L26 recap) + T5 span corruption.
- **Embeddings live here and in L24, not in the RNN chapter** (DECISIONS #64): L24 gives the
  three-frame primer (a token is a vector, directions carry meaning); this lecture shows how they
  are trained (word2vec). L21's old embeddings frame used a hand-placed 2D map
  (`ml/13_rnns/py_src/embedding_2d.py`); replace it with real vectors when built.
- The data: crawl -> filtering funnel (dedup, language ID, quality classifiers, PII) - FineWeb;
  mixtures with code and math; synthetic data. Llama 3: ~15.6T tokens vs 1.8T for Llama 2 (from
  llm_training 07).
- Perplexity = exp(cross-entropy) (callback: log-loss [11]); worked example; caveat: only
  comparable under the same tokenizer (callback LLM-1).
- GPT-1 -> GPT-2 -> GPT-3 / BERT / T5 as a timeline of *lessons*: fine-tune -> zero-shot ->
  in-context learning.
- Base-model demos (both are parked instructor asks in `DEFERRED_TODO.md`): it continues a famous
  Wikipedia opening **verbatim** (memorization); asked a question, it writes more questions.
- **Sources:** dl4nlp 07 sec "Pre-training objectives" (7 frames), 06 (22 -> ~6), 05 sec
  "Perplexity" (4), 01 sec "Word embeddings" (5 -> 3); Karpathy "Deep Dive into LLMs" notes
  (`misc/dl4nlp/_yt_videos/karpathy_deep_dive_llms/`); llm_training 07 (Llama 3 data).
- **Hands-on:** perplexity of a small base model on Armenian vs English, and on a memorized vs a
  novel passage.

#### LLM-4 Scaling laws and emergence

- Power laws (Kaplan); C ~ 6ND; the training-memory bill (~16 bytes/param with Adam: weights,
  gradients, two moments, before activations - this number comes back in LLM-9).
- Chinchilla (~20 tokens/param); over-training for cheap inference (small models on far more
  tokens); the three eras of compute allocation.
- Emergent abilities, the mirage argument (the metric makes the jump), where the debate stands;
  grokking (`misc/grokking/`, Welch Labs material already extracted).
- Teaser: test-time compute as the second scaling axis -> L32g.
- **Sources:** dl4nlp 13 (18) + 17 (18), deduplicated to ~20; llm_training 09.
- **First merge candidate** into LLM-3 if sessions are short.

#### LLM-5 Post-training: from base model to assistant

- Why a base model is not a chatbot (callback: the LLM-3 demo).
- SFT on conversations; chat templates and special tokens (ChatML; the cost of adding tokens - from
  dl4nlp 03); "the assistant is a statistical imitation of a human labeler" (Karpathy).
- Instruction tuning at scale (FLAN); quality over quantity (LIMA, 1,000 examples).
- Preference tuning **at idea level** (D3): comparisons -> reward model (Bradley-Terry in one line),
  the RLHF pipeline as a picture, reward hacking, DPO as "skip the RL". Full math: L32f.
- Verifiable rewards and "thinking" models with a budget dial (Qwen3) as a teaser -> L32g.
- Distillation: SFT on a bigger model's outputs (the R1 distills).
- **Sources:** dl4nlp 07 sec 2-4 (~15 frames); llm_training 18 (FLAN), 04 (LIMA), 10
  (InstructGPT), 08 (Qwen3); distillation frames from dl4nlp 11.
- LoRA/QLoRA are **not** here any more - they get their own lecture (LLM-9).

### Part B - Using it, and knowing when it is wrong

#### LLM-6 Prompting and in-context learning

- Zero-shot, few-shot, and how fragile few-shot is (example order, format, label balance).
- Chain of thought, zero-shot CoT, why it works ("models need tokens to think"); self-consistency;
  reasoning models do this internally, which changes the advice.
- System prompts; structured output (callback: constrained decoding, LLM-2); context engineering -
  what goes into the window and in what order.
- Prompt injection, direct and indirect (indirect injection sets up the agents chapter).
- **Sources:** dl4nlp 08 (22 -> ~17; Tree of Thoughts cut, ReAct -> L44); llm_training 14 (CoT
  half).
- **Hands-on:** few-shot fragility on the local model from LLM-2 - permute example order and
  labels, measure the accuracy swing.

#### LLM-7 Evaluation and hallucination

- Why evaluation is hard; intrinsic vs extrinsic; BLEU / ROUGE in 2-3 frames and their limits;
  BERTScore.
- LLM-as-judge (pointwise vs pairwise; position, verbosity and self-preference biases).
- Benchmarks (knowledge, reasoning, code, agents), contamination, leaderboards and Goodhart.
- Hallucination: taxonomy (factuality vs faithfulness), why it happens - including Kalai et al.
  2025: training and evaluation reward guessing over "I don't know" (arXiv 2509.04664); detection
  (SelfCheckGPT, semantic entropy); mitigation overview.
- **Cliffhanger:** give the model the documents -> RAG (L41).
- **Sources:** dl4nlp 05 (27 -> ~16; perplexity -> LLM-3, MAP/nDCG -> L43 owns them) and 09
  (26 -> ~14; RAG mitigation -> ch17, duplicate self-consistency dropped).

**Then the built chapters continue the story:** RAG (`ch17`, L41-L43, 3 lectures, far deeper than
dl4nlp 12) and Agents (`ch18`, L44, replaces dl4nlp 14).

### Part C - Run it and adapt it yourself

#### LLM-8 Quantization: an LLM on your laptop

An 8B model is 16 GB in BF16. The laptop has 16 GB in total. Make it ~5 GB and lose almost nothing.

- **Memory arithmetic:** params x bytes; KV-cache bytes per token (2 x layers x kv_heads x
  head_dim x bytes) - context costs memory too, and the KV cache can be quantized as well (flag
  names: check llama.cpp docs at build). Decoding is memory-bandwidth bound, so
  tokens/s <= bandwidth / bytes read per token (dense model, batch 1; for MoE only the active
  experts count): fewer bytes is also *faster*.
- **Number formats:** FP32 / FP16 / BF16 (range vs precision, from llm_training 19); FP8 E4M3/E5M2
  (DeepSeek-V3 trains in it, llm_training 17); INT8 / INT4; FP4 - MXFP4 (OCP standard, blocks of
  32, shared E8M0 scale) vs NVFP4 (blocks of 16, FP8 scale + FP32 tensor scale).
- **The math, by hand:** absmax and zero-point (affine) quantization; rounding error; per-tensor vs
  per-channel vs block-wise, and why blocks win.
- **Outliers:** a few huge activation dimensions (LLM.int8) - why weight-only quantization is the
  common path.
- **Post-training quantization:** round-to-nearest -> GPTQ (second-order error correction, layer by
  layer) -> AWQ (protect channels with large activations). llama.cpp's importance matrix
  (`llama-imatrix`) is the same idea for GGUF.
- **Quantization-aware training:** train with fake quantization so the model learns to live with
  it. Real examples: Gemma 3 QAT checkpoints (2025, shipped as Q4_0 GGUF for llama.cpp/Ollama);
  gpt-oss ships its MoE weights in MXFP4, which is why the 20B model fits in 16 GB.
- **Extreme end:** 1.58-bit ternary weights (BitNet), trained from scratch - one frame.
- **GGUF decoded:** one file with weights + tokenizer + chat template + metadata; the k-quant
  names (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K, IQ-quants) and what K/M/S mean; bits per weight
  for each - take from the llama.cpp quantize docs at build time.
- **Measured on our laptop** (the lecture's own frame): one small model at F16 / Q8_0 / Q4_K_M /
  Q3_K_M / Q2_K -> file size, peak RAM, tokens/s (`llama-bench`), perplexity on Armenian and
  English text (`llama-perplexity`). Needs a compute OK before running (machine limits).
- **The local stack (as of 2026-10-02)** - see the table right after this list.
- **Gotchas (verified):** Ollama does not quantize GGUF on import - quantize first with
  `llama-quantize`. FP4 vs Q4_K_M quality is unsettled as of spring 2026.
- **Sources:** dl4nlp 11 (memory arithmetic + the ~7 quantization frames), llm_training 12 (block
  quantization, NF4), 19 (float anatomy), 17 (FP8).
- **Homework:** the quantization ladder on your own laptop - table + one plot.

The local stack, as of 2026-10-02 (section 9 has the sources):

| Tool | What it is | When to reach for it |
|---|---|---|
| llama.cpp | C/C++ engine, GGUF, CPU / CUDA / Metal / Vulkan / SYCL; `llama cli`, `llama serve` (OpenAI-compatible); standalone `llama-quantize`, `llama-perplexity`, `llama-bench` | Full control, measuring, odd hardware |
| Ollama | One-command runner + model library; Modelfile; OpenAI- and Anthropic-compatible APIs; MLX backend on Apple Silicon (2026); cloud models | Easiest start; apps that need a local API |
| LM Studio | GUI over the same GGUF models (features: verify at build) | Non-programmers |
| MLX | Apple's array framework, native on M-series | Macs |
| vLLM / SGLang | GPU serving engines, built for throughput across many users | Serving, not laptops -> LLM-10 |
| HF transformers | Earlier GGUF loading dequantized to float; since Sep 2026 (main branch, Metal first) it runs the quantized llama.cpp kernels | Staying in Python |

#### LLM-9 Fine-tuning on a budget: LoRA and QLoRA

Full fine-tuning a 7B model needs ~112 GB (16 bytes/param, LLM-4). LoRA trains under 1% of the
parameters; QLoRA keeps the frozen base in 4 bits. Result: a free Colab T4.

- **Should you fine-tune at all?** The ladder: prompt -> RAG -> fine-tune (callback: L43 "when not
  to use RAG"). Fine-tuning is good at format, style, narrow tasks; new facts are better served by
  RAG. Continued pretraining as the third option, e.g. for a language (llm_training 06,
  Don't Stop Pretraining).
- The full fine-tuning bill: weights + gradients + Adam m and v.
- PEFT family in one frame (adapters, prefix tuning; LoRA won).
- **LoRA:** dW = BA with rank r; B = 0 at init so training starts exactly at the base model;
  alpha/r scaling; which matrices (QLoRA's finding: adapters on *all* linear layers are needed to
  match full fine-tuning - llm_training 12; library defaults differ, check at build); the
  parameter count by hand; merge after training -> zero extra latency; swap adapters per task. Why low rank works
  (intrinsic dimension, from llm_training 03).
- **QLoRA:** NF4 (quantiles of a normal - callback: block quantization in LLM-8), double
  quantization, paged optimizers; store in NF4, compute in BF16; matches 16-bit (Guanaco).
- Variants in one frame (DoRA and friends).
- **Recipe:** rank, targets, learning rate, epochs - take current defaults from the library docs at
  build time; data quality over quantity (LIMA callback); the chat template must match (LLM-5);
  evaluate before and after (LLM-7); catastrophic forgetting.
- **From adapter to laptop:** merge -> convert to GGUF -> `llama-quantize` -> Ollama
  (`FROM ./model.gguf`, `ollama create`). Unsloth does the first three in one call:
  `model.save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")`.
- **Gotcha (verified 2026-10-02 on Ollama `main`):** Ollama **no longer supports LoRA adapters** -
  `server/create.go:40` defines `errAdaptersUnsupported`, and `:903` rejects adapter-type GGUFs
  too. Merge the adapter first. Older tutorials using `ADAPTER` in a Modelfile are out of date.
- **Sources:** llm_training 03 (LoRA, 19 frames) and 12 (QLoRA, 21 frames); dl4nlp 07 sec "PEFT"
  (4 frames); dl4nlp 11's QLoRA frame is a duplicate - dropped.

### Part D - What changed since 2017

#### LLM-10 Anatomy of a 2026 LLM (likely 2 sessions)

Put L25's 2017 block next to a 2026 model card. Almost every change since exists to save memory or
compute at inference time.

- **KV cache:** why decoding is memory-bound; prefill vs decode; KV memory arithmetic -> MQA ->
  GQA -> MLA (DeepSeek; also GLM-5, Kimi K2.5, Mistral Small 4).
- **Attention cost:** sliding-window + global layers (gpt-oss alternates full attention with a
  128-token window, plus learned attention sinks); hybrid linear attention (Gated DeltaNet + full
  attention in Qwen3-Next / Qwen3.5); mostly-Mamba-2 hybrids (Nemotron 3 Nano) - pays off L21's
  "recurrence comeback" teaser (since 2026-10-03 L21 keeps only two sentences; the Qwen3-Next /
  Qwen3.5 "three of every four layers" detail and the Mamba / xLSTM names land here).
- **Position:** RoPE (llm_training 16), NoPE layers, long-context extension (dl4nlp 16).
- **FFN cost - MoE:** router, top-k, load balancing, shared + fine-grained experts; total vs
  active parameters (gpt-oss-20b 21B/3.6B, Qwen3.5 397B/17B, Kimi K2.5 1T/32B). Why huge models
  run cheaply.
- **More than one token per step:** speculative decoding (draft + verify, lossless) and
  multi-token prediction heads (DeepSeek-V3; Gemma 4 MTP checkpoints, Apr 2026; Step 3.5 Flash).
- **FlashAttention** (exact, IO-aware) in 2 frames; **serving** many users: continuous batching,
  PagedAttention.
- **The landscape, dated** (re-verify before delivery): MoE with small active counts everywhere;
  small on-device models (Gemma 4 E2B/E4B, the 0.8B Qwen3.5 in llama.cpp's README); native
  multimodality; thinking modes with budgets; agentic coding as the benchmark everyone chases;
  diffusion language models as the non-autoregressive alternative (forward pointer to the
  diffusion chapter).
- **What did not change:** residual stream, attention, next-token training. Raschka's conclusion
  from the Jan-Feb 2026 releases: performance comes mostly from data and training recipes, not
  architecture.
- **Sources:** dl4nlp 11 (KV cache, MQA/GQA, FlashAttention, speculative decoding, continuous
  batching, ~12 frames), 10 (MoE, 20 -> ~6), 16 (long context, 22 -> ~6); llm_training 13 (MoE),
  15 (FlashAttention), 16 (RoPE), 17 (DeepSeek-V3: MLA, MTP); `misc/dl4nlp/_reference_welchlabs_mla/`
  (MLA video frames); Raschka's LLM Architecture Gallery for figures to redraw.

---

## 6. Practicals

| Practical | After | What students do | Compute |
|---|---|---|---|
| **P1 - A tiny GPT that speaks Armenian** | LLM-3/4 | Sequel to the ch11 surname inventor: same spirit, now a transformer. Tokenizer (char or the LLM-1 BPE) -> CLM pretraining -> perplexity -> decoding knobs | CPU, minutes (estimate - measure when built) |
| **P2 - Your own model, on your own laptop** | LLM-9 | QLoRA fine-tune a small model on Colab (T4) on a small Armenian instruction set -> merge -> GGUF -> quantize -> run it in the D4 runtime; evaluate before/after | Colab GPU (`colab-gpu` skill) + laptop |
| HW - The quantization ladder | LLM-8 | Size, RAM, tokens/s, perplexity per quant level, on their own machine | Laptop, small model |

P2's open choices: D5 (library), D6 (base model), and the instruction data (needs a source - not
chosen yet).

---

## 7. Source map - where every existing deck goes

### `misc/dl4nlp/` (18 decks)

| Deck | Destination |
|---|---|
| 01 pre_transformer | Covered by L20/L21/L24; n-grams, word2vec, ELMo -> LLM-3 |
| 02 transformers | Already absorbed into L24-L26 |
| 03 tokenization | LLM-1; chat tokens -> LLM-5 |
| 04 decoding_strategies | LLM-2; speculative decoding -> LLM-10 |
| 05 evaluation | LLM-7; perplexity -> LLM-3; MAP/nDCG stay with L43 |
| 06 early_notable_models | LLM-3 (timeline, compressed) |
| 07 pretraining_finetuning | Objectives -> LLM-3; SFT + alignment idea -> LLM-5; PEFT -> LLM-9 |
| 08 prompting | LLM-6; ReAct -> L44; Tree of Thoughts cut |
| 09 hallucinations | LLM-7 |
| 10 mixture_of_experts | LLM-10 |
| 11 inference_optimization | Quantization + memory arithmetic -> LLM-8; distillation -> LLM-5; KV cache, GQA, FlashAttention, speculative, batching -> LLM-10 |
| 12 rag | **Retired** - ch17 L41-L43 |
| 13 scaling_laws | LLM-4 |
| 14 agents_tool_use | **Retired** - ch18 L44 |
| 15 reasoning_test_time | CoT / self-consistency -> LLM-6; two scaling axes -> LLM-4; PRM/ORM, o1/R1 -> L32g |
| 16 long_context_attention | LLM-10 |
| 17 emergence | Merged into LLM-4 |
| 18 reinforcement_learning | **Retired** - ch11 L32-L32g |

### `ml/llm_training/slides/` (19 paper decks)

They stay as the registered "LLM Training & Alignment" reading list; the lectures borrow from them.

| Deck | Feeds |
|---|---|
| 01 GRPO, 11 DeepSeek-R1 | L32g (already borrowed) |
| 02 DPO, 10 InstructGPT | L32f; InstructGPT picture also LLM-5 |
| 03 LoRA, 12 QLoRA | LLM-9 (QLoRA's NF4 section also LLM-8) |
| 04 LIMA, 18 FLAN, 08 Qwen3 | LLM-5 |
| 05 BPE-dropout | LLM-1 (optional frame) |
| 06 Don't Stop Pretraining | LLM-9 (continued pretraining) |
| 07 Llama 3 | LLM-3 (data), LLM-4 (over-training) |
| 09 Scaling laws | LLM-4 |
| 13 MoE, 15 FlashAttention, 16 RoPE | LLM-10 |
| 14 Chain of thought | LLM-6 |
| 17 DeepSeek-V3 | LLM-10 (MLA, MTP), LLM-8 (FP8) |
| 19 Making training fast | LLM-8 (float formats) |

---

## 8. Session budget

| Option | Lectures | Sessions (+ RAG 3, agents 1) |
|---|---|---|
| Short | LLM-1, LLM-2, LLM-3+4 merged, LLM-5, LLM-6+7 merged, LLM-8+9 merged (lighter) | 6 (+4) |
| **Core (recommended)** | LLM-1 to LLM-9 | 9 (+4) |
| Full | Core + LLM-10 (2 sessions) | 11 (+4) |

Practicals add 1-2 sessions in any option. Context: `ml/00_plan.md` (Aug 8) gave the LLM topics 5
sessions and projected a mid-to-late November finish. Counted 2026-10-02: **45 lecture decks are
already built after L16** (L17 through L48, including the 8 mech-interp decks), before this
block's 9-11 and any practicals. Something gets cut whatever is chosen here.

---

## 9. Recent developments - verified 2026-10-02

Primary sources unless marked. **Re-verify before delivery.**

- **Gemma 4** - released 2026-03-31 in E2B, E4B, 26B A4B (MoE) and 31B; MTP checkpoints
  2026-04-16; a 12B "Unified" 2026-06-03. [Gemma releases](https://ai.google.dev/gemma/docs/releases)
- **Gemma 3 QAT** - quantization-aware-trained checkpoints, including Q4_0 GGUFs for llama.cpp
  and Ollama (2025). [Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- **gpt-oss** - 2025-08-05; 120B (117B total / 5.1B active) and 20B (21B / 3.6B); MoE weights in
  MXFP4; 120B fits one 80 GB GPU, 20B fits 16 GB; alternating full and 128-token sliding-window
  attention; learned attention sinks; 128K context. [HF blog](https://huggingface.co/blog/welcome-openai-gpt-oss)
- **Jan-Mar 2026 open-weight architectures** - Qwen3.5 (397B/17B, Gated DeltaNet hybrid), GLM-5
  (744B/40B, MLA + DeepSeek sparse attention), Kimi K2.5 (1T/32B, MLA), Step 3.5 Flash (MTP-3 in
  training and inference), MiniMax M2.5 (classic GQA), Nemotron 3 Nano (mostly Mamba-2), Mistral
  Small 4 (119B/6.63B, MLA). Author's conclusion: data and recipes matter more than architecture.
  [Raschka, Jan-Feb 2026](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight),
  [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- **llama.cpp** - README quick start is now `llama cli -hf ...` / `llama serve -hf ...`, with an
  installer at llama.app; `llama-quantize`, `llama-perplexity`, `llama-bench`, `llama-imatrix`
  are still standalone tools; 1.5- to 8-bit quantization.
  [README](https://github.com/ggml-org/llama.cpp), `tools/CMakeLists.txt` (via Context7).
  NVFP4 kernels merged March-April 2026 (secondary source citing PR numbers; check the PRs at
  build). [insiderllm](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)
- **Ollama** - MLX backend on Apple Silicon (preview 2026-03-30); Anthropic API compatibility
  (2026-01-16); `ollama launch` (2026-01-23); cloud models (2025-09-19); gpt-oss (2025-08-05).
  [Ollama blog](https://ollama.com/blog). LoRA adapters no longer supported
  (`server/create.go:40`, `:903`); no quantization on GGUF import (`docs/import.mdx:35`) - both
  read from the live `main` branch.
- **HF transformers runs llama.cpp quants** - 2026-09-22: `from_pretrained(..., gguf_file=...)`
  now keeps the weights packed and reuses the ggml kernels (Metal first; transformers `main`, not
  yet a release) instead of dequantizing on load.
  [HF blog](https://huggingface.co/blog/transformers-llama-cpp-quants)
- **Unsloth** - `save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")`.
  [Unsloth docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md)
- **FP4 formats** - MXFP4: OCP standard, blocks of 32, E8M0 shared scale. NVFP4: blocks of 16,
  FP8 E4M3 block scale + FP32 tensor scale; native on Blackwell. Quality vs Q4_K_M unsettled
  (secondary). [insiderllm](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/),
  [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **Why LMs hallucinate** - Kalai, Nachum, Zhang (OpenAI) and Vempala (Georgia Tech),
  2025-09-04: training and evaluation reward guessing. [arXiv 2509.04664](https://arxiv.org/abs/2509.04664)
- **Diffusion LMs** - Mercury (Inception), Gemini Diffusion, LLaDA 2 reported as deployed in 2026.
  **Secondary sources only** - confirm before putting a number on a slide.

**Seen in search, not used:** "Kimi K3", "DeepSeek V4 Pro", "GLM-5.2", "Qwen3.8" appeared only on
SEO aggregator pages. Not in this plan until a primary source confirms them.

---

## 10. Known issues found while planning

- **`misc/dl4nlp/03_tokenization.pdf` is 0 bytes**, committed that way in b7424f8 (2026-07-26).
  The `.tex` (38 frames) is intact; the GitHub PDF link serves an empty file. Recompile.
- `ml/00_plan.md` is from 2026-08-08 and predates the reorder and this plan.
- The dl4nlp decks use `misc/dl4nlp/preamble.tex`; macros differ from `ml/preamble.tex`
  (`13_rnns/RNN_CHAPTER_PLAN.md` lists known deltas: `lightbg`, `-Stealth`).

## 11. Next steps, after approval

1. Answer D1 and D3-D7 (D2 resolved 2026-10-03).
2. Reorder edits from section 2 (L23, L23c, `_quarto.yml`, qmd titles, `00_plan.md`).
3. Create the chapter folder (D7), move this file into it.
4. Per lecture: outline -> approval -> build (port from the source decks in section 7) ->
   compile-deck -> student review (opt-in).
5. D6 measurement (Armenian tokens/word + perplexity for 3-4 candidate models) before P1/P2 are
   built.
