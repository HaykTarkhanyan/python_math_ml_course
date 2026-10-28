# LLM-13 - Fine-tuning on a budget: LoRA and QLoRA - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-13) and the queue (instructor:
"do all queue items"; D5 answered: library-free slides - DECISIONS #75). Session 13. One question:
**full fine-tuning a 7B model needs ~112 GB - how do you adapt one on hardware you can get?**

Deck file: `LLM13_lora.tex`. Figures: `py_src/lora_figs.py` - the local demo is LoRA written by hand
in PyTorch on the cached GPT-2 (rank 8 on c_attn x 12, 150 steps, 32 conversations whose answers
open with "Good question!", answers-only loss; 20 held-out questions before and after; step 0 vs
the base; merged vs unmerged).

**Decisions:** D5 -> library-free (#75); the local demo: yes (the queue's R); D6 (the practical's base
model) stays open: measured 2026-10-06, only GPT-2 is fully cached (Qwen2.5-0.5B-Instruct: its
tokenizer only; arm-gemma-e4b: nothing), so every candidate needs a download - asked at the end.

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| LoRA: freeze W, inject trainable rank-decomposition matrices; GPT-3 175B: 10,000x fewer trainable parameters, 3x less GPU memory; on par or better than fine-tuning; no added inference latency | Hu et al. 2021 abstract |
| Accuracy flat from rank 1 for W_q, W_v (attention only adapted) | `sources/llm_training/03_lora` (Hu et al. Table 6) |
| QLoRA: 65B on one 48 GB GPU, 16-bit fine-tuning performance preserved; Guanaco 99.3% of ChatGPT on Vicuna; NF4, double quantization, paged optimizers; adapters on every linear layer | Dettmers et al. 2023 abstract; `sources/llm_training/12_qlora` |
| 7B: 112 GB full fine-tuning (16 B/param), 14 GB bf16, 3.5 GB 4-bit | by hand (LLM-8, LLM-12) |

---

## Frames

Title; two maps; cold open (predict the share of weights to train) -> "0.24% was enough" (the
demo); outline. **Should you fine-tune?** the ladder (prompt, RAG, fine-tune; continued pretraining).
**LoRA:** the bill; W + BA (to scale; B = 0, step 0 = base, measured); the count by hand (1.4%, 0.24%);
why low rank (intrinsic rank, Hu et al.); which matrices, alpha/r, merge (measured). **QLoRA:** the
memory bars (112 / 14 / 3.5 GB); NF4, double quantization, paged optimizers, Guanaco. **The recipe:**
data, template, answers-only loss, before/after, forgetting; merge before converting. Recap; next
(LLM-14); sources.
