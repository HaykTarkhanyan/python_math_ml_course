# Cross-reference: my DL4NLP decks vs LMU s26 course

Compares **this folder** (`misc/dl4nlp/` - my own 18-deck DL4NLP course) against the
**LMU s26 reference course** archived at `ml/dl4nlp/moodle_s26_course/` (weeks 1-13 slides +
5 past exams). I am on both sides: author of these 18 decks, and a student sitting the LMU
exam (29.07.2026). So this doc answers two things at once - does my course cover the reference
course, and where are my exam gaps.

Built: 2026-07-22. Deck content taken from `dl4nlp_outline.md` (topics 01-13) and the actual
`\frametitle`s of decks 14-18 (which go beyond the written outline). Exam frequencies are from
`../../ml/dl4nlp/moodle_s26_course/exams/exam_topics_crossref.md`.

## Deck -> LMU chapter -> exam relevance

| My deck (misc/dl4nlp) | LMU s26 chapter/week | Exam freq (of 5) |
|---|---|---|
| 01 pre_transformer (RNN, n-gram, Word2Vec) | Ch01 NPLM+embeddings, Ch02 RNN | embeddings 2/5 |
| 02 transformers (attn, pos-enc, enc-dec) | Ch03 transformer | internals 3/5, params 3/5 |
| 03 tokenization (BPE, FastText) | Ch02 slides-24-tokenization | 3/5 |
| 04 decoding_strategies | Ch08 decoding | 3/5 |
| 05 evaluation (PPL/BLEU/ROUGE) | Ch04 metrics, Ch08 eval_metrics | via decoding |
| 06 early_notable_models (GPT/BERT/T5) | Ch04 BERT, Ch06 T5, Ch07 GPT | pretrain-obj 4/5, T5 3/5 |
| 07 pretraining_finetuning (MLM/NSP/CLM; SFT/RLHF/DPO/GRPO/LoRA) | Ch04 pretrain-finetune, Ch05, wk9 finetuning, wk10 rlhf | paradigms 5/5 |
| 08 prompting (zero/few-shot, CoT) | wk9 finetuning-and-prompting, CoT-fewshot | paradigms 5/5, CoT 3/5 |
| 09 hallucinations | exam topic (appears 23M/24R) | 2/5 |
| 10 mixture_of_experts | - (not a distinct LMU s26 week) | - |
| 11 inference_optimization (KV-cache, quant, distill) | wk11 compute/fast, Ch05 distilbert | - (wk9-13, unexamined) |
| 12 rag | - (not in LMU weeks 1-13) | - |
| 13 scaling_laws (Chinchilla) | wk11 scale | 1/5 (24R) |
| 14 agents_tool_use (ReAct, MCP, SWE-Bench) | wk misc "agentic-ai" (different course, per README) | - |
| 15 reasoning_test_time (o1/o3, DeepSeek-R1, ToT) | 25M exam "reasoning models" only | 1/5 (25M) |
| 16 long_context_attention (FlashAttn, MQA/GQA, Mamba) | wk11 fast, Ch03 efficient/trafo-xl | 1/5 (distributed, 23R) |
| 17 emergence (mirage debate, grokking) | upstream ch09 "emergent-abilities" (1 slide) | - |
| 18 reinforcement_learning (MDP->PPO, RLHF) | wk10 rlhf | 1/5 (23R) |

## Gaps - LMU / exam topics my decks do NOT cover

1. **PyTorch + HuggingFace hands-on coding - the single biggest gap.** Appears in **all 5
   exams** and is the largest point block (19 pts in 24M, ~25 in 24R, 30 in 25M): writing a
   training loop, completing `MultiHeadAttention`, spotting bugs in a Transformer block,
   `Trainer`/`TrainingArguments`, gradient accumulation & effective batch size. **All 18 of my
   decks are conceptual/theory - none are coding decks.** For exam prep, my own material earns
   zero of these points. Fix this first.
2. **Multilinguality** (LMU upstream ch12; exam 23R: BPE across languages, continued
   pretraining, data rebalancing) - not in any of my 18 decks.
3. **Lighter than LMU:** ELMo (Ch02), BERTology / probing (Ch05), Transformer-XL specifically
   (Ch03), NSP as a first-class topic. I cover distillation (deck 11) but not the BERTology
   probing angle.

## My decks that go BEYOND the LMU s26 course

My course is broader and more current on the modern-LLM end:

- **10 mixture_of_experts** - not a distinct LMU s26 week.
- **12 rag** - not in LMU weeks 1-13 at all.
- **14 agents_tool_use** (ReAct, MCP, SWE-Bench, computer use) - LMU only has an "agentic-ai
  overview" that its README flags as a promo for a *different future course*, not DL4NLP.
- **15 reasoning_test_time** (o1/o3, DeepSeek-R1, ToT, PRM/ORM) - LMU only touches "reasoning
  models" as the newest 25M exam topic; my deck is far deeper.
- **17 emergence** (mirage debate, grokking) - LMU upstream has one "emergent-abilities" slide;
  I have a full deck.

## Bottom line

- As a **course**, my 18 decks cover essentially every *conceptual* topic in the LMU reference
  and extend well past it (MoE, RAG, agents, reasoning, emergence, long-context/Mamba).
  Content-wise mine is the more modern of the two.
- As **exam prep**, one glaring hole: **hands-on PyTorch/HuggingFace coding**, the #1 guaranteed
  exam block, completely absent from my theory decks. Multilinguality is a smaller second gap.
  `exam_topics_crossref.md` already says to weight 24M/24R/25M style - those lean hardest on
  exactly the coding I have no decks for.

## Notes

- **Deck 18 (RL) has no compiled PDF** yet - only `18_reinforcement_learning.tex`.
- `materials_dl4nlp.pdf` in this folder is a separate materials/reading list (not cross-refd here).
- **Third overlapping source in the repo:** `ml/llm_training/` (paper decks: FlashAttention,
  RoPE, DeepSeek-V3, GRPO, DPO, LoRA, LIMA, FLAN). Overlaps decks 07/16/18 - if consolidating,
  these three sources partly duplicate each other.
