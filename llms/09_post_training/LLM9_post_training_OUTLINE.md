# LLM-9 - Post-training: from base model to assistant - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-9) and the recommended answers in
`INTERVIEW_QUEUE.md` (instructor: "do all queue items"). Session 9. One question: **LLM-7 ended with
a base model that continues documents - how does it become something that answers?** Three verbs
carry the deck: **imitate** (SFT, distillation), **compare** (preferences: reward model, RLHF, DPO),
**check** (verifiable rewards). Papers are evidence inside the spine, not stops on a tour. Idea
level only (D3): the RL maths is the RL chapter's.

Deck file: `LLM9_post_training.tex`. Figures: `py_src/posttrain_figs.py` (one `--gpt2` run of the
cached GPT-2; the chat template tokenized with Qwen2.5's cached `tokenizer.json`; toys asserted).

**Decisions (queue answers, 2026-10-06):** the three-verb spine; the cold open is one run of the
cached GPT-2 on "How do I reset my password?"; Bradley-Terry by hand (2.0 vs 0.5 -> 0.82);
reasoning models as a one-frame teaser.

**Not available locally:** an instruction-tuned model. Only Qwen2.5-0.5B-Instruct's `tokenizer.json`
is cached, not its weights, so the cold open compares GPT-2 raw with GPT-2 inside a "User: /
Assistant:" frame, not with an instruct model. Downloading one is the instructor's call
(recorded in DEFERRED_TODO-style notes in the session log, not done).

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| InstructGPT 1.3B preferred to GPT-3 175B | Ouyang et al. 2022; local `sources/llm_training/10_instructgpt` |
| FLAN 137B (zero-shot) beats zero-shot GPT-3 175B on 20 of 25 datasets; 62 datasets as instructions; instruction tuning hurt models of 8B and below | Wei et al. 2021; local `sources/llm_training/18_flan` |
| LIMA: LLaMa 65B, 1,000 curated examples, plain SFT; equivalent or preferred to GPT-4 in 43% of cases | Zhou et al. 2023 abstract (arXiv 2305.11206) |
| Over-optimising a proxy reward model hurts the "gold" reward (Goodhart) | Gao et al. 2022 abstract (arXiv 2210.10760) |
| DeepSeek-R1: distilled into Qwen and Llama models with about 800k samples (600k reasoning + 200k other); R1-Zero used rule-based accuracy and format rewards, no neural reward model "may suffer from reward hacking" | DeepSeek-AI 2025 (arXiv 2501.12948, full text) |
| ChatML in Qwen2.5: `<|im_start|>` 151644, `<|im_end|>` 151645, `<|endoftext|>` 151643; default system message "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." | Qwen2.5-0.5B-Instruct `tokenizer_config.json` (fetched); IDs re-checked against the cached `tokenizer.json` |
| "a simulation of a human labeler" | Karpathy, Deep Dive into LLMs (2025), transcript line 155 |
| DPO trains on preference pairs directly, no reward model and no RL loop | Rafailov et al. 2023 |

---

## Frames

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Post-training" / "From a base model to an assistant" | - |
| 1-2 | Where we are | Post-train lit | `roadmap_post_training_*` |
| 3 | Cold open: ask a base model | **Predict:** what does GPT-2 write after "How do I reset my password?" | - |
| 4 | It writes an FAQ page | The real run: "please contact us", then the question again, in a loop. Right: the same question inside "User: / Assistant:" (real). The knowledge is not the problem; the habit is. | `post_cold_open.pdf` |
| 5 | Outline | | |
| 6 | Three ways to change a habit | imitate / compare / check, and what each needs: written answers / two answers and a choice / an automatic checker. | `post_three_verbs.pdf` |
| 7 | [plain] Imitate | "Show it thousands of good conversations." | |
| 8 | Supervised fine-tuning | LLM-7's loss on different data: conversations written by labelers; loss on the assistant's tokens only. "A simulation of a human labeler" (Karpathy). | - |
| 9 | The chat template | A conversation becomes one token stream (ChatML, Qwen2.5): the special tokens and their IDs, the default system message; the model learns to end its turn with `<|im_end|>` (LLM-6's stop rule). | `post_chatml.pdf` |
| 10 | How much data? | FLAN: breadth (62 datasets; 20 of 25 vs GPT-3; hurts at 8B and below). LIMA: 1,000 examples, 43% vs GPT-4 - the knowledge was already there. | - |
| 11 | Imitating a bigger model | Distillation: SFT on a teacher's outputs (R1 -> Qwen/Llama, about 800k samples). | - |
| 12 | [plain] Compare | "Judging is easier than writing." | |
| 13 | Why compare | People rank two answers more reliably than they write a perfect one; preference pairs. | - |
| 14 | Bradley-Terry by hand | r_A = 2.0, r_B = 0.5: P(A preferred) = sigma(1.5) = 0.82; loss -ln 0.82 = 0.20 when people chose A; 1.70 when they chose B. | `post_bradley_terry.pdf` |
| 15 | RLHF in one picture | generate -> reward model scores -> nudge the model up the reward, held near the SFT model (a KL leash). Maths: the RL chapter. | `post_rlhf.pdf` |
| 16 | It works: 1.3B beats 175B | InstructGPT. | - |
| 17 | Reward hacking | The reward model is a proxy; push too hard and real quality falls (Gao et al.: Goodhart). Toy curve, labelled. | `post_goodhart.pdf` |
| 18 | DPO: skip the reward model | The same Bradley-Terry comparison as a loss on the model itself. | - |
| 19 | [plain] Check | "When an answer can be checked, the checker is the teacher." | |
| 20 | Verifiable rewards | Maths answers, unit tests; R1-Zero's accuracy + format rewards; no reward model, so less to hack. Thinking models (LLM-8's second axis): the RL chapter. | - |
| 21 | Recap | | |
| 22 | Next on the map | LLM-10: prompting. | `roadmap_prompting_life.pdf` |
| 23 | Sources | | |

**Self-review:** (1) idea level for RLHF/DPO (D3). (2) The cold open does not claim an instruct model's
output - none is local. (3) LIMA's 43% is "equivalent or preferred", not "better". (4) Reward-hacking
curve is a toy shape; only Goodhart's direction is sourced. (5) System prompts and prompt craft stay in
LLM-10; LoRA in LLM-13.
