# Outline — [LLM-18] FLAN (Instruction Tuning at Scale)

**Paper:** Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai, Le, "Finetuned Language Models Are
Zero-Shot Learners" (ICLR 2022), arXiv:2109.01652.
Source text: `materials_md/papers/24_flan_2109.01652.txt`.

**Title:** `[LLM-18] FLAN: Instruction Tuning at Scale`
**Subtitle:** `Finetune on many tasks phrased as instructions -> zero-shot generalization to unseen tasks (Wei et al., 2021)`

## Core idea (one paragraph)
A pretrained LM is a strong few-shot learner but clunky **zero-shot** — it wasn't trained to obey
instructions. **FLAN** takes a **137B LaMDA-PT** model and finetunes it on a **large collection of
NLP datasets, each phrased as a natural-language instruction**. This teaches "**follow the
instruction**" as a general skill that **transfers to task types it never saw**: FLAN beats
zero-shot GPT-3 175B on **20 of 25 datasets**. The evaluation is rigorous — to score a task
cluster, that **whole cluster is held out** of finetuning. The ablations are the payload: the
benefit grows with the **number of task clusters**, **requires scale** (instruction tuning *hurts*
models ≤ 8B and only helps at 68B / 137B), and **depends on the instructions themselves** (strip
them and multitask finetuning loses the gain). FLAN is the "scale up instruction data" thesis;
**LIMA (LLM-4)** is its "less is more" counterpoint.

## Chapter fit
The mainstream **counterpoint to LIMA (04)** (many tasks at scale vs 1,000 curated examples) and
the **SFT-data** story behind step 1 of InstructGPT (10). Its scale-emergence ablation rhymes with
CoT's emergence (14). No architecture overlap with dl4nlp.

## Section + frame plan

**Hook — "Great at examples, clumsy at instructions"**
GPT-3 shines few-shot but often flubs a bare zero-shot instruction (echo the InstructGPT framing,
LLM-10): next-token pretraining never rewarded "do what this instruction says." FLAN's question:
can we make a model good at **brand-new tasks with zero examples**, just by teaching it the *habit*
of following instructions? TikZ: tiny "few-shot ok / zero-shot poor -> instruction-tuned zero-shot
strong" sketch.

**§1 The idea: teach the habit, not the tasks**
- *Intuition frame (before the setup).* You're not teaching the model the tasks — you're teaching
  the **format**: "read an instruction, produce what it asks." Learn that on enough tasks and it
  **generalizes to instructions you never trained on**. Analogy: a new hire who's handled 60 kinds
  of ticket, then handles a 61st kind they've never seen because they've learned *how to take a
  ticket*.
- *Instruction tuning, defined.* Sits between pretraining and use: multitask finetuning where every
  example is `(instruction, input) -> output`. Fig: `instruction_tuning.pdf` (pretrain -> instruction
  tune -> zero-shot on unseen cluster).

**§2 The method**
- **62 NLP datasets** grouped into **12 task clusters** (NLI, QA, summarization, translation, ...);
  each dataset gets ~**10 hand-written instruction templates** (up to 3 "turned around"). Base
  model: **137B LaMDA-PT** (decoder-only). Fig: `task_clusters.pdf` (the 12-cluster map).
- *The rigorous eval.* To measure zero-shot on a cluster, **hold out that entire cluster** during
  finetuning — so "unseen task" really means unseen. Keybox: this conservative held-out-cluster
  design is what makes the zero-shot claim honest.

**§3 Results**
- FLAN 137B > **zero-shot** GPT-3 175B on **20/25 datasets**, and beats **few-shot** GPT-3 on
  **10** (the "large margin" set: ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, StoryCloze). Fig:
  `flan_vs_gpt3.pdf` (real).
- *Honest caveat (add).* Instruction tuning does **not** help tasks already close to the LM
  objective — commonsense reasoning / coreference posed as sentence completions (FLAN beat
  LaMDA-PT on only **3 of 7** such tasks), because the instruction is redundant there. This
  sharpens "it's the *instructions*, not just multitask exposure."

**§4 What makes it work (the ablations — the heart)**
- *Number of task clusters.* Add more clusters -> held-out performance keeps rising (still not
  saturating at 7 clusters). Fig: `num_clusters.pdf` (real/trend). Message: instruction *diversity*
  drives generalization.
- *Predict-first: does this help a small model?* Sizes tested: **422M, 2B, 8B, 68B, 137B**.
  Instruction tuning **helps at 68B and 137B** and **HURTS at 8B and below** (small models spend
  limited capacity fitting many tasks, losing zero-shot); the crossover is **between 8B and 68B**.
  Fig: `scale_emergence.pdf` (real). Parallels CoT emergence (LLM-14) — cross-reference it.
- *Are the instructions necessary?* Ablate the templates (train on plain input->output, no
  instruction): the gain **collapses**. It's the *instructions*, not merely multitask exposure.
  Fig: `instructions_ablation.pdf` — 4-cluster avg: FLAN **55.2**, FT-dataset-name **46.6 / 47.0**,
  FT-no-template **37.3**.

**§5 Recap + Next**
- Recap: instruction tuning = multitask finetuning phrased as instructions; teaches a transferable
  "follow instructions" skill; needs many clusters + scale (helps ≥68B, hurts ≤8B); instructions
  are essential.
- Next box: **LIMA (04)** is the deliberate counterpoint (quality over quantity); FLAN is the
  SFT-data engine behind InstructGPT (10) and modern post-training. (One-line pointer to the
  follow-up FLAN-T5 / "Scaling Instruction-Finetuned LMs," 2022 — do not conflate with this paper.)

## Figures (Python/matplotlib)
- `instruction_tuning.pdf` — pretrain -> instruction-tune -> zero-shot-on-held-out schematic.
- `task_clusters.pdf` — the 62 datasets / 12 clusters, held-out marked (schematic).
- `flan_vs_gpt3.pdf` — REAL, FLAN vs zero/few-shot GPT-3 (20/25 datasets; few-shot on 10).
- `num_clusters.pdf` — REAL/trend, performance vs number of clusters (rising, non-saturating).
- `scale_emergence.pdf` — REAL, instruction tuning vs model size (422M/2B/8B hurt; 68B/137B help)
  — the signature predict-first figure.
- `instructions_ablation.pdf` — 4-cluster avg: 55.2 / 46.6 / 47.0 / 37.3.

## Pedagogical notes
- Intuition-before-method (the "new hire / take a ticket" analogy) then the setup.
- Predict-first on the scale-emergence result (§4) — it's genuinely counter-intuitive.
- Minimal/no code; this is a data+method paper.
- Make the **LIMA contrast** explicit at both hook-adjacent and recap (matched pair).

## Build notes / cautions
- **Numbers locked by review:** 137B LaMDA-PT, 62 datasets / **12 clusters**, ~10 templates each,
  20/25 **datasets** (not "tasks"), few-shot beaten on 10; emergence sizes 422M/2B/8B/68B/137B with
  crossover between 8B and 68B; instructions-ablation 55.2 / 46.6 / 47.0 / 37.3.
- Don't conflate this original FLAN (Wei 2021, LaMDA-PT) with FLAN-T5 / the FLAN Collection (2022/
  2023) — mention the follow-up only as a one-line pointer.
- Conventions: `\input{../../../preamble}`, local callout macros (copy from `13_moe.tex`),
  `[plain]` transitions, `% Provenance:` block. Compile twice, clean, eyeball for overflow.
