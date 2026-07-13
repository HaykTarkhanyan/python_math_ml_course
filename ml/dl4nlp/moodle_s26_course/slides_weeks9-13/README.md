# DL4NLP slides — weeks 9–13 (Prof. Schütze's decks)

The Moodle "Lecture Material" entries for **weeks 9–13** all point to Prof. Schütze's course
page rather than the slds-lmu web chapters:

- Source page: <https://www.cis.lmu.de/~hs/teach/26s/dl4nlp/>
- Downloaded: 2026-07-12

Unlike weeks 1–8 (see `../slides_tex/`, built from repo `.tex`), these are kept as **PDFs**
because they are Schütze's **curated lecture decks** (`9*.pdf`) that do **not** map 1:1 to the
repo's `.tex` filenames, and they are **not in the local `_reference/lecture_dl4nlp` clone**
(HEAD 2026-05-13). They live on the repo's live `main` branch and were fetched via the raw
GitHub URLs the page links to.

## Readable text (`.md` next to each `.pdf`)

Each deck has a readable `.md` companion so the content can be read without PDF parsing:
- **7 lecture decks** — auto-extracted with `pdftotext` and cleaned (slide titles → `##`).
  Body text is accurate; inline `�` marks a glyph the tool couldn't map (often ×, a Greek
  letter, or a bullet). Most diagrams are not captured (see the PDF), **but the important
  diagram-only slides — where the figure carries the meaning and the text alone is not
  obvious — have hand-written visual notes added inline** (marked `> **Figure**`). Covered so
  far: `9fast` (data & tensor parallelism, SRAM/HBM + FlashAttention, tiling, operation
  fusion) and `9rlhf` (InstructGPT 3-step pipeline). Spot-checked `9compute`/`9scale` — those
  turned out to be text/formula slides already captured, so no notes needed.
- **2 misc decks** (image-only, no extractable text) — **visually transcribed**:
  - `misc/slido-brunskill3.md` — an RLHF True/False quiz with the **instructor's handwritten
    answers** (authoritative).
  - `misc/agentic-ai_oview.md` — a promo for a *different future* course ("WP3 Agentic AI",
    WiSe 26/27), not DL4NLP content.

## Contents (229 pages, 9 decks)

| Chapter | File | Topic (link text) | Pages |
|---|---|---|---|
| 09 LLMs | `09_llm/9tune_finetuning-and-prompting.pdf` | Finetuning and prompting | 31 |
| 09 LLMs | `09_llm/9cotp_chain-of-thought-fewshot.pdf` | Chain-of-thought few-shot prompting | 27 |
| 10 RLHF/Instr. Tuning | `10_rlhf/9itune_instruction-tuning.pdf` | Instruction tuning | 43 |
| 10 RLHF/Instr. Tuning | `10_rlhf/9rlhf.pdf` | RLHF | 61 |
| 11 Training LLMs | `11_training-llms/9compute.pdf` | compute | 20 |
| 11 Training LLMs | `11_training-llms/9fast.pdf` | fast | 15 |
| 11 Training LLMs | `11_training-llms/9scale.pdf` | scale | 27 |
| Misc | `misc/agentic-ai_oview.pdf` | Agentic AI (overview) | 3 |
| Misc | `misc/slido-brunskill3.pdf` | slido (poll export) | 2 |

## Source URLs

Chapters 09–11 (raw GitHub, `slds-lmu/lecture_dl4nlp/main/slides/…`):
`chapter09-llm/9tune.pdf`, `chapter09-llm/9cotp.pdf`, `chapter10-rlhf/9itune.pdf`,
`chapter10-rlhf/9rlhf.pdf`, `chapter11-training-llms/{9compute,9fast,9scale}.pdf`.
Misc (cis.lmu.de): `~hs/teach/26s/dl4nlp/oview.pdf`, `~hs/teach/26s/dl4nlp/slido,brunskill3.pdf`.

## Related `.tex` in the repo (different names, for reference)

The nearest source `.tex` in `_reference/lecture_dl4nlp/slides/` (names don't match the curated
decks, so not copied into `slides_tex`):
- `chapter09-llm/`: `slides-91-instruction-tuning.tex`, `slides-92-chain-of-thought.tex`, `slides-93-emergent-abilities.tex`, `slides-94-big-models.tex`, `slides-94-deepseek.tex`, `slides-95-big-model-benchmarks.tex`, `reinforce.tex`
- `chapter10-rlhf/`: `slides-10-rlhf.tex`
- `chapter11-training-llms/`: `slides-111-compute-memory.tex`, `slides-112-reduce-comp.tex`, `slides-113-scaling.tex`, `slides-114-x-optimize.tex`

To refresh, re-fetch the URLs above (or `git pull` the repo and copy the `9*.pdf` from
chapters 09/10/11).
