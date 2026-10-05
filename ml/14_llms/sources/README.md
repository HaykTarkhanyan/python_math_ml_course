# sources/ - raw material for the LLM chapter

Copied here on 2026-10-04 (DECISIONS #66) so that everything the chapter is built from sits in one
folder. These are **copies**: the originals were not touched. Port from these into the chapter's
own decks (one level up); do not polish the copies themselves. Which deck feeds which lecture:
`../LLM_CHAPTER_PLAN.md`, section 7.

Only files tracked by git were copied (202 files, 17.7 MB).

## `dl4nlp/` - from `misc/dl4nlp/`

The instructor's earlier DL4NLP course: 15 survey decks (`01_pre_transformer` to
`17_emergence`), their `preamble.tex` and `fig/`, `dl4nlp_outline.md`, `CROSSREF_vs_lmu_s26.md`,
and the notes for Karpathy's "Deep Dive into LLMs" (`_yt_videos/karpathy_deep_dive_llms/`: transcript
and README; the video itself is git-ignored and was not copied).

- **Left out:** `12_rag` (replaced by `ml/ch17_rag`), `14_agents_tool_use` (`ml/ch18_agents`),
  `18_reinforcement_learning` (`ml/ch11_rl`), and `materials_dl4nlp.pdf` (a merged bundle of the
  decks).
- **Compile** from inside `dl4nlp/`: `pdflatex 03_tokenization.tex` (twice). The decks load
  `\input{preamble}` from the same folder.
- `03_tokenization.pdf` was 0 bytes in the original (committed that way in b7424f8); it was
  recompiled here on 2026-10-04 and copied back over the original.

## `llm_training/` - from `ml/llm_training/slides/`

16 paper decks, one folder each (`.tex`, `.pdf`, `fig/`, and `py_src/` where the deck has one).

- **Left out:** `01_grpo`, `02_dpo`, `11_deepseek_r1` - they feed the RL chapter (L32f, L32g).
- **One edit per copy:** `\input{../../../preamble}` became `\input{../../../../preamble}`, so it
  still reaches `ml/preamble.tex` from one folder deeper. Checked by compiling `03_lora` (0 errors).
- The originals stay registered on the site (`ml/llm_training/llm_training.qmd`).

## Not copied - use them where they are

| Where | What | For |
|---|---|---|
| `misc/grokking/` | Welch Labs grokking material, extracted (11.3 MB tracked) | LLM-5 |
| `ml/claude_projects/armenian_glitch_token_hunt/` | Our own glitch-token project | LLM-1 |
| `misc/dl4nlp/_reference_welchlabs_mla/` | MLA video frames (git-ignored, local only) | LLM-11 |
| `ml/llm_training/materials/` | The papers and blog posts behind the paper decks | all |
| `ml/dl4nlp/moodle_s26_course/` | LMU DL4NLP summer 2026 slides and exams (66 MB tracked) | cross-checks |
| `ml/text_embedding/` | Embedding papers | LLM-4, RAG |
| `ml/13_rnns/py_src/tokenizer_demo.py`, `embedding_2d.py` | Tokenizer panels, the 2D embedding schematic | LLM-1, LLM-4 (move when LLM-1 is built; see the plan) |
