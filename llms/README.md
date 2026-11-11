# llms/ - the LLM part of the course

Moved out of `ml/14_llms/` on 2026-10-28 (instructor: one folder per deck; DECISIONS #102).

| Folder | Deck | What is in it |
|---|---|---|
| `01_tokenization/` | LLM-1 Tokenization | deck, outline, `tokenization.qmd` (the website page), `fig/`, `py_src/`, `results/` |
| `02_decoding/` | LLM-2 Decoding (taught as video 03: deck `03_decoding.tex`) | same layout, plus `decoding.qmd` (website page) and the API notebook `04_llm_api.ipynb` |
| `03_embeddings_position/` | LLM-3 Embeddings and position | same layout (`long_context_figs.py` also lives here) |
| `04_attention/` | LLM-4 Attention | same layout, plus `attention.qmd` (the older page covering LLM-4 to LLM-6) |
| `05_transformer_block/` | LLM-5 The Transformer block | same layout |
| `06_transformers_in_the_world/` | LLM-6 Transformers in the world | same layout |
| `07_pretraining/` ... `14_anatomy_2026/` | LLM-7 to LLM-14 | same layout |
| `00_shared/` | used by several decks | `fig/roadmap_*.pdf` (the course map, from `py_src/llm_roadmap.py`), `fig/borrowed/` (3Blue1Brown, paper and Welch Labs stills), `research/`, `sources/`, `archive/` |

Chapter-level plans sit here at the top: `LLM_CHAPTER_PLAN.md`, `INTERVIEW_QUEUE.md`, `ATTENTION_CHAPTER_PLAN.md`.

Working with a deck:

- Compile from inside the deck's folder (the decks load `\input{../../ml/preamble}`), twice:
  `cd llms/01_tokenization && pdflatex -interaction=nonstopmode -halt-on-error 01_tokenization.tex`
- Each deck's `\graphicspath` searches its own `fig/` and `../00_shared/fig/` (plus `borrowed/` where used).
- Figure scripts run from the repo root, e.g. `./ma/Scripts/python.exe llms/01_tokenization/py_src/tokenization_figs.py`.
  They find their folders relative to themselves (`parents[1]` = the deck folder, `parents[3]` = the repo root).
- `NN_topic_notes.pdf` is the copy the instructor annotates in class. Open that one for the instructor, never the
  clean PDF (see `CLAUDE.md`).
