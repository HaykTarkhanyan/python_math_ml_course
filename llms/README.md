# llms/ - the LLM part of the course

Moved out of `ml/14_llms/` on 2026-10-28 (instructor: one folder per deck; DECISIONS #102). Folders follow the
teaching order: on 2026-11-18 the tools practical became `03_tools_agents/` and the LLM-3 to LLM-14 folders moved up
by one (DECISIONS #117); the same night the RAG decks moved in from `ml/ch17_rag/` as `05_rag/` and the LLM
folders moved up once more (DECISIONS #119); on 2026-11-24 the Kargin agent session became `04_kargin_agent/`, so RAG is
`05_rag/` and from `06_` on folder NN holds deck LLM-(NN-3) (DECISIONS #121). Files inside are named by their video
number (`05_tool_calls.pdf`, `08_rag_retrieval.pdf`); untaught LLM decks keep their `LLMN_` names.

| Folder | Deck | What is in it |
|---|---|---|
| `01_tokenization/` | LLM-1 Tokenization | deck, outline, `tokenization.qmd` (the website page), `fig/`, `py_src/`, `results/` |
| `02_decoding/` | LLM-2 Decoding (taught as video 03: deck `03_decoding.tex`) | same layout, plus `decoding.qmd` (website page) and the API notebook `04_llm_api.ipynb` |
| `03_tools_agents/` | Tools practical, two sessions (videos 05 and 06, taught 2026-11-18) | decks `05_tool_calls` and `06_mcp_skills` (+ `_notes.pdf`), notebooks `05_tool_calls_solution.ipynb` and `06_mcp_skills_solution.ipynb`, `server.py`, `tools_lib.py`, `recorder.py`, `skills/`, outline `05_06_tools_agents_OUTLINE.md` |
| `04_kargin_agent/` | "Find that sketch": an agent over Kargin Haghordum, a project plus a notebook tour (video 07, built 2026-11-24) | `sketch_finder/` (a uv project: package, tests, `sketch-finder` command, eval, cost ledger, MCP server, data, results; see its README), notebook `07_kargin_agent_solution.ipynb`, `py_src/` (notebook builder, ArmBench table), `results/armbench_2026-11-24.json`, outline `07_kargin_agent_OUTLINE.md`, `archive/` (the prototype runs) |
| `05_rag/` | RAG, three lectures (videos 08, 09, 10; moved from `ml/ch17_rag/`) | decks `08_rag_retrieval`, `09_rag_hybrid_rerank`, `10_rag_generation_eval` (+ `_notes.pdf`), `rag.qmd` (page, not on the site yet), `rag_demo.py`, `RAG_CHAPTER_PLAN.md`, `papers/`, `fig/`, `py_src/` (figure scripts keep their `l41_`/`l42_`/`l43_` names), `results/` |
| `06_embeddings_position/` | LLM-3 Embeddings and position | same layout (`long_context_figs.py` also lives here) |
| `07_attention/` | LLM-4 Attention | same layout, plus `attention.qmd` (the older page covering LLM-4 to LLM-6) |
| `08_transformer_block/` | LLM-5 The Transformer block | same layout |
| `09_transformers_in_the_world/` | LLM-6 Transformers in the world | same layout |
| `10_pretraining/` ... `17_anatomy_2026/` | LLM-7 to LLM-14 | same layout |
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
