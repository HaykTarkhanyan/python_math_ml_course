# Put the RAG chapter page on the site: video 08 already links it

**Bring this up at the start of the next session here.**

Video [08] (RAG lecture 1, retrieval: chunking, BM25, embeddings; deck `llms/04_rag/08_rag_retrieval`) was taught on
2026-11-27 and uploaded as https://youtu.be/s7MCThtCnfo (unlisted until Hayk makes it public). Hayk chose the RAG
chapter page as its materials link, so the YouTube description points at
https://hayktarkhanyan.github.io/python_math_ml_course/llms/04_rag/rag.html, which is a 404 today. Source: Hayk's
answer in the timestamp_generation_agentic session of 2026-11-27 ("Page URL + inbox note").

What was checked on 2026-11-27 ~22:00:
- `llms/04_rag/rag.qmd` exists but is not in `_quarto.yml` (local and origin/main); the site's LLM menu stops at
  `llms/03_tools_agents/tools_agents.qmd`.
- The page still says the chapter "has not been taught yet" and numbers the decks 07/08/09; the videos are numbered
  in teaching order, so retrieval is video 08 (`08_rag_retrieval`), then presumably 09 and 10.
- `llms/04_rag/08_rag_retrieval_notes.pdf` (the annotated notes from the lecture, saved 18:15) is untracked; the
  tools page links each lesson's `_notes.pdf`, so it likely belongs in git and on the page.

To do:
1. Add `llms/04_rag/rag.qmd` to `_quarto.yml` after the tools page, render, push, and check that the URL above loads.
2. Update the page for a taught lecture: the video link (above), slides and notes PDFs, and drop or reword the
   "not taught yet" warning for lecture 1.

When handled, move this file to `_claude_inbox/done/` with one line on what was done.
