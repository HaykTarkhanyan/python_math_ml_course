# Deck 11 - RAG, part 4: graphs, and chunks that lost their document (L44, video 11)

**Status:** draft deck built 2026-11-27 while the instructor was away ("search up on graph rag and other topics i may
want to teach and start to put them in a new deck 11_"). The usual order (interview, outline, build) was skipped on
that instruction. The same evening the instructor decided scope and numbering (Q2-Q4, DECISIONS #126). 37 pages:
`11_rag_advanced.tex` / `.pdf` (+ `_notes.pdf` copy, uncommitted until taught).

## Decided (2026-11-27)

- **Q2:** taught as **video 11**. The RAG practical becomes video 12, vision and documents 13, voice 14, shipping it
  15, theory from 16 (`llms/TWO_STREAMS_PROPOSAL.html` renumbered).
- **Q3:** graph RAG and the chunk fixes only. Self-checking pipelines, long context and attacks (13 frames, built and
  sourced) are parked in `parked/11_rag_advanced_sections_3-5.tex`; see `DEFERRED_TODO.md`.
- **Q4:** added a by-hand modularity frame, one ColPali frame, and a short section on agent memory.

## Why this deck

Video 10 ends with one frame each on agentic retrieval and graph RAG, and the inbox note of 2026-11-26 asked for graph
RAG material. Decks 08-10 already cover chunking, BM25, dense retrieval, rewriting, RRF, reranking, ColBERT, HNSW,
context assembly, evaluation, HyDE and multi-hop.

## Structure (as built)

| # | Section | Frames | What it teaches | Evidence |
|---|---|---|---|---|
| 0 | Cold open | 2 | Video 10's recall question; top-k gets part of the chain | **measured** on a 38-chunk archive: BM25 top-5 names 2/4 right suppliers, dense 3/4, dense needs k=15 and then has 3 wrong ones; search over an **LLM-built** graph: 4/4, 0 wrong, 8 chunks |
| 1 | Graph RAG | 19 | extraction, what real LLMs extract, entity resolution, local search, global questions, **modularity by hand**, communities, map-reduce, GraphRAG's numbers, the bill, LazyGraphRAG, PPR (concept, 7-node example by hand, on our graph), HippoRAG's numbers, who is who + library status, predict-first "does the graph win on facts?", GraphRAG-Bench by task, when to use | two real extractions (gpt-4o-mini $0.006, GPT-6.1 Terra $0.063); Louvain, modularity and PPR computed and asserted; papers: Edge 2024, Han 2025, Xiang 2025, Gutierrez 2024 |
| 2 | Chunks that forgot their document | 9 | two contract clauses one number apart (predict-first), contextual retrieval, Anthropic's numbers, late chunking, small-to-big, RAPTOR, **ColPali**, which fix for which problem | **measured**: BM25 exact tie 4.07/4.07, dense prefers the wrong clause; one context sentence makes the right one win under both |
| 3 | Retrieval as memory | 1 | an agent's memory is an index it writes itself: MemGPT, Mem0, update and delete | papers (Mem0 numbers are the vendor's own) |
| - | Recap + Next (video 12, the RAG practical) | 1 | | |

## Files

- `py_src/l44_data.py` - archive (recalls, incidents, suppliers), hand-written teaching graph, contract clauses,
  poisoned chunk. Committed before any measurement (3fe0d831).
- `py_src/l44_probe_claims.py` -> `results/l44_probe.json`, `logs/l44_probe_claims.log` (~20 s, e5-small).
- `py_src/l44_extract.py` (paid, OpenRouter) -> `results/l44_extraction_*.json`, ledger `results/l44_calls.jsonl`.
- `py_src/l44_llm_graph.py` -> `results/l44_llm_graph.json` (no API).
- `py_src/l44_figs.py` -> `fig/l44_01..13_*.pdf` (~10 s); asserts the PPR and modularity tables.
- `parked/11_rag_advanced_sections_3-5.tex` - the three parked sections, verbatim.

## What the measurements changed

- The intended cold open was "top-k misses the chain". Dense top-5 already names 3 of 4, so the slide says "part of
  the chain" and makes the honest point (k=15 for all four, with three wrong suppliers along).
- The first draft compared top-k with a walk over the **hand-written** graph, which made the graph's 4/4 a built-in
  result. Replaced by two real LLM extractions: 94-99 entities (by hand: 27), in-between nodes (INC-031 -> PRS-220 ->
  hydraulic seal -> Kotayk Hydraulics), "RC-2019-02" split from "2019 recall"; a fixed 2-hop walk finds 0-1 of 4,
  while recall nodes + neighbours -> their chunks finds 4/4 with 0 wrong for both models.

## Still open

- Instructor review of the frames themselves.
- `rag.qmd` has no deck 11 entry (the page is not on the site yet).
