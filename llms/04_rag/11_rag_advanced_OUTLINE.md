# Deck 11 - RAG, part 4: beyond the standard pipeline (L44)

**Status:** draft deck built 2026-11-27 while the instructor was away ("search up on graph rag and other topics i may
want to teach and start to put them in a new deck 11_"). The usual order (interview, outline, build) was skipped on
that instruction, so everything below is open to change. 49 pages: `11_rag_advanced.tex` / `.pdf` (+ `_notes.pdf`
copy, uncommitted until taught).

## Why this deck

Video 10 ends with one frame each on agentic retrieval and graph RAG, and the inbox note of 2026-11-26 asked for graph
RAG material. Decks 08-10 already cover chunking, BM25, dense retrieval, rewriting, RRF, reranking, ColBERT, HNSW,
context assembly, evaluation, HyDE and multi-hop. This deck covers what comes after.

## Structure (as built)

| # | Section | Frames | What it teaches | Evidence |
|---|---|---|---|---|
| 0 | Cold open | 2 | Video 10's recall question; top-k gets part of the chain | **measured** on a 38-chunk archive: BM25 top-5 names 2/4 right suppliers, dense 3/4, dense needs k=15 and then has 3 wrong ones; graph search over an **LLM-built** graph: 4/4, 0 wrong, 8 chunks |
| 1 | Graph RAG | 18 | extraction, what real LLMs extract, entity resolution, local search, global questions, communities + modularity, map-reduce, GraphRAG's numbers, the bill, LazyGraphRAG, PPR (concept, by-hand 7-node example, on our graph), HippoRAG's numbers, who is who + library status, predict-first "does the graph win on facts?", GraphRAG-Bench by task, when to use | two real extractions (gpt-4o-mini $0.006, GPT-6.1 Terra $0.063); Louvain and PPR computed; papers: Edge 2024, Han 2025, Xiang 2025, Gutierrez 2024 |
| 2 | Chunks that forgot their document | 8 | two contract clauses one number apart (predict-first), contextual retrieval, Anthropic's numbers, late chunking, small-to-big, RAPTOR, which fix for which problem | **measured**: BM25 exact tie 4.07/4.07, dense prefers the wrong clause; with one context sentence the right one wins under both |
| 3 | Pipelines that check their own work | 4 | Adaptive-RAG (router is the bottleneck), CRAG, Self-RAG, deep research as the agent loop with a budget | papers |
| 4 | Long context, caching or retrieval | 4 | context windows Nov 2026 (~1M), long context vs RAG (Li 2024, Self-Route), the arithmetic by hand, CAG | papers + by-hand arithmetic |
| 5 | Attacking a RAG system | 5 | corpus as attack surface, one injected chunk (predict-first), PoisonedRAG, indirect prompt injection, defences, OWASP 2025 | **measured**: the injected chunk ranks 1st under BM25 and dense, also for a reworded question |
| - | Recap + Next | 1 | | |

## Files

- `py_src/l44_data.py` - archive (recalls, incidents, suppliers), hand-written teaching graph, contract clauses,
  poisoned chunk. Committed before any measurement (3fe0d831).
- `py_src/l44_probe_claims.py` -> `results/l44_probe.json`, `logs/l44_probe_claims.log` (~20 s, e5-small).
- `py_src/l44_extract.py` (paid, OpenRouter) -> `results/l44_extraction_*.json`, ledger `results/l44_calls.jsonl`.
- `py_src/l44_llm_graph.py` -> `results/l44_llm_graph.json` (no API).
- `py_src/l44_figs.py` -> `fig/l44_01..12_*.pdf` (~5 s).

## What the measurements changed

- The intended cold open was "top-k misses the chain". Dense top-5 already names 3 of 4, so the slide says "part of
  the chain" and makes the honest point (k=15 for all four, with three wrong suppliers along).
- The first draft compared top-k with a walk over the **hand-written** graph, which made the graph's 4/4 a built-in
  result. Replaced by two real LLM extractions. Findings worth teaching: the LLMs produce 94-99 entities (by hand: 27),
  add in-between nodes (INC-031 -> PRS-220 -> hydraulic seal -> Kotayk Hydraulics), split "RC-2019-02" from
  "2019 recall"; a fixed 2-hop walk then finds 0-1 of 4, while GraphRAG-style local search (recall nodes + neighbours
  -> their chunks) finds 4/4 with 0 wrong for both models.

## Researched, left out (say if you want them in)

- **ColPali / page-image retrieval** (ViDoRe; ColPali 81.3 vs 67.0 nDCG@5 for the best text pipeline; ~257 KB per
  page). Belongs to the planned vision-and-documents session (DECISIONS #120).
- **Agent memory** (MemGPT 2023, Mem0 2025: vendor-reported numbers only).
- **DRIFT search** (GraphRAG's mix of local and global), **HippoRAG 2 details**, **LightRAG's numbers** (its win
  rates are LLM-judge pairwise, like GraphRAG's).

## Open questions for the instructor

- Video numbering: as video 11, this deck pushes the RAG practical to 12 and the industry block (vision and
  documents, voice, shipping) to 13-15. Or should it become an optional/bonus deck?
- Scope: keep all five sections? Section 4 (long context) and section 5 (security) could also stand alone.
- Depth: add a by-hand modularity computation on a 6-node graph? Show one raw extraction JSON on a slide?
- The deck has no `rag.qmd` entry yet (the page itself is not on the site).
