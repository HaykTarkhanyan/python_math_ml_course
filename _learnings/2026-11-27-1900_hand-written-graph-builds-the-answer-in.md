# A comparison against a hand-written structure builds the answer in

**Symptom.** Deck 11's cold open compared measured top-k retrieval (BM25 top-5: 2 of 4 right suppliers, dense 3 of
4) with a walk over a knowledge graph. The graph walk scored 4 of 4 with 0 wrong, from exactly the 6 evidence chunks.
It looked like a clean win for graph RAG.

**Cause.** The graph was written by hand from the same chunks, with canonical names and exactly the edges the answer
needs. The probe even raised an error if the walk did not return the hand-written answer. So the 4 of 4 was an
assumption, not a measurement. The session's heads-up note caught it.

**What the real thing does.** Two LLMs extracted graphs from the same 38 chunks (`py_src/l44_extract.py`, $0.0695 in
total, `results/l44_llm_graph.json`):

| | by hand | gpt-4o-mini | GPT-6.1 Terra |
|---|---|---|---|
| entities / relationships | 27 / 29 | 94 / 80 | 99 / 92 |
| 2-hop walk from the recalls, right suppliers | 4/4 | 1/4 | 0/4 |
| recall nodes + neighbours -> their chunks | 4/4, 6 chunks | 4/4, 8 chunks | 4/4, 8 chunks |

The models add in-between nodes (INC-031 -> PRS-220 -> hydraulic seal -> Kotayk Hydraulics). The stronger model
adds more of them, so a fixed-length walk finds less. Both split "RC-2019-02" from "2019 recall". Handing over the
chunks behind the nearby nodes (GraphRAG's local search) survives all of it.

**Consequences.** The deck now measures graph RAG on the LLM-built graphs and uses the hand-written graph only to
teach mechanics, labelled as hand-written. When a slide compares a measured baseline with a structured method, build
the structure the way a real tool would (an LLM, a parser) before claiming a result. It costs cents and can change
the lesson, as it did here ("the graph finds the chunks, the text carries the answer").
