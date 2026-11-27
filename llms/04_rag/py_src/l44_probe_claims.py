"""Measure every quantitative claim L44 (deck 11) wants to make, BEFORE any slide is written.

Same discipline as l43_probe_claims.py: the inputs in l44_data.py were committed before this
ran, and nothing goes on a slide until it has appeared in this log and in the JSON.

What it measures:
  1. coverage     - for the recall question, how many of the four right suppliers (and of
                    the six evidence chunks) BM25 and dense retrieval put in their top k
  2. traversal    - what walking the hand-written graph from the recall nodes returns
  3. communities  - Louvain communities of that graph (what a global search would summarise)
  4. PPR          - Personalized PageRank from the recall nodes: do the right suppliers
                    outrank the distractors?
  5. contextual   - the two contract clauses, bare and with their context prefix
  6. poisoning    - rank of one injected chunk for the target question and a paraphrase
  7. tokens       - corpus size in tokens, for the long-context arithmetic

Writes: logs/l44_probe_claims.log and llms/04_rag/results/l44_probe.json
Run (~20 s on CPU, the embedder must be in the HF cache):
    USE_TF=0 ./ma/.venv/Scripts/python.exe llms/04_rag/py_src/l44_probe_claims.py
"""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import networkx as nx
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from l44_data import (
    ARCHIVE_IDS, ARCHIVE_OFFSET, CONTRACT_CLAUSES, CONTRACT_CONTEXT, CONTRACT_GOLD,
    CONTRACT_QUESTION, CORPUS, NODE_TYPE, POISON_CHUNK, POISON_QUESTION, RECALL_ANSWER,
    RECALL_DISTRACTORS, RECALL_EVIDENCE, RECALL_QUESTION, SUPPLIERS, TRIPLES,
    TRUE_ANSWER_INDEX, corpus_index, tokenize,
)

SEED = 509
MODEL_NAME = "intfloat/multilingual-e5-small"
# A user would rarely type the attacker's exact target string. Fixed before the first run.
POISON_PARAPHRASE = "How much pressure does the Lori press need?"
K_MAX = 15

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l44_probe_claims.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def bm25_scores(query, docs_tokens, k1=1.5, b=0.75):
    """BM25 with the Lucene idf variant, identical to l43_probe_claims.py."""
    n_docs = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / n_docs
    out = []
    for toks in docs_tokens:
        norm = (1 - b) + b * len(toks) / avgdl
        s = 0.0
        for term in tokenize(query):
            tf = toks.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in docs_tokens if term in d)
            s += tf / (k1 * norm + tf) * np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        out.append(s)
    return np.array(out)


def rank_of(scores, target):
    return int((np.argsort(-scores, kind="stable") == target).nonzero()[0][0]) + 1


def label(i):
    if i < ARCHIVE_OFFSET:
        return f"manual[{i}]"
    return ARCHIVE_IDS[i - ARCHIVE_OFFSET]


def build_graph():
    g = nx.MultiDiGraph()
    for h, r, t, src in TRIPLES:
        g.add_edge(h, t, relation=r, source=src)
    for n in g.nodes:
        g.nodes[n]["type"] = NODE_TYPE[n]
    return g


def main():
    t0 = time.time()
    np.random.seed(SEED)
    out = {"model": MODEL_NAME, "corpus_size": len(CORPUS), "seed": SEED}

    model = SentenceTransformer(MODEL_NAME)

    def encode_passages(texts):
        return model.encode([f"passage: {c}" for c in texts], normalize_embeddings=True,
                            show_progress_bar=False)

    def encode_query(q):
        return model.encode([f"query: {q}"], normalize_embeddings=True,
                            show_progress_bar=False)[0]

    docs_tokens = [tokenize(c) for c in CORPUS]
    doc_vecs = encode_passages(CORPUS)

    # ------------------------------------------------------------ 1. coverage
    log.info("=" * 78)
    log.info("1. COVERAGE - %r", RECALL_QUESTION)
    evidence_idx = [corpus_index(k) for k in RECALL_EVIDENCE]
    scores = {"BM25": bm25_scores(RECALL_QUESTION, docs_tokens),
              "dense": doc_vecs @ encode_query(RECALL_QUESTION)}
    out["coverage"] = {}
    for name, sc in scores.items():
        order = np.argsort(-sc, kind="stable")
        log.info("  %s top-10:", name)
        for r, i in enumerate(order[:10], 1):
            log.info("    %2d. %-12s %.3f  %s", r, label(i), sc[i], CORPUS[i][:64])
        rows = []
        for k in range(1, K_MAX + 1):
            top = order[:k]
            text = " ".join(CORPUS[i] for i in top)
            right = sorted(s for s in RECALL_ANSWER if s in text)
            wrong = sorted(s for s in RECALL_DISTRACTORS if s in text)
            ev = sorted(label(i) for i in top if i in evidence_idx)
            rows.append({"k": k, "right_suppliers": right, "distractors": wrong,
                         "evidence_chunks": ev, "all_evidence": len(ev) == len(evidence_idx)})
        for row in rows:
            if row["k"] in (3, 5, 10, 15):
                log.info("  %s k=%2d: right %d/4 %s | distractors %d %s | evidence %d/6",
                         name, row["k"], len(row["right_suppliers"]), row["right_suppliers"],
                         len(row["distractors"]), row["distractors"],
                         len(row["evidence_chunks"]))
        ranks = {label(i): rank_of(sc, i) for i in evidence_idx}
        log.info("  %s ranks of the six evidence chunks: %s", name, ranks)
        out["coverage"][name] = {"rows": rows, "evidence_ranks": ranks,
                                 "top10": [label(i) for i in order[:10]],
                                 "order": [label(i) for i in order]}

    # ------------------------------------------------------------ 2. traversal
    log.info("=" * 78)
    log.info("2. TRAVERSAL - recall -> follows -> incident -> involves -> supplier")
    g = build_graph()
    recalls = [n for n in g.nodes if g.nodes[n]["type"] == "recall"]
    found, used_chunks, paths = set(), set(), []
    for rc in recalls:
        for _, inc, d1 in g.out_edges(rc, data=True):
            if d1["relation"] != "follows":
                continue
            used_chunks.add(d1["source"])
            for _, sup, d2 in g.out_edges(inc, data=True):
                if d2["relation"] == "involves" and g.nodes[sup]["type"] == "supplier":
                    found.add(sup)
                    used_chunks.add(d2["source"])
                    paths.append([rc, inc, sup])
    for p in paths:
        log.info("  %s", " -> ".join(p))
    log.info("  found %s; right %d/4, distractors %d; chunks behind the edges: %s",
             sorted(found), len(found & RECALL_ANSWER), len(found & RECALL_DISTRACTORS),
             sorted(used_chunks))
    out["traversal"] = {"paths": paths, "found": sorted(found),
                        "chunks_used": sorted(used_chunks)}
    if found != RECALL_ANSWER:
        raise RuntimeError(f"traversal disagrees with the hand-written answer: {found}")

    # ------------------------------------------------------------ 3. communities
    log.info("=" * 78)
    log.info("3. COMMUNITIES - Louvain on the undirected graph, seed %d", SEED)
    ug = nx.Graph(g.to_undirected())
    comms = nx.community.louvain_communities(ug, seed=SEED)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c))
    q = nx.community.modularity(ug, comms)
    for i, c in enumerate(comms):
        log.info("  community %d (%d nodes): %s", i, len(c), c)
    log.info("  modularity %.3f; %d nodes, %d edges", q, ug.number_of_nodes(),
             ug.number_of_edges())
    out["communities"] = {"communities": comms, "modularity": q,
                          "nodes": ug.number_of_nodes(), "edges": ug.number_of_edges()}

    # ------------------------------------------------------------ 4. PPR
    log.info("=" * 78)
    log.info("4. PERSONALIZED PAGERANK - restart at the two recall nodes, alpha 0.85")
    pers = {n: (1.0 if n in recalls else 0.0) for n in ug.nodes}
    ppr = nx.pagerank(ug, alpha=0.85, personalization=pers)
    sup_rank = sorted(SUPPLIERS, key=lambda s: -ppr[s])
    for r, s in enumerate(sup_rank, 1):
        tag = "RIGHT" if s in RECALL_ANSWER else "distractor"
        log.info("  %d. %-18s %.4f  %s", r, s, ppr[s], tag)
    top4 = set(sup_rank[:4])
    log.info("  top-4 suppliers by PPR == the right answer: %s", top4 == RECALL_ANSWER)
    plain = nx.pagerank(ug, alpha=0.85)
    plain_rank = sorted(SUPPLIERS, key=lambda s: -plain[s])
    log.info("  plain PageRank order (no personalization), for contrast: %s", plain_rank)
    out["ppr"] = {"alpha": 0.85, "scores": {n: ppr[n] for n in ug.nodes},
                  "supplier_order": sup_rank, "top4_is_answer": top4 == RECALL_ANSWER,
                  "plain_supplier_order": plain_rank}

    # ------------------------------------------------------------ 5. contextual
    log.info("=" * 78)
    log.info("5. CONTEXTUAL - %r", CONTRACT_QUESTION)
    out["contextual"] = {}
    names = list(CONTRACT_CLAUSES)
    gold_pos = len(CORPUS) + names.index(CONTRACT_GOLD)
    for variant in ("bare", "with context"):
        extra = [CONTRACT_CLAUSES[n] if variant == "bare"
                 else f"{CONTRACT_CONTEXT[n]} {CONTRACT_CLAUSES[n]}" for n in names]
        docs = CORPUS + extra
        dt = [tokenize(c) for c in docs]
        bm = bm25_scores(CONTRACT_QUESTION, dt)
        dv = np.vstack([doc_vecs, encode_passages(extra)])
        de = dv @ encode_query(CONTRACT_QUESTION)
        rec = {}
        for name, sc in (("BM25", bm), ("dense", de)):
            clause_scores = {n: float(sc[len(CORPUS) + j]) for j, n in enumerate(names)}
            rec[name] = {"clause_scores": clause_scores, "gold_rank": rank_of(sc, gold_pos),
                         "tie": abs(clause_scores[names[0]] - clause_scores[names[1]]) < 1e-9}
            log.info("  %-12s %-5s scores %s | gold rank %d | exact tie %s", variant, name,
                     {k: round(v, 4) for k, v in clause_scores.items()},
                     rec[name]["gold_rank"], rec[name]["tie"])
        out["contextual"][variant] = rec

    # ------------------------------------------------------------ 6. poisoning
    log.info("=" * 78)
    log.info("6. POISONING - one injected chunk: %r", POISON_CHUNK)
    docs = CORPUS + [POISON_CHUNK]
    dt = [tokenize(c) for c in docs]
    dv = np.vstack([doc_vecs, encode_passages([POISON_CHUNK])])
    poison_pos = len(CORPUS)
    out["poisoning"] = {}
    for qname, qtext in (("target question", POISON_QUESTION), ("paraphrase", POISON_PARAPHRASE)):
        rec = {}
        for name, sc in (("BM25", bm25_scores(qtext, dt)), ("dense", dv @ encode_query(qtext))):
            rec[name] = {"poison_rank": rank_of(sc, poison_pos),
                         "true_rank": rank_of(sc, TRUE_ANSWER_INDEX),
                         "poison_score": float(sc[poison_pos]),
                         "true_score": float(sc[TRUE_ANSWER_INDEX])}
            log.info("  %-15s %-5s poison rank %d (%.3f) | true answer rank %d (%.3f)",
                     qname, name, rec[name]["poison_rank"], rec[name]["poison_score"],
                     rec[name]["true_rank"], rec[name]["true_score"])
        out["poisoning"][qname] = {"query": qtext, **rec}

    # ------------------------------------------------------------ 7. tokens
    log.info("=" * 78)
    enc = tiktoken.get_encoding("cl100k_base")
    toks = [len(enc.encode(c)) for c in CORPUS]
    log.info("7. TOKENS - corpus %d tokens over %d chunks (mean %.1f, max %d)",
             sum(toks), len(toks), np.mean(toks), max(toks))
    out["tokens"] = {"total": sum(toks), "mean_chunk": float(np.mean(toks)), "max_chunk": max(toks)}

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "l44_probe.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s in %.0f s", path, time.time() - t0)


if __name__ == "__main__":
    main()
