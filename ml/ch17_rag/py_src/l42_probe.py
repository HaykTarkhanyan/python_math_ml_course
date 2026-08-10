"""Measurement harness for L42. Run this BEFORE writing any frame that makes a claim.

Every number the deck asserts about hybrid retrieval, keyword extraction or query
expansion is produced here first. If a measurement contradicts the intended story, the
story changes - see _learnings/2026-08-10-2016_rag-figure-contradicted-its-own-claim.md.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_probe.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS, CHUNK_LABELS
from l42_data import (EVAL_QUERIES, LLM_KEYWORDS, RAMBLING, RAMBLING_TARGET, REVISIONS,
                      REVISION_QUERY, RRF_K, bm25_scores, corpus_tokens, ranks_from_scores,
                      rrf_fuse)
from l42_keywords import keybert, rake, yake

MODEL_NAME = "intfloat/multilingual-e5-small"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_probe.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def dense_scores(model, chunk_vecs, query):
    qv = model.encode([f"query: {query}"], normalize_embeddings=True)[0]
    return chunk_vecs @ qv


def main():
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    chunk_vecs = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True)
    docs_tok = corpus_tokens()

    # ---------------------------------------------------------- 1. per-query ranks
    log.info("=" * 78)
    log.info("1. BM25 vs dense vs RRF hybrid, %d queries, %d chunks", len(EVAL_QUERIES), len(CHUNKS))
    rows = []
    for text, target, kind in EVAL_QUERIES:
        bs, ds = bm25_scores(text, docs_tok), dense_scores(model, chunk_vecs, text)
        br, dr = ranks_from_scores(bs), ranks_from_scores(ds)
        hr = ranks_from_scores(rrf_fuse([br, dr]))
        rows.append((text, target, kind, int(br[target]), int(dr[target]),
                     int(hr[target]), float(bs[target])))
        log.info("  %-56s %-10s BM25 %2d (%.3f) | dense %2d | RRF %2d",
                 text[:56], kind, br[target], bs[target], dr[target], hr[target])

    b = np.array([r[3] for r in rows], float)
    d = np.array([r[4] for r in rows], float)
    h = np.array([r[5] for r in rows], float)
    for name, arr in (("BM25", b), ("dense", d), ("RRF hybrid", h)):
        log.info("  %-11s  MRR %.3f | top-1 %2d/%d | top-3 %2d/%d | mean rank %.2f | worst %d",
                 name, np.mean(1 / arr), int((arr == 1).sum()), len(arr),
                 int((arr <= 3).sum()), len(arr), arr.mean(), int(arr.max()))
    log.info("  disagreements (|BM25 rank - dense rank| >= 3): %d of %d",
             int((np.abs(b - d) >= 3).sum()), len(rows))
    log.info("  BM25 scored EXACTLY 0 on %d queries", int(sum(1 for r in rows if r[6] < 1e-9)))

    # ---------------------------------------------------------- 1b. fusing TRUNCATED lists
    # Production hybrid search fuses two top-k lists, not two complete rankings. That
    # difference matters here, because BM25's tail below its top-k is pure tie-breaking
    # noise (its score there is 0) and full-list RRF lets that noise vote.
    log.info("=" * 78)
    log.info("1b. RRF over truncated top-k lists (0 contribution outside a list's top-k)")

    def rrf_trunc(rank_arrays, cut, k=RRF_K):
        total = np.zeros(len(CHUNKS))
        for r in rank_arrays:
            total += np.where(r <= cut, 1.0 / (k + r), 0.0)
        return total

    per_q = []
    for text, target, kind in EVAL_QUERIES:
        bs, ds = bm25_scores(text, docs_tok), dense_scores(model, chunk_vecs, text)
        per_q.append((target, ranks_from_scores(bs), ranks_from_scores(ds)))

    for cut in (3, 5, 10, len(CHUNKS)):
        arr = np.array([ranks_from_scores(rrf_trunc([br, dr], cut))[t] for t, br, dr in per_q],
                       float)
        log.info("  fuse top-%-2d  MRR %.3f | top-1 %2d/%d | mean rank %.2f | worst %2d | "
                 "paraphrase rank %d",
                 cut, np.mean(1 / arr), int((arr == 1).sum()), len(arr), arr.mean(),
                 int(arr.max()), int(arr[1]))

    # ---------------------------------------------------------- 2. keyword extraction
    log.info("=" * 78)
    log.info("2. keyword extraction on the rambling question")
    rk, rake_table, degfreq = rake(RAMBLING)
    log.info("  RAKE top-5: %s", [(p, round(s, 2)) for p, s in rk])
    log.info("  RAKE word scores (top 8): %s", [(w, round(s, 2)) for w, s in rake_table[:8]])
    yk, yterm, yparts = yake(RAMBLING)
    log.info("  YAKE top-5 (lower better): %s", [(p, round(s, 4)) for p, s in yk])
    kb, _ = keybert(RAMBLING, model)
    log.info("  KeyBERT top-5: %s", [(p, round(s, 3)) for p, s in kb])
    log.info("  LLM (authored, no API key): %s", LLM_KEYWORDS)

    log.info("-" * 78)
    log.info("  retrieval with each keyword set (target = chunk %d: %r)",
             RAMBLING_TARGET, CHUNKS[RAMBLING_TARGET][:50])
    variants = {
        "full rambling question": RAMBLING,
        "RAKE top-3": " ".join(p for p, _ in rk[:3]),
        "YAKE top-3": " ".join(p for p, _ in yk[:3]),
        "KeyBERT top-3": " ".join(p for p, _ in kb[:3]),
        "LLM keywords": " ".join(LLM_KEYWORDS),
    }
    for name, q in variants.items():
        bs, ds = bm25_scores(q, docs_tok), dense_scores(model, chunk_vecs, q)
        br, dr = ranks_from_scores(bs), ranks_from_scores(ds)
        log.info("  %-24s BM25 rank %2d (top hit: %-18s) | dense rank %2d | q=%r",
                 name, br[RAMBLING_TARGET], CHUNK_LABELS[int(np.argmax(bs))],
                 dr[RAMBLING_TARGET], q[:60])

    # ---------------------------------------------------------- 3. query expansion
    log.info("=" * 78)
    log.info("3. query expansion on L41's paraphrase failure")
    # The expansion terms are synonyms of the QUERY's own words. They were written without
    # looking at the target chunk - picking words out of the answer would make this test
    # meaningless. Two levels, so the deck can show that WHICH synonyms you add matters.
    base = "How warm is the room where the cheese matures?"
    generic = base + " temperature degrees heat"
    domain = generic + " chamber cellar ripening ageing"
    for name, q in (("original", base),
                    ("+ generic synonyms", generic),
                    ("+ domain synonyms", domain)):
        bs = bm25_scores(q, docs_tok)
        br = ranks_from_scores(bs)
        log.info("  %-22s BM25 rank %2d, score %.3f   q=%r", name, br[9], bs[9], q)

    # ---------------------------------------------------------- 4. score scales
    log.info("=" * 78)
    log.info("4. why you cannot add the two scores")
    q = EVAL_QUERIES[0][0]
    bs, ds = bm25_scores(q, docs_tok), dense_scores(model, chunk_vecs, q)
    log.info("  query %r", q)
    log.info("  BM25 : min %.3f max %.3f mean %.3f", bs.min(), bs.max(), bs.mean())
    log.info("  cosine: min %.3f max %.3f mean %.3f", ds.min(), ds.max(), ds.mean())
    naive = bs + ds
    log.info("  naive sum ranks target at %d (BM25 alone %d, dense alone %d)",
             ranks_from_scores(naive)[0], ranks_from_scores(bs)[0], ranks_from_scores(ds)[0])
    log.info("  top-3 by naive sum: %s",
             [CHUNK_LABELS[i] for i in np.argsort(-naive)[:3]])
    log.info("  top-3 by BM25     : %s", [CHUNK_LABELS[i] for i in np.argsort(-bs)[:3]])
    log.info("  top-3 by cosine   : %s", [CHUNK_LABELS[i] for i in np.argsort(-ds)[:3]])

    # ---------------------------------------------------------- 5. contradictory revisions
    log.info("=" * 78)
    log.info("5. two contradictory revisions, same query")
    rv = model.encode([f"passage: {r['text']}" for r in REVISIONS], normalize_embeddings=True)
    qv = model.encode([f"query: {REVISION_QUERY}"], normalize_embeddings=True)[0]
    sims = rv @ qv
    for r, s in zip(REVISIONS, sims):
        log.info("  %s (%s, %s) cosine %.4f", r["id"], r["date"], r["status"], s)
    log.info("  gap between them: %.4f", abs(sims[0] - sims[1]))

    log.info("=" * 78)
    log.info("RRF constant k = %d", RRF_K)


if __name__ == "__main__":
    main()
