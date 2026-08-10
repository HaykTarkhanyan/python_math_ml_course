"""Measure every quantitative claim L43 wants to make, BEFORE any slide is written.

Written for the same reason as probe_retrieval_claims.py: the previous deck in this chapter
specified a figure to prove a claim, then measured the opposite. So nothing goes on a slide
here until it has appeared in this log.

What it measures:
  1. cold open      - what the press-pressure query actually retrieves, with scores
  2. token budget   - real tiktoken counts for the corpus chunks
  3. duplication    - cosine and token overlap between near-duplicate revisions
  4. answer rel.    - real e5 cosines for the RAGAS answer-relevance construction
  5. eval-set trap  - lexical overlap and retrieval quality, copied vs asked questions
  6. HyDE           - rank of the gold chunk from the question vs from a fake answer
  7. multi-hop      - can one retrieval call get both chunks the question needs

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l43_probe_claims.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from l41_data import CHUNK_LABELS
from l43_data import (
    AR_GOOD_ANSWER, AR_GOOD_BACKQUESTIONS, AR_QUESTION, AR_VAGUE_ANSWER,
    AR_VAGUE_BACKQUESTIONS, COLD_OPEN_QUERY, CORPUS, EVAL_SET, HYDE_DOCUMENT, HYDE_QUERY,
    IDX_CELLAR, IDX_PRESS_ANSWER, IDX_PRESS_FAULT, MULTIHOP_NEEDED, MULTIHOP_QUERY,
    REVISION_CHUNKS, tokenize,
)

SEED = 509
MODEL_NAME = "intfloat/multilingual-e5-small"

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l43_probe_claims.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "at", "in", "on", "of", "for", "to", "is", "are", "was", "were", "be",
    "and", "or", "what", "how", "when", "where", "which", "does", "do", "did", "it", "its",
    "must", "should", "with", "before", "after", "into", "by", "that", "this", "we", "us",
    "so", "much", "long", "small", "up", "run", "used", "use", "there", "than", "from",
}


def bm25_scores(query, docs_tokens, k1=1.5, b=0.75):
    """BM25 with the Lucene idf variant, identical to probe_retrieval_claims.py."""
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
    return int((np.argsort(-scores) == target).nonzero()[0][0]) + 1


def content_tokens(text):
    return {t for t in tokenize(text) if t not in STOPWORDS}


def jaccard(a, b):
    ta, tb = content_tokens(a), content_tokens(b)
    if not (ta | tb):
        raise ValueError("empty token sets - refusing to divide by zero")
    return len(ta & tb) / len(ta | tb)


def main():
    np.random.seed(SEED)
    labels = CHUNK_LABELS + ["rev.2 (2019)", "rev.4 (2024)"]
    if len(labels) != len(CORPUS):
        raise ValueError(f"label/corpus mismatch: {len(labels)} vs {len(CORPUS)}")

    model = SentenceTransformer(MODEL_NAME)
    docs_tokens = [tokenize(c) for c in CORPUS]
    doc_vecs = model.encode([f"passage: {c}" for c in CORPUS],
                            normalize_embeddings=True, show_progress_bar=False)

    def dense(query):
        qv = model.encode([f"query: {query}"], normalize_embeddings=True,
                          show_progress_bar=False)[0]
        return doc_vecs @ qv

    # ---------------------------------------------------------------- 1. cold open
    log.info("=" * 78)
    log.info("1. COLD OPEN - what does %r actually retrieve?", COLD_OPEN_QUERY)
    bs = bm25_scores(COLD_OPEN_QUERY, docs_tokens)
    ds = dense(COLD_OPEN_QUERY)
    for name, sc in (("BM25", bs), ("dense", ds)):
        order = np.argsort(-sc)[:5]
        log.info("  %s top-5:", name)
        for r, i in enumerate(order, 1):
            log.info("    %d. [%2d] %-20s %.3f  %s", r, i, labels[i], sc[i], CORPUS[i][:70])
        log.info("  %s: answer chunk rank %d, fault chunk rank %d",
                 name, rank_of(sc, IDX_PRESS_ANSWER), rank_of(sc, IDX_PRESS_FAULT))

    # ---------------------------------------------------------------- 2. token budget
    log.info("=" * 78)
    log.info("2. TOKEN BUDGET - real tiktoken (cl100k_base) counts")
    enc = tiktoken.get_encoding("cl100k_base")
    counts = np.array([len(enc.encode(c)) for c in CORPUS])
    log.info("  chunk tokens: min %d, median %.1f, mean %.1f, max %d, total %d",
             counts.min(), float(np.median(counts)), counts.mean(), counts.max(), counts.sum())
    for i in np.argsort(-counts)[:3]:
        log.info("    longest: [%2d] %-20s %3d tokens", i, labels[i], counts[i])
    log.info("  whole 20-chunk corpus = %d tokens", counts.sum())

    # ---------------------------------------------------------------- 3. duplication
    log.info("=" * 78)
    log.info("3. DUPLICATION - two revisions of the same procedure")
    rev_vecs = model.encode([f"passage: {c}" for c in REVISION_CHUNKS],
                            normalize_embeddings=True, show_progress_bar=False)
    log.info("  cosine(rev2, rev4)          = %.3f", float(rev_vecs[0] @ rev_vecs[1]))
    log.info("  token overlap (Jaccard)     = %.3f", jaccard(*REVISION_CHUNKS))
    log.info("  cosine(rev4, original ch.0) = %.3f",
             float(rev_vecs[1] @ doc_vecs[IDX_PRESS_ANSWER]))
    log.info("  cosine(rev2, original ch.0) = %.3f",
             float(rev_vecs[0] @ doc_vecs[IDX_PRESS_ANSWER]))
    # a genuinely unrelated pair, for scale
    log.info("  cosine(ch.0, ch.10 gloves)  = %.3f",
             float(doc_vecs[IDX_PRESS_ANSWER] @ doc_vecs[10]))

    # ---------------------------------------------------------------- 4. answer relevance
    log.info("=" * 78)
    log.info("4. ANSWER RELEVANCE - cosine(original question, question rebuilt from answer)")
    qv = model.encode([f"query: {AR_QUESTION}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    for tag, backqs in (("good ", AR_GOOD_BACKQUESTIONS), ("vague", AR_VAGUE_BACKQUESTIONS)):
        bv = model.encode([f"query: {q}" for q in backqs], normalize_embeddings=True,
                          show_progress_bar=False)
        sims = bv @ qv
        log.info("  %s answer -> sims %s  mean %.3f",
                 tag, np.round(sims, 3).tolist(), float(sims.mean()))

    # ---------------------------------------------------------------- 5. eval-set trap
    log.info("=" * 78)
    log.info("5. EVAL-SET TRAP - questions copied from the chunk vs questions as asked")
    results = {}
    for kind, col in (("copied", 1), ("asked", 2)):
        ovl, bm_rank, de_rank = [], [], []
        for row in EVAL_SET:
            gold, q = row[0], row[col]
            ovl.append(jaccard(q, CORPUS[gold]))
            bm_rank.append(rank_of(bm25_scores(q, docs_tokens), gold))
            de_rank.append(rank_of(dense(q), gold))
        results[kind] = (np.array(ovl), np.array(bm_rank), np.array(de_rank))
        log.info("  %-6s overlap mean %.3f | BM25 ranks %s | dense ranks %s",
                 kind, np.mean(ovl), bm_rank, de_rank)
        for k in (1, 3):
            log.info("         recall@%d  BM25 %d/%d   dense %d/%d", k,
                     int(np.sum(np.array(bm_rank) <= k)), len(EVAL_SET),
                     int(np.sum(np.array(de_rank) <= k)), len(EVAL_SET))

    # ---------------------------------------------------------------- 6. HyDE
    log.info("=" * 78)
    log.info("6. HyDE - %r", HYDE_QUERY)
    ds_q = dense(HYDE_QUERY)
    log.info("  plain question              : rank %d, score %.3f",
             rank_of(ds_q, IDX_CELLAR), ds_q[IDX_CELLAR])
    for prefix in ("query", "passage"):
        hv = model.encode([f"{prefix}: {HYDE_DOCUMENT}"], normalize_embeddings=True,
                          show_progress_bar=False)[0]
        hs = doc_vecs @ hv
        log.info("  hypothetical doc (%-7s): rank %d, score %.3f", prefix + ":",
                 rank_of(hs, IDX_CELLAR), hs[IDX_CELLAR])
    bs_q = bm25_scores(HYDE_QUERY, docs_tokens)
    bs_h = bm25_scores(HYDE_DOCUMENT, docs_tokens)
    log.info("  BM25 question   : rank %d, score %.3f", rank_of(bs_q, IDX_CELLAR), bs_q[IDX_CELLAR])
    log.info("  BM25 fake answer: rank %d, score %.3f", rank_of(bs_h, IDX_CELLAR), bs_h[IDX_CELLAR])

    # ---------------------------------------------------------------- 7. multi-hop
    log.info("=" * 78)
    log.info("7. MULTI-HOP - %r", MULTIHOP_QUERY)
    for name, sc in (("BM25", bm25_scores(MULTIHOP_QUERY, docs_tokens)),
                     ("dense", dense(MULTIHOP_QUERY))):
        ranks = [rank_of(sc, i) for i in MULTIHOP_NEEDED]
        log.info("  %s ranks of the two needed chunks: %s (top-5 contains both: %s)",
                 name, ranks, all(r <= 5 for r in ranks))
        order = np.argsort(-sc)[:5]
        log.info("    top-5: %s", [labels[i] for i in order])
    # second hop: once you know it is the PRS-400, ask again
    hop2 = "What pressure does the Lori press operate at?"
    sc2 = dense(hop2)
    log.info("  after rewriting to %r: answer chunk rank %d", hop2,
             rank_of(sc2, IDX_PRESS_ANSWER))

    log.info("=" * 78)
    log.info("done - nothing above may be rounded up on a slide")


if __name__ == "__main__":
    main()
