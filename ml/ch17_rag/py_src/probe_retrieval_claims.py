"""Probe: do the deck's retrieval claims actually reproduce on this corpus?

Written after figure 13's first run produced the OPPOSITE of the intended teaching story.
This script measures rather than assumes. It reports, for each candidate query, both the
rank AND the raw score of the correct chunk under BM25 and under dense retrieval, so a
"rank" that is really arbitrary tie-breaking among zero scores is visible as such.

Two fixes to the experiment, both about statistical power rather than cherry-picking:
  - part-number distractors are added, so "dense blurs exact identifiers" is testable at all;
  - scores are reported, so near-zero BM25 signal is not disguised as a mid-table rank.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/probe_retrieval_claims.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS, tokenize

MODEL_NAME = "intfloat/multilingual-e5-small"

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "probe_retrieval_claims.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# Distractors so the identifier claim has something to fail on. Without near-miss part
# numbers there is nothing for an embedder to confuse, and the test is vacuous.
DISTRACTORS = [
    "Model PRS-220 was withdrawn from service after the 2019 recall.",
    "The PRS-380 press is rated for harder cheeses and higher pressures.",
    "Spare parts for the PRS-410 are no longer stocked by the supplier.",
    "Line three uses a VAC-400 vacuum unit rather than a press.",
]
CORPUS = CHUNKS + DISTRACTORS

# (label, query, index of the chunk that actually answers it)
CANDIDATES = [
    ("direct",      "What pressure should the press run at for Lori cheese?", 0),
    ("paraphrase1", "How hard does the machine squeeze the cheese?",          0),
    ("paraphrase2", "How much force is applied while the cheese is formed?",  0),
    ("paraphrase3", "What setting is used when the cheese is compacted?",     0),
    ("partnum",     "PRS-400",                                               8),
    ("partnum_ctx", "Which press replaced the old one on line two?",          8),
    ("safety",      "What should staff wear when working with the knives?",  10),
    ("temp",        "How warm is the room where the cheese matures?",         9),
]


def bm25_scores(query, docs_tokens, k1=1.5, b=0.75):
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


def main():
    log.info("corpus: %d chunks (%d original + %d distractors)",
             len(CORPUS), len(CHUNKS), len(DISTRACTORS))
    model = SentenceTransformer(MODEL_NAME)
    docs_tokens = [tokenize(c) for c in CORPUS]
    doc_vecs = model.encode([f"passage: {c}" for c in CORPUS],
                            normalize_embeddings=True, show_progress_bar=False)

    log.info("%-13s | %-24s | %-24s", "query", "BM25", "dense")
    log.info("%-13s | %-24s | %-24s", "", "rank  score  top-score", "rank  score  top-score")
    for label, query, target in CANDIDATES:
        bs = bm25_scores(query, docs_tokens)
        qv = model.encode([f"query: {query}"], normalize_embeddings=True,
                          show_progress_bar=False)[0]
        ds = doc_vecs @ qv
        log.info(
            "%-13s | %2d  %7.3f  %7.3f | %2d  %7.3f  %7.3f",
            label, rank_of(bs, target), bs[target], bs.max(),
            rank_of(ds, target), ds[target], ds.max(),
        )
        if bs.max() < 1e-6:
            log.info("%-13s   -> BM25 has NO lexical signal at all (every score 0)", "")

    log.info("")
    log.info("Reading guide: a BM25 rank is meaningless when the score column is ~0 -")
    log.info("that is arbitrary tie-breaking, not retrieval.")


if __name__ == "__main__":
    main()
