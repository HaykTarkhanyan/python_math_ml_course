"""Second probe for L43: the retrieval-metric arithmetic and the token budget.

Separate from l43_probe_claims.py because none of this needs the embedding model except
the ranked lists, which are read back from the first probe's measured output.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l43_probe_metrics.py
"""

import logging
import re
from pathlib import Path

import numpy as np
import tiktoken

from l43_data import CORPUS

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l43_probe_metrics.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# Measured by l43_probe_claims.py, run 2026-08-10, on the 20-chunk corpus with
# intfloat/multilingual-e5-small. Rank of the gold chunk for each of the 8 eval questions.
DENSE_RANKS_ASKED = [10, 2, 5, 6, 1, 4, 1, 1]
BM25_RANKS_ASKED = [3, 2, 7, 11, 13, 20, 15, 18]
DENSE_RANKS_COPIED = [1, 1, 1, 1, 1, 1, 1, 1]

# Cold-open ranking, dense, measured: rev.4 / rev.2 / chunk 0 / brining / PRS-380.
# Graded relevance assigned by one rule, stated on the slide:
#   3 = current and complete, 2 = part of the answer, 1 = right topic but superseded, 0 = no
COLD_OPEN_GAINS_RETURNED = [3, 1, 2, 0, 0]
COLD_OPEN_LABELS = ["rev.4 (2024)", "rev.2 (2019)", "press 2.5 bar", "brining", "PRS-380"]

# The 5 retrieved chunks of the cold open, for the context-relevance count.
COLD_OPEN_CONTEXT_IDX = [19, 18, 0, 2, 15]
# Which sentences in that context a human marks as needed to answer the question.
# Sentence indices are (chunk index, sentence index within chunk).
COLD_OPEN_NEEDED = {(19, 0), (0, 0)}


def recall_at_k(ranks, k):
    return float(np.mean([r <= k for r in ranks]))


def mrr(ranks):
    return float(np.mean([1.0 / r for r in ranks]))


def dcg(gains):
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def sentences(text):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not parts:
        raise ValueError(f"no sentences found in {text!r}")
    return parts


def main():
    log.info("=" * 78)
    log.info("A. RECALL@K on the 8-question eval set (gold chunk rank per question)")
    for name, ranks in (("dense/asked", DENSE_RANKS_ASKED),
                        ("bm25/asked", BM25_RANKS_ASKED),
                        ("dense/copied", DENSE_RANKS_COPIED)):
        curve = [round(recall_at_k(ranks, k), 3) for k in range(1, 11)]
        log.info("  %-13s recall@1..10 = %s", name, curve)

    log.info("=" * 78)
    log.info("B. MRR (Mean Reciprocal Rank)")
    log.info("  dense/asked  MRR = %.4f", mrr(DENSE_RANKS_ASKED))
    log.info("  bm25/asked   MRR = %.4f", mrr(BM25_RANKS_ASKED))
    log.info("  dense/copied MRR = %.4f", mrr(DENSE_RANKS_COPIED))
    hand = [1, 2, 10]
    log.info("  hand example ranks %s -> reciprocals %s -> MRR = %.4f",
             hand, [round(1 / r, 4) for r in hand], mrr(hand))

    log.info("=" * 78)
    log.info("C. nDCG@5 on the cold-open ranking (linear gain, log2 discount)")
    returned = COLD_OPEN_GAINS_RETURNED
    ideal = sorted(returned, reverse=True)
    for i, (lab, g) in enumerate(zip(COLD_OPEN_LABELS, returned), start=1):
        log.info("    rank %d  %-14s gain %d  discount 1/log2(%d) = %.4f  contrib %.4f",
                 i, lab, g, i + 1, 1 / np.log2(i + 1), g / np.log2(i + 1))
    log.info("  returned gains %s -> DCG@5  = %.4f", returned, dcg(returned))
    log.info("  ideal    gains %s -> IDCG@5 = %.4f", ideal, dcg(ideal))
    log.info("  nDCG@5 = %.4f", dcg(returned) / dcg(ideal))

    log.info("=" * 78)
    log.info("D. CONTEXT RELEVANCE on the cold-open context (RAGAS sentence count)")
    total = 0
    kept = 0
    for ci in COLD_OPEN_CONTEXT_IDX:
        sents = sentences(CORPUS[ci])
        for si, s in enumerate(sents):
            total += 1
            mark = "NEEDED" if (ci, si) in COLD_OPEN_NEEDED else "      "
            if (ci, si) in COLD_OPEN_NEEDED:
                kept += 1
            log.info("    [%2d.%d] %s %s", ci, si, mark, s[:66])
    log.info("  context relevance = %d / %d = %.3f", kept, total, kept / total)

    log.info("=" * 78)
    log.info("E. TOKEN BUDGET on a real technical document (tiktoken cl100k_base)")
    doc = Path(__file__).resolve().parent.parent / "papers" / "llm_readable" / \
        "01_bm25_robertson_zaragoza_2009.txt"
    if not doc.exists():
        raise FileNotFoundError(doc)
    text = doc.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("cl100k_base")
    log.info("  document: %s", doc.name)
    log.info("  characters %d, tokens %d", len(text), len(enc.encode(text)))
    for size in (500, 800, 1200):
        chunks = [text[i:i + size] for i in range(0, len(text), size)]
        counts = np.array([len(enc.encode(c)) for c in chunks])
        log.info("  chunk size %4d chars -> %4d chunks, tokens per chunk: "
                 "mean %.1f, median %.1f, p90 %.1f, max %d",
                 size, len(chunks), counts.mean(), float(np.median(counts)),
                 float(np.percentile(counts, 90)), counts.max())
        if size == 800:
            for budget in (1000, 2000, 4000):
                cum = np.cumsum(counts)
                fit = int(np.sum(cum <= budget))
                log.info("      budget %5d tokens -> %2d chunks fit "
                         "(%.1f%% of the document)", budget, fit, 100 * fit / len(chunks))

    log.info("=" * 78)
    log.info("F. AGGREGATION - what a top-k retriever can see")
    for n_reports, n_relevant, k in ((400, 37, 5), (400, 37, 20)):
        log.info("  %d reports, %d mention the press, top-%d retrieved -> the model sees at "
                 "most %d of them (%.0f%% of the true count)",
                 n_reports, n_relevant, k, min(k, n_relevant),
                 100 * min(k, n_relevant) / n_relevant)

    log.info("=" * 78)
    log.info("done")


if __name__ == "__main__":
    main()
