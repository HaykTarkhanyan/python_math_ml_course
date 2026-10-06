"""Numbers for two frames added after the L42 pedagogy review.

1. YAKE, by hand. The review flagged that RAKE gets real arithmetic and YAKE does not:
   "I can say 'five features, lower is better', but I never saw how they combine."
   This dumps the five features and the combination for real words of the rambling question.

2. Score normalisation, demonstrated. The review called the normalisation slide the single
   frame most needing five minutes, because three failure modes are asserted as prose with
   no numbers "on a deck that otherwise loves showing arithmetic". This measures all three
   on the real corpus.

3. Trying to make BM25 win (added after the 2026-10-06 student review, which never saw hybrid
   help). Eleven identifier questions on the 18 chunks, then the same corpus plus 19
   near-identical spare-parts notes (PRS-401 ... PRS-420). Every query tried is logged.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_worked_extras.py   (~20 s)
"""

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from l42_data import RAMBLING, bm25_scores, ranks_from_scores, rrf_fuse
from l42_keywords import _yake_term_scores
from l41_data import CHUNK_LABELS, CHUNKS, tokenize

MODEL_NAME = "intfloat/multilingual-e5-small"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_worked_extras.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})

QUERY = "What pressure should the press run at for Lori cheese?"
SHOW_WORDS = ["lori", "press", "pressure", "week"]


def bm25(query, docs_tokens, k1=1.5, b=0.75):
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


def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        raise ValueError("cannot min-max a constant vector")
    return (v - lo) / (hi - lo)


def yake_by_hand():
    scores, parts, tf = _yake_term_scores(RAMBLING)
    log.info("YAKE features on the rambling question (lower score = better keyword)")
    log.info("  %-9s %6s %6s %6s %6s %6s | %7s", "word", "case", "pos", "freq", "rel", "diff", "score")
    for w in SHOW_WORDS:
        if w not in scores:
            raise KeyError(f"{w!r} not in the rambling question")
        p = parts[w]
        log.info("  %-9s %6.3f %6.3f %6.3f %6.3f %6.3f | %7.3f",
                 w, p["case"], p["pos"], p["freq"], p["rel"], p["diff"], scores[w])
        # Reproduce the combination explicitly, so the slide can show it as one line.
        recomputed = (p["rel"] * p["pos"]) / (p["case"] + p["freq"] / p["rel"] + p["diff"] / p["rel"])
        if abs(recomputed - scores[w]) > 1e-9:
            raise ValueError(f"{w}: combination does not reproduce the score")
    log.info("  combination:  score = (rel * pos) / (case + freq/rel + diff/rel)")


def normalisation_failures(model):
    docs_tok = [tokenize(c) for c in CHUNKS]
    bm = bm25(QUERY, docs_tok)
    qv = model.encode([f"query: {QUERY}"], normalize_embeddings=True, show_progress_bar=False)[0]
    dv = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True,
                      show_progress_bar=False)
    cos = dv @ qv

    log.info("raw ranges: BM25 %.3f to %.3f | cosine %.3f to %.3f",
             bm.min(), bm.max(), cos.min(), cos.max())

    # Failure A: min-max scale depends on which documents happen to be in the set.
    keep = np.argsort(-bm)[:8]
    full = minmax(bm)
    subset = minmax(bm[keep])
    doc = int(keep[3])
    log.info("A) same chunk, different candidate set: normalised %.3f (all 18) vs %.3f (top 8)",
             full[doc], subset[3])
    second = np.argsort(-bm)[1]
    got = (round(float(full[doc]), 3), round(float(subset[3]), 3),
           round(float(minmax(bm)[second]), 3), round(float(minmax(bm.copy() * np.where(
               np.arange(len(bm)) == bm.argmax(), 3.0, 1.0))[second]), 3))
    if got != (0.394, 0.211, 0.672, 0.224):
        raise ValueError(f"normalisation numbers changed: {got}; the slides quote (0.394, 0.211, 0.672, 0.224)")

    # Failure B: one strong outlier squashes everything else.
    inflated = bm.copy()
    inflated[int(bm.argmax())] *= 3.0
    log.info("B) tripling the top score squashes the rest: 2nd place %.3f -> %.3f",
             float(minmax(bm)[np.argsort(-bm)[1]]),
             float(minmax(inflated)[np.argsort(-bm)[1]]))

    # Failure C: min-max destroys BM25's honest zero.
    zeros = int((bm == 0).sum())
    log.info("C) %d of %d chunks score EXACTLY 0.000 under BM25; after min-max they become "
             "%.3f, indistinguishable from a genuine weak match",
             zeros, len(bm), float(minmax(bm)[bm == 0][0]) if zeros else float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(4.63, 1.5))   # drawn at slot size: 0.84 width
    order = np.argsort(-bm)

    axes[0].bar(range(len(bm)), bm[order], color=ARM_ORANGE)
    axes[0].set_title("BM25, raw\n(zeros mean 'no signal')", fontsize=6.5)
    axes[0].set_ylabel("score")

    axes[1].bar(range(len(cos)), cos[order], color=ARM_BLUE)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("cosine, raw\n(never zero, narrow band)", fontsize=6.5)

    mm = minmax(bm)[order]
    colors = [ARM_RED if bm[order][i] == 0 else ARM_ORANGE for i in range(len(mm))]
    axes[2].bar(range(len(mm)), mm, color=colors)
    axes[2].set_title("BM25 after min-max\n(the zeros are now 0.00 too -\nsame number, lost meaning)",
                      fontsize=6.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_xlabel("chunks, best first")
    fig.tight_layout(w_pad=0.8)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "l42_18_normalisation.pdf"
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error("could not write %s: %s", out, e)
                raise
            log.warning("errno 22 on %s, retry %d/4", out.name, attempt)
            time.sleep(1)
    plt.close(fig)
    log.info("wrote %s", out)


IDENTIFIER_QUERIES = [  # (as a user might type it, index of the chunk that answers it)
    ("PRS-410", 16), ("PRS-410 spare parts", 16), ("can I still get parts for the PRS-410?", 16),
    ("PRS-220", 14), ("is the PRS-220 still in service?", 14),
    ("PRS-380", 15), ("which cheeses is the PRS-380 for?", 15),
    ("VAC-400", 17), ("where is the VAC-400?", 17),
    ("PRS-400", 8), ("what did the PRS-400 replace?", 8),
]
PRS410 = 16


def ranks_three_ways(model, corpus_vecs, docs, query):
    bs = bm25_scores(query, docs)
    qv = model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)[0]
    ds = corpus_vecs @ qv
    b, d = ranks_from_scores(bs), ranks_from_scores(ds)
    return b, d, ranks_from_scores(rrf_fuse([b, d])), ds


def identifier_attempts(model):
    """Attempt 1: identifier questions on the 18 chunks. Attempt 2: add near-identical notes."""
    docs = [tokenize(c) for c in CHUNKS]
    vecs = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True,
                        show_progress_bar=False)
    dense_first = 0
    for q, t in IDENTIFIER_QUERIES:
        b, d, h, _ = ranks_three_ways(model, vecs, docs, q)
        dense_first += int(d[t] == 1)
        log.info("ids  %-40s %-15s bm25 %2d dense %2d rrf %2d", q, CHUNK_LABELS[t], b[t], d[t], h[t])
        if q == "PRS-220":
            prs220 = (b[t], d[t])

    extra = [f"Spare parts for the PRS-{n} are no longer stocked by the supplier."
             for n in range(401, 421) if n != 410]
    corpus = CHUNKS + extra
    docs2 = [tokenize(c) for c in corpus]
    vecs2 = model.encode([f"passage: {c}" for c in corpus], normalize_embeddings=True,
                         show_progress_bar=False)
    margins, near_first = [], 0
    for q in ["PRS-410", "PRS-410 spare parts", "can I still get parts for the PRS-410?"]:
        b, d, h, ds = ranks_three_ways(model, vecs2, docs2, q)
        margin = float(ds[PRS410] - max(ds[i] for i in range(len(corpus)) if i != PRS410))
        margins.append(margin)
        near_first += int(d[PRS410] == 1)
        log.info("near %-40s bm25 %2d dense %2d rrf %2d  cosine margin %.4f", q, b[PRS410],
                 d[PRS410], h[PRS410], margin)

    # The frame "We tried to make BM25 win" quotes these.
    got = (len(IDENTIFIER_QUERIES), dense_first, prs220, len(extra), near_first,
           round(min(margins), 3), round(max(margins), 3))
    if got != (11, 10, (2, 2), 19, 3, 0.009, 0.018):
        raise ValueError(f"identifier numbers changed: {got}; the frame quotes "
                         "(11, 10, (2, 2), 19, 3, 0.009, 0.018)")
    log.info("identifier attempts checked: dense first on 10 of 11, then 3 of 3 against 19 "
             "near-identical notes by 0.009-0.018")


def main():
    yake_by_hand()
    log.info("")
    model = SentenceTransformer(MODEL_NAME)
    normalisation_failures(model)
    log.info("")
    identifier_attempts(model)


if __name__ == "__main__":
    main()
