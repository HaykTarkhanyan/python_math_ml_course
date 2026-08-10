"""Numbers for two frames added after the L42 pedagogy review.

1. YAKE, by hand. The review flagged that RAKE gets real arithmetic and YAKE does not:
   "I can say 'five features, lower is better', but I never saw how they combine."
   This dumps the five features and the combination for real words of the rambling question.

2. Score normalisation, demonstrated. The review called the normalisation slide the single
   frame most needing five minutes, because three failure modes are asserted as prose with
   no numbers "on a deck that otherwise loves showing arithmetic". This measures all three
   on the real corpus.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_worked_extras.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from l42_data import RAMBLING
from l42_keywords import _yake_term_scores
from l41_data import CHUNKS, tokenize

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
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
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

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    order = np.argsort(-bm)

    axes[0].bar(range(len(bm)), bm[order], color=ARM_ORANGE)
    axes[0].set_title("BM25, raw\n(zeros mean 'no signal')", fontsize=10)
    axes[0].set_ylabel("score")

    axes[1].bar(range(len(cos)), cos[order], color=ARM_BLUE)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("cosine, raw\n(never zero, narrow band)", fontsize=10)

    mm = minmax(bm)[order]
    colors = [ARM_RED if bm[order][i] == 0 else ARM_ORANGE for i in range(len(mm))]
    axes[2].bar(range(len(mm)), mm, color=colors)
    axes[2].set_title("BM25 after min-max\n(the zeros are now 0.00 too -\nsame number, lost meaning)",
                      fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_xlabel("chunks, best first")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "l42_18_normalisation.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def main():
    yake_by_hand()
    log.info("")
    model = SentenceTransformer(MODEL_NAME)
    normalisation_failures(model)


if __name__ == "__main__":
    main()
