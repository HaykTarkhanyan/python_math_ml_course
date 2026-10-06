"""Model-backed figures for L43 (generation and evaluation): 01, 03, 10, 12, 13.

Every number here is measured at run time with intfloat/multilingual-e5-small over the
20-chunk cheese-factory corpus (L41's 18 chunks plus two revisions of one procedure).

  01  REAL - what the cold-open query actually retrieves, with scores.
  03  REAL - cosine similarity between near-duplicate chunks, and the floor set by a pair
             that has nothing to do with each other.
  10  REAL - answer relevance: cosine between the original question and the questions you
             get back from a good answer vs a vague one.
  12  REAL - the evaluation-set trap: word overlap and recall@1 for questions copied from
             a chunk vs questions phrased the way a user asks them.
  13  REAL - HyDE: rank of the correct chunk when searching with the question vs with a
             fabricated answer to that question.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l43_eval_figs.py
"""

import logging
import time
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from l41_data import CHUNK_LABELS
from l43_data import (
    AR_GOOD_BACKQUESTIONS, AR_QUESTION, AR_VAGUE_BACKQUESTIONS, COLD_OPEN_QUERY, CORPUS,
    EVAL_SET, HYDE_DOCUMENT, HYDE_QUERY, IDX_CELLAR, IDX_PRESS_ANSWER, REVISION_CHUNKS,
    tokenize,
)

SEED = 509
MODEL_NAME = "intfloat/multilingual-e5-small"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l43_eval_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
    "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})

LABELS = CHUNK_LABELS + ["rev.4 (2024)", "rev.2 (2019)"]  # order fixed below
STOPWORDS = {
    "the", "a", "an", "at", "in", "on", "of", "for", "to", "is", "are", "was", "were", "be",
    "and", "or", "what", "how", "when", "where", "which", "does", "do", "did", "it", "its",
    "must", "should", "with", "before", "after", "into", "by", "that", "this", "we", "us",
    "so", "much", "long", "small", "up", "run", "used", "use", "there", "than", "from",
}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
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


def content_tokens(text):
    return {t for t in tokenize(text) if t not in STOPWORDS}


def jaccard(a, b):
    ta, tb = content_tokens(a), content_tokens(b)
    if not (ta | tb):
        raise ValueError("empty token sets - refusing to divide by zero")
    return len(ta & tb) / len(ta | tb)


# --- figure 01 -------------------------------------------------------------------------
def fig_cold_open(model, doc_vecs, labels):
    qv = model.encode([f"query: {COLD_OPEN_QUERY}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    scores = doc_vecs @ qv
    order = np.argsort(-scores)[:5]
    log.info("fig01: top-5 = %s", [(labels[i], round(float(scores[i]), 3)) for i in order])

    press_like = {19, 18, IDX_PRESS_ANSWER}
    colors = [ARM_RED if i in press_like else GREY for i in order]
    fig, ax = plt.subplots(figsize=(4.3, 1.5), layout="constrained")   # drawn at slot size
    y = np.arange(len(order))[::-1]
    bars = ax.barh(y, [scores[i] for i in order], color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("cosine similarity to the question   (red: chunks about the press)")
    ax.set_title(f'"{COLD_OPEN_QUERY}"')
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "l43_01_cold_open_retrieval")


# --- figure 03 -------------------------------------------------------------------------
def fig_dedup(model, doc_vecs, labels):
    rev = model.encode([f"passage: {c}" for c in REVISION_CHUNKS],
                       normalize_embeddings=True, show_progress_bar=False)
    pairs = [
        ("rev.2 (2019)  vs  rev.4 (2024)", float(rev[0] @ rev[1]), ARM_RED),
        ("rev.2 (2019)  vs  original chunk", float(rev[0] @ doc_vecs[IDX_PRESS_ANSWER]),
         ARM_RED),
        ("rev.4 (2024)  vs  original chunk", float(rev[1] @ doc_vecs[IDX_PRESS_ANSWER]),
         ARM_RED),
        ("press chunk  vs  safety gloves", float(doc_vecs[IDX_PRESS_ANSWER] @ doc_vecs[10]),
         ARM_BLUE),
    ]
    for name, val, _ in pairs:
        log.info("fig03: %-36s %.3f", name, val)
    if (round(pairs[0][1], 3), round(pairs[3][1], 3)) != (0.952, 0.767):
        raise ValueError(f"dedup similarities changed: {pairs[0][1]:.3f}, {pairs[3][1]:.3f}; the slides quote 0.952 and 0.767")

    fig, ax = plt.subplots(figsize=(3.97, 1.45), layout="constrained")   # drawn at slot size
    y = np.arange(len(pairs))[::-1]
    bars = ax.barh(y, [v for _, v, _ in pairs], color=[c for _, _, c in pairs], height=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=6.5, label_type="center", color="white", fontweight="bold")
    ax.axvline(0.90, color=GREEN, lw=1.2, ls="--", ymax=0.84)
    ax.text(0.89, len(pairs) - 0.42, 'a "drop near-duplicates" cut at 0.90',
            color=GREEN, fontsize=6.5, va="bottom", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in pairs])
    ax.set_xlim(0, 1.14)
    ax.set_ylim(-0.7, 3.85)
    ax.set_xlabel("cosine similarity")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "l43_03_dedup_similarity")


# --- figure 10 ------------------------------------------------------------------------
def fig_answer_relevance(model):
    qv = model.encode([f"query: {AR_QUESTION}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    out = {}
    for tag, backqs in (("good answer", AR_GOOD_BACKQUESTIONS),
                        ("vague answer", AR_VAGUE_BACKQUESTIONS)):
        bv = model.encode([f"query: {q}" for q in backqs], normalize_embeddings=True,
                          show_progress_bar=False)
        sims = np.asarray(bv @ qv, dtype=float)
        out[tag] = sims
        log.info("fig10: %-13s sims %s mean %.3f", tag, np.round(sims, 3).tolist(),
                 sims.mean())
    got = (round(float(out["good answer"].mean()), 3), round(float(out["vague answer"].mean()), 3))
    if got != (0.948, 0.790):
        raise ValueError(f"answer-relevance means changed: {got}; the slides quote 0.948 and 0.790")

    fig, ax = plt.subplots(figsize=(3.31, 1.5))   # drawn at slot size
    x = np.arange(3)
    width = 0.36
    b1 = ax.bar(x - width / 2, out["good answer"], width, color=ARM_BLUE,
                label="from the good answer")
    b2 = ax.bar(x + width / 2, out["vague answer"], width, color=ARM_ORANGE,
                label="from the vague answer")
    # inside the bar tops: on top they sat on the dashed mean lines
    ax.bar_label(b1, fmt="%.3f", fontsize=5.8, padding=-7, color="white")
    ax.bar_label(b2, fmt="%.3f", fontsize=5.8, padding=-7, color="#222222")
    for tag, color, off in (("good answer", ARM_BLUE, -0.18), ("vague answer", ARM_ORANGE, 0.18)):
        m = float(out[tag].mean())
        ax.axhline(m, color=color, ls="--", lw=0.9)
        ax.text(1.005, m, f"mean {m:.3f}", color=color, fontsize=6.5, va="center",
                fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3"])
    ax.set_xlabel("reconstructed question")
    ax.set_ylim(0, 1.05)   # from zero: a vague answer still scores 0.79 (student review)
    ax.set_ylabel("cosine to the question")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "l43_10_answer_relevance")


# --- figure 12 ------------------------------------------------------------------------
def fig_eval_trap(model, doc_vecs, docs_tokens):
    res = {}
    for kind, col in (("copied from\nthe chunk", 1), ("asked by\na user", 2)):
        ovl, dranks, branks = [], [], []
        for row in EVAL_SET:
            gold, q = row[0], row[col]
            ovl.append(jaccard(q, CORPUS[gold]))
            qv = model.encode([f"query: {q}"], normalize_embeddings=True,
                              show_progress_bar=False)[0]
            dranks.append(rank_of(doc_vecs @ qv, gold))
            branks.append(rank_of(bm25_scores(q, docs_tokens), gold))
        res[kind] = (float(np.mean(ovl)),
                     float(np.mean([r == 1 for r in dranks])),
                     float(np.mean([r == 1 for r in branks])))
        log.info("fig12: %-22s overlap %.3f  dense recall@1 %.3f  bm25 recall@1 %.3f",
                 kind.replace("\n", " "), *res[kind])
    asked_overlap = res["asked by\na user"][0]
    if round(asked_overlap, 3) != 0.008:
        raise ValueError(f"asked-question overlap changed: {asked_overlap:.3f}; the slide quotes 0.008")

    kinds = list(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.52, 1.5))   # drawn at slot size

    bars = ax1.bar(kinds, [res[k][0] for k in kinds], color=[ARM_ORANGE, ARM_BLUE],
                   width=0.5)
    ax1.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1.5)
    ax1.set_ylim(0, 0.58)
    ax1.set_ylabel("word overlap\nwith the chunk")
    ax1.set_title("Do the question and the chunk share words?", fontsize=7)

    x = np.arange(2)
    width = 0.34
    b1 = ax2.bar(x - width / 2, [res[k][1] for k in kinds], width, color=ARM_BLUE,
                 label="embeddings")
    b2 = ax2.bar(x + width / 2, [res[k][2] for k in kinds], width, color=ARM_RED,
                 label="BM25")
    ax2.bar_label(b1, fmt="%.2f", fontsize=6.5, padding=1)
    ax2.bar_label(b2, fmt="%.2f", fontsize=6.5, padding=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(kinds)
    ax2.set_ylim(0, 1.40)
    ax2.set_ylabel("recall@1")
    ax2.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    ax2.set_title("Same 8 facts, same corpus", fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "l43_12_eval_set_trap")


# --- figure 13 ------------------------------------------------------------------------
def fig_hyde(model, doc_vecs, docs_tokens, labels):
    qv = model.encode([f"query: {HYDE_QUERY}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    hv = model.encode([f"query: {HYDE_DOCUMENT}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    d_q, d_h = doc_vecs @ qv, doc_vecs @ hv
    b_q = bm25_scores(HYDE_QUERY, docs_tokens)
    b_h = bm25_scores(HYDE_DOCUMENT, docs_tokens)

    rows = [
        ("embeddings", "search with the question", rank_of(d_q, IDX_CELLAR), ARM_BLUE),
        ("embeddings", "search with the fake answer", rank_of(d_h, IDX_CELLAR), ARM_BLUE),
        ("BM25", "search with the question", rank_of(b_q, IDX_CELLAR), ARM_RED),
        ("BM25", "search with the fake answer", rank_of(b_h, IDX_CELLAR), ARM_RED),
    ]
    for meth, what, r, _ in rows:
        log.info("fig13: %-11s %-28s rank %2d", meth, what, r)
    log.info("fig13: BM25 score with question %.3f, with fake answer %.3f",
             b_q[IDX_CELLAR], b_h[IDX_CELLAR])
    if rows[0][2] != 4 or round(float(b_h[IDX_CELLAR]), 3) != 3.857:
        raise ValueError(f"HyDE numbers changed: dense rank {rows[0][2]}, BM25 {b_h[IDX_CELLAR]:.3f}; the slide quotes rank 4 and 3.857")

    fig, ax = plt.subplots(figsize=(4.08, 1.5), layout="constrained")   # drawn at slot size
    y = np.arange(len(rows))[::-1]
    bars = ax.barh(y, [r for _, _, r, _ in rows], color=[c for _, _, _, c in rows],
                   height=0.55)
    ax.bar_label(bars, labels=[f"rank {r}" for _, _, r, _ in rows], fontsize=6.5,
                 padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}\n{w}" for m, w, _, _ in rows], fontsize=6)
    ax.set_xlim(0, 23)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("rank of the correct chunk among 20 (shorter is better)")
    ax.set_title(f'"{HYDE_QUERY}"')
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "l43_13_hyde")


def main():
    np.random.seed(SEED)
    labels = CHUNK_LABELS + ["rev.2 (2019)", "rev.4 (2024)"]
    if len(labels) != len(CORPUS):
        raise ValueError(f"label/corpus mismatch: {len(labels)} vs {len(CORPUS)}")

    model = SentenceTransformer(MODEL_NAME)
    doc_vecs = model.encode([f"passage: {c}" for c in CORPUS],
                            normalize_embeddings=True, show_progress_bar=False)
    docs_tokens = [tokenize(c) for c in CORPUS]

    fig_cold_open(model, doc_vecs, labels)
    fig_dedup(model, doc_vecs, labels)
    fig_answer_relevance(model)
    fig_eval_trap(model, doc_vecs, docs_tokens)
    fig_hyde(model, doc_vecs, docs_tokens, labels)
    log.info("all model-backed figures written")


if __name__ == "__main__":
    main()
