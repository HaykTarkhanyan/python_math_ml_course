"""L42 figures 01, 05, 06, 07, 08, 09 - all REAL, all measured by this script.

  01  where BM25 and dense disagree, over 16 queries with a known target chunk
  05  what each keyword-extraction method does to retrieval of the same target
  06  query expansion rescues the paraphrase BM25 could not see
  07  BM25 and cosine live on different scales, so adding them is not fusion
  08  the RRF weight 1/(k + rank), for three values of k
  09  BM25 vs dense vs RRF hybrid on the 16-query set - the honest result

Numbers here are the ones on the slides. Anything that disagreed with the intended story
changed the story: see the provenance block of L42_rag_hybrid_rerank.tex.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_retrieval_figs.py
"""

import logging
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS, CHUNK_LABELS
from l42_data import (EVAL_QUERIES, LLM_KEYWORDS, RAMBLING, RAMBLING_TARGET, RRF_K,
                      bm25_scores, corpus_tokens, ranks_from_scores, rrf_fuse)
from l42_keywords import keybert, rake, yake

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"
MODEL_NAME = "intfloat/multilingual-e5-small"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_retrieval_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def dense(model, chunk_vecs, query):
    return chunk_vecs @ model.encode([f"query: {query}"], normalize_embeddings=True)[0]


# --- figure 01 -------------------------------------------------------------------------
def fig_win_scatter(model, chunk_vecs, docs_tok):
    pts, zero_score = [], []
    for text, target, _kind in EVAL_QUERIES:
        bs = bm25_scores(text, docs_tok)
        ds = dense(model, chunk_vecs, text)
        pts.append((int(ranks_from_scores(bs)[target]), int(ranks_from_scores(ds)[target])))
        zero_score.append(bs[target] < 1e-9)

    counts = Counter(pts)
    log.info("01 win scatter: %s", dict(counts))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    lim = 19.5
    ax.plot([0.6, lim], [0.6, lim], color=GREY, lw=1.0, ls=":", zorder=1)
    ax.text(11.0, 11.6, "both agree", fontsize=11, color=GREY, rotation=45,
            ha="center", va="bottom")

    for (bx, dy), n in counts.items():
        colour = ARM_RED if abs(bx - dy) >= 3 else ARM_BLUE
        ax.scatter([bx], [dy], s=90 + 45 * n, color=colour, alpha=0.85, zorder=3,
                   edgecolor="white", lw=1.2)

    big, nbig = max(counts.items(), key=lambda kv: kv[1])
    ax.annotate(f"{nbig} questions here\n(both found it first)",
                xy=big, xytext=(big[0] + 4.2, big[1] + 2.6), fontsize=11.5,
                color=ARM_BLUE, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE, lw=1.4))

    bad = [i for i, z in enumerate(zero_score) if z][0]
    bx, dy = pts[bad]
    ax.annotate("BM25 score 0.000\nnot one shared word",
                xy=(bx, dy), xytext=(bx - 1.6, dy + 6.0), fontsize=11.5, color=ARM_RED,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.4))

    ax.set_xlim(-1.2, lim)
    ax.set_ylim(-1.2, lim)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("rank the correct chunk got from BM25", fontsize=12)
    ax.set_ylabel("rank it got from embeddings", fontsize=12)
    ax.set_title("16 questions, 18 chunks: they agree - until one collapses",
                 fontsize=12.5)
    save(fig, "l42_01_win_scatter")
    return pts


# --- figures 05 and 06 -----------------------------------------------------------------
def fig_keyword_payoff(model, chunk_vecs, docs_tok):
    rk, _, _ = rake(RAMBLING)
    yk, _, _ = yake(RAMBLING)
    kb, _ = keybert(RAMBLING, model)
    variants = [
        ("the whole\nquestion", RAMBLING),
        ("RAKE\ntop-3", " ".join(p for p, _ in rk[:3])),
        ("YAKE\ntop-3", " ".join(p for p, _ in yk[:3])),
        ("KeyBERT\ntop-3", " ".join(p for p, _ in kb[:3])),
        ("LLM\nkeywords", " ".join(LLM_KEYWORDS)),
    ]

    br, dr, tops = [], [], []
    for name, q in variants:
        bs, ds = bm25_scores(q, docs_tok), dense(model, chunk_vecs, q)
        br.append(int(ranks_from_scores(bs)[RAMBLING_TARGET]))
        dr.append(int(ranks_from_scores(ds)[RAMBLING_TARGET]))
        tops.append(CHUNK_LABELS[int(np.argmax(bs))])
        log.info("05 %-16s BM25 rank %d (top hit %r) | dense rank %d | %r",
                 name.replace("\n", " "), br[-1], tops[-1], dr[-1], q)

    x = np.arange(len(variants))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    b1 = ax.bar(x - w / 2, br, w, color=ARM_ORANGE, label="BM25 (keywords)")
    b2 = ax.bar(x + w / 2, dr, w, color=ARM_BLUE, label="dense (embeddings)")
    ax.bar_label(b1, fmt="%d", fontsize=10.5, fontweight="bold", padding=2)
    ax.bar_label(b2, fmt="%d", fontsize=10.5, fontweight="bold", padding=2)

    for xi, t in enumerate(tops):
        colour = ARM_RED if t != "press 2.5 bar" else "#2E8B57"
        ax.text(xi - w / 2, -0.62, t, fontsize=7.4, color=colour, ha="center",
                rotation=0, va="top")
    ax.text(-1.02, -0.62, "BM25's\ntop hit:", fontsize=7.4, color=GREY, ha="left", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in variants], fontsize=9.5)
    ax.set_ylabel("rank of the chunk that answers\n(1 = found it)")
    ax.set_ylim(0, max(br + dr) + 1.1)
    ax.set_title("Extracting keywords is not free improvement", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.tick_params(axis="x", length=0)
    save(fig, "l42_05_keyword_payoff")


def fig_expansion(docs_tok):
    base = "How warm is the room where the cheese matures?"
    steps = [
        ("the question\nas typed", base),
        ("+ generic synonyms\ntemperature, degrees, heat", base + " temperature degrees heat"),
        ("+ domain synonyms\nchamber, cellar, ripening, ageing",
         base + " temperature degrees heat chamber cellar ripening ageing"),
    ]
    ranks, scores = [], []
    for name, q in steps:
        bs = bm25_scores(q, docs_tok)
        ranks.append(int(ranks_from_scores(bs)[9]))
        scores.append(float(bs[9]))
        log.info("06 %-22s BM25 rank %2d score %.3f", name.replace("\n", " "), ranks[-1], scores[-1])

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar([s[0] for s in steps], ranks,
                  color=[ARM_RED, ARM_ORANGE, "#2E8B57"], width=0.55)
    ax.bar_label(bars, labels=[f"rank {r}\nscore {s:.2f}" for r, s in zip(ranks, scores)],
                 fontsize=10, fontweight="bold", padding=3)
    ax.set_ylim(0, max(ranks) * 1.35)
    ax.set_ylabel("BM25 rank of the ripening-cellar chunk\n(1 = found it)")
    ax.set_title("The query BM25 scored 0.000 on, after adding words to it", fontsize=11)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    save(fig, "l42_06_expansion")


# --- figure 07 -------------------------------------------------------------------------
def fig_score_scales(model, chunk_vecs, docs_tok):
    q = EVAL_QUERIES[0][0]
    bs, ds = bm25_scores(q, docs_tok), dense(model, chunk_vecs, q)
    naive = bs + ds
    top_bm = [CHUNK_LABELS[i] for i in np.argsort(-bs)[:3]]
    top_sum = [CHUNK_LABELS[i] for i in np.argsort(-naive)[:3]]
    log.info("07 BM25 range [%.3f, %.3f]  cosine range [%.3f, %.3f]",
             bs.min(), bs.max(), ds.min(), ds.max())
    log.info("07 top-3 by BM25 %s | by naive sum %s | identical: %s",
             top_bm, top_sum, top_bm == top_sum)

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.scatter(bs, np.full_like(bs, 1.0), s=90, color=ARM_ORANGE, alpha=0.8,
               edgecolor="white", lw=1.0, zorder=3)
    ax.scatter(ds, np.full_like(ds, 0.35), s=90, color=ARM_BLUE, alpha=0.8,
               edgecolor="white", lw=1.0, zorder=3)
    ax.hlines([1.0, 0.35], -0.05, 3.25, color=GREY, lw=0.8, zorder=1)

    ax.annotate(f"BM25: {bs.min():.2f} to {bs.max():.2f}", (3.3, 1.0), fontsize=10,
                color=ARM_ORANGE, fontweight="bold", va="center")
    ax.annotate(f"cosine: {ds.min():.2f} to {ds.max():.2f}", (3.3, 0.35), fontsize=10,
                color=ARM_BLUE, fontweight="bold", va="center")

    ax.set_yticks([])
    ax.set_ylim(0.05, 1.42)
    ax.set_xlim(-0.15, 4.9)
    ax.set_xlabel("score for each of the 18 chunks, same query")
    ax.set_title("One score runs 0 to 3, the other lives inside a 0.15-wide band",
                 fontsize=11)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.04,
             "Adding them: the top 3 is exactly BM25's top 3. The embedder gets no vote.",
             ha="center", fontsize=9.5, color=ARM_RED, style="italic")
    save(fig, "l42_07_score_scales")


# --- figure 08 -------------------------------------------------------------------------
def fig_rrf_curve():
    ranks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for k, colour in [(0, ARM_RED), (10, ARM_ORANGE), (60, ARM_BLUE)]:
        w = 1.0 / (k + ranks)
        ax.plot(ranks, w / w[0], "o-", ms=4.5, color=colour, lw=1.8,
                label=f"k = {k}")
        log.info("08 k=%2d: weight(1)=%.4f weight(2)=%.4f ratio %.2f",
                 k, 1 / (k + 1), 1 / (k + 2), (1 / (k + 1)) / (1 / (k + 2)))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("rank in one list")
    ax.set_ylabel("weight, relative to rank 1")
    ax.set_title(r"$1/(k + \mathrm{rank})$: bigger $k$ means rank 1 is not king",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "l42_08_rrf_curve")


# --- figure 09 -------------------------------------------------------------------------
def fig_hybrid_gain(model, chunk_vecs, docs_tok):
    b, d, h = [], [], []
    for text, target, _ in EVAL_QUERIES:
        bs, ds = bm25_scores(text, docs_tok), dense(model, chunk_vecs, text)
        br, dr = ranks_from_scores(bs), ranks_from_scores(ds)
        b.append(br[target])
        d.append(dr[target])
        h.append(ranks_from_scores(rrf_fuse([br, dr]))[target])
    b, d, h = np.array(b, float), np.array(d, float), np.array(h, float)

    names = ["BM25\nalone", "embeddings\nalone", "RRF\nhybrid"]
    mrr = [np.mean(1 / a) for a in (b, d, h)]
    worst = [a.max() for a in (b, d, h)]
    for n, m, w in zip(names, mrr, worst):
        log.info("09 %-16s MRR %.3f  worst rank %d", n.replace("\n", " "), m, int(w))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    colours = [ARM_ORANGE, ARM_BLUE, "#2E8B57"]

    bars = axes[0].bar(names, mrr, color=colours, width=0.55)
    axes[0].bar_label(bars, fmt="%.3f", fontsize=10.5, fontweight="bold", padding=3)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("average quality over 16 questions\n(1.0 = always rank 1, higher is better)",
                      fontsize=9.5)
    axes[0].set_ylabel("mean reciprocal rank")

    bars = axes[1].bar(names, worst, color=colours, width=0.55)
    axes[1].bar_label(bars, fmt="%d", fontsize=10.5, fontweight="bold", padding=3)
    axes[1].set_ylim(0, max(worst) * 1.3)
    axes[1].set_title("the single worst question\n(rank of the answer, lower is better)",
                      fontsize=9.5)
    axes[1].set_ylabel("worst rank")

    for ax in axes:
        ax.tick_params(axis="x", length=0, labelsize=9)
    fig.suptitle("Measured on our 18-chunk toy corpus - fusion did not beat the better half",
                 fontsize=11, y=1.04)
    save(fig, "l42_09_hybrid_gain")
    return mrr, worst


def main():
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    chunk_vecs = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True)
    docs_tok = corpus_tokens()

    fig_win_scatter(model, chunk_vecs, docs_tok)
    fig_keyword_payoff(model, chunk_vecs, docs_tok)
    fig_expansion(docs_tok)
    fig_score_scales(model, chunk_vecs, docs_tok)
    fig_rrf_curve()
    mrr, _ = fig_hybrid_gain(model, chunk_vecs, docs_tok)
    if mrr[2] > max(mrr[0], mrr[1]):
        raise ValueError("hybrid now beats both halves - the deck's honest-negative frames "
                         "and the provenance block must be rewritten, not silently left")
    log.info("RRF k = %d; done: 6 figures", RRF_K)


if __name__ == "__main__":
    main()
