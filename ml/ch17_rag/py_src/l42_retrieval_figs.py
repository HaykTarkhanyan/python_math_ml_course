"""L42 figures 01, 05, 06, 07, 08, 09 - all REAL, all measured by this script.

  01  where BM25 and dense disagree, over 16 queries with a known target chunk
  05  what each keyword-extraction method does to retrieval of the same target
  06  query expansion rescues the paraphrase BM25 could not see
  07  BM25 and cosine live on different scales, so adding them is not fusion
  08  the RRF weight 1/(k + rank), for three values of k
  09  BM25 vs dense vs RRF hybrid on the 16-query set - the honest result

Numbers here are the ones on the slides. Anything that disagreed with the intended story
changed the story: see the provenance block of L42_rag_hybrid_rerank.tex.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_retrieval_figs.py   (~30 s)
"""

import logging
import os
import time
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
    "font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
    "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


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
    dense_first = sum(n for (_bx, dy), n in counts.items() if dy == 1)   # L42 quotes "14 of the 16"
    if counts[(1, 1)] != 12 or counts[(17, 4)] != 1 or dense_first != 14:
        raise ValueError(f"win scatter changed: {dict(counts)}; the frame quotes 12 at (1, 1) and one at (17, 4)")

    fig, ax = plt.subplots(figsize=(2.53, 2.25))   # drawn at slot size: 0.46 column
    lim = 19.5
    ax.plot([0.6, lim], [0.6, lim], color=GREY, lw=0.8, ls=":", zorder=1)
    ax.text(11.0, 11.6, "both agree", fontsize=6.5, color=GREY, rotation=45,
            ha="center", va="bottom")

    for (bx, dy), n in counts.items():
        colour = ARM_RED if abs(bx - dy) >= 3 else ARM_BLUE
        ax.scatter([bx], [dy], s=18 + 9 * n, color=colour, alpha=0.85, zorder=3,
                   edgecolor="white", lw=0.6)

    big, nbig = max(counts.items(), key=lambda kv: kv[1])
    ax.annotate(f"{nbig} questions here\n(both found it first)",
                xy=big, xytext=(big[0] + 3.0, big[1] + 3.2), fontsize=6.5,
                color=ARM_BLUE, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE, lw=0.8))

    bad = [i for i, z in enumerate(zero_score) if z][0]
    bx, dy = pts[bad]
    ax.annotate("BM25 score\n0.000: not one\nshared word",
                xy=(bx, dy), xytext=(bx - 1.5, dy + 3.4), fontsize=6.5, color=ARM_RED,
                fontweight="bold", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=0.8))

    ax.set_xlim(-1.2, lim)
    ax.set_ylim(-1.2, lim)
    ax.set_xlabel("BM25 rank of the correct chunk")
    ax.set_ylabel("embedding rank of the correct chunk")
    ax.set_title("16 questions, 18 chunks:\nthey agree - until one collapses")
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
    if br[:4] != [2, 2, 2, 1]:
        raise ValueError(f"keyword payoff changed: BM25 ranks {br}; the frame quotes 2, 2, 2, then KeyBERT 1")

    x = np.arange(len(variants))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.42, 1.62))   # drawn at slot size: 0.62 width
    b1 = ax.bar(x - w / 2, br, w, color=ARM_ORANGE, label="BM25 (keywords)")
    b2 = ax.bar(x + w / 2, dr, w, color=ARM_BLUE, label="dense (embeddings)")
    ax.bar_label(b1, fmt="%d", fontsize=6.5, fontweight="bold", padding=1.5)
    ax.bar_label(b2, fmt="%d", fontsize=6.5, fontweight="bold", padding=1.5)

    for xi, t in enumerate(tops):
        colour = ARM_RED if t != "press 2.5 bar" else "#2E8B57"
        ax.text(xi, -0.95, t.replace(" ", "\n", 1), fontsize=5.8, color=colour, ha="center",
                va="top")
    ax.text(-0.62, -0.95, "BM25's\ntop hit:", fontsize=5.8, color=GREY, ha="right", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in variants])
    ax.set_ylabel("rank of the answer\n(1 = found it)")
    ax.set_ylim(0, max(br + dr) + 1.5)
    ax.set_title("Extracting keywords is not free improvement")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.tick_params(axis="x", length=0)
    save(fig, "l42_05_keyword_payoff")


def fig_expansion(docs_tok):
    base = "How warm is the room where the cheese matures?"
    steps = [
        ("the question\nas typed", base),
        ("+ generic:\ntemperature,\ndegrees, heat", base + " temperature degrees heat"),
        ("+ domain:\nchamber, cellar,\nripening, ageing",
         base + " temperature degrees heat chamber cellar ripening ageing"),
    ]
    ranks, scores = [], []
    for name, q in steps:
        bs = bm25_scores(q, docs_tok)
        ranks.append(int(ranks_from_scores(bs)[9]))
        scores.append(float(bs[9]))
        log.info("06 %-22s BM25 rank %2d score %.3f", name.replace("\n", " "), ranks[-1], scores[-1])
    if ranks != [17, 6, 2]:
        raise ValueError(f"expansion ranks changed: {ranks}; the frame quotes 17 -> 6 -> 2")

    fig, ax = plt.subplots(figsize=(2.75, 1.38))   # drawn at slot size: 0.50 width
    bars = ax.bar([s[0] for s in steps], ranks,
                  color=[ARM_RED, ARM_ORANGE, "#2E8B57"], width=0.55)
    ax.bar_label(bars, labels=[f"rank {r}\nscore {s:.2f}" for r, s in zip(ranks, scores)],
                 fontsize=6, fontweight="bold", padding=1.5)
    ax.set_ylim(0, max(ranks) * 1.4)
    ax.set_ylabel("BM25 rank of the answer\n(1 = found it)")
    ax.set_title("The query BM25 scored 0.000 on,\nafter adding words to it", fontsize=7)
    ax.tick_params(axis="x", length=0, labelsize=6)
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
    got = tuple(round(float(v), 3) for v in (bs.min(), bs.max(), ds.min(), ds.max()))
    if got != (0.0, 3.021, 0.710, 0.858) or top_bm != top_sum:
        raise ValueError(f"score ranges changed: {got}; the slides quote 0.000-3.021 and 0.710-0.858, identical top 3")

    fig, ax = plt.subplots(figsize=(3.42, 1.25))   # drawn at slot size: 0.62 width
    ax.scatter(bs, np.full_like(bs, 1.0), s=22, color=ARM_ORANGE, alpha=0.8,
               edgecolor="white", lw=0.5, zorder=3)
    ax.scatter(ds, np.full_like(ds, 0.35), s=22, color=ARM_BLUE, alpha=0.8,
               edgecolor="white", lw=0.5, zorder=3)
    ax.hlines([1.0, 0.35], -0.05, 3.25, color=GREY, lw=0.6, zorder=1)

    ax.annotate(f"BM25: {bs.min():.2f} to {bs.max():.2f}", (3.3, 1.0), fontsize=6.5,
                color=ARM_ORANGE, fontweight="bold", va="center")
    ax.annotate(f"cosine: {ds.min():.2f} to {ds.max():.2f}", (3.3, 0.35), fontsize=6.5,
                color=ARM_BLUE, fontweight="bold", va="center")

    ax.set_yticks([])
    ax.set_ylim(0.15, 1.2)
    ax.set_xlim(-0.15, 4.9)
    ax.set_xlabel("score for each of the 18 chunks, same query")
    ax.set_title("One score runs 0 to 3, the other lives inside a 0.15-wide band")
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.16,
             "Adding them: the top 3 is exactly BM25's top 3. The embedder gets no vote.",
             ha="center", va="top", fontsize=6.5, color=ARM_RED, style="italic")
    save(fig, "l42_07_score_scales")


# --- figure 08 -------------------------------------------------------------------------
def fig_rrf_curve():
    ranks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(2.75, 1.5))   # drawn at slot size: 0.50 width
    for k, colour in [(0, ARM_RED), (10, ARM_ORANGE), (60, ARM_BLUE)]:
        w = 1.0 / (k + ranks)
        ax.plot(ranks, w / w[0], "o-", ms=2.5, color=colour, lw=1.1)
        ax.text(20.7, (w / w[0])[-1], f"k = {k}", color=colour, fontsize=6.5,
                fontweight="bold", va="center")
        log.info("08 k=%2d: weight(1)=%.4f weight(2)=%.4f ratio %.2f",
                 k, 1 / (k + 1), 1 / (k + 2), (1 / (k + 1)) / (1 / (k + 2)))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("rank in one list")
    ax.set_ylabel("weight, relative to rank 1")
    ax.set_title(r"$1/(k + \mathrm{rank})$: bigger $k$ means rank 1 is not king")
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
    got = (round(float(mrr[1]), 3), round(float(mrr[2]), 3), int(worst[1]), int(worst[2]))
    if got != (0.922, 0.913, 4, 9):
        raise ValueError(f"hybrid numbers changed: {got}; the frame quotes 0.922, 0.913, rank 4 -> 9")

    fig, axes = plt.subplots(1, 2, figsize=(3.31, 1.6))   # drawn at slot size: 0.60 width
    colours = [ARM_ORANGE, ARM_BLUE, "#2E8B57"]

    bars = axes[0].bar(names, mrr, color=colours, width=0.55)
    axes[0].bar_label(bars, fmt="%.3f", fontsize=6, fontweight="bold", padding=1.5)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("average of 1/rank, 16 questions\n(MRR; higher is better)", fontsize=6.5)
    axes[0].set_ylabel("mean reciprocal rank")

    bars = axes[1].bar(names, worst, color=colours, width=0.55)
    axes[1].bar_label(bars, fmt="%d", fontsize=6, fontweight="bold", padding=1.5)
    axes[1].set_ylim(0, max(worst) * 1.3)
    axes[1].set_title("the single worst question\n(answer's rank, lower is better)", fontsize=6.5)
    axes[1].set_ylabel("worst rank")

    for ax in axes:
        ax.tick_params(axis="x", length=0, labelsize=6)
    fig.tight_layout(w_pad=1.0)
    fig.suptitle("Our 18-chunk toy corpus: fusion did not beat the better half",
                 fontsize=7.5, y=1.04)
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
