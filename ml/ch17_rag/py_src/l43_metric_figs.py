"""Metric figures for L43 (generation and evaluation): 02, 06, 07, 08, 09, 11, 16.

Provenance per figure:
  02  REAL - token counts from tiktoken (cl100k_base) over a real 24.5k-token technical
             document chunked at 800 characters. Nothing here is estimated.
  06  REAL - recall@k on the 8-question evaluation set, from ranks measured by
             l43_probe_claims.py with intfloat/multilingual-e5-small over the 20-chunk corpus.
  07  REAL - Mean Reciprocal Rank, three of those measured ranks, arithmetic exact.
  08  REAL - nDCG@5 on the measured cold-open ranking. Relevance grades are a human
             judgement made by one stated rule; the arithmetic on them is exact.
  09  REAL - faithfulness arithmetic on a stated four-statement decomposition (3/4).
             The answer being decomposed is an illustrative failure, and the slide says so.
  11  REAL - agreement with human annotators, transcribed from Ragas (Es et al., 2023),
             Table 1. Not re-measured here.
  16  REAL - exact arithmetic on stated archive parameters (400 reports, 37 relevant).

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l43_metric_figs.py   (~5 s)
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken

from l43_data import FAITH_STATEMENTS, N_INCIDENT_REPORTS, N_MENTIONING_PRESS

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
PAPERS = Path(__file__).resolve().parent.parent / "papers" / "llm_readable"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l43_metric_figs.log", encoding="utf-8")],
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

# --- measured inputs -------------------------------------------------------------------
# Rank of the gold chunk for each of the 8 evaluation questions. Measured 2026-08-10 by
# l43_probe_claims.py: intfloat/multilingual-e5-small, 20-chunk corpus, BM25 k1=1.5 b=0.75.
DENSE_RANKS_ASKED = [10, 2, 5, 6, 1, 4, 1, 1]
BM25_RANKS_ASKED = [3, 2, 7, 11, 13, 20, 15, 18]
DENSE_RANKS_COPIED = [1, 1, 1, 1, 1, 1, 1, 1]

# Three of the measured "asked" ranks, used for the by-hand MRR frame.
MRR_HAND = [
    ("How small do we chop it up?", 1),
    ("How is the raw milk made safe...?", 2),
    ("How hard does the machine squeeze the cheese?", 10),
]

# The measured cold-open ranking (dense). Grades by the rule stated on the slide:
#   3 = current and complete, 2 = part of the answer, 1 = right topic but superseded, 0 = no
COLD_OPEN = [
    ("rev.4 (2024)", 3),
    ("rev.2 (2019)", 1),
    ("press 2.5 bar", 2),
    ("brining", 0),
    ("PRS-380", 0),
]

# Ragas (Es et al., 2023), Table 1: accuracy of agreement with two human annotators on
# WikiEval pairwise comparisons. Chance on a two-way comparison is 0.50.
JUDGE_AGREEMENT = {
    "Ragas": (0.95, 0.78, 0.70),
    "ask for a 0-10 score": (0.72, 0.52, 0.63),
    "ask which is better": (0.54, 0.40, 0.52),
}
JUDGE_DIMS = ["faithfulness", "answer\nrelevance", "context\nrelevance"]


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


def recall_at_k(ranks, k):
    return float(np.mean([r <= k for r in ranks]))


# --- figure 02 -------------------------------------------------------------------------
def fig_token_budget():
    """Real tiktoken counts over a real technical document chunked at 800 characters."""
    doc = PAPERS / "01_bm25_robertson_zaragoza_2009.txt"
    if not doc.exists():
        raise FileNotFoundError(f"missing source document: {doc}")
    text = doc.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("cl100k_base")
    size = 800
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    counts = np.array([len(enc.encode(c)) for c in chunks])
    log.info("fig02: %d chunks, mean %.1f tokens, median %.1f, max %d",
             len(chunks), counts.mean(), float(np.median(counts)), counts.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.74, 1.55))   # drawn at slot size: 0.86 width

    ax1.hist(counts, bins=24, color=ARM_BLUE, alpha=0.85)
    ax1.axvline(counts.mean(), color=ARM_RED, lw=1.2,
                label=f"mean {counts.mean():.0f} tokens")
    ax1.set_xlabel("tokens in one 800-character chunk")
    ax1.set_ylabel("number of chunks")
    ax1.set_title("Same character budget, different token cost")
    ax1.legend(frameon=False)

    budgets = [1000, 2000, 4000, 8000]
    cum = np.cumsum(counts)
    fits = [int(np.sum(cum <= b)) for b in budgets]
    log.info("fig02: budgets %s -> chunks %s (of %d)", budgets, fits, len(chunks))
    bars = ax2.bar([f"{b//1000}k" for b in budgets], fits, color=ARM_ORANGE,
                   edgecolor="white")
    ax2.bar_label(bars, fmt="%d", fontsize=6.5, padding=1.5)
    ax2.axhline(len(chunks), color=GREY, ls="--", lw=0.9)
    ax2.text(-0.4, len(chunks) + 2, f"whole document = {len(chunks)} chunks",
             ha="left", va="bottom", fontsize=6.5, color=GREY)
    ax2.set_ylim(0, len(chunks) * 1.22)
    ax2.set_xlabel("token budget for retrieved context")
    ax2.set_ylabel("chunks that fit")
    ax2.set_title("One document, 24,519 tokens")

    fig.tight_layout(w_pad=1.2)
    save(fig, "l43_02_token_budget")


# --- figure 06 -------------------------------------------------------------------------
def fig_recall_at_k():
    ks = np.arange(1, 11)
    series = [
        ("questions copied\nfrom the chunk (dense)", DENSE_RANKS_COPIED, ARM_ORANGE, "s"),
        ("questions as a\nuser asks them (dense)", DENSE_RANKS_ASKED, ARM_BLUE, "o"),
        ("questions as a\nuser asks them (BM25)", BM25_RANKS_ASKED, ARM_RED, "^"),
    ]
    fig, ax = plt.subplots(figsize=(4.41, 1.65))   # drawn at slot size: 0.80 width
    for label, ranks, color, marker in series:
        curve = [recall_at_k(ranks, k) for k in ks]
        log.info("fig06: %-42s %s", label.replace("\n", " "), [round(c, 3) for c in curve])
        ax.plot(ks, curve, marker=marker, color=color, lw=1.3, ms=3.2, label=label)

    ax.set_xticks(ks)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("k  (how many chunks go into the prompt)")
    ax.set_ylabel("recall@k")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), labelspacing=0.9)
    ax.set_title("8 questions, 20 chunks, one correct chunk each")
    fig.tight_layout()
    save(fig, "l43_06_recall_at_k")


# --- figure 07 -------------------------------------------------------------------------
def fig_mrr():
    labels = [q for q, _ in MRR_HAND]
    ranks = [r for _, r in MRR_HAND]
    recip = [1.0 / r for r in ranks]
    mean = float(np.mean(recip))
    log.info("fig07: ranks %s -> reciprocals %s -> MRR %.4f",
             ranks, [round(x, 4) for x in recip], mean)

    fig, ax = plt.subplots(figsize=(4.96, 1.3), layout="constrained")   # slot: 0.90 width
    ypos = np.arange(len(ranks))[::-1]
    ax.barh(ypos, recip, color=[ARM_BLUE, ARM_ORANGE, ARM_RED], height=0.5)
    for y, r, v in zip(ypos, ranks, recip):
        ax.text(1.08, y, f"rank {r}  ->  1/{r} = {v:.2f}", fontsize=6.5, va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'"{l}"' for l in labels], fontsize=6)
    ax.set_xlim(0, 1.85)
    ax.set_ylim(-0.75, 2.5)
    ax.set_xlabel("reciprocal rank of the correct chunk")
    ax.axvline(mean, color=GREEN, lw=1.2, ls="--", ymax=0.82)
    ax.text(mean, 2.62, f"MRR = mean of the three = {mean:.3f}", color=GREEN,
            fontsize=7, fontweight="bold", va="bottom", ha="center")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "l43_07_mrr")


# --- figure 08 -------------------------------------------------------------------------
def fig_ndcg():
    labels = [l for l, _ in COLD_OPEN]
    gains = np.array([g for _, g in COLD_OPEN], dtype=float)
    ideal = np.sort(gains)[::-1]
    pos = np.arange(1, len(gains) + 1)
    disc = 1.0 / np.log2(pos + 1)
    contrib = gains * disc
    icontrib = ideal * disc
    dcg, idcg = contrib.sum(), icontrib.sum()
    log.info("fig08: DCG %.4f  IDCG %.4f  nDCG %.4f", dcg, idcg, dcg / idcg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.85, 1.85),   # drawn at slot size
                                   gridspec_kw={"width_ratios": [2.3, 1]})

    width = 0.38
    b1 = ax1.bar(pos - width / 2, gains, width, color=ARM_BLUE, label="relevance grade")
    b2 = ax1.bar(pos + width / 2, contrib, width, color=ARM_ORANGE,
                 label="grade / log$_2$(rank+1)")
    ax1.bar_label(b2, fmt="%.2f", fontsize=6, padding=1)
    ax1.set_xticks(pos)
    ax1.set_xticklabels([f"{i}\n{l.replace(' ', chr(10), 1)}\n\u00f7 {np.log2(i + 1):.2f}"
                         for i, l in zip(pos, labels)],
                        fontsize=6)
    ax1.set_ylim(0, 3.6)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_ylabel("value")
    ax1.set_xlabel("rank returned by the retriever, and what its grade is divided by")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title("Later positions are discounted")

    bars = ax2.bar(["DCG@5\n(what we got)", "IDCG@5\n(best possible)"], [dcg, idcg],
                   color=[ARM_ORANGE, GREY], width=0.55)
    ax2.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1.5)
    ax2.set_ylim(0, idcg * 1.25)
    ax2.tick_params(axis="x", labelsize=6)
    ax2.set_title(f"nDCG@5 = {dcg:.3f} / {idcg:.3f} = {dcg/idcg:.3f}",
                  fontsize=7.5, color=GREEN, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, "l43_08_ndcg")


# --- figure 09 -------------------------------------------------------------------------
def fig_faithfulness():
    supported = sum(1 for _, ok in FAITH_STATEMENTS if ok)
    total = len(FAITH_STATEMENTS)
    log.info("fig09: faithfulness = %d/%d = %.3f", supported, total, supported / total)

    fig, ax = plt.subplots(figsize=(4.63, 1.45))   # drawn at slot size
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.75, len(FAITH_STATEMENTS) + 0.4)
    ax.axis("off")

    for i, (text, ok) in enumerate(FAITH_STATEMENTS):
        y = len(FAITH_STATEMENTS) - i - 0.4
        color = GREEN if ok else ARM_RED
        mark = "supported by the context" if ok else "NOT in the context"
        ax.add_patch(plt.Rectangle((0.0, y - 0.30), 7.25, 0.62,
                                   facecolor=color, alpha=0.10, edgecolor=color, lw=0.7))
        ax.text(0.12, y, f'"{text}"', fontsize=6.3, va="center")
        ax.text(7.4, y, mark, fontsize=6.3, va="center", color=color, fontweight="bold")

    ax.text(5.0, -0.55, f"faithfulness  =  {supported} supported / {total} statements  =  "
                       f"{supported/total:.2f}",
            fontsize=7.5, ha="center", color=ARM_BLUE, fontweight="bold")
    fig.tight_layout()
    save(fig, "l43_09_faithfulness")


# --- figure 11 -------------------------------------------------------------------------
def fig_judge_agreement():
    fig, ax = plt.subplots(figsize=(4.3, 1.5))   # drawn at slot size: 0.78 width
    x = np.arange(len(JUDGE_DIMS))
    width = 0.26
    colors = [ARM_BLUE, ARM_ORANGE, ARM_RED]
    for i, ((name, vals), color) in enumerate(zip(JUDGE_AGREEMENT.items(), colors)):
        bars = ax.bar(x + (i - 1) * width, vals, width, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=5.8, padding=1.5)

    ax.axhline(0.5, color=GREY, ls="--", lw=0.9)
    ax.text(1.005, 0.5, "coin\nflip", fontsize=6.5, color=GREY, va="center",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(JUDGE_DIMS)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("agreement\nwith humans")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.12, 0.5), handlelength=1.2)
    ax.set_title("How often the automatic judge picks what the humans picked")
    fig.tight_layout()
    save(fig, "l43_11_judge_agreement")


# --- figure 16 -------------------------------------------------------------------------
def fig_aggregation():
    ks = [3, 5, 10, 20, 50]
    seen = [min(k, N_MENTIONING_PRESS) for k in ks]
    log.info("fig16: %d relevant of %d; top-k %s -> seen %s",
             N_MENTIONING_PRESS, N_INCIDENT_REPORTS, ks, seen)

    fig, ax = plt.subplots(figsize=(3.53, 1.55))   # drawn at slot size
    bars = ax.bar([f"top-{k}" for k in ks], seen, color=ARM_BLUE, width=0.58)
    ax.bar_label(bars, fmt="%d", fontsize=6.5, padding=1.5)
    ax.axhline(N_MENTIONING_PRESS, color=ARM_RED, lw=1.2)
    ax.text(-0.42, N_MENTIONING_PRESS + 1.6,
            f"true answer: {N_MENTIONING_PRESS} reports mention the press",
            color=ARM_RED, fontsize=6.5, ha="left", fontweight="bold")
    ax.set_ylim(0, N_MENTIONING_PRESS * 1.42)
    ax.set_ylabel("reports the model\ncan actually see")
    ax.set_title(f'"How many incidents involved the press?"  -  '
                 f'{N_INCIDENT_REPORTS} reports in the archive', fontsize=7)
    fig.tight_layout()
    save(fig, "l43_16_aggregation")


def main():
    np.random.seed(SEED)
    fig_token_budget()
    fig_recall_at_k()
    fig_mrr()
    fig_ndcg()
    fig_faithfulness()
    fig_judge_agreement()
    fig_aggregation()
    log.info("all metric figures written")


if __name__ == "__main__":
    main()
