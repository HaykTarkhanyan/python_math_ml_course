"""Diagram and benchmark figures for L41 (RAG retrieval): 01, 03, 11, 12, 14, 15.

Provenance per figure:
  01  ILLUSTRATIVE - the RAG loop, offline indexing vs online querying.
  03  ILLUSTRATIVE - the recall ceiling. Stage percentages are invented to show the shape
                     of the argument; the slide says so. The point (stage 1 caps everything)
                     is structural, not empirical.
  11  ILLUSTRATIVE - bi-encoder vs cross-encoder wiring.
  12  ILLUSTRATIVE - what runs once vs per query.
  14  REAL          - ArmBench-TextEmbed scores transcribed from the ATE-2 announcement
                     (huggingface.co/blog/Metric-AI/ate-2). These are ATE-2 numbers; do not
                     mix them with the ATE-1 paper's numbers.
  15  ILLUSTRATIVE - the ATE-2 recipe, from the same post.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_diagram_figs.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
              logging.FileHandler(LOGS / "l41_diagram_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "figure.dpi": 140})

# ArmBench-TextEmbed, native Armenian. Source: huggingface.co/blog/Metric-AI/ate-2
ATE2_SCORES = [
    ("ATE-2-large\n(560M, open)", 0.805, ARM_RED),
    ("gemini-embedding-001\n(commercial)", 0.774, GREY),
    ("ATE-2-base\n(278M, open)", 0.767, ARM_ORANGE),
]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def box(ax, x, y, w, h, label, color, fontsize=9, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.5))


# --- figure 01 -------------------------------------------------------------------------
def fig_rag_loop():
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.1, 2.55), 10.8, 1.85, boxstyle="round,pad=0.02",
                                facecolor="#F2F6FB", edgecolor="#C8D6E8", lw=1.2))
    ax.text(0.32, 4.16, "ONCE, OFFLINE  -  when documents change",
            fontsize=9.5, color=ARM_BLUE, fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.1, 0.18), 10.8, 1.85, boxstyle="round,pad=0.02",
                                facecolor="#FDF3F3", edgecolor="#EFCFCF", lw=1.2))
    ax.text(0.32, 1.79, "EVERY QUERY  -  in milliseconds",
            fontsize=9.5, color=ARM_RED, fontweight="bold")

    stages = [("4,000\ndocuments", 0.45), ("chunk", 2.5), ("embed", 4.55), ("index", 6.6)]
    for label, x in stages:
        box(ax, x, 2.95, 1.6, 0.85, label, ARM_BLUE)
    for _, x in stages[:-1]:
        arrow(ax, x + 1.62, 3.37, x + 2.46, 3.37)
    arrow(ax, 8.24, 3.37, 9.05, 3.37)
    box(ax, 9.05, 2.95, 1.6, 0.85, "vector\nstore", "#0B2A57")

    q = [("question", 0.45), ("embed", 2.5), ("search", 4.55), ("top-k\nchunks", 6.6)]
    for label, x in q:
        box(ax, x, 0.58, 1.6, 0.85, label, ARM_RED)
    for _, x in q[:-1]:
        arrow(ax, x + 1.62, 1.0, x + 2.46, 1.0)
    arrow(ax, 8.24, 1.0, 9.05, 1.0)
    box(ax, 9.05, 0.58, 1.6, 0.85, "LLM\nanswer", "#7A0A12")

    ax.add_patch(FancyArrowPatch((9.85, 2.9), (5.35, 1.5), arrowstyle="-|>",
                                 mutation_scale=13, color=GREY, lw=1.5,
                                 linestyle="--", connectionstyle="arc3,rad=0.28"))
    ax.text(7.9, 2.16, "the index is read, never rebuilt",
            fontsize=8.5, color=GREY, style="italic", ha="center")

    ax.set_title("RAG is two pipelines, and only one of them runs when a user asks",
                 fontsize=11.5, pad=6)
    save(fig, "01_rag_loop")


# --- figure 03 -------------------------------------------------------------------------
def fig_recall_ceiling():
    stages = ["answer survives\nchunking", "retrieved in\ntop-100", "survives\nreranking",
              "actually used\nby the LLM"]
    good = [72, 68, 65, 61]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))

    bars = ax.bar(stages, good, color=[ARM_RED] + [ARM_BLUE] * 3, width=0.62)
    # padding=8 keeps each label clear of the dashed ceiling line, which previously
    # struck through the 68% label (student review, 2026-08-10).
    ax.bar_label(bars, fmt="%d%%", fontsize=10.5, fontweight="bold", padding=8)

    ax.axhline(good[0], color=ARM_RED, ls="--", lw=1.4)
    ax.text(3.42, good[0] + 2.2, "ceiling set by chunking", fontsize=9,
            color=ARM_RED, ha="right", fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_ylabel("questions still answerable (%)")
    ax.set_title("No later stage can recover what chunking threw away", fontsize=11)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    fig.text(0.5, -0.055, "Illustrative percentages - the shape is the point, not the values.",
             ha="center", fontsize=8.5, color=GREY, style="italic")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    save(fig, "03_recall_ceiling")


# --- figure 11 -------------------------------------------------------------------------
def fig_bi_vs_cross():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    ax = axes[0]
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.35, 0.35, 1.6, 0.7, "query", ARM_RED)
    box(ax, 3.05, 0.35, 1.6, 0.7, "chunk", ARM_BLUE)
    box(ax, 0.35, 1.75, 1.6, 0.8, "encoder", "#555555")
    box(ax, 3.05, 1.75, 1.6, 0.8, "encoder", "#555555")
    arrow(ax, 1.15, 1.1, 1.15, 1.7); arrow(ax, 3.85, 1.1, 3.85, 1.7)
    arrow(ax, 1.15, 2.6, 1.15, 3.15); arrow(ax, 3.85, 2.6, 3.85, 3.15)
    box(ax, 0.35, 3.15, 1.6, 0.6, "vector", ARM_ORANGE)
    box(ax, 3.05, 3.15, 1.6, 0.6, "vector", ARM_ORANGE)
    ax.annotate("", xy=(3.0, 3.45), xytext=(2.0, 3.45),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2))
    ax.text(2.5, 3.75, "cosine", ha="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.text(2.5, 4.55, "Bi-encoder", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(2.5, 4.18, "they never meet - so chunks precompute", ha="center",
            fontsize=8.5, color=GREY, style="italic")

    ax = axes[1]
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.35, 0.35, 1.6, 0.7, "query", ARM_RED)
    box(ax, 3.05, 0.35, 1.6, 0.7, "chunk", ARM_BLUE)
    box(ax, 0.35, 1.85, 4.3, 0.95, "one encoder, full attention\nacross both", "#555555")
    arrow(ax, 1.15, 1.1, 1.6, 1.8); arrow(ax, 3.85, 1.1, 3.4, 1.8)
    arrow(ax, 2.5, 2.85, 2.5, 3.35)
    box(ax, 1.7, 3.35, 1.6, 0.6, "score", GREEN)
    ax.text(2.5, 4.55, "Cross-encoder", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(2.5, 4.18, "accurate, but one pass per pair", ha="center",
            fontsize=8.5, color=GREY, style="italic")

    save(fig, "11_bi_vs_cross_encoder")


# --- figure 12 -------------------------------------------------------------------------
def fig_offline_vs_online():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    labels = ["embed 4,000 chunks\n(once, offline)", "embed 1 query\n(per request)"]
    counts = [4000, 1]
    bars = ax.barh(labels, counts, color=[ARM_BLUE, ARM_RED], height=0.5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 20000)
    for bar, c in zip(bars, counts):
        ax.text(c * 1.35, bar.get_y() + bar.get_height() / 2,
                f"{c:,} encoder pass{'es' if c > 1 else ''}",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("encoder passes (log scale)")
    ax.set_title("Why a bi-encoder is fast: the expensive half already happened",
                 fontsize=11)
    ax.tick_params(axis="y", length=0, labelsize=9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    save(fig, "12_offline_vs_online")


# --- figure 14 -------------------------------------------------------------------------
def fig_ate2_scores():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    names = [n for n, _, _ in ATE2_SCORES]
    vals = [v for _, v, _ in ATE2_SCORES]
    colors = [c for _, _, c in ATE2_SCORES]

    bars = ax.bar(names, vals, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=11, fontweight="bold", padding=3)

    ax.set_ylim(0.70, 0.84)
    ax.set_ylabel("ArmBench-TextEmbed score\n(native Armenian, higher is better)")
    ax.set_title("A 560M open model, tuned on 10,000 synthetic pairs,\n"
                 "beats the commercial embedder on Armenian", fontsize=11)
    ax.tick_params(axis="x", length=0, labelsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    save(fig, "14_ate2_scores")


# --- figure 15 -------------------------------------------------------------------------
def fig_ate2_recipe():
    fig, ax = plt.subplots(figsize=(10.2, 2.5))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 2.5); ax.axis("off")

    box(ax, 0.15, 0.75, 2.1, 1.0, "mE5\nmultilingual encoder", GREY, fontsize=9)
    box(ax, 3.0, 0.75, 2.5, 1.0, "10,000 pairs\ntranslated from\nEnglish Reddit",
        ARM_ORANGE, fontsize=8.5)
    box(ax, 6.25, 0.75, 1.9, 1.0, "fine-tune", ARM_BLUE, fontsize=9)
    box(ax, 8.85, 0.75, 1.2, 1.0, "ATE-2", ARM_RED, fontsize=10)

    arrow(ax, 2.3, 1.25, 2.95, 1.25)
    arrow(ax, 5.55, 1.25, 6.2, 1.25)
    arrow(ax, 8.2, 1.25, 8.8, 1.25)

    ax.text(5.1, 0.32, "noisy synthetic data, not a curated corpus",
            ha="center", fontsize=9, color=GREY, style="italic")
    ax.text(5.1, 2.18, "The whole recipe", ha="center", fontsize=11.5, fontweight="bold")
    save(fig, "15_ate2_recipe")


def main():
    top = max(ATE2_SCORES, key=lambda r: r[1])
    if "ATE-2-large" not in top[0]:
        raise ValueError(f"ATE-2-large is not top of the chart: {top[0]}")
    log.info("ATE-2 chart: %s", {n.split(chr(10))[0]: v for n, v, _ in ATE2_SCORES})

    fig_rag_loop()
    fig_recall_ceiling()
    fig_bi_vs_cross()
    fig_offline_vs_online()
    fig_ate2_scores()
    fig_ate2_recipe()
    log.info("done: 6 figures")


if __name__ == "__main__":
    main()
