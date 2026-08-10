"""Diagram figures for L43 (generation and evaluation): 04, 05, 14, 15.

All four are ILLUSTRATIVE - schematics carrying an argument, not plots of data. None of
them shows a number, so there is nothing here to mislabel.

  04  the four outcomes of a grounded answer: answerable or not, answered or refused.
  05  where a RAG pipeline leaks, and which metric is watching each leak.
  14  the agentic retrieval loop: search, read, decide, search again.
  15  when the answer is not RAG.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l43_diagram_figs.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"
LIGHT = "#EFEFF4"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l43_diagram_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "figure.dpi": 140})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def box(ax, x, y, w, h, label, color, fontsize=9.5, text_color="white", weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=lw))


# --- figure 04 -------------------------------------------------------------------------
def fig_grounding_outcomes():
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # col 0 = the system answers, col 1 = the system refuses.
    # row 1 = the context contains the answer, row 0 = it does not.
    # The two off-diagonal cells were swapped until 2026-08-10 (caught in student review):
    # fabrication is answering WITHOUT support, i.e. (answers, no context) = (0, 0).
    # Refusing when the answer was present is merely a wasted opportunity, i.e. (1, 1).
    cells = [
        # (col, row, title, subtitle, colour)
        (0, 1, "Correct answer", "cited, checkable", GREEN),
        (1, 1, "Refused anyway", "annoying, but safe", ARM_ORANGE),
        (0, 0, "Made up", "the failure everyone fears", ARM_RED),
        (1, 0, '"I do not know"', "the answer you licensed", GREEN),
    ]
    for col, row, title, sub, color in cells:
        x = 3.0 + col * 3.3
        y = 0.75 + row * 1.75
        ax.add_patch(FancyBboxPatch((x, y), 3.05, 1.5, boxstyle="round,pad=0.02",
                                    facecolor=color, alpha=0.16, edgecolor=color, lw=1.6))
        ax.text(x + 1.52, y + 0.92, title, ha="center", fontsize=11, fontweight="bold",
                color=color)
        ax.text(x + 1.52, y + 0.45, sub, ha="center", fontsize=8.8, color="#444444")

    ax.text(4.52, 4.45, "the system answers", ha="center", fontsize=10, fontweight="bold")
    ax.text(7.82, 4.45, "the system refuses", ha="center", fontsize=10, fontweight="bold")
    ax.text(2.75, 3.25, "the context\ncontains\nthe answer", ha="right", va="center",
            fontsize=10, fontweight="bold")
    ax.text(2.75, 1.5, "it does\nnot", ha="right", va="center",
            fontsize=10, fontweight="bold")

    # Name the boxes rather than their positions, so the caption cannot drift out of sync
    # with the layout the way the swapped labels did.
    ax.text(6.15, 0.18, 'Tighten the prompt and "Made up" shrinks - '
                        'but "Refused anyway" grows.',
            ha="center", fontsize=9.5, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "l43_04_grounding_outcomes")


# --- figure 05 -------------------------------------------------------------------------
def fig_failure_ladder():
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    stages = [
        (0.15, "chunk the\ndocuments", ARM_BLUE),
        (2.85, "retrieve\ntop-k", ARM_BLUE),
        (5.55, "assemble\nthe prompt", ARM_BLUE),
        (8.25, "generate\nthe answer", ARM_BLUE),
    ]
    for x, label, color in stages:
        box(ax, x, 3.35, 2.4, 0.85, label, color, fontsize=10)
    for x, _, _ in stages[:-1]:
        arrow(ax, x + 2.42, 3.78, x + 2.68, 3.78)

    leaks = [
        (0.15, "the answer got\ncut in half", "nothing downstream\ncan recover it", ARM_RED),
        (2.85, "the right chunk\nis not in the top-k", "the retrieval metrics\nsee this", GREEN),
        (5.55, "two retrieved chunks\ncontradict each other", "no standard metric\nsees this",
         ARM_RED),
        (8.25, "the answer ignores\nthe context", "faithfulness\nsees this", GREEN),
    ]
    for x, failure, caught, color in leaks:
        arrow(ax, x + 1.2, 3.30, x + 1.2, 2.55, color=ARM_RED, lw=1.8)
        ax.add_patch(FancyBboxPatch((x, 1.55), 2.4, 0.95, boxstyle="round,pad=0.02",
                                    facecolor=ARM_RED, alpha=0.12, edgecolor=ARM_RED,
                                    lw=1.2))
        ax.text(x + 1.2, 2.02, failure, ha="center", va="center", fontsize=8.8,
                color="#333333")
        ax.add_patch(FancyBboxPatch((x, 0.35), 2.4, 0.9, boxstyle="round,pad=0.02",
                                    facecolor=color, alpha=0.14, edgecolor=color, lw=1.2))
        ax.text(x + 1.2, 0.80, caught, ha="center", va="center", fontsize=8.8,
                color=color, fontweight="bold")

    ax.text(0.05, 2.70, "what goes wrong", fontsize=9, color=ARM_RED, fontweight="bold")
    ax.text(0.05, 1.34, "who notices", fontsize=9, color=GREY, fontweight="bold")
    fig.tight_layout()
    save(fig, "l43_05_failure_ladder")


# --- figure 14 -------------------------------------------------------------------------
def fig_agentic_loop():
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    box(ax, 0.1, 1.75, 1.7, 0.85, "question", GREY, fontsize=10)
    box(ax, 2.35, 1.75, 1.9, 0.85, "write a\nsearch query", ARM_BLUE, fontsize=9.5)
    box(ax, 4.75, 1.75, 1.7, 0.85, "retrieve\nand read", ARM_BLUE, fontsize=9.5)
    box(ax, 6.95, 1.75, 2.5, 0.85, "enough to answer?", ARM_ORANGE, fontsize=9.5)
    box(ax, 6.95, 0.25, 2.5, 0.85, "answer, with sources", GREEN, fontsize=9.5)

    arrow(ax, 1.85, 2.17, 2.30, 2.17)
    arrow(ax, 4.30, 2.17, 4.70, 2.17)
    arrow(ax, 6.50, 2.17, 6.90, 2.17)
    arrow(ax, 8.20, 1.70, 8.20, 1.15, color=GREEN)
    ax.text(8.32, 1.40, "yes", fontsize=9.5, color=GREEN, fontweight="bold")

    ax.add_patch(FancyArrowPatch((8.20, 2.65), (3.30, 2.65),
                                 connectionstyle="arc3,rad=0.40", arrowstyle="-|>",
                                 mutation_scale=13, color=ARM_RED, lw=1.8))
    ax.text(5.75, 3.85, "no - search again, with what you just learned",
            ha="center", fontsize=9.5, color=ARM_RED, fontweight="bold")

    ax.text(4.8, 0.02, "The number of retrieval calls is decided at run time, "
                       "not fixed in advance.",
            ha="center", fontsize=9.5, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "l43_14_agentic_loop")


# --- figure 15 -------------------------------------------------------------------------
def fig_when_not_rag():
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    questions = [
        (0.15, "Is the data\nin tables?", "write SQL", ARM_ORANGE),
        (2.75, "Does the whole\ncorpus fit in\nthe context?", "just paste it", ARM_ORANGE),
        (5.35, "Is the problem\nstyle, format\nor tone?", "fine-tune", ARM_ORANGE),
        (7.95, "Does it need\ncounting over\neverything?", "aggregate first", ARM_ORANGE),
    ]
    for x, q, answer, color in questions:
        ax.add_patch(FancyBboxPatch((x, 2.55), 2.35, 1.35, boxstyle="round,pad=0.02",
                                    facecolor=ARM_BLUE, alpha=0.14, edgecolor=ARM_BLUE,
                                    lw=1.4))
        ax.text(x + 1.17, 3.22, q, ha="center", va="center", fontsize=9.5,
                color="#222222", fontweight="bold")
        arrow(ax, x + 1.17, 2.50, x + 1.17, 1.95, color=color, lw=1.8)
        box(ax, x + 0.15, 1.05, 2.05, 0.85, answer, color, fontsize=10)
        ax.text(x + 1.30, 2.22, "yes", ha="left", fontsize=8.5, color=color,
                fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.15, 0.10), 10.15, 0.72, boxstyle="round,pad=0.02",
                                facecolor=GREEN, alpha=0.14, edgecolor=GREEN, lw=1.5))
    ax.text(5.22, 0.46, "no to all four  ->  RAG is the right tool", ha="center",
            va="center", fontsize=11, color=GREEN, fontweight="bold")
    fig.tight_layout()
    save(fig, "l43_15_when_not_rag")


def main():
    fig_grounding_outcomes()
    fig_failure_ladder()
    fig_agentic_loop()
    fig_when_not_rag()
    log.info("all diagram figures written")


if __name__ == "__main__":
    main()
