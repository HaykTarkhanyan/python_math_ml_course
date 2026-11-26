"""Figures for the practical frames added to 10_rag_generation_eval.tex (instructor, Q50, 2026-11-19).

m01_sources_ui   what the user sees: an answer with numbered citations and a sources panel (a mock-up)
m02_eval_loop    growing the eval set from real failures: logs -> failures -> label -> eval set -> rerun -> ship

Both are schematic (illustrative content, the deck's own cheese-factory example).
Run (repo root, ~2 s):  ./ma/.venv/Scripts/python.exe llms/04_rag/py_src/l43_explain_figs.py
"""
import logging
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
FIG = HERE.parents[1] / "fig"
LOGS = HERE.parents[3] / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOGS / "l43_explain_figs.log", encoding="utf-8")])
log = logging.getLogger("l43_explain_figs")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RED, BLUE, ORANGE, GREEN, GREY = "#D90012", "#0033A0", "#F2A800", "#2E8B57", "#666666"


def save(fig, name):
    out = FIG / f"{name}.pdf"
    for attempt in range(1, 6):
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            break
        except OSError as e:
            if attempt == 5:
                log.error("could not write %s: %s", out, e)
                raise
            time.sleep(1)
    plt.close(fig)
    log.info("wrote %s", out.name)


def sources_ui():
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 6.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#F7F7F7",
                                ec="#BBBBBB", lw=0.8))
    ax.text(0.5, 6.0, "What pressure should the Lori press run at?", fontsize=6.8, color=GREY, style="italic")
    lines = [("The Lori press starts at 2.5 bar ", "[1]"), ("and is raised to 3.2 bar after twenty minutes ", "[1]"),
             ("Both readings go in the shift log ", "[2]")]
    y = 5.15
    for text, cite in lines:
        t = ax.text(0.5, y, text, fontsize=7.2)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.text(bb.x1, y + 0.08, cite, fontsize=6.2, color=BLUE, fontweight="bold")
        y -= 0.62
    ax.plot([0.5, 9.5], [3.2, 3.2], color="#CCCCCC", lw=0.7)
    ax.text(0.5, 2.7, "Sources", fontsize=7, fontweight="bold")
    srcs = [("[1]", "Press manual, rev. 4 (2024), sec. 3.2", "open"),
            ("[2]", "Shift procedures, rev. 2 (2023), page 4", "open")]
    y = 2.05
    for num, title, link in srcs:
        ax.text(0.5, y, num, fontsize=6.6, color=BLUE, fontweight="bold")
        ax.text(1.3, y, title, fontsize=6.6)
        ax.text(9.4, y, link, fontsize=6.6, color=BLUE, ha="right", style="italic")
        y -= 0.6
    ax.text(0.5, 0.45, "every [n] checked against the chunks actually retrieved", fontsize=6.0, color=GREEN,
            style="italic")
    save(fig, "m01_sources_ui")


def eval_loop():
    steps = [("logs", "question,\nchunks, answer,\nfeedback", GREY),
             ("failures", "thumbs down,\n\"I don't know\",\nno citation", RED),
             ("label", "what should\nhave been\nretrieved", ORANGE),
             ("eval set", "add the case\nto your\ngolden set", BLUE),
             ("rerun", "on every\nchange: prompt,\nchunker, model", BLUE),
             ("ship", "only if\nnothing got\nworse", GREEN)]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(5.3, 1.65))        # slot: the full text width; units are inches
    ax.set_xlim(0, 5.3)
    ax.set_ylim(0, 1.65)
    ax.axis("off")
    w, h, y = 0.78, 0.86, 1.08
    xs = [0.42 + k * 0.89 for k in range(n)]
    for k, (x, (title, body, c)) in enumerate(zip(xs, steps)):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=c, alpha=0.12, lw=0))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="none", ec=c, lw=1.0))
        ax.text(x, y + 0.29, title, ha="center", va="center", fontsize=6.5, fontweight="bold")
        ax.text(x, y - 0.08, body, ha="center", va="center", fontsize=5.7)
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((x + w / 2, y), (xs[k + 1] - w / 2, y), arrowstyle="-|>", mutation_scale=7,
                                         color=GREY, lw=0.8, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((xs[-1], y - h / 2), (xs[0], y - h / 2), arrowstyle="-|>", mutation_scale=8,
                                 color=BLUE, lw=0.9, connectionstyle="arc3,rad=-0.08"))
    ax.text(5.3 / 2, 0.06, "and again next week: the eval set grows from what real users break", ha="center",
            fontsize=6.4, color=BLUE, style="italic")
    save(fig, "m02_eval_loop")


def main():
    t0 = time.perf_counter()
    sources_ui()
    eval_loop()
    log.info("done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
