"""Go-board schematic for the L32 cold open (AlphaGo move 37).

DELIBERATELY A SCHEMATIC, NOT THE REAL POSITION. Reproducing the actual 37-stone board from
AlphaGo vs Lee Sedol game 2 would require a verified game record; what the slide needs is the
*concept* - that opening play conventionally lives on the third and fourth lines, and AlphaGo
played a shoulder hit on the fifth. The figure is labelled as a schematic on the slide.

Verified facts used on the slide (web-checked 2026-08-06):
  - game 2 of the March 2016 match, move 37, a shoulder hit on the FIFTH line
  - AlphaGo's own policy network put it at ~1 in 10,000 for a human player
  - played while Lee Sedol was away from the board; he took 12+ minutes to reply
  - commentators initially read it as a mistake

Generates ml/ch11_rl/fig/go_lines.pdf

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/go_schematic.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

SIZE = 19
STAR_POINTS = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_go_schematic.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_facecolor("#f2d9a0")

    for i in range(SIZE):
        ax.plot([0, SIZE - 1], [i, i], color="0.35", lw=0.7, zorder=1)
        ax.plot([i, i], [0, SIZE - 1], color="0.35", lw=0.7, zorder=1)

    for x, y in STAR_POINTS:
        ax.plot(x, y, "o", color="0.25", ms=4.5, zorder=2)

    # The conventional lines, counted from the edge. Labels are single-line and pushed to
    # staggered x-positions: at one board-unit apart, two-line labels collide.
    for line, color, label in [(2, BLUE, "3rd line - territory"),
                               (3, "#2e7d32", "4th line - influence"),
                               (4, RED, "5th line - AlphaGo played here")]:
        ax.plot([0, SIZE - 1], [line, line], color=color, lw=2.6, alpha=0.85, zorder=3)
        ax.text(SIZE - 0.4, line, label, color=color, fontsize=8.5, va="center", ha="left",
                fontweight="bold")

    # a black stone on the fifth line - the shoulder hit, schematically
    ax.plot(9, 4, "o", color="black", ms=17, zorder=4)
    ax.annotate("move 37", xy=(9, 4), xytext=(11.6, 7.6), fontsize=11, fontweight="bold",
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.8), zorder=5)

    ax.set_xlim(-1.2, SIZE + 7.2)
    ax.set_ylim(-1.2, SIZE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Schematic: opening play lives on the 3rd and 4th lines.\n"
                 "AlphaGo played the 5th.", fontsize=11)

    fig.tight_layout()
    out = FIG / "go_lines.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.relative_to(REPO_ROOT)} (schematic, not the real board position)")


if __name__ == "__main__":
    main()
