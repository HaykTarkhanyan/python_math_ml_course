"""The four phases of Monte Carlo tree search, for L32e (deck 5).

A teaching diagram, not an experiment. MCTS is a loop of four steps on the same tree,
and seeing the same tree four times is the fastest way to understand it.

Generates into ml/ch11_rl/fig/:
  mcts_phases.pdf

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/mcts_illustration.py

Conventions (repo CLAUDE.md): console + logs/ logging, f-strings, Armenian-flag
colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN, GREY = "#2e7d32", "#999999"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

# The tree, drawn identically in all four panels.
POS = {"root": (0.0, 3.0), "a": (-1.05, 2.0), "b": (1.05, 2.0),
       "ba": (0.35, 1.0), "bb": (1.75, 1.0)}
EDGES = [("root", "a"), ("root", "b"), ("b", "ba"), ("b", "bb")]
PATH = ["root", "b", "bb"]          # the branch this iteration selects
NEW = (2.45, 0.0)                   # the node expansion adds


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_mcts.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def draw_tree(ax, highlight=(), dim=False):
    base = GREY if dim else "0.45"
    for u, v in EDGES:
        on_path = u in highlight and v in highlight
        ax.plot(*zip(POS[u], POS[v]), color=BLUE if on_path else base,
                lw=2.6 if on_path else 1.3, zorder=2)
    for name, (x, y) in POS.items():
        on_path = name in highlight
        ax.add_patch(Circle((x, y), 0.30, facecolor="white",
                            edgecolor=BLUE if on_path else base,
                            lw=2.6 if on_path else 1.3, zorder=3))


def fig_phases():
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.5))
    titles = ["1. Selection", "2. Expansion", "3. Simulation", "4. Backpropagation"]
    captions = [
        "walk down using UCT,\nbalancing wins against\nhow little you have tried",
        "add one child for a\nmove not yet explored",
        "play to the end\n(random, or with a\nlearned policy)",
        "push the result back up\nthrough every node\non the path",
    ]
    colors = [BLUE, GREEN, ORANGE, RED]

    for ax, title, caption, color in zip(axes, titles, captions, colors):
        ax.set_title(title, fontsize=12, color=color, fontweight="bold", pad=8)
        ax.text(0.5, -1.62, caption, ha="center", va="top", fontsize=8.2,
                color="0.3", transform=ax.transData)
        ax.set_xlim(-2.0, 3.3)
        ax.set_ylim(-1.75, 3.6)
        ax.axis("off")

    # 1. selection
    draw_tree(axes[0], highlight=PATH)

    # 2. expansion
    draw_tree(axes[1], highlight=PATH, dim=True)
    axes[1].plot(*zip(POS["bb"], NEW), color=GREEN, lw=2.2, ls="--", zorder=2)
    axes[1].add_patch(Circle(NEW, 0.30, facecolor=GREEN, alpha=0.25,
                             edgecolor=GREEN, lw=2.4, zorder=3))

    # 3. simulation
    draw_tree(axes[2], dim=True)
    axes[2].plot(*zip(POS["bb"], NEW), color=GREY, lw=1.3, ls="--", zorder=2)
    axes[2].add_patch(Circle(NEW, 0.30, facecolor="white", edgecolor=GREY, lw=1.3, zorder=3))
    axes[2].add_patch(FancyArrowPatch(NEW, (2.75, -1.05), arrowstyle="-|>",
                                      mutation_scale=13, color=ORANGE, lw=2.0,
                                      connectionstyle="arc3,rad=0.28", zorder=4))
    axes[2].text(2.75, -1.30, "win", ha="center", fontsize=9.5, color=ORANGE,
                 fontweight="bold")

    # 4. backpropagation
    draw_tree(axes[3], dim=True)
    axes[3].plot(*zip(POS["bb"], NEW), color=GREY, lw=1.3, ls="--", zorder=2)
    axes[3].add_patch(Circle(NEW, 0.30, facecolor="white", edgecolor=GREY, lw=1.3, zorder=3))
    for a, b in [(NEW, POS["bb"]), (POS["bb"], POS["b"]), (POS["b"], POS["root"])]:
        axes[3].add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                          color=RED, lw=2.2, shrinkA=13, shrinkB=13, zorder=5))

    fig.suptitle("One MCTS iteration. Run it thousands of times, then play the most-visited "
                 "move from the root.",
                 fontsize=11.5, color="0.15", y=1.02)
    fig.tight_layout()
    out = FIG / "mcts_phases.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    log.info(f"wrote {fig_phases().relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
