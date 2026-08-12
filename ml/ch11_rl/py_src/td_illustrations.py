"""Illustrations for L32b (deck 2: learning from experience).

These are teaching diagrams, not experiments - nothing here is measured. They exist
because the two ideas they show (when an update happens, and why on-policy and
off-policy disagree) are spatial ideas that prose handles badly.

Generates into ml/ch11_rl/fig/:
  mc_vs_td.pdf       -- when Monte Carlo updates vs when TD updates
  cliff_walking.pdf  -- the safe path and the optimal path, and why they differ

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/td_illustrations.py

Conventions (repo CLAUDE.md): console + logs/ logging, f-strings, Armenian-flag
colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN, GREY = "#2e7d32", "#666666"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_td_illustrations.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def fig_mc_vs_td():
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    n = 5
    xs = [i * 2.0 for i in range(n)]
    y_td, y_mc = 2.5, 0.0

    for y, color, name in [(y_td, BLUE, "TD(0)"), (y_mc, ORANGE, "Monte Carlo")]:
        for i, x in enumerate(xs):
            last = i == n - 1
            ax.add_patch(Rectangle((x - 0.42, y - 0.32), 0.84, 0.64,
                                   facecolor="0.35" if last else "white",
                                   edgecolor=color, lw=1.8, zorder=3))
            ax.text(x, y, "end" if last else f"$s_{i}$", ha="center", va="center",
                    fontsize=11, color="white" if last else "black", zorder=4)
            if not last:
                ax.annotate("", xy=(xs[i + 1] - 0.44, y), xytext=(x + 0.44, y),
                            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.4))
                ax.text((x + xs[i + 1]) / 2, y + 0.2, f"$r_{i}$", ha="center",
                        fontsize=9, color=GREY)
        ax.text(-2.6, y, name, ha="left", va="center", fontsize=12,
                color=color, fontweight="bold")

    # TD: an update fires immediately after every single step.
    for i, x in enumerate(xs[:-1]):
        ax.add_patch(FancyArrowPatch((x, y_td - 0.36), (x, y_td - 0.86),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=BLUE, lw=1.6))
        ax.text(x, y_td - 1.02, "update", ha="center", va="top", fontsize=8, color=BLUE)

    ax.text(sum(xs) / n, y_td - 1.45,
            "$V(s_i) \\leftarrow V(s_i) + \\alpha\\,[\\,r_i + \\gamma V(s_{i+1}) - V(s_i)\\,]$"
            "     one step of real reward, then a guess",
            ha="center", fontsize=9.5, color=BLUE)

    # MC: nothing happens until the episode ends, then every state updates at once.
    bus_y = y_mc - 0.9
    ax.plot([xs[0], xs[-1]], [bus_y, bus_y], color=ORANGE, lw=1.8)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x, bus_y), (x, y_mc - 0.36),
                                     arrowstyle="-|>", mutation_scale=11, color=ORANGE, lw=1.5))
    ax.add_patch(FancyArrowPatch((xs[-1], y_mc - 0.36), (xs[-1], bus_y),
                                 arrowstyle="-|>", mutation_scale=11, color=ORANGE, lw=1.8))
    ax.text(xs[-1] + 0.6, bus_y + 0.18, "the episode ends,\n$G_i$ becomes known",
            ha="left", va="center", fontsize=8.5, color=ORANGE)

    ax.text(sum(xs) / n, y_mc - 1.25,
            "$V(s_i) \\leftarrow V(s_i) + \\alpha\\,[\\,G_i - V(s_i)\\,]$"
            "     the whole actual return, no guessing",
            ha="center", fontsize=9.5, color=ORANGE)

    ax.text(-2.6, y_td + 1.0, "Same episode. The only question is when you are allowed to learn.",
            fontsize=11.5, color="0.2")

    ax.set_xlim(-2.9, 12.4)
    ax.set_ylim(-1.8, 4.0)
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "mc_vs_td.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cliff_walking():
    rows, cols = 4, 12
    fig, ax = plt.subplots(figsize=(9.6, 3.6))

    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], color="0.8", lw=1)
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], color="0.8", lw=1)

    # the cliff runs along the bottom row between start and goal
    ax.add_patch(Rectangle((1, 0), cols - 2, 1, facecolor=RED, alpha=0.75))
    ax.text(cols / 2, 0.5, "T H E   C L I F F      (reward $-100$, back to start)",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.85))
    ax.add_patch(Rectangle((cols - 1, 0), 1, 1, facecolor=GREEN, alpha=0.85))

    # The two paths share their endpoints, so offset the vertical runs to keep both visible.
    opt = [(0.62, 0.5), (0.62, 1.5), (cols - 0.62, 1.5), (cols - 0.62, 0.5)]
    ax.plot(*zip(*opt), color=ORANGE, lw=3.0, solid_capstyle="round", zorder=5)
    ax.text(cols / 2, 1.72, "optimal path -- what Q-learning learns", ha="center",
            fontsize=9.5, color=ORANGE, fontweight="bold")

    safe = [(0.38, 0.5), (0.38, 2.5), (cols - 0.38, 2.5), (cols - 0.38, 0.5)]
    ax.plot(*zip(*safe), color=BLUE, lw=3.0, solid_capstyle="round", zorder=5)
    ax.text(cols / 2, 2.72, "safe path -- what SARSA learns", ha="center",
            fontsize=9.5, color=BLUE, fontweight="bold")

    ax.text(0.5, 0.5, "S", ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", zorder=7)
    ax.text(cols - 0.5, 0.5, "G", ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", zorder=7)

    ax.text(cols / 2, 3.55,
            "Both agents explore with the same $\\epsilon$. They disagree because only one of "
            "them admits it.",
            ha="center", fontsize=10.5, color="0.2")

    ax.set_xlim(-0.2, cols + 0.2)
    ax.set_ylim(-0.2, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "cliff_walking.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    for out in (fig_mc_vs_td(), fig_cliff_walking()):
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
