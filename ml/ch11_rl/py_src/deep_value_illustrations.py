"""Illustrations for L32c (deck 3: deep value-based RL).

Teaching diagrams, not experiments. The deadly triad is a three-way overlap that only
a Venn diagram makes memorable, and DQN's two fixes are easier to see than to read.

Generates into ml/ch11_rl/fig/:
  deadly_triad.pdf -- the three ingredients, and the region where they diverge
  dqn_loop.pdf     -- where the replay buffer and the target network sit

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/deep_value_illustrations.py

Conventions (repo CLAUDE.md): console + logs/ logging, f-strings, Armenian-flag
colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN, GREY = "#2e7d32", "#555555"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_deep_value.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def fig_deadly_triad():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    r = 1.65
    centres = {"bootstrapping": (-0.85, 0.55), "function\napproximation": (0.85, 0.55),
               "off-policy": (0.0, -0.85)}
    colors = {"bootstrapping": BLUE, "function\napproximation": ORANGE, "off-policy": RED}
    label_xy = {"bootstrapping": (-2.15, 1.75), "function\napproximation": (2.15, 1.75),
                "off-policy": (0.0, -2.75)}
    detail = {
        "bootstrapping": "targets built from\nyour own estimates\n(TD, Q-learning)",
        "function\napproximation": "one weight vector\nshared across states\n(a neural net)",
        "off-policy": "learning about a policy\nyou are not following\n(the max, replay)",
    }

    for name, (cx, cy) in centres.items():
        ax.add_patch(Circle((cx, cy), r, facecolor=colors[name], alpha=0.16,
                            edgecolor=colors[name], lw=2.0))
        lx, ly = label_xy[name]
        ha = "center" if name == "off-policy" else ("right" if lx < 0 else "left")
        ax.text(lx, ly, name.replace("\n", " "), fontsize=12, color=colors[name],
                fontweight="bold", ha=ha, va="center")
        ax.text(lx, ly - 0.42, detail[name], fontsize=8, color=GREY, ha=ha, va="top")

    ax.text(0.0, 0.05, "danger", fontsize=13, color="black", fontweight="bold",
            ha="center", va="center")
    ax.text(0.0, -0.32, "values can\ndiverge", fontsize=8.5, color="black",
            ha="center", va="top")

    ax.text(0, 3.05, "The deadly triad", fontsize=14, color=RED, fontweight="bold",
            ha="center")
    ax.text(0, 2.62, "Any two are safe. All three together, and there is no convergence "
                     "guarantee at all.",
            fontsize=9.5, color="0.25", ha="center")

    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-3.6, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "deadly_triad.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _box(ax, xy, w, h, text, color, fontsize=9, fill=0.10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.06", facecolor=color, alpha=fill,
                                edgecolor=color, lw=1.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="black")


def _arrow(ax, a, b, color, text=None, rad=0.0, text_dy=0.22, fontsize=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + text_dy, text,
                ha="center", fontsize=fontsize, color=color)


def fig_dqn_loop():
    fig, ax = plt.subplots(figsize=(10.4, 4.6))

    _box(ax, (1.1, 2.1), 2.0, 0.85, "environment", GREEN)
    _box(ax, (4.5, 2.1), 2.5, 0.95, "replay buffer\n$(s,a,r,s')$, millions", ORANGE)
    _box(ax, (8.1, 2.6), 2.3, 0.95, "$Q_\\theta$\nonline network", BLUE)
    _box(ax, (8.1, 0.5), 2.3, 0.95, "$Q_{\\theta^-}$\nfrozen copy", RED)

    _arrow(ax, (2.15, 2.1), (3.2, 2.1), GREEN, "store every step", text_dy=0.14)
    _arrow(ax, (5.8, 2.2), (6.9, 2.5), ORANGE, "random\nminibatch", text_dy=0.26)
    _arrow(ax, (8.1, 1.0), (8.1, 2.1), RED, None)
    ax.text(7.9, 1.55, "supplies the target\n$y = r + \\gamma \\max_{a'} Q_{\\theta^-}(s',a')$",
            fontsize=8, color=RED, ha="right", va="center")
    _arrow(ax, (9.3, 2.5), (9.3, 0.7), GREY, None, rad=-0.55)
    ax.text(10.0, 1.6, "copy $\\theta$ into $\\theta^-$\nevery $C$ steps",
            fontsize=8, color=GREY, ha="center", va="center")
    # Route the action back over the top so it does not cut through the buffer box.
    ax.plot([8.1, 8.1, 1.1], [3.08, 3.95, 3.95], color=BLUE, lw=1.6, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((1.1, 3.95), (1.1, 2.56), arrowstyle="-|>",
                                 mutation_scale=13, color=BLUE, lw=1.6))
    ax.text(4.6, 4.08, "$\\epsilon$-greedy action", fontsize=8.5, color=BLUE, ha="center")

    ax.text(4.6, 4.62, "DQN: the two boxes that are not in tabular Q-learning",
            fontsize=12.5, color="0.15", ha="center")
    ax.text(4.6, -0.55,
            "Replay breaks the correlation between consecutive frames.  "
            "The frozen copy stops the network chasing its own output.",
            fontsize=9.5, color="0.3", ha="center")

    ax.set_xlim(-0.5, 11.4)
    ax.set_ylim(-0.9, 5.0)
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "dqn_loop.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    for out in (fig_deadly_triad(), fig_dqn_loop()):
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
