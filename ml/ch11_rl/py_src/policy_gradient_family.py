"""The policy-gradient family tree, for L32d (deck 4) and recalled in L32h (deck 7).

A teaching diagram, not an experiment. The point it makes: every method students will
meet in the LLM lectures is one edit away from REINFORCE, and the edits are all
variance reduction or step-size control.

Generates into ml/ch11_rl/fig/:
  pg_family.pdf

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/policy_gradient_family.py

Conventions (repo CLAUDE.md): console + logs/ logging, f-strings, Armenian-flag
colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
VIOLET, GREY = "#7832A0", "#555555"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

# (x, y, title, subtitle, colour). The spine runs left to right; continuous control
# drops down from actor-critic, LLM methods drop down from PPO.
NODES = [
    (0.0, 2.0, "REINFORCE", "1992\nplay, score, push", GREY),
    (2.45, 2.0, "+ baseline", "subtract $b(s)$\nvariance, not bias", GREY),
    (4.9, 2.0, "actor-critic", "learn the baseline\nA2C / A3C", BLUE),
    (7.35, 2.0, "TRPO", "2015\nKL trust region", BLUE),
    (9.8, 2.0, "PPO", "2017\nclip the ratio", RED),
    (4.9, 0.15, "DDPG / TD3 / SAC", "continuous control\ndeterministic, entropy", ORANGE),
    (9.8, 0.15, "GRPO", "2024\ndrop the critic", VIOLET),
    (12.25, 0.15, "DAPO / GSPO", "2025\npatch what broke", VIOLET),
]

EDGES = [
    (0, 1, "the gradient is\ntoo noisy"),
    (1, 2, "where does\n$b(s)$ come from?"),
    (2, 3, "one bad step\nis unrecoverable"),
    (3, 4, "second-order\nis too expensive"),
    (2, 5, None),
    (4, 6, None),
    (6, 7, None),
]


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_pg_family.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def fig_family():
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    w, h = 2.0, 0.95

    for x, y, title, sub, color in NODES:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.07", facecolor=color, alpha=0.12,
                                    edgecolor=color, lw=2.0))
        ax.text(x, y + 0.18, title, ha="center", va="center", fontsize=11,
                color=color, fontweight="bold")
        ax.text(x, y - 0.22, sub, ha="center", va="center", fontsize=7.5, color="0.25")

    for i, j, label in EDGES:
        xi, yi = NODES[i][0], NODES[i][1]
        xj, yj = NODES[j][0], NODES[j][1]
        if yi == yj:
            a, b = (xi + w / 2, yi), (xj - w / 2, yj)
        else:
            a, b = (xi, yi - h / 2), (xj, yj + h / 2)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     color="0.45", lw=1.6))
        if label:
            ax.text((a[0] + b[0]) / 2, yi + 0.72, label, ha="center", va="center",
                    fontsize=7.5, color="0.4", style="italic")

    ax.text(-1.15, 3.25, "Every one of these is REINFORCE plus one edit.",
            fontsize=12.5, color="0.15", ha="left")
    ax.text(-1.15, 2.92,
            "The edits do two jobs only: reduce the variance of the gradient, "
            "or stop the step being too big.",
            fontsize=9, color="0.4", ha="left")

    ax.text(-1.15, 0.15, "control\n(robots, games)", fontsize=8.5, color=ORANGE,
            ha="left", va="center", fontweight="bold")
    ax.text(13.45, 1.05, "language\nmodels", fontsize=8.5, color=VIOLET,
            ha="center", va="center", fontweight="bold")

    ax.set_xlim(-1.3, 13.9)
    ax.set_ylim(-0.6, 3.5)
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "pg_family.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    log.info(f"wrote {fig_family().relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
