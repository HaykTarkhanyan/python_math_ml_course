"""Real figure for the L17 CNN Architectures deck (Section 2, VGG / receptive field).

Shows how the receptive field of one output neuron grows when 3x3 convolutions are
stacked: after 1 layer it sees 3x3, after 2 layers 5x5, after 3 layers 7x7 - the same
reach as a single 7x7 filter, but built from small filters (deeper, cheaper, more
nonlinearities). Includes the parameter-coefficient comparison (27 C^2 vs 49 C^2).

Generates into ml/ch6_cnn/fig/:
  receptive_field.pdf

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/receptive_field.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_receptive_field")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "receptive_field.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- left: nested receptive fields on a 7x7 grid ----
    N = 7
    for i in range(N):
        for j in range(N):
            axL.add_patch(Rectangle((j, i), 1, 1, fc="#f2f2f2", ec="#dddddd", lw=0.8))
    center = N // 2
    axL.add_patch(Rectangle((center, center), 1, 1, fc="#cccccc", ec="#999", lw=0.8))

    # squares of size 3,5,7 centered on the grid center cell
    specs = [(3, ARM_RED, "after 1 layer: 3x3"),
             (5, ARM_BLUE, "after 2 layers: 5x5"),
             (7, ARM_ORANGE, "after 3 layers: 7x7")]
    for size, color, _ in specs:
        half = size // 2
        x0 = center - half
        y0 = center - half
        axL.add_patch(Rectangle((x0, y0), size, size, fc="none", ec=color, lw=3.0))
    axL.set_xlim(-0.3, N + 0.3)
    axL.set_ylim(-1.2, N + 0.3)
    axL.set_aspect("equal")
    axL.axis("off")
    axL.set_title("Receptive field grows with depth\n(stacked 3x3 convolutions)",
                  fontsize=12)
    # legend below the grid
    for idx, (size, color, label) in enumerate(specs):
        y = -0.55 - idx * 0.42
        axL.plot([0.2, 0.9], [y + 0.15, y + 0.15], color=color, lw=3.0)
        axL.text(1.1, y + 0.15, label, color=color, fontsize=10, va="center")

    # ---- right: parameter-coefficient comparison ----
    labels = ["three 3x3\nlayers", "one 7x7\nlayer"]
    coeffs = [27, 49]   # 3*(3*3)=27 C^2  vs  7*7 = 49 C^2
    colors = [ARM_BLUE, ARM_RED]
    bars = axR.bar(labels, coeffs, width=0.6, color=colors)
    axR.bar_label(bars, labels=[f"{c} C²" for c in coeffs], padding=4,
                  fontsize=13, fontweight="bold")
    axR.set_ylim(0, 58)
    axR.set_ylabel("weights per channel-pair (coefficient of C²)", fontsize=10.5)
    axR.set_title("Same 7x7 reach, fewer weights", fontsize=12)
    axR.grid(axis="y", alpha=0.2)
    axR.set_axisbelow(True)
    axR.text(0.04, 0.94, "+ two extra ReLUs for free", transform=axR.transAxes,
             ha="left", va="top", fontsize=10, color=ARM_BLUE, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "receptive_field.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
