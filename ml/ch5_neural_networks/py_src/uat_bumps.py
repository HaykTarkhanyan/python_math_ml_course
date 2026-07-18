"""Real figure for the dl_multilayer_nets deck (universal approximation section).

Generates into ml/ch5_neural_networks/fig/:
  uat_bumps.pdf -- the intuition behind the universal approximation theorem: each hidden
                   neuron pair contributes one localized "bump"
                   [ u * (sigma(a(x-l)) - sigma(a(x-r))) ], and summing bumps traces any
                   continuous curve. More neurons -> closer fit ("m -> infinity" in the theorem).
                   Left: the individual bumps. Right: their sum vs the target, for 6 and 20 bumps.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/uat_bumps.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED, ARMORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

X0, X1 = 0.0, 6.0


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("uat_bumps")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "uat_bumps.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def target(x):
    return 0.9 * np.sin(1.1 * x) + 0.35 * np.sin(2.7 * x)


def bumps(x, K):
    """Return the list of K localized bumps whose sum approximates `target`."""
    edges = np.linspace(X0, X1, K + 1)
    a = 1.4 * K                      # step sharpness scales with bin count
    out = []
    for j in range(K):
        l, r = edges[j], edges[j + 1]
        h = target(0.5 * (l + r))
        out.append(h * (sigmoid(a * (x - l)) - sigmoid(a * (x - r))))
    return out


def fig_bumps(log):
    x = np.linspace(X0, X1, 800)
    g = target(x)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # ---- left: the individual bumps (K = 6) ----
    palette = [ARMBLUE, ARMORANGE, ARMRED]
    for j, b in enumerate(bumps(x, 6)):
        c = palette[j % len(palette)]
        axL.plot(x, b, color=c, lw=1.6)
        axL.fill_between(x, 0, b, color=c, alpha=0.12)
    axL.plot(x, g, color="#888888", lw=1.4, ls="--", label="target $g(x)$")
    axL.axhline(0, color="#bbbbbb", lw=0.7)
    axL.set_title("Each neuron pair adds one bump  (6 shown)", fontsize=12.5)
    axL.legend(loc="upper right", fontsize=9, frameon=False)

    # ---- right: sums vs target, 6 and 20 bumps ----
    s6 = np.sum(bumps(x, 6), axis=0)
    s20 = np.sum(bumps(x, 20), axis=0)
    axR.plot(x, g, color="black", lw=2.2, ls="--", label="target $g(x)$")
    axR.plot(x, s6, color=ARMORANGE, lw=1.8, label="sum of 6 bumps")
    axR.plot(x, s20, color=ARMRED, lw=1.8, label="sum of 20 bumps")
    axR.set_title("Their sum approximates the target", fontsize=12.5)
    axR.legend(loc="upper right", fontsize=9, frameon=False)
    log.info(f"max|g - sum6| = {np.max(np.abs(g - s6)):.3f}, "
             f"max|g - sum20| = {np.max(np.abs(g - s20)):.3f}")

    for ax in (axL, axR):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_xlim(X0, X1)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel("$x$", fontsize=11)
        ax.grid(color="#EEEEEE", lw=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Universal approximation, by hand: any continuous curve is a sum of bumps",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "uat_bumps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_bumps(log)
    log.info("done")


if __name__ == "__main__":
    main()
