"""Real figure for the L20 RNN Foundations deck ("Predict-first: the fading number").

Generates into ml/ch7_rnn/fig/:
  vanish_curve.pdf -- 0.8^k (vanishing) and 1.25^k (exploding) vs k, log scale,
                      both curves labeled, k=29 annotated with its exact value.

LOCKED worked numbers: 0.8^29 ~ 0.0015 (vanishing), 1.25^29 ~ 646 (exploding).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/vanish_curve.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V_VANISH, V_EXPLODE = 0.8, 1.25
K_MARK = 29
BLUE, RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_vanish_curve")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "vanish_curve.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def fig_vanish_curve(log):
    k = np.arange(0, 31)
    vanish = V_VANISH ** k
    explode = V_EXPLODE ** k
    v_mark = V_VANISH ** K_MARK
    e_mark = V_EXPLODE ** K_MARK
    log.info(f"0.8^{K_MARK} = {v_mark:.6f}")
    log.info(f"1.25^{K_MARK} = {e_mark:.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(k, explode, color=RED, lw=2.2, label=f"$V={V_EXPLODE}$ (exploding)")
    ax.plot(k, vanish, color=BLUE, lw=2.2, label=f"$V={V_VANISH}$ (vanishing)")
    ax.set_yscale("log")
    ax.axvline(K_MARK, color="gray", lw=1.0, linestyle="--")

    ax.annotate(f"{V_EXPLODE}$^{{{K_MARK}}}\\approx${e_mark:.0f}",
                xy=(K_MARK, e_mark), xytext=(K_MARK - 13, e_mark * 1.6),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.annotate(f"{V_VANISH}$^{{{K_MARK}}}\\approx${v_mark:.4f}",
                xy=(K_MARK, v_mark), xytext=(K_MARK - 15, v_mark * 6),
                fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

    ax.set_xlabel("k (steps back)", fontsize=11)
    ax.set_ylabel("$V^k$ (log scale)", fontsize=11)
    ax.set_title("One multiplication, repeated: starves or explodes", fontsize=12)
    ax.legend(fontsize=10, loc="center left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "vanish_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_vanish_curve(log)
    log.info("done")


if __name__ == "__main__":
    main()
