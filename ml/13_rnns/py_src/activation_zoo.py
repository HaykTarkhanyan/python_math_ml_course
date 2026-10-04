"""Real figure for the L20 RNN Foundations deck ("Picking the activation: why tanh").

Generates into ml/13_rnns/fig/:
  activation_zoo.pdf -- tanh, sigmoid and ReLU plotted together, tanh highlighted
                        (thicker line) since it is the RNN default. Both bounded
                        curves' asymptotes are marked; ReLU's unboundedness is the
                        visual contrast.

Run with the project venv:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/activation_zoo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Armenian flag palette (3 colors): tanh highlighted in blue, sigmoid red, ReLU orange.
BLUE, RED, ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_activation_zoo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "activation_zoo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def fig_activation_zoo(log):
    x = np.linspace(-4, 4, 400)

    # drawn at its slot size (L20: 0.52 column x 0.92 = ~2.65 in) so fonts are true size
    fig, ax = plt.subplots(figsize=(2.65, 2.15))
    ax.plot(x, relu(x), color=ORANGE, lw=1.3, label="ReLU: unbounded")
    ax.plot(x, sigmoid(x), color=RED, lw=1.3, label="sigmoid: (0, 1)")
    ax.plot(x, np.tanh(x), color=BLUE, lw=2.2, label="tanh: (-1, 1), RNN default")

    ax.axhline(1.0, color=BLUE, lw=0.8, linestyle=":", alpha=0.6)
    ax.axhline(-1.0, color=BLUE, lw=0.8, linestyle=":", alpha=0.6)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.axvline(0.0, color="gray", lw=0.6)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.6, 4)
    ax.set_xlabel("pre-activation", fontsize=7.5)
    ax.set_ylabel("output", fontsize=7.5)
    ax.set_title("tanh: bounded and zero-centered", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "activation_zoo.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_activation_zoo(log)
    log.info("done")


if __name__ == "__main__":
    main()
