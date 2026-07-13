"""Real figure for the L20 RNN Foundations deck ("Picking the activation: why tanh").

Generates into ml/ch7_rnn/fig/:
  activation_zoo.pdf -- tanh, sigmoid and ReLU plotted together, tanh highlighted
                        (thicker line) since it is the RNN default. Both bounded
                        curves' asymptotes are marked; ReLU's unboundedness is the
                        visual contrast.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/activation_zoo.py
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

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x, relu(x), color=ORANGE, lw=1.8, label="ReLU (unbounded)")
    ax.plot(x, sigmoid(x), color=RED, lw=1.8, label="sigmoid (bounded 0 to 1)")
    ax.plot(x, np.tanh(x), color=BLUE, lw=3.0, label="tanh (bounded -1 to 1) - RNN default")

    ax.axhline(1.0, color=BLUE, lw=0.8, linestyle=":", alpha=0.6)
    ax.axhline(-1.0, color=BLUE, lw=0.8, linestyle=":", alpha=0.6)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.axvline(0.0, color="gray", lw=0.6)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.6, 4)
    ax.set_xlabel("pre-activation", fontsize=11)
    ax.set_ylabel("activation output", fontsize=11)
    ax.set_title("tanh stays bounded and zero-centered; ReLU does not", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "activation_zoo.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_activation_zoo(log)
    log.info("done")


if __name__ == "__main__":
    main()
