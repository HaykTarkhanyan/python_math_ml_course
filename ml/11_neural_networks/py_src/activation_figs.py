"""Figures for the activation-equation frames of xx_optimization_init_activations.

Generates into ml/11_neural_networks/fig/:
  act_sigmoid_tanh.pdf  -- sigmoid and tanh (left) and their derivatives (right).
  act_relu_family.pdf   -- ReLU, Leaky ReLU, ELU, SELU, GELU (left) and derivatives (right).
                           Leaky ReLU is drawn with slope 0.1 (PyTorch's default 0.01 would be
                           invisible at slide scale); ELU alpha = 1 and the SELU constants are
                           PyTorch's defaults.

Run with the project venv:
    ./ma/Scripts/python.exe ml/11_neural_networks/py_src/activation_figs.py
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 509  # nothing random here; kept for the course convention
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
LEAKY_PLOT_SLOPE = 0.1
SELU_ALPHA, SELU_SCALE = 1.6732632423543772, 1.0507009873554805   # torch.nn.SELU constants

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]


def setup_logging() -> logging.Logger:
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "activation_figs.log", encoding="utf-8")],
    )
    return logging.getLogger("activation_figs")


def style(axes) -> None:
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("z", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3)


def sigmoid_tanh(out: Path, log: logging.Logger) -> None:
    z = np.linspace(-5, 5, 1000)
    sig = 1 / (1 + np.exp(-z))
    th = np.tanh(z)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axes[0].plot(z, sig, color=BLUE, lw=2.4, label=r"$\sigma(z)$")
    axes[0].plot(z, th, color=RED, lw=2.4, label=r"$\tanh(z)$")
    axes[0].set_ylabel("output", fontsize=10)
    axes[1].plot(z, sig * (1 - sig), color=BLUE, lw=2.4, label=r"$\sigma'$ (max $0.25$)")
    axes[1].plot(z, 1 - th ** 2, color=RED, lw=2.4, label=r"$\tanh'$ (max $1$)")
    axes[1].set_ylabel("derivative", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    # right half is empty above 0.2; short handles keep the legend clear of the tanh' peak
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.03, 0.8),
                   handlelength=1.0)
    style(axes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}; max sigma' = {np.max(sig * (1 - sig)):.4f}, max tanh' = {np.max(1 - th ** 2):.4f}")
    log.info(f"check tanh(z) = 2 sigma(2z) - 1: max |diff| = {np.max(np.abs(th - (2 / (1 + np.exp(-2 * z)) - 1))):.1e}")


def relu_family(out: Path, log: logging.Logger) -> None:
    z = np.linspace(-4, 3, 1400)
    elu = np.where(z >= 0, z, np.exp(z) - 1)
    curves = {
        "ReLU": (np.maximum(0, z), (z > 0).astype(float), BLUE, "-"),
        f"Leaky ReLU (slope {LEAKY_PLOT_SLOPE})": (np.where(z >= 0, z, LEAKY_PLOT_SLOPE * z),
                                                 np.where(z >= 0, 1.0, LEAKY_PLOT_SLOPE), ORANGE, "--"),
        "ELU": (elu, np.where(z >= 0, 1.0, np.exp(z)), "gray", "-."),
        "SELU": (SELU_SCALE * np.where(z >= 0, z, SELU_ALPHA * (np.exp(z) - 1)),
                 SELU_SCALE * np.where(z >= 0, 1.0, SELU_ALPHA * np.exp(z)), "#6A3D9A", ":"),
        "GELU": (z * norm.cdf(z), norm.cdf(z) + z * norm.pdf(z), RED, "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for name, (f, df, color, ls) in curves.items():
        axes[0].plot(z, f, color=color, ls=ls, lw=2.0, label=name)
        axes[1].plot(z, df, color=color, ls=ls, lw=2.0, label=name)
        log.info(f"{name}: min output {f.min():.3f}")
    axes[0].set_ylabel("output", fontsize=10)
    axes[0].set_ylim(-2.0, 3.1)
    axes[1].set_ylabel("derivative", fontsize=10)
    axes[1].set_ylim(-0.2, 1.9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    style(axes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main() -> None:
    log = setup_logging()
    fig_dir = CH_DIR / "fig"
    fig_dir.mkdir(exist_ok=True)
    sigmoid_tanh(fig_dir / "act_sigmoid_tanh.pdf", log)
    relu_family(fig_dir / "act_relu_family.pdf", log)
    log.info("done")


if __name__ == "__main__":
    main()
