"""Figures for the modern-components frames of ch5 (GELU, LayerNorm).

Outputs (sized for a 4:3 slide, large fonts, no titles - the slide supplies context):
    fig/gelu_vs_relu.pdf  - ReLU / GELU / SiLU shapes and their derivatives
    fig/norm_axes.pdf     - what BatchNorm normalises over vs what LayerNorm normalises over
    fig/lr_schedules.pdf  - constant / linear decay / warmup + inverse sqrt / warmup + cosine

    ./ma/Scripts/python.exe ml/11_neural_networks/py_src/modern_components_figs.py
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SEED = 509

# Armenian flag palette (3+ colours rule, CLAUDE.md).
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

CHAPTER = Path(__file__).resolve().parent.parent
REPO = CHAPTER.parent.parent


def setup_logging() -> logging.Logger:
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "modern_components_figs.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("modern_components_figs")


def gelu_figure(out: Path, log: logging.Logger) -> None:
    """ReLU vs GELU vs SiLU, and their derivatives. The point: GELU is smooth and its
    derivative is non-zero for negative inputs, so a unit that goes negative is not dead."""
    x = np.linspace(-4, 4, 800)

    relu = np.maximum(0.0, x)
    gelu = x * norm.cdf(x)                       # exact GELU: x * Phi(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    silu = x * sig

    d_relu = (x > 0).astype(float)
    d_gelu = norm.cdf(x) + x * norm.pdf(x)       # Phi(x) + x phi(x)
    d_silu = sig * (1.0 + x * (1.0 - sig))

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))

    axes[0].plot(x, relu, color=BLUE, lw=2.2, ls="--", label="ReLU")
    axes[0].plot(x, gelu, color=RED, lw=2.6, label="GELU")
    axes[0].plot(x, silu, color=ORANGE, lw=2.0, ls=":", label="SiLU")
    axes[0].set_ylabel("output", fontsize=12)

    # The whole point of the frame is what happens for negative inputs, and at full scale the
    # three curves look identical there. Mark the exact value the slide quotes.
    g_at_m1 = float(-1 * norm.cdf(-1))
    axes[0].plot([-1], [g_at_m1], "o", color=RED, ms=6, zorder=5)
    axes[0].plot([-1], [0.0], "o", color=BLUE, ms=5, zorder=5)
    axes[0].annotate(
        f"GELU$(-1)={g_at_m1:.2f}$\nReLU$(-1)=0$",
        xy=(-1, g_at_m1), xytext=(-3.85, 1.75), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )
    axes[0].set_ylim(-0.9, 4.2)

    axes[1].plot(x, d_relu, color=BLUE, lw=2.2, ls="--", label="ReLU")
    axes[1].plot(x, d_gelu, color=RED, lw=2.6, label="GELU")
    axes[1].plot(x, d_silu, color=ORANGE, lw=2.0, ls=":", label="SiLU")
    axes[1].set_ylabel("derivative", fontsize=12)

    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("z", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)
    log.info("GELU(-1)=%.4f  dGELU(-1)=%.4f  (ReLU and its derivative are both 0 there)",
             float(-1 * norm.cdf(-1)), float(norm.cdf(-1) + (-1) * norm.pdf(-1)))


def norm_axes_figure(out: Path, log: logging.Logger) -> None:
    """Which numbers get averaged: BatchNorm down the batch, LayerNorm across features."""
    rng = np.random.default_rng(SEED)
    n_rows, n_cols = 5, 6                     # 5 examples in the batch, 6 features each
    vals = rng.normal(0, 1, size=(n_rows, n_cols))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    for ax, mode in zip(axes, ("BatchNorm", "LayerNorm")):
        ax.imshow(vals, cmap="Greys", vmin=-3, vmax=3, alpha=0.25, aspect="auto")
        colour = RED if mode == "BatchNorm" else BLUE
        if mode == "BatchNorm":
            # one column = one feature, averaged over the whole batch
            ax.add_patch(Rectangle((1.5, -0.5), 1, n_rows, fill=False,
                                   edgecolor=colour, lw=3))
        else:
            # one row = one example, averaged over its own features
            ax.add_patch(Rectangle((-0.5, 1.5), n_cols, 1, fill=False,
                                   edgecolor=colour, lw=3))
        ax.set_title(mode, fontsize=12, color=colour, fontweight="bold")
        ax.set_xlabel("features", fontsize=11)
        ax.set_ylabel("batch", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def lr_schedule_figure(out: Path, log: logging.Logger) -> None:
    """Schedule SHAPES on an illustrative 10k-step run (warmup stretched to 4% so it is visible;
    the papers' real step counts are quoted on the slide, not drawn)."""
    total, warm = 10_000, 400
    t = np.arange(1, total + 1)

    constant = np.ones_like(t, dtype=float)
    linear = 1.0 - 0.9 * (t - 1) / (total - 1)                        # LMU's linear decay to 10%
    inv_sqrt = np.minimum(t / warm, np.sqrt(warm / t))                # Vaswani et al. (2017) shape
    progress = np.clip((t - warm) / (total - warm), 0.0, 1.0)
    cosine = np.where(t < warm, t / warm, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axvspan(0, warm, color=ORANGE, alpha=0.15, lw=0)
    ax.text(warm * 1.15, 0.05, "warmup", fontsize=9, color="gray")
    ax.plot(t, constant, color="gray", lw=1.6, ls=":", label="constant")
    ax.plot(t, linear, color=BLUE, lw=2.0, ls="--", label="linear decay (LMU)")
    ax.plot(t, inv_sqrt, color=ORANGE, lw=2.2, label="warmup + inverse sqrt (Vaswani)")
    ax.plot(t, cosine, color=RED, lw=2.6, label="warmup + cosine to 10% (Llama 2)")
    ax.set_xlabel("training step", fontsize=12)
    ax.set_ylabel("learning rate / peak", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, total)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    # Legend above the axes: inside, every corner is crossed by at least one curve.
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    log.info("schedules at step %d: linear=%.2f inv_sqrt=%.2f cosine=%.2f",
             total, linear[-1], inv_sqrt[-1], cosine[-1])


def main() -> None:
    log = setup_logging()
    fig_dir = CHAPTER / "fig"
    fig_dir.mkdir(exist_ok=True)
    gelu_figure(fig_dir / "gelu_vs_relu.pdf", log)
    norm_axes_figure(fig_dir / "norm_axes.pdf", log)
    lr_schedule_figure(fig_dir / "lr_schedules.pdf", log)
    log.info("done")


if __name__ == "__main__":
    main()
