"""Figures for ch16 (JEPA) - decks L39 and L40.

Every number plotted here is transcribed from a primary source; nothing is measured
or simulated. Provenance per figure is in the docstring of its builder.

Run:  ./ma/Scripts/python.exe ml/ch16_jepa/py_src/ch16_figs.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "ch16_figs.log")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


# ----------------------------------------------------------------------------
def three_architectures():
    """The chapter's reference diagram: JEA / generative / JEPA side by side.

    Not data - a schematic. Drawn in Python rather than TikZ per house style.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    def box(ax, xy, w, h, label, fc="white", ec=GREY, fs=9):
        ax.add_patch(Rectangle(xy, w, h, fc=fc, ec=ec, lw=1.3, zorder=2))
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=3)

    def arrow(ax, a, b, color=GREY, style="-|>"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11,
                                     lw=1.2, color=color, zorder=1))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # --- 1. joint-embedding architecture -------------------------------------
    ax = axes[0]
    ax.set_title("Joint-embedding\n(CLIP, DINO)", fontsize=11, color=ARM_BLUE, pad=6)
    box(ax, (0.6, 0.6), 2.6, 1.1, "$x$")
    box(ax, (6.8, 0.6), 2.6, 1.1, "$y$")
    box(ax, (0.6, 3.4), 2.6, 1.3, "Encoder", fc="#eef2fa")
    box(ax, (6.8, 3.4), 2.6, 1.3, "Encoder", fc="#eef2fa")
    arrow(ax, (1.9, 1.7), (1.9, 3.4)); arrow(ax, (8.1, 1.7), (8.1, 3.4))
    arrow(ax, (1.9, 4.7), (1.9, 6.4)); arrow(ax, (8.1, 4.7), (8.1, 6.4))
    box(ax, (0.6, 6.4), 2.6, 1.1, "$s_x$", fc="#f7f7f7")
    box(ax, (6.8, 6.4), 2.6, 1.1, "$s_y$", fc="#f7f7f7")
    ax.annotate("", xy=(6.8, 6.95), xytext=(3.2, 6.95),
                arrowprops=dict(arrowstyle="<|-|>", color=ARM_BLUE, lw=1.6))
    ax.text(5.0, 7.5, "make them\nagree", ha="center", fontsize=9, color=ARM_BLUE)
    ax.text(5.0, 9.1, "answers: do these match?", ha="center", fontsize=9,
            style="italic", color=GREY)

    # --- 2. generative -------------------------------------------------------
    ax = axes[1]
    ax.set_title("Generative\n(autoencoder, MAE, next token)", fontsize=11,
                 color=ARM_ORANGE, pad=6)
    box(ax, (0.6, 0.6), 2.6, 1.1, "$x$")
    box(ax, (0.6, 3.4), 2.6, 1.3, "Encoder", fc="#fdf3e0")
    box(ax, (4.0, 3.4), 2.6, 1.3, "Decoder", fc="#fdf3e0")
    box(ax, (7.2, 3.4), 2.4, 1.3, "$\\hat{y}$", fc="#f7f7f7")
    box(ax, (7.2, 0.6), 2.4, 1.1, "$y$")
    arrow(ax, (1.9, 1.7), (1.9, 3.4))
    arrow(ax, (3.2, 4.05), (4.0, 4.05))
    arrow(ax, (6.6, 4.05), (7.2, 4.05))
    ax.annotate("", xy=(8.4, 3.4), xytext=(8.4, 1.7),
                arrowprops=dict(arrowstyle="<|-|>", color=ARM_ORANGE, lw=1.6))
    ax.text(9.6, 2.6, "loss in\ninput space", ha="right", fontsize=9, color=ARM_ORANGE)
    ax.text(5.0, 9.1, "pays for every unpredictable pixel", ha="center", fontsize=9,
            style="italic", color=GREY)

    # --- 3. JEPA -------------------------------------------------------------
    ax = axes[2]
    ax.set_title("JEPA\n(I-JEPA, V-JEPA)", fontsize=11, color=ARM_RED, pad=6)
    box(ax, (0.4, 0.6), 2.4, 1.1, "$x$")
    box(ax, (7.0, 0.6), 2.4, 1.1, "$y$")
    box(ax, (0.4, 3.0), 2.4, 1.3, "Encoder", fc="#fbecec")
    box(ax, (7.0, 3.0), 2.4, 1.3, "Encoder", fc="#fbecec")
    arrow(ax, (1.6, 1.7), (1.6, 3.0)); arrow(ax, (8.2, 1.7), (8.2, 3.0))
    arrow(ax, (1.6, 4.3), (1.6, 5.6)); arrow(ax, (8.2, 4.3), (8.2, 5.6))
    box(ax, (0.4, 5.6), 2.4, 1.0, "$s_x$", fc="#f7f7f7")
    box(ax, (7.0, 5.6), 2.4, 1.0, "$s_y$", fc="#f7f7f7")
    box(ax, (3.5, 5.55), 2.8, 1.1, "Predictor", fc="#fbecec")
    arrow(ax, (2.8, 6.1), (3.5, 6.1))
    arrow(ax, (6.3, 6.1), (7.0, 6.1), color=ARM_RED)
    ax.text(4.9, 4.9, "$z$", ha="center", fontsize=10, color=GREY)
    arrow(ax, (4.9, 5.15), (4.9, 5.55), color=GREY)
    ax.text(4.9, 7.5, "loss in\nrepresentation space", ha="center", fontsize=9,
            color=ARM_RED)
    ax.text(5.0, 9.1, "predicts a target it invented", ha="center", fontsize=9,
            style="italic", color=GREY)

    save(fig, "three_architectures")


# ----------------------------------------------------------------------------
def ablation_masking():
    """I-JEPA Table 6. ViT-B/16, 300 epochs, linear eval on 1% ImageNet."""
    names = ["multi-block\n(proposed)", "block", "random", "rasterized"]
    vals = [54.2, 20.2, 17.6, 15.5]
    ctx = ["25%", "40%", "40%", "25%"]
    colors = [ARM_RED, GREY, GREY, ARM_BLUE]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    bars = ax.bar(names, vals, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=11, fontweight="bold")
    ax.set_ylabel("1% ImageNet top-1")
    ax.set_ylim(0, 63)
    for i, c in enumerate(ctx):
        ax.text(i, 1.6, f"context {c}", ha="center", fontsize=8.5, color="white")
    ax.set_title("Same architecture, same compute. Only the masking changes.",
                 fontsize=10.5, color=GREY)
    save(fig, "ablation_masking")


def ablation_target():
    """I-JEPA Table 7. ViT-L/16, linear eval on 1% ImageNet."""
    names = ["target-encoder output\n(500 epochs)", "pixels\n(800 epochs)"]
    vals = [66.9, 40.7]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    bars = ax.bar(names, vals, color=[ARM_RED, ARM_BLUE], width=0.55)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=12, fontweight="bold")
    ax.set_ylabel("1% ImageNet top-1")
    ax.set_ylim(0, 78)
    ax.annotate("", xy=(1, 42.5), xytext=(1, 65.4),
                arrowprops=dict(arrowstyle="<|-|>", color=ARM_ORANGE, lw=2))
    ax.text(1.12, 54, "26.2\npoints", fontsize=11, color=ARM_ORANGE,
            fontweight="bold", va="center")
    ax.set_title("The pixel run got 60% more training.", fontsize=10.5, color=GREY)
    save(fig, "ablation_target")


# ----------------------------------------------------------------------------
def efficiency():
    """I-JEPA Table 1: linear ImageNet-1k vs pretraining epochs.

    Epoch counts and accuracies are the paper's; the x axis is epochs, NOT
    GPU-hours, so it understates I-JEPA's advantage rather than overstating it.
    """
    pts = [
        ("MAE ViT-H/14",      1600, 77.2, ARM_BLUE),
        ("data2vec ViT-L/16", 1600, 77.3, ARM_BLUE),
        ("CAE ViT-L/16",      1600, 78.1, ARM_BLUE),
        ("iBOT ViT-L/16",      250, 81.0, ARM_ORANGE),
        ("DINO ViT-B/8",       300, 80.1, ARM_ORANGE),
        ("I-JEPA ViT-H/14",    300, 79.3, ARM_RED),
        ("I-JEPA ViT-L/16",    600, 77.5, ARM_RED),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for name, ep, acc, c in pts:
        ax.scatter(ep, acc, s=95, color=c, zorder=3,
                   marker="o" if c != ARM_ORANGE else "^")
        dx, ha = (-40, "right") if ep > 900 else (40, "left")
        ax.annotate(name, (ep, acc), textcoords="offset points",
                    xytext=(dx / 4, 7), ha=ha, fontsize=8.5)
    ax.set_xlabel("pretraining epochs")
    ax.set_ylabel("ImageNet-1k linear top-1")
    ax.set_xlim(100, 1850)
    ax.set_ylim(76.3, 82.3)
    ax.scatter([], [], color=ARM_RED, label="I-JEPA")
    ax.scatter([], [], color=ARM_BLUE, label="no augmentations")
    ax.scatter([], [], color=ARM_ORANGE, marker="^",
               label="uses hand-crafted augmentations")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("I-JEPA gets there in fewer epochs. It does not end up highest.",
                 fontsize=10.5, color=GREY)
    save(fig, "efficiency")


# ----------------------------------------------------------------------------
def energy_landscape():
    """Schematic: a sculpted energy landscape vs a collapsed (flat) one."""
    rng = np.random.default_rng(SEED)
    x = np.linspace(-3, 3, 400)
    sculpted = 0.9 * (x ** 2 - 1.6) ** 2 + 0.25
    flat = np.full_like(x, 0.30) + 0.004 * rng.standard_normal(x.size)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), sharey=True)
    for ax, y, title, c in [
        (axes[0], sculpted, "What you want", ARM_RED),
        (axes[1], flat, "Collapse", ARM_BLUE),
    ]:
        ax.plot(x, y, color=c, lw=2.4)
        ax.fill_between(x, y, 6.5, color=c, alpha=0.06)
        ax.set_xlabel("all possible $(x, y)$ pairs")
        ax.set_title(title, fontsize=11, color=c)
        ax.set_xticks([])
        ax.set_ylim(0, 6.5)
    axes[0].set_ylabel("energy  $F(x,y)$")
    for xd in (-1.265, 1.265):
        axes[0].scatter([xd], [0.25], s=55, color=ARM_ORANGE, zorder=4)
    axes[0].text(0, 3.4, "high off the data", ha="center", fontsize=9, color=GREY)
    axes[0].text(1.35, 1.0, "low on\nobserved pairs", fontsize=8.5, color=ARM_ORANGE)
    axes[1].text(0, 3.4, "low everywhere:\ndistinguishes nothing", ha="center",
                 fontsize=9, color=GREY)
    axes[1].text(0, 0.75, "loss looks excellent", ha="center", fontsize=8.5,
                 color=ARM_BLUE)
    save(fig, "energy_landscape")


# ----------------------------------------------------------------------------
def data_asymmetry():
    """V-JEPA 2 / V-JEPA 2-AC training budget, from the paper and Meta's blog."""
    labels = ["internet video\n(stage 1, no actions)", "robot video\n(stage 2, DROID)"]
    hours = [1_000_000, 62]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    bars = ax.barh(labels, hours, color=[ARM_BLUE, ARM_RED], height=0.5)
    ax.set_xscale("log")
    ax.set_xlim(10, 6e6)
    ax.set_xlabel("hours of video (log scale)")
    for b, h in zip(bars, hours):
        ax.text(h * 1.35, b.get_y() + b.get_height() / 2, f"{h:,} h",
                va="center", fontsize=11, fontweight="bold")
    ax.set_title("Essentially all of the knowledge comes from watching.",
                 fontsize=10.5, color=GREY)
    save(fig, "data_asymmetry")


# ----------------------------------------------------------------------------
def physics_gap():
    """IntPhys 2 / MVPBench / CausalVQA - the honest version.

    Meta's blog reports human performance as a BAND (85-95%) and does not give
    per-benchmark human scores, so the band is drawn as a band. Model performance
    is reported qualitatively as 'at or near chance', so chance is drawn as a line
    and no fabricated model bars are plotted. See _knowledge/jepa/sources.md.
    """
    names = ["IntPhys 2", "MVPBench", "CausalVQA"]
    xs = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.axhspan(85, 95, color=ARM_BLUE, alpha=0.16, zorder=1)
    ax.text(2.42, 90, "humans\n85-95%", fontsize=9.5, color=ARM_BLUE,
            va="center", ha="left", fontweight="bold")

    ax.hlines(50, -0.5, 2.5, color=ARM_RED, lw=2.2, zorder=3)
    ax.text(2.42, 50, "chance", fontsize=9.5, color=ARM_RED, va="center",
            ha="left", fontweight="bold")
    ax.scatter(xs, [50, 50, 50], s=110, color=ARM_RED, zorder=4, marker="o")
    ax.text(1.0, 41,
            "models, including V-JEPA 2, sit at or near chance",
            ha="center", fontsize=9.5, color=ARM_RED)

    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.5, 3.35)
    ax.set_ylim(30, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Meta published these alongside V-JEPA 2, and its own model fails them.",
                 fontsize=10.5, color=GREY)
    ax.text(-0.45, 32.2,
            "Human range is reported as a band, not per benchmark; model scores are "
            "reported qualitatively. Drawn as published.",
            fontsize=7.3, color=GREY, style="italic")
    save(fig, "physics_gap")


# ----------------------------------------------------------------------------
def masking_strategy():
    """I-JEPA's actual multi-block sampling, drawn on the ch12 market photo.

    4 targets, scale (0.15, 0.2), aspect (0.75, 1.5); 1 context, scale (0.85, 1.0),
    unit aspect, minus overlap. Paper section 4.
    """
    img_path = (Path(__file__).resolve().parents[2]
                / "ch12_vlm" / "fig" / "img" / "yerevan_market.jpg")
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    img = plt.imread(img_path)
    H, W = img.shape[0], img.shape[1]

    # sample 4 target blocks the way the paper does
    targets = []
    for _ in range(4):
        scale = rng.uniform(0.15, 0.20)
        ar = rng.uniform(0.75, 1.5)
        h = np.sqrt(scale * H * W / ar)
        w = ar * h
        y0 = rng.uniform(0, H - h)
        x0 = rng.uniform(0, W - w)
        targets.append((x0, y0, w, h))

    for ax, mode in zip(axes, ("context", "targets")):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if mode == "context":
            ax.add_patch(Rectangle((0, 0), W, H, fc="white", alpha=0.55, zorder=2))
            cw, ch = 0.94 * W, 0.94 * H
            ax.add_patch(Rectangle(((W - cw) / 2, (H - ch) / 2), cw, ch,
                                   fc="none", ec=ARM_BLUE, lw=2.4, zorder=4))
            for (x0, y0, w, h) in targets:
                ax.add_patch(Rectangle((x0, y0), w, h, fc="white", alpha=0.92,
                                       ec=ARM_RED, lw=1.6, ls="--", zorder=5))
            ax.set_title("Context: one large block, minus any overlap\n"
                         "scale (0.85, 1.0), about 25% of patches survive",
                         fontsize=9.5, color=ARM_BLUE)
        else:
            ax.add_patch(Rectangle((0, 0), W, H, fc="white", alpha=0.72, zorder=2))
            for (x0, y0, w, h) in targets:
                ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=ARM_RED,
                                       lw=2.4, zorder=5))
            ax.set_title("4 targets: scale (0.15, 0.2), aspect (0.75, 1.5)\n"
                         "large enough to be semantic, not texture",
                         fontsize=9.5, color=ARM_RED)
    save(fig, "masking_strategy")


if __name__ == "__main__":
    log.info("building ch16 figures (seed=%d)", SEED)
    three_architectures()
    ablation_masking()
    ablation_target()
    efficiency()
    energy_landscape()
    data_asymmetry()
    physics_gap()
    masking_strategy()
    log.info("done")
