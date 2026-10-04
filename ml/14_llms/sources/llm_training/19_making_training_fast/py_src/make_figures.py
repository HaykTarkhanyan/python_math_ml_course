"""Figures for the "Making Training Fast" deck ([LLM-19]).

Run with the project venv:
    ./ma/Scripts/python.exe py_src/make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Real-number figures use A100 80GB SXM datasheet values and the measured
timings from Karpathy, "Let's reproduce GPT-2 (124M)" (2024), section 2.
Schematic figures are labeled as such on the figure itself.
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

SEED = 509
np.random.seed(SEED)

# ---- logging (stream + file) -------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "fig")
LOGDIR = os.path.join(HERE, "logs")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGDIR, "make_figures.log"), mode="w"),
    ],
)
log = logging.getLogger("fast_training_figs")

# ---- shared style (matches the other llm_training decks) ---------------
ARM_RED = "#C81E28"      # armred
ARM_BLUE = "#1E46A0"     # armblue
ARM_ORANGE = "#E6A01E"   # armorange
GREEN = "#008C46"        # paramgreen
VIOLET = "#7832A0"       # violet1
GREY = "#8a8a8a"
LIGHT = "#d9d9d9"

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.titlesize": 13,
    "figure.dpi": 140,
})


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", os.path.relpath(path, HERE))


def schematic_note(ax, text="schematic, not to scale"):
    ax.text(0.99, -0.13, text, transform=ax.transAxes, ha="right",
            fontsize=8.2, color=GREY, style="italic")


# ---- Fig A: A100 throughput by precision (REAL) ------------------------
def fig_a100_tflops():
    """A100 80GB SXM datasheet, dense (non-sparsity) numbers."""
    labels = ["FP64", "FP32", "TF32\ntensor core", "BF16 / FP16\ntensor core",
              "INT8\ntensor core"]
    vals = [9.7, 19.5, 156.0, 312.0, 624.0]
    colors = [GREY, ARM_RED, ARM_BLUE, GREEN, LIGHT]

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    bars = ax.bar(labels, vals, color=colors, zorder=3, width=0.62)
    ax.bar_label(bars, labels=[f"{v:g}" for v in vals], padding=3,
                 fontsize=11, weight="bold")

    # the 8x TF32 promise, drawn between FP32 and TF32
    ax.annotate("", xy=(2, 168), xytext=(1, 78),
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE, lw=2.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.5, 232, "8x on paper", ha="center", color=ARM_BLUE,
            fontsize=11.5, weight="bold")

    ax.text(4, 330, "inference\nonly", ha="center", va="center", color="#666666",
            fontsize=10, weight="bold")
    ax.set_ylabel("TFLOPS (dense)")
    ax.set_ylim(0, 730)
    ax.set_title("NVIDIA A100 80GB SXM: throughput by numeric format")
    ax.text(0.99, -0.22,
            "datasheet values, sparsity numbers excluded",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "a100_tflops.pdf")


# ---- Fig B: float bit layouts (REAL) -----------------------------------
def fig_float_formats():
    """Sign / exponent / mantissa layouts. The centerpiece figure."""
    # (name, exponent bits, mantissa bits, dropped mantissa bits, stored bits)
    fmts = [
        ("FP32",  8, 23, 0,  32),
        ("TF32",  8, 10, 13, 32),
        ("FP16",  5, 10, 0,  16),
        ("BF16",  8,  7, 0,  16),
    ]
    unit = 1.0          # width of one bit
    fig, ax = plt.subplots(figsize=(10.4, 4.4))

    for i, (name, ne, nm, ndrop, stored) in enumerate(fmts):
        y = len(fmts) - 1 - i
        x = 0.0
        # sign
        ax.add_patch(Rectangle((x, y), unit, 0.62, facecolor=GREY,
                               edgecolor="white", lw=1.0))
        ax.text(x + unit / 2, y + 0.31, "s", ha="center", va="center",
                color="white", fontsize=9.5, weight="bold")
        x += unit
        # exponent
        ax.add_patch(Rectangle((x, y), ne * unit, 0.62, facecolor=ARM_BLUE,
                               edgecolor="white", lw=1.0))
        ax.text(x + ne * unit / 2, y + 0.31, f"{ne}", ha="center", va="center",
                color="white", fontsize=10.5, weight="bold")
        x += ne * unit
        # mantissa (kept)
        ax.add_patch(Rectangle((x, y), nm * unit, 0.62, facecolor=ARM_ORANGE,
                               edgecolor="white", lw=1.0))
        ax.text(x + nm * unit / 2, y + 0.31, f"{nm}", ha="center", va="center",
                color="#3a2a00", fontsize=10.5, weight="bold")
        x += nm * unit
        # mantissa dropped inside the tensor-core instruction (TF32 only)
        if ndrop:
            ax.add_patch(Rectangle((x, y), ndrop * unit, 0.62, facecolor="white",
                                   edgecolor=ARM_RED, lw=1.2, hatch="////"))
            ax.text(x + ndrop * unit / 2, y + 0.31, f"{ndrop} bits dropped",
                    ha="center", va="center", color=ARM_RED, fontsize=9.5,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.9))
            x += ndrop * unit

        ax.text(-0.6, y + 0.31, name, ha="right", va="center",
                fontsize=12.5, weight="bold")
        ax.text(33.4, y + 0.31, f"{stored} bits stored", ha="left", va="center",
                fontsize=10, color="#444444")

    # header labels
    ax.text(-4.1, 4.15, "s = sign", ha="left", fontsize=9.5, color=GREY)
    ax.text(5.2, 4.15, "exponent  =  RANGE", ha="center", fontsize=11,
            color=ARM_BLUE, weight="bold")
    ax.text(21.0, 4.15, "mantissa  =  PRECISION", ha="center", fontsize=11,
            color="#8a6000", weight="bold")

    # the alignment point: FP32, TF32 and BF16 share an 8-bit exponent
    ax.plot([9.0, 9.0], [-0.15, 3.95], color=ARM_BLUE, ls=":", lw=1.6, zorder=0)
    ax.text(9.25, -0.42, "same 8-bit exponent as FP32 -> same range, no gradient scaler",
            ha="left", fontsize=9.5, color=ARM_BLUE)

    ax.set_xlim(-4.2, 40)
    ax.set_ylim(-0.75, 4.6)
    ax.axis("off")
    ax.set_title("Four ways to spend bits on a number", fontsize=13.5, pad=14)
    save(fig, "float_formats.pdf")


# ---- Fig C: representable range, BF16 vs FP16 (REAL) -------------------
def fig_bf16_vs_fp16_range():
    """IEEE normal-number ranges. FP16's narrow range is why scalers exist."""
    # (name, smallest normal, largest finite, color)
    rows = [
        ("FP32", 1.175e-38, 3.403e38, ARM_RED),
        ("BF16", 1.175e-38, 3.390e38, GREEN),
        ("FP16", 6.104e-05, 6.5504e4, ARM_BLUE),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    for i, (name, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([np.log10(lo), np.log10(hi)], [y, y], color=c, lw=11,
                solid_capstyle="butt", zorder=3)
        ax.text(-42.5, y, name, ha="right", va="center", fontsize=12.5,
                weight="bold")
        # short bars get their endpoint labels pushed outward, not stacked
        narrow = (np.log10(hi) - np.log10(lo)) < 20
        ax.text(np.log10(lo) - (2.6 if narrow else 0), y + (0.0 if narrow else 0.30),
                f"{lo:.2g}", ha="right" if narrow else "center",
                va="center" if narrow else "bottom", fontsize=8.5, color="#444444")
        ax.text(np.log10(hi) + (2.6 if narrow else 0), y + (0.0 if narrow else 0.30),
                f"{hi:.2g}", ha="left" if narrow else "center",
                va="center" if narrow else "bottom", fontsize=8.5, color="#444444")

    # a small gradient that FP16 cannot represent as a normal number
    ax.axvline(-7, color=VIOLET, ls="--", lw=1.6, zorder=2)
    ax.text(-7.3, -0.72, "a small gradient ($10^{-7}$)\nunderflows in FP16",
            ha="center", va="top", fontsize=9.5, color=VIOLET, weight="bold")

    ax.set_xlim(-42, 42)
    ax.set_ylim(-1.6, 2.7)
    ax.set_yticks([])
    ax.set_xlabel("magnitude (log$_{10}$)")
    ax.spines["left"].set_visible(False)
    ax.set_title("BF16 keeps FP32's range. FP16 does not - hence gradient scalers.")
    ax.text(0.99, -0.42, "IEEE normal-number limits; the $10^{-7}$ marker is illustrative",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "bf16_vs_fp16_range.pdf")


# ---- Fig D: why not INT8 for training (SCHEMATIC) ----------------------
def fig_int8_vs_float():
    """Uniform integer spacing vs float spacing under a bell-shaped spread."""
    x = np.linspace(-4, 4, 400)
    pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6), sharex=True)

    for ax, kind in zip(axes, ["int", "float"]):
        ax.plot(x, pdf, color="#333333", lw=1.8, zorder=3)
        ax.fill_between(x, pdf, color="#333333", alpha=0.07, zorder=1)
        if kind == "int":
            ticks = np.linspace(-4, 4, 25)          # uniform spacing
            c, title = ARM_RED, "INT8: evenly spaced steps everywhere"
        else:
            # float-like: dense near zero, sparse in the tails
            mags = np.concatenate([-np.logspace(np.log10(4), -2, 12),
                                   np.logspace(-2, np.log10(4), 12)])
            ticks, c = mags, GREEN
            title = "Floating point: fine steps near zero, coarse in the tails"
        for t in ticks:
            ax.axvline(t, ymin=0.0, ymax=0.14, color=c, lw=1.4, zorder=4)
        ax.set_title(title, fontsize=11.5, color=c, loc="left")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    axes[1].set_xlabel("weight / activation value")
    axes[0].text(2.55, 0.30, "most values live here,\nand get few distinct levels",
                 fontsize=9.5, color=ARM_RED, ha="center")
    fig.suptitle("Weights and activations cluster near zero - so floats fit them, "
                 "integers do not", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.text(0.99, -0.03, "schematic; spacing exaggerated for legibility",
             ha="right", fontsize=8.2, color=GREY, style="italic")
    save(fig, "int8_vs_float.pdf")


# ---- Fig E: memory hierarchy (REAL) ------------------------------------
def fig_memory_hierarchy():
    """A100-class numbers: fast is tiny, big is slow."""
    tiers = [
        ("on-chip SRAM\n(L1 + registers)", 19_000.0, 0.02, GREEN),
        ("HBM\n(GPU memory)", 2_000.0, 80.0, ARM_BLUE),
        ("CPU DRAM\n(system memory)", 12.8, 1_000.0, ARM_RED),
    ]
    names = [t[0] for t in tiers]
    bw = [t[1] for t in tiers]
    cap = [t[2] for t in tiers]
    cols = [t[3] for t in tiers]
    y = np.arange(len(tiers))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)

    b1 = ax1.barh(y, bw, color=cols, height=0.55, zorder=3)
    ax1.bar_label(b1, labels=["19 TB/s", "2.0 TB/s", "12.8 GB/s"],
                  padding=4, fontsize=10.5, weight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(5, 2e6)
    ax1.set_xlabel("bandwidth, GB/s (log)")
    ax1.set_title("How fast you can read it", fontsize=12)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10.5)

    b2 = ax2.barh(y, cap, color=cols, height=0.55, zorder=3)
    ax2.bar_label(b2, labels=["~20 MB", "80 GB", ">1 TB"],
                  padding=4, fontsize=10.5, weight="bold")
    ax2.set_xscale("log")
    ax2.set_xlim(0.005, 3e5)
    ax2.set_xlabel("capacity, GB (log)")
    ax2.set_title("How much of it there is", fontsize=12)

    fig.suptitle("The memory wall: the fast memory is tiny, the big memory is slow",
                 fontsize=13.5, y=1.04)
    ax1.text(0.0, -0.42, "A100 80GB SXM class numbers",
             transform=ax1.transAxes, fontsize=8.2, color=GREY)
    fig.tight_layout()
    save(fig, "memory_hierarchy.pdf")


# ---- Fig F: kernel fusion round trips (SCHEMATIC) ----------------------
def fig_kernel_fusion():
    """The GELU elementwise chain, unfused vs fused."""
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.0))
    ops = [r"$x^3$", r"$\times\,0.044715$", r"$+\,x$"]

    def draw(ax, fused):
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 2.5)
        ax.axis("off")
        # HBM strip along the bottom, chip strip along the top
        ax.add_patch(Rectangle((0.2, 0.05), 10.0, 0.42, facecolor=ARM_BLUE,
                               alpha=0.13, edgecolor=ARM_BLUE, lw=1.2))
        ax.text(0.35, 0.26, "HBM (slow, big)", va="center", fontsize=10,
                color=ARM_BLUE, weight="bold")
        ax.add_patch(Rectangle((0.2, 1.95), 10.0, 0.42, facecolor=GREEN,
                               alpha=0.13, edgecolor=GREEN, lw=1.2))
        ax.text(0.35, 2.16, "on-chip SRAM (fast, tiny)", va="center",
                fontsize=10, color=GREEN, weight="bold")

        if not fused:
            xs = [2.0, 5.0, 8.0]
            for xc, op in zip(xs, ops):
                ax.add_patch(FancyArrowPatch((xc - 0.75, 0.47), (xc - 0.35, 1.95),
                                             arrowstyle="-|>", mutation_scale=13,
                                             color=ARM_RED, lw=1.7))
                ax.add_patch(FancyArrowPatch((xc + 0.35, 1.95), (xc + 0.75, 0.47),
                                             arrowstyle="-|>", mutation_scale=13,
                                             color=ARM_RED, lw=1.7))
                ax.text(xc, 2.16, op, ha="center", va="center", fontsize=11.5,
                        color=GREEN, weight="bold")
                ax.text(xc, 1.15, "read\n+ write", ha="center", va="center",
                        fontsize=8.5, color=ARM_RED)
            ax.set_title("Eager: one kernel per operation, a full round trip each time "
                         "(3 reads + 3 writes)",
                         fontsize=12, color=ARM_RED, loc="left")
        else:
            ax.add_patch(FancyArrowPatch((2.0, 0.47), (2.6, 1.95),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=GREEN, lw=2.2))
            ax.add_patch(FancyArrowPatch((7.8, 1.95), (8.4, 0.47),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=GREEN, lw=2.2))
            ax.add_patch(Rectangle((3.0, 1.99), 4.4, 0.34, facecolor="white",
                                   edgecolor=GREEN, lw=1.6))
            ax.text(5.2, 2.16, "  ".join(ops) + "   all on chip",
                    ha="center", va="center", fontsize=11.5, color=GREEN,
                    weight="bold")
            ax.text(5.2, 1.15, "the tensor never goes back to HBM in between",
                    ha="center", va="center", fontsize=9.5, color=GREEN)
            ax.set_title("Fused (torch.compile): one round trip for the whole chain "
                         "(1 read + 1 write)",
                         fontsize=12, color=GREEN, loc="left")

    draw(axes[0], fused=False)
    draw(axes[1], fused=True)
    axes[1].text(0.99, -0.02, "schematic; the GELU elementwise chain",
                 transform=axes[1].transAxes, ha="right", fontsize=8.2,
                 color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "kernel_fusion.pdf")


# ---- Fig G: what autocast changes (SCHEMATIC) --------------------------
def fig_autocast_what_changes():
    cast = ["linear layers (matmuls)", "attention $QK^\\top$ and $PV$",
            "convolutions", "-> activations become BF16"]
    keep = ["layer norm", "softmax", "the loss (cross-entropy)",
            "-> parameters stay FP32"]

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    for x0, title, items, c in [(0.2, "Cast to BF16", cast, GREEN),
                                (5.2, "Left in FP32", keep, ARM_BLUE)]:
        ax.add_patch(Rectangle((x0, 0.5), 4.6, 4.1, facecolor=c, alpha=0.07,
                               edgecolor=c, lw=1.5))
        ax.text(x0 + 2.3, 4.15, title, ha="center", fontsize=13,
                color=c, weight="bold")
        for j, it in enumerate(items):
            bold = it.startswith("->")
            ax.text(x0 + 0.35, 3.45 - 0.72 * j, it, ha="left", fontsize=11,
                    color="#222222" if not bold else c,
                    weight="bold" if bold else "normal")

    ax.text(5.0, 5.25,
            "torch.autocast wraps the forward pass and the loss ONLY - "
            "not backward, not the optimizer step",
            ha="center", fontsize=11.5, color=ARM_RED, weight="bold")
    ax.text(9.8, 0.12, "schematic; matmul-like ops are cast, "
            "reduction-heavy ops are not", ha="right", fontsize=8.2,
            color=GREY, style="italic")
    save(fig, "autocast_what_changes.pdf")


# ---- Fig H: the speedup ladder (REAL) ----------------------------------
def fig_speedup_ladder():
    """Measured on one A100, GPT-2 124M, B=16, T=1024 (16,384 tokens/step)."""
    labels = ["FP32\nbaseline", "TF32", "BF16\nautocast", "torch\n.compile",
              "Flash\nAttention", "vocab\n50304"]
    ms = [1000, 333, 300, 130, 96, 93]
    tok = [16384 / (m / 1000) for m in ms]
    cum = [1000 / m for m in ms]
    cols = [GREY, ARM_BLUE, GREEN, VIOLET, ARM_ORANGE, ARM_RED]

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    bars = ax.bar(labels, ms, color=cols, width=0.62, zorder=3)
    ax.bar_label(bars,
                 labels=[f"{m} ms\n{t/1000:.0f}k tok/s" for m, t in zip(ms, tok)],
                 padding=3, fontsize=10, weight="bold")

    for i, c in enumerate(cum):
        ax.text(i, ms[i] + 232, f"{c:.1f}x", ha="center", fontsize=12,
                color=cols[i], weight="bold")

    ax.set_ylabel("milliseconds per training step")
    ax.set_ylim(0, 1420)
    ax.set_title("Five changes, none of which alter the model: 1000 ms -> 93 ms "
                 "(10.8x)")
    ax.text(0.99, -0.26,
            "A100 80GB SXM, GPT-2 124M, batch 16 x 1024 tokens "
            "(Karpathy, 2024)",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "speedup_ladder.pdf")


# ---- Fig I: which rungs exist on which GPU (REAL) ----------------------
def fig_hardware_reality():
    rungs = ["TF32", "BF16 autocast", "FP16 + gradient scaler",
             "torch.compile", "FlashAttention kernel", "power-of-two shapes"]
    gpus = ["T4\n(Turing, Colab free)", "A100\n(Ampere)", "H100\n(Hopper)"]
    # 1 = yes, 0.5 = partial / fallback, 0 = no
    grid = np.array([
        [0.0, 1.0, 1.0],   # TF32: Ampere+
        [0.0, 1.0, 1.0],   # BF16: Ampere+
        [1.0, 1.0, 1.0],   # FP16 + scaler: Volta+
        [1.0, 1.0, 1.0],   # torch.compile
        [0.5, 1.0, 1.0],   # FlashAttention kernel needs Ampere+; T4 falls back
        [1.0, 1.0, 1.0],   # nice shapes always help
    ])
    marks = {1.0: ("yes", GREEN), 0.5: ("fallback", ARM_ORANGE), 0.0: ("no", ARM_RED)}

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for i in range(len(rungs)):
        for j in range(len(gpus)):
            v = grid[i, j]
            txt, c = marks[v]
            ax.add_patch(Rectangle((j, len(rungs) - 1 - i), 1, 1,
                                   facecolor=c, alpha=0.16, edgecolor="white",
                                   lw=2.0))
            ax.text(j + 0.5, len(rungs) - 1 - i + 0.5, txt, ha="center",
                    va="center", fontsize=11, color=c, weight="bold")

    ax.set_xlim(0, len(gpus))
    ax.set_ylim(0, len(rungs))
    ax.set_xticks(np.arange(len(gpus)) + 0.5)
    ax.set_xticklabels(gpus, fontsize=10.5)
    ax.set_yticks(np.arange(len(rungs)) + 0.5)
    ax.set_yticklabels(rungs[::-1], fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_title("Not every rung exists on every GPU", fontsize=13.5, pad=34)
    ax.text(0.99, -0.13,
            "TF32 and BF16 arrived with Ampere; on Turing the FlashAttention "
            "kernel is unavailable and PyTorch uses its memory-efficient backend",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "hardware_reality.pdf")


# ---- Fig J: roofline (REAL) --------------------------------------------
def fig_roofline():
    """Why matmuls are compute-bound and everything else is not.

    Machine balance = peak FLOPS / HBM bandwidth. An operation whose arithmetic
    intensity sits left of the ridge point can never reach peak: it is waiting
    on memory. All intensities computed for B=16, T=1024, C=768, V=50257.
    """
    BW = 2.0e12                                  # A100 80GB SXM HBM, B/s
    ceilings = [("FP32  19.5", 19.5e12, ARM_RED),
                ("TF32  156", 156e12, ARM_BLUE),
                ("BF16  312", 312e12, GREEN)]
    I = np.logspace(-2, 3.4, 600)

    fig, ax = plt.subplots(figsize=(10.0, 4.9))

    # memory-bound region (left of the TF32 ridge point)
    ridge_tf32 = 156e12 / BW
    ax.axvspan(1e-2, ridge_tf32, color=ARM_RED, alpha=0.05, zorder=0)
    ax.text(3.0, 0.048, "memory-bound: waiting on HBM", ha="center",
            fontsize=10.5, color=ARM_RED, weight="bold", alpha=0.9)

    for name, peak, c in ceilings:
        ax.loglog(I, np.minimum(BW * I, peak) / 1e12, color=c, lw=2.4, zorder=3)
        ax.text(2200, peak / 1e12, f"  {name}", va="center", ha="left",
                fontsize=9.5, color=c, weight="bold")
        ax.plot([peak / BW], [peak / 1e12], "o", color=c, ms=6, zorder=4)

    ax.text(0.028, 0.10, "slope = HBM\n2.0 TB/s", fontsize=9.5, color=GREY,
            rotation=34, ha="left", va="bottom")

    # where our operations actually sit
    ops = [(0.125, "one elementwise op\n(GELU, scale, residual add)", ARM_ORANGE, 230.0, "left"),
           (362.0, "the classifier matmul\n768 $\\rightarrow$ 50257", VIOLET, 3.2, "right")]
    for x, label, c, ly, side in ops:
        y = min(BW * x, 312e12) / 1e12
        ax.axvline(x, color=c, ls="--", lw=1.6, zorder=2)
        ax.plot([x], [y], "D", color=c, ms=8, zorder=5)
        ax.text(x * (1.3 if side == "left" else 0.78), ly, label, fontsize=10,
                color=c, weight="bold", ha=side, va="center")

    ax.set_xlim(0.02, 2000)
    ax.set_ylim(0.02, 900)
    ax.set_xlabel("arithmetic intensity (FLOPs performed per byte moved)")
    ax.set_ylabel("attainable TFLOPS")
    ax.grid(True, which="both", alpha=0.18)
    ax.set_title("The one plot behind all five rungs: you only reach peak on the right")
    ax.text(0.99, -0.235,
            "A100 80GB SXM; intensities for B=16, T=1024, C=768, FP32 operands",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "roofline.pdf")


if __name__ == "__main__":
    fig_roofline()
    fig_a100_tflops()
    fig_float_formats()
    fig_bf16_vs_fp16_range()
    fig_int8_vs_float()
    fig_memory_hierarchy()
    fig_kernel_fusion()
    fig_autocast_what_changes()
    fig_speedup_ladder()
    fig_hardware_reality()
    log.info("all [LLM-19] figures done")
