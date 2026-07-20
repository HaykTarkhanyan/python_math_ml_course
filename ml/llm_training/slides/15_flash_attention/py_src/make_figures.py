"""Figures for the FlashAttention deck ([LLM-15]).

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Real-number figures (gpu_memory_hierarchy, speedup_bar) use values from
Dao et al. 2022, arXiv:2205.14135. Schematic figures are labeled as such.
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SEED = 509
np.random.seed(SEED)

# ---- logging (stream + file) -------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")
os.makedirs(LOGDIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGDIR, "make_figures.log"), mode="w"),
    ],
)
log = logging.getLogger("flash_attention_figs")

# ---- shared style (matches 13_moe) -------------------------------------
ARM_RED = "#C81E28"      # armred
ARM_BLUE = "#1E46A0"     # armblue
ARM_ORANGE = "#E6A01E"   # armorange
GREEN = "#008C46"        # paramgreen
VIOLET = "#7832A0"       # violet1
GREY = "#8a8a8a"

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.titlesize": 13,
    "figure.dpi": 140,
})

FIG = os.path.join(HERE, "..", "fig")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", os.path.normpath(path))


def _box(ax, xy, w, h, text, fc, ec, tc="white", fs=11, lw=1.6):
    b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(b)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=tc, fontsize=fs, zorder=4, weight="bold")


def _arrow(ax, p, q, color="#444444", lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, zorder=2, ls=ls,
                                 shrinkA=3, shrinkB=3))


# ---- Fig A: GPU memory hierarchy (REAL, A100) --------------------------
def fig_gpu_memory_hierarchy():
    # Real A100 numbers from the paper (Sec 2.1 / Fig 1 left).
    tiers = ["SRAM\n(on-chip)", "HBM\n(GPU main)", "DRAM\n(CPU)"]
    bw = [19000.0, 1500.0, 12.8]          # GB/s
    size = ["~20 MB\n(192 KB/SM x 108)", "40-80 GB", ">1 TB"]
    cols = [ARM_RED, ARM_BLUE, ARM_ORANGE]

    y = np.arange(len(tiers))[::-1]        # SRAM on top
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    bars = ax.barh(y, bw, color=cols, height=0.62, zorder=3, log=True)
    # one two-line label per bar (bandwidth bold + size grey), left-aligned
    # just past the bar end -> no overlap between separate texts.
    for yi, b, s, v in zip(y, bars, size, bw):
        w = b.get_width()
        bw_txt = f"{v:,.0f} GB/s" if v >= 100 else f"{v:g} GB/s"
        ax.text(w * 1.45, yi + 0.14, bw_txt, va="center", ha="left",
                fontsize=10, weight="bold", color="#222222")
        ax.text(w * 1.45, yi - 0.18, s.replace("\n", " "), va="center",
                ha="left", fontsize=8.8, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels(tiers, fontsize=11)
    ax.set_xlabel("bandwidth (GB/s, log scale)")
    ax.set_xlim(5, 3e6)
    ax.set_title("GPU memory: the fast tier is tiny, the big tier is slow  (A100)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.99, -0.22, "size labels on the right; bandwidth ratio SRAM:HBM ~ 10x, "
            "HBM:DRAM ~ 100x", transform=ax.transAxes, ha="right",
            fontsize=8.2, color=GREY)
    save(fig, "gpu_memory_hierarchy.pdf")


# ---- Fig B: standard attention writes N x N to HBM (schematic) ----------
def fig_standard_attention():
    fig, ax = plt.subplots(figsize=(10.6, 3.9))

    # HBM slab on the left holding the big N x N matrix
    ax.add_patch(FancyBboxPatch((0.15, 0.35), 2.7, 3.1,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc="#f3f4f8", ec="#888888", lw=1.4, zorder=1))
    ax.text(1.5, 3.62, "GPU HBM  (slow, large)", ha="center",
            fontsize=10.5, color="#555555", weight="bold")
    _box(ax, (0.85, 2.75), 0.9, 0.5, "Q,K,V", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=9.5)
    # the big N x N intermediate, highlighted red
    ax.add_patch(FancyBboxPatch((1.55, 1.05), 1.15, 1.55,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 fc="#fbe9ea", ec=ARM_RED, lw=2.0, zorder=3))
    ax.text(2.13, 1.82, "$S,P$\n$N\\times N$", ha="center", va="center",
            color=ARM_RED, fontsize=11, weight="bold")
    _box(ax, (0.85, 0.72), 0.9, 0.44, "O", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=9.5)

    # three round-trip steps on the right
    steps = [
        ("1.  $S = QK^{\\top}$", "write $N\\times N$", ARM_RED),
        ("2.  $P = \\mathrm{softmax}(S)$", "read + write $N\\times N$", ARM_RED),
        ("3.  $O = PV$", "read $N\\times N$, write $O$", ARM_RED),
    ]
    x0 = 3.9
    yv = [2.95, 1.9, 0.85]
    for (title, sub, col), yy in zip(steps, yv):
        ax.add_patch(FancyBboxPatch((x0, yy - 0.4), 5.8, 0.82,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc="white", ec="#bbbbbb", lw=1.3, zorder=3))
        ax.text(x0 + 0.25, yy + 0.07, title, ha="left", va="center",
                fontsize=11.5, color="#222222", weight="bold")
        ax.text(x0 + 0.25, yy - 0.24, sub, ha="left", va="center",
                fontsize=9.5, color=col, style="italic")
        _arrow(ax, (2.75, yy), (x0 - 0.05, yy), color=col, lw=1.7)

    ax.text(6.8, 3.6, "several full $N\\times N$ round-trips to slow HBM",
            ha="center", fontsize=10.5, color=ARM_RED, weight="bold")
    ax.text(6.8, 0.12, "softmax / mask / dropout are cheap in FLOPs but "
            "memory-bound", ha="center", fontsize=9, color=GREY, style="italic")
    ax.set_xlim(0, 9.9)
    ax.set_ylim(-0.15, 3.95)
    ax.axis("off")
    ax.text(0.01, 0.02, "schematic", transform=ax.transAxes, fontsize=8,
            color=GREY, style="italic")
    save(fig, "standard_attention.pdf")


# ---- Fig C: online (running) softmax across blocks (schematic) ---------
def fig_online_softmax():
    fig, ax = plt.subplots(figsize=(10.4, 4.1))

    # running state box (top)
    _box(ax, (1.7, 3.4), 2.6, 0.7,
         "running state\n$m,\\ \\ell,\\ o$", "#eef2fb", ARM_BLUE,
         tc=ARM_BLUE, fs=11)

    # incoming blocks
    blocks = [
        (0.9, "block 1", "local max $m_1$"),
        (3.4, "block 2", "local max $m_2 > m_1$"),
        (5.9, "block 3", "local max $m_3$"),
    ]
    for xx, name, sub in blocks:
        _box(ax, (xx, 1.4), 1.7, 0.66, name, ARM_ORANGE, ARM_ORANGE,
             tc="#333333", fs=10.5)
        ax.text(xx, 0.72, sub, ha="center", fontsize=9, color="#555555")

    # combine node
    _box(ax, (8.2, 1.4), 2.0, 1.5,
         "update\n$m' = \\max(m, m_b)$\nrescale by\n$e^{m-m'},\\, e^{m_b-m'}$",
         "#e9f5ee", GREEN, tc=GREEN, fs=9.5)

    for xx, _, _ in blocks:
        _arrow(ax, (xx, 1.73), (7.3, 1.55), color=GREY, lw=1.3)
    # loop back to running state
    _arrow(ax, (8.2, 2.15), (2.6, 3.15), color=GREEN, lw=1.7)
    ax.text(5.2, 2.95, "carry a small summary, one block at a time",
            ha="center", fontsize=9.5, color=GREEN, style="italic")

    ax.text(5.0, 0.08, "never hold the whole $N\\times N$ row at once - just "
            "$m$ (max), $\\ell$ (sum), $o$ (partial output)",
            ha="center", fontsize=9.5, color="#333333")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.15, 3.95)
    ax.axis("off")
    ax.text(0.01, 0.02, "schematic", transform=ax.transAxes, fontsize=8,
            color=GREY, style="italic")
    save(fig, "online_softmax.pdf")


# ---- Fig D (CENTERPIECE): tiling diagram (schematic) -------------------
def fig_tiling_diagram():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))

    # K,V column blocks (outer loop) across the top
    kx = [4.2, 5.5, 6.8, 8.1]
    for j, x in enumerate(kx):
        fc = "#e9f5ee" if j == 1 else "white"
        ec = GREEN if j == 1 else "#bbbbbb"
        ax.add_patch(FancyBboxPatch((x, 4.05), 1.05, 0.55,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     fc=fc, ec=ec, lw=(2.0 if j == 1 else 1.2), zorder=3))
        ax.text(x + 0.52, 4.32, f"$K_{j+1},V_{j+1}$", ha="center", va="center",
                fontsize=9, color=("#222222" if j == 1 else "#888888"))
    ax.text(6.15, 4.85, "outer loop: K,V blocks", ha="center",
            fontsize=10, color=GREEN, weight="bold")

    # Q row blocks (inner loop) down the left
    qy = [3.3, 2.35, 1.4, 0.45]
    for i, y in enumerate(qy):
        fc = "#eef2fb" if i == 1 else "white"
        ec = ARM_BLUE if i == 1 else "#bbbbbb"
        ax.add_patch(FancyBboxPatch((0.35, y), 1.05, 0.62,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     fc=fc, ec=ec, lw=(2.0 if i == 1 else 1.2), zorder=3))
        ax.text(0.87, y + 0.31, f"$Q_{i+1}$", ha="center", va="center",
                fontsize=9.5, color=("#222222" if i == 1 else "#888888"))
    ax.text(0.87, 4.05, "inner loop:\nQ blocks", ha="center",
            fontsize=10, color=ARM_BLUE, weight="bold")

    # the S grid (blocks); highlight the active tile S_ij
    gx0, gy0 = 4.2, 0.45
    cw, ch = 1.3, 0.72
    for i in range(4):
        for j in range(4):
            active = (i == 1 and j == 1)
            x = gx0 + j * cw
            y = gy0 + (3 - i) * ch
            ax.add_patch(Rectangle((x, y), cw - 0.12, ch - 0.10,
                         fc=("#fdf2d6" if active else "#fafafa"),
                         ec=(ARM_ORANGE if active else "#dddddd"),
                         lw=(2.0 if active else 0.8), zorder=2))
            if active:
                ax.text(x + (cw - 0.12) / 2, y + (ch - 0.10) / 2,
                        "$S_{ij}$", ha="center", va="center",
                        fontsize=10, color=ARM_ORANGE, weight="bold")

    # SRAM callout on the active tile
    ax.add_patch(FancyBboxPatch((5.55, 2.05), 1.9, 0.62,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=ARM_ORANGE, ec=ARM_ORANGE, lw=1.4, zorder=5))
    ax.text(6.5, 2.36, "in fast SRAM:\ncompute $S_{ij}$, update $m,\\ell,O_i$",
            ha="center", va="center", fontsize=8.5, color="#333333",
            weight="bold", zorder=6)
    _arrow(ax, (6.4, 2.7), (5.9, 3.05), color=ARM_ORANGE, lw=1.5)

    # only O written back
    _box(ax, (9.9, 1.9), 1.2, 0.9, "write\nonly $O$\n$N\\times d$",
         "#e9f5ee", GREEN, tc=GREEN, fs=9.5)
    _arrow(ax, (9.0, 1.9), (9.25, 1.9), color=GREEN, lw=1.8)
    ax.text(9.9, 0.95, "$S,P$ never\nwritten to HBM", ha="center",
            fontsize=8.8, color=ARM_RED, weight="bold")

    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    ax.text(0.01, 0.01, "schematic", transform=ax.transAxes, fontsize=8,
            color=GREY, style="italic")
    save(fig, "tiling_diagram.pdf")


# ---- Fig E: end-to-end speedups (REAL) ---------------------------------
def fig_speedup_bar():
    # End-to-end wall-clock speedups (paper Tables 1-3).
    labels = ["BERT-large\n(vs MLPerf 1.1)", "GPT-2 small\n(vs HF)",
              "GPT-2 medium\n(vs HF)", "LRA\n(vs standard)"]
    speed = [1.15, 3.5, 3.0, 2.4]          # 15% = 1.15x
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    bars = ax.bar(x, speed, width=0.6, color=ARM_BLUE, zorder=3)
    ax.bar_label(bars, labels=["+15%\n(1.15x)", "3.5x", "3.0x", "2.4x"],
                 padding=3, fontsize=10.5, weight="bold")

    # the kernel-only 7.6x, set clearly apart
    xk = len(labels) + 0.7
    kb = ax.bar([xk], [7.6], width=0.6, color=VIOLET, zorder=3)
    ax.bar_label(kb, labels=["7.6x"], padding=3, fontsize=11, weight="bold")
    ax.axvline(len(labels) - 0.15, color="#cccccc", ls="--", lw=1.2)
    ax.text(xk, -0.9, "attention\nkernel only\n(not end-to-end)",
            ha="center", va="top", fontsize=9, color=VIOLET, weight="bold")

    ax.axhline(1.0, color=GREY, ls=":", lw=1.3)
    ax.text(-0.4, 1.12, "baseline 1x", ha="left", color=GREY, fontsize=9)
    ax.set_xticks(list(x) + [xk])
    ax.set_xticklabels(labels + ["kernel"], fontsize=9.5)
    ax.set_ylabel("wall-clock speedup")
    ax.set_ylim(0, 8.6)
    ax.set_title("FlashAttention: up to ~3x end-to-end, exact result "
                 "(7.6x on the attention op alone)")
    save(fig, "speedup_bar.pdf")


# ---- Fig F: memory scaling, quadratic vs linear ------------------------
def fig_memory_scaling():
    # Illustrative shapes: standard adds an O(N^2) term, flash stays O(N).
    # Tuned so the gap reaches ~20x at 64K (paper: up to 20x less memory).
    N = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    b = 1.0
    a = 19.0 / 65536.0 * b                  # gap = 1 + (a/b)N -> 20x at 64K
    flash = b * N
    standard = b * N + a * N * N
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.loglog(N, standard, "o-", color=ARM_RED, lw=2.2, ms=5,
              label="standard: $O(N^2)$ memory")
    ax.loglog(N, flash, "s-", color=GREEN, lw=2.2, ms=5,
              label="FlashAttention: $O(N)$ memory")
    # 20x annotation at the largest N
    ax.annotate("", xy=(N[-1], standard[-1]), xytext=(N[-1], flash[-1]),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
    ax.text(N[-1] * 0.62, np.sqrt(standard[-1] * flash[-1]), "up to\n20x",
            ha="right", va="center", fontsize=11, color="#333333",
            weight="bold")
    ax.set_xlabel("sequence length $N$")
    ax.set_ylabel("attention memory (arb. units, log)")
    ax.set_title("Linear vs quadratic memory: FlashAttention unlocks long context")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    ax.text(0.99, -0.20, "shapes are the real asymptotics ($N^2$ vs $N$); "
            "magnitudes illustrative, 20x from paper Fig. 3",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "memory_scaling.pdf")


if __name__ == "__main__":
    fig_gpu_memory_hierarchy()
    fig_standard_attention()
    fig_online_softmax()
    fig_tiling_diagram()
    fig_speedup_bar()
    fig_memory_scaling()
    log.info("all FlashAttention figures done")
