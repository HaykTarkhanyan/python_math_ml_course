"""Figures for the RoPE (Rotary Position Embedding) deck [LLM-16].

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Attribution note: `theta_scaling.pdf` shows DOWNSTREAM work (Llama-3 / NTK / YaRN),
NOT the RoFormer paper, and is labeled as such. The RoFormer paper fixes base
theta = 10000 throughout.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge

SEED = 509
np.random.seed(SEED)

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

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


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


# ---- Fig A: permutation invariance (schematic) -------------------------
def fig_permutation_invariance():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6))
    toks = ["the", "cat", "sat"]
    outs = ["$o_1$", "$o_2$", "$o_3$"]

    def draw(ax, order, title, tcol):
        ax.set_title(title, color=tcol, weight="bold", fontsize=12, loc="left")
        xs = [1.2, 3.0, 4.8]
        # input tokens
        for x, t in zip(xs, [toks[i] for i in order]):
            _box(ax, (x, 0.5), 1.2, 0.55, t, "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=12)
        # attention slab
        ax.add_patch(FancyBboxPatch((0.55, 1.35), 4.9, 0.7,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc=ARM_ORANGE, ec=ARM_ORANGE, lw=1.4, zorder=3))
        ax.text(3.0, 1.7, "self-attention  (no position signal)",
                ha="center", va="center", color="#333333", fontsize=11, weight="bold")
        # outputs, permuted the same way
        for x, o in zip(xs, [outs[i] for i in order]):
            _box(ax, (x, 2.85), 1.2, 0.55, o, "#eafaf0", GREEN, tc=GREEN, fs=12)
        for x in xs:
            _arrow(ax, (x, 0.78), (x, 1.32))
            _arrow(ax, (x, 2.08), (x, 2.55))
        ax.set_xlim(0.3, 6.6); ax.set_ylim(0.0, 3.3); ax.axis("off")

    draw(axes[0], [0, 1, 2], "Original order:  the cat sat", ARM_BLUE)
    draw(axes[1], [2, 1, 0], "Shuffled order:  sat cat the", ARM_RED)
    axes[1].text(6.05, 2.85, "same set of\noutputs, just\npermuted",
                 ha="center", va="center", color=GREY, fontsize=9)
    fig.suptitle("Permutation invariance: shuffle the inputs, outputs just shuffle with them"
                 "  (schematic)", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "permutation_invariance.pdf")


# ---- Fig B (CENTERPIECE): clock-hand 2-D rotation ----------------------
def fig_rotation_2d():
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    th = np.deg2rad(30.0)          # base angle theta
    ang_q = 2 * 30.0               # m = 2  ->  m*theta = 60 deg
    ang_k = 5 * 30.0               # n = 5  ->  n*theta = 150 deg
    # unit circle
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tt), np.sin(tt), color="#cccccc", lw=1.2, zorder=1)
    ax.axhline(0, color="#e2e2e2", lw=1.0, zorder=0)
    ax.axvline(0, color="#e2e2e2", lw=1.0, zorder=0)

    def hand(a_deg, color, label, lab_r=1.16):
        a = np.deg2rad(a_deg)
        ax.add_patch(FancyArrowPatch((0, 0), (np.cos(a), np.sin(a)),
                     arrowstyle="-|>", mutation_scale=18, color=color, lw=2.6, zorder=4))
        ax.text(lab_r * np.cos(a), lab_r * np.sin(a), label, color=color,
                ha="center", va="center", fontsize=12, weight="bold")

    hand(ang_q, ARM_BLUE, r"$R(m\theta)\,q$")
    hand(ang_k, ARM_RED, r"$R(n\theta)\,k$")
    # angle-between wedge
    ax.add_patch(Wedge((0, 0), 0.42, ang_q, ang_k, width=0.42,
                       facecolor=GREEN, alpha=0.18, zorder=2))
    mid = np.deg2rad((ang_q + ang_k) / 2)
    ax.text(0.62 * np.cos(mid), 0.62 * np.sin(mid), r"$(n-m)\theta$",
            color=GREEN, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0, -1.42,
            "angle between the two hands $= (n-m)\\theta$ : depends only on the\n"
            "relative position, so $\\;R(m\\theta)q \\cdot R(n\\theta)k = q^{\\top}R((n-m)\\theta)k$",
            ha="center", va="center", fontsize=11, color="#333333")
    ax.set_title("Each 2-D slice is a clock hand: position = how far you rotate it",
                 color="#333333", fontsize=12.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.7, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "rotation_2d.pdf")


# ---- Fig C: per-pair rotation frequencies (sinusoidal bands) -----------
def fig_frequencies():
    m = np.linspace(0, 60, 600)
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    bands = [
        (1.0,    ARM_RED,    r"pair $i=0$: high freq  $\theta_0=1$"),
        (0.15,   ARM_BLUE,   r"mid freq  $\theta_i\approx 0.15$"),
        (0.03,   ARM_ORANGE, r"pair $i\!\to\! d/2$: low freq  $\theta_i\to 10000^{-1}$"),
    ]
    for theta, col, lab in bands:
        ax.plot(m, np.cos(m * theta), color=col, lw=2.2, label=lab)
    ax.set_xlabel("position $m$")
    ax.set_ylabel(r"$\cos(m\,\theta_i)$")
    ax.set_ylim(-1.35, 1.55)
    ax.set_title(r"Each coordinate pair rotates at its own frequency "
                 r"$\theta_i = 10000^{-2i/d}$")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9.5)
    ax.text(0.5, -0.30, "high-frequency pairs track local position; "
            "low-frequency pairs track global position  (same spirit as sinusoidal bands)",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=GREY)
    save(fig, "frequencies.pdf")


# ---- Fig D: long-term decay (reproduces paper Fig. 2 shape) ------------
def fig_decay():
    d = 128
    half = d // 2
    i = np.arange(half)
    theta = 10000.0 ** (-2.0 * i / d)      # theta_i = 10000^{-2i/d}
    dist = np.arange(0, 257)
    bound = np.empty_like(dist, dtype=float)
    for t_idx, t in enumerate(dist):
        # S_j = sum_{i=0}^{j-1} exp(i * t * theta_i);  bound = mean_j |S_j|
        phases = np.exp(1j * t * theta)
        S = np.cumsum(phases)
        bound[t_idx] = np.mean(np.abs(S))
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    ax.plot(dist, bound, color=ARM_BLUE, lw=2.0, zorder=3)
    ax.fill_between(dist, bound, color=ARM_BLUE, alpha=0.10)
    ax.set_xlabel("relative distance  $|m-n|$")
    ax.set_ylabel("relative upper bound\n(mean $|S_j|$)")
    ax.set_title("Long-term decay: far-apart tokens attend less  "
                 "(paper Fig. 2, proved via Abel transform)")
    ax.set_xlim(0, 256)
    ax.annotate("nearby: strong", (6, bound[6]), (40, bound[6] + 3),
                fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("far: weak", (230, bound[230]), (150, bound[230] + 6),
                fontsize=9.5, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    save(fig, "decay.pdf")


# ---- Fig E: base-theta rescaling (DOWNSTREAM, NOT RoFormer) ------------
def fig_theta_scaling():
    models = ["RoFormer\n(this paper)", "Llama / Llama-2", "Llama-3", "Qwen2.5 / Qwen3"]
    theta = [1e4, 1e4, 5e5, 1e6]
    cols = [ARM_BLUE, GREY, ARM_ORANGE, ARM_RED]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    bars = ax.bar(x, theta, color=cols, width=0.6, zorder=3)
    ax.set_yscale("log")
    for b, v in zip(bars, theta):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                f"{v:,.0f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(r"base $\theta$ (log scale)")
    ax.set_ylim(3e3, 5e6)
    ax.set_title("Base-$\\theta$ rescaling: bigger $\\theta$ = slower rotation = usable at longer context")
    # shade the downstream region
    ax.axvspan(0.5, 3.5, color=ARM_RED, alpha=0.05, zorder=0)
    ax.text(2.0, 2.6e6, "DOWNSTREAM (Llama-3 / NTK / YaRN, 2023+)  --  NOT in the RoFormer paper",
            ha="center", fontsize=9.2, color=ARM_RED, weight="bold")
    ax.annotate("paper fixes\n$\\theta=10000$", (0.0, 1.05e4), (0.0, 1.2e5),
                ha="center", fontsize=8.8, color=ARM_BLUE,
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE))
    save(fig, "theta_scaling.pdf")


# ---- Fig F: adoption timeline (curated, NOT from paper) ----------------
def fig_adoption():
    # (model, year, marker-color)
    data = [
        ("RoFormer / RoPE", 2021.3, ARM_BLUE),
        ("GPT-NeoX", 2022.1, GREY),
        ("PaLM", 2022.3, GREY),
        ("LLaMA", 2023.1, ARM_ORANGE),
        ("Llama-2", 2023.5, ARM_ORANGE),
        ("Mistral 7B", 2023.7, GREY),
        ("Qwen", 2023.8, GREY),
        ("Llama-3", 2024.3, ARM_RED),
        ("DeepSeek", 2024.5, ARM_RED),
        ("Qwen3", 2025.3, ARM_RED),
    ]
    data = sorted(data, key=lambda r: r[1])
    ys = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x0 = 2021.0
    for y, (name, yr, col) in zip(ys, data):
        ax.plot([x0, yr], [y, y], color="#dddddd", lw=1.2, zorder=1)
        ax.scatter([yr], [y], s=90, color=col, zorder=3)
        ax.text(yr + 0.08, y, name, va="center", ha="left", fontsize=10.5)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlim(2021.0, 2026.2)
    ax.set_xticks(range(2021, 2027))
    ax.set_xlabel("year")
    ax.set_title("RoPE became the default position encoding")
    ax.text(0.5, -0.16, "curated list, not from the RoFormer paper (2021); "
            "release years approximate",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=GREY)
    ax.spines["left"].set_visible(False)
    save(fig, "adoption.pdf")


# ---- Fig G: why rotate instead of add (2-panel intuition) --------------
def fig_add_vs_rotate():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.3))
    x = np.array([0.9, 0.35])           # base token "content" vector

    # LEFT: adding a position vector changes length + direction
    axL.set_title("ADD:  position leaks into content", color=ARM_RED,
                  fontsize=12, weight="bold")
    p_m = np.array([-0.15, 0.75])
    xm = x + p_m
    for vec, col, lab, lr in [(x, ARM_BLUE, r"$x$ (content)", 1.12),
                              (p_m, GREY, r"$p_m$ (position)", 1.18),
                              (xm, ARM_RED, r"$x+p_m$", 1.10)]:
        axL.add_patch(FancyArrowPatch((0, 0), tuple(vec), arrowstyle="-|>",
                      mutation_scale=15, color=col, lw=2.4, zorder=4))
        axL.text(vec[0] * lr, vec[1] * lr, lab, color=col, fontsize=10.5,
                 weight="bold", ha="center", va="center")
    axL.text(0.5, -0.16, "length & direction change:\nthe token's meaning is altered",
             transform=axL.transAxes, ha="center", fontsize=9.3, color=ARM_RED)
    axL.set_xlim(-0.7, 1.35); axL.set_ylim(-0.15, 1.3)
    axL.set_aspect("equal"); axL.axis("off")

    # RIGHT: rotation preserves norm; position is a pure phase
    axR.set_title("ROTATE:  content fixed, position = phase", color=GREEN,
                  fontsize=12, weight="bold")
    r = float(np.linalg.norm(x))
    tt = np.linspace(0, 2 * np.pi, 200)
    axR.plot(r * np.cos(tt), r * np.sin(tt), color="#cccccc", lw=1.1, zorder=1)

    def rot(v, deg):
        a = np.deg2rad(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return R @ v

    for deg, col, lab in [(0, ARM_BLUE, r"$x$"), (55, GREEN, r"$R(m\theta)\,x$")]:
        v = rot(x, deg)
        axR.add_patch(FancyArrowPatch((0, 0), tuple(v), arrowstyle="-|>",
                      mutation_scale=15, color=col, lw=2.4, zorder=4))
        axR.text(v[0] * 1.2, v[1] * 1.2, lab, color=col, fontsize=10.5,
                 weight="bold", ha="center", va="center")
    axR.add_patch(Wedge((0, 0), 0.32, 0, 55, width=0.32, facecolor=GREEN,
                        alpha=0.18, zorder=2))
    axR.text(0.44 * np.cos(np.deg2rad(27)), 0.44 * np.sin(np.deg2rad(27)),
             r"$m\theta$", color=GREEN, fontsize=11, weight="bold",
             ha="center", va="center")
    axR.text(0.5, -0.16, "same length (norm preserved):\nonly the angle carries position",
             transform=axR.transAxes, ha="center", fontsize=9.3, color=GREEN)
    axR.set_xlim(-0.75, 1.15); axR.set_ylim(-0.35, 1.15)
    axR.set_aspect("equal"); axR.axis("off")

    fig.tight_layout()
    save(fig, "add_vs_rotate.pdf")


if __name__ == "__main__":
    fig_permutation_invariance()
    fig_rotation_2d()
    fig_frequencies()
    fig_decay()
    fig_theta_scaling()
    fig_adoption()
    fig_add_vs_rotate()
    print("all RoPE figures done")
