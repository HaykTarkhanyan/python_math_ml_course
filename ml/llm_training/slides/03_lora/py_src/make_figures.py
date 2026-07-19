"""Figures for the LoRA deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

ARM_RED = "#C81E28"; ARM_BLUE = "#1E46A0"; ARM_ORANGE = "#E6A01E"
GREEN = "#008C46"; VIOLET = "#7832A0"; GREY = "#8a8a8a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.titlesize": 13, "figure.dpi": 140})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig"); os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", os.path.normpath(p))


def fig_param_count():
    # GPT-3 175B trainable params (millions), log scale
    methods = ["Full FT", "Adapter$_H$", "Prefix-layer", "LoRA (37.7M)", "LoRA (4.7M)"]
    params = [175255.8, 40.1, 20.2, 37.7, 4.7]
    colors = [ARM_RED, GREY, GREY, ARM_BLUE, ARM_BLUE]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.barh(y, params, color=colors, zorder=3, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(methods)
    ax.invert_yaxis()
    for yi, p in zip(y, params):
        ax.text(p * 1.3, yi, f"{p:,.1f}M", va="center", fontsize=10)
    ax.set_xlabel("trainable parameters (millions, log scale)")
    ax.set_title("GPT-3 175B: LoRA trains ~10,000x fewer parameters")
    ax.set_xlim(1, 6e5)
    ax.annotate("", xy=(4.7, 4), xytext=(175255.8, 0),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
    ax.text(300, 2, r"$\approx$10,000$\times$", color=GREEN, fontsize=13, weight="bold")
    save(fig, "param_count.pdf")


def fig_rank():
    # Table 6, WikiSQL accuracy vs rank r
    r = [1, 2, 4, 8, 64]
    x = np.arange(len(r))
    wq = [68.8, 69.6, 70.5, 70.4, 70.0]
    wqv = [73.4, 73.3, 73.7, 73.8, 73.5]
    all4 = [74.1, 73.7, 74.0, 74.0, 73.9]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(x, wq, "-o", color=GREY, lw=2, label=r"$W_q$ only")
    ax.plot(x, wqv, "-o", color=ARM_BLUE, lw=2.4, label=r"$\{W_q, W_v\}$")
    ax.plot(x, all4, "-o", color=GREEN, lw=2, label=r"$\{W_q,W_k,W_v,W_o\}$")
    ax.set_xticks(x); ax.set_xticklabels(r)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("WikiSQL accuracy (%)")
    ax.set_title("Rank as low as $r=1$ already suffices for $\\{W_q,W_v\\}$")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(67, 75)
    ax.annotate("flat from r=1:\nlow intrinsic rank", (0, wqv[0]), (1.3, 71.3),
                fontsize=9.5, color=ARM_BLUE,
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE))
    save(fig, "rank.pdf")


def fig_which_matrices():
    # Table 5, fixed 18M budget
    cfg = [r"$W_q$", r"$W_k$", r"$W_v$", r"$W_o$", r"$W_q,W_k$", r"$W_q,W_v$", "all 4"]
    wikisql = [70.4, 70.0, 73.0, 73.2, 71.4, 73.7, 73.7]
    mnli = [91.0, 90.8, 91.0, 91.3, 91.3, 91.3, 91.7]
    x = np.arange(len(cfg)); w = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, data, title, lo, hi in [
        (axes[0], wikisql, "WikiSQL", 68, 75),
        (axes[1], mnli, "MultiNLI", 90, 92)]:
        cols = [GREEN if c in (r"$W_q,W_v$", "all 4") else ARM_BLUE for c in cfg]
        ax.bar(x, data, color=cols, zorder=3, width=0.62)
        ax.set_xticks(x); ax.set_xticklabels(cfg, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(lo, hi); ax.set_title(title)
        ax.set_ylabel("accuracy (%)")
        for xi, d in zip(x, data):
            ax.text(xi, d + (hi - lo) * 0.01, f"{d:.1f}", ha="center", fontsize=8)
    fig.suptitle("Same 18M budget: spread it over $\\{W_q,W_v\\}$ (green), not one matrix",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "which_matrices.pdf")


def fig_latency():
    # Table 1, GPT-2 medium, ms
    labels = ["FT / LoRA", "Adapter$_L$", "Adapter$_H$"]
    ms = [19.8, 23.9, 25.8]
    colors = [ARM_BLUE, ARM_ORANGE, ARM_RED]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.bar(x, ms, color=colors, zorder=3, width=0.55)
    for xi, m in zip(x, ms):
        ax.text(xi, m + 0.2, f"{m} ms", ha="center", fontsize=10)
    ax.text(1, 24.6, "+20.7%", ha="center", color=ARM_ORANGE, fontsize=10, weight="bold")
    ax.text(2, 26.5, "+30.3%", ha="center", color=ARM_RED, fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("inference latency (ms)")
    ax.set_title("Online inference (batch 1): adapters add latency, LoRA adds none")
    ax.set_ylim(0, 29)
    save(fig, "latency.pdf")


def fig_amplification():
    # Table 7, layer 48, Frobenius norms of delta-W projected onto top-r subspace of W
    cats = [r"$\|U^\top W_q V\|$" + "\n(self)", r"$\|U^\top \Delta W_q V\|$", "random"]
    r4 = [0.32, 21.67, 0.02]
    r64 = [1.90, 37.71, 0.33]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w / 2, r4, w, label="r=4", color=ARM_BLUE, zorder=3)
    ax.bar(x + w / 2, r64, w, label="r=64", color=ARM_ORANGE, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Frobenius norm (log)")
    ax.set_title(r"$\Delta W$ amplifies task directions already in $W$ (~21$\times$ at r=4)")
    ax.legend(frameon=False)
    ax.text(1, 28, "amplify 21.5x", ha="center", color=GREEN, fontsize=10, weight="bold")
    save(fig, "amplification.pdf")


if __name__ == "__main__":
    fig_param_count(); fig_rank(); fig_which_matrices(); fig_latency(); fig_amplification()
    print("all LoRA figures done")
