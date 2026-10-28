"""Figures for the Mixture-of-Experts deck (Switch Transformer + Mixtral).

Run with the project venv (or any env with matplotlib + numpy):
    python3 make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- shared style -------------------------------------------------------
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


# ---- Fig A (CENTERPIECE): dense FFN block vs MoE block -------------------
def fig_dense_vs_moe():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))

    # -- Dense panel --
    ax = axes[0]
    ax.set_title("Dense Transformer block", color=GREY, weight="bold")
    _box(ax, (2.0, 0.35), 1.2, 0.5, "token $x$", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=11)
    _box(ax, (2.0, 1.65), 2.4, 1.0, "FFN\n(one big MLP)", ARM_BLUE, ARM_BLUE, fs=13)
    _box(ax, (2.0, 2.95), 1.2, 0.5, "output $y$", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=11)
    _arrow(ax, (2.0, 0.6), (2.0, 1.15))
    _arrow(ax, (2.0, 2.15), (2.0, 2.7))
    ax.text(2.0, 3.55, "every token uses ALL the params",
            ha="center", color="#333333", fontsize=10, style="italic")
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.1, 3.8); ax.axis("off")

    # -- MoE panel --
    ax = axes[1]
    ax.set_title("MoE block: router picks top-$k$ of $N$ experts",
                 color=GREEN, weight="bold")
    _box(ax, (2.5, 0.3), 1.2, 0.44, "token $x$", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=10.5)
    _box(ax, (2.5, 1.02), 1.5, 0.48, "Router $W_r$", ARM_ORANGE, ARM_ORANGE,
         tc="#333333", fs=11)
    # 4 experts, top-2 selected (E2, E3)
    ex_x = [0.7, 1.95, 3.2, 4.45]
    sel = [False, True, True, False]
    gates = [None, "0.65", "0.35", None]
    for xi, s, g in zip(ex_x, sel, gates):
        col = GREEN if s else GREY
        fc = GREEN if s else "#e6e6e6"
        tc = "white" if s else "#777777"
        _box(ax, (xi, 1.95), 1.05, 0.54, "FFN", fc, col, tc=tc, fs=10.5,
             lw=(1.9 if s else 1.1))
        ls = "-" if s else (0, (2, 2))
        acol = GREEN if s else "#bbbbbb"
        _arrow(ax, (2.5, 1.26), (xi, 1.68), color=acol, lw=(1.7 if s else 1.0), ls=ls)
        if s:
            _arrow(ax, (xi, 2.22), (2.5, 2.68), color=GREEN, lw=1.7)
            ax.text(xi, 2.46, f"$p={g}$", ha="center",
                    color=GREEN, fontsize=8.5, weight="bold")
    ax.text(4.45, 2.42, "not selected", ha="center", va="center",
            color="#999999", fontsize=8)
    _box(ax, (2.5, 2.9), 0.5, 0.44, "$+$", "white", "#444444", tc="#222222", fs=14)
    _box(ax, (2.5, 3.55), 1.2, 0.44, "output $y$", "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=10.5)
    _arrow(ax, (2.5, 0.52), (2.5, 0.78))
    _arrow(ax, (2.5, 3.12), (2.5, 3.33))
    ax.set_xlim(-0.1, 5.3); ax.set_ylim(-0.1, 3.9); ax.axis("off")

    fig.suptitle(r"Replace the FFN with $N$ experts + a router: params grow with $N$, "
                 r"FLOPs/token set by $k$", fontsize=12, y=1.02, color="#333333")
    save(fig, "dense_vs_moe.pdf")


# ---- Fig B: total vs active parameters ---------------------------------
def fig_total_vs_active():
    models = ["Switch-Base", "Switch-Large", "Mixtral 8x7B"]
    total = [7.0, 26.0, 47.0]        # billions, sparse/total param count
    active = [0.2, 0.7, 13.0]        # active-per-token (B): Switch = FLOP-matched
                                     # dense baseline (T5-Base/Large); Mixtral reported
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    b1 = ax.bar(x - w / 2, total, w, label="total params", color=ARM_BLUE, zorder=3)
    b2 = ax.bar(x + w / 2, active, w, label="active per token", color=GREEN, zorder=3)
    ax.set_yscale("log")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                    f"{b.get_height():g}B", ha="center", fontsize=9.5)
    for i in range(len(models)):
        ax.annotate(f"{total[i] / active[i]:.0f}x", (x[i], total[i] * 1.9),
                    ha="center", color=ARM_RED, fontsize=11, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("parameters (billions, log)")
    ax.set_ylim(0.1, 200)
    ax.set_title("MoE decouples capacity from compute: huge total, small active")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.5, -0.30, "Switch active = FLOP-matched dense baseline (T5-Base/Large);  "
            "Mixtral 13B is the paper's reported active count.",
            transform=ax.transAxes, ha="center", fontsize=8.2, color=GREY)
    save(fig, "total_vs_active.pdf")


# ---- Fig C: FLOP-matched speedup (real numbers) ------------------------
def fig_flop_speedup():
    labels = ["Switch-Base\nvs T5-Base", "Switch-Base\nvs T5-Large",
              "Switch-C 1.6T\nvs T5-XXL"]
    speedup = [7.0, 2.5, 4.0]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    bars = ax.bar(x, speedup, color=[ARM_BLUE, ARM_BLUE, VIOLET], width=0.55, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                f"{b.get_height():g}x", ha="center", fontsize=12, weight="bold")
    ax.axhline(1.0, color=GREY, ls=":", lw=1.3)
    ax.text(2.42, 1.05, "dense baseline (1x)", ha="right", color=GREY, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("speedup to a fixed quality\n(equal compute budget)")
    ax.set_ylim(0, 8.2)
    ax.set_title("Same FLOPs/token, far faster: sparse beats dense at equal compute")
    ax.text(1, 4.3, "T5-Large uses 3.5x more\nFLOPs/token, still slower",
            ha="center", fontsize=8.6, color="#555555", style="italic")
    save(fig, "flop_speedup.pdf")


# ---- Fig D: load balancing (schematic) ---------------------------------
def fig_load_balancing():
    N = 8
    idx = np.arange(1, N + 1)
    unbal = np.array([0.40, 0.27, 0.15, 0.08, 0.05, 0.03, 0.02, 0.00])
    bal = np.array([0.135, 0.12, 0.13, 0.115, 0.125, 0.13, 0.12, 0.125])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)

    ax = axes[0]
    cols = [ARM_RED if v < 0.03 else ARM_BLUE for v in unbal]
    ax.bar(idx, unbal, color=cols, width=0.7, zorder=3)
    ax.axhline(1 / N, color=GREEN, ls="--", lw=1.6)
    ax.text(N, 1 / N + 0.005, "uniform $1/N$", color=GREEN, ha="right", fontsize=9.5)
    ax.set_title("Without aux loss: collapse", color=ARM_RED, weight="bold")
    ax.annotate("dead experts\n(never picked)", (7.5, 0.02), (5.0, 0.20),
                fontsize=9, color=ARM_RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xlabel("expert id"); ax.set_ylabel("fraction of tokens routed")
    ax.set_xticks(idx)

    ax = axes[1]
    ax.bar(idx, bal, color=ARM_BLUE, width=0.7, zorder=3)
    ax.axhline(1 / N, color=GREEN, ls="--", lw=1.6)
    ax.text(N, 1 / N + 0.005, "uniform $1/N$", color=GREEN, ha="right", fontsize=9.5)
    ax.set_title(r"With aux loss $\;\alpha N\sum_i f_i P_i\;$: balanced",
                 color=GREEN, weight="bold")
    ax.set_xlabel("expert id"); ax.set_xticks(idx)

    fig.suptitle("Load-balancing auxiliary loss keeps all experts used  (schematic; "
                 r"$N=8$)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "load_balancing.pdf")


# ---- Fig E: expert capacity / token dropping (schematic) ---------------
def fig_capacity_factor():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
    # each expert: a stack of capacity slots; tokens fill from bottom.
    # routed[i] = how many tokens the router sent to expert i (same in both panels)
    routed = [2, 3, 1]
    experts = ["Expert 1", "Expert 2", "Expert 3"]
    slot_h, slot_w, gap = 0.46, 0.9, 1.6

    def draw(ax, cap, title):
        ax.set_title(title, weight="bold", fontsize=12)
        for e, (name, n) in enumerate(zip(experts, routed)):
            x0 = 0.6 + e * gap
            for s in range(cap):
                filled = s < n
                fc = ARM_BLUE if filled else "#f0f0f0"
                ec = ARM_BLUE if filled else "#c8c8c8"
                ax.add_patch(FancyBboxPatch((x0, 0.5 + s * (slot_h + 0.06)), slot_w, slot_h,
                             boxstyle="round,pad=0.01,rounding_size=0.03",
                             fc=fc, ec=ec, lw=1.4, zorder=3))
                if not filled:
                    ax.text(x0 + slot_w / 2, 0.5 + s * (slot_h + 0.06) + slot_h / 2,
                            "pad", ha="center", va="center", color="#a0a0a0", fontsize=8)
            # overflow (dropped) tokens above the stack
            if n > cap:
                for d in range(n - cap):
                    yd = 0.5 + (cap + d) * (slot_h + 0.06)
                    ax.add_patch(FancyBboxPatch((x0, yd), slot_w, slot_h,
                                 boxstyle="round,pad=0.01,rounding_size=0.03",
                                 fc="white", ec=ARM_RED, lw=1.6, ls=(0, (2, 2)), zorder=3))
                    ax.text(x0 + slot_w / 2, yd + slot_h / 2, "drop",
                            ha="center", va="center", color=ARM_RED, fontsize=8.5,
                            weight="bold")
            ax.text(x0 + slot_w / 2, 0.18, name, ha="center", fontsize=9.5)
            ax.text(x0 + slot_w / 2, -0.12, f"{n} routed", ha="center", fontsize=8.5,
                    color=GREY)
        ax.set_xlim(0.1, 0.6 + 3 * gap); ax.set_ylim(-0.35, 2.75); ax.axis("off")

    draw(axes[0], 2, "Capacity factor 1.0  (2 slots)")
    axes[0].text(2.7, 2.55, "overflow token dropped\n$\\rightarrow$ skips layer via residual",
                 ha="center", color=ARM_RED, fontsize=9)
    draw(axes[1], 3, "Capacity factor 1.5  (3 slots)")
    axes[1].text(2.7, 2.55, "fits all tokens, but\nempty slots = wasted compute",
                 ha="center", color="#777777", fontsize=9)
    fig.suptitle("Expert capacity $=\\;$(tokens / experts) $\\times$ capacity factor: "
                 "the drop-vs-waste trade-off", fontsize=12, y=1.01)
    save(fig, "capacity_factor.pdf")


# ---- Fig F: Mixtral vs Llama-2-70B / GPT-3.5 (real numbers) -------------
def fig_mixtral_benchmarks():
    bench = ["MMLU", "HellaSwag", "ARC-C", "MBPP", "GSM8K"]
    llama = [69.9, 87.1, 85.1, 49.8, 53.6]
    gpt35 = [70.0, 85.5, 85.2, 52.2, 57.1]
    mixtral = [70.6, 86.7, 85.8, 60.7, 58.4]
    x = np.arange(len(bench)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    b1 = ax.bar(x - w, llama, w, label="Llama-2 70B  (70B active)", color=GREY, zorder=3)
    b2 = ax.bar(x, gpt35, w, label="GPT-3.5", color=ARM_ORANGE, zorder=3)
    b3 = ax.bar(x + w, mixtral, w, label="Mixtral 8x7B  (13B active)",
                color=ARM_BLUE, zorder=3)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Mixtral 8x7B matches/beats Llama-2 70B & GPT-3.5 at 5x fewer active params")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "mixtral_benchmarks.pdf")


if __name__ == "__main__":
    fig_dense_vs_moe()
    fig_total_vs_active()
    fig_flop_speedup()
    fig_load_balancing()
    fig_capacity_factor()
    fig_mixtral_benchmarks()
    print("all MoE figures done")
