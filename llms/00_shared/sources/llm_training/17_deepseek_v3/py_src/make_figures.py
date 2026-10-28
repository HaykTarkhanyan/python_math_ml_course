"""Figures for the DeepSeek-V3 deck ([LLM-17]).

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/ (resolved relative to THIS file).
Fails loud on any error (no silent fallback).

All schematic figures are labelled "schematic". Benchmark / cost figures use
REAL numbers from the DeepSeek-V3 technical report (arXiv:2412.19437, Tables 1
and 6). Attribution note: MLA and DeepSeekMoE are INHERITED from DeepSeek-V2 and
re-validated in V3; auxiliary-loss-free balancing, large-scale FP8 training and
Multi-Token Prediction are V3's own contributions. four_levers.pdf marks this.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SEED = 509
np.random.seed(SEED)

# ---- shared style (matches 13_moe) --------------------------------------
ARM_RED = "#C81E28"      # armred
ARM_BLUE = "#1E46A0"     # armblue
ARM_ORANGE = "#E6A01E"   # armorange
GREEN = "#008C46"        # paramgreen
VIOLET = "#7832A0"       # violet1
GREY = "#8a8a8a"
LGREY = "#c8c8c8"

# Armenian-flag "pure" palette for 3+ color charts (per CLAUDE.md).
FLAG_RED = "#D90012"
FLAG_BLUE = "#0033A0"
FLAG_ORANGE = "#F2A800"

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


# ---- Fig 1 (CENTERPIECE): four/five levers, inherited vs V3-new ---------
def fig_four_levers():
    fig, ax = plt.subplots(figsize=(11.6, 5.2))

    # top: model spec
    _box(ax, (5.0, 5.15), 6.4, 0.7,
         "DeepSeek-V3  --  671B total params, 37B active per token",
         ARM_BLUE, ARM_BLUE, tc="white", fs=13)

    # column headers
    ax.text(2.35, 4.25, "Inherited from DeepSeek-V2\n(thoroughly re-validated in V3)",
            ha="center", va="center", color="#555555", fontsize=11, weight="bold")
    ax.text(7.65, 4.25, "New in DeepSeek-V3",
            ha="center", va="center", color=GREEN, fontsize=11.5, weight="bold")

    # inherited boxes (muted blue-grey)
    inh_fc, inh_ec = "#dde3f0", "#8a95b5"
    _box(ax, (2.35, 3.30), 3.7, 0.72, "MLA -- compress the KV cache",
         inh_fc, inh_ec, tc="#2b3a66", fs=11)
    _box(ax, (2.35, 2.30), 3.7, 0.72, "DeepSeekMoE -- fine-grained experts",
         inh_fc, inh_ec, tc="#2b3a66", fs=11)

    # V3-new boxes (green)
    _box(ax, (7.65, 3.55), 3.9, 0.66, "Auxiliary-loss-free load balancing",
         GREEN, GREEN, tc="white", fs=11)
    _box(ax, (7.65, 2.70), 3.9, 0.66, "FP8 training at large scale",
         GREEN, GREEN, tc="white", fs=11)
    _box(ax, (7.65, 1.85), 3.9, 0.66, "Multi-Token Prediction (MTP)",
         GREEN, GREEN, tc="white", fs=11)

    # divider
    ax.plot([5.0, 5.0], [1.35, 4.05], color=LGREY, lw=1.2, ls=(0, (3, 3)), zorder=1)

    # bottom band: DualPipe co-design
    _box(ax, (5.0, 0.72), 9.2, 0.72,
         "DualPipe co-design: computation-communication overlap  ->  a big reason it was cheap",
         "#fbf1d8", ARM_ORANGE, tc="#7a5300", fs=11)

    # legend swatches
    ax.add_patch(FancyBboxPatch((0.35, -0.35), 0.32, 0.26,
                 boxstyle="round,pad=0.01", fc=inh_fc, ec=inh_ec, lw=1.3))
    ax.text(0.8, -0.22, "inherited from V2", va="center", fontsize=9, color="#555555")
    ax.add_patch(FancyBboxPatch((3.3, -0.35), 0.32, 0.26,
                 boxstyle="round,pad=0.01", fc=GREEN, ec=GREEN, lw=1.3))
    ax.text(3.75, -0.22, "new in V3", va="center", fontsize=9, color=GREEN)

    ax.text(9.9, -0.22, "schematic", va="center", ha="right",
            fontsize=8.5, color=GREY, style="italic")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-0.55, 5.6)
    ax.axis("off")
    fig.suptitle("A frontier model from a stack of efficiency levers, not brute scale",
                 fontsize=12.5, y=1.0, color="#333333")
    save(fig, "four_levers.pdf")


# ---- Fig 2: MLA -- KV cache -> low-rank latent -> per-head ---------------
def fig_mla_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # -- left: the flow --
    ax = axes[0]
    ax.set_title("MLA: cache one small latent, rebuild per head",
                 color=ARM_BLUE, weight="bold", fontsize=12)
    _box(ax, (1.4, 3.5), 2.0, 0.6, "token hidden $h_t$", "#eef2fb", ARM_BLUE,
         tc=ARM_BLUE, fs=10.5)
    _box(ax, (1.4, 2.35), 2.3, 0.62, "down-project $W^{DKV}$", ARM_ORANGE, ARM_ORANGE,
         tc="#5a4300", fs=10)
    _box(ax, (1.4, 1.2), 2.5, 0.7, "latent $c_t$\n(this is ALL you cache)",
         GREEN, GREEN, tc="white", fs=10)
    _arrow(ax, (1.4, 3.18), (1.4, 2.68))
    _arrow(ax, (1.4, 2.02), (1.4, 1.56))

    # up-projections to per-head K,V, drawn on the right of the latent
    for i, yy in enumerate([2.0, 1.2, 0.4]):
        lab = "$K_1,V_1$" if i == 0 else ("$K_2,V_2$" if i == 1 else "$K_h,V_h$")
        _box(ax, (4.4, yy), 1.5, 0.5, lab, "#eef7f0", GREEN, tc="#155f38", fs=9.5)
        _arrow(ax, (2.68, 1.2), (3.62, yy), color=GREEN, lw=1.3)
    ax.text(4.4, 2.55, "up-project $W^{UK},W^{UV}$\nper head, at use time",
            ha="center", fontsize=8.6, color="#155f38", style="italic")
    ax.text(1.4, 0.35, "small $d_c \\ll d_h n_h$", ha="center",
            fontsize=8.6, color=GREEN)
    ax.set_xlim(0.0, 5.4)
    ax.set_ylim(0.0, 4.0)
    ax.axis("off")

    # -- right: KV-cache size contrast (schematic, relative) --
    ax = axes[1]
    labels = ["standard\nMHA cache", "MLA\nlatent cache"]
    sizes = [1.0, 0.07]      # schematic relative bytes/token
    cols = [ARM_RED, GREEN]
    bars = ax.bar(labels, sizes, color=cols, width=0.6, zorder=3)
    for b, s in zip(bars, sizes):
        txt = "full K,V\nper head" if s > 0.5 else "just $c_t$"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                txt, ha="center", fontsize=9, weight="bold",
                color=(ARM_RED if s > 0.5 else GREEN))
    ax.set_ylabel("KV cache per token\n(relative, schematic)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Much smaller cache,\nnear-lossless quality", fontsize=11)
    ax.annotate("", xy=(1, 0.16), xytext=(0, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6))
    ax.text(0.5, 0.62, "big cut", ha="center", color="#555555",
            fontsize=9.5, style="italic")
    fig.suptitle("MLA (inherited from DeepSeek-V2): project K,V into a shared "
                 "low-rank latent  --  schematic", fontsize=11.5, y=1.02, color="#333333")
    fig.tight_layout()
    save(fig, "mla_diagram.pdf")


# ---- Fig 3: auxiliary-loss-free balancing -------------------------------
def fig_aux_free_balancing():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    # -- left: the routing rule --
    ax = axes[0]
    ax.set_title("A per-expert bias steers routing only",
                 color=ARM_BLUE, weight="bold", fontsize=11.5)
    _box(ax, (1.6, 3.3), 2.6, 0.56, "affinity $s_i$", "#eef2fb", ARM_BLUE,
         tc=ARM_BLUE, fs=10.5)
    _box(ax, (1.6, 2.25), 2.6, 0.56, "add bias  $s_i + b_i$", GREEN, GREEN,
         tc="white", fs=10.5)
    _box(ax, (1.6, 1.2), 2.6, 0.56, "top-$K$ routing", ARM_ORANGE, ARM_ORANGE,
         tc="#5a4300", fs=10.5)
    _arrow(ax, (1.6, 3.02), (1.6, 2.53))
    _arrow(ax, (1.6, 1.97), (1.6, 1.48))
    # gate value uses original s_i (bypasses the bias)
    _box(ax, (4.6, 2.25), 2.3, 0.62, "gate value\nuses $s_i$ only", "#eef7f0",
         GREEN, tc="#155f38", fs=9.5)
    _arrow(ax, (2.9, 3.3), (4.6, 2.56), color=GREEN, lw=1.3, ls=(0, (3, 2)))
    ax.text(1.6, 0.55, "bias only picks WHO;\ngate weight is untouched",
            ha="center", fontsize=9, color="#555555", style="italic")
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.2, 3.8)
    ax.axis("off")

    # -- right: load converges to uniform as bias is nudged --
    ax = axes[1]
    steps = np.arange(0, 60)
    target = 1.0
    # overloaded expert cooling down, underloaded warming up (schematic)
    hot = 1.9 * np.exp(-steps / 14) + target * (1 - np.exp(-steps / 14))
    cold = 0.2 + (target - 0.2) * (1 - np.exp(-steps / 14))
    hot += np.random.normal(0, 0.015, steps.shape)
    cold += np.random.normal(0, 0.015, steps.shape)
    ax.plot(steps, hot, color=ARM_RED, lw=2.0, label="overloaded expert  ($-\\gamma$)")
    ax.plot(steps, cold, color=ARM_BLUE, lw=2.0, label="underloaded expert  ($+\\gamma$)")
    ax.axhline(target, color=GREEN, ls="--", lw=1.6)
    ax.text(59, target + 0.06, "balanced load", color=GREEN, ha="right", fontsize=9.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("relative expert load")
    ax.set_ylim(0, 2.1)
    ax.set_title("Bias nudged each step:  $-\\gamma$ if overloaded,  $+\\gamma$ if "
                 "underloaded", fontsize=10.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.text(0.99, -0.22, "schematic", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=GREY, style="italic")
    fig.suptitle("Auxiliary-loss-free balancing (V3): equalize experts with a routing "
                 "bias, no aux-loss quality tax", fontsize=11.5, y=1.03, color="#333333")
    fig.tight_layout()
    save(fig, "aux_free_balancing.pdf")


# ---- Fig 4: FP8 map -- which ops FP8 vs higher precision -----------------
def fig_fp8_map():
    fig, ax = plt.subplots(figsize=(11.4, 4.7))

    # left panel: FP8 (the heavy GEMMs)
    ax.text(2.55, 4.35, "FP8 (8-bit): the heavy GEMMs", ha="center",
            color=GREEN, weight="bold", fontsize=12)
    for i, t in enumerate(["Fprop  (forward matmul)",
                           "Dgrad  (activation gradient)",
                           "Wgrad  (weight gradient)"]):
        _box(ax, (2.55, 3.55 - i * 0.85), 4.0, 0.62, t, GREEN, GREEN,
             tc="white", fs=10.5)
    _box(ax, (2.55, 0.95), 4.0, 0.6, "~2x compute throughput, less memory",
         "#eef7f0", GREEN, tc="#155f38", fs=10)

    # right panel: kept high precision
    ax.text(7.65, 4.35, "Kept in BF16 / FP32: the sensitive parts", ha="center",
            color=ARM_ORANGE, weight="bold", fontsize=12)
    sens = ["embedding & output head", "MoE gating / router",
            "normalization (RMSNorm)", "attention", "FP32 accumulation of FP8 GEMMs"]
    for i, t in enumerate(sens):
        _box(ax, (7.65, 3.75 - i * 0.62), 4.2, 0.5, t, "#fbf1d8", ARM_ORANGE,
             tc="#7a5300", fs=10)

    ax.plot([5.05, 5.05], [0.6, 4.15], color=LGREY, lw=1.2, ls=(0, (3, 3)))

    # bottom note
    _box(ax, (5.1, 0.05), 9.2, 0.62,
         "Fine-grained scaling: 1x128 tiles (activations), 128x128 blocks (weights)"
         "  ->  loss error < 0.25% vs BF16",
         "#eef2fb", ARM_BLUE, tc="#2b3a66", fs=10)

    ax.text(9.85, -0.5, "schematic", ha="right", fontsize=8.5,
            color=GREY, style="italic")
    ax.set_xlim(0.2, 10.0)
    ax.set_ylim(-0.6, 4.7)
    ax.axis("off")
    fig.suptitle("FP8 training (V3): 8-bit for the big matmuls, higher precision "
                 "where it matters", fontsize=12, y=1.0, color="#333333")
    save(fig, "fp8_map.pdf")


# ---- Fig 5: MTP -- next-1 vs sequential next-k modules -------------------
def fig_mtp_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3),
                             gridspec_kw={"width_ratios": [0.8, 1.25]})

    # -- left: standard next-token --
    ax = axes[0]
    ax.set_title("Standard: predict next token only",
                 color="#555555", weight="bold", fontsize=11)
    _box(ax, (1.5, 1.4), 2.2, 0.9, "Main model", ARM_BLUE, ARM_BLUE,
         tc="white", fs=11)
    _box(ax, (1.5, 3.1), 2.0, 0.6, "token $t{+}1$", "#eef7f0", GREEN,
         tc="#155f38", fs=10)
    _arrow(ax, (1.5, 1.88), (1.5, 2.78), color=GREEN, lw=1.6)
    ax.text(1.5, 0.55, "one target per position", ha="center",
            fontsize=9, color="#777777", style="italic")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.2, 3.7)
    ax.axis("off")

    # -- right: MTP sequential modules keeping the causal chain --
    ax = axes[1]
    ax.set_title("MTP: sequential modules keep the full causal chain",
                 color=GREEN, weight="bold", fontsize=11)
    xs = [1.1, 3.15, 5.2]
    names = ["Main model", "MTP module 1", "MTP module 2"]
    tgts = ["token $t{+}1$", "token $t{+}2$", "token $t{+}3$"]
    cols = [ARM_BLUE, GREEN, GREEN]
    for x, nm, tg, c in zip(xs, names, tgts, cols):
        _box(ax, (x, 1.4), 1.75, 0.85, nm, c, c, tc="white", fs=9.8)
        _box(ax, (x, 3.05), 1.6, 0.55, tg, "#eef7f0", GREEN, tc="#155f38", fs=9.3)
        _arrow(ax, (x, 1.85), (x, 2.75), color=GREEN, lw=1.5)
    # sequential hidden-state chain between modules
    _arrow(ax, (1.98, 1.4), (2.28, 1.4), color="#444444", lw=1.7)
    _arrow(ax, (4.03, 1.4), (4.33, 1.4), color="#444444", lw=1.7)
    ax.text(3.6, 0.62, "each module reuses the previous depth's representation "
            "(not parallel independent heads)",
            ha="center", fontsize=8.6, color="#555555", style="italic")
    ax.text(5.2, 3.75, "shared embedding + output head",
            ha="center", fontsize=8.4, color=GREEN)
    ax.set_xlim(0.0, 6.2)
    ax.set_ylim(0.3, 4.0)
    ax.axis("off")

    fig.suptitle("Multi-Token Prediction (V3): denser signal per position, and it "
                 "enables speculative decoding  --  schematic",
                 fontsize=11.5, y=1.02, color="#333333")
    fig.tight_layout()
    save(fig, "mtp_diagram.pdf")


# ---- Fig 6: benchmarks (REAL, Table 6) ----------------------------------
def fig_benchmarks():
    bench = ["MMLU", "GPQA-Diamond", "MATH-500", "Codeforces\n(pctile)"]
    # DeepSeek-V3 tech report, Table 6.
    v3     = [88.5, 59.1, 90.2, 51.6]
    qwen   = [85.3, 49.0, 80.0, 24.8]
    llama  = [88.6, 51.1, 73.8, 25.3]
    claude = [88.3, 65.0, 78.3, 20.3]
    gpt4o  = [87.2, 49.9, 74.6, 23.6]

    x = np.arange(len(bench))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    series = [
        ("Qwen2.5-72B (open)", qwen, GREY, -2),
        ("Llama-3.1-405B (open)", llama, "#5a5a5a", -1),
        ("GPT-4o-0513 (closed)", gpt4o, FLAG_ORANGE, 0),
        ("Claude-3.5-Sonnet (closed)", claude, FLAG_RED, 1),
        ("DeepSeek-V3 (37B active)", v3, FLAG_BLUE, 2),
    ]
    for name, vals, col, off in series:
        bars = ax.bar(x + off * w, vals, w, label=name, color=col, zorder=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.0f}", ha="center", fontsize=6.6,
                    color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(bench)
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Best open model, competitive with frontier closed models "
                 "(DeepSeek-V3 report, Table 6)")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.22)
    save(fig, "benchmarks.pdf")


# ---- Fig 7: training cost (REAL, Table 1) -------------------------------
def fig_cost():
    phases = ["Pre-training", "Context ext.\n(-> 128K)", "Post-training"]
    gpu_k = [2664, 119, 5]        # thousands of H800 GPU-hours (Table 1)
    usd = ["\\$5.328M", "\\$0.238M", "\\$0.01M"]
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    bars = ax.bar(phases, gpu_k, color=[FLAG_BLUE, FLAG_ORANGE, GREEN],
                  width=0.6, zorder=3)
    for b, u in zip(bars, usd):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 45,
                f"{b.get_height():,}K\n{u}", ha="center", fontsize=9.5,
                weight="bold", color="#333333")
    ax.set_ylabel("H800 GPU-hours (thousands)")
    ax.set_ylim(0, 3050)
    ax.set_title("DeepSeek-V3 full training: 2.788M H800-hours = \\$5.576M "
                 "(at \\$2 / GPU-hour)")
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.5, -0.26, "180K GPU-hours per trillion tokens  --  ~3.7 days/T "
            "on a 2048-GPU H800 cluster.  (Table 1; \\$2/GPU-hour assumption.)",
            transform=ax.transAxes, ha="center", fontsize=8.4, color=GREY)
    save(fig, "cost.pdf")


# ---- Fig H: FP8 number line -- why fine-grained scaling (schematic) -----
def fig_fp8_numberline():
    rng = np.random.default_rng(SEED)
    R = 4.0
    # schematic FP8 representable levels: few, non-uniform, denser near 0.
    pos = np.array([0.05, 0.1, 0.18, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 3.6, R])
    levels = np.unique(np.concatenate([-pos[::-1], [0.0], pos]))
    block = rng.normal(0, 0.06, 14)          # a tile of small weights

    def snap(v):
        v = np.asarray(v)
        return levels[np.argmin(np.abs(levels[None, :] - v[:, None]), axis=1)]

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.8, 4.5))

    def draw(ax, vals, title, tcol):
        ax.set_title(title, color=tcol, fontsize=11.5, weight="bold", loc="left")
        ax.axhline(0, color="#bbbbbb", lw=1.0, zorder=1)
        for lv in levels:
            ax.plot([lv, lv], [-0.10, 0.10], color=LGREY, lw=1.0, zorder=1)
        ax.scatter(vals, np.zeros_like(vals), s=32, color=FLAG_BLUE,
                   zorder=3, alpha=0.85)
        sn = snap(vals)
        ax.scatter(sn, np.zeros_like(sn) - 0.30, s=34, marker="^",
                   color=FLAG_ORANGE, zorder=3)
        ax.set_xlim(-R * 1.12, R * 1.12)
        ax.set_ylim(-0.62, 0.42)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    draw(axA, block,
         "Raw: a tile of small weights collapses onto 1-2 FP8 levels (info lost)", ARM_RED)
    draw(axB, block * 20.0,
         "Scaled per tile (x scale): the same weights span many levels (info kept)", GREEN)
    axB.text(0.5, -0.55, "grey ticks = FP8 representable levels   |   "
             "blue = true values   |   orange = nearest FP8 level",
             transform=axB.transAxes, ha="center", fontsize=8.4, color=GREY)
    axA.text(0.005, 1.02, "schematic", transform=axA.transAxes, fontsize=8,
             color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "fp8_numberline.pdf")


if __name__ == "__main__":
    fig_four_levers()
    fig_mla_diagram()
    fig_aux_free_balancing()
    fig_fp8_map()
    fig_fp8_numberline()
    fig_mtp_diagram()
    fig_benchmarks()
    fig_cost()
    print("all DeepSeek-V3 figures done")
