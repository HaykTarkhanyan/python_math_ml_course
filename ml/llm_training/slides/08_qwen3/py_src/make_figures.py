"""Figures for the Qwen3 deck. Run: python make_figures.py -> ../fig/*.pdf"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


def fig_model_lineup():
    # Table 1 (dense) + Table 2 (MoE): total vs activated params (billions).
    names = ["0.6B", "1.7B", "4B", "8B", "14B", "30B-A3B\n(MoE)", "32B",
             "235B-A22B\n(MoE)"]
    total = [0.6, 1.7, 4, 8, 14, 30, 32, 235]
    active = [0.6, 1.7, 4, 8, 14, 3, 32, 22]
    is_moe = [False] * 8
    is_moe[5] = is_moe[7] = True
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    for yi, tot, act, moe in zip(y, total, active, is_moe):
        if moe:
            ax.barh(yi, tot, color=ARM_ORANGE + "55", zorder=3, height=0.62,
                    edgecolor=ARM_ORANGE)
            ax.barh(yi, act, color=ARM_ORANGE, zorder=4, height=0.62)
            ax.text(tot * 1.35, yi, f"{tot:g}B total", va="center", fontsize=9)
            ax.text(act * 0.9, yi, f"{act:g}B", va="center", ha="right",
                    fontsize=9, color="white", weight="bold")
        else:
            ax.barh(yi, tot, color=ARM_BLUE, zorder=3, height=0.62)
            ax.text(tot * 1.35, yi, f"{tot:g}B", va="center", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 800)
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_title("Qwen3 family: 6 dense + 2 MoE models; MoE activates a small fraction")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ARM_BLUE, label="dense (all params active)"),
                       Patch(color=ARM_ORANGE, label="MoE activated params"),
                       Patch(facecolor=ARM_ORANGE + "55", edgecolor=ARM_ORANGE,
                             label="MoE total params")],
              frameon=False, loc="center right", fontsize=9,
              bbox_to_anchor=(1.0, 0.62))
    save(fig, "model_lineup.pdf")


def fig_base_efficiency():
    # Table 3: Qwen3-235B-A22B-Base vs strong open base models.
    bench = ["MMLU-Pro", "GPQA", "MATH"]
    models = ["Qwen2.5-72B\n(72B/72B)", "Llama-4-Maverick\n(402B/17B)",
              "DeepSeek-V3\n(671B/37B)", "Qwen3-235B-A22B\n(235B/22B)"]
    # rows: model -> [mmlu-pro, gpqa, math]
    data = {
        models[0]: [58.07, 45.88, 62.12],
        models[1]: [63.91, 43.94, 63.32],
        models[2]: [59.84, 41.92, 62.62],
        models[3]: [68.18, 47.47, 71.84],
    }
    colors = [GREY, GREY, GREY, ARM_BLUE]
    x = np.arange(len(bench)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    for i, (m, c) in enumerate(zip(models, colors)):
        off = (i - 1.5) * w
        bars = ax.bar(x + off, data[m], w, label=m, color=c, zorder=3)
        for b, v in zip(bars, data[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylabel("base-model score (%)")
    ax.set_ylim(35, 80)
    ax.set_title("Pre-trained base: Qwen3-235B-A22B tops peers with far fewer params")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    save(fig, "base_efficiency.pdf")


def fig_thinking_budget():
    # Figure 2 (p.20), read from the plot (approximate). x = thinking budget (K tokens).
    x = np.array([1, 2, 4, 8, 16, 32])
    aime24 = [42, 43, 56, 72, 83, 85.5]
    aime25 = [30.5, 35.5, 42.5, 59, 77, 81]
    lcb = [45, 48, 52, 58.5, 65, 67.5]
    gpqa = [64.2, 64.4, 68.1, 70.3, 70.6, 72]
    base = {"AIME'24": 40, "AIME'25": 26, "LiveCodeBench": 36, "GPQA-Diamond": 65}
    series = [("AIME'24", aime24, ARM_RED), ("AIME'25", aime25, ARM_ORANGE),
              ("LiveCodeBench", lcb, GREEN), ("GPQA-Diamond", gpqa, ARM_BLUE)]
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    for name, ys, c in series:
        ax.plot(x, ys, "-o", color=c, lw=2.2, ms=5, label=name)
        ax.axhline(base[name], color=c, ls=":", lw=1.1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{v}K" for v in x])
    ax.set_xlabel("thinking budget (tokens, log scale)")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("More thinking tokens -> higher accuracy, then it plateaus")
    ax.set_ylim(24, 90)
    ax.legend(frameon=False, loc="center right", fontsize=9.5)
    ax.text(1.05, 34, "dotted = non-thinking mode (fixed)", fontsize=8.5,
            color=GREY, style="italic")
    fig.text(0.5, -0.02, "values read from Figure 2 (approximate)", ha="center",
             fontsize=8.5, color=GREY, style="italic")
    save(fig, "thinking_budget.pdf")


def fig_pipeline():
    # Figure 1 schematic: 4-stage post-training + strong-to-weak distillation.
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(cx, cy, w, h, text, fc, ec, tc="black", fs=9.5, bold=True):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4, weight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=15, lw=1.8, color=color, zorder=2))

    yT = 3.2
    box(1.0, yT, 1.5, 0.9, "Base\nmodel", "#eeeeee", GREY)
    stages = [
        (3.1, "Stage 1\nLong-CoT\nCold Start", GREY + "33", GREY),
        (5.0, "Stage 2\nReasoning RL\n(GRPO)", ARM_BLUE + "22", ARM_BLUE),
        (6.9, "Stage 3\nThinking\nMode Fusion", ARM_ORANGE + "33", ARM_ORANGE),
        (8.8, "Stage 4\nGeneral RL", GREEN + "22", GREEN),
    ]
    for cx, txt, fc, ec in stages:
        box(cx, yT, 1.55, 1.0, txt, fc, ec, fs=8.8)
    box(10.4, yT, 1.15, 1.0, "Qwen3\nflagship", ARM_RED + "18", ARM_RED, fs=8.8)
    xs = [1.0, 3.1, 5.0, 6.9, 8.8, 10.4]
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 0.78, yT, b - 0.78, yT)

    # brackets: stages 1-2 build thinking, 3-4 add non-thinking
    ax.annotate("", xy=(5.85, 4.05), xytext=(2.3, 4.05),
                arrowprops=dict(arrowstyle="-", color=ARM_BLUE, lw=1.2))
    ax.text(4.05, 4.25, "build THINKING (reasoning)", ha="center", fontsize=8.5,
            color=ARM_BLUE, weight="bold")
    ax.annotate("", xy=(9.6, 4.05), xytext=(6.1, 4.05),
                arrowprops=dict(arrowstyle="-", color=ARM_ORANGE, lw=1.2))
    ax.text(7.85, 4.25, "fuse NON-THINKING + generalize", ha="center", fontsize=8.5,
            color=ARM_ORANGE, weight="bold")

    # distillation branch for lightweight models
    yB = 1.0
    box(4.4, yB, 3.6, 0.95, "Strong-to-Weak Distillation\n(off-policy + on-policy, match logits)",
        VIOLET + "10", VIOLET, fs=8.6)
    box(8.7, yB, 2.2, 0.95, "Lightweight Qwen3 models\n(5 dense + 1 MoE, cheap)",
        VIOLET + "18", VIOLET, fs=8.2)
    # flagship (teacher) feeds the distillation process
    ax.add_patch(FancyArrowPatch((10.4, yT - 0.55), (4.9, yB + 0.5),
                 connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                 mutation_scale=15, lw=1.7, color=VIOLET, zorder=2))
    ax.text(7.6, 2.15, "teacher logits", fontsize=8.0, color=VIOLET,
            ha="center", style="italic")
    arrow(6.25, yB, 7.55, yB, color=VIOLET)   # distillation -> lightweight models
    save(fig, "pipeline.pdf")


def fig_distill_vs_rl():
    # Table 21: from same off-policy 8B checkpoint, +RL vs +on-policy distillation.
    bench = ["AIME'24", "AIME'25", "MATH500", "LiveCodeBench", "GPQA-Diamond"]
    start = [55.0, 42.8, 92.4, 42.0, 55.6]        # off-policy distilled checkpoint
    rl = [67.6, 55.5, 94.8, 52.9, 61.3]           # + RL (17,920 GPU-hours)
    distill = [74.4, 65.5, 97.0, 60.3, 63.3]      # + on-policy distill (1,800 GPU-h)
    x = np.arange(len(bench)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    b0 = ax.bar(x - w, start, w, label="start: off-policy distilled", color=GREY,
                zorder=3)
    b1 = ax.bar(x, rl, w, label="+ RL  (17,920 GPU-h)", color=ARM_BLUE, zorder=3)
    b2 = ax.bar(x + w, distill, w, label="+ on-policy distillation  (1,800 GPU-h)",
                color=GREEN, zorder=3)
    for bars in (b0, b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.0f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(bench, fontsize=9.5)
    ax.set_ylabel("Qwen3-8B score (%)")
    ax.set_ylim(38, 105)
    ax.set_title("Distillation beats RL on every benchmark -- at ~1/10 the GPU hours")
    ax.legend(frameon=False, fontsize=8.6, loc="upper right", ncol=1)
    ax.annotate(r"~10$\times$ cheaper", xy=(0.26, 75), xytext=(1.15, 90),
                fontsize=10, color=GREEN, weight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    save(fig, "distill_vs_rl.pdf")


def fig_headline_bench():
    # Table 11: Qwen3-235B-A22B (Thinking) vs top reasoning models.
    models = ["OpenAI-o1", "DeepSeek-R1", "Grok-3-Beta", "Gemini2.5-Pro",
              "Qwen3-235B-A22B"]
    aime24 = [74.3, 79.8, 83.9, 92.0, 85.7]
    aime25 = [79.2, 70.0, 77.3, 86.7, 81.5]
    colors = [GREY, GREY, GREY, VIOLET, ARM_RED]
    x = np.arange(2); w = 0.16
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    for i, (m, c) in enumerate(zip(models, colors)):
        off = (i - 2) * w
        vals = [aime24[i], aime25[i]]
        bars = ax.bar(x + off, vals, w, label=m, color=c, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=7.8,
                    weight="bold" if i == 4 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(["AIME'24", "AIME'25"])
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(60, 98)
    ax.set_title("Qwen3-235B-A22B (Thinking): competitive with the strongest reasoners")
    ax.legend(frameon=False, fontsize=8.6, ncol=3, loc="lower center")
    save(fig, "headline_bench.pdf")


if __name__ == "__main__":
    fig_model_lineup()
    fig_base_efficiency()
    fig_thinking_budget()
    fig_pipeline()
    fig_distill_vs_rl()
    fig_headline_bench()
    print("all Qwen3 figures done")
