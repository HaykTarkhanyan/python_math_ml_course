"""Figures for the DeepSeek-R1 deck.

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Number provenance: DeepSeek-AI, "DeepSeek-R1" (2025), arXiv:2501.12948.
  - AIME emergence endpoints (15.6 -> 77.9 pass@1, cons 86.7): Section 2.3, Fig 1a.
  - Response-length growth: Fig 1b (per-step values not tabulated -> curve is schematic).
  - Multi-stage pipeline: Fig 2 (schematic box flow).
  - Headline benchmarks: Table 8 (real numbers).
  - Distilled-model sizes: Table 15 (real numbers).
  - Distillation vs RL-from-scratch: Table 16 (real numbers).
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


# ---- Fig A: R1-Zero AIME accuracy emerges over RL steps -----------------
def fig_aime_emergence():
    # Real anchors (Fig 1a): pass@1 15.6 -> 77.9; cons@16 ends ~86.7.
    # Curve SHAPE between the anchors is schematic (per-step values not tabulated).
    steps = np.linspace(0, 10400, 300)
    s = steps / 10400.0
    shape = 1 - np.exp(-2.6 * s)          # saturating rise
    shape = shape / shape[-1]
    pass1 = 15.6 + (77.9 - 15.6) * shape
    cons = 24.0 + (86.7 - 24.0) * shape   # cons@16 starts higher, ends 86.7
    human = 40.0                          # AIME human-participant average (Fig 1a line)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(steps, cons, color=ARM_ORANGE, lw=2.6, label="cons@16 (majority vote)")
    ax.plot(steps, pass1, color=ARM_BLUE, lw=2.8, label="pass@1")
    ax.axhline(human, color=GREY, ls="--", lw=1.8)
    ax.text(300, human + 1.5, "human participant avg", color=GREY, fontsize=9.5)

    for y, txt, col in [(15.6, "15.6", ARM_BLUE), (77.9, "77.9", ARM_BLUE),
                        (86.7, "86.7", ARM_ORANGE)]:
        xx = 0 if y == 15.6 else 10400
        ax.scatter([xx], [y], color=col, s=34, zorder=5)
        ax.annotate(f"{txt}%", (xx, y), (xx + (300 if y == 15.6 else -1400), y + 3),
                    color=col, fontsize=10.5, weight="bold")

    ax.set_xlabel("RL training steps")
    ax.set_ylabel("AIME 2024 accuracy (%)")
    ax.set_title("R1-Zero: reasoning emerges from pure RL  (endpoints real, shape schematic)")
    ax.set_xlim(-200, 10800); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "aime_emergence.pdf")


# ---- Fig B: response length grows -> the model thinks longer ------------
def fig_response_length():
    # Fig 1b: average response length rises over training; a jump near the
    # 8.2k step (max gen length raised 32k -> 64k). Per-step values not
    # tabulated -> curve is schematic (labeled as such).
    steps = np.linspace(0, 10400, 300)
    base = 600 + 6800 * (1 - np.exp(-steps / 3800.0))
    jump = 3200 / (1 + np.exp(-(steps - 8200) / 250.0))   # step-up near 8.2k
    length = base + jump

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(steps, length, color=GREEN, lw=2.8)
    ax.fill_between(steps, 0, length, color=GREEN, alpha=0.08)
    ax.axvline(8200, color=ARM_ORANGE, ls=":", lw=1.8)
    ax.annotate("jump at ~8.2k step\n(max length 32k -> 64k)", (8200, 3200),
                (4200, 9600), color=ARM_ORANGE, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.text(300, 700, "starts short:\na few hundred tokens", color=GREY, fontsize=9.5,
            va="bottom")
    ax.annotate("hundreds -> thousands\nof 'thinking' tokens", (10400, length[-1]),
                (6400, 2500), color=GREEN, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("RL training steps")
    ax.set_ylabel("avg response length (tokens)")
    ax.set_title("The model spontaneously thinks longer  (schematic of the reported trend)")
    ax.set_xlim(-200, 10800); ax.set_ylim(0, 13000)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "response_length.pdf")


# ---- Fig C: the multi-stage R1 pipeline (box flow) ----------------------
def fig_r1_pipeline():
    fig, ax = plt.subplots(figsize=(12.2, 4.4))

    def box(cx, cy, w, h, title, sub, fc, ec, tc="white", fs=10.5):
        b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           fc=fc, ec=ec, lw=1.8, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy + 0.16, title, ha="center", va="center",
                color=tc, fontsize=fs, weight="bold", zorder=4)
        ax.text(cx, cy - 0.26, sub, ha="center", va="center",
                color=tc, fontsize=8.2, zorder=4)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=15, color="#444444", lw=1.8, zorder=2))

    y = 1.6
    xs = [0.9, 3.1, 5.5, 8.0, 10.4, 12.6]
    box(xs[0], y, 1.5, 1.3, "V3-Base", "pretrained\ncheckpoint", GREY, GREY)
    box(xs[1], y, 1.7, 1.3, "Cold-start\nSFT", "a little curated\nlong-CoT", ARM_BLUE, ARM_BLUE)
    box(xs[2], y, 1.9, 1.3, "Reasoning\nRL", "rule reward\n+ lang. consistency", ARM_ORANGE,
        ARM_ORANGE, tc="#333333")
    box(xs[3], y, 1.9, 1.3, "Reject-sample\n+ SFT", "~800k best outputs\n+ non-reasoning",
        ARM_BLUE, ARM_BLUE)
    box(xs[4], y, 1.7, 1.3, "Final RL", "all scenarios\nrule + pref. RM", ARM_ORANGE,
        ARM_ORANGE, tc="#333333")
    box(xs[5], y, 1.5, 1.3, "R1", "reasoning +\naligned", GREEN, GREEN)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 0.78, b - 0.78, y)

    ax.text(4.3, 0.55, "SFT stages (blue) teach readable CoT;  RL stages (orange) "
            "reward correctness, then broad alignment", ha="center", fontsize=9.5,
            color="#444444", style="italic")
    ax.set_xlim(-0.1, 13.5); ax.set_ylim(0.1, 2.5); ax.axis("off")
    save(fig, "r1_pipeline.pdf")


# ---- Fig D: headline benchmarks R1 vs o1-1217 vs V3 --------------------
def fig_headline_bench():
    # Table 8 (real numbers).
    bench = ["AIME 2024\n(pass@1)", "MATH-500\n(pass@1)", "GPQA Diamond\n(pass@1)",
             "Codeforces\n(percentile)"]
    v3 = [39.2, 90.2, 59.1, 58.7]
    o1 = [79.2, 96.4, 75.7, 96.6]
    r1 = [79.8, 97.3, 71.5, 96.3]
    x = np.arange(len(bench)); w = 0.27
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    b1 = ax.bar(x - w, v3, w, label="DeepSeek-V3", color=GREY, zorder=3)
    b2 = ax.bar(x, o1, w, label="OpenAI o1-1217", color=ARM_ORANGE, zorder=3)
    b3 = ax.bar(x + w, r1, w, label="DeepSeek-R1", color=ARM_BLUE, zorder=3)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", fontsize=8.4)
    ax.set_ylabel("score (%)")
    ax.set_title("DeepSeek-R1 vs OpenAI o1-1217: on par on math, ahead of the base V3")
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylim(0, 112); ax.legend(frameon=False, loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "headline_bench.pdf")


# ---- Fig E: distilled models across sizes (AIME) -----------------------
def fig_distill_sizes():
    # Table 15: AIME 2024 pass@1 for R1-distilled dense models.
    names = ["Qwen\n1.5B", "Llama\n8B", "Qwen\n7B", "Qwen\n14B", "Llama\n70B", "Qwen\n32B"]
    aime = [28.9, 50.4, 55.5, 69.7, 70.0, 72.6]
    cols = [ARM_BLUE if "Qwen" in n else VIOLET for n in names]
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    bars = ax.bar(range(len(names)), aime, color=cols, width=0.62, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{b.get_height():.1f}", ha="center", fontsize=9.5, weight="bold")
    # non-reasoning reference lines
    ax.axhline(16.0, color=ARM_RED, ls="--", lw=1.6)
    ax.text(5.55, 16.0 + 0.8, "Claude-3.5-Sonnet  16.0", color=ARM_RED, fontsize=8.6,
            ha="right")
    ax.axhline(9.3, color="#b06a00", ls=":", lw=1.6)
    ax.text(5.55, 9.3 - 3.0, "GPT-4o  9.3", color="#b06a00", fontsize=8.6, ha="right")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("AIME 2024 pass@1 (%)")
    ax.set_title("Distilled from R1 (SFT only): even 1.5B beats big non-reasoning models on math")
    ax.set_ylim(0, 82)
    ax.text(0.02, 0.93, "blue = Qwen base,  violet = Llama base", transform=ax.transAxes,
            fontsize=8.6, color="#555555")
    save(fig, "distill_sizes.pdf")


# ---- Fig F: distillation vs RL-from-scratch on a 32B model --------------
def fig_distill_vs_rl():
    # Table 16 (real numbers). The paper's key distillation result.
    bench = ["AIME 2024\n(pass@1)", "MATH-500\n(pass@1)", "GPQA Diamond\n(pass@1)",
             "LiveCodeBench\n(pass@1)"]
    rl = [47.0, 91.6, 55.0, 40.2]        # Qwen2.5-32B-Zero (RL from scratch, 10k+ steps)
    distill = [72.6, 94.3, 62.1, 57.2]   # DeepSeek-R1-Distill-Qwen-32B
    x = np.arange(len(bench)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    b1 = ax.bar(x - w / 2, rl, w, label="RL from scratch (Qwen-32B-Zero)", color=GREY,
                zorder=3)
    b2 = ax.bar(x + w / 2, distill, w, label="Distilled from R1 (Qwen-32B)", color=GREEN,
                zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", fontsize=9.0)
    for i in range(len(bench)):
        d = distill[i] - rl[i]
        ax.annotate(f"+{d:.1f}", (x[i], max(rl[i], distill[i]) + 5.0), ha="center",
                    color=GREEN, fontsize=10.0, weight="bold")
    ax.set_ylabel("score (%)")
    ax.set_title("Same 32B base: distilling R1 beats running RL directly on it")
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylim(0, 112); ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "distill_vs_rl.pdf")


if __name__ == "__main__":
    fig_aime_emergence()
    fig_response_length()
    fig_r1_pipeline()
    fig_headline_bench()
    fig_distill_sizes()
    fig_distill_vs_rl()
    print("all DeepSeek-R1 figures done")
