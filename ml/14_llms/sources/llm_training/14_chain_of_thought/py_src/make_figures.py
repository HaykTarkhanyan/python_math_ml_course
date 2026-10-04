"""Figures for the Chain-of-Thought & Process Supervision deck.

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Sources for the REAL numbers:
  - Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
    (2022), arXiv:2201.11903. Table 2 (PaLM standard vs CoT).
  - Wang et al., "Self-Consistency" (2022), arXiv:2203.11171 (PaLM 540B GSM8K
    56.5 -> 74.4 at 40 paths; curve shape schematic).
  - Lightman et al., "Let's Verify Step by Step" (2023), arXiv:2305.20050.
    Best-of-1860 MATH (ORM 72.4, PRM 78.2, majority 69.6); Table 1 OOD STEM.
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


# ---- Fig A: emergence -- CoT overtakes only at large scale --------------
def fig_emergence():
    # PaLM on GSM8K, Wei et al. Table 2 (exact).
    scale = np.array([8, 62, 540])          # PaLM size in billions
    standard = np.array([4.9, 9.6, 17.9])
    cot = np.array([4.1, 29.9, 56.9])

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.plot(scale, standard, "--o", color=GREY, lw=2.2, ms=7,
            label="standard prompting")
    ax.plot(scale, cot, "-o", color=ARM_BLUE, lw=2.6, ms=7,
            label="chain-of-thought prompting")
    ax.set_xscale("log")
    ax.set_xticks(scale)
    ax.set_xticklabels(["8B", "62B", "540B"])
    ax.set_xlabel("PaLM model scale (parameters, log)")
    ax.set_ylabel("GSM8K solve rate (%)")
    ax.set_ylim(0, 65)

    # value labels
    for xi, s, c in zip(scale, standard, cot):
        ax.annotate(f"{s:.1f}", (xi, s), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, color=GREY)
        ax.annotate(f"{c:.1f}", (xi, c), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9.5, color=ARM_BLUE,
                    weight="bold")

    # small model: CoT actually HURTS
    ax.annotate("small model:\nCoT below standard\n(4.1 < 4.9)",
                (8, 4.5), (11, 22), fontsize=9, color=ARM_RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.4))
    # large model: the jump
    ax.annotate("+39.0", (540, 56.9), textcoords="offset points",
                xytext=(-42, -4), fontsize=12, color=GREEN, weight="bold")
    ax.annotate("", (540, 56.9), (540, 17.9),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))

    ax.set_title("Chain-of-thought is an EMERGENT ability of scale (PaLM, GSM8K)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "emergence.pdf")


# ---- Fig B: CoT gains across arithmetic benchmarks ----------------------
def fig_benchmark_gains():
    # PaLM 540B, Wei et al. Table 2 (exact).
    bench = ["GSM8K", "SVAMP", "ASDiv", "AQuA", "MAWPS"]
    standard = [17.9, 69.4, 72.1, 25.2, 79.2]
    cot = [56.9, 79.0, 73.9, 35.8, 93.3]
    x = np.arange(len(bench)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    b1 = ax.bar(x - w / 2, standard, w, label="standard", color=GREY, zorder=3)
    b2 = ax.bar(x + w / 2, cot, w, label="chain-of-thought", color=ARM_BLUE,
                zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)
    for i in range(len(bench)):
        d = cot[i] - standard[i]
        ax.annotate(f"+{d:.1f}", (x[i], max(cot[i], standard[i]) + 5.5),
                    ha="center", color=GREEN, fontsize=10, weight="bold")
    ax.set_ylabel("solve rate (%)")
    ax.set_title("PaLM 540B: standard vs chain-of-thought (arithmetic)")
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylim(0, 108); ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.0, 101, "biggest gain on the hardest set", color=ARM_ORANGE,
            fontsize=9, ha="left", style="italic")
    save(fig, "benchmark_gains.pdf")


# ---- Fig C: self-consistency -- sample many chains, majority vote -------
def fig_self_consistency():
    # PaLM 540B on GSM8K: greedy CoT 56.5 -> self-consistency 74.4 at 40 paths
    # (Wang et al. 2022). Endpoints REAL; saturating shape is schematic.
    n = np.arange(1, 41)
    lo, hi = 56.5, 74.4
    acc = hi - (hi - lo) * np.exp(-(n - 1) / 7.5)
    acc[0] = lo
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.axhline(lo, color=GREY, ls="--", lw=1.8, label="greedy CoT (1 chain)")
    ax.plot(n, acc, "-", color=GREEN, lw=2.8,
            label="self-consistency (majority vote)")
    ax.scatter([1, 40], [lo, hi], color=[GREY, GREEN], zorder=5, s=45)
    ax.annotate(f"{lo:.1f}", (1, lo), textcoords="offset points",
                xytext=(6, -14), fontsize=10, color=GREY, weight="bold")
    ax.annotate(f"{hi:.1f}", (40, hi), textcoords="offset points",
                xytext=(-6, 6), ha="right", fontsize=11, color=GREEN,
                weight="bold")
    ax.annotate("+17.9 from voting alone\n(no extra training)", (28, 68),
                fontsize=9.5, color=GREEN, ha="center")
    ax.set_xlabel("number of sampled reasoning chains")
    ax.set_ylabel("GSM8K solve rate (%)")
    ax.set_ylim(52, 78)
    ax.set_title("Self-consistency: sample many chains, take the majority answer")
    ax.legend(frameon=False, loc="center right", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.98, 0.03, "endpoints real; saturating shape schematic",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=GREY, style="italic")
    save(fig, "self_consistency.pdf")


# ---- Fig D: ORM vs PRM vs majority -- best-of-N on MATH (money shot) -----
def fig_orm_prm_bestofn():
    # Lightman et al. Fig 3: best-of-N on MATH subset. Endpoints at N=1860 are
    # EXACT (PRM 78.2, ORM 72.4, majority 69.6); intermediate shape schematic.
    N = np.array([4, 16, 64, 256, 1024, 1860])
    x = np.log10(N)
    def curve(end, start, k):
        y = end - (end - start) * np.exp(-(x - x[0]) / k)
        return y
    prm = curve(78.2, 66.0, 1.15)
    orm = curve(72.4, 65.0, 0.85)
    maj = curve(69.6, 64.5, 0.80)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(x, prm, "-o", color=GREEN, lw=2.8, ms=5, label="PRM (process)")
    ax.plot(x, orm, "-o", color=ARM_BLUE, lw=2.4, ms=5, label="ORM (outcome)")
    ax.plot(x, maj, "--o", color=GREY, lw=2.2, ms=5, label="majority vote")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in N])
    ax.set_xlabel("N = solutions per problem (best-of-N)")
    ax.set_ylabel("MATH problems solved (%)")
    ax.set_ylim(62, 80)
    # exact endpoint labels
    for y, c in [(78.2, GREEN), (72.4, ARM_BLUE), (69.6, GREY)]:
        ax.annotate(f"{y:.1f}", (x[-1], y), textcoords="offset points",
                    xytext=(8, -2), fontsize=10.5, color=c, weight="bold")
    ax.annotate("gap widens with N", (x[3], 74.5), (x[1], 77),
                fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.set_title("Let's Verify: PRM beats ORM and majority vote on MATH")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.45)
    save(fig, "orm_prm_bestofn.pdf")


# ---- Fig E: step-scored reasoning chain (illustrative) ------------------
def fig_step_scoring():
    fig, ax = plt.subplots(figsize=(10.6, 4.2))

    def solution(x0, title, tcol, marks, verdict, vcol):
        w, h = 3.9, 0.5
        ax.text(x0 + w / 2, 3.75, title, ha="center", fontsize=11.5,
                weight="bold", color=tcol)
        for k, ok in enumerate(marks):
            y = 3.15 - k * 0.62
            col = GREEN if ok else ARM_RED
            ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         fc=(col + "22") if False else "#f6f8fb",
                         ec=col, lw=1.8, zorder=2))
            ax.text(x0 + 0.28, y, f"step {k + 1}", va="center", fontsize=10,
                    color="#333333")
            sym = "✓" if ok else "✗"
            ax.text(x0 + w - 0.32, y, sym, va="center", ha="center",
                    fontsize=15, color=col, weight="bold", zorder=3)
        ax.add_patch(FancyBboxPatch((x0, 0.15), w, 0.5,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc=vcol + "20", ec=vcol, lw=1.6, zorder=2))
        ax.text(x0 + w / 2, 0.4, verdict, ha="center", va="center",
                fontsize=10, color=vcol, weight="bold")

    # left: genuinely correct chain
    solution(0.3, "Correct reasoning", GREEN,
             [True, True, True, True], "final answer right", GREEN)
    # right: wrong step but a lucky-correct final answer
    solution(6.0, "Flawed reasoning", ARM_RED,
             [True, True, False, True], "final answer right (by luck)", ARM_ORANGE)

    ax.text(5.15, 2.1, "ORM sees\nonly this\n(final token)", ha="center",
            va="center", fontsize=8.5, color=ARM_BLUE, style="italic")
    ax.annotate("", (9.9, 0.9), (9.9, 1.55),
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.4))
    ax.text(9.95, 1.9, "PRM flags\nthe bad step", ha="left", va="center",
            fontsize=8.5, color=ARM_RED, style="italic")

    ax.set_title("Same final answer, different reasoning: outcome scoring "
                 "rewards luck, process scoring does not", fontsize=11)
    ax.text(0.99, 0.005, "illustrative (schematic)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GREY, style="italic")
    ax.set_xlim(-0.1, 11.6); ax.set_ylim(-0.05, 4.1); ax.axis("off")
    save(fig, "step_scoring.pdf")


# ---- Fig F: OOD generalization on fresh STEM exams ----------------------
def fig_ood_generalization():
    # Lightman et al. Table 1 (exact), best-of-100.
    subj = ["AP\nCalculus", "AP\nChemistry", "AP\nPhysics", "AMC\n10/12",
            "Aggregate"]
    orm = [68.9, 68.9, 77.8, 49.1, 63.8]
    prm = [86.7, 80.0, 86.7, 53.2, 72.9]
    maj = [80.0, 71.7, 82.2, 32.8, 61.3]
    x = np.arange(len(subj)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    bars = [
        (ax.bar(x - w, orm, w, label="ORM (outcome)", color=ARM_BLUE, zorder=3)),
        (ax.bar(x, prm, w, label="PRM (process)", color=GREEN, zorder=3)),
        (ax.bar(x + w, maj, w, label="majority vote", color=GREY, zorder=3)),
    ]
    for bs in bars:
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_ylabel("solved (%), best-of-100")
    ax.set_title("Fresh STEM exams (out-of-distribution): PRM generalizes best")
    ax.set_xticks(x); ax.set_xticklabels(subj)
    ax.set_ylim(0, 100); ax.legend(frameon=False, loc="lower left", ncol=3,
                                    fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "ood_generalization.pdf")


if __name__ == "__main__":
    fig_emergence()
    fig_benchmark_gains()
    fig_self_consistency()
    fig_orm_prm_bestofn()
    fig_step_scoring()
    fig_ood_generalization()
    print("all CoT figures done")
