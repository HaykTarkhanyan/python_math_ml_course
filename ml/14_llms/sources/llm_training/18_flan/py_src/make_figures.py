"""Figures for the FLAN deck ([LLM-18] Instruction Tuning at Scale).

Run with the project venv (or any env with matplotlib + numpy):
    ../../../../ma/Scripts/python.exe py_src/make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Numbers are from Wei et al. (2021), "Finetuned Language Models Are Zero-Shot
Learners" (arXiv:2109.01652). Where a figure has no per-point numbers in the
paper (scale_emergence, Fig. 7), values are read/approximated off the plot and
labelled as such on the slide.
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

SEED = 509
np.random.seed(SEED)

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


# ---- Fig A (SCHEMATIC): instruction tuning pipeline --------------------
def fig_instruction_tuning():
    fig, ax = plt.subplots(figsize=(11.0, 3.6))

    _box(ax, (1.4, 2.0), 2.2, 1.15,
         "Pretrain\nLaMDA-PT 137B\n(next-token)", ARM_BLUE, ARM_BLUE, fs=11)
    _box(ax, (5.0, 2.0), 2.6, 1.15,
         "Instruction-tune\non 11 task clusters\n(instruction, input)$\\to$output",
         GREEN, GREEN, fs=10.5)
    _box(ax, (8.7, 2.0), 2.3, 1.15,
         "Zero-shot eval\non the HELD-OUT\ncluster (NLI)", ARM_ORANGE, ARM_ORANGE,
         tc="#333333", fs=10.5)

    _arrow(ax, (2.5, 2.0), (3.7, 2.0), lw=2.0)
    _arrow(ax, (6.3, 2.0), (7.55, 2.0), lw=2.0)

    ax.text(5.0, 0.75, "NLI never seen during tuning $\\Rightarrow$ a genuinely unseen task type",
            ha="center", color=ARM_RED, fontsize=10, style="italic")
    ax.text(1.4, 3.05, "generic language model", ha="center", color=GREY, fontsize=9)
    ax.text(5.0, 3.05, "learns the FORMAT: follow instructions", ha="center",
            color=GREY, fontsize=9)
    ax.text(8.7, 3.05, "generalises to a new instruction", ha="center",
            color=GREY, fontsize=9)

    ax.set_xlim(0, 10.2); ax.set_ylim(0.3, 3.4); ax.axis("off")
    fig.suptitle("Instruction tuning sits between pretraining and use",
                 fontsize=12.5, y=1.0, color="#333333")
    save(fig, "instruction_tuning.pdf")


# ---- Fig B (REAL): 62 datasets in 12 clusters --------------------------
def fig_task_clusters():
    # Cluster sizes from Figure 3 (sum = 62).
    clusters = [
        ("Summarization", 11),
        ("Translation", 8),
        ("Natural language inference", 7),
        ("Misc.", 7),
        ("Reading comprehension", 5),
        ("Commonsense", 4),
        ("Sentiment", 4),
        ("Struct to text", 4),
        ("Paraphrase", 4),
        ("Closed-book QA", 3),
        ("Coreference", 3),
        ("Read. comp. w/ commonsense", 2),
    ]
    names = [c[0] for c in clusters]
    sizes = [c[1] for c in clusters]
    y = np.arange(len(clusters))[::-1]  # first cluster on top

    # highlight NLI as the held-out cluster
    held = "Natural language inference"
    cols = [ARM_RED if n == held else ARM_BLUE for n in names]

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    bars = ax.barh(y, sizes, color=cols, height=0.68, zorder=3)
    ax.bar_label(bars, labels=[f"{s}" for s in sizes], padding=3, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("number of datasets in the cluster")
    ax.set_xlim(0, 13)
    ax.set_title("62 datasets grouped into 12 task clusters  (~10 instruction templates each)")
    ax.annotate("held out for evaluation\n(none of its datasets seen in tuning)",
                xy=(7, y[names.index(held)]), xytext=(9.2, y[names.index(held)] + 1.2),
                fontsize=9, color=ARM_RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.text(0.99, -0.14, "sizes from Fig. 3 (Wei et al., 2021); total = 62",
            transform=ax.transAxes, ha="right", fontsize=8.2, color=GREY)
    save(fig, "task_clusters.pdf")


# ---- Fig C (REAL): FLAN vs GPT-3, three unseen task types --------------
def fig_flan_vs_gpt3():
    # Figure 1 (bottom) averaged numbers.
    tasks = ["Natural language\ninference", "Reading\ncomprehension", "Closed-book\nQA"]
    gpt3_zero = [42.9, 63.7, 49.8]
    gpt3_few = [53.2, 72.6, 55.7]
    flan_zero = [56.2, 77.4, 56.6]
    x = np.arange(len(tasks)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    b1 = ax.bar(x - w, gpt3_zero, w, label="GPT-3 175B  zero-shot", color=GREY, zorder=3)
    b2 = ax.bar(x, gpt3_few, w, label="GPT-3 175B  few-shot", color=ARM_ORANGE, zorder=3)
    b3 = ax.bar(x + w, flan_zero, w, label="FLAN 137B  zero-shot", color=ARM_BLUE, zorder=3)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("performance on unseen task type")
    ax.set_ylim(0, 95)
    ax.set_title("Zero-shot FLAN beats zero-shot GPT-3 - and its few-shot too (Fig. 1)")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.20, "Headline: FLAN > zero-shot GPT-3 on 20 of 25 datasets, "
            "and > few-shot GPT-3 on 10.",
            transform=ax.transAxes, ha="center", fontsize=9, color=ARM_RED)
    save(fig, "flan_vs_gpt3.pdf")


# ---- Fig D (REAL): performance vs number of task clusters --------------
def fig_num_clusters():
    n_clusters = np.array([1, 2, 3, 4, 5, 6, 7])
    n_datasets = [11, 20, 26, 30, 34, 37, 39]
    avg = np.array([49.9, 55.0, 59.3, 59.2, 60.8, 61.9, 63.5])  # Fig. 6 "Average"
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.plot(n_clusters, avg, "-o", color=ARM_BLUE, lw=2.4, ms=8, zorder=3,
            label="held-out avg (NLI, closed-book QA, commonsense)")
    for xc, v, nd in zip(n_clusters, avg, n_datasets):
        ax.annotate(f"{v:.1f}", (xc, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=ARM_BLUE)
        ax.annotate(f"({nd})", (xc, v), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8, color=GREY)
    ax.annotate("still rising at 7 clusters\n(no sign of saturating)",
                xy=(7, 63.5), xytext=(4.7, 57.0), fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("number of task clusters used for instruction tuning  "
                  "(grey = # datasets)")
    ax.set_ylabel("zero-shot accuracy on\nheld-out clusters (%)")
    ax.set_xticks(n_clusters)
    ax.set_ylim(46, 68)
    ax.set_title("More instruction clusters $\\to$ better zero-shot on unseen tasks (Fig. 6)")
    save(fig, "num_clusters.pdf")


# ---- Fig E (SIGNATURE, reconstructed): instruction tuning vs scale -----
def fig_scale_emergence():
    # Fig. 7 has no per-point numbers; values approximated off the plot.
    # Qualitative facts (locked): untuned rises with scale; instruction tuning
    # HURTS at 422M/2B/8B, HELPS at 68B/137B; crossover between 8B and 68B.
    sizes = ["422M", "2B", "8B", "68B", "137B"]
    x = np.arange(len(sizes))
    untuned = np.array([40.0, 42.0, 47.0, 54.0, 58.0])
    tuned = np.array([34.0, 38.0, 45.0, 58.5, 64.0])

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.plot(x, untuned, "-o", color=GREY, lw=2.2, ms=8, zorder=3,
            label="Untuned model (LaMDA-PT)")
    ax.plot(x, tuned, "-o", color=ARM_BLUE, lw=2.6, ms=8, zorder=3,
            label="Instruction tuning (FLAN)")

    # shade where instruction tuning HURTS
    ax.axvspan(-0.4, 2.5, color=ARM_RED, alpha=0.06, zorder=0)
    ax.axvspan(2.5, 4.4, color=GREEN, alpha=0.06, zorder=0)
    ax.text(1.0, 31.5, "instruction tuning HURTS", ha="center", color=ARM_RED,
            fontsize=10.5, weight="bold")
    ax.text(3.5, 31.5, "instruction tuning HELPS", ha="center", color=GREEN,
            fontsize=10.5, weight="bold")
    ax.axvline(2.5, color="#999999", ls="--", lw=1.4)
    ax.text(2.5, 66.5, "crossover\n(8B - 68B)", ha="center", color="#555555",
            fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("avg zero-shot accuracy\non 13 held-out tasks (%)")
    ax.set_ylim(28, 70)
    ax.set_xlim(-0.4, 4.4)
    ax.set_title("Instruction tuning is an emergent ability of scale (Fig. 7)")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.text(0.99, -0.16, "values approximate, read off Fig. 7 (paper reports no per-point numbers)",
            transform=ax.transAxes, ha="right", fontsize=8.0, color=GREY)
    save(fig, "scale_emergence.pdf")


# ---- Fig F (REAL): role of instructions ablation -----------------------
def fig_instructions_ablation():
    # Figure 8, 4-task-cluster average zero-shot performance.
    labels = [
        "FT: instruction\nEval: instruction\n(FLAN)",
        "FT: dataset name\nEval: dataset name",
        "FT: dataset name\nEval: instruction",
        "FT: no template\nEval: instruction",
    ]
    vals = [55.2, 47.0, 46.6, 37.3]
    cols = [ARM_BLUE, GREY, GREY, ARM_RED]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    bars = ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=11, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("zero-shot performance (4 task-cluster avg.)")
    ax.set_xlim(0, 62)
    ax.set_title("Strip the instructions and the gain collapses (Fig. 8)")
    ax.annotate("", xy=(55.2, y[0]), xytext=(37.3, y[3]),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.text(50.0, y[2] + 0.15, "$-18$ points\nwithout templates", color=GREEN,
            fontsize=9.5, ha="center")
    save(fig, "instructions_ablation.pdf")


if __name__ == "__main__":
    fig_instruction_tuning()
    fig_task_clusters()
    fig_flan_vs_gpt3()
    fig_num_clusters()
    fig_scale_emergence()
    fig_instructions_ablation()
    print("all FLAN figures done")
