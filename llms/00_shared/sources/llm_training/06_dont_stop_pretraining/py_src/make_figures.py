"""Figures for the Don't Stop Pretraining deck. Run: python3 make_figures.py -> ../fig/*.pdf

All numbers are REAL, taken from Gururangan et al. (2020), arXiv:2004.10964:
  vocab_overlap  -> Figure 2 (vocabulary overlap %)
  dapt_gains     -> Table 5 (ROBERTA vs DAPT, test F1)
  irrelevant     -> Table 3 (ROBERTA / DAPT / not-DAPT, i.e. irrelevant domain)
  tapt_combined  -> Table 5 (ROBERTA / DAPT / TAPT / DAPT+TAPT)
  compute        -> Table 9 (compute + storage for RCT-500)
  augment_tapt   -> Table 7 (Curated-TAPT)
"""
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


def fig_vocab_overlap():
    # Figure 2 (exact). Rows/cols: PT, News, Reviews, BioMed, CS.
    labels = ["PT", "News", "Reviews", "BioMed", "CS"]
    M = np.array([
        [100.0, 54.1, 34.5, 27.3, 19.2],
        [54.1, 100.0, 40.0, 24.9, 17.3],
        [34.5, 40.0, 100.0, 18.3, 12.7],
        [27.3, 24.9, 18.3, 100.0, 21.4],
        [19.2, 17.3, 12.7, 21.4, 100.0],
    ])
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="white" if M[i, j] > 55 else "#222222",
                    fontsize=11, weight="bold" if i == 0 or j == 0 else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    # Highlight the PT column: overlap with ROBERTA's pretraining corpus.
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 5, fill=False,
                               edgecolor=ARM_RED, lw=2.6))
    ax.set_title("Vocabulary overlap (%) with each domain")
    ax.text(0, 5.15, "far from PT  ->  more to gain from DAPT",
            color=ARM_RED, fontsize=10, ha="left", style="italic")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overlap (%)")
    save(fig, "vocab_overlap.pdf")


def fig_dapt_gains():
    # Table 5, ROBERTA vs DAPT (test F1). Ordered by domain.
    tasks = ["CHEMPROT", "RCT", "ACL-ARC", "SCIERC", "HYP.", "AGNEWS", "HELPFUL.", "IMDB"]
    doms = ["BioMed", "BioMed", "CS", "CS", "News", "News", "Reviews", "Reviews"]
    roberta = [81.9, 87.2, 63.0, 77.3, 86.6, 93.9, 65.1, 95.0]
    dapt = [84.2, 87.6, 75.4, 80.8, 88.2, 93.9, 66.5, 95.4]
    x = np.arange(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    b1 = ax.bar(x - w / 2, roberta, w, color=GREY, zorder=3, label="RoBERTa (baseline)")
    b2 = ax.bar(x + w / 2, dapt, w, color=ARM_BLUE, zorder=3, label="+ DAPT")
    ax.bar_label(b1, fmt="%.1f", fontsize=7.5, padding=1)
    ax.bar_label(b2, fmt="%.1f", fontsize=7.5, padding=1, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({d})" for t, d in zip(tasks, doms)], fontsize=8.5)
    ax.set_ylabel("test F1"); ax.set_ylim(55, 100)
    ax.set_title("DAPT helps on all 8 tasks - most where the domain is far from PT")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    # Flag the biggest gain (ACL-ARC, a far CS domain).
    ax.annotate("+12.4", xy=(2 + w / 2, 75.4), xytext=(2 + w / 2, 90),
                ha="center", color=ARM_BLUE, fontsize=10, weight="bold",
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE, lw=1.5))
    save(fig, "dapt_gains.pdf")


def fig_irrelevant():
    # Table 3: ROBERTA / DAPT / not-DAPT (adapt to an irrelevant domain).
    tasks = ["CHEMPROT\n(BioMed)", "SCIERC\n(CS)", "ACL-ARC\n(CS)", "HYP.\n(News)"]
    roberta = [81.9, 77.3, 63.0, 86.6]
    dapt = [84.2, 80.8, 75.4, 88.2]
    notdapt = [79.4, 79.2, 66.4, 76.4]
    x = np.arange(len(tasks)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    b1 = ax.bar(x - w, roberta, w, color=GREY, zorder=3, label="RoBERTa (baseline)")
    b2 = ax.bar(x, dapt, w, color=GREEN, zorder=3, label="relevant DAPT")
    b3 = ax.bar(x + w, notdapt, w, color=ARM_RED, zorder=3, label="irrelevant DAPT")
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.1f", fontsize=8, padding=1)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("test F1"); ax.set_ylim(55, 95)
    ax.set_title("Adapting to the WRONG domain often falls below the baseline")
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)
    save(fig, "irrelevant.pdf")


def fig_tapt_combined():
    # Table 5: ROBERTA / DAPT / TAPT / DAPT+TAPT (test F1), readable subset.
    tasks = ["ACL-ARC\n(CS)", "SCIERC\n(CS)", "HYP.\n(News)", "HELPFUL.\n(Rev.)"]
    roberta = [63.0, 77.3, 86.6, 65.1]
    dapt = [75.4, 80.8, 88.2, 66.5]
    tapt = [67.4, 79.3, 90.4, 68.5]
    both = [75.6, 81.3, 90.0, 68.7]
    x = np.arange(len(tasks)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    series = [("RoBERTa", roberta, GREY), ("DAPT", dapt, ARM_BLUE),
              ("TAPT", tapt, ARM_ORANGE), ("DAPT+TAPT", both, GREEN)]
    for k, (lab, vals, col) in enumerate(series):
        b = ax.bar(x + (k - 1.5) * w, vals, w, color=col, zorder=3, label=lab)
        ax.bar_label(b, fmt="%.1f", fontsize=7, padding=1, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("test F1"); ax.set_ylim(58, 100)
    ax.set_title("TAPT alone is strong (tiny data); DAPT + TAPT is best")
    ax.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)
    save(fig, "tapt_combined.pdf")


def fig_compute():
    # Table 9 (RCT-500): storage vs F1 for TAPT / Curated-TAPT / DAPT.
    names = ["TAPT", "Curated-TAPT", "DAPT"]
    storage_mb = [80 / 1024.0, 27.0, 47.0 * 1024.0]   # 80KB, 27MB, 47GB in MB
    stor_txt = ["80 KB", "27 MB", "47 GB"]
    steps = ["0.2K steps", "8.8K steps", "12.5K steps"]
    f1 = [79.8, 83.4, 82.5]
    colors = [GREEN, ARM_ORANGE, ARM_BLUE]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b = ax.bar(x, storage_mb, width=0.55, color=colors, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 3e5)
    for xi, (s, st, sp, f) in enumerate(zip(storage_mb, stor_txt, steps, f1)):
        ax.text(xi, s * 1.7, st, ha="center", fontsize=10, weight="bold")
        ax.text(xi, s * 1.7 * 3.2, f"F1 {f}", ha="center", fontsize=9.5,
                color="#222222")
        ax.text(xi, 0.045, sp, ha="center", fontsize=8.5, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("pretraining corpus size (MB, log scale)")
    ax.set_title("TAPT: ~60x fewer steps, ~600,000x less data, F1 within ~3 pts of DAPT")
    save(fig, "compute.pdf")


def fig_augment_tapt():
    # Table 7: Curated-TAPT vs TAPT / DAPT+TAPT (test F1).
    tasks = ["RCT-500", "HYP.", "IMDB"]
    tapt = [79.8, 90.4, 95.5]
    dapt_tapt = [83.0, 90.0, 95.6]
    curated = [83.4, 89.9, 95.7]
    dapt_cur = [83.8, 92.1, 95.8]
    x = np.arange(len(tasks)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    series = [("TAPT", tapt, ARM_ORANGE), ("DAPT+TAPT", dapt_tapt, GREEN),
              ("Curated-TAPT", curated, VIOLET), ("DAPT+Curated-TAPT", dapt_cur, ARM_BLUE)]
    for k, (lab, vals, col) in enumerate(series):
        b = ax.bar(x + (k - 1.5) * w, vals, w, color=col, zorder=3, label=lab)
        ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=1, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("test F1"); ax.set_ylim(75, 100)
    ax.set_title("More curated in-domain text for TAPT helps further")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.12))
    save(fig, "augment_tapt.pdf")


if __name__ == "__main__":
    fig_vocab_overlap(); fig_dapt_gains(); fig_irrelevant()
    fig_tapt_combined(); fig_compute(); fig_augment_tapt()
    print("all Don't-Stop-Pretraining figures done")
