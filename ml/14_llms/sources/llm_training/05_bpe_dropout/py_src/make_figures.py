"""Figures for the BPE-Dropout deck. Run: python make_figures.py -> ../fig/*.pdf"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ARM_RED = "#C81E28"; ARM_BLUE = "#1E46A0"; ARM_ORANGE = "#E6A01E"
GREEN = "#008C46"; GREY = "#8a8a8a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.titlesize": 13, "figure.dpi": 140})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig"); os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", os.path.normpath(p))


def fig_seg_example():
    # SCHEMATIC (illustrative splits) reproducing Figure 1's message: standard BPE
    # emits ONE deterministic segmentation of a word, while BPE-dropout samples many
    # different segmentations from the SAME merge table. Word: "unrelated" (paper's ex.).
    word = "unrelated"
    n = len(word)
    rows = [
        ("standard BPE  (p=0)", ["un", "related"], ARM_BLUE, True),
        ("BPE-dropout sample", ["un", "rel", "ated"], ARM_ORANGE, False),
        ("BPE-dropout sample", ["u", "n", "related"], ARM_ORANGE, False),
        ("BPE-dropout sample", ["unre", "lat", "ed"], ARM_ORANGE, False),
        ("BPE-dropout sample", ["u", "nre", "lated"], ARM_ORANGE, False),
    ]
    cw = 1.0           # width of one character cell
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    top = len(rows)
    for ri, (label, toks, color, emph) in enumerate(rows):
        y = top - ri - 1
        x = 0
        for tk in toks:
            w = len(tk) * cw
            box = FancyBboxPatch((x + 0.06, y + 0.12), w - 0.12, 0.66,
                                 boxstyle="round,pad=0.02,rounding_size=0.10",
                                 linewidth=2.0 if emph else 1.6,
                                 edgecolor=color,
                                 facecolor=color + "22" if not emph else color + "33")
            ax.add_patch(box)
            ax.text(x + w / 2, y + 0.45, tk, ha="center", va="center",
                    fontsize=15, color="#222", family="monospace",
                    weight="bold" if emph else "normal")
            x += w
        ax.text(-0.3, y + 0.45, label, ha="right", va="center", fontsize=11,
                color=color, weight="bold" if emph else "normal")
    # thin guide lines at every character boundary to show tokens share one grid
    for c in range(n + 1):
        ax.plot([c, c], [0.05, top - 0.02], color="#dddddd", lw=0.7, zorder=0)
    ax.set_xlim(-3.6, n + 0.2)
    ax.set_ylim(-0.15, top + 0.15)
    ax.axis("off")
    ax.set_title("One word, one BPE split vs many BPE-dropout splits "
                 "(same merge table)  — illustrative", fontsize=12.5)
    save(fig, "seg_example.pdf")


def fig_bleu_gains():
    # Table 2 (real). Bars = BPE-dropout gain over BPE; grey diamonds = Kudo(2018) gain
    # over BPE. Shows gains on every pair and that BPE-dropout usually matches/beats Kudo.
    pairs = ["En-Vi", "Vi-En", "En-Zh", "Zh-En", "En-Fr", "Fr-En",
             "En-Ar", "Ar-En", "En-De", "De-En", "En-Ja", "Ja-En"]
    bpe = np.array([31.78, 30.83, 20.48, 19.72, 39.37, 38.18,
                    13.89, 31.90, 27.41, 32.69, 54.51, 30.77])
    kudo = np.array([32.43, 32.36, 23.01, 21.10, 39.45, 38.88,
                     14.43, 32.80, 27.82, 33.65, 55.46, 31.23])
    drop = np.array([33.27, 32.99, 22.84, 21.45, 40.02, 39.39,
                     15.05, 33.72, 28.01, 34.19, 55.00, 31.29])
    gain_d = drop - bpe
    gain_k = kudo - bpe
    order = np.argsort(gain_d)          # ascending, so biggest ends at top of barh
    pairs = [pairs[i] for i in order]
    gain_d = gain_d[order]; gain_k = gain_k[order]
    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.barh(y, gain_d, color=ARM_BLUE, height=0.62, zorder=3,
            label="BPE-dropout − BPE")
    ax.scatter(gain_k, y, color=GREY, marker="D", s=34, zorder=4,
               label="Kudo (2018) − BPE")
    ax.axvline(0, color="#999999", lw=1.0)
    for yi, g in zip(y, gain_d):
        ax.text(g + 0.04, yi, f"+{g:.2f}", va="center", fontsize=9.5,
                color=ARM_BLUE)
    ax.set_yticks(y); ax.set_yticklabels(pairs, fontsize=10)
    ax.set_xlabel("BLEU gain over standard BPE")
    ax.set_xlim(-0.15, 2.75)
    ax.set_title("BPE-dropout improves BLEU on every pair (up to +2.36)")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    save(fig, "bleu_gains.pdf")


def fig_dropout_p():
    # Effect of dropout rate p. WMT14 En-Fr, 500k. SCHEMATIC curve shape; the p=0 and
    # p~0.1 anchors are from Table 3 (BPE 29.28; best BPE-dropout ~30.1). High p pushes
    # training toward char-level and the model collapses ("unable to translate", Fig 2).
    p = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    bleu = np.array([29.28, 29.9, 30.12, 29.7, 29.0, 27.8, 24.5, 18.0, 11.0])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(p, bleu, "-o", color=ARM_BLUE, lw=2.4, ms=5, zorder=3)
    peak = 2
    ax.scatter([p[peak]], [bleu[peak]], color=GREEN, s=90, zorder=5)
    ax.annotate("best: p = 0.1", (p[peak], bleu[peak]), (0.22, 30.6),
                fontsize=11, color=GREEN, weight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.axvspan(0.55, 1.02, color=ARM_RED, alpha=0.08)
    ax.text(0.78, 22.5, "close to char-level:\nmodel collapses", ha="center",
            fontsize=10, color=ARM_RED)
    ax.annotate("p = 0: standard BPE", (0.0, 29.28), (0.05, 25.5),
                fontsize=10, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xlabel("dropout probability $p$ (chance a merge is skipped)")
    ax.set_ylabel("BLEU")
    ax.set_title("A moderate $p$ is best; too high shatters words into characters "
                 "(shape schematic)", fontsize=12)
    ax.set_ylim(8, 32)
    ax.set_xlim(-0.03, 1.03)
    save(fig, "dropout_p.pdf")


def fig_robustness():
    # Table 5 (real): BLEU lost when the source is corrupted with 10% misspellings.
    # BPE-dropout was never trained on noise yet degrades markedly less.
    sets = ["En-De", "De-En", "En-Fr (4m)", "En-Fr (16m)"]
    bpe_drop = np.array([27.41 - 24.45, 32.69 - 29.71, 33.38 - 30.30, 34.37 - 31.23])
    bd_drop = np.array([28.01 - 26.03, 34.19 - 32.03, 33.85 - 32.13, 34.82 - 32.94])
    x = np.arange(len(sets)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    b1 = ax.bar(x - w / 2, bpe_drop, w, color=ARM_RED, zorder=3, label="BPE")
    b2 = ax.bar(x + w / 2, bd_drop, w, color=ARM_BLUE, zorder=3, label="BPE-dropout")
    ax.bar_label(b1, fmt="-%.2f", padding=2, fontsize=9, color=ARM_RED)
    ax.bar_label(b2, fmt="-%.2f", padding=2, fontsize=9, color=ARM_BLUE)
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_ylabel("BLEU lost to 10% misspellings")
    ax.set_title("Robustness to noisy input: BPE-dropout degrades far less "
                 "(lower is better)", fontsize=12)
    ax.set_ylim(0, 3.7)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "robustness.pdf")


def fig_corpus_size():
    # Table 3 (real): WMT14 En-Fr random subsets. BPE vs best BPE-dropout config
    # (both sides for small/medium, source-only for 4m/16m). Gain shrinks as data grows.
    sizes = np.array([0.25, 0.5, 1.0, 4.0, 16.0])            # millions of sentence pairs
    bpe = np.array([26.94, 29.28, 30.53, 33.38, 34.37])
    drop = np.array([28.40, 30.12, 31.23, 33.89, 34.82])    # best BPE-dropout per size
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.fill_between(sizes, bpe, drop, color=ARM_ORANGE, alpha=0.18, zorder=1)
    ax.plot(sizes, bpe, "-o", color=GREY, lw=2.2, ms=6, label="BPE")
    ax.plot(sizes, drop, "-o", color=ARM_BLUE, lw=2.6, ms=6, label="BPE-dropout")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["250k", "500k", "1m", "4m", "16m"])
    for xi, lo, hi in zip(sizes, bpe, drop):
        ax.text(xi, hi + 0.18, f"+{hi - lo:.2f}", ha="center", fontsize=9.5,
                color=ARM_ORANGE, weight="bold")
    ax.set_xlabel("training corpus size (WMT14 En-Fr subset)")
    ax.set_ylabel("BLEU")
    ax.set_title("Gains are largest on small / low-resource data", fontsize=12.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(26, 36)
    save(fig, "corpus_size.pdf")


def fig_analysis():
    # Two analysis results, both REAL numbers from the paper.
    # (a) char 4-gram precision of top-10 embedding neighbours: BPE 0.18 vs 0.29 (Sec 6.2).
    # (b) relative inference time (Table 4): 32k and 4k vocab, BPE vs BPE-dropout.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax = axes[0]
    labels = ["BPE", "BPE-dropout"]
    prec = [0.18, 0.29]
    b = ax.bar(labels, prec, color=[GREY, ARM_BLUE], width=0.55, zorder=3)
    ax.bar_label(b, fmt="%.2f", padding=3, fontsize=11)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("char 4-gram precision of top-10 neighbours")
    ax.set_title("Better embeddings for rare tokens", fontsize=12)

    ax = axes[1]
    voc = ["32k vocab", "4k vocab"]
    bpe_t = [1.00, 1.44]
    bd_t = [1.03, 1.46]
    x = np.arange(len(voc)); w = 0.36
    b1 = ax.bar(x - w / 2, bpe_t, w, color=GREY, zorder=3, label="BPE")
    b2 = ax.bar(x + w / 2, bd_t, w, color=ARM_BLUE, zorder=3, label="BPE-dropout")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9.5)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(voc)
    ax.set_ylabel("relative inference time")
    ax.set_ylim(0, 1.75)
    ax.set_title("Inference cost basically unchanged", fontsize=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)

    fig.tight_layout()
    save(fig, "analysis.pdf")


if __name__ == "__main__":
    fig_seg_example(); fig_bleu_gains(); fig_dropout_p()
    fig_robustness(); fig_corpus_size(); fig_analysis()
    print("all BPE-dropout figures done")
