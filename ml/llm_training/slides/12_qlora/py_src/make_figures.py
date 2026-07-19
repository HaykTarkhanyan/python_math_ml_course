"""Figures for the QLoRA deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
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


def fig_memory_wall():
    # Memory to FINETUNE LLaMA 65B. Full 16-bit FT: >780 GB (paper, intro).
    # 16-bit LoRA: frozen base in fp16 alone is 65B*2B = 130 GB (a floor).
    # QLoRA: 45.0 GB (paper Appendix G / Fig 6, bs=1 seqlen=512, grad ckpt).
    methods = ["16-bit full\nfine-tuning", "16-bit LoRA\n(frozen base fp16)",
               "QLoRA\n(4-bit NF4 base)"]
    gb = [780, 130, 45.0]
    labels = [">780 GB", "≥130 GB", "45 GB"]
    colors = [ARM_RED, ARM_ORANGE, GREEN]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.4, 3.7))
    ax.barh(y, gb, color=colors, zorder=3, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(methods)
    ax.invert_yaxis()
    for yi, g, lab in zip(y, gb, labels):
        ax.text(g + 12, yi, lab, va="center", fontsize=11, weight="bold")
    ax.axvline(48, color=ARM_BLUE, ls="--", lw=1.8)
    ax.text(48, -0.72, "one 48 GB GPU", color=ARM_BLUE, fontsize=10,
            ha="center", weight="bold")
    ax.set_xlabel("GPU memory to finetune a 65B model (GB)")
    ax.set_title("QLoRA drops 65B finetuning from >780 GB onto a single 48 GB GPU")
    ax.set_xlim(0, 900)
    save(fig, "memory_wall.pdf")


def fig_where_memory():
    # Where the memory goes in LoRA finetuning of LLaMA 7B (paper, Background sec).
    # 4-bit base model 5048 MB; LoRA input gradients w/ ckpt 18 MB; LoRA params 26 MB.
    labels = ["4-bit base model\n(frozen)", "LoRA input grads\n(grad-checkpointed)",
              "LoRA parameters\n(trainable)"]
    mb = [5048, 18, 26]
    colors = [ARM_RED, GREY, ARM_BLUE]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.barh(y, mb, color=colors, zorder=3, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, m in zip(y, mb):
        ax.text(m * 1.25, yi, f"{m:,} MB", va="center", fontsize=10.5)
    ax.set_xlim(8, 2e4)
    ax.set_xlabel("memory footprint (MB, log scale) — LLaMA 7B, batch 1")
    ax.set_title("The frozen base weights ARE the wall (~200x the adapters)")
    save(fig, "where_memory.pdf")


def fig_nf4_buckets():
    # SCHEMATIC. Int4 (uniform levels) vs NF4 (equal-mass quantiles) over a normal.
    # NF4 packs more levels where the weight mass is; uniform wastes levels in tails.
    rng = np.random.default_rng(509)
    x = np.linspace(-3, 3, 600)
    pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    nbins = 8  # schematic: real NF4 = 16 levels; 8 bins shown for clarity
    # uniform boundaries across the clipped range [-1,1] (absmax-normalized)
    uni = np.linspace(-1, 1, nbins + 1) * 2.6
    # equal-mass boundaries: empirical quantiles of a standard normal
    samp = rng.standard_normal(400000)
    q = np.quantile(samp, np.linspace(0, 1, nbins + 1))
    q[0], q[-1] = -3, 3

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, bnd, title, col in [
            (axes[0], uni, "Int4 / uniform: equal-width levels", ARM_RED),
            (axes[1], q, "NF4: equal-MASS quantile levels", GREEN)]:
        ax.plot(x, pdf, color="#333333", lw=1.8)
        ax.fill_between(x, pdf, color=ARM_BLUE, alpha=0.08)
        for b in bnd:
            ax.axvline(b, color=col, lw=1.4, alpha=0.85)
        # shade the mass in each bin to make "equal mass" visible on the right
        for lo, hi in zip(bnd[:-1], bnd[1:]):
            xm = np.linspace(lo, hi, 60)
            ax.fill_between(xm, np.exp(-xm ** 2 / 2) / np.sqrt(2 * np.pi),
                            color=col, alpha=0.10)
        ax.set_title(title, fontsize=12, color=col)
        ax.set_xlim(-3, 3); ax.set_ylim(0, 0.46)
        ax.set_yticks([]); ax.set_xlabel("weight value (absmax-normalized)")
    axes[0].text(0, 0.42, "levels wasted\nin the empty tails", ha="center",
                 fontsize=9.5, color=ARM_RED)
    axes[1].text(0, 0.42, "each bin holds an\nequal # of weights", ha="center",
                 fontsize=9.5, color=GREEN)
    fig.suptitle("SCHEMATIC: NF4 spends its 16 levels where the weights actually are",
                 fontsize=12.5, y=1.04)
    fig.tight_layout()
    save(fig, "nf4_buckets.pdf")


def fig_nf4_perplexity():
    # Table 2: Pile Common Crawl mean perplexity, 125M-13B models. Lower is better.
    labels = ["Int4", "Float4\n(E2M1)", "Float4\n(E3M0)", "NFloat4\n+ DQ"]
    ppl = [34.34, 31.07, 29.48, 27.41]
    colors = [GREY, ARM_ORANGE, ARM_ORANGE, GREEN]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.bar(x, ppl, color=colors, zorder=3, width=0.6)
    for xi, p in zip(x, ppl):
        ax.text(xi, p + 0.25, f"{p:.2f}", ha="center", fontsize=10.5, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean perplexity (lower is better)")
    ax.set_title("NF4 is the most accurate 4-bit type (Pile CC perplexity)")
    ax.set_ylim(25, 36)
    ax.annotate("", xy=(3, 27.9), xytext=(0, 34.9),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    save(fig, "nf4_perplexity.pdf")


def fig_gpu_fit():
    # Appendix G / Fig 6: QLoRA finetuning memory (bs 1, seqlen 512, grad ckpt).
    sizes = ["7B", "13B", "33B", "65B"]
    gb = [6.9, 11.3, 24.7, 45.0]
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    cols = [GREEN, GREEN, ARM_ORANGE, GREEN]
    ax.bar(x, gb, color=cols, zorder=3, width=0.56)
    for xi, g in zip(x, gb):
        ax.text(xi, g + 0.7, f"{g:.1f} GB", ha="center", fontsize=10.5, weight="bold")
    ax.axhline(24, color=ARM_BLUE, ls="--", lw=1.6)
    ax.text(3.35, 24.4, "24 GB GPU", color=ARM_BLUE, fontsize=9.5, ha="right")
    ax.axhline(48, color=ARM_RED, ls="--", lw=1.6)
    ax.text(3.35, 48.4, "48 GB GPU", color=ARM_RED, fontsize=9.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("QLoRA finetuning memory (GB)")
    ax.set_xlabel("LLaMA model size")
    ax.set_title("Every size fits one GPU (33B just overflows 24 GB → paged optimizers)")
    ax.set_ylim(0, 56)
    save(fig, "gpu_fit.pdf")


def fig_parity():
    # Table 4: mean 5-shot MMLU (mean of Alpaca + FLAN v2) per size,
    # 16-bit BFloat16 vs 4-bit NFloat4+DQ. Bars are ~identical => no gap.
    sizes = ["7B", "13B", "33B", "65B"]
    bf16 = [(38.4 + 45.6) / 2, (47.2 + 50.6) / 2, (57.7 + 60.5) / 2, (61.8 + 62.5) / 2]
    nf4 = [(39.0 + 44.5) / 2, (47.5 + 50.7) / 2, (57.3 + 59.2) / 2, (61.8 + 63.9) / 2]
    x = np.arange(len(sizes)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(x - w / 2, bf16, w, label="16-bit BFloat16", color=GREY, zorder=3)
    ax.bar(x + w / 2, nf4, w, label="4-bit NF4 + DQ (QLoRA)", color=GREEN, zorder=3)
    for xi, b, n in zip(x, bf16, nf4):
        ax.text(xi - w / 2, b + 0.6, f"{b:.1f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, n + 0.6, f"{n:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("LLaMA model size")
    ax.set_ylabel("mean 5-shot MMLU accuracy (%)")
    ax.set_title("4-bit QLoRA matches 16-bit finetuning at every scale — no gap")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(30, 70)
    save(fig, "parity.pdf")


def fig_guanaco_vicuna():
    # Table 6: Vicuna benchmark score as % of ChatGPT (GPT-4 judge, mean of both orders).
    models = ["GPT-4", "Guanaco 65B (4-bit, 41 GB)", "Guanaco 33B (4-bit, 21 GB)",
              "Vicuna 13B (16-bit, 26 GB)", "Bard", "Guanaco 13B (4-bit, 10 GB)",
              "Guanaco 7B (4-bit, 5 GB)", "Alpaca 65B (4-bit, 41 GB)"]
    pct = [114.5, 99.3, 97.8, 94.9, 94.8, 90.4, 87.0, 70.7]
    colors = [ARM_BLUE] + [GREEN if "Guanaco" in m else GREY for m in models[1:]]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.barh(y, pct, color=colors, zorder=3, height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(models, fontsize=10)
    ax.invert_yaxis()
    for yi, p in zip(y, pct):
        ax.text(p + 1, yi, f"{p:.1f}%", va="center", fontsize=9.5, weight="bold")
    ax.axvline(100, color=ARM_RED, ls="--", lw=1.7)
    ax.text(100, len(models) - 0.3, "ChatGPT = 100%", color=ARM_RED,
            fontsize=9.5, ha="center")
    ax.set_xlabel("Vicuna benchmark score, % of ChatGPT (GPT-4 judge)")
    ax.set_title("Guanaco 65B reaches 99.3% of ChatGPT — one GPU, 24 hours")
    ax.set_xlim(0, 128)
    save(fig, "guanaco_vicuna.pdf")


if __name__ == "__main__":
    fig_memory_wall(); fig_where_memory(); fig_nf4_buckets(); fig_nf4_perplexity()
    fig_gpu_fit(); fig_parity(); fig_guanaco_vicuna()
    print("all QLoRA figures done")
