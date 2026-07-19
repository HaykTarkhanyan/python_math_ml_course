"""Figures for the Scaling Laws deck. Run: python make_figures.py -> ../fig/*.pdf

Real numbers come from Kaplan et al. 2020 (arXiv:2001.08361) and Hoffmann et al.
2022 / Chinchilla (arXiv:2203.15556). Power-law / IsoFLOP shapes are labeled
schematics that reproduce the papers' figure shapes with the reported exponents.
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


def fig_power_laws():
    # Kaplan Fig 1 shape: loss is a straight line on log-log vs each of C, N, D.
    # Uses the paper's exponents and constants; schematic (single clean curve each).
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    C = np.logspace(-8, 2, 120)                     # PF-days
    axes[0].plot(C, (3.1e8 / C) ** 0.050, color=ARM_BLUE, lw=2.6)
    axes[0].set_xlabel("compute  $C$ (PF-days)")
    axes[0].set_title(r"$L \propto C^{-0.050}$")
    axes[0].set_ylabel("test loss")

    N = np.logspace(3, 9, 120)                       # non-embedding params
    axes[1].plot(N, (8.8e13 / N) ** 0.076, color=ARM_RED, lw=2.6)
    axes[1].set_xlabel("parameters  $N$")
    axes[1].set_title(r"$L \propto N^{-0.076}$")

    D = np.logspace(7, 10, 120)                      # tokens
    axes[2].plot(D, (5.4e13 / D) ** 0.095, color=ARM_ORANGE, lw=2.6)
    axes[2].set_xlabel("dataset  $D$ (tokens)")
    axes[2].set_title(r"$L \propto D^{-0.095}$")

    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major", ls=":", alpha=0.4)
    fig.suptitle("Straight lines on log-log: loss is a power law in each factor "
                 "(schematic, Kaplan 2020, Fig 1)", fontsize=12, y=1.05)
    fig.tight_layout()
    save(fig, "power_laws.pdf")


def fig_isoflop():
    # Chinchilla Approach 2 / Fig 3: a loss valley in model size at each FLOP
    # budget; the minima trace the efficient frontier. Schematic (clean parabolas).
    fig, ax = plt.subplots(figsize=(7.9, 4.5))
    budgets = [6e18, 1e20, 1e21, 1e22, 1e23]
    labels = [r"$6\times10^{18}$", r"$10^{20}$", r"$10^{21}$", r"$10^{22}$", r"$10^{23}$"]
    cols = [GREY, ARM_ORANGE, ARM_BLUE, GREEN, VIOLET]
    N = np.logspace(7.5, 11.3, 260)
    minN, minL = [], []
    for C, lab, col in zip(budgets, labels, cols):
        Nopt = 0.09 * C ** 0.5                        # optimal N ~ C^0.5 (schematic)
        floor = 1.6 + 28.9 * C ** (-0.07)
        L = floor + 0.42 * (np.log10(N) - np.log10(Nopt)) ** 2
        ax.plot(N, L, color=col, lw=2.2, label=lab)
        minN.append(Nopt); minL.append(floor)
    ax.plot(minN, minL, "k--", lw=1.5, zorder=5)
    ax.scatter(minN, minL, color="k", s=28, zorder=6)
    ax.annotate("efficient frontier:\noptimal $N$ grows with compute",
                (minN[3], minL[3]), (minN[0] * 1.15, minL[0] - 0.15),
                fontsize=9.5, color="k",
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xscale("log")
    ax.set_xlabel("model size  $N$ (parameters)")
    ax.set_ylabel("final training loss")
    ax.set_title("IsoFLOP valleys: each budget has one optimal $N$\n"
                 "(schematic, Chinchilla 2022, Approach 2)")
    ax.legend(title="FLOP budget", frameon=False, fontsize=9, ncol=2, loc="upper right")
    ax.set_ylim(1.9, 4.3)
    save(fig, "isoflop.pdf")


def fig_kaplan_vs_chinchilla():
    # Optimal model size vs compute: Kaplan N ~ C^0.73 vs Chinchilla N ~ C^0.50,
    # anchored so both pass through the Gopher FLOP budget. Schematic + real anchors.
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    C = np.logspace(21, 25, 120)
    Cg = 5.76e23                                      # Gopher compute budget (FLOPs)
    ax.plot(C, 2.8e11 * (C / Cg) ** 0.73, color=ARM_RED, lw=2.6,
            label=r"Kaplan 2020:  $N \propto C^{0.73}$")
    ax.plot(C, 7.0e10 * (C / Cg) ** 0.50, color=GREEN, lw=2.6,
            label=r"Chinchilla 2022:  $N \propto C^{0.50}$")
    ax.axvline(Cg, color=GREY, ls=":", lw=1.3)
    ax.scatter([Cg, Cg], [2.8e11, 7.0e10], color="k", zorder=6, s=34)
    ax.annotate("Gopher 280B", (Cg, 2.8e11), (Cg * 0.10, 2.8e11 * 1.7),
                fontsize=9.5, color=ARM_RED)
    ax.annotate("Chinchilla 70B", (Cg, 7.0e10), (Cg * 1.25, 7.0e10 * 0.42),
                fontsize=9.5, color=GREEN)
    ax.text(Cg * 1.15, 3e9, "Gopher\nbudget", color=GREY, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute  $C$ (FLOPs)")
    ax.set_ylabel("compute-optimal model size  $N$")
    ax.set_title("Same compute, opposite advice: Kaplan's steeper slope pours\n"
                 "compute into size; Chinchilla splits it evenly")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    save(fig, "kaplan_vs_chinchilla.pdf")


def fig_tokens_per_param():
    # Chinchilla Table 3 (Approach 1): optimal tokens per parameter ~ 20 across scales.
    sizes = ["400M", "1B", "10B", "67B", "175B", "280B"]
    N = np.array([0.4e9, 1e9, 10e9, 67e9, 175e9, 280e9])
    D = np.array([8.0e9, 20.2e9, 205.1e9, 1.5e12, 3.7e12, 5.9e12])
    ratio = D / N
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    ax.bar(x, ratio, color=ARM_BLUE, zorder=3, width=0.6)
    ax.axhline(20, color=ARM_RED, ls="--", lw=1.6)
    ax.text(len(sizes) - 0.45, 20.7, "20 tokens / parameter", color=ARM_RED,
            ha="right", fontsize=10.5, weight="bold")
    for xi, r in zip(x, ratio):
        ax.text(xi, r + 0.3, f"{r:.0f}x", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("compute-optimal model size")
    ax.set_ylabel("optimal tokens per parameter  $D/N$")
    ax.set_title(r"The 20:1 rule: optimal $D/N \approx 20$ at every scale "
                 "(Chinchilla Table 3)")
    ax.set_ylim(0, 26)
    save(fig, "tokens_per_param.pdf")


def fig_chinchilla_vs_gopher():
    # Real numbers: same compute, Chinchilla 70B/1.4T vs Gopher 280B/0.3T.
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))
    names = ["Gopher", "Chinchilla"]
    cols = [ARM_RED, GREEN]
    panels = [
        ([280, 70], "parameters (B)", "Model size", "{:.0f}B"),
        ([300, 1400], "training tokens (B)", "Training data", "{:.0f}B"),
        ([60.0, 67.6], "MMLU 5-shot (%)", "MMLU accuracy", "{:.1f}%"),
    ]
    for ax, (vals, ylab, title, fmt) in zip(axes, panels):
        x = np.arange(2)
        ax.bar(x, vals, color=cols, zorder=3, width=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.02, fmt.format(v), ha="center", fontsize=10.5,
                    weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Equal training compute: Chinchilla 70B (1.4T tokens) beats "
                 "Gopher 280B (0.3T tokens)", fontsize=12, y=1.04)
    fig.tight_layout()
    save(fig, "chinchilla_vs_gopher.pdf")


def fig_undertrained():
    # Real numbers (Chinchilla Table 1): 2020-2022 LLMs vs the 20:1 optimum line.
    models = ["LaMDA", "GPT-3", "Jurassic-1", "Gopher", "MT-NLG", "Chinchilla"]
    N = np.array([137, 175, 178, 280, 530, 70]) * 1e9
    D = np.array([168, 300, 300, 300, 270, 1400]) * 1e9
    is_chin = [False] * 5 + [True]
    off = [(1.06, 1.10), (1.06, 0.80), (0.62, 1.14), (1.06, 1.12),
           (1.06, 1.10), (0.42, 1.14)]
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    nn = np.logspace(10.7, 11.85, 50)
    ax.plot(nn, 20 * nn, color=ARM_RED, ls="--", lw=1.8,
            label=r"compute-optimal  $D = 20\,N$")
    for m, n, d, c, (ox, oy) in zip(models, N, D, is_chin, off):
        col = GREEN if c else ARM_BLUE
        ax.scatter(n, d, color=col, s=72, zorder=5)
        ax.annotate(m, (n, d), (n * ox, d * oy), fontsize=9, color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e10, 9e11); ax.set_ylim(1.1e11, 8e12)
    ax.set_xlabel("model size  $N$ (parameters)")
    ax.set_ylabel("training tokens  $D$")
    ax.set_title("2020-2022 LLMs cluster at ~300B tokens, far below the 20:1 line\n"
                 "(under-trained); only Chinchilla sits on the optimum")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "undertrained.pdf")


if __name__ == "__main__":
    fig_power_laws()
    fig_isoflop()
    fig_kaplan_vs_chinchilla()
    fig_tokens_per_param()
    fig_chinchilla_vs_gopher()
    fig_undertrained()
    print("all scaling-law figures done")
