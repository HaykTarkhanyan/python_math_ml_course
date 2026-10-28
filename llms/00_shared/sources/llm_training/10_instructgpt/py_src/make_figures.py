"""Figures for the InstructGPT / RLHF deck.

Run with the project venv (or any env with matplotlib + numpy):
    python make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).

Number provenance:
  - win_rate: trend reconstructed from Ouyang et al. 2022 Figure 1 (exact plotted
    values approximate); the two head-to-head numbers (85%, 71%) are exact from text.
  - rm_vs_human: exact (RM 5-fold CV 69.6%, RM train 72.4%, labeler agreement 72.6%).
  - honesty_harm: hallucination 41% vs 21% exact; toxicity -25% (respectful) exact.
  - ppo_clip: schematic (standard PPO Fig 1 shape), fully labeled.
  - rlhf_pipeline, alignment_tax: schematic diagrams, labeled as such on the slide.
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


# ---- Fig A: the 3-step RLHF pipeline (centerpiece) ---------------------
def fig_rlhf_pipeline():
    fig, ax = plt.subplots(figsize=(12.6, 5.0))

    def box(xy, w, h, text, fc, ec, tc="white", fs=10.5, lw=1.8, weight="bold"):
        b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(b)
        ax.text(xy[0], xy[1], text, ha="center", va="center",
                color=tc, fontsize=fs, zorder=4, weight=weight)

    def arrow(p, q, color="#444444", lw=1.8, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14,
                                     color=color, lw=lw, zorder=2,
                                     shrinkA=3, shrinkB=3))

    xs = [2.0, 6.3, 10.6]           # column centres for the 3 steps
    # column dividers / headers
    for x, title, col in zip(xs,
                             ["STEP 1  -  SFT", "STEP 2  -  Reward model",
                              "STEP 3  -  RL (PPO)"],
                             [ARM_BLUE, GREEN, ARM_ORANGE]):
        ax.text(x, 4.75, title, ha="center", fontsize=12.5, weight="bold", color=col)

    # human-feedback input boxes (top row)
    box((xs[0], 3.9), 3.1, 0.7, "Human demonstrations\n(13k prompts)", "#eef2fb",
        ARM_BLUE, tc=ARM_BLUE, fs=9.5)
    box((xs[1], 3.9), 3.1, 0.7, "Human rankings of\n4-9 outputs (33k prompts)", "#eaf5ee",
        GREEN, tc=GREEN, fs=9.5)
    box((xs[2], 3.9), 3.1, 0.7, "Prompts only\n(31k, no labels)", "#fdf4e2",
        ARM_ORANGE, tc="#8a6400", fs=9.5)
    ax.text(6.3, 4.42, "human feedback", ha="center", fontsize=9.5,
            style="italic", color="#666666")

    # model / process boxes (middle row)
    box((xs[0], 2.35), 2.9, 0.95, "Fine-tune GPT-3\nsupervised\n$\\Rightarrow\\ \\pi_{SFT}$",
        ARM_BLUE, ARM_BLUE, fs=10)
    box((xs[1], 2.35), 2.9, 0.95, "Train reward model\n(Bradley-Terry)\n$\\Rightarrow\\ r_\\theta$ (6B)",
        GREEN, GREEN, fs=10)
    box((xs[2], 2.35), 2.9, 0.95,
        "Maximize $r_\\theta$\n$-\\ \\beta\\,$KL to $\\pi_{SFT}$\n$\\Rightarrow$ InstructGPT",
        ARM_ORANGE, ARM_ORANGE, tc="#5a4400", fs=10)

    # outputs (bottom row)
    box((xs[0], 0.85), 2.5, 0.55, "$\\pi_{SFT}$", "white", ARM_BLUE, tc=ARM_BLUE, fs=11)
    box((xs[1], 0.85), 2.5, 0.55, "$r_\\theta$", "white", GREEN, tc=GREEN, fs=11)
    box((xs[2], 0.85), 2.5, 0.55, "$\\pi_{RL}$  (policy)", "white", ARM_ORANGE,
        tc="#8a6400", fs=11)

    # vertical arrows within each column
    for x in xs:
        arrow((x, 3.55), (x, 2.85))     # feedback -> model
        arrow((x, 1.88), (x, 1.15))     # model -> output

    # cross-column dependency arrows
    # SFT initializes the RM and the PPO policy
    arrow((xs[0] + 1.25, 0.85), (xs[1] - 1.25, 0.85), color=ARM_BLUE, lw=1.6)
    ax.text((xs[0] + xs[1]) / 2, 1.08, "init", ha="center", fontsize=8.5,
            color=ARM_BLUE, style="italic")
    # RM provides the reward signal to PPO
    arrow((xs[1] + 1.25, 0.85), (xs[2] - 1.25, 0.85), color=GREEN, lw=1.6)
    ax.text((xs[1] + xs[2]) / 2, 1.08, "reward", ha="center", fontsize=8.5,
            color=GREEN, style="italic")
    # SFT is also the frozen KL anchor for step 3 (long curved arrow)
    ax.add_patch(FancyArrowPatch((xs[0], 0.58), (xs[2], 0.58),
                 connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
                 mutation_scale=13, color="#999999", lw=1.4, ls="--", zorder=1))
    ax.text(6.3, -0.35, "$\\pi_{SFT}$ frozen as the KL anchor for step 3",
            ha="center", fontsize=9, color="#777777", style="italic")

    ax.set_xlim(-0.1, 12.7)
    ax.set_ylim(-0.7, 5.1)
    ax.axis("off")
    fig.suptitle("RLHF: supervised demos $\\rightarrow$ a learned reward $\\rightarrow$ RL against it",
                 fontsize=13, y=1.0, color="#333333")
    save(fig, "rlhf_pipeline.pdf")


# ---- Fig B: headline win rate vs 175B SFT ------------------------------
def fig_win_rate():
    sizes = ["1.3B", "6B", "175B"]
    x = np.arange(3)
    # trend reconstructed from Figure 1 (exact plotted values approximate)
    gpt = [0.18, 0.21, 0.28]
    gpt_p = [0.40, 0.42, 0.44]
    sft = [0.43, 0.47, 0.50]
    ppo = [0.53, 0.63, 0.70]
    ppo_ptx = [0.52, 0.62, 0.71]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.axhline(0.5, color="#bbbbbb", ls=":", lw=1.4)
    ax.text(2.02, 0.505, "175B SFT baseline", color="#888888", fontsize=8.5, va="bottom")

    ax.plot(x, gpt, "-s", color=GREY, lw=2, ms=5, label="GPT-3")
    ax.plot(x, gpt_p, "-^", color=ARM_ORANGE, lw=2, ms=5, label="GPT-3 (prompted)")
    ax.plot(x, sft, "-o", color=VIOLET, lw=2, ms=5, label="SFT")
    ax.plot(x, ppo, "--o", color=GREEN, lw=2, ms=5, label="PPO")
    ax.plot(x, ppo_ptx, "-o", color=ARM_BLUE, lw=3, ms=7,
            label="PPO-ptx (InstructGPT)")

    # the striking comparison: 1.3B InstructGPT beats 175B GPT-3
    ax.scatter([0], [ppo_ptx[0]], s=150, facecolors="none", edgecolors=ARM_RED, lw=2, zorder=5)
    ax.scatter([2], [gpt[2]], s=150, facecolors="none", edgecolors=ARM_RED, lw=2, zorder=5)
    ax.annotate("1.3B InstructGPT (0.52)\nbeats 175B GPT-3 (0.28)",
                xy=(2, gpt[2]), xytext=(0.35, 0.30), fontsize=9.5, color=ARM_RED,
                weight="bold",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.set_ylabel("win rate vs 175B SFT")
    ax.set_title("Labelers prefer InstructGPT across every model size (from Fig 1)")
    ax.set_ylim(0.1, 0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "win_rate.pdf")


# ---- Fig C: PPO clipped surrogate objective ----------------------------
def fig_ppo_clip():
    eps = 0.2
    r = np.linspace(0.0, 2.0, 500)

    def clipped_term(r, A):
        return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=False)

    # ---- A > 0 ----
    ax = axes[0]
    A = 1.0
    ax.plot(r, r * A, color=GREY, lw=1.6, ls="--", label=r"unclipped $rA$")
    ax.plot(r, clipped_term(r, A), color=ARM_BLUE, lw=2.8, label=r"$\min(rA,\,\mathrm{clip}(r)A)$")
    ax.axvline(1, color="#cccccc", lw=1)
    ax.axvline(1 + eps, color=ARM_RED, ls=":", lw=1.4)
    ax.scatter([1], [1], s=70, color=ARM_RED, zorder=5)
    ax.text(1.0, 0.08, "r=1", color="#888888", fontsize=9, ha="center")
    ax.text(1 + eps, 1.85, r"$1+\epsilon$", color=ARM_RED, fontsize=10, ha="center")
    ax.annotate("clipped: no reward for\npushing r past $1+\\epsilon$",
                xy=(1.55, 1.2), xytext=(0.75, 1.6), fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))
    ax.set_title(r"advantage $A>0$ (good action)", color=GREEN)
    ax.set_xlabel(r"ratio $r=\pi_\theta/\pi_{\theta_{old}}$")
    ax.set_ylabel("objective (one term)")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # ---- A < 0 ----
    ax = axes[1]
    A = -1.0
    ax.plot(r, r * A, color=GREY, lw=1.6, ls="--", label=r"unclipped $rA$")
    ax.plot(r, clipped_term(r, A), color=ARM_BLUE, lw=2.8, label=r"$\min(rA,\,\mathrm{clip}(r)A)$")
    ax.axvline(1, color="#cccccc", lw=1)
    ax.axvline(1 - eps, color=ARM_RED, ls=":", lw=1.4)
    ax.scatter([1], [-1], s=70, color=ARM_RED, zorder=5)
    ax.text(1.0, -0.12, "r=1", color="#888888", fontsize=9, ha="center")
    ax.text(1 - eps, -1.9, r"$1-\epsilon$", color=ARM_RED, fontsize=10, ha="center")
    ax.annotate("clipped: no reward for\npushing r below $1-\\epsilon$",
                xy=(0.55, -1.1), xytext=(0.9, -1.75), fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))
    ax.set_title(r"advantage $A<0$ (bad action)", color=ARM_RED)
    ax.set_xlabel(r"ratio $r=\pi_\theta/\pi_{\theta_{old}}$")
    ax.set_ylabel("objective (one term)")
    ax.set_xlim(0, 2); ax.set_ylim(-2, 0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.suptitle("Clipped surrogate: the trust region that keeps updates small "
                 "($\\epsilon=0.2$, schematic)", fontsize=12, y=1.01, color="#333333")
    fig.tight_layout()
    save(fig, "ppo_clip.pdf")


# ---- Fig D: reward model accuracy vs human agreement -------------------
def fig_rm_vs_human():
    labels = ["RM\n(held-out labelers)", "RM\n(training labelers)",
              "Humans\n(inter-labeler agree)"]
    vals = [69.6, 72.4, 72.6]
    cols = [ARM_BLUE, ARM_BLUE, GREY]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(range(3), vals, width=0.6, color=cols, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{b.get_height():.1f}%", ha="center", fontsize=11, weight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("agreement / accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("The reward model predicts a preference about as well\nas two humans agree",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "rm_vs_human.pdf")


# ---- Fig E: honesty & harmlessness (real numbers) ----------------------
def fig_honesty_harm():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # ---- hallucination on closed-domain tasks (lower is better) ----
    ax = axes[0]
    vals = [41, 21]
    cols = [GREY, ARM_BLUE]
    bars = ax.bar([0, 1], vals, width=0.55, color=cols, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{int(b.get_height())}%", ha="center", fontsize=12, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GPT-3", "InstructGPT"])
    ax.set_ylabel("hallucination rate (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Makes up facts half as often\n(closed-domain tasks)", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25)

    # ---- toxicity, respectful prompt (relative index, lower is better) ----
    ax = axes[1]
    vals = [100, 75]
    cols = [GREY, ARM_BLUE]
    bars = ax.bar([0, 1], vals, width=0.55, color=cols, zorder=3)
    for b, lab in zip(bars, ["100", "75  (-25%)"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                lab, ha="center", fontsize=11.5, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GPT-3", "InstructGPT"])
    ax.set_ylabel("toxic outputs (index, GPT-3 = 100)")
    ax.set_ylim(0, 120)
    ax.set_title("25% fewer toxic outputs\n(when prompted to be respectful)", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "honesty_harm.pdf")


# ---- Fig F: the alignment tax (schematic) ------------------------------
def fig_alignment_tax():
    labels = ["GPT-3\n(baseline)", "PPO", "PPO-ptx"]
    vals = [100, 88, 99]
    cols = [GREY, ARM_RED, GREEN]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.axhline(100, color="#bbbbbb", ls=":", lw=1.4)
    bars = ax.bar(range(3), vals, width=0.58, color=cols, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                f"{int(b.get_height())}", ha="center", fontsize=11.5, weight="bold")
    # the "tax" gap
    ax.annotate("", xy=(1, 100), xytext=(1, 88),
                arrowprops=dict(arrowstyle="<->", color=ARM_RED, lw=1.6))
    ax.text(1.12, 94, "alignment\ntax", color=ARM_RED, fontsize=10, weight="bold",
            va="center")
    ax.text(2, 82, "pretraining mix\nrecovers it", color=GREEN, fontsize=9.5,
            ha="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("public NLP benchmark score (schematic)")
    ax.set_ylim(0, 115)
    ax.set_title("Plain PPO regresses on public NLP tasks; mixing in pretraining\n"
                 "gradients (PPO-ptx) pays back the tax", fontsize=11.5)
    save(fig, "alignment_tax.pdf")


if __name__ == "__main__":
    fig_rlhf_pipeline()
    fig_win_rate()
    fig_ppo_clip()
    fig_rm_vs_human()
    fig_honesty_harm()
    fig_alignment_tax()
    print("all InstructGPT figures done")
