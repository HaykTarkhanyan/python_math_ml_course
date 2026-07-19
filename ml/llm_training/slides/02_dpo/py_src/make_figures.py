"""Figures for the DPO deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
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


def fig_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2))

    def box(ax, x, y, w, h, text, fc, ec, tc="white", fs=10.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text(x, y, text, ha="center", va="center", color=tc,
                fontsize=fs, weight="bold", zorder=4)

    def arr(ax, p, q, c="#444444", lw=1.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                     color=c, lw=lw, shrinkA=3, shrinkB=3, zorder=2))

    # RLHF row
    ax = axes[0]
    ax.set_title("RLHF: reward model + RL loop", color=ARM_RED, loc="left", weight="bold")
    box(ax, 1.2, 0.5, 2.0, 0.7, "Preference\ndata", GREY, GREY)
    box(ax, 4.0, 0.5, 2.2, 0.7, "Reward model\n$r_\\varphi$  (train)", ARM_ORANGE, ARM_ORANGE, tc="#333")
    box(ax, 7.0, 0.5, 2.6, 0.7, "PPO: max $r-\\beta$KL\n(sample from $\\pi$)", ARM_RED, ARM_RED)
    box(ax, 9.7, 0.5, 1.6, 0.7, "Policy\n$\\pi$", ARM_BLUE, ARM_BLUE)
    arr(ax, (2.2, 0.5), (2.9, 0.5)); arr(ax, (5.1, 0.5), (5.7, 0.5))
    arr(ax, (8.3, 0.5), (8.9, 0.5))
    ax.text(5.5, 1.15, "2 models trained  +  RL loop  +  in-loop sampling",
            ha="center", color=ARM_RED, fontsize=10, style="italic")
    ax.set_xlim(0, 10.8); ax.set_ylim(-0.1, 1.5); ax.axis("off")

    # DPO row
    ax = axes[1]
    ax.set_title("DPO: one classification loss", color=GREEN, loc="left", weight="bold")
    box(ax, 1.6, 0.5, 2.8, 0.7, "Preference data\n$(x, y_w, y_l)$", GREY, GREY)
    box(ax, 5.6, 0.5, 3.6, 0.8, "max-likelihood on\nimplicit-reward log-ratios", GREEN, GREEN, fs=10.5)
    box(ax, 9.5, 0.5, 2.0, 0.7, "Policy $\\pi$\n(= reward)", ARM_BLUE, ARM_BLUE, fs=10)
    arr(ax, (3.0, 0.5), (3.75, 0.5), c=GREEN); arr(ax, (7.45, 0.5), (8.45, 0.5), c=GREEN)
    ax.text(5.6, 1.2, "1 loss  -  no reward model  -  no RL  -  no sampling",
            ha="center", color=GREEN, fontsize=10, style="italic")
    ax.set_xlim(0, 10.8); ax.set_ylim(-0.1, 1.6); ax.axis("off")
    fig.tight_layout()
    save(fig, "pipeline.pdf")


def fig_frontier():
    kl = np.linspace(0, 25, 300)
    def curve(a, tau): return a * (1 - np.exp(-kl / tau))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(kl, curve(1.00, 3.5), color=ARM_BLUE, lw=2.8, label="DPO")
    ax.plot(kl, curve(0.86, 4.5), color=GREEN, lw=2.2, ls="-", label="PPO-GT (oracle)")
    ax.plot(kl, curve(0.78, 5.5), color=ARM_RED, lw=2.2, label="PPO (learned r)")
    ax.plot(kl, curve(0.55, 6.5), color=GREY, lw=1.8, ls="--", label="Preferred-FT")
    ax.plot(kl, curve(0.42, 7.5), color=ARM_ORANGE, lw=1.8, ls=":", label="Unlikelihood")
    ax.set_xlabel(r"KL$(\pi_\theta \,\|\, \pi_{ref})$")
    ax.set_ylabel("expected reward")
    ax.set_title("Reward vs KL frontier (IMDb sentiment) - DPO dominates everywhere")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.text(0.02, 0.95, "schematic reproduction of the reported ordering",
            transform=ax.transAxes, fontsize=8.5, color=GREY, style="italic")
    save(fig, "frontier.pdf")


def fig_loss_shape():
    d = np.linspace(-4, 4, 400)   # implicit-reward margin r_w - r_l
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for beta, c in [(0.1, GREY), (0.5, ARM_ORANGE), (1.0, ARM_BLUE)]:
        ax.plot(d, -np.log(1 / (1 + np.exp(-beta * d))), color=c, lw=2.3,
                label=fr"loss  $\beta={beta}$")
    ax2 = ax.twinx()
    ax2.plot(d, 1 / (1 + np.exp(1.0 * d)), color=GREEN, lw=2.3, ls="--",
             label=r"grad weight $\sigma(\hat r_l-\hat r_w)$")
    ax2.set_ylabel("gradient weight", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN); ax2.set_ylim(0, 1.05)
    ax.axvline(0, color="#bbbbbb", ls=":", lw=1)
    ax.set_xlabel(r"implicit-reward margin  $\hat r(y_w)-\hat r(y_l)$")
    ax.set_ylabel("per-example loss")
    ax.set_title("DPO loss and its gradient weight")
    ax.text(-3.7, 2.4, "ordering WRONG:\nbig update", color=ARM_RED, fontsize=9, weight="bold")
    ax.text(1.4, 0.35, "already correct:\ntiny update", color=GREEN, fontsize=9, weight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "loss_shape.pdf")


def fig_temperature():
    t = np.linspace(0, 1, 100)
    dpo = 0.61 - 0.05 * t                       # high and flat
    ppo = 0.57 - 0.42 * t**1.6                   # decays sharply
    sft = 0.28 + 0.0 * t
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(t, dpo, color=ARM_BLUE, lw=2.6, label="DPO")
    ax.plot(t, ppo, color=ARM_RED, lw=2.4, label="PPO")
    ax.plot(t, sft, color=GREY, lw=1.8, ls="--", label="SFT")
    ax.axhline(0.5, color="#cccccc", lw=1, ls=":")
    ax.scatter([0, 0], [0.61, 0.57], color=[ARM_BLUE, ARM_RED], zorder=5)
    ax.annotate("61%", (0, 0.61), (0.08, 0.66), color=ARM_BLUE, fontsize=10, weight="bold")
    ax.annotate("57%", (0, 0.57), (0.08, 0.50), color=ARM_RED, fontsize=10, weight="bold")
    ax.set_xlabel("sampling temperature")
    ax.set_ylabel("win rate vs reference (GPT-4 judge)")
    ax.set_title("TL;DR summarization: DPO is higher AND temperature-robust")
    ax.legend(frameon=False, loc="lower left"); ax.set_ylim(0, 0.7)
    save(fig, "temperature.pdf")


def fig_winrate():
    methods = ["DPO", "SFT", "PPO-1"]
    gpt4s = [47, 27, 13]; gpt4c = [54, 32, 12]; human = [58, 43, 17]
    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(x - w, gpt4s, w, label="GPT-4 (simple)", color=GREY, zorder=3)
    ax.bar(x, gpt4c, w, label="GPT-4 (concise)", color=ARM_ORANGE, zorder=3)
    ax.bar(x + w, human, w, label="Human", color=ARM_BLUE, zorder=3)
    for i, vals in enumerate(zip(gpt4s, gpt4c, human)):
        for j, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * w, v + 1, f"{v}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("win rate vs PPO@temp0 (%)")
    ax.set_title("Anthropic-HH: DPO preferred by both GPT-4 and humans")
    ax.axhline(50, color="#cccccc", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=9.5); ax.set_ylim(0, 70)
    save(fig, "winrate.pdf")


if __name__ == "__main__":
    fig_pipeline(); fig_frontier(); fig_loss_shape(); fig_temperature(); fig_winrate()
    print("all DPO figures done")
