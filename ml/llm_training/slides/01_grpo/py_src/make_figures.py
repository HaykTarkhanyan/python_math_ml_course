"""Figures for the GRPO / DeepSeekMath deck.

Run with the project venv (or any env with matplotlib + numpy):
    python3 make_figures.py
Outputs PDFs into ../fig/. Fails loud on any error (no silent fallback).
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


# ---- Fig A: PPO vs GRPO architecture schematic --------------------------
def fig_ppo_vs_grpo():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    def box(ax, xy, w, h, text, fc, ec, tc="white", fs=11, lw=1.6):
        b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.06",
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(b)
        ax.text(xy[0], xy[1], text, ha="center", va="center",
                color=tc, fontsize=fs, zorder=4, weight="bold")

    def arrow(ax, p, q, color="#444444", lw=1.6, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                     color=color, lw=lw, zorder=2,
                                     shrinkA=2, shrinkB=2))

    # ---- PPO panel ----
    ax = axes[0]
    ax.set_title("PPO: actor-critic (4 models)", color=ARM_RED, weight="bold")
    box(ax, (0.5, 3.4), 1.5, 0.7, "Policy\n$\\pi_\\theta$", ARM_BLUE, ARM_BLUE)
    box(ax, (2.7, 3.4), 1.5, 0.7, "Reference\n$\\pi_{ref}$", GREY, GREY)
    box(ax, (0.5, 1.9), 1.5, 0.7, "Reward\n$r_\\varphi$", GREEN, GREEN)
    box(ax, (2.7, 1.9), 1.5, 0.7, "Value / critic\n$V_\\psi$", ARM_RED, ARM_RED)
    box(ax, (1.6, 0.5), 1.7, 0.6, "GAE  $\\rightarrow A_t$", "white", "#444444", tc="#222222")
    arrow(ax, (0.5, 3.05), (0.5, 2.25))          # policy -> reward
    arrow(ax, (2.7, 1.55), (2.1, 0.75))          # value -> GAE
    arrow(ax, (0.5, 1.55), (1.1, 0.75))          # reward -> GAE
    arrow(ax, (1.6, 0.8), (0.5, 3.05), color=ARM_BLUE, lw=1.3)  # A -> policy update
    ax.text(2.7, 1.15, "trained, ~policy-sized,\nexpensive", ha="center",
            va="top", color=ARM_RED, fontsize=9.5, style="italic")
    ax.set_xlim(-0.6, 3.9); ax.set_ylim(-0.1, 4.1); ax.axis("off")

    # ---- GRPO panel ----
    ax = axes[1]
    ax.set_title("GRPO: group baseline (3 models)", color=GREEN, weight="bold")
    box(ax, (0.6, 3.5), 1.5, 0.7, "Policy\n$\\pi_\\theta$", ARM_BLUE, ARM_BLUE)
    box(ax, (3.0, 3.5), 1.5, 0.7, "Reference\n$\\pi_{ref}$", GREY, GREY)
    # group of outputs
    ys = [2.55, 2.05, 1.55, 1.05]
    labels = ["$o_1$", "$o_2$", "$\\vdots$", "$o_G$"]
    for y, lab in zip(ys, labels):
        box(ax, (1.15, y), 0.7, 0.34, lab, "#eef2fb", ARM_BLUE, tc=ARM_BLUE, fs=10, lw=1.1)
        arrow(ax, (0.6, 3.15), (0.95, y + 0.12), color=ARM_BLUE, lw=1.0)
    box(ax, (2.35, 1.8), 1.0, 0.55, "Reward\n$r_\\varphi$", GREEN, GREEN, fs=10)
    for y in ys:
        arrow(ax, (1.5, y), (1.95, 1.9), color="#999999", lw=0.9)
    box(ax, (3.55, 1.8), 1.35, 0.7, "Group\nmean / std", ARM_ORANGE, ARM_ORANGE, tc="#333333", fs=10)
    arrow(ax, (2.85, 1.8), (2.9, 1.8), color="#444444")
    box(ax, (3.55, 0.55), 1.35, 0.5, "$\\hat A_i$", "white", "#444444", tc="#222222")
    arrow(ax, (3.55, 1.45), (3.55, 0.8))
    arrow(ax, (2.9, 0.55), (0.6, 3.15), color=ARM_BLUE, lw=1.3)
    ax.text(3.55, 2.7, "no value net!", ha="center", color=GREEN,
            fontsize=10.5, weight="bold", style="italic")
    ax.set_xlim(-0.2, 4.5); ax.set_ylim(-0.1, 4.2); ax.axis("off")

    fig.suptitle("4 models  $\\rightarrow$  3 models: the critic is replaced by a group-relative baseline",
                 fontsize=12.5, y=1.02, color="#333333")
    save(fig, "ppo_vs_grpo.pdf")


# ---- Fig B: group-relative advantage worked example --------------------
def fig_group_advantage():
    # illustrative rewards for G=6 outputs (labelled illustrative on slide)
    r = np.array([0.9, 0.2, 0.8, -0.1, 0.5, -0.3])
    idx = np.arange(1, len(r) + 1)
    mu, sd = r.mean(), r.std()
    adv = (r - mu) / sd

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    ax.bar(idx, r, color=ARM_BLUE, width=0.62, zorder=3)
    ax.axhline(mu, color=ARM_ORANGE, lw=2, zorder=2)
    ax.fill_between([0.4, 6.6], mu - sd, mu + sd, color=ARM_ORANGE, alpha=0.15, zorder=1)
    ax.text(6.5, mu, "  mean", color=ARM_ORANGE, va="center", fontsize=10, weight="bold")
    ax.text(6.5, mu + sd, "  +std", color=ARM_ORANGE, va="center", fontsize=9)
    ax.set_title("Raw rewards $r_i$ within the group")
    ax.set_xlabel("output $i$"); ax.set_xticks(idx)
    ax.set_xlim(0.4, 7.4)

    ax = axes[1]
    colors = [GREEN if a >= 0 else ARM_RED for a in adv]
    ax.bar(idx, adv, color=colors, width=0.62, zorder=3)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_title(r"Advantage $\hat A_i = (r_i-\mathrm{mean})/\mathrm{std}$")
    ax.set_xlabel("output $i$"); ax.set_xticks(idx)
    ax.text(0.5, 0.92, "better than avg", transform=ax.transAxes, color=GREEN,
            fontsize=9.5, weight="bold")
    ax.text(0.5, 0.06, "worse than avg", transform=ax.transAxes, color=ARM_RED,
            fontsize=9.5, weight="bold")

    fig.tight_layout()
    save(fig, "group_advantage.pdf")


# ---- Fig C: Instruct -> RL benchmark gains -----------------------------
def fig_benchmark_gains():
    bench = ["GSM8K", "MATH", "MGSM-zh", "CMATH*"]
    instruct = [82.9, 46.8, 73.2, 84.6]
    rl = [88.2, 51.7, 79.6, 88.8]
    x = np.arange(len(bench)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    b1 = ax.bar(x - w / 2, instruct, w, label="Instruct (SFT)", color=GREY, zorder=3)
    b2 = ax.bar(x + w / 2, rl, w, label="RL (GRPO)", color=ARM_BLUE, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", fontsize=9.5)
    for i in range(len(bench)):
        d = rl[i] - instruct[i]
        ax.annotate(f"+{d:.1f}", (x[i], max(rl[i], instruct[i]) + 4.2),
                    ha="center", color=GREEN, fontsize=10.5, weight="bold")
    ax.set_ylabel("accuracy (%)  CoT")
    ax.set_title("DeepSeekMath 7B: SFT $\\rightarrow$ GRPO gains  (*CMATH = out-of-domain)")
    ax.set_xticks(x); ax.set_xticklabels(bench)
    ax.set_ylim(0, 100); ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "benchmark_gains.pdf")


# ---- Fig D: Maj@K vs Pass@K (illustrative trend) -----------------------
def fig_majk_passk():
    K = np.array([1, 4, 8, 16, 32, 64])
    x = np.log2(K)
    # schematic monotone curves reproducing Figure 7's qualitative message
    passk_inst = 82 + 12 * (1 - np.exp(-x / 2.2))
    passk_rl = 83 + 12 * (1 - np.exp(-x / 2.4))   # pass@k barely changes
    majk_inst = 82 + 3.0 * (1 - np.exp(-x / 3.0))
    majk_rl = 88 + 2.5 * (1 - np.exp(-x / 3.0))   # maj@k lifted by RL

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(x, passk_inst, "--", color=GREY, lw=2, label="Pass@K  Instruct")
    ax.plot(x, passk_rl, "--", color=ARM_BLUE, lw=2, label="Pass@K  RL")
    ax.plot(x, majk_inst, "-o", color=GREY, lw=2, ms=4, label="Maj@K  Instruct")
    ax.plot(x, majk_rl, "-o", color=ARM_BLUE, lw=2, ms=4, label="Maj@K  RL")
    ax.set_xticks(x); ax.set_xticklabels(K)
    ax.set_xlabel("K (samples per question)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("RL lifts Maj@K, not Pass@K  (schematic of the reported trend)")
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower right")
    ax.annotate("Pass@K nearly unchanged:\nRL adds no new solutions",
                (x[-1], passk_rl[-1]), (x[2], 97), fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_ylim(80, 100)
    save(fig, "majk_passk.pdf")


# ---- Fig E: method ladder (ablation ordering) --------------------------
def fig_method_ladder():
    methods = ["RFT\n(offline)", "Online RFT", "GRPO+OS", "GRPO+PS"]
    # schematic ordering (exact y-values not in text); increasing quality
    vals = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    cols = [GREY, ARM_ORANGE, ARM_BLUE, GREEN]
    ax.plot(range(4), vals, color="#bbbbbb", lw=2, zorder=1)
    for i, (m, c) in enumerate(zip(methods, cols)):
        ax.scatter(i, vals[i], s=260, color=c, zorder=3)
        ax.text(i, vals[i] + 0.25, m, ha="center", fontsize=10.5, weight="bold")
    ax.text(0.5, 0.5, "offline $\\rightarrow$ online", color=ARM_ORANGE, fontsize=10)
    ax.text(2.5, 3.5, "richer credit:\nreward magnitude\n+ per-step", color=GREEN,
            fontsize=9.5, ha="center")
    ax.set_ylim(0, 5); ax.set_xlim(-0.5, 3.6)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_title("Ablation ordering: online + richer gradient coefficient wins")
    save(fig, "method_ladder.pdf")


# ---- Fig F: KL estimator comparison ------------------------------------
def fig_kl_estimator():
    x = np.linspace(0.15, 3.0, 400)      # x = pi_ref / pi_theta
    k1 = -np.log(x)                       # naive log-ratio estimator
    k3 = x - np.log(x) - 1                # Schulman k3 (unbiased, >= 0)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.axhline(0, color="#999999", lw=1)
    ax.plot(x, k1, color=ARM_RED, lw=2.2, label=r"naive $k_1=-\log x$")
    ax.plot(x, k3, color=GREEN, lw=2.6, label=r"GRPO $k_3=x-\log x-1$")
    ax.axvline(1, color="#bbbbbb", ls=":", lw=1)
    ax.fill_between(x, k1, 0, where=(k1 < 0), color=ARM_RED, alpha=0.12)
    ax.set_xlabel(r"$x=\pi_{ref}/\pi_\theta$")
    ax.set_ylabel("per-token estimate")
    ax.set_title(r"Unbiased KL estimator $k_3\geq 0$ everywhere; naive $k_1$ goes negative")
    ax.legend(frameon=False, loc="upper center")
    ax.set_ylim(-1.6, 3.0)
    save(fig, "kl_estimator.pdf")


# ---- Fig G: process supervision (per-step credit assignment) -----------
def fig_process_supervision():
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    steps = [(ARM_BLUE, 0.8), (ARM_ORANGE, -0.3), (GREEN, 0.5)]
    ytok, width = 2.7, 2.2
    x0s = [0.3, 3.5, 6.7]
    centers = []
    for k, (col, rew) in enumerate(steps):
        x0 = x0s[k]; x1 = x0 + width; c = (x0 + x1) / 2; centers.append(c)
        ax.add_patch(FancyBboxPatch((x0, ytok - 0.32), width, 0.64,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc="#f4f6fb", ec=col, lw=1.8, zorder=2))
        for xi in np.linspace(x0 + 0.4, x1 - 0.4, 3):
            ax.add_patch(FancyBboxPatch((xi - 0.15, ytok - 0.17), 0.3, 0.34,
                         boxstyle="round,pad=0.01,rounding_size=0.03",
                         fc="white", ec=col, lw=1.0, zorder=3))
        ax.text(c, ytok + 0.52, f"Step {k + 1}", ha="center", fontsize=11,
                weight="bold", color=col)
        rcol = GREEN if rew >= 0 else ARM_RED
        ax.text(c, ytok + 1.02, rf"$\tilde r_{k + 1}={rew:+.1f}$", ha="center",
                fontsize=12, weight="bold", color=rcol)
    xend = x0s[2] + width
    ax.add_patch(FancyArrowPatch((centers[0], 1.7), (xend + 0.1, 1.7), arrowstyle="-|>",
                 mutation_scale=13, color=ARM_BLUE, lw=2.4, zorder=2))
    ax.text(0.3, 1.28, r"token in Step 1  $\Rightarrow\ \hat A=\tilde r_1+\tilde r_2+\tilde r_3=+1.0$",
            fontsize=10.5, color=ARM_BLUE, weight="bold", va="top")
    ax.add_patch(FancyArrowPatch((centers[2], 0.75), (xend + 0.1, 0.75), arrowstyle="-|>",
                 mutation_scale=13, color=GREEN, lw=2.4, zorder=2))
    ax.text(0.3, 0.33, r"token in Step 3  $\Rightarrow\ \hat A=\tilde r_3=+0.5$   "
            r"(only steps at or after it count)", fontsize=10.5, color=GREEN,
            weight="bold", va="top")
    ax.set_title("Process supervision: each token is credited with the rewards of the "
                 "steps ending at or after it", fontsize=11.5)
    ax.set_xlim(-0.2, 10.4); ax.set_ylim(-0.1, 4.2); ax.axis("off")
    save(fig, "process_supervision.pdf")


if __name__ == "__main__":
    fig_ppo_vs_grpo()
    fig_group_advantage()
    fig_benchmark_gains()
    fig_majk_passk()
    fig_method_ladder()
    fig_kl_estimator()
    fig_process_supervision()
    print("all GRPO figures done")
