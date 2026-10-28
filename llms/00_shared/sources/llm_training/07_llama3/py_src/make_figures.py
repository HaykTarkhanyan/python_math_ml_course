"""Figures for the Llama 3 deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
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


def fig_herd():
    # The three members of the herd -- parameter counts, log scale (Table 3).
    names = ["Llama 3 8B", "Llama 3 70B", "Llama 3 405B"]
    params = [8, 70, 405]  # billions
    colors = [GREY, ARM_BLUE, ARM_RED]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(x, params, color=colors, zorder=3, width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("parameters (billions, log scale)")
    ax.set_title("One recipe, three sizes -- all dense Transformers (no MoE)")
    ax.set_ylim(4, 900)
    for xi, p in zip(x, params):
        ax.text(xi, p * 1.12, f"{p}B", ha="center", fontsize=12, weight="bold")
    ax.text(2, 20, "flagship\n$\\approx$ compute-optimal\nfor the budget", ha="center",
            fontsize=9.5, color=ARM_RED)
    save(fig, "herd.pdf")


def fig_data_mix():
    # Final pre-training data mix (Section 3.1.2): real approximate shares.
    cats = ["General\nknowledge", "Math &\nreasoning", "Code", "Multilingual"]
    share = [50, 25, 17, 8]
    colors = [ARM_BLUE, GREEN, ARM_ORANGE, VIOLET]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(x, share, color=colors, zorder=3, width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("share of the 15.6T-token mix (%)")
    ax.set_title("The pre-training data mix (15.6T tokens, curated & de-duplicated)")
    ax.set_ylim(0, 58)
    for xi, s in zip(x, share):
        ax.text(xi, s + 1.0, f"{s}%", ha="center", fontsize=12, weight="bold")
    save(fig, "data_mix.pdf")


def fig_scaling_law():
    # Compute-optimal token count vs compute (Section 3.2.1). SCHEMATIC:
    # fit points in the measured range, dashed extrapolation to the flagship
    # budget, calibrated to the paper's reported endpoint (402B / 16.55T @ 3.8e25).
    alpha = 0.53
    C_fit = np.logspace(18.78, 22.0, 8)         # 6e18 .. 1e22, the measured budgets
    C_star = 3.8e25                             # flagship compute budget
    N_star = 16.55e12                           # paper's extrapolated optimal tokens
    A = N_star / C_star ** alpha                 # calibrate A to hit the endpoint
    N_fit = A * C_fit ** alpha
    rng = np.random.default_rng(509)
    N_noisy = N_fit * (1 + 0.06 * rng.standard_normal(len(C_fit)))

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(C_fit, N_fit, "-", color=ARM_BLUE, lw=2,
            label=r"power-law fit  $N(C)=A\,C^{0.53}$")
    ax.plot(C_fit, N_noisy, "o", color=ARM_BLUE, ms=6,
            label="compute-optimal models\n(IsoFLOPs minima)")
    C_ext = np.logspace(22.0, np.log10(C_star), 50)
    ax.plot(C_ext, A * C_ext ** alpha, "--", color=GREY, lw=1.8,
            label="extrapolation")
    ax.plot([C_star], [N_star], "*", color=ARM_RED, ms=20, zorder=5)
    ax.annotate("flagship budget $3.8\\times10^{25}$ FLOPs\n$\\to$ 402B params on 16.5T tokens\n$\\Rightarrow$ chose 405B",
                xy=(C_star, N_star), xytext=(6e21, 3.2e11),
                fontsize=9.5, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("compute-optimal training tokens")
    ax.set_title("Extrapolate a power law to pick the model size (schematic)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_ylim(3e9, 4e13)
    save(fig, "scaling_law.pdf")


def fig_downstream_pred():
    # Two-stage scaling law that predicts downstream ACCURACY, not just loss
    # (Section 3.2.1, Figure 4, ARC Challenge). SCHEMATIC of the method.
    C = np.logspace(19, 25.6, 200)
    # sigmoidal accuracy vs log-compute
    z = (np.log10(C) - 21.3) / 1.15
    acc = 0.5 + 0.47 / (1 + np.exp(-z))
    C_star = 3.8e25
    z_star = (np.log10(C_star) - 21.3) / 1.15
    acc_pred = 0.5 + 0.47 / (1 + np.exp(-z_star))
    acc_actual = acc_pred + 0.006          # paper: prediction slightly underestimates

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    mask = C <= 1e22
    ax.plot(C[mask], acc[mask], "-", color=ARM_BLUE, lw=2.4,
            label=r"fitted on small runs ($\leq 10^{22}$ FLOPs)")
    ax.plot(C[~mask], acc[~mask], "--", color=GREY, lw=1.8,
            label="extrapolation (4 orders of magnitude)")
    rng = np.random.default_rng(509)
    Cs = np.logspace(19.2, 21.9, 7)
    zs = (np.log10(Cs) - 21.3) / 1.15
    accs = 0.5 + 0.47 / (1 + np.exp(-zs)) + 0.008 * rng.standard_normal(len(Cs))
    ax.plot(Cs, accs, "o", color=ARM_BLUE, ms=6, label="scaling-law models")
    ax.plot([C_star], [acc_pred], "*", color=ARM_ORANGE, ms=20, zorder=5,
            label="predicted 405B")
    ax.plot([C_star], [acc_actual], "P", color=ARM_RED, ms=11, zorder=6,
            label="actual 405B")
    ax.annotate("prediction lands\nalmost exactly", xy=(C_star, acc_pred),
                xytext=(4e22, 0.72), fontsize=9.5, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xscale("log")
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("downstream accuracy (ARC Challenge)")
    ax.set_title("Predicting benchmark accuracy before training (schematic)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.3)
    ax.set_ylim(0.48, 1.0)
    save(fig, "downstream_pred.pdf")


def fig_posttrain_pipeline():
    # Post-training: reward model -> SFT (on rejection-sampled data) -> DPO,
    # repeated over 6 iterative rounds (Section 4.1, Figure 7). Clean box flow.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")

    def box(x, w, label, color, y=1.9, h=1.0, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.8, edgecolor=color, facecolor=color + "22"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color="#222", weight="bold")
        return x + w

    def arrow(x0, x1, y=2.4):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                     arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444"))

    xs = 0.2
    x = box(xs, 1.7, "Pre-trained\ncheckpoint", GREY); arrow(x, x + 0.35)
    x = box(x + 0.35, 1.6, "Reward\nmodel", ARM_BLUE); arrow(x, x + 0.35)
    x = box(x + 0.35, 1.9, "Rejection\nsampling", ARM_ORANGE); arrow(x, x + 0.35)
    x = box(x + 0.35, 1.5, "SFT", GREEN); arrow(x, x + 0.35)
    x = box(x + 0.35, 1.5, "DPO", VIOLET); arrow(x, x + 0.35)
    x = box(x + 0.35, 1.6, "Aligned\nmodel", ARM_RED)

    # iterative-rounds loop back arrow
    ax.add_patch(FancyArrowPatch((9.55, 1.9), (2.05, 1.55),
                 connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                 mutation_scale=16, lw=1.7, color=ARM_RED, linestyle="--"))
    ax.text(5.7, 0.42, "6 iterative rounds: fresh preference + SFT data each cycle",
            ha="center", fontsize=10, color=ARM_RED, style="italic")
    ax.text(5.7, 3.35, "No PPO / RLHF -- DPO is cheaper and more stable at 405B scale",
            ha="center", fontsize=10.5, color=ARM_BLUE, weight="bold")
    save(fig, "posttrain_pipeline.pdf")


def fig_benchmarks():
    # Table 2, exact numbers: Llama 3 405B vs GPT-4o vs Claude 3.5 Sonnet.
    bench = ["MMLU\n(0-shot CoT)", "IFEval", "HumanEval", "GSM8K", "MATH", "GPQA"]
    llama = [88.6, 88.6, 89.0, 96.8, 73.8, 51.1]
    gpt4o = [88.7, 85.6, 90.2, 96.1, 76.6, 53.6]
    claude = [88.3, 88.0, 92.0, 96.4, 71.1, 59.4]
    x = np.arange(len(bench)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.3))
    b1 = ax.bar(x - w, llama, w, label="Llama 3 405B", color=ARM_BLUE, zorder=3)
    b2 = ax.bar(x, gpt4o, w, label="GPT-4o", color=GREY, zorder=3)
    b3 = ax.bar(x + w, claude, w, label="Claude 3.5 Sonnet", color=ARM_ORANGE, zorder=3)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(bench, fontsize=10)
    ax.set_ylabel("score (%)")
    ax.set_title("Open 405B is on par with GPT-4o / Claude 3.5 -- wins some, loses some")
    ax.legend(frameon=False, loc="lower center", ncol=3, fontsize=10)
    ax.set_ylim(40, 108)
    save(fig, "benchmarks.pdf")


if __name__ == "__main__":
    fig_herd(); fig_data_mix(); fig_scaling_law(); fig_downstream_pred()
    fig_posttrain_pipeline(); fig_benchmarks()
    print("all Llama 3 figures done")
