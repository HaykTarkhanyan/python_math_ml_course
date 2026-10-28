"""Figures for the LIMA deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
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


def _pref_bars(ax, baselines, wins, ties, losses, title):
    y = np.arange(len(baselines))[::-1]
    ax.barh(y, wins, color=GREEN, zorder=3, label="LIMA wins", height=0.62)
    ax.barh(y, ties, left=wins, color=GREY, zorder=3, label="Tie", height=0.62)
    ax.barh(y, losses, left=[w + t for w, t in zip(wins, ties)], color=ARM_RED,
            zorder=3, label="LIMA loses", height=0.62)
    for yi, (w, t) in enumerate(zip(wins, ties)):
        yy = y[yi]
        ax.text(w / 2, yy, f"{w}", ha="center", va="center", color="white", fontsize=9.5, weight="bold")
        eq = w + t
        ax.text(101, yy, f"{eq}% eq+", va="center", fontsize=9, color="#333333")
    ax.set_yticks(y); ax.set_yticklabels(baselines)
    ax.set_xlim(0, 100); ax.set_xlabel("% of 300 test prompts")
    ax.set_title(title)


def fig_human_pref():
    baselines = ["Alpaca 65B", "DaVinci003", "Bard", "Claude", "GPT-4"]
    wins = [53, 44, 33, 24, 18]; ties = [21, 21, 25, 22, 25]; losses = [26, 35, 42, 54, 57]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    _pref_bars(ax, baselines, wins, ties, losses,
               "Human preference: LIMA (1,000 examples) vs the field")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "human_pref.pdf")


def fig_gpt4_pref():
    baselines = ["Alpaca 65B", "DaVinci003", "Bard", "Claude", "GPT-4"]
    wins = [64, 54, 27, 14, 19]; ties = [19, 23, 26, 23, 15]; losses = [17, 23, 47, 63, 66]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    _pref_bars(ax, baselines, wins, ties, losses,
               "The trend replicates with GPT-4 as the judge")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "gpt4_pref.pdf")


def fig_quantity():
    # Figure 6: exact y-values NOT in text -> draw a flat plateau, qualitative.
    n = np.array([2, 4, 8, 16, 32])         # thousands of examples
    x = np.log2(n)
    plateau = np.array([3.55, 3.6, 3.58, 3.62, 3.6])   # essentially flat
    naive = 3.55 + 0.28 * (x - x[0])                    # hypothetical scaling law
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(x, naive, "--", color=GREY, lw=2, label="naive scaling-law expectation")
    ax.plot(x, plateau, "-o", color=ARM_BLUE, lw=2.6, ms=6, label="actual (filtered Stack Exchange)")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}K" for k in n])
    ax.set_xlabel("training examples")
    ax.set_ylabel("ChatGPT helpfulness (1-6)")
    ax.set_title("16x more data, no gain: quantity alone does not help")
    ax.set_ylim(3.2, 4.6)
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("plateau", (x[-1], plateau[-1]), (x[-2], 3.35), color=ARM_BLUE,
                fontsize=10, weight="bold")
    ax.text(0.98, 0.5, "y-axis qualitative:\nexact values not\nreported",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY, style="italic")
    save(fig, "quantity.pdf")


def fig_diversity_quality():
    labels = ["wikiHow\n(quality,\nlow diversity)", "Stack Exch.\nUNfiltered\n(diverse,\nlow quality)",
              "Stack Exch.\nFiltered\n(diverse +\nquality)"]
    vals = [3.49, 3.33, 3.83]
    colors = [ARM_ORANGE, GREY, GREEN]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(x, vals, color=colors, zorder=3, width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{v}", ha="center", fontsize=11, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("ChatGPT helpfulness (1-6)")
    ax.set_title("Quality + diversity beat either alone")
    ax.set_ylim(3.2, 4.0)
    ax.annotate("", xy=(2, 3.83), xytext=(1, 3.33),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.text(1.5, 3.66, "+0.5\nquality", color=GREEN, fontsize=9.5, ha="center", weight="bold")
    save(fig, "diversity_quality.pdf")


def fig_dialogue():
    cats = ["Zero-Shot\n(1,000)", "+30 Dialogue\n(1,030)"]
    excellent = [45.2, 76.1]; passq = [35.7, 21.7]; fail = [19.1, 2.2]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(x, excellent, color=GREEN, zorder=3, width=0.5, label="Excellent")
    ax.bar(x, passq, bottom=excellent, color=GREY, zorder=3, width=0.5, label="Pass")
    ax.bar(x, fail, bottom=[e + p for e, p in zip(excellent, passq)],
           color=ARM_RED, zorder=3, width=0.5, label="Fail")
    for xi in x:
        ax.text(xi, excellent[xi] / 2, f"{excellent[xi]:.0f}%", ha="center",
                va="center", color="white", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("% of dialogue turns"); ax.set_ylim(0, 100)
    ax.set_title("Multi-turn dialogue: 30 examples unlock it")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.annotate("+30\nexamples", xy=(0.75, 80), fontsize=10, color=GREEN, weight="bold")
    save(fig, "dialogue.pdf")


def fig_composition():
    src = ["SE STEM", "SE Other", "wikiHow", "WritingPrompts", "Nat. Instr.", "Authors"]
    n = [200, 200, 200, 150, 50, 200]
    colors = [ARM_BLUE, "#4a6fbf", GREEN, ARM_ORANGE, VIOLET, ARM_RED]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    wedges, _, autotext = ax.pie(n, colors=colors, autopct=lambda p: f"{int(round(p*10))}",
                                 startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, [f"{s} ({c})" for s, c in zip(src, n)],
              loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=9.5)
    ax.set_title("The 1,000 examples (~750k tokens)")
    save(fig, "composition.pdf")


if __name__ == "__main__":
    fig_human_pref(); fig_gpt4_pref(); fig_quantity()
    fig_diversity_quality(); fig_dialogue(); fig_composition()
    print("all LIMA figures done")
