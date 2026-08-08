"""Figures for L38 (Vision-Language-Action models), ml/ch15_vla.

NO MODEL IS TRAINED, DOWNLOADED OR RUN anywhere in this chapter (instructor decision,
2026-08-08). Every figure here is either a labelled conceptual diagram or a plot of
numbers published in a paper / a company post. Each number's source is in the docstring
of the function that plots it, and again in the deck's provenance block.

Generates into ml/ch15_vla/fig/:
  vla_loop.pdf          -- the observation-to-action loop (conceptual diagram)
  control_rates.pdf     -- published control frequencies, log scale
  model_timeline.pdf    -- parameter count vs release date, open vs closed weights
  data_gap.pdf          -- hours of training signal, log scale (two entries are estimates)
  gemini2_measured.pdf  -- Gemini Robotics 2's own published bars (company-reported)
  language_collapse.pdf -- clean vs perturbed success, three independent studies
  redvla_coupling.pdf   -- benign success and attack success rate, OpenVLA vs OpenVLA-OFT
  wilson_ci.pdf         -- exact Wilson score intervals vs trial count n

Run with the project venv, from the repo root:
    ./ma/Scripts/python.exe ml/ch15_vla/py_src/l38_vla_figs.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#555555"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "l38_vla_figs.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


log = build_logger()


def save(fig, name):
    out = FIG / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    size = out.stat().st_size
    if size < 2000:
        raise RuntimeError(f"{out} is only {size} bytes - the figure did not render")
    log.info(f"wrote {out} ({size} bytes)")


# ---------------------------------------------------------------------------------------
def fig_vla_loop():
    """Conceptual: what happens between a camera frame and a joint torque.

    Nothing measured here. The structure follows OpenVLA (Kim et al., 2024) and pi-0
    (Black et al., 2024): image tokens + instruction tokens + proprioceptive state into a
    pretrained VLM, an action head emitting a chunk of H actions, and a CONVENTIONAL
    low-level controller underneath that the VLA never replaces.
    """
    fig, ax = plt.subplots(figsize=(11.4, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fs=8.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="black", fontweight=weight)

    def arrow(x0, y0, x1, y1, color=GREY, lw=1.6, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=13, color=color, linewidth=lw,
                                     shrinkA=0, shrinkB=0))

    # Inputs
    box(1.5, 27.5, 17, 8.5, "scene camera\n+ wrist camera", "#eaf0fa", BLUE)
    box(1.5, 16.5, 17, 8.0, '"put the bowl\nin the sink"', "#fdf3e0", ORANGE)
    box(1.5, 6.0, 17, 7.5, "proprioception\n(joint angles, grip)", "#f0f0f0", GREY)

    # Backbone + head
    box(25.5, 12.5, 19, 20.0, "pretrained\nvision-language\nmodel\n\n(0.5B - 7B)",
        "#eaf0fa", BLUE, fs=9, weight="bold")
    box(50.5, 12.5, 17, 20.0, "action head\n\ndiscrete tokens,\nor a diffusion /\nflow head",
        "#e6f4ec", "#008C46", fs=8.5, weight="bold")

    # Output chunk
    box(73.0, 21.0, 25.0, 11.5, "chunk of $H$ actions\n"
        r"$a_t \ldots a_{t+H-1}$" "\n7-DoF end-effector deltas",
        "#fdf3e0", ORANGE, fs=8.5)
    box(73.0, 6.0, 25.0, 9.5, "low-level controller\n100 - 1000 Hz\n(NOT the VLA)",
        "#fbe6e8", RED, fs=8.5, weight="bold")

    for y in (31.7, 20.5, 9.7):
        arrow(18.5, y, 25.5, 22.5 if y != 22.5 else y)
    arrow(44.5, 22.5, 50.5, 22.5)
    arrow(67.5, 22.5, 73.0, 26.7)
    arrow(85.5, 21.0, 85.5, 15.5)

    # Feedback: the world changes, so go round again.
    ax.plot([98.0, 99.2, 99.2, 20.0, 20.0, 10.0], [10.7, 10.7, 39.5, 39.5, 38.0, 38.0],
            color=GREY, lw=1.4, ls="--")
    arrow(10.0, 38.0, 10.0, 36.3, color=GREY, lw=1.4)
    ax.text(58, 40.6, "the world has moved - capture again", ha="center", va="center",
            fontsize=8.5, color=GREY, style="italic")

    ax.text(35.0, 8.0, "one forward pass:\n50 - 200 ms", ha="center", va="center",
            fontsize=8.5, color=RED, fontweight="bold")
    ax.text(59.0, 8.0, "so re-querying every\nstep is impossible", ha="center", va="center",
            fontsize=8.5, color=RED)

    fig.suptitle("From two camera frames and a sentence to a joint command",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "vla_loop.pdf")


# ---------------------------------------------------------------------------------------
def fig_control_rates():
    """Published control frequencies. Log scale, because the spread is three decades.

    Sources: RT-1 3 Hz (Brohan et al., 2022); pi-0 up to 50 Hz (Black et al., 2024);
    Helix S2 7-9 Hz and S1 200 Hz (Figure AI blog, Feb 2025, company-reported, no paper);
    low-level joint controllers 100-1000 Hz (standard robotics, not a paper).
    """
    rows = [
        ("RT-1 policy queries\n(Brohan et al., 2022)", 3.0, None, BLUE),
        ("Helix System 2, 7B VLM\n(Figure, company-reported)", 7.0, 9.0, GREY),
        ("pi-0 action output\n(Black et al., 2024)", 50.0, None, BLUE),
        ("Helix System 1, 80M policy\n(Figure, company-reported)", 200.0, None, GREY),
        ("low-level joint controller\n(not the VLA)", 100.0, 1000.0, RED),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ypos = np.arange(len(rows))
    for y, (label, lo, hi, color) in zip(ypos, rows):
        if hi is None:
            ax.plot([lo], [y], "o", color=color, ms=11)
            ax.annotate(f"{lo:g} Hz", (lo, y), textcoords="offset points", xytext=(13, 0),
                        va="center", fontsize=10, fontweight="bold", color=color)
        else:
            ax.plot([lo, hi], [y, y], color=color, lw=6, solid_capstyle="round", alpha=0.85)
            ax.annotate(f"{lo:g} - {hi:g} Hz", (hi, y), textcoords="offset points",
                        xytext=(13, 0), va="center", fontsize=10, fontweight="bold",
                        color=color)

    ax.axvspan(5.0, 20.0, color=ORANGE, alpha=0.16, zorder=0)
    ax.text(10.0, len(rows) - 0.35, "a 50-200 ms forward pass,\nre-queried every step",
            ha="center", va="center", fontsize=8.5, color="#8a6000", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(1.5, 4000)
    ax.set_ylim(-0.7, len(rows) - 0.15)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("control rate (Hz, log scale)", fontsize=10)
    ax.set_title("The latency wall: the model is 1-2 decades too slow for contact\n"
                 "Action chunking and dual-system designs both exist to close this gap",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.tight_layout()
    save(fig, "control_rates.pdf")


# ---------------------------------------------------------------------------------------
def fig_model_timeline():
    """Parameter count vs release date. The point: it went DOWN, unlike LLMs.

    All sizes from the paper or the releasing lab's own page (see the deck provenance).
    Gemini Robotics 1.0 / 1.5 / 2 are omitted because their sizes are undisclosed - which
    is itself annotated on the chart.
    """
    # (label, decimal year, params in billions, open weights)
    models = [
        ("RT-1",        2022.95, 0.035, True),
        ("RT-2",        2023.57, 55.0,  False),
        ("Octo",        2024.37, 0.093, True),
        ("OpenVLA",     2024.45, 7.0,   True),
        ("pi-0",        2024.83, 3.3,   True),
        ("Helix",       2025.12, 7.08,  False),
        ("GR00T N1",    2025.21, 2.2,   True),
        ("SmolVLA",     2025.42, 0.45,  True),
        ("pi-0.6",      2025.88, 5.0,   False),
        ("Dream-VLA",   2025.99, 7.0,   True),
        ("LingBot-VLA 2.0", 2026.52, 6.0, True),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for label, year, params, is_open in models:
        color = BLUE if is_open else RED
        marker = "o" if is_open else "s"
        ax.plot([year], [params], marker, color=color, ms=10,
                markerfacecolor=color if is_open else "white",
                markeredgecolor=color, markeredgewidth=2.0, zorder=3)

    offsets = {
        "RT-1": (0, 13), "RT-2": (0, 14), "Octo": (-4, -20), "OpenVLA": (-2, 13),
        "pi-0": (10, -6), "Helix": (2, 13), "GR00T N1": (12, -4), "SmolVLA": (-6, -20),
        "pi-0.6": (-30, 6), "Dream-VLA": (4, 11), "LingBot-VLA 2.0": (-42, -20),
    }
    for label, year, params, is_open in models:
        dx, dy = offsets[label]
        ax.annotate(label, (year, params), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color=BLUE if is_open else RED, fontweight="bold")

    ax.axhspan(0.4, 8.0, color=ORANGE, alpha=0.13, zorder=0)
    # "everything" was false on this chart's own data - Octo (2024, 0.093B) sits below the
    # band. Say "most" and let the visible exception stand.
    ax.text(2023.62, 1.9, "after RT-2, most models\nlive in this band: 0.45B - 7B",
            fontsize=9, color="#8a6000", fontweight="bold", ha="center")

    ax.set_yscale("log")
    ax.set_ylim(0.02, 130)
    ax.set_xlim(2022.6, 2026.95)
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_xticklabels(["2023", "2024", "2025", "2026"])
    ax.set_ylabel("parameters (billions, log scale)", fontsize=10)
    ax.set_title("Model size went down, not up\n"
                 "Filled circles = open weights, hollow squares = closed", fontsize=11)
    ax.text(2024.35, 0.026,
            "Gemini Robotics 1.0 / 1.5 / 2 are not plotted: sizes undisclosed",
            fontsize=8, color=GREY, style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    save(fig, "model_timeline.pdf")


# ---------------------------------------------------------------------------------------
def fig_data_gap():
    """Hours of training signal, one unit, log scale.

    DROID 350 h and pi-0 ~10,000 h are from the papers. LingBot-VLA 2.0's ~50,000 h of
    robot data is from a secondary source. The top two bars are INDUSTRY ESTIMATES, not
    measurements, and are labelled as such on the chart itself.
    """
    rows = [
        ("DROID, 2024\nthe careful academic dataset", 350, False),
        ("pi-0's whole training set, 2024\n(proprietary + open)", 10_000, False),
        ("LingBot-VLA 2.0 robot data, 2026\n(secondary source)", 50_000, False),
        ("ALL robot manipulation data\nthat exists", 300_000, True),
        ("internet video", 1_000_000_000, True),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    ypos = np.arange(len(rows))
    colors = [ORANGE if est else BLUE for _, _, est in rows]
    bars = ax.barh(ypos, [r[1] for r in rows], color=colors, height=0.6)
    for y, (label, val, est) in zip(ypos, rows):
        txt = f"{val:,} h" + ("   (estimate)" if est else "")
        ax.annotate(txt, (val, y), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=9.5, fontweight="bold",
                    color="#8a6000" if est else BLUE)

    ax.set_xscale("log")
    ax.set_xlim(100, 3e11)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("hours of recorded data (log scale)", fontsize=10)
    ax.set_title("There is no internet of robot data\n"
                 "Thousands of times smaller than video - and it cannot be scraped",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.text(0.5, -0.07,
             "Orange bars are industry estimates (Bessemer; the gogoduck912 robotics "
             "scaling-law analysis), not measurements.",
             ha="center", fontsize=8.2, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "data_gap.pdf")


# ---------------------------------------------------------------------------------------
def fig_gemini2_measured():
    """Gemini Robotics 2's own published success rates (DeepMind blog, 30 July 2026).

    Company-reported on tasks the company chose, with an unpublished protocol and no n
    and no intervals. Plotted anyway because they are numbers, and because the spread
    inside a single "dexterity" category is the honest part.
    """
    groups = [
        ("General whole-body\nmanipulation\n(Apollo, Inspire hands)",
         [("pick from floor", 45.7), ("pick from table", 68.4), ("pick from shelf", 76.3)]),
        ("Multi-finger dexterity\n(Apollo, SharpaWave hands)",
         [("dustpan", 32), ("screw bulb", 36), ("ziplock", 40), ("tie trash bag", 44),
          ("unscrew bulb", 92)]),
        ("Gripper dexterity\n(Franka Duo)",
         [("general pick+place", 74.2), ("diverse tool kitting", 78.9),
          ("precise insertion", 89.6)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9),
                            gridspec_kw={"width_ratios": [3, 5, 3]})
    for ax, (title, items) in zip(axes, groups):
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = [RED if v < 50 else (ORANGE if v < 75 else BLUE) for v in vals]
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.66)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8, rotation=32, ha="right")
        ax.set_ylim(0, 108)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("success rate (%)", fontsize=9.5)
        else:
            ax.set_yticklabels([])

    fig.suptitle("The frontier, in the vendor's own numbers - Gemini Robotics 2, 30 July 2026",
                 fontsize=11.5, y=1.03)
    fig.text(0.5, -0.14,
             "Company-reported, self-chosen tasks, protocol and trial counts not published. "
             "Screwing a bulb IN: 36%. Unscrewing one: 92%.",
             ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "gemini2_measured.pdf")


# ---------------------------------------------------------------------------------------
def fig_language_collapse():
    """Percentage points of task success lost when only the WORDING changes.

    One unit throughout (percentage points of success), because the three studies report
    different things and a shared axis is the only honest way to put them together.

    LIBERO-Para (Kim et al., arXiv 2603.28301): 22-52 point degradation across seven
      configurations of four families, 0.6B-7.5B, under plain paraphrase.
    DAERT (Tong et al., arXiv 2604.05595): 93.33% -> 5.85% on pi-0 and OpenVLA under
      RL-searched rephrasings. 87.5 points.
    LIBERO-PRO (Zhou et al., arXiv 2510.03827): >90% -> 0.0%. Flagged on the chart,
      because LIBERO-PRO perturbs four axes at once, not wording alone.
    """
    rows = [
        ("LIBERO-Para (Kim et al., 2026)\nordinary paraphrase, 7 configs 0.6B-7.5B",
         22.0, 52.0, "22 to 52 points, every model tested"),
        ("DAERT (Tong et al., 2026)\nRL-searched rephrasings, pi-0 + OpenVLA",
         87.5, None, "93.3%  ->  5.9%"),
        ("LIBERO-PRO (Zhou et al., 2026)\nfour axes at once, wording among them",
         90.0, None, ">90%  ->  0.0%"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ypos = np.arange(len(rows))
    for y, (label, lo, hi, note) in zip(ypos, rows):
        if hi is None:
            ax.barh([y], [lo], color=RED, height=0.5)
            ax.annotate(note, (lo, y), textcoords="offset points", xytext=(9, 0),
                        va="center", fontsize=10, fontweight="bold", color=RED)
        else:
            ax.barh([y], [hi - lo], left=lo, color=ORANGE, height=0.5)
            ax.annotate(note, (hi, y), textcoords="offset points", xytext=(9, 0),
                        va="center", fontsize=10, fontweight="bold", color="#8a6000")

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("task success lost (percentage points)", fontsize=10)
    ax.set_title("Same robot, same scene, same request - different words\n"
                 "Three independent groups, one conclusion", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.text(0.5, -0.07,
             "Different papers, different models, different perturbation strengths. The "
             "shared quantity is points of success lost, not a ranking.",
             ha="center", fontsize=8.2, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "language_collapse.pdf")


# ---------------------------------------------------------------------------------------
def fig_redvla_coupling():
    """RedVLA (Zhang et al., arXiv 2604.22591, 24 Apr 2026), Table: OpenVLA vs OpenVLA-OFT.

    Benign success 76.5% -> 97.1% (+20.6 points). Attack success rate 64.9% -> 90.5%
    (+25.6 points). The better policy is the more attackable one.
    """
    models = ["OpenVLA", "OpenVLA-OFT"]
    benign = [76.5, 97.1]
    asr = [64.9, 90.5]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar(x - w / 2, benign, w, color=BLUE, label="benign task success")
    b2 = ax.bar(x + w / 2, asr, w, color=RED, label="attack success rate (RedVLA)")
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=10.5, fontweight="bold", color=BLUE)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=10.5, fontweight="bold", color=RED)

    ax.annotate("", xy=(1 - w / 2, 97.1), xytext=(0 - w / 2, 76.5),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8, ls="--"))
    ax.annotate("+20.6 pts", (0.5 - w / 2, 88.5), fontsize=10, color=BLUE,
                fontweight="bold", ha="center", va="bottom")
    ax.annotate("", xy=(1 + w / 2, 90.5), xytext=(0 + w / 2, 64.9),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, ls="--"))
    # The dashed red trend line passes ~77.7 at this x, and the gap between the bar groups is
    # empty below 64.9 - so drop the label well clear rather than nudging it.
    ax.annotate("+25.6 pts", (0.5 + w / 2, 58.0), fontsize=10, color=RED,
                fontweight="bold", ha="center", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_ylabel("percent", fontsize=10)
    # Neutral descriptor only. The frame title ("The better policy is the more attackable
    # one") carries the argument; repeating it here said the same thing twice on one slide.
    ax.set_title("OpenVLA vs OpenVLA-OFT under RedVLA evaluation", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "redvla_coupling.pdf")


# ---------------------------------------------------------------------------------------
def wilson(p_hat, n, z=1.959963985):
    """Exact Wilson score interval. No approximation, no simulation."""
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def fig_wilson_ci():
    """How many real-robot trials you need before a 10-point difference means anything.

    Pure arithmetic: 95% Wilson score intervals for observed 70% and 80% success.
    Motivated by PhAIL (Arkhangelskiy, arXiv 2605.29710), whose abstract states that
    real-world VLA evaluation "still rests on binary success rate at a fixed timeout with
    N <= 25 rollouts per condition, almost always without confidence intervals or paired
    statistical comparison".
    """
    ns = np.arange(5, 401, 1)
    lo70, hi70 = wilson(0.70, ns)
    lo80, hi80 = wilson(0.80, ns)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.fill_between(ns, lo70 * 100, hi70 * 100, color=BLUE, alpha=0.22,
                    label="observed 70% success")
    ax.fill_between(ns, lo80 * 100, hi80 * 100, color=RED, alpha=0.22,
                    label="observed 80% success")
    ax.plot(ns, np.full_like(ns, 70.0, dtype=float), color=BLUE, lw=1.6)
    ax.plot(ns, np.full_like(ns, 80.0, dtype=float), color=RED, lw=1.6)

    a, b = wilson(0.70, 20)
    c, d = wilson(0.80, 20)
    log.info(f"n=20: 70% -> [{a * 100:.1f}, {b * 100:.1f}]; "
             f"80% -> [{c * 100:.1f}, {d * 100:.1f}]")
    ax.axvline(20, color=ORANGE, lw=2.2, ls="--")
    ax.annotate(f"n = 20, typical published trial count\n"
                f"70% means [{a * 100:.0f}%, {b * 100:.0f}%]\n"
                f"80% means [{c * 100:.0f}%, {d * 100:.0f}%]\n"
                f"the two are indistinguishable",
                xy=(20, 34), xytext=(46, 27), fontsize=9.5, color="#8a6000",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))

    # Non-overlapping intervals is a CONSERVATIVE criterion. A pooled two-proportion
    # z-test calls the same difference much sooner, so report both rather than the
    # flattering one: n_test = p_bar(1-p_bar)*2*(z/delta)^2.
    first_clean = next((int(n) for n in ns if wilson(0.70, n)[1] < wilson(0.80, n)[0]), None)
    if first_clean is None:
        raise RuntimeError("intervals never separated - widen the n range")
    z = 1.959963985
    p_bar, delta = 0.75, 0.10
    n_test = int(np.ceil(p_bar * (1 - p_bar) * 2 * (z / delta) ** 2))
    log.info(f"intervals stop overlapping at n = {first_clean}; "
             f"two-proportion z-test needs n = {n_test} per arm")
    ax.axvline(first_clean, color="#008C46", lw=2.0, ls=":")
    ax.annotate(f"n = {first_clean}: even the intervals\nthemselves stop overlapping",
                xy=(first_clean, 90), xytext=(first_clean - 128, 96),
                fontsize=9.5, color="#008C46", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#008C46", lw=1.6))

    ax.set_xlim(5, 400)
    ax.set_ylim(25, 108)
    ax.set_xlabel("trials per condition, n", fontsize=10)
    ax.set_ylabel("95% Wilson interval on the success rate (%)", fontsize=10)
    ax.set_title("Why most published real-robot comparisons are noise", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, ls=":")
    fig.text(0.5, -0.05,
             # Ratio computed against the n = 20 annotated on this same chart, not against
             # the N <= 25 figure quoted two slides earlier - a reader checking the claim
             # uses the number in front of them.
             f"Exact 95% Wilson intervals. A pooled two-proportion z-test is less "
             f"conservative and separates 70% from 80% at n = {n_test} per arm - still "
             f"{n_test / 20:.0f}x the n = 20 annotated above.",
             ha="center", fontsize=8.2, color=GREY, style="italic")
    fig.tight_layout()
    save(fig, "wilson_ci.pdf")


# ---------------------------------------------------------------------------------------
# Slow-down diagrams. Added 2026-08-08 (instructor: expand, explain slower). Three of this
# deck's hardest ideas lived inside table cells. These are conceptual diagrams - nothing is
# measured, trained or run.

def _panel(ax, x, y, w, h, text, color, fs=8, alpha=0.16):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, color="0.4", lw=1.2, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls))


def fig_action_multimodality():
    """Why averaging demonstrations is not a neutral act.

    Two humans route around an obstacle in opposite directions. Both are correct. Their
    average goes straight through it. This is the entire argument for a diffusion or flow
    head instead of a head trained with mean squared error, and in the deck it was one
    sentence inside a table cell.
    """
    t = np.linspace(0, 1, 120)
    start, goal = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    obstacle = np.array([2.0, 0.0])

    def route(sign):
        mid = obstacle + np.array([0.0, sign * 1.35])
        pts = (1 - t)[:, None] ** 2 * start + 2 * (1 - t)[:, None] * t[:, None] * mid \
            + t[:, None] ** 2 * goal
        return pts

    up, down = route(+1), route(-1)
    mean = 0.5 * (up + down)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), sharex=True, sharey=True)
    for ax in axes:
        ax.add_patch(plt.Circle(obstacle, 0.55, fc="0.35", ec="0.2", alpha=0.85, zorder=3))
        ax.scatter(*start, s=45, color="0.25", zorder=4)
        ax.scatter(*goal, s=45, marker="*", color="#008C46", zorder=4)
        ax.set_xlim(-0.6, 4.6); ax.set_ylim(-2.1, 2.1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    axes[0].plot(*up.T, lw=2.4, color=BLUE)
    axes[0].set_title("demonstration 1: go over", fontsize=9.5, color=BLUE)
    axes[1].plot(*down.T, lw=2.4, color=ORANGE)
    axes[1].set_title("demonstration 2: go under", fontsize=9.5, color="#B07800")
    axes[2].plot(*up.T, lw=1.0, color=BLUE, alpha=0.35)
    axes[2].plot(*down.T, lw=1.0, color=ORANGE, alpha=0.35)
    axes[2].plot(*mean.T, lw=2.8, color=RED)
    axes[2].set_title("their average: straight through it", fontsize=9.5, color=RED)
    axes[2].text(2.0, 0.95, "collision", ha="center", fontsize=9, color=RED,
                 fontweight="bold")

    fig.suptitle("Both demonstrations are correct. A head trained to predict the MEAN action "
                 "learns the one behaviour that is not.", fontsize=9.5, y=0.02)
    fig.tight_layout()
    out = FIG / "action_multimodality.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_compounding_error():
    """Why a robot cannot be trained like a language model: errors move the world."""
    rng = np.random.default_rng(SEED)
    T = 60
    demo = np.c_[np.linspace(0, 6, T), 0.9 * np.sin(np.linspace(0, 2.4, T))]

    def rollout(chunk, sigma=0.06):
        """Each decision injects a small HEADING error that then persists.

        Modelled as a rotation of the commanded step rather than an additive offset: an
        offset re-applied every step compounds quadratically and flies off the canvas, which
        is not what drift looks like. Accumulated heading is a random walk, so deviation grows
        with the square root of the number of DECISIONS - exactly the quantity chunking cuts.
        """
        out, heading = [demo[0].copy()], 0.0
        for k in range(1, T):
            if (k - 1) % chunk == 0:               # the policy was asked for a new action
                heading += rng.normal(0, sigma)
            c, s = np.cos(heading), np.sin(heading)
            dx, dy = demo[k] - demo[k - 1]
            out.append(out[-1] + np.array([c * dx - s * dy, s * dx + c * dy]))
        return np.array(out)

    # Draw MANY rollouts per panel, not one. A single draw is not representative when the
    # chunked arm only makes 5 random decisions - the first version of this figure happened to
    # show chunking drifting MORE, which is the opposite of the point it was captioned with.
    n_show, n_stat = 14, 400
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=True)
    for ax, chunk, title, color in [
            (axes[0], 1, "ask every step: 60 chances to drift", RED),
            (axes[1], 12, "action chunking, $H=12$: 5 chances", "#008C46")]:
        for _ in range(n_show):
            ax.plot(*rollout(chunk).T, lw=1.0, color=color, alpha=0.45)
        gaps = [np.linalg.norm(rollout(chunk)[-1] - demo[-1]) for _ in range(n_stat)]
        ax.plot(*demo.T, lw=2.8, color="0.3", zorder=5)
        ax.scatter(*demo[0], s=40, color="0.2", zorder=6)
        ax.set_title(f"{title}\nmean final deviation over {n_stat} runs: "
                     f"{np.mean(gaps):.2f}", fontsize=9.5, color=color)
        ax.set_xticks([]); ax.set_yticks([])
        log.info(f"compounding-error ({title}): mean final gap {np.mean(gaps):.2f} "
                 f"over {n_stat} rollouts (schematic units)")
    axes[0].plot([], [], lw=2.8, color="0.3", label="the demonstration")
    axes[0].plot([], [], lw=1.4, color=RED, label="rollouts")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Schematic, 14 rollouts drawn per panel. An error does not just cost "
                 "accuracy - it moves the arm somewhere the demonstrations never went.",
                 fontsize=9, y=0.02)
    fig.tight_layout()
    out = FIG / "compounding_error.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_three_action_heads():
    """Discrete bins vs compressed tokens vs a continuous head, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    specs = [
        ("1. discrete bins", BLUE, "RT-2, OpenVLA",
         [("action dimension", ORANGE), ("cut into 256 bins", BLUE),
          ("each bin = a TOKEN\n(reuse rare text tokens)", BLUE),
          ("the language head,\nunchanged", "#7832A0")],
         "7 dims x 50 steps\n= 350 tokens. Slow."),
        ("2. compressed tokens", RED, "FAST",
         [("the whole chunk", ORANGE), ("cosine transform\nalong time", RED),
          ("drop small high-\nfrequency terms, then BPE", RED),
          ("the language head,\nunchanged", "#7832A0")],
         "~10x fewer tokens.\nThe JPEG trick."),
        ("3. continuous head", "#008C46", "Octo, pi-0, GR00T N1",
         [("backbone features", ORANGE), ("a SEPARATE head", "#008C46"),
          ("diffusion or\nflow matching (ch10)", "#008C46"),
          ("real numbers out,\nno bins at all", "#7832A0")],
         "Can represent BOTH\nways around the mug."),
    ]
    for ax, (title, color, who, boxes, note) in zip(axes, specs):
        ax.set_xlim(0, 3); ax.set_ylim(0, 4.7); ax.axis("off")
        ax.set_title(f"{title}\n{who}", fontsize=10, color=color)
        for i, (label, c) in enumerate(boxes):
            y = 3.65 - i * 0.85
            _panel(ax, 0.25, y, 2.5, 0.62, label, c, fs=7.5)
            if i < len(boxes) - 1:
                _arrow(ax, 1.5, y, 1.5, y - 0.23)
        ax.text(1.5, 0.2, note, fontsize=8.5, ha="center", color="0.25", style="italic")
    fig.suptitle("The same fork as chapter 10 and chapter 13: discretize and reuse the "
                 "language head, or model the numbers directly.", fontsize=9.5, y=1.0)
    fig.tight_layout()
    out = FIG / "three_action_heads.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    np.random.seed(SEED)
    fig_vla_loop()
    fig_control_rates()
    fig_model_timeline()
    fig_data_gap()
    fig_gemini2_measured()
    fig_language_collapse()
    fig_redvla_coupling()
    fig_wilson_ci()
    fig_action_multimodality()
    fig_compounding_error()
    fig_three_action_heads()
    log.info("done - 11 figures")
