"""Real figure for the dl_intro_history deck (Everyday-AI section).

Generates into ml/ch5_neural_networks/fig/:
  adoption_speed.pdf -- horizontal bar chart: how long popular apps took to reach
                        100 million users. ChatGPT (2 months) is the fastest standalone
                        app ever; Threads (5 days) rode Instagram's existing user base.
                        Data: Visual Capitalist / UBS (time-to-100M-users), accessed 2026.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/adoption_speed.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED, ARMORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("adoption_speed")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "adoption_speed.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# (app, months_to_100M, human_label); ordered longest -> shortest.
DATA = [
    ("Facebook",   54.0, "4.5 years"),
    ("YouTube",    49.0, "4 yr 1 mo"),
    ("Snapchat",   45.0, "3 yr 9 mo"),
    ("WhatsApp",   40.0, "3 yr 4 mo"),
    ("Instagram",  28.0, "2 yr 4 mo"),
    ("TikTok",      9.0, "9 months"),
    ("ChatGPT",     2.0, "2 months"),
    ("Threads*",    0.17, "5 days"),
]


def fig_adoption(log):
    apps = [d[0] for d in DATA]
    months = [d[1] for d in DATA]
    labels = [d[2] for d in DATA]

    def color(app):
        if app == "ChatGPT":
            return ARMRED
        if app.startswith("Threads"):
            return ARMORANGE
        return ARMBLUE

    colors = [color(a) for a in apps]

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    y = range(len(apps))
    bars = ax.barh(list(y), months, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(apps, fontsize=12)
    ax.invert_yaxis()  # Facebook (longest) on top, Threads (shortest) at bottom

    ax.bar_label(bars, labels=labels, padding=4, fontsize=11, fontweight="bold")

    ax.set_xlabel("Time to reach 100 million users (months)", fontsize=11)
    ax.set_xlim(0, 64)  # right padding so end-aligned labels do not clip
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("How long each app took to reach 100 million users", fontsize=14)
    fig.text(0.125, 0.01,
             "* Threads launched on Instagram's existing user base.  "
             "Source: Visual Capitalist / UBS.",
             fontsize=8.5, color="#666666")

    for a, m in zip(apps, months):
        log.info(f"{a}: {m} months")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIG_DIR / "adoption_speed.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_adoption(log)
    log.info("done")


if __name__ == "__main__":
    main()
