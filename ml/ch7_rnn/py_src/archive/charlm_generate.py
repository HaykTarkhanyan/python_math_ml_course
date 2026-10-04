"""ILLUSTRATIVE ANIM for the L21 "Road to Attention" deck (Section 3: the char-LSTM
generation demo).

Generates into ml/ch7_rnn/fig/:
  charlm_gen_0.pdf .. charlm_gen_4.pdf -- five flip-book frames shown via \\only<n>: a
                     TYPICAL progression of char-LSTM samples across training, from
                     random noise to a seeded, almost-literary continuation.
  charlm_loss.pdf -- a stylized SCHEMATIC loss curve (no real numbers -- axes carry no
                     unit ticks, just a smooth decreasing shape with the five checkpoint
                     positions marked).

SCOPE NOTE (instructor scope change, logged in L21_DECISIONS.md): the outline originally
called for fetching a public-domain Armenian corpus (Tumanyan/Charents, Wikisource) and
training a real char-LSTM. The instructor later ruled out ANY dataset/model download and
ANY real training run for this deck. This script instead hand-writes a TYPICAL
progression of what such a demo usually looks like (gibberish -> letter statistics ->
word-like fragments -> almost-literary -> seeded continuation) and renders it with the
same per-checkpoint ANIM format the outline specifies. Every frame says "illustrative /
typical progression, not measured" -- this is never presented as real model output.

Armenian rendering follows forward_pass_anim.py's pattern (Segoe UI, glyph-error
promotion, utf-8 logging, no Armenian inside mathtext).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/charlm_generate.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("error", message=".*missing from current font.*")

ARM_FONT = "Segoe UI"
BLUE, GREEN, GRAY, RED, ORANGE = "#0033A0", "#008C46", "#999999", "#D90012", "#F2A800"

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
DATA_FILE = HERE.parent / "data" / "armenian_line.txt"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l21_charlm_generate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "charlm_generate.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def seed_words(log) -> str:
    text = DATA_FILE.read_text(encoding="utf-8").strip()
    words = text.split()
    seed = " ".join(words[:3])  # "Ես այս ամենինչ"
    log.info(f"seed for the final checkpoint (opening of the Armenian line): {seed}")
    return seed


# Hand-written, clearly-labeled TYPICAL samples (never claimed as measured). Each entry:
# (checkpoint label, sample text, one-line stage description).
CHECKPOINTS = [
    ("step 0 (untrained)",
     "թզԽղրջ ևնիաուրծ խքացնեռ ընթղմս ցվատեղժ ոքրիմբ",
     "pure noise -- random characters from the alphabet"),
    ("step ~100",
     "անի որնա տեմա անու ևանիս որա նամի անե",
     "letter statistics emerge -- vowel-consonant rhythm, no real words yet"),
    ("step ~1,000",
     "նա ամեն լավ խոսք չէր գտնում մեջքով վարդի աչքով",
     "word-like fragments -- some real short words, invented ones in between"),
    ("step ~5,000",
     "Ես տեսնում եմ այս ամենը լուռ ու մտածում իմ մասին",
     "almost-literary -- real words, loose grammar, a plausible-sounding line"),
    ("final (seeded)",
     "Ես այս ամենինչ նայում եմ լուռ ու մտքումս խոսում եմ ինքս ինձ հետ",
     "seeded with the chapter's opening words -- the model continues the line"),
]


def draw_frame(step_idx, ckpt_label, sample, stage_note, log):
    fig = plt.figure(figsize=(11.0, 5.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.90, "a char-LSTM learns to write Armenian (typical progression)",
            ha="center", fontsize=13.5, fontweight="bold", color="black")
    ax.text(0.5, 0.80, ckpt_label, ha="center", fontsize=12, color=BLUE,
            fontweight="bold")

    # Progress ribbon: 5 stops, current one highlighted.
    n = len(CHECKPOINTS)
    xs = np.linspace(0.18, 0.82, n)
    for i, x in enumerate(xs):
        c = ORANGE if i == step_idx else "#DDDDDD"
        ax.plot([x], [0.68], marker="o", markersize=14 if i == step_idx else 9, color=c)
    ax.plot([xs[0], xs[-1]], [0.68, 0.68], color="#DDDDDD", lw=1.5, zorder=0)

    ax.add_patch(plt.Rectangle((0.08, 0.34), 0.84, 0.22, ec=GREEN, fc=GREEN + "10",
                                lw=1.4))
    ax.text(0.5, 0.45, sample, ha="center", va="center", fontsize=15,
            fontfamily=ARM_FONT, color="black", wrap=True)

    ax.text(0.5, 0.22, stage_note, ha="center", fontsize=11.5, color="black",
            style="italic")
    ax.text(0.98, 0.98, "ILLUSTRATIVE -- typical progression, not a measured run",
            ha="right", va="top", fontsize=8.5, color=RED, style="italic")
    ax.text(0.02, 0.98, f"checkpoint {step_idx + 1} of {n}", ha="left", va="top",
            fontsize=9, color=GRAY)

    out = FIG_DIR / f"charlm_gen_{step_idx}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out} ({ckpt_label}): {sample}")


def fig_loss_schematic(log):
    """A stylized, schematic loss curve -- NO real numbers. Axes are unitless/labeled
    'schematic'; the curve is a smooth monotone decay with markers at the 5 checkpoint
    positions used by the ANIM above, for visual continuity only."""
    x = np.linspace(0, 10, 300)
    y = 2.6 * np.exp(-0.55 * x) + 0.35  # illustrative shape only
    checkpoint_x = np.array([0.0, 1.2, 3.5, 6.5, 9.5])
    checkpoint_y = 2.6 * np.exp(-0.55 * checkpoint_x) + 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, y, color=BLUE, lw=2.4)
    ax.scatter(checkpoint_x, checkpoint_y, color=ORANGE, s=70, zorder=3,
               edgecolor="black", linewidth=0.6)
    for i, (cx, cy) in enumerate(zip(checkpoint_x, checkpoint_y)):
        ax.annotate(f"ckpt {i}", (cx, cy), textcoords="offset points", xytext=(6, 8),
                    fontsize=9, color=ORANGE)
    ax.set_xlabel("training progress (schematic)", fontsize=11)
    ax.set_ylabel("loss (schematic)", fontsize=11)
    ax.set_title("Illustrative loss curve -- shape only, no measured run", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "charlm_loss.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seed = seed_words(log)
    final_sample = CHECKPOINTS[-1][1]
    if not final_sample.startswith(seed):
        raise ValueError(
            f"final checkpoint sample must start with the Armenian line's opening "
            f"words ('{seed}'); got: '{final_sample}'"
        )
    for i, (label, sample, note) in enumerate(CHECKPOINTS):
        draw_frame(i, label, sample, note, log)
    fig_loss_schematic(log)
    log.info("done")


if __name__ == "__main__":
    main()
