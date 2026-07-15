"""ILLUSTRATIVE ANIM for the L21 "Road to Attention" deck (Section 3: language
modeling mechanics).

Generates into ml/ch7_rnn/fig/:
  charlm_anim_0.pdf .. charlm_anim_5.pdf -- six flip-book frames shown via \\only<n> on
                     one beamer frame: feeding the course's cheese-joke word Պանիր
                     (Պ-ա-ն-ի-ր) one character at a time, and showing -- at each step --
                     a next-character probability bar chart (top-5 candidates).

SCOPE NOTE (instructor scope change, logged in L21_DECISIONS.md): no char-LSTM is
trained for this deck. The probability bars are ILLUSTRATIVE -- hand-chosen, plausible
values (e.g. vowels dominate right after a consonant), NOT measured from a real model.
Every frame is labeled "illustrative" so nobody mistakes these for real inference
output. The mechanics being taught (one character in, a full distribution over the
next character out, same net at every step) do not depend on the numbers being real.

Word choice (interview-locked, hardcoded here -- there is no data file for this word;
py_src/data/armenian_line.txt is a separate chapter constant that does not contain it):
  Պանիր (Armenian for "cheese"), split into Պ-ա-ն-ի-ր.

Armenian rendering follows forward_pass_anim.py's pattern: Segoe UI (Armenian
coverage confirmed on this machine), "Glyph ... missing from current font" warnings
promoted to hard errors, utf-8 console/log reconfiguration, and no Armenian text mixed
into mathtext `$...$` strings.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/charlm_anim.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# FAIL LOUDLY: a missing Armenian glyph must crash the script, never render as tofu.
warnings.filterwarnings("error", message=".*missing from current font.*")

ARM_FONT = "Segoe UI"
BLUE, GREEN, GRAY, RED, ORANGE = "#0033A0", "#008C46", "#999999", "#D90012", "#F2A800"

SEED = 509
WORD = "Պանիր"  # course cheese joke, interview-locked 2026-07-13

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l21_charlm_anim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "charlm_anim.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# Each step: (context typed so far, [(candidate_char, illustrative_prob), ...], note)
# Probabilities are hand-chosen and need not sum to 1 (an implicit "other" mass fills
# the remainder) -- they are illustrative of PLAUSIBLE model behavior, not measurements.
STEPS = [
    {
        "ctx": "",
        "bars": [],
        "note": "no character fed yet -- the net has only its initial state",
    },
    {
        "ctx": "Պ",
        "bars": [("ա", 0.34), ("ո", 0.16), ("ի", 0.12), ("ե", 0.09), ("այ", 0.06)],
        "note": "after a consonant, vowels dominate the guess",
    },
    {
        "ctx": "Պա",
        "bars": [("ն", 0.29), ("տ", 0.13), ("ր", 0.10), ("կ", 0.08), ("ս", 0.07)],
        "note": "ն is a common continuation after Պա-",
    },
    {
        "ctx": "Պան",
        "bars": [("ի", 0.31), ("դ", 0.11), ("ն", 0.09), ("ց", 0.08), ("ա", 0.07)],
        "note": "ի keeps the word on track toward Պանիր",
    },
    {
        "ctx": "Պանի",
        "bars": [("ր", 0.42), ("կ", 0.10), ("ն", 0.07), ("ց", 0.06), ("տ", 0.05)],
        "note": "ր is now the clear favorite -- Պանիր is nearly complete",
    },
    {
        "ctx": "Պանիր",
        "bars": [(" ", 0.22), ("ը", 0.15), (",", 0.10), ("ի", 0.07), ("ն", 0.06)],
        "note": "the word is complete; the model now bets on what comes AFTER it",
    },
]


def draw_frame(step_idx, step, log):
    fig = plt.figure(figsize=(10.5, 5.8))
    head_ax = fig.add_axes([0, 0.72, 1, 0.26])  # word display strip, on its own axes
    head_ax.axis("off")
    head_ax.set_xlim(0, 1); head_ax.set_ylim(0, 1)

    head_ax.text(0.5, 0.85, "feeding Պանիր one character at a time", ha="center",
                  fontsize=13, color="black", fontfamily=ARM_FONT, fontweight="bold")
    x0 = 0.5 - 0.06 * (len(WORD) - 1)
    for i, ch in enumerate(WORD):
        typed = i < len(step["ctx"])
        color = BLUE if typed else "#CCCCCC"
        head_ax.text(x0 + 0.12 * i, 0.30, ch, ha="center", fontsize=22, color=color,
                      fontfamily=ARM_FONT, fontweight="bold")

    if step["bars"]:
        chars = [c for c, _ in step["bars"]]
        probs = [p for _, p in step["bars"]]
        y_pos = range(len(chars))
        bar_ax = fig.add_axes([0.30, 0.14, 0.46, 0.48])
        bars = bar_ax.barh(list(y_pos), probs, color=ORANGE, height=0.6)
        bar_ax.set_yticks(list(y_pos))
        bar_ax.set_yticklabels(chars, fontsize=15, fontfamily=ARM_FONT)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(0, 0.5)
        bar_ax.set_xlabel("illustrative probability (top 5, hand-chosen)", fontsize=9.5)
        for b, p in zip(bars, probs):
            bar_ax.text(p + 0.01, b.get_y() + b.get_height() / 2, f"{p:.2f}",
                        va="center", fontsize=9.5)
        bar_ax.spines[["top", "right"]].set_visible(False)
    else:
        fig.text(0.5, 0.42, "(distribution appears once the first character is fed)",
                 ha="center", fontsize=11, color=GRAY, style="italic")

    fig.text(0.5, 0.045, step["note"], ha="center", fontsize=10.5, color="black",
              fontfamily=ARM_FONT)
    fig.text(0.98, 0.98, "ILLUSTRATIVE -- not measured from a trained model",
              ha="right", va="top", fontsize=8.5, color=RED, style="italic")
    fig.text(0.02, 0.98, f"step {step_idx + 1} of {len(STEPS)}", ha="left", va="top",
              fontsize=9, color=GRAY)

    out = FIG_DIR / f"charlm_anim_{step_idx}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out} (ctx='{step['ctx']}', bars={step['bars']})")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"word: {WORD} (letters: {list(WORD)})")
    assert list(WORD) == ["Պ", "ա", "ն", "ի", "ր"], f"unexpected split: {list(WORD)}"
    for i, step in enumerate(STEPS):
        draw_frame(i, step, log)
    log.info("done")


if __name__ == "__main__":
    main()
