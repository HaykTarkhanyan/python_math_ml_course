"""Real figure for the L20 RNN Foundations deck (Section 1: "The shuffle test").

Generates into ml/13_rnns/fig/:
  word_shuffle.pdf -- the chapter's running review sentence vs a fixed shuffle of its
                      words, with the two identical bag-of-words histograms below.
                      Demonstrates that an order-blind (bag-of-words) representation
                      literally cannot tell the two apart. Callback: the L16 pixel-shuffle
                      frame did the same demo on an image's pixels.

Run with the project venv:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/word_shuffle.py
"""

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
REVIEW_SENTENCE = "The pomegranates arrived fresh and sweet."
BLUE, RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_word_shuffle")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "word_shuffle.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def tokenize_display(sentence: str) -> list[str]:
    """Words as displayed (punctuation kept attached)."""
    return sentence.rstrip(".").split()


def tokenize_bow(sentence: str) -> list[str]:
    """Lowercased, punctuation-stripped tokens for the bag-of-words count."""
    return re.findall(r"[a-zA-Z']+", sentence.lower())


# Drawn at the size the figure gets on the slide (L20 shuffle frame: 0.70 column, ~3.85 in).
# The sentence rows use an axes whose data units are inches, so the word boxes are sized
# from the true 7.5 pt monospace character width. Redrawn 2026-10-03: the old 9.6 in canvas
# left ~4 pt text.
FIG_W, FIG_H = 3.85, 2.55
SENT_BOTTOM = 0.60                      # sentence axes: top 40% of the figure
MONO_PT = 7.5
CHAR_IN = 0.6 * MONO_PT / 72            # monospace advance width, inches


def box_width(word: str) -> float:
    """Box width in inches: the word at MONO_PT monospace plus padding."""
    return CHAR_IN * len(word) + 0.10


def draw_sentence_row(ax, y, words, label, origin, log):
    """One sentence as word boxes at height y (inches). origin[i] = the word's position in
    the original sentence (1-based), printed under each box when given."""
    x = 0.68
    ax.text(0.05, y, label, ha="left", va="center", fontsize=7, color="#666666")
    for i, word in enumerate(words):
        w = box_width(word)
        ax.add_patch(plt.Rectangle((x, y - 0.12), w, 0.24, ec=BLUE, fc=BLUE + "1A", lw=0.9))
        ax.text(x + w / 2, y, word, ha="center", va="center", fontsize=MONO_PT,
                family="monospace")
        if origin is not None:
            ax.text(x + w / 2, y - 0.17, f"was {origin[i]}", ha="center", va="top",
                    fontsize=7, color="#888888")
        x += w + 0.05
    if x > FIG_W:
        raise ValueError(f"{label} row is {x:.2f} in wide, figure is {FIG_W} in")
    log.info(f"{label}: {' '.join(words)}")


def fig_word_shuffle(log):
    words = tokenize_display(REVIEW_SENTENCE)
    n = len(words)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    shuffled_words = [words[i] for i in perm]

    bow_orig = tokenize_bow(REVIEW_SENTENCE)
    bow_shuf = [tokenize_bow(REVIEW_SENTENCE)[i] for i in perm]  # same multiset, reordered
    vocab = sorted(set(bow_orig))
    counts_orig = [bow_orig.count(v) for v in vocab]
    counts_shuf = [bow_shuf.count(v) for v in vocab]
    assert counts_orig == counts_shuf, "bag-of-words counts must be identical by construction"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    sent_h = FIG_H * (1 - SENT_BOTTOM)
    ax_s = fig.add_axes([0, SENT_BOTTOM, 1, 1 - SENT_BOTTOM])
    ax_s.set_xlim(0, FIG_W)
    ax_s.set_ylim(0, sent_h)
    ax_s.axis("off")
    ax_s.text(FIG_W / 2, sent_h - 0.03, "Same words, all meaning gone - identical histograms",
              ha="center", va="top", fontsize=8, fontweight="bold")
    draw_sentence_row(ax_s, sent_h - 0.33, words, "original", None, log)
    draw_sentence_row(ax_s, sent_h - 0.68, shuffled_words, "shuffled",
                      [int(i) + 1 for i in perm], log)

    for k, (counts, title, color) in enumerate([
        (counts_orig, "original: bag of words", BLUE),
        (counts_shuf, "shuffled: bag of words", RED),
    ]):
        ax = fig.add_axes([0.10 + k * 0.50, 0.255, 0.38, 0.27])
        bars = ax.bar(vocab, counts, color=color)
        ax.bar_label(bars, fontsize=7, padding=1)
        ax.set_ylim(0, max(counts_orig) + 0.6)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for t in ax.get_xticklabels():
            t.set_ha("right")
            t.set_rotation_mode("anchor")
        ax.set_yticks(range(0, max(counts_orig) + 1))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    out = FIG_DIR / "word_shuffle.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_word_shuffle(log)
    log.info("done")


if __name__ == "__main__":
    main()
