"""Real figure for the L20 RNN Foundations deck (Section 1: "The shuffle test").

Generates into ml/ch7_rnn/fig/:
  word_shuffle.pdf -- the chapter's running review sentence vs a fixed shuffle of its
                      words, with the two identical bag-of-words histograms below.
                      Demonstrates that an order-blind (bag-of-words) representation
                      literally cannot tell the two apart. Callback: the L16 pixel-shuffle
                      frame did the same demo on an image's pixels.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/word_shuffle.py
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


def box_width(word: str) -> float:
    """Monospace-ish width estimate (data units) sized to fit the word plus padding."""
    return 0.34 * len(word) + 0.55


def draw_sentence_row(ax, words, title, highlight_order, log):
    widths = [box_width(w) for w in words]
    gap = 0.18
    centers, x = [], 0.0
    for w in widths:
        centers.append(x + w / 2)
        x += w + gap
    total = x - gap
    ax.set_xlim(-0.3, total + 0.3)
    ax.set_ylim(0, 1)
    for cx, w, word in zip(centers, widths, words):
        ax.add_patch(plt.Rectangle((cx - w / 2, 0.15), w, 0.7, ec=BLUE,
                                    fc="#0033A0" + "1A", lw=1.4))
        ax.text(cx, 0.5, word, ha="center", va="center", fontsize=11, family="monospace")
    for i, cx in enumerate(centers):
        ax.text(cx, 0.03, str(highlight_order[i] + 1), ha="center", va="top",
                fontsize=8, color="#888")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    log.info(f"{title}: {' '.join(words)}")


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

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6),
                              gridspec_kw={"height_ratios": [1, 1.6]})

    draw_sentence_row(axes[0, 0], words, "Original review", np.arange(n), log)
    draw_sentence_row(axes[0, 1], shuffled_words, "Same words, one fixed shuffle",
                       perm, log)

    for ax, counts, title, color in [
        (axes[1, 0], counts_orig, "Bag-of-words histogram", BLUE),
        (axes[1, 1], counts_shuf, "Bag-of-words histogram", RED),
    ]:
        bars = ax.bar(vocab, counts, color=color)
        ax.bar_label(bars, fontsize=9)
        ax.set_ylim(0, max(counts_orig) + 1)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8.5)
        ax.set_yticks(range(0, max(counts_orig) + 2))

    fig.suptitle("Every word identical. All meaning gone. Histograms: pixel-identical.",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "word_shuffle.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_word_shuffle(log)
    log.info("done")


if __name__ == "__main__":
    main()
