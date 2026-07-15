"""ILLUSTRATIVE figure for the L21 "Road to Attention" deck (Section 2: embeddings).

Generates into ml/ch7_rnn/fig/:
  embedding_2d.pdf -- a hand-placed 2-D SCHEMATIC of a word-embedding space: ~28 words
                      in six semantic clusters, plus the king/queen/man/woman quartet
                      drawn as an exact parallelogram with the analogy arrows.

SCOPE NOTE (instructor scope change, logged in L21_DECISIONS.md): the outline originally
asked for a small GloVe subset. The instructor later ruled out any embedding download --
no gensim, no glove-wiki-gigaword-50, no network fetch. This figure is a hand-placed
ILLUSTRATIVE schematic instead: the coordinates are chosen by hand to keep semantically
related words close together and to make the king - man + woman = queen arithmetic
exact (not fit from real vectors). The frame using this figure must say so explicitly
("schematic of a real embedding space").

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/embedding_2d.py
"""

import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
BLUE, GREEN, GRAY, RED, ORANGE = "#0033A0", "#008C46", "#555555", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l21_embedding_2d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "embedding_2d.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# Hand-placed clusters (x, y). The royalty quartet is placed so that
# king - man == queen - woman EXACTLY (both = (0.0, 1.35)), so the analogy arrows are
# genuinely parallel and equal-length, not just "close".
MAN, WOMAN = (0.0, 0.0), (2.1, 0.05)
ROYAL_VEC = (0.05, 1.35)
KING = (MAN[0] + ROYAL_VEC[0], MAN[1] + ROYAL_VEC[1])
QUEEN = (WOMAN[0] + ROYAL_VEC[0], WOMAN[1] + ROYAL_VEC[1])
PRINCE = (MAN[0] + 0.85, MAN[1] + 0.55)
PRINCESS = (WOMAN[0] + 0.85, WOMAN[1] + 0.55)

ROYALTY = {
    "man": MAN, "woman": WOMAN, "king": KING, "queen": QUEEN,
    "prince": PRINCE, "princess": PRINCESS,
}

CLUSTERS = {
    "royalty": ROYALTY,
    "fruit": {
        "pomegranate": (8.6, 5.4), "apple": (9.3, 5.9), "orange": (9.9, 5.2),
        "banana": (8.9, 4.6), "grape": (9.6, 4.3),
    },
    "animals": {
        "dog": (-6.2, 5.6), "cat": (-5.4, 6.1), "puppy": (-6.7, 6.4),
        "kitten": (-4.9, 6.7), "wolf": (-6.9, 4.9),
    },
    "places": {
        "Yerevan": (-7.5, -3.6), "Armenia": (-6.6, -4.2), "Paris": (-4.7, -3.3),
        "France": (-4.0, -4.0), "Berlin": (-5.4, -2.6),
    },
    "sentiment": {
        "good": (5.6, -4.4), "great": (6.3, -3.9), "excellent": (6.9, -4.6),
        "bad": (5.2, -6.3), "terrible": (5.9, -6.9), "awful": (6.6, -6.2),
    },
    "verb_tense": {
        "walk": (1.2, -7.3), "walked": (1.2, -9.0), "run": (3.4, -7.3),
        "ran": (3.4, -9.0),
    },
}


def n_words(clusters) -> int:
    return sum(len(d) for d in clusters.values())


def fig_embedding_2d(log):
    total = n_words(CLUSTERS)
    log.info(f"plotting {total} hand-placed words across {len(CLUSTERS)} clusters")

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    for cname, words in CLUSTERS.items():
        if cname == "royalty":
            continue
        for w, (x, y) in words.items():
            ax.scatter([x], [y], s=26, color=GRAY, zorder=2)
            ax.annotate(w, (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=9.5, color=GRAY)

    # Royalty quartet, highlighted.
    for w, (x, y) in ROYALTY.items():
        ax.scatter([x], [y], s=60, color=BLUE, zorder=3)
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=12, color=BLUE, fontweight="bold")

    # Analogy arrows: man -> king, woman -> queen (parallel by construction).
    ax.annotate("", xy=KING, xytext=MAN,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
    ax.annotate("", xy=QUEEN, xytext=WOMAN,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
    mid_x = (MAN[0] + KING[0]) / 2 - 1.35
    mid_y = (MAN[1] + KING[1]) / 2
    ax.text(mid_x, mid_y, "king - man\n= queen - woman", fontsize=10, color=RED,
            ha="center", fontweight="bold")

    for cname in ["fruit", "animals", "places", "sentiment", "verb_tense"]:
        xs = [p[0] for p in CLUSTERS[cname].values()]
        ys = [p[1] for p in CLUSTERS[cname].values()]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        ax.text(cx, max(ys) + 0.9, cname.replace("_", " "), fontsize=10, color=GRAY,
                ha="center", style="italic")

    ax.set_title("Illustrative schematic of a word-embedding space (2-D projection)\n"
                  "real embeddings: 100-1000 dimensions, learned from huge text corpora",
                  fontsize=12)
    ax.set_xlim(-10.5, 12.5)
    ax.set_ylim(-10.5, 8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "embedding_2d.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    assert math.isclose(KING[0] - MAN[0], QUEEN[0] - WOMAN[0]), "parallelogram broken (x)"
    assert math.isclose(KING[1] - MAN[1], QUEEN[1] - WOMAN[1]), "parallelogram broken (y)"
    fig_embedding_2d(log)
    log.info("done")


if __name__ == "__main__":
    main()
