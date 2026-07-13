"""Real figure for the L20 RNN Foundations deck (Section 1: "The shoehorn hacks").

Generates into ml/ch7_rnn/fig/:
  pad_truncate.pdf -- three reviews of very different lengths (~3 / ~11 / ~42 words)
                      forced into a fixed window of 8 tokens: the short one is mostly
                      <pad>, the long one loses its ending -- including the verdict.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/pad_truncate.py
"""

import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIXED_LEN = 8
GREEN = "#008C46"
GRAY = "#AAAAAA"
RED = "#D90012"

# The three reviews (short/medium reused verbatim from the L20 cold open;
# long is this script's own 42-word complaint, verdict word last).
REVIEWS = {
    "short": "Fresh and sweet!",
    "medium": "The pomegranates arrived fresh and sweet, best I've had since Yerevan.",
    "long": (
        "I ordered a large box of exported pomegranates for my mother's birthday "
        "expecting the same deep red arils and rich tart sweetness I remembered "
        "from childhood markets back home but after the long shipping delay they "
        "arrived bruised dry and honestly disappointing"
    ),
}

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_pad_truncate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "pad_truncate.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def box_width(word: str) -> float:
    """Monospace-ish width estimate (data units) sized to fit the word plus padding."""
    return 0.30 * len(word) + 0.5


def column_layout(rows):
    """Shared per-column widths (max needed across the three rows) so the fixed-length
    boundary lines up at the same x for every row."""
    col_words = []
    for j in range(FIXED_LEN):
        candidates = []
        for _, tokens in rows:
            w = tokens[j] if j < len(tokens) else "<pad>"
            candidates.append(w)
        col_words.append(candidates)
    col_widths = [max(box_width(w) for w in cands) for cands in col_words]
    gap = 0.12
    centers, lefts, x = [], [], 0.0
    for w in col_widths:
        lefts.append(x)
        centers.append(x + w / 2)
        x += w + gap
    boundary = x - gap / 2
    return col_widths, lefts, centers, boundary


def draw_row(ax, y, h, label, tokens, col_widths, lefts, centers, boundary, log):
    n = len(tokens)
    kept = tokens[:FIXED_LEN]
    for j, w in enumerate(kept):
        ax.add_patch(plt.Rectangle((lefts[j], y), col_widths[j], h, ec=GREEN,
                                    fc=GREEN + "22", lw=1.4))
        ax.text(centers[j], y + h / 2, w, ha="center", va="center", fontsize=8.5,
                family="monospace")
    if n < FIXED_LEN:
        pad_count = FIXED_LEN - n
        for j in range(n, FIXED_LEN):
            ax.add_patch(plt.Rectangle((lefts[j], y), col_widths[j], h, ec=GRAY,
                                        fc="#EEEEEE", lw=1.0, linestyle="--"))
            ax.text(centers[j], y + h / 2, "<pad>", ha="center", va="center", fontsize=7.5,
                    color=GRAY)
        log.info(f"{label.splitlines()[0]}: {n} words, kept all {n}, padded {pad_count} "
                 f"-> {pad_count}/{FIXED_LEN} slots wasted")
    else:
        lost = tokens[FIXED_LEN:]
        wrapped = textwrap.wrap("LOST: " + " ".join(lost), width=58)
        n_lines = len(wrapped)
        box_h = max(h, 0.34 * n_lines + 0.14)
        y_box = y + h / 2 - box_h / 2
        ax.add_patch(plt.Rectangle((boundary + 0.15, y_box), 8.6, box_h, ec=RED,
                                    fc=RED + "11", lw=1.2, linestyle=":"))
        line_h = box_h / (n_lines + 1)
        for k, line in enumerate(wrapped):
            ax.text(boundary + 0.35, y_box + box_h - line_h * (k + 1), line, ha="left",
                    va="center", fontsize=7.5, color=RED, style="italic")
        log.info(f"{label.splitlines()[0]}: {n} words, kept first {FIXED_LEN}, "
                 f"{len(lost)} words truncated away: {' '.join(lost)}")
    ax.text(-0.3, y + h / 2, label, ha="right", va="center", fontsize=10, fontweight="bold")
    return y + h  # row top, for layout bookkeeping


def fig_pad_truncate(log):
    rows = [
        ("short\n(3 words)", REVIEWS["short"].split()),
        ("medium\n(11 words)", REVIEWS["medium"].split()),
        ("long\n(42 words)", REVIEWS["long"].split()),
    ]
    col_widths, lefts, centers, boundary = column_layout(rows)

    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    h = 0.8
    row_gap = 1.5
    y_positions = [3.0, 3.0 - row_gap, 3.0 - 2 * row_gap]
    for (label, tokens), y in zip(rows, y_positions):
        draw_row(ax, y, h, label, tokens, col_widths, lefts, centers, boundary, log)

    ax.axvline(boundary, color="black", lw=1.0, linestyle="-")
    ax.text(boundary, y_positions[0] + h + 0.35, f"fixed length = {FIXED_LEN}",
            ha="center", fontsize=10)

    ax.set_xlim(-3.4, boundary + 9.2)
    ax.set_ylim(y_positions[-1] - 0.5, y_positions[0] + h + 0.8)
    ax.axis("off")
    fig.suptitle("Pad + truncate to a fixed length: waste on the short one, "
                 "amputation on the long one", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "pad_truncate.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for label, text in REVIEWS.items():
        log.info(f"review[{label}] ({len(text.split())} words): {text}")
    fig_pad_truncate(log)
    log.info("done")


if __name__ == "__main__":
    main()
