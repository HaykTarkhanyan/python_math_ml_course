"""Real figure for the L20 RNN Foundations deck (Section 1: "The shoehorn hacks").

Generates into ml/13_rnns/fig/:
  pad_truncate.pdf -- three reviews of very different lengths (~3 / ~11 / ~42 words)
                      forced into a fixed window of 8 tokens: the short one is mostly
                      <pad>, the long one loses its ending -- including the verdict.

Run with the project venv:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/pad_truncate.py
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


# Drawn at the size the figure gets on the slide. Redrawn 2026-10-03: the old 12 in canvas
# shown at 0.82 x linewidth left ~3 pt text. Now the frame embeds it at the full line width
# (~5.5 in), the axes span the figure and data units are inches, so token boxes are sized
# from the true 7 pt monospace character width. Lost words go under their row (there is
# no room to the right of eight columns at this size).
FIG_W, FIG_H = 5.5, 1.85
MONO_PT = 7
CHAR_IN = 0.6 * MONO_PT / 72            # monospace advance width, inches
X0 = 0.76                                # left edge of the first token column
ROW_H = 0.21


def box_width(word: str) -> float:
    """Box width in inches: the word at MONO_PT monospace plus padding."""
    return CHAR_IN * len(word) + 0.055


def column_layout(rows):
    """Shared per-column widths (max needed across the three rows) so the fixed-length
    boundary lines up at the same x for every row."""
    col_widths = []
    for j in range(FIXED_LEN):
        cands = [tokens[j] if j < len(tokens) else "<pad>" for _, tokens in rows]
        col_widths.append(max(box_width(w) for w in cands))
    gap = 0.022
    centers, lefts, x = [], [], X0
    for w in col_widths:
        lefts.append(x)
        centers.append(x + w / 2)
        x += w + gap
    boundary = x - gap / 2
    if boundary > FIG_W:
        raise ValueError(f"eight columns need {boundary:.2f} in, figure is {FIG_W} in")
    return col_widths, lefts, centers, boundary


def draw_row(ax, y, label, tokens, col_widths, lefts, centers, log):
    """One review as eight fixed slots at height y (row centre, inches). Returns the y of
    the lowest text it drew."""
    n = len(tokens)
    for j, w in enumerate(tokens[:FIXED_LEN]):
        ax.add_patch(plt.Rectangle((lefts[j], y - ROW_H / 2), col_widths[j], ROW_H, ec=GREEN,
                                    fc=GREEN + "22", lw=0.9))
        ax.text(centers[j], y, w, ha="center", va="center", fontsize=MONO_PT,
                family="monospace")
    ax.text(X0 - 0.07, y, label, ha="right", va="center", fontsize=7, fontweight="bold",
            linespacing=1.0)
    lowest = y - ROW_H / 2
    if n < FIXED_LEN:
        for j in range(n, FIXED_LEN):
            ax.add_patch(plt.Rectangle((lefts[j], y - ROW_H / 2), col_widths[j], ROW_H,
                                        ec=GRAY, fc="#EEEEEE", lw=0.7, linestyle="--"))
            ax.text(centers[j], y, "<pad>", ha="center", va="center", fontsize=MONO_PT,
                    family="monospace", color=GRAY)
        log.info(f"{label.splitlines()[0]}: {n} words, padded {FIXED_LEN - n} "
                 f"-> {FIXED_LEN - n}/{FIXED_LEN} slots wasted")
    else:
        lost = tokens[FIXED_LEN:]
        lines = textwrap.wrap(f"LOST ({len(lost)} words): " + " ".join(lost), width=92)
        for k, line in enumerate(lines):
            lowest = y - ROW_H / 2 - 0.06 - 0.13 * k
            ax.text(X0, lowest, line, ha="left", va="top", fontsize=7, color=RED,
                    style="italic")
        lowest -= 0.12
        log.info(f"{label.splitlines()[0]}: {n} words, kept first {FIXED_LEN}, "
                 f"{len(lost)} words truncated away: {' '.join(lost)}")
    return lowest


def fig_pad_truncate(log):
    rows = [
        ("short\n(3 words)", REVIEWS["short"].split()),
        ("medium\n(11 words)", REVIEWS["medium"].split()),
        ("long\n(42 words)", REVIEWS["long"].split()),
    ]
    col_widths, lefts, centers, boundary = column_layout(rows)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    ax.text(0.03, FIG_H - 0.04, "Pad + truncate to a fixed length: waste on the short one, "
            "amputation on the long one", ha="left", va="top", fontsize=8, fontweight="bold")
    y = FIG_H - 0.48
    top = y + ROW_H / 2
    for label, tokens in rows:
        lowest = draw_row(ax, y, label, tokens, col_widths, lefts, centers, log)
        y = lowest - 0.06 - ROW_H / 2
    ax.plot([boundary, boundary], [lowest + 0.1, top + 0.08], color="black", lw=0.9)
    ax.text(boundary - 0.03, top + 0.1, f"fixed length = {FIXED_LEN}", ha="right", va="bottom",
            fontsize=7)
    if lowest < 0:
        raise ValueError(f"rows run {-lowest:.2f} in below the figure")
    out = FIG_DIR / "pad_truncate.pdf"
    fig.savefig(out)
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
