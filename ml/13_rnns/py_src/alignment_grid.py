"""Figure for L21 "Each step needs different words": the word alignment between the chapter's
Armenian line (what the seq2seq decoder writes) and its English gloss (what the encoder reads).

Two PDFs for a predict-first frame:
    fig/alignment_grid_0.pdf - the empty grid (students fill it in)
    fig/alignment_grid_1.pdf - the filled grid

The point of the frame: every output word needs a different, small part of the input, and in a
different order (Armenian puts "this all" before the verb, and says "my inside" for "inside me"),
while seq2seq hands every decoder step the same context vector C.

The alignment below is hand-made (drafted 2026-10-03, to be confirmed by a native speaker).
Dark cells = the word that carries the meaning; light cells = a weaker link (an auxiliary verb's
person agreement, a preposition folded into an adverb). "the" is aligned to nothing on purpose.

The Armenian words are read from py_src/data/armenian_line.txt (never retyped). The English
gloss is the chapter constant used in L21's encoder-decoder frame.

Runtime: ~2 s. Run with the project venv:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/alignment_grid.py
"""

import logging
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# FAIL LOUDLY: a missing Armenian glyph must crash the script, never render as tofu.
warnings.filterwarnings("error", message=".*missing from current font.*")

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
LINE_FILE = HERE.parent / "data" / "armenian_line.txt"

ARM_FONT = "Segoe UI"
BLUE, LIGHT, RED, GRID = "#0033A0", "#A9BCE3", "#D90012", "#BBBBBB"

ENGLISH = "I watch all of this in silence , and the connoisseurs speak inside me".split()
ENGLISH = [w for w in ENGLISH if w != ","]

# Armenian word index -> (strong English indices, weak English indices)
ALIGN = {
    0: ([0], []),          # Ես        -> I
    1: ([4], []),          # այս       -> this
    2: ([2], [3]),         # ամենին   -> all (of)
    3: ([1], []),          # նայում    -> watch
    4: ([1], [0]),         # եմ        -> (I) watch: auxiliary, 1st person
    5: ([6], [5]),         # լուռ      -> (in) silence
    6: ([7], []),          # և         -> and
    7: ([9], []),          # գիտակներ  -> connoisseurs
    8: ([10], [9]),        # են        -> speak: auxiliary, agrees with the subject
    9: ([10], []),         # խոսում    -> speak
    10: ([12], []),        # իմ        -> me
    11: ([11], []),        # մեջ       -> inside
}
# Rows where the order runs backwards, outlined in the filled version: (rows, cols), inclusive.
FLIPS = [((1, 3), (1, 4)), ((10, 11), (11, 12))]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("alignment_grid")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "alignment_grid.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def armenian_words() -> list[str]:
    line = LINE_FILE.read_text(encoding="utf-8").strip()
    words = [w.strip(",.") for w in line.split()]
    words = [w for w in words if w]
    if len(words) != len(ALIGN):
        raise ValueError(f"{LINE_FILE} has {len(words)} words, ALIGN covers {len(ALIGN)}")
    return words


def draw(words: list[str], filled: bool, out: Path) -> None:
    n_r, n_c = len(words), len(ENGLISH)
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    for r in range(n_r):
        for c in range(n_c):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="white", edgecolor=GRID, lw=0.5))
        if filled:
            strong, weak = ALIGN[r]
            for c in weak:
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=LIGHT, edgecolor=GRID, lw=0.5))
            for c in strong:
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=BLUE, edgecolor=GRID, lw=0.5))
        ax.text(-0.25, r + 0.5, f"{r + 1}  {words[r]}", ha="right", va="center", fontsize=7,
                fontfamily=ARM_FONT)
    for c, w in enumerate(ENGLISH):
        ax.text(c + 0.5, -0.25, w, ha="left", va="bottom", rotation=55, fontsize=7,
                rotation_mode="anchor")
    if filled:
        for (r0, r1), (c0, c1) in FLIPS:
            ax.add_patch(FancyBboxPatch((c0 - 0.1, r0 - 0.1), c1 - c0 + 1.2, r1 - r0 + 1.2,
                                        boxstyle="round,pad=0,rounding_size=0.3",
                                        fill=False, edgecolor=RED, lw=1.1))
    ax.text(-0.25, -0.9, "rows: decoder\nwrites\n\ncolumns:\nencoder read",
            ha="right", va="bottom", fontsize=6.3, color="#555555", style="italic",
            linespacing=1.1)
    ax.set_xlim(-0.1, n_c + 1.6)
    ax.set_ylim(n_r + 0.1, -0.1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0.27, right=0.99, top=0.75, bottom=0.02)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    log = setup_logging()
    t0 = time.perf_counter()
    words = armenian_words()
    log.info(f"Armenian words: {words}")
    log.info(f"English words: {ENGLISH}")
    for r, (strong, weak) in ALIGN.items():
        log.info(f"{r + 1:>2} {words[r]:<10} -> strong {[ENGLISH[c] for c in strong]}, "
                 f"weak {[ENGLISH[c] for c in weak]}")
    FIG_DIR.mkdir(exist_ok=True)
    for k, filled in enumerate([False, True]):
        out = FIG_DIR / f"alignment_grid_{k}.pdf"
        draw(words, filled, out)
        log.info(f"wrote {out}")
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
