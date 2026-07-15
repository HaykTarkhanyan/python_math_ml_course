"""Real figure for the L21 "Road to Attention" deck (Section 2: tokens).

Generates into ml/ch7_rnn/fig/:
  tokenizer_panel1.pdf -- the review sentence split three ways: char / word / subword.
  tokenizer_panel2.pdf -- the review sentence through a real subword tokenizer, token
                          boundaries and IDs visible.
  tokenizer_panel3.pdf -- the Armenian tax: the chapter's Armenian line vs its English
                          gloss, MEASURED token counts (bar chart, ax.bar_label).

Uses tiktoken's cl100k_base encoding (a real production BPE tokenizer, GPT-3.5/4
family) -- installed into the `ma` venv with
    uv pip install --python ./ma/Scripts/python.exe tiktoken
The encoding file downloads once on first use (~a few seconds) and is then cached by
tiktoken under the user's cache dir; this script fails loudly if tiktoken is missing or
the encoding cannot be loaded (no silent fallback to a fake tokenizer).

Chapter constants reused verbatim (never retyped):
  REVIEW_SENTENCE: "The pomegranates arrived fresh and sweet." (from L20)
  Armenian line: read from py_src/data/armenian_line.txt
  English gloss: locked at the L21 interview (see ENGLISH_GLOSS below)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/tokenizer_demo.py
"""

import logging
import sys
import warnings
from pathlib import Path

try:
    import tiktoken
except ImportError as e:
    raise ImportError(
        "tiktoken is required for tokenizer_demo.py and must be installed into the "
        "project venv: uv pip install --python ./ma/Scripts/python.exe tiktoken "
        "-- refusing to silently fall back to a fake tokenizer."
    ) from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# FAIL LOUDLY: a missing glyph must crash the script, never render as tofu.
warnings.filterwarnings("error", message=".*missing from current font.*")

ARM_FONT = "Segoe UI"
BLUE, GREEN, GRAY, RED, ORANGE = "#0033A0", "#008C46", "#999999", "#D90012", "#F2A800"

SEED = 509
REVIEW_SENTENCE = "The pomegranates arrived fresh and sweet."
ENGLISH_GLOSS = ("I watch all of this in silence, and the connoisseurs speak inside "
                  "me.")
ENCODING_NAME = "cl100k_base"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
DATA_FILE = HERE.parent / "data" / "armenian_line.txt"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l21_tokenizer_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "tokenizer_demo.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_armenian_line(log) -> str:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"canonical Armenian line missing: {DATA_FILE}")
    line = DATA_FILE.read_text(encoding="utf-8").strip()
    log.info(f"Armenian line ({len(line.split())} words): {line}")
    return line


def get_encoding(log):
    log.info(f"loading tiktoken encoding '{ENCODING_NAME}' (cached after first run)")
    enc = tiktoken.get_encoding(ENCODING_NAME)
    log.info("encoding loaded")
    return enc


# --- panel 1: char / word / subword split of the review sentence ------------------

def split_char(sentence: str) -> list[str]:
    return list(sentence)


def split_word(sentence: str) -> list[str]:
    # Simple whitespace + trailing-punctuation split (illustrative word tokenizer).
    out = []
    for w in sentence.split():
        core = w.rstrip(".,!?")
        trail = w[len(core):]
        if core:
            out.append(core)
        if trail:
            out.append(trail)
    return out


def split_subword(sentence: str, enc) -> list[str]:
    ids = enc.encode(sentence)
    return [enc.decode([i]) for i in ids]


def draw_token_row(fig, ax, y, tokens, color, box_h=0.6, fontsize=9, x0=0.0, pad=0.22,
                    gap=0.16):
    """Lay out token boxes left to right. Box width is measured from the ACTUAL
    rendered text extent (not a guessed char-count formula) so long subword pieces
    never overlap their neighbors -- measured once, in points, then converted to data
    units via the axes' fixed points-per-data-unit ratio (independent of xlim, so it
    is safe to call this before or after ax.set_xlim)."""
    # points-per-data-unit: render a probe string, compare its point size (known) to
    # its data-unit width under the CURRENT xlim/figure size.
    x = x0
    for tok in tokens:
        disp = tok.replace(" ", "␣")  # visible space marker (open box, U+2423)
        t = ax.text(x, y, disp, fontsize=fontsize, family="monospace", alpha=0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # re-fetch: draw() invalidates the old one
        bbox = t.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        (dx0, _), (dx1, _) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        text_w = dx1 - dx0
        t.remove()
        w = max(0.5, text_w + 2 * pad)
        ax.add_patch(Rectangle((x, y), w, box_h, ec=color, fc=color + "1A", lw=1.1))
        ax.text(x + w / 2, y + box_h / 2, disp, ha="center", va="center", fontsize=fontsize,
                family="monospace")
        x += w + gap
    return x  # right edge


def fig_panel1(enc, log):
    chars = split_char(REVIEW_SENTENCE)
    words = split_word(REVIEW_SENTENCE)
    subs = split_subword(REVIEW_SENTENCE, enc)
    log.info(f"panel1 char split ({len(chars)} tokens): {chars}")
    log.info(f"panel1 word split ({len(words)} tokens): {words}")
    log.info(f"panel1 subword split ({len(subs)} tokens): {subs}")

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    row_h = 0.6
    ys = [3.6, 1.9, 0.2]
    labels = [
        f"characters  ({len(chars)} tokens, vocab ≈ 100)",
        f"words  ({len(words)} tokens, vocab explodes; “pomegranatey” = OOV)",
        f"subwords (real BPE tokenizer)  ({len(subs)} tokens, vocab ≈ 100k)",
    ]
    colors = [GRAY, BLUE, GREEN]
    # Freeze the data <-> pixel mapping BEFORE measuring any text extents -- box
    # widths are measured from real rendered text, and that measurement is only
    # stable if xlim/ylim (and hence points-per-data-unit) never change afterward.
    ax.set_xlim(-7.6, 39.5)
    ax.set_ylim(-0.3, 4.6)
    ax.axis("off")
    for y, toks, color, label in zip(ys, [chars, words, subs], colors, labels):
        draw_token_row(fig, ax, y, toks, color, box_h=row_h,
                       fontsize=10 if color != GRAY else 8.5)
        ax.text(-0.3, y + row_h / 2, label, ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=color)
    fig.suptitle('"The pomegranates arrived fresh and sweet." split three ways',
                 fontsize=12.5)
    # NOTE: no fig.tight_layout() here -- it repositions the Axes bbox within the
    # Figure, which would silently invalidate the data<->pixel scale that
    # draw_token_row() already used to size every box from measured text extents
    # (found by rendering a debug PDF and seeing boxes too narrow for their own
    # text). bbox_inches="tight" on savefig() below only crops the exported page and
    # does not have this effect.
    out = FIG_DIR / "tokenizer_panel1.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def fig_panel2(enc, log):
    ids = enc.encode(REVIEW_SENTENCE)
    toks = [enc.decode([i]) for i in ids]
    log.info(f"panel2 subword tokens ({len(toks)}): {list(zip(toks, ids))}")

    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    x = 0.0
    for tok, tid in zip(toks, ids):
        disp = tok.replace(" ", "␣")
        w = max(0.75, 0.22 * len(disp) + 0.35)
        ax.add_patch(Rectangle((x, 1.1), w, 0.75, ec=GREEN, fc=GREEN + "1A", lw=1.3))
        ax.text(x + w / 2, 1.475, disp, ha="center", va="center", fontsize=11,
                family="monospace")
        ax.text(x + w / 2, 0.75, str(tid), ha="center", va="center", fontsize=7.5,
                color=GRAY)
        x += w + 0.1

    ax.text(x / 2, 2.35,
            f"{len(toks)} tokens for a {len(REVIEW_SENTENCE.split())}-word sentence "
            "-- vocabulary built from frequent character pairs merged together (BPE, "
            "one breath; full mechanics: ch8)", ha="center", fontsize=10.5, color="black")
    ax.set_xlim(-0.3, x + 0.3)
    ax.set_ylim(0.3, 2.9)
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "tokenizer_panel2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def fig_panel3(enc, armenian_line, log):
    ids_hy = enc.encode(armenian_line)
    ids_en = enc.encode(ENGLISH_GLOSS)
    n_hy, n_en = len(ids_hy), len(ids_en)
    ratio = n_hy / n_en
    log.info(f"panel3 Armenian line tokens: {n_hy}")
    log.info(f"panel3 English gloss tokens: {n_en}")
    log.info(f"panel3 ratio (hy/en): {ratio:.2f}x")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.bar(["Armenian line", "English gloss"], [n_hy, n_en],
                   color=[RED, BLUE], width=0.55)
    ax.bar_label(bars, fontsize=13, fontweight="bold")
    ax.set_ylabel("tokens (cl100k_base)", fontsize=11)
    ax.set_title(f"Same meaning, {ratio:.1f}× more tokens", fontsize=13)
    ax.set_ylim(0, max(n_hy, n_en) * 1.22)

    # Armenian line rendered below as plain text (font-safe: no mathtext mixing).
    fig.text(0.5, 0.02, armenian_line, ha="center", fontsize=9.5, fontfamily=ARM_FONT,
              color=GRAY)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG_DIR / "tokenizer_panel3.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    enc = get_encoding(log)
    armenian_line = load_armenian_line(log)
    log.info(f"English gloss ({len(ENGLISH_GLOSS.split())} words): {ENGLISH_GLOSS}")

    fig_panel1(enc, log)
    fig_panel2(enc, log)
    fig_panel3(enc, armenian_line, log)
    log.info("done")


if __name__ == "__main__":
    main()
