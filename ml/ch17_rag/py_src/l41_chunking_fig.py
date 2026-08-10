"""Chunking figure for L41 (RAG retrieval): figure 02.

REAL: the chunk boundaries are computed by actually running each strategy over a real
manual excerpt. Nothing is hand-placed. The script asserts that fixed-size chunking really
does split the answer sentence, because that is the claim the slide makes.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_chunking_fig.py
"""

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l41_chunking_fig.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "figure.dpi": 140})

# Two paragraphs, so paragraph-based chunking is a real strategy and not a no-op.
PARAGRAPHS = [
    "The Lori press operates at 2.5 bar during the first pressing stage. "
    "After twenty minutes the pressure is raised to 3.2 bar for the second stage.",
    "Operators must record both readings in the shift log. "
    "If the gauge drifts by more than 0.2 bar, stop the line and call maintenance.",
]
PARA_SEP = "\n\n"
TEXT = PARA_SEP.join(PARAGRAPHS)
ANSWER = "The Lori press operates at 2.5 bar during the first pressing stage."


def fixed_size(text, size, overlap=0):
    """Cut every `size` characters, optionally stepping back by `overlap`."""
    step = size - overlap
    if step <= 0:
        raise ValueError(f"overlap {overlap} must be smaller than size {size}")
    spans, start = [], 0
    while start < len(text):
        spans.append((start, min(start + size, len(text))))
        start += step
    return spans


def sentences(text):
    """Split on sentence-ending punctuation, keeping the punctuation."""
    spans, start = [], 0
    for m in re.finditer(r"[.!?]\s+", text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def recursive(text, max_chars=120):
    """Merge whole sentences until adding the next would exceed max_chars."""
    spans, cur_start, cur_end = [], None, None
    for s, e in sentences(text):
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start <= max_chars:
            cur_end = e
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


def paragraphs(text):
    """Split on blank lines. The cheapest strategy that respects the author's own structure."""
    spans, cursor = [], 0
    for para in text.split(PARA_SEP):
        start = text.index(para, cursor)
        spans.append((start, start + len(para)))
        cursor = start + len(para)
    return spans


def splits_answer(spans, text, answer):
    """True if no single chunk contains the whole answer sentence."""
    a0 = text.index(answer)
    a1 = a0 + len(answer)
    return not any(s <= a0 and e >= a1 for s, e in spans)


def main():
    strategies = [
        ("Fixed size, 60 chars", fixed_size(TEXT, 60), ARM_RED),
        ("Fixed size + 15 overlap", fixed_size(TEXT, 60, 15), ARM_ORANGE),
        ("By sentence", sentences(TEXT), ARM_BLUE),
        ("By paragraph", paragraphs(TEXT), "#7A4FBF"),
        ("Recursive (merge to 150)", recursive(TEXT, 150), "#2E8B57"),
    ]

    for name, spans, _ in strategies:
        broke = splits_answer(spans, TEXT, ANSWER)
        log.info("%-26s %2d chunks, answer split: %s", name, len(spans), broke)

    if not splits_answer(TEXT and strategies[0][1], TEXT, ANSWER):
        raise ValueError("fixed-size chunking did not split the answer - the slide's claim fails")
    if splits_answer(strategies[2][1], TEXT, ANSWER):
        raise ValueError("sentence chunking split the answer - unexpected")

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    a0 = TEXT.index(ANSWER)
    a1 = a0 + len(ANSWER)

    ax.axvspan(a0, a1, color="#FFE9A8", zorder=0)
    ax.text((a0 + a1) / 2, len(strategies) - 0.28,
            "the sentence that answers the question",
            ha="center", fontsize=9, color="#7A5C00")

    for row, (name, spans, color) in enumerate(strategies):
        y = len(strategies) - 1 - row
        for i, (s, e) in enumerate(spans):
            ax.barh(y, e - s, left=s, height=0.5,
                    color=color, alpha=0.55 if i % 2 else 0.85,
                    edgecolor="white", linewidth=1.4, zorder=3)
        broke = splits_answer(spans, TEXT, ANSWER)
        ax.text(len(TEXT) + 6, y, "answer split" if broke else "answer intact",
                va="center", fontsize=8.5, fontweight="bold",
                color=ARM_RED if broke else "#2E8B57")

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([n for n, _, _ in strategies][::-1], fontsize=9.5)
    ax.set_xlabel("character position in the paragraph")
    ax.set_xlim(0, len(TEXT) + 78)
    ax.set_title("Same text, five chunking strategies", fontsize=11)
    ax.tick_params(axis="y", length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "02_chunking_strategies.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
