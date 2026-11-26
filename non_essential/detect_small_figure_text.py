"""Flag slide pages whose text renders too small to read - usually a figure drawn too wide.

**Why this exists.** A matplotlib figure saved at 10 x 4 inches and placed in a 3.5-inch slot is
scaled to 35%, and so is every label in it: 10 pt becomes 3.5 pt. Nothing warns about it.
`detect_clipped_slides.py` and `detect_footer_collisions.py` both pass, and a contact sheet at
45 dpi makes 3 pt text look like ordinary small print. Found 2026-10-01 in `ml/12_cnn`: almost
every figure of the four CNN decks rendered its labels at 2.3-5 pt against 10.9 pt body text.

**How it works.** PyMuPDF reports every text span with its size *as rendered on the page*,
including text inside embedded figure PDFs (their scaling is applied). The detector lists, per
page, the spans smaller than the threshold. Raster figures (PNG/JPG) contain no text and cannot be
checked - look at those.

Usage::

    ./ma/.venv/Scripts/python.exe non_essential/detect_small_figure_text.py DECK.pdf [MORE.pdf ...]
    ./ma/.venv/Scripts/python.exe non_essential/detect_small_figure_text.py ml/12_cnn --threshold 6

Options: ``--threshold PT`` (default 5.5), ``--min-chars N`` (default 2: single characters are
skipped, because math subscripts in normal slide text are legitimately small).

Exits 1 if anything is flagged. The fix is to draw the figure at its slot size (figsize = the
width it gets on the slide, fonts 7-10 pt), not to scale fonts up blindly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_logging() -> logging.Logger:
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(REPO_ROOT / "logs" / "detect_small_figure_text.log",
                                                      mode="w", encoding="utf-8")])
    return logging.getLogger("detect_small_figure_text")


def pdfs_from(args: list[str]) -> list[Path]:
    out = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            # compiled decks only: a .pdf with a sibling .tex (figures in fig/ are not decks)
            out += sorted(q for q in p.glob("*.pdf") if q.with_suffix(".tex").exists())
        elif p.suffix.lower() == ".pdf" and p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(f"not a PDF or a directory: {a}")
    if not out:
        raise FileNotFoundError(f"no deck PDFs found in {args}")
    return out


def small_spans(page, threshold: float, min_chars: int):
    found = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = "".join(c["c"] for c in span["chars"]).strip()
                if len(text) >= min_chars and span["size"] < threshold:
                    found.append((round(span["size"], 1), text[:40]))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=5.5)
    ap.add_argument("--min-chars", type=int, default=2)
    args = ap.parse_args()
    log = setup_logging()

    total = 0
    for pdf in pdfs_from(args.paths):
        doc = fitz.open(pdf)
        flagged = []
        for pno, page in enumerate(doc, start=1):
            spans = small_spans(page, args.threshold, args.min_chars)
            if spans:
                flagged.append((pno, spans))
        if not flagged:
            log.info(f"{pdf.name}: clean (no text under {args.threshold} pt, {len(doc)} pages)")
            continue
        total += len(flagged)
        log.info(f"{pdf.name}: {len(flagged)} page(s) with text under {args.threshold} pt")
        for pno, spans in flagged:
            smallest = min(s for s, _ in spans)
            sample = "; ".join(f"{s}pt '{t}'" for s, t in sorted(set(spans))[:3])
            log.info(f"    page {pno:3d}  min {smallest:4.1f} pt  ({len(spans)} spans)  {sample}")
    log.info(f"\nTOTAL pages flagged: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
