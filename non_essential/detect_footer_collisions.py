"""Flag Beamer frames whose content runs into the page-number footer.

**Why this exists, separately from `detect_clipped_slides.py`.**

`detect_clipped_slides.py` looks for content *clipped at the frame edge* - a TikZ picture or a
figure running past the slide boundary. It does not catch the other silent Beamer failure: a
`tcolorbox` or a paragraph that simply grows downward until it sits on top of the page number.
The text is still drawn, so nothing is clipped in the sense that detector tests for, but the last
line is unreadable and often overlaps the footer.

Found 2026-08-13: `detect_clipped_slides.py` reported 0 flagged frames for all three decks of
`ml/ch19_mech_interp`, while **12 frames** across those decks had content in the footer band -
including one that lost the final sentence of a worked-numbers frame and one that cut a citation
off mid-word. Both decks had also passed two `pdflatex` runs with 0 overfull-vbox warnings,
because Beamer does not warn about this.

**How it works.** Render each page, look at the bottom strip, and ignore the right-hand corner
where the page number legitimately lives. Any ink left over is content that should not be there.

Usage::

    ./ma/Scripts/python.exe non_essential/detect_footer_collisions.py DECK.pdf [MORE.pdf ...]
    ./ma/Scripts/python.exe non_essential/detect_footer_collisions.py ml/ch19_mech_interp

Exits 1 if anything is flagged, so it can gate a build. Every flag must still be checked against
the rendered page - a full-bleed figure frame legitimately fills the bottom of the slide.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Bottom strip of the page to inspect, as a fraction of page height. The footer template in
# ml/preamble.tex sits inside roughly the bottom 5%.
BAND_FRACTION = 0.055

# Ignore this far into the page from the right: the page number's own territory.
RIGHT_MARGIN_FRACTION = 0.20

# Grey level below which a pixel counts as ink (0 = black, 255 = white).
INK_THRESHOLD = 200

# Fraction of the inspected band that must be ink before we flag the page. Antialiasing and the
# occasional descender put a handful of dark pixels in the band on a clean slide.
FLAG_FRACTION = 0.002

RENDER_DPI = 105


def setup_logging() -> logging.Logger:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "detect_footer_collisions.log", mode="w",
                                encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def render_pages(pdf: Path, out_dir: Path, log: logging.Logger) -> list[Path]:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "pdftoppm not found on PATH. It ships with TeX Live (poppler); this script cannot "
            "run without it."
        )
    prefix = out_dir / pdf.stem
    result = subprocess.run(
        ["pdftoppm", "-png", "-r", str(RENDER_DPI), str(pdf), str(prefix)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed on {pdf}: {result.stderr.strip()}")

    pages = sorted(out_dir.glob(f"{pdf.stem}-*.png"))
    if not pages:
        raise RuntimeError(f"pdftoppm produced no pages for {pdf}")
    return pages


def check_page(png: Path) -> float:
    """Fraction of the footer band (excluding the page-number corner) that is ink."""
    image = np.asarray(Image.open(png).convert("L"))
    height, width = image.shape
    band = image[int(height * (1 - BAND_FRACTION)):, : int(width * (1 - RIGHT_MARGIN_FRACTION))]
    return float((band < INK_THRESHOLD).mean())


def check_pdf(pdf: Path, log: logging.Logger) -> list[tuple[int, float]]:
    flagged = []
    with tempfile.TemporaryDirectory() as tmp:
        for png in render_pages(pdf, Path(tmp), log):
            page = int(png.stem.rsplit("-", 1)[1])
            ink = check_page(png)
            if ink > FLAG_FRACTION:
                flagged.append((page, ink))

    if flagged:
        log.info(f"{pdf.name}: {len(flagged)} frame(s) with content in the footer band")
        for page, ink in flagged:
            log.info(f"    page {page:3d}   ink {ink * 100:5.2f}%")
    else:
        log.info(f"{pdf.name}: clean")
    return flagged


def collect_pdfs(targets: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            # Never touch a hand-annotated lecture export; those are not reproducible.
            pdfs.extend(sorted(p for p in path.glob("*.pdf") if not p.stem.endswith("_notes")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
        else:
            raise FileNotFoundError(f"{target} is neither a .pdf nor a directory")
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found in {targets}")
    return pdfs


def main() -> int:
    log = setup_logging()
    if len(sys.argv) < 2:
        log.info(__doc__)
        return 2

    pdfs = collect_pdfs(sys.argv[1:])
    total = 0
    for pdf in pdfs:
        total += len(check_pdf(pdf, log))

    log.info("")
    log.info(f"TOTAL frames flagged: {total}  (verify each against the rendered page)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
