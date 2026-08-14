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

# A full-bleed image frame (an approved default in ml/SLIDE_STYLE.md) covers the whole slide, so
# its footer band is nearly all ink - correct, not a collision. Detect it by what "full-bleed"
# literally means: the image reaches the page edge. A Beamer text frame always leaves white at
# the extreme border, so the outer ring is ~0% covered; a still that bleeds to the edge is ~100%,
# whether the image is dark or light. (An earlier attempt keyed on how dark the page body was,
# which missed light stills - a photo of a white page read as "not a picture".)
BORDER_RING_FRACTION = 0.015     # how far in from each edge counts as the ring
BORDER_WHITE_LEVEL = 240         # >= this is effectively page white
FULL_BLEED_BORDER_COVERAGE = 0.90

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


def check_page(png: Path) -> tuple[float, float]:
    """Return (ink fraction of the footer band, coverage of the outer border ring).

    The band excludes the right-hand corner where the page number legitimately lives.
    Border coverage near 1.0 means the page is a full-bleed image rather than a text frame.
    """
    image = np.asarray(Image.open(png).convert("L"))
    height, width = image.shape
    split = int(height * (1 - BAND_FRACTION))
    band = image[split:, : int(width * (1 - RIGHT_MARGIN_FRACTION))]

    dy = max(1, int(height * BORDER_RING_FRACTION))
    dx = max(1, int(width * BORDER_RING_FRACTION))
    ring = np.concatenate([
        image[:dy, :].ravel(), image[-dy:, :].ravel(),
        image[:, :dx].ravel(), image[:, -dx:].ravel(),
    ])
    return (float((band < INK_THRESHOLD).mean()),
            float((ring < BORDER_WHITE_LEVEL).mean()))


def check_pdf(pdf: Path, log: logging.Logger) -> list[tuple[int, float]]:
    flagged: list[tuple[int, float]] = []
    full_bleed: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for png in render_pages(pdf, Path(tmp), log):
            page = int(png.stem.rsplit("-", 1)[1])
            band_ink, border = check_page(png)
            if band_ink <= FLAG_FRACTION:
                continue
            if border > FULL_BLEED_BORDER_COVERAGE:
                full_bleed.append(page)
            else:
                flagged.append((page, band_ink))

    if flagged:
        log.info(f"{pdf.name}: {len(flagged)} frame(s) with content in the footer band")
        for page, ink in flagged:
            log.info(f"    page {page:3d}   ink {ink * 100:5.2f}%")
    else:
        log.info(f"{pdf.name}: clean")
    if full_bleed:
        log.info(f"    ({len(full_bleed)} full-bleed image frame(s) not counted: "
                 f"{', '.join(str(p) for p in full_bleed)})")
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
