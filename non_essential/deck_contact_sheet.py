"""Render a Beamer PDF into numbered contact sheets (3x3 pages per image) for a fast visual pass.

The detectors next to this file catch clipped text and footer collisions; they cannot see a figure
whose labels are too small to read, a diagram that jumps between overlay steps, or a table that
wraps badly. Those need eyes, and looking at 50 pages one by one is slow. A contact sheet shows
nine pages at once with their page numbers stamped in red.

Usage:
    ./ma/Scripts/python.exe non_essential/deck_contact_sheet.py DECK.pdf OUT_DIR [DPI] [PAGES]

    DPI    default 45 - enough to spot layout problems; use 70-110 to read small text
    PAGES  optional comma-separated page numbers, e.g. 2,13,41 (default: every page)

Writes OUT_DIR/p-NN.png per page and OUT_DIR/sheet_NN.png per nine pages. Use a fresh OUT_DIR
each time rather than clearing an old one. Needs pdftoppm (ships with TeX Live).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_logging() -> logging.Logger:
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(REPO_ROOT / "logs" / "deck_contact_sheet.log",
                                                      mode="w", encoding="utf-8")])
    return logging.getLogger("deck_contact_sheet")


def main() -> None:
    log = setup_logging()
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    pdf, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    if pdf.suffix.lower() != ".pdf" or not pdf.exists():
        raise FileNotFoundError(f"not an existing PDF: {pdf}")
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    wanted = {int(p) for p in sys.argv[4].split(",")} if len(sys.argv) > 4 else None

    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(outdir / "p")], check=True)
    pages = sorted(outdir.glob("p-*.png"), key=lambda f: int(f.stem.split("-")[1]))
    if wanted:
        pages = [f for f in pages if int(f.stem.split("-")[1]) in wanted]
        missing = wanted - {int(f.stem.split("-")[1]) for f in pages}
        if missing:
            raise ValueError(f"pages not in the PDF: {sorted(missing)}")

    w, h = Image.open(pages[0]).size
    for start in range(0, len(pages), 9):
        sheet = Image.new("RGB", (3 * w + 20, 3 * h + 20), "white")
        draw = ImageDraw.Draw(sheet)
        for k, f in enumerate(pages[start:start + 9]):
            x, y = (k % 3) * (w + 10), (k // 3) * (h + 10)
            sheet.paste(Image.open(f), (x, y))
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline="gray")
            draw.text((x + 4, y + 2), f.stem.split("-")[1], fill="red")
        sheet.save(outdir / f"sheet_{start // 9 + 1:02d}.png")
    log.info(f"{len(pages)} pages -> {outdir}")


if __name__ == "__main__":
    main()
