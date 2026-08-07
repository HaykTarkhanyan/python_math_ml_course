"""Extract only the ՊԱՆԻՐ letter folders from the Mashtots zip.

The full archive is 70,060 PNGs across 78 class folders; the homework needs five of them.
Class folders inside the zip are named by the competition's label id, so the mapping below
is the one published on the competition data page (0->Ա ... 77->ֆ).

Re-run with a different LETTERS dict to change the word.
"""

import logging
import zipfile
from pathlib import Path

CHAPTER = Path(__file__).resolve().parent.parent
DATA = CHAPTER / "data"
ZIP = DATA / "mashtots-dataset-v2.zip"
RAW = DATA / "mashtots_raw"

LOGS = CHAPTER.parent.parent / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "extract_mashtots.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ՊԱՆԻՐ - "cheese", which is also this course's difficulty unit.
LETTERS = {26: "PEH", 0: "AYB", 21: "NOW", 10: "INI", 32: "REH"}


def main():
    if not ZIP.exists():
        raise FileNotFoundError(f"missing archive: {ZIP}")
    RAW.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        wanted = {c: [n for n in names if n.startswith(f"Train/Train/{c}/") and n.endswith(".png")]
                  for c in LETTERS}
        for c, files in wanted.items():
            if not files:
                raise RuntimeError(f"class {c} ({LETTERS[c]}) has no PNGs in the archive - wrong layout?")
            log.info("class %-2d %-4s  %5d images", c, LETTERS[c], len(files))

        total = sum(len(f) for f in wanted.values())
        log.info("extracting %d files (of %d in the archive) -> %s", total, len(names), RAW)
        for c, files in wanted.items():
            z.extractall(RAW, members=files)
            log.info("  done class %d (%s)", c, LETTERS[c])

    on_disk = sum(1 for _ in (RAW / "Train" / "Train").rglob("*.png"))
    log.info("extracted %d PNGs, expected %d", on_disk, total)
    if on_disk != total:
        raise RuntimeError(f"extraction incomplete: {on_disk} != {total}")


if __name__ == "__main__":
    main()
