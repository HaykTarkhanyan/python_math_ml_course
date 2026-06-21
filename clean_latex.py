#!/usr/bin/env python
"""Remove LaTeX/Beamer build junk (.aux, .log, .nav, .toc, ...) from the repo.

Safe by construction: an auxiliary file is deleted only when a sibling
``<stem>.tex`` exists in the same folder, so Python ``logs/*.log`` and other
non-LaTeX files are never touched. The compiled ``.pdf`` and the ``.tex``
source are always kept. Files literally named ``texput.*`` are also removed,
since ``texput`` is TeX's reserved jobname for a no-name run and is never real
content.

Run it after a PDF is ready.

Usage (works from any directory):
    python clean_latex.py            # clean the whole repo
    python clean_latex.py PATH ...   # clean only the given dirs / .tex files
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Repo root = the folder this script lives in (so it works from any cwd).
ROOT = Path(__file__).resolve().parent

# Directories we never descend into (own git data, third-party LaTeX sources).
EXCLUDE_DIRS = {".git", "_reference"}

# LaTeX / Beamer auxiliary extensions. Never includes .tex / .pdf / .bib.
JUNK_EXTS = (
    ".aux", ".bbl", ".blg", ".bcf", ".fdb_latexmk", ".fls", ".idx", ".ilg",
    ".ind", ".lof", ".log", ".lot", ".nav", ".out", ".run.xml", ".snm",
    ".synctex.gz", ".toc", ".vrb", ".dvi", ".xdv", ".auxlock",
)


def setup_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "clean_latex.log", encoding="utf-8"),
        ],
    )


def _excluded(path: Path, root: Path) -> bool:
    return bool(EXCLUDE_DIRS & set(path.relative_to(root).parts))


def junk_for_tex(tex: Path):
    """Yield existing junk files that share ``tex``'s stem in its folder."""
    stem = tex.name[: -len(".tex")]
    for ext in JUNK_EXTS:
        cand = tex.with_name(stem + ext)
        if cand.is_file():
            yield cand


def orphan_texput(root: Path):
    """Yield stray ``texput.*`` files (TeX's reserved no-name jobname)."""
    for path in root.rglob("texput.*"):
        if _excluded(path, root):
            continue
        if path.is_file() and path.suffix not in (".tex", ".pdf"):
            yield path


def collect(roots) -> list[Path]:
    targets: set[Path] = set()
    for raw in roots:
        root = Path(raw).resolve()
        if not root.exists():
            logging.error("path does not exist: %s", root)
            raise FileNotFoundError(root)
        if root.is_file():
            if root.suffix == ".tex":
                targets.update(junk_for_tex(root))
            continue
        for tex in root.rglob("*.tex"):
            if not _excluded(tex, root):
                targets.update(junk_for_tex(tex))
        targets.update(orphan_texput(root))
    return sorted(targets)


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    setup_logging()
    roots = sys.argv[1:] or [ROOT]
    targets = collect(roots)
    if not targets:
        logging.info("No LaTeX build junk found.")
        return
    removed = 0
    for f in targets:
        try:
            f.unlink()
        except OSError as exc:
            # Fail loud: a locked / unremovable file is a real problem.
            logging.error("could not remove %s: %s", _display(f), exc)
            raise
        logging.info("removed %s", _display(f))
        removed += 1
    logging.info("Done: removed %d file(s).", removed)


if __name__ == "__main__":
    main()
