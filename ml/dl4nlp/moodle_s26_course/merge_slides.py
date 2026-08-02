"""Merge the DL4NLP Moodle lecture decks into one navigable PDF.

The result is ``_merged/slides.pdf``.  Its bookmark tree mirrors the source
folders (weeks -> chapters -> source deck) and nests any bookmarks embedded in
the individual PDFs below their source-deck bookmark.

Usage:
    ./ma/Scripts/python.exe ml/dl4nlp/moodle_s26_course/merge_slides.py
    ./ma/Scripts/python.exe ml/dl4nlp/moodle_s26_course/merge_slides.py --out path/to/slides.pdf

The script deliberately includes only the two lecture-slide folders.  It does
not merge the separate ``exams/`` PDFs in this Moodle archive.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCES = (
    ("Weeks 1-8", SCRIPT_DIR / "slides_weeks1-8"),
    ("Weeks 9-13", SCRIPT_DIR / "slides_weeks9-13"),
)


def find_repo_root(start: Path) -> Path:
    """Return the nearest Git repository root."""
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    return SCRIPT_DIR


def setup_logging() -> logging.Logger:
    """Log both to the terminal and the repository's existing logs directory."""
    logger = logging.getLogger("merge_dl4nlp_slides")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logs_dir = find_repo_root(SCRIPT_DIR) / "logs"
    logs_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        logs_dir / "merge_dl4nlp_slides.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Beamer slide annotations can emit cosmetic pypdf warnings while copying.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    return logger


def list_pdfs(folder: Path) -> list[Path]:
    """Return every source PDF below *folder* in deterministic course order."""
    pdfs: list[Path] = []
    for directory, subdirectories, filenames in os.walk(folder):
        subdirectories.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(".pdf"):
                pdfs.append(Path(directory) / filename)
    return pdfs


def copy_internal_outline(
    writer: PdfWriter,
    reader: PdfReader,
    items: list,
    page_offset: int,
    parent: object,
    log: logging.Logger,
) -> None:
    """Recreate a source PDF outline under *parent*, with adjusted page numbers."""
    last_item = None
    for item in items:
        if isinstance(item, list):
            copy_internal_outline(
                writer,
                reader,
                item,
                page_offset,
                last_item if last_item is not None else parent,
                log,
            )
            continue
        try:
            page_number = reader.get_destination_page_number(item)
        except Exception as error:  # A bad source bookmark should not stop the merge.
            log.warning("    skipped a bookmark with a bad destination: %s", error)
            continue
        if page_number is None or page_number < 0:
            continue
        title = str(getattr(item, "title", None) or "untitled")
        last_item = writer.add_outline_item(
            title, page_offset + page_number, parent=parent
        )


def add_folder_bookmarks(
    writer: PdfWriter,
    source_root: Path,
    file_directory: Path,
    top_bookmark: object,
    cache: dict[Path, object],
    page: int,
) -> object:
    """Create the relative folder-bookmark chain once and return its deepest item."""
    parent = top_bookmark
    current = source_root
    for part in file_directory.relative_to(source_root).parts:
        current /= part
        if current not in cache:
            cache[current] = writer.add_outline_item(part, page, parent=parent)
        parent = cache[current]
    return parent


def merge_slides(output: Path, log: logging.Logger) -> tuple[int, int, list[Path]]:
    """Create the merged slide PDF and return (files, pages, unreadable_files)."""
    writer = PdfWriter()
    file_count = 0
    failures: list[Path] = []

    for source_title, source_root in SOURCES:
        pdfs = list_pdfs(source_root)
        if not pdfs:
            log.error("No PDFs found in %s", source_root)
            failures.append(source_root)
            continue

        # pypdf cannot resolve a bookmark added to an as-yet-empty writer.  Add
        # the group bookmark immediately after its first deck has supplied pages.
        top_bookmark = None
        folder_bookmarks: dict[Path, object] = {}
        for source_pdf in pdfs:
            start_page = len(writer.pages)
            try:
                reader = PdfReader(str(source_pdf))
                writer.append(reader, import_outline=False)
            except Exception as error:
                log.error("FAILED to read/append %s: %s", source_pdf, error)
                failures.append(source_pdf)
                continue

            if top_bookmark is None:
                top_bookmark = writer.add_outline_item(source_title, start_page)
            parent = add_folder_bookmarks(
                writer,
                source_root,
                source_pdf.parent,
                top_bookmark,
                folder_bookmarks,
                start_page,
            )
            deck_bookmark = writer.add_outline_item(
                source_pdf.name, start_page, parent=parent
            )
            try:
                copy_internal_outline(
                    writer,
                    reader,
                    reader.outline,
                    start_page,
                    deck_bookmark,
                    log,
                )
            except Exception as error:
                log.warning(
                    "Could not import internal bookmarks from %s: %s",
                    source_pdf.name,
                    error,
                )
            file_count += 1
            log.info("+ %s", source_pdf.relative_to(SCRIPT_DIR))

    if not file_count:
        writer.close()
        return 0, 0, failures

    page_count = len(writer.pages)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".tmp.pdf")
    try:
        with temporary_output.open("wb") as file_handle:
            writer.write(file_handle)
        temporary_output.replace(output)
    finally:
        writer.close()
        if temporary_output.exists():
            temporary_output.unlink()
    return file_count, page_count, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all DL4NLP lecture decks, preserving their bookmarks."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "_merged" / "slides.pdf",
        help="output PDF (default: %(default)s)",
    )
    args = parser.parse_args()

    log = setup_logging()
    output = args.out.resolve()
    log.info("Writing %s", output)
    file_count, page_count, failures = merge_slides(output, log)
    if failures:
        log.error("%d source(s) could not be merged:", len(failures))
        for failure in failures:
            log.error("- %s", failure)
        raise SystemExit(1)
    log.info("DONE: %d decks, %d pages -> %s", file_count, page_count, output)


if __name__ == "__main__":
    main()
