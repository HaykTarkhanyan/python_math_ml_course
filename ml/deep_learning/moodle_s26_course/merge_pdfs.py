"""Merge the PDFs under each subfolder into a single navigable PDF, preserving bookmarks.

For a target ROOT directory, this produces one merged PDF per immediate subfolder of ROOT
(default mode), so you can open a whole subfolder as one file. Each merged file gets a
**nested bookmark tree**:

    <sub-subfolder>            (folder bookmark)
        <source-file.pdf>      (one bookmark per source PDF)
            <its own bookmarks> (the source PDF's internal outline, preserved and nested)

Example (ROOT = this moodle course folder) produces `_merged/slides.pdf`,
`_merged/lab_materials.pdf`, `_merged/mock_exam.pdf`, each viewable end to end.

Usage (project venv):
    ./ma/Scripts/python.exe ml/deep_learning/moodle_s26_course/merge_pdfs.py            # merge this folder
    ./ma/Scripts/python.exe .../merge_pdfs.py  <ROOT>                                   # merge another folder
    ./ma/Scripts/python.exe .../merge_pdfs.py  <ROOT> --leaf                            # one file per leaf folder

Notes:
- Output goes to  <ROOT>/_merged/  (that folder is never itself merged, so re-running is safe).
- Requires `pypdf` (install: uv pip install --python ./ma/Scripts/python.exe pypdf).
- Fails loud: unreadable source PDFs are logged at ERROR and the run exits non-zero, but every
  PDF that CAN be merged still is (a broken file never silently vanishes without a logged error).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    """Nearest ancestor containing a .git directory; fall back to the script's folder."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return SCRIPT_DIR


def setup_logging() -> logging.Logger:
    logs_dir = find_repo_root(SCRIPT_DIR) / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("merge_pdfs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(logs_dir / "merge_pdfs.log", encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    # pypdf emits cosmetic warnings ("Annotation sizes differ") while merging beamer link
    # annotations; they do not affect the output, so keep them out of our clean log.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    return logger


def list_pdfs(directory: Path, recursive: bool):
    """PDF files under `directory`, sorted; case-insensitive suffix; skips any `_merged` dir."""
    out = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(directory):
            dirnames[:] = sorted(d for d in dirnames if d != "_merged")
            for fn in sorted(filenames):
                if fn.lower().endswith(".pdf"):
                    out.append(Path(dirpath) / fn)
    else:
        out = sorted(p for p in directory.iterdir()
                     if p.is_file() and p.suffix.lower() == ".pdf")
    return out


def _copy_internal_outline(writer, reader, items, page_offset, parent, log):
    """Recreate a source PDF's outline under `parent`, offsetting page numbers."""
    last = None
    for item in items:
        if isinstance(item, list):                      # a sublist = children of the previous item
            _copy_internal_outline(writer, reader, item, page_offset,
                                   last if last is not None else parent, log)
            continue
        try:
            pno = reader.get_destination_page_number(item)
        except Exception as e:                          # malformed destination
            log.warning(f"    skipped a bookmark (bad destination): {e}")
            continue
        if pno is None or pno < 0:
            continue
        title = str(getattr(item, "title", None) or "untitled")
        last = writer.add_outline_item(title, page_offset + pno, parent=parent)


def _ensure_folder_bookmarks(writer, root: Path, file_dir: Path, cache: dict, page: int, log):
    """Create (once) a chain of folder bookmarks for file_dir relative to root; return the deepest."""
    rel = file_dir.relative_to(root)
    if str(rel) == ".":
        return None                                     # file sits directly in the merged root
    parent = None
    cum = root
    for part in rel.parts:
        cum = cum / part
        key = str(cum)
        if key not in cache:
            cache[key] = writer.add_outline_item(part, page, parent=parent)
        parent = cache[key]
    return parent


def merge_folder(folder: Path, out_path: Path, log) -> dict:
    """Merge every PDF under `folder` (recursively) into out_path with a nested bookmark tree."""
    writer = PdfWriter()
    folder_bookmarks: dict = {}
    n_files = n_pages = 0
    failures = []

    for src in list_pdfs(folder, recursive=True):
        if src.resolve() == out_path.resolve():
            continue
        start_page = len(writer.pages)
        try:
            reader = PdfReader(str(src))
            writer.append(reader, import_outline=False)  # pages only; we rebuild bookmarks ourselves
        except Exception as e:
            log.error(f"  FAILED to read/append {src}: {e}")
            failures.append(src)
            continue
        parent_ref = _ensure_folder_bookmarks(writer, folder, src.parent,
                                              folder_bookmarks, start_page, log)
        file_ref = writer.add_outline_item(src.name, start_page, parent=parent_ref)
        try:
            _copy_internal_outline(writer, reader, reader.outline, start_page, file_ref, log)
        except Exception as e:
            log.warning(f"  could not import internal bookmarks of {src.name}: {e}")
        n_files += 1
        n_pages = len(writer.pages)
        log.info(f"  + {src.relative_to(folder)}  (now {n_pages} pages)")

    if n_files == 0:
        log.info(f"  (no PDFs) skipping {folder.name}")
        return {"files": 0, "pages": 0, "failures": failures}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        writer.write(fh)
    writer.close()
    log.info(f"  -> wrote {out_path.name}: {n_files} files, {n_pages} pages")
    return {"files": n_files, "pages": n_pages, "failures": failures}


def main():
    ap = argparse.ArgumentParser(description="Merge PDFs per subfolder, keeping bookmarks.")
    ap.add_argument("root", nargs="?", default=str(SCRIPT_DIR),
                    help="folder to process (default: this script's folder)")
    ap.add_argument("--leaf", action="store_true",
                    help="one merged file per LEAF folder (dir that directly holds PDFs) "
                         "instead of one per immediate subfolder")
    ap.add_argument("--out", default="_merged", help="output subfolder name (default: _merged)")
    args = ap.parse_args()

    log = setup_logging()
    root = Path(args.root).resolve()
    if not root.is_dir():
        log.error(f"not a directory: {root}")
        sys.exit(2)
    merged_dir = root / args.out
    log.info(f"root: {root}")
    log.info(f"mode: {'leaf folders' if args.leaf else 'immediate subfolders'}  ->  {merged_dir}")

    jobs = []  # (source_folder, output_path)
    if args.leaf:
        # every directory (any depth) that directly contains PDFs
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != args.out)
            d = Path(dirpath)
            if any(f.lower().endswith(".pdf") for f in filenames):
                rel = d.relative_to(root)
                name = "root" if str(rel) == "." else str(rel).replace(os.sep, "_")
                jobs.append((d, merged_dir / f"{name}.pdf"))
    else:
        # one file per immediate subfolder of root (recursively including everything under it)
        for child in sorted(p for p in root.iterdir() if p.is_dir() and p.name != args.out):
            if list_pdfs(child, recursive=True):
                jobs.append((child, merged_dir / f"{child.name}.pdf"))
        loose = list_pdfs(root, recursive=False)          # any PDFs sitting directly in root
        if loose:
            jobs.append((root, merged_dir / f"{root.name}_rootfiles.pdf"))

    if not jobs:
        log.error("no PDFs found under root")
        sys.exit(1)

    total_files = total_pages = 0
    all_failures = []
    for folder, out_path in jobs:
        # in leaf mode we only want the folder's OWN pdfs, not descendants:
        if args.leaf:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            res = _merge_leaf(folder, out_path, log)
        else:
            log.info(f"[{out_path.name}] merging {folder}")
            res = merge_folder(folder, out_path, log)
        total_files += res["files"]; total_pages += res["pages"]
        all_failures += res["failures"]

    log.info(f"DONE: {len([j for j in jobs])} target(s), "
             f"{total_files} source PDFs, {total_pages} total pages -> {merged_dir}")
    if all_failures:
        log.error(f"{len(all_failures)} file(s) could not be merged:")
        for f in all_failures:
            log.error(f"  - {f}")
        sys.exit(1)


def _merge_leaf(folder: Path, out_path: Path, log) -> dict:
    """Leaf mode: merge only the PDFs sitting directly in `folder` (one bookmark per file)."""
    writer = PdfWriter()
    n_files = n_pages = 0
    failures = []
    log.info(f"[{out_path.name}] merging direct PDFs of {folder}")
    for src in list_pdfs(folder, recursive=False):
        if src.resolve() == out_path.resolve():
            continue
        start_page = len(writer.pages)
        try:
            reader = PdfReader(str(src))
            writer.append(reader, import_outline=False)
        except Exception as e:
            log.error(f"  FAILED to read/append {src}: {e}")
            failures.append(src)
            continue
        file_ref = writer.add_outline_item(src.name, start_page, parent=None)
        try:
            _copy_internal_outline(writer, reader, reader.outline, start_page, file_ref, log)
        except Exception as e:
            log.warning(f"  could not import internal bookmarks of {src.name}: {e}")
        n_files += 1
        n_pages = len(writer.pages)
    if n_files:
        with open(out_path, "wb") as fh:
            writer.write(fh)
        writer.close()
        log.info(f"  -> wrote {out_path.name}: {n_files} files, {n_pages} pages")
    return {"files": n_files, "pages": n_pages, "failures": failures}


if __name__ == "__main__":
    main()
