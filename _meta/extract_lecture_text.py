"""
Extract text from Beamer lecture PDFs (L00–L14) and save as .txt files in _meta/lecture_texts/.
Uses pymupdf (fitz) to read each page/slide and writes one file per lecture.

Usage:
    python _meta/extract_lecture_text.py
"""

import os
import glob
import pymupdf  # pip install pymupdf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LECTURES_DIR = os.path.join(REPO_ROOT, "math", "Lectures")
OUTPUT_DIR = os.path.join(REPO_ROOT, "_meta", "lecture_texts")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF, labelled by slide number."""
    doc = pymupdf.open(pdf_path)
    parts = []
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            parts.append(f"--- Slide {i} ---\n{text}")
    doc.close()
    return "\n\n".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find L00–L14 PDFs (skip notes, merged, etc.)
    pdfs = sorted(glob.glob(os.path.join(LECTURES_DIR, "L[0-9][0-9]_*.pdf")))

    if not pdfs:
        print(f"No lecture PDFs found in {LECTURES_DIR}")
        return

    for pdf_path in pdfs:
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")

        print(f"Extracting: {os.path.basename(pdf_path)} ... ", end="")
        text = extract_pdf_text(pdf_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{len(text):,} chars -> {os.path.basename(out_path)}")

    print(f"\nDone. {len(pdfs)} files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
