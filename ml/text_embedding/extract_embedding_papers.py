"""Extract LLM-readable text from the text-embedding PDFs.

Mirrors ml/llm_training/materials_md/: plain pdftotext output in reading order.
Adds `=== PAGE n ===` markers so any passage can be traced back to a PDF page.
"""

import logging
import subprocess
from pathlib import Path

from pypdf import PdfReader

REPO = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course")
SRC = REPO / "ml" / "text_embedding" / "papers"
OUT = SRC / "llm_readable"


def setup_logging() -> None:
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "extract_embedding_papers.log", encoding="utf-8"),
        ],
    )


def extract(pdf: Path) -> str:
    """pdftotext to stdout, then split its form-feed page breaks into markers."""
    proc = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True,
        check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return "\n".join(
        f"=== PAGE {n} ===\n{body.strip()}" for n, body in enumerate(pages, 1)
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)
    OUT.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(SRC.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {SRC}")

    for pdf in pdfs:
        n_pages = len(PdfReader(pdf).pages)
        text = extract(pdf)

        n_marked = text.count("=== PAGE ")
        if n_marked != n_pages:
            raise ValueError(f"{pdf.name}: {n_pages} PDF pages but {n_marked} extracted")

        words = len(text.split())
        if words < 500:
            raise ValueError(f"{pdf.name}: only {words} words extracted, likely a scanned PDF")

        dest = OUT / f"{pdf.stem}.txt"
        dest.write_text(text, encoding="utf-8")
        log.info(f"{pdf.name}: {n_pages} pages -> {dest.name} ({words:,} words)")

    log.info(f"extracted {len(pdfs)} papers to {OUT}")


if __name__ == "__main__":
    main()
