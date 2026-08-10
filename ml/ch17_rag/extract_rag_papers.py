"""Extract LLM-readable text from the RAG source papers.

Mirrors ml/text_embedding/extract_embedding_papers.py: plain pdftotext output in
reading order, with `=== PAGE n ===` markers so any passage can be traced back to
a PDF page. Also asserts each PDF is the paper we think it is, by checking an
expected title fragment against the extracted page-1 text.
"""

import logging
import subprocess
from pathlib import Path

from pypdf import PdfReader

REPO = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course")
SRC = REPO / "ml" / "ch17_rag" / "papers"
OUT = SRC / "llm_readable"

# stem -> lowercase fragment that must appear in the first page of extracted text
EXPECTED_TITLE = {
    "01_bm25_robertson_zaragoza_2009": "the probabilistic relevance framework",
    "02_dpr_2004.04906": "dense passage retrieval",
    "03_colbert_2004.12832": "colbert",
    "04_hnsw_1603.09320": "hierarchical navigable small world",
    "05_rag_lewis_2005.11401": "retrieval-augmented generation",
    "06_ragas_2309.15217": "ragas",
}


def setup_logging() -> None:
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "extract_rag_papers.log", encoding="utf-8"),
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

    total_pages = 0
    for pdf in pdfs:
        n_pages = len(PdfReader(pdf).pages)
        text = extract(pdf)

        n_marked = text.count("=== PAGE ")
        if n_marked != n_pages:
            raise ValueError(f"{pdf.name}: {n_pages} PDF pages but {n_marked} extracted")

        words = len(text.split())
        if words < 500:
            raise ValueError(f"{pdf.name}: only {words} words extracted, likely a scanned PDF")

        if pdf.stem not in EXPECTED_TITLE:
            raise KeyError(f"{pdf.name}: no expected title registered in EXPECTED_TITLE")
        fragment = EXPECTED_TITLE[pdf.stem]
        first_page = text.split("=== PAGE 2 ===")[0].lower()
        if fragment not in first_page:
            raise ValueError(f"{pdf.name}: expected title fragment {fragment!r} not on page 1")

        dest = OUT / f"{pdf.stem}.txt"
        dest.write_text(text, encoding="utf-8")
        total_pages += n_pages
        log.info(f"{pdf.name}: {n_pages} pages -> {dest.name} ({words:,} words)")

    log.info(f"extracted {len(pdfs)} papers ({total_pages} pages) to {OUT}")


if __name__ == "__main__":
    main()
