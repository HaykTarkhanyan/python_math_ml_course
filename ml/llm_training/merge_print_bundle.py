"""Merge the print-worthy page ranges of four LLM-training papers into one PDF.

Reading order: PPO -> DPO -> GRPO -> RoPE (each assumes the previous).
References and non-mathematical appendices are dropped; the two appendices that
carry the actual derivations (DPO A.x, GRPO A.x) are kept.
"""

import logging
from pathlib import Path

from pypdf import PdfReader, PdfWriter

REPO = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course")
SRC = REPO / "ml" / "llm_training" / "materials" / "papers"
OUT = Path(__file__).parent / "llm_rl_print_bundle.pdf"

# (filename, title, 1-based inclusive page ranges to keep)
BUNDLE = [
    ("14_ppo_1707.06347.pdf", "1. PPO (Schulman et al. 2017)", [(1, 8)]),
    ("08_dpo_2305.18290.pdf", "2. DPO (Rafailov et al. 2023)", [(1, 10)]),
    (
        "09_grpo_deepseekmath_2402.03300.pdf",
        "3. GRPO / DeepSeekMath (Shao et al. 2024)",
        [(1, 3), (11, 22)],
    ),
]

EXPECTED_PAGES = 33


def setup_logging() -> None:
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "merge_print_bundle.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)

    writer = PdfWriter()
    for filename, title, ranges in BUNDLE:
        path = SRC / filename
        reader = PdfReader(path)
        n_src = len(reader.pages)
        start_in_bundle = len(writer.pages)

        for first, last in ranges:
            if last > n_src:
                raise ValueError(f"{filename}: range {first}-{last} exceeds {n_src} pages")
            for page_no in range(first, last + 1):
                writer.add_page(reader.pages[page_no - 1])

        added = len(writer.pages) - start_in_bundle
        writer.add_outline_item(title, start_in_bundle)
        log.info(
            f"{filename}: kept {added} of {n_src} pages "
            f"({', '.join(f'{a}-{b}' for a, b in ranges)})"
        )

    total = len(writer.pages)
    if total != EXPECTED_PAGES:
        raise ValueError(f"expected {EXPECTED_PAGES} pages, built {total}")

    with open(OUT, "wb") as fh:
        writer.write(fh)
    log.info(f"wrote {OUT} ({total} pages, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
