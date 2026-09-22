"""Light shared helpers for the chapter 19 extension scripts (intro, vision, circuits, facts, diffing).

Same conventions as ``ioi_core.py`` - seed 509, logs to ``logs/<script>.log`` at the repo root,
raw results as JSON in ``results/`` before any figure is drawn - but without importing
TransformerLens, so the torch-only scripts (the toy MLP, the ResNet-18 vision figures) start fast.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

SEED = 509

CHAPTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = CHAPTER_DIR / "fig"
RESULTS_DIR = CHAPTER_DIR / "results"

# Armenian-flag colours for 3+ series (global rule), plus two neutrals.
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY, GREEN = "#666666", "#008C46"


def setup_logging(script_name: str) -> logging.Logger:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{script_name}.log", mode="w", encoding="utf-8"),
        ],
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(script_name)


def save_results(name: str, payload: dict, log: logging.Logger) -> Path:
    """Raw results as JSON first - every figure and slide number derives from this file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"wrote {path.relative_to(REPO_ROOT)}")
    return path


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the experiment script that writes it first")
    return json.loads(path.read_text(encoding="utf-8"))
