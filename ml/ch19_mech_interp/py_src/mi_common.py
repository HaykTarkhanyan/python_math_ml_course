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


def stream_ticks(tl_labels: list[str]) -> list[str]:
    """Tick labels for a logit-lens x axis, in the decks' convention: "emb", then "after layer L"
    with layers counted from 0 like the head names (L9H9 is in layer 9).

    TransformerLens's accumulated_resid labels a point by the block it ENTERS: "k_pre" is the
    stream after layers 0..k-1, and "final_post" is the stream after the last layer. Printing
    those raw names put "9_pre" under the point that is really "after layer 8", and one deck frame
    read a layer number off by one because of it (2026-10-17). Fails loudly on any other label.
    """
    out = []
    for k, lab in enumerate(tl_labels):
        if lab == f"{k}_pre":
            out.append("emb" if k == 0 else str(k - 1))
        elif lab == "final_post" and k == len(tl_labels) - 1:
            out.append(str(k - 1))
        else:
            raise ValueError(f"unexpected accumulated_resid label {lab!r} at index {k}")
    return out


def pick_device() -> str:
    """"cuda" when a GPU is present (the heavy runs go to Colab), otherwise "cpu"."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def save_results(name: str, payload: dict, log: logging.Logger, compact: bool = False) -> Path:
    """Raw results as JSON first - every figure and slide number derives from this file.

    compact=True drops the indentation: for files that store images, indent=2 puts every pixel on its
    own line and roughly triples the size (vision_features.json: 12 MB indented, ~4 MB compact)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    text = json.dumps(payload, separators=(",", ":")) if compact else json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    log.info(f"wrote {path.relative_to(REPO_ROOT)}")
    return path


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the experiment script that writes it first")
    return json.loads(path.read_text(encoding="utf-8"))
