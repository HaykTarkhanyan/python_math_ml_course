"""Payoff figure for the L18 Transfer Learning deck (Section 1).

Generates into ml/ch6_cnn/fig/:
  transfer_curves.pdf -- from-scratch vs feature-extraction vs fine-tune validation
                         accuracy on small data.

Two modes:
  1. REAL: if ml/ch6_cnn/py_src/data/hw3_metrics.csv exists (saved by the HW3 Colab
     notebook), plot those measured curves. Required columns: epoch, from_scratch,
     feature_extraction, fine_tune (val accuracy in [0, 1]). Fails loudly on a
     malformed file.
  2. SCHEMATIC (fallback while the HW3 notebook does not exist yet): draw the three
     canonical curve SHAPES with unlabeled axes and a printed disclaimer
     "schematic - typical shapes, real curves come with HW3". No invented numbers
     are presented as measurements.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/transfer_curves.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette for 3-color charts.
"""

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
METRICS = HERE.parent / "data" / "hw3_metrics.csv"

# Armenian flag palette (3+ colors rule).
COL_SCRATCH = "#D90012"   # red
COL_FEAT = "#0033A0"      # blue
COL_FINETUNE = "#F2A800"  # orange

REQUIRED_COLS = ["epoch", "from_scratch", "feature_extraction", "fine_tune"]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l18_transfer_curves")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "transfer_curves.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_metrics(path: Path, log) -> dict:
    """Parse the HW3 metrics CSV; raise ValueError on anything malformed."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = {c: [] for c in REQUIRED_COLS}
    for i, row in enumerate(rows):
        for c in REQUIRED_COLS:
            try:
                out[c].append(float(row[c]))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path} row {i + 1}, column '{c}': "
                                 f"non-numeric value {row[c]!r}") from e
    for c in REQUIRED_COLS[1:]:
        bad = [v for v in out[c] if not 0.0 <= v <= 1.0]
        if bad:
            raise ValueError(f"{path} column '{c}' has accuracies outside [0, 1]: {bad[:3]}")
    log.info(f"loaded {len(rows)} epochs of measured HW3 metrics from {path}")
    return out


def plot_real(m: dict, log):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(m["epoch"], m["from_scratch"], color=COL_SCRATCH, lw=2.2,
            label="from scratch")
    ax.plot(m["epoch"], m["feature_extraction"], color=COL_FEAT, lw=2.2,
            label="feature extraction (frozen trunk)")
    ax.plot(m["epoch"], m["fine_tune"], color=COL_FINETUNE, lw=2.2,
            label="fine-tune (small LR)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Transfer learning on small data (HW3, measured)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_schematic(log):
    log.warning(f"metrics file not found at {METRICS} - emitting the SCHEMATIC "
                f"(HW3 notebook does not exist yet); no numbers are presented as measured")
    x = np.linspace(0, 30, 300)
    scratch = 0.06 + 0.50 * (1 - np.exp(-x / 14.0))
    feat = 0.15 + 0.63 * (1 - np.exp(-x / 2.5))
    fine = 0.15 + 0.73 * (1 - np.exp(-x / 5.0))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(x, scratch, color=COL_SCRATCH, lw=2.4)
    ax.plot(x, feat, color=COL_FEAT, lw=2.4)
    ax.plot(x, fine, color=COL_FINETUNE, lw=2.4)
    ax.text(30.5, scratch[-1], "from scratch:\nslow, plateaus low",
            color=COL_SCRATCH, fontsize=10, va="center")
    ax.text(30.5, feat[-1] - 0.03, "feature extraction:\nfast, good",
            color=COL_FEAT, fontsize=10, va="center")
    ax.text(30.5, fine[-1] + 0.04, "fine-tune:\nbest ceiling",
            color=COL_FINETUNE, fontsize=10, va="center")

    # deliberately unlabeled axes: shapes, not measurements
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("epochs", fontsize=11)
    ax.set_ylabel("validation accuracy", fontsize=11)
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same small dataset, three ways to train", fontsize=13)
    ax.text(0.02, 0.965,
            "schematic - typical shapes, real curves come with HW3",
            transform=ax.transAxes, fontsize=9.5, color="gray",
            style="italic", va="top")
    return fig


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    if METRICS.exists():
        fig = plot_real(load_metrics(METRICS, log), log)
    else:
        fig = plot_schematic(log)

    out = FIG_DIR / "transfer_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
