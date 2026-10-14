"""Where ResNet-18's weights live - per-stage parameter counts for the [50] Transfer Learning deck.

Frame "How much does each recipe actually train?": the bars show that the TOP block
(layer4) holds three quarters of all weights, so "unfreeze just the top block" trains
75% of the network, while Recipe 1 (feature extraction) trains only the 3-class head.

Counts are computed from torchvision's resnet18 architecture (weights=None - no download;
counts do not depend on the weights) with the 1000-class fc swapped for a 3-class head, and
asserted against the numbers quoted on the slide:
  conv1+bn1 9,536 | layer1 147,968 | layer2 525,568 | layer3 2,099,712 | layer4 8,393,728
  head 1,539 | total 11,178,051 (the original 1000-class model: 11,689,512 = the deck's 11.7M)

Generates into ml/12_cnn/fig/:
  trainable_params.pdf

Run with the project venv (~5 s, CPU):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/trainable_params.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
FROZEN_GRAY = "#b9b9b9"
TRAIN_GREEN = "#008c46"   # paramgreen in ml/preamble.tex

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

EXPECTED = {"conv1+bn1": 9_536, "block 1": 147_968, "block 2": 525_568,
            "block 3": 2_099_712, "block 4": 8_393_728, "new head": 1_539}
EXPECTED_TOTAL_3CLASS = 11_178_051
EXPECTED_TOTAL_1000CLASS = 11_689_512


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l51_trainable_params")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "trainable_params.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def stage_counts(log: logging.Logger) -> dict:
    torch.manual_seed(SEED)
    model = torchvision.models.resnet18(weights=None)
    total_1000 = n_params(model)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    counts = {
        "conv1+bn1": n_params(model.conv1) + n_params(model.bn1),
        "block 1": n_params(model.layer1),
        "block 2": n_params(model.layer2),
        "block 3": n_params(model.layer3),
        "block 4": n_params(model.layer4),
        "new head": n_params(model.fc),
    }
    total = n_params(model)
    for k, v in counts.items():
        log.info(f"{k:10s} {v:>10,d}  {100 * v / total:6.2f}%")
    log.info(f"total (3-class head) {total:,}; original 1000-class model {total_1000:,}")
    if counts != EXPECTED or total != EXPECTED_TOTAL_3CLASS \
            or total_1000 != EXPECTED_TOTAL_1000CLASS:
        raise AssertionError("parameter counts drifted from the numbers on the slide")
    return counts


def plot(counts: dict, log: logging.Logger) -> None:
    total = sum(counts.values())
    names = list(counts)[::-1]                    # barh draws bottom-up: conv1 on top, head last
    vals = [counts[n] for n in names]
    trained = {"block 4", "new head"}             # "unfreeze the top block" step
    colors = [TRAIN_GREEN if n in trained else FROZEN_GRAY for n in names]

    fig, ax = plt.subplots(figsize=(2.5, 1.85))   # tight bbox ~2.95 in = the 0.54 column
    bars = ax.barh(names, vals, color=colors, height=0.62)
    labels = [f"{v:,}  ({100 * v / total:.2g}%)" if v / total >= 0.01
              else f"{v:,}  ({100 * v / total:.2f}%)" for v in vals]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=7.5)
    ax.set_xlim(0, max(vals) * 1.62)              # room for the end labels
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, length=0)
    for tick in ax.get_yticklabels():             # the slide says "green = trained"
        if tick.get_text() in trained:
            tick.set_color(TRAIN_GREEN); tick.set_fontweight("bold")
    ax.set_title("ResNet-18 weights, by stage (3-class head)", fontsize=8.5, pad=4)

    out = FIG_DIR / "trainable_params.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    counts = stage_counts(log)
    plot(counts, log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
