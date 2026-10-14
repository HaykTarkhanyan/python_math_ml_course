"""Figures for the L16 CNN Foundations deck (Section 6, what the layers learn).

Generates into ml/12_cnn/fig/:
  feature_depth.pdf      -- a pretrained ResNet-18 (torchvision, ImageNet weights; the network of
                            L17 and L18) looks at the deck's astronaut photo. For each of its four
                            stages: the two maps with the most spatial contrast, with the stage's
                            map count and grid size. Shows maps getting coarser (56 -> 7 px) and
                            more numerous (64 -> 512) with depth. Channels are picked by the
                            script, not by hand.
  feature_hierarchy.pdf  -- the Distill "Feature Visualization" hero (Olah, Mordvintsev & Schubert,
                            2017, CC BY 4.0): images optimised to excite one GoogLeNet channel,
                            five depths - edges (conv2d0), textures (mixed3a), patterns (mixed4a),
                            parts (mixed4b/4c), objects (mixed4d/4e). Source sprite:
                            fig/borrowed/distill_featviz_sprite.jpg = public/images/sprite_hero.png
                            from github.com/distillpub/post--feature-visualization (5 columns =
                            the 5 depths, 6 examples each; column labels from its
                            src/diagrams/GoogleNetExamples.html). Two examples per depth, picked
                            by hand for legibility.

Both are drawn at their size on the slide (5.5 / 4.75 in wide), so font sizes are slide sizes.

Run with the project venv (~15 s, CPU: one ResNet-18 forward pass; weights from the torch cache,
a one-time ~45 MB download if missing):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/feature_hierarchy.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SEED = 509
STAGES = ["layer1", "layer2", "layer3", "layer4"]
# Distill sprite: (column = depth, rows to show), labels as in GoogleNetExamples.html
DISTILL = [("Edges", "conv2d0", (0, 2)), ("Textures", "mixed3a", (1, 4)),
           ("Patterns", "mixed4a", (0, 5)), ("Parts", "mixed4b, 4c", (4, 1)),
           ("Objects", "mixed4d, 4e", (5, 0))]
TILE = 200                                   # sprite tile size in px

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
SPRITE = FIG_DIR / "borrowed" / "distill_featviz_sprite.jpg"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_feature_hierarchy")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "feature_hierarchy.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def feature_depth(log):
    import torch
    from skimage import data
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    photo = Image.fromarray(data.astronaut()).resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(photo, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((x - mean) / std)[None]

    acts = {}
    for name in STAGES:
        getattr(model, name).register_forward_hook(
            lambda _m, _i, out, name=name: acts.__setitem__(name, out[0].detach().numpy()))
    with torch.no_grad():
        logits = model(x)
    top = weights.meta["categories"][int(logits.argmax())]
    log.info(f"ResNet-18 top-1 on the astronaut: {top}")

    expected = {"layer1": (64, 56), "layer2": (128, 28), "layer3": (256, 14), "layer4": (512, 7)}
    picks = {}
    for name in STAGES:
        a = acts[name]
        if (a.shape[0], a.shape[1]) != expected[name]:
            raise ValueError(f"{name}: shape {a.shape}, expected {expected[name]}")
        # the two maps with the most spatial contrast (std across positions): they show WHERE
        # something was found, rather than a channel that fires a little everywhere
        order = np.argsort(a.reshape(a.shape[0], -1).std(1))[::-1]
        picks[name] = order[:2].tolist()
        log.info(f"{name}: {a.shape[0]} maps of {a.shape[1]}x{a.shape[2]}, picked channels {picks[name]}")

    fig = plt.figure(figsize=(5.5, 2.25))
    gs = fig.add_gridspec(2, 5, left=0.005, right=0.995, top=0.85, bottom=0.01,
                          wspace=0.06, hspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(photo); ax.axis("off")
    ax.set_title("input\n3 x 224 x 224", fontsize=8.5, pad=3)
    for col, name in enumerate(STAGES, start=1):
        a = acts[name]
        for r, ch in enumerate(picks[name]):
            axm = fig.add_subplot(gs[r, col])
            axm.imshow(a[ch], cmap="magma", interpolation="nearest", vmin=0,
                       vmax=np.percentile(a[ch], 99.5))   # one hot pixel must not wash out the rest
            axm.set_xticks([]); axm.set_yticks([])
            if r == 0:
                axm.set_title(f"{name}\n{a.shape[0]} x {a.shape[1]} x {a.shape[2]}",
                              fontsize=8.5, pad=3)
    out = FIG_DIR / "feature_depth.pdf"
    fig.savefig(out, dpi=200); plt.close(fig)
    log.info(f"saved {out}")


def feature_hierarchy(log):
    if not SPRITE.exists():
        raise FileNotFoundError(f"missing {SPRITE} - see the module docstring for its source")
    sprite = np.asarray(Image.open(SPRITE).convert("RGB"))
    if sprite.shape[:2] != (6 * TILE, 5 * TILE):
        raise ValueError(f"sprite is {sprite.shape}, expected 6x5 tiles of {TILE} px")

    fig, axes = plt.subplots(2, 5, figsize=(4.75, 2.15))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.83, bottom=0.01, wspace=0.07, hspace=0.05)
    for col, (label, layer, rows) in enumerate(DISTILL):
        for r, row in enumerate(rows):
            tile = sprite[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]
            axes[r, col].imshow(tile); axes[r, col].axis("off")
        axes[0, col].set_title(f"$\\bf{{{label}}}$\n{layer}", fontsize=8.5, pad=3)
    log.info(f"feature hierarchy from {SPRITE.name}: " +
             ", ".join(f"{lab} rows {rows}" for lab, _, rows in DISTILL))
    out = FIG_DIR / "feature_hierarchy.pdf"
    fig.savefig(out, dpi=220); plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.perf_counter()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    feature_hierarchy(log)
    feature_depth(log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
