"""ResNet-18, stage by stage, with its tensor shapes - for the [50] Transfer Learning deck.

Frame "ResNet-18: a trunk and a head": the net every recipe in the deck reuses. Five stages
shrink the photo and grow the channels (56x56x64 -> 7x7x512), global average pooling turns
the 512 final maps into 512 numbers, and one linear layer (the head) turns those into class
scores. The braces mark the trunk (conv1 .. avgpool) and the head (fc) - the names the
"Two recipes" frame freezes and swaps.

Shapes come from pushing a zero photo through torchvision's resnet18 (weights=None - no
download; shapes do not depend on the weights) with forward hooks, and are asserted, so the
labels cannot drift from what PyTorch produces. The script also checks the name "18":
conv1 + 16 3x3 convs in the four blocks + fc = 18 weight layers (the three 1x1 shortcut
convs are not counted, as in He et al. 2015).

Generates into ml/12_cnn/fig/:
  resnet18_arch.pdf

Run with the project venv (~4 s, CPU):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/resnet18_arch.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch
import torchvision

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
TRUNK_FILL = "#cfdcf2"
HEAD_GREEN = "#008c46"        # paramgreen in ml/preamble.tex
HEAD_FILL = "#cfeedd"
DARK = "#333333"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
PHOTO = FIG_DIR / "src_pomegranate.jpg"

STAGES = ["maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
EXPECTED = {"maxpool": (56, 56, 64), "layer1": (56, 56, 64), "layer2": (28, 28, 128),
            "layer3": (14, 14, 256), "layer4": (7, 7, 512), "avgpool": (1, 1, 512),
            "fc": (3,)}
FS = 7.5


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l51_resnet18_arch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "resnet18_arch.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def traced_shapes(log: logging.Logger) -> dict:
    """Push a zero 224x224 photo through resnet18 (3-class fc) and record each stage's output."""
    torch.manual_seed(SEED)
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.eval()
    shapes = {}

    def hook(name):
        def fn(_module, _inp, out):
            s = tuple(out.shape[1:])
            shapes[name] = (s[1], s[2], s[0]) if len(s) == 3 else s   # (C,H,W) -> (H,W,C)
        return fn

    for name in STAGES:
        getattr(model, name).register_forward_hook(hook(name))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    for name, s in shapes.items():
        log.info(f"{name:8s} {'x'.join(map(str, s))}")
    if shapes != EXPECTED:
        raise AssertionError(f"ResNet-18 shapes drifted: {shapes}")

    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    n_weight_layers = sum(m.kernel_size != (1, 1) for m in convs) + 1     # + fc
    log.info(f"conv layers {len(convs)} (incl. 1x1 shortcuts); weight layers {n_weight_layers}")
    if n_weight_layers != 18:
        raise AssertionError(f"expected 18 weight layers, counted {n_weight_layers}")
    return shapes


def draw(shapes: dict, log: logging.Logger) -> None:
    W, H = 5.45, 1.60
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    yc = 0.80                                    # centre line of every box
    y_lab = 0.31                                 # top of the shape labels under the boxes

    def arrow(x0, x1, y=yc):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
                                     color="black", lw=0.9, shrinkA=0, shrinkB=0))

    # photo, centre-cropped to a square like the net's 224x224 input
    img = plt.imread(PHOTO)
    h_img, w_img = img.shape[:2]
    side = min(h_img, w_img)
    top, left = (h_img - side) // 2, (w_img - side) // 2
    img = img[top:top + side, left:left + side]
    x_ph, ph = 0.10, 0.46
    ax.imshow(img, extent=(x_ph, x_ph + ph, yc - ph / 2, yc + ph / 2), aspect="auto",
              zorder=1)
    ax.text(x_ph + ph / 2, y_lab, "photo\n224x224x3", ha="center", va="top", fontsize=FS,
            linespacing=1.1)

    # five stages: box height ~ map size, box width ~ channels
    heights = {56: 0.85, 28: 0.62, 14: 0.44, 7: 0.32}
    widths = {64: 0.42, 128: 0.46, 256: 0.50, 512: 0.54}
    names = {"maxpool": "conv1\n+ pool", "layer1": "block 1", "layer2": "block 2",
             "layer3": "block 3", "layer4": "block 4"}
    gap = 0.11
    x = x_ph + ph + 0.15
    arrow(x_ph + ph + 0.02, x)
    edges = {}
    for i, stage in enumerate(["maxpool", "layer1", "layer2", "layer3", "layer4"]):
        h_px, _, c = shapes[stage]
        w, h = widths[c], heights[h_px]
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, fc=TRUNK_FILL, ec="black", lw=0.6))
        ax.text(x + w / 2, yc, names[stage], ha="center", va="center", fontsize=FS,
                linespacing=1.0)
        ax.text(x + w / 2, y_lab, f"{c} maps\n{h_px}x{h_px}", ha="center", va="top",
                fontsize=FS, linespacing=1.1)
        edges[stage] = (x, x + w)
        if i < 4:
            arrow(x + w, x + w + gap)
        x += w + gap

    # global average pool: 512 maps -> 512 numbers (a thin tall bar = a vector)
    x_b4_end = edges["layer4"][1]
    x_vec = x_b4_end + 0.50
    arrow(x_b4_end, x_vec - 0.02)
    ax.text((x_b4_end + x_vec) / 2, yc + 0.05, "average\neach map", ha="center", va="bottom",
            fontsize=6.8, linespacing=1.0, color=DARK)
    vec_h = 0.72
    ax.add_patch(Rectangle((x_vec, yc - vec_h / 2), 0.07, vec_h, fc=ARM_ORANGE, ec="black",
                           lw=0.6))
    n_feat = shapes["avgpool"][2]
    ax.text(x_vec + 0.035, y_lab, f"{n_feat}\nnumbers", ha="center", va="top", fontsize=FS,
            linespacing=1.1)

    # head: one linear layer -> class scores
    x_fc = x_vec + 0.07 + 0.17
    arrow(x_vec + 0.07, x_fc)
    fc_w, fc_h = 0.30, 0.40
    ax.add_patch(Rectangle((x_fc, yc - fc_h / 2), fc_w, fc_h, fc=HEAD_FILL, ec=HEAD_GREEN,
                           lw=1.0))
    ax.text(x_fc + fc_w / 2, yc, "fc", ha="center", va="center", fontsize=FS)
    x_out = x_fc + fc_w + 0.13
    arrow(x_fc + fc_w, x_out - 0.02)
    ax.add_patch(Rectangle((x_out, yc - 0.10), 0.07, 0.20, fc=HEAD_GREEN, ec="black", lw=0.6))
    ax.text(x_out + 0.12, yc, f"{shapes['fc'][0]} grades\n(ImageNet:\n1000)",
            ha="left", va="center", fontsize=FS, linespacing=1.1)

    # brackets: trunk (conv1 .. avgpool) and head (fc)
    def bracket(x0, x1, y, color, text):
        ax.plot([x0, x0, x1, x1], [y - 0.06, y, y, y - 0.06], color=color, lw=1.0)
        ax.text((x0 + x1) / 2, y + 0.04, text, ha="center", va="bottom", fontsize=FS,
                color=color, fontweight="bold")

    y_br = yc + heights[56] / 2 + 0.10
    bracket(edges["maxpool"][0], x_vec + 0.07, y_br, ARM_BLUE,
            "trunk (backbone): photo in, 512 numbers out")
    bracket(x_fc - 0.03, x_fc + fc_w + 0.03, y_br, HEAD_GREEN, "head")

    out = FIG_DIR / "resnet18_arch.pdf"
    fig.savefig(out)                              # fixed geometry - no tight bbox
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    draw(traced_shapes(log), log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
