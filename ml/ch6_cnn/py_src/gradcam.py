"""Real figures for the L18 Transfer Learning deck (Section 2, Grad-CAM).

Generates into ml/ch6_cnn/fig/:
  gradcam.pdf         -- ImageNet-pretrained resnet18's Grad-CAM heatmap overlaid on the
                         chapter's pomegranate photo (40% alpha), with the model's actual
                         top-1 prediction printed honestly in the panel title.
  gradcam_anim_0.pdf  -- ANIM flip-book frame 1: the input photo (224 x 224 center crop).
  gradcam_anim_1.pdf  -- frame 2: the raw 7 x 7 class-weighted activation map.
  gradcam_anim_2.pdf  -- frame 3: the map upsampled and overlaid on the photo.

Grad-CAM per Selvaraju et al. 2017 (arXiv:1610.02391): take the last conv block's
feature maps A^k, weight each by alpha_k = the global-average-pooled gradient of the
top class score w.r.t. A^k, sum, ReLU, upsample. ONE forward + ONE backward on CPU -
nothing trains locally.

Fails loudly if the photo or the pretrained weights are unavailable.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/gradcam.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
PHOTO = FIG_DIR / "src_pomegranate.jpg"

# ImageNet normalization statistics (the pretrained model's training stats -
# same values as the official PyTorch transfer-learning tutorial).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ANIM_FIGSIZE = (5.4, 5.9)  # identical for all flip-book frames: no click-jitter


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l18_gradcam")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "gradcam.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_inputs(log):
    """Return (normalized model input 1x3x224x224, display image 224x224x3 float)."""
    if not PHOTO.exists():
        raise FileNotFoundError(f"chapter photo missing: {PHOTO}")
    img = Image.open(PHOTO).convert("RGB")
    log.info(f"loaded {PHOTO.name}, size {img.size}")
    base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    cropped = base(img)
    display = np.asarray(cropped, dtype=float) / 255.0
    model_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(cropped).unsqueeze(0)
    return model_in, display


def grad_cam(model_in, log):
    """One forward + one backward through pretrained resnet18; return cam7, cam224,
    top-1 (name, prob) and the top-3 list."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)  # raises if weights cannot be found/downloaded
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"resnet18 loaded, {n_params:,} parameters")

    store = {}

    def fwd_hook(module, inp, out):
        store["acts"] = out
        out.register_hook(lambda g: store.__setitem__("grads", g))

    handle = model.layer4.register_forward_hook(fwd_hook)  # last conv block
    logits = model(model_in)
    probs = F.softmax(logits.detach(), dim=1)[0]
    top3 = torch.topk(probs, 3)
    categories = weights.meta["categories"]
    for p, i in zip(top3.values, top3.indices):
        log.info(f"predicted: {categories[i]} (p = {p.item():.3f})")
    top_idx = int(top3.indices[0])
    top_name = categories[top_idx]
    top_prob = float(top3.values[0])

    model.zero_grad()
    logits[0, top_idx].backward()
    handle.remove()
    if "grads" not in store:
        raise RuntimeError("backward hook never fired - no gradients captured")

    acts = store["acts"][0]            # (512, 7, 7)
    grads = store["grads"][0]          # (512, 7, 7)
    alpha = grads.mean(dim=(1, 2))     # GAP of gradients -> channel weights
    cam = F.relu((alpha[:, None, None] * acts).sum(0))  # (7, 7)
    if float(cam.max()) <= 0:
        raise RuntimeError("Grad-CAM map is all zeros - something is wrong")
    cam = cam / cam.max()
    cam224 = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear",
                           align_corners=False)[0, 0]
    log.info(f"cam 7x7 range [0, 1], upsampled to {tuple(cam224.shape)}")
    top3_list = [(categories[int(i)], float(p)) for p, i in zip(top3.values, top3.indices)]
    return cam.detach().numpy(), cam224.detach().numpy(), top_name, top_prob, top3_list


def anim_frame(path, title, draw, log):
    fig, ax = plt.subplots(figsize=ANIM_FIGSIZE)
    draw(ax)
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {path}")


def main():
    log = setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    model_in, display = load_inputs(log)
    cam7, cam224, top_name, top_prob, top3 = grad_cam(model_in, log)

    # --- static figure: photo | overlay ---------------------------------
    # per-pixel alpha baked into an RGBA overlay (peak ~0.45, fading where the
    # evidence is low): the heatmap reads as a spotlight instead of a uniform
    # tint over an already-red photo
    overlay_rgba = plt.get_cmap("jet")(cam224)
    overlay_rgba[..., 3] = 0.5 * cam224 ** 1.5
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    axes[0].imshow(display)
    axes[0].set_title("input photo", fontsize=12)
    axes[1].imshow(display)
    axes[1].imshow(overlay_rgba)
    axes[1].set_title(f"Grad-CAM for '{top_name}' (p = {top_prob:.2f})", fontsize=12)
    for ax in axes:
        ax.axis("off")
    out = FIG_DIR / "gradcam.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")

    # --- ANIM flip-book: photo -> coarse 7x7 -> overlay ------------------
    anim_frame(FIG_DIR / "gradcam_anim_0.pdf",
               "the photo the network sees (224 x 224)",
               lambda ax: ax.imshow(display), log)
    anim_frame(FIG_DIR / "gradcam_anim_1.pdf",
               f"class-weighted activation map: 7 x 7 ('{top_name}')",
               lambda ax: ax.imshow(cam7, cmap="jet", interpolation="nearest"), log)

    def draw_overlay(ax):
        ax.imshow(display)
        ax.imshow(overlay_rgba)

    anim_frame(FIG_DIR / "gradcam_anim_2.pdf",
               "upsampled and overlaid on the photo",
               draw_overlay, log)

    log.info(f"done. top-3: {top3}")


if __name__ == "__main__":
    main()
