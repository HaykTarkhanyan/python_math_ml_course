"""Real figures for the L18 Transfer Learning deck (Section 2, Grad-CAM).

Generates into ml/12_cnn/fig/:
  gradcam.pdf              -- ImageNet-pretrained resnet18's Grad-CAM heatmap overlaid on the
                              chapter's pomegranate photo, with the model's actual top-1 prediction
                              printed honestly in the panel title.
  gradcam_anim_0..2.pdf    -- overview flip-book: the photo, the 7 x 7 heatmap, the overlay.
  gradcam_step1_maps.pdf   -- (2026-10-01) step 1: the photo and six of the 512 feature maps (7 x 7)
                              the last conv block produces for it.
  gradcam_step2_grads.pdf  -- step 2: for the same six maps, the gradient of the class score with
                              respect to every cell, and its average alpha_k (the map's weight).
  gradcam_step3_sum.pdf    -- step 3: alpha_k * A^k for the three heaviest maps, "+ ... (512
                              terms)", the weighted sum, and ReLU of it (the 7 x 7 heatmap).
  gradcam_step4_upsample.pdf -- step 4: the 7 x 7 heatmap drawn blocky (nearest), the same map
                              bilinearly upsampled to 224 x 224, and the overlay on the photo.
The six maps of steps 1-3 are picked by the script: the three with the largest positive alpha and
the three with the most negative alpha (instructor, l18_todo.md: "show the gradients themselves").

Grad-CAM per Selvaraju et al. 2017 (arXiv:1610.02391): take the last conv block's feature maps
A^k, weight each by alpha_k = the average over the 7 x 7 positions of d(score)/d(A^k), sum, ReLU,
upsample. ONE forward + ONE backward on CPU - nothing trains locally.

All figures are drawn at their size on the slide, so font sizes are slide sizes (7.5-9 pt).
Fails loudly if the photo or the pretrained weights are unavailable.

Run with the project venv (~15 s, mostly importing torch):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/gradcam.py
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
GREEN, RED = "#008C46", "#D90012"

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

ANIM_FIGSIZE = (1.95, 2.05)  # identical for all flip-book frames: no click-jitter; slide size


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
    """One forward + one backward through pretrained resnet18. Returns a dict with the maps,
    their gradients, the weights alpha, the raw weighted sum, the 7x7 and 224x224 heatmaps,
    and the top-1 / top-3 predictions."""
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

    model.zero_grad()
    logits[0, top_idx].backward()
    handle.remove()
    if "grads" not in store:
        raise RuntimeError("backward hook never fired - no gradients captured")

    acts = store["acts"][0].detach()        # (512, 7, 7)
    grads = store["grads"][0].detach()      # (512, 7, 7)
    if tuple(acts.shape) != (512, 7, 7):
        raise ValueError(f"unexpected layer4 shape {tuple(acts.shape)}")
    alpha = grads.mean(dim=(1, 2))          # one weight per map
    # ResNet-18's head is global average pooling + one dense layer: score = sum_k w_k * mean(A^k),
    # so d(score)/d(A^k_ij) = w_k / 49 in EVERY cell, and alpha_k is the head's weight / 49.
    # Checked here, because the slide says so (Grad-CAM reduces to CAM, Zhou et al. 2016).
    w_head = model.fc.weight[top_idx].detach() / 49.0
    spread = float((grads - grads.mean(dim=(1, 2), keepdim=True)).abs().max())
    gap = float((alpha - w_head).abs().max())
    if spread > 1e-7 or gap > 1e-7:
        raise ValueError(f"gradient not constant per map (spread {spread:.1e}) or alpha != w/49 "
                         f"(gap {gap:.1e}) - the GAP + linear-head argument no longer holds")
    log.info(f"checked: each map's gradient is constant over its 49 cells (max spread "
             f"{spread:.1e}) and alpha_k = fc.weight[class, k] / 49 (max gap {gap:.1e})")
    weighted = (alpha[:, None, None] * acts).sum(0)          # (7, 7), can be negative
    cam = F.relu(weighted)
    if float(cam.max()) <= 0:
        raise RuntimeError("Grad-CAM map is all zeros - something is wrong")
    cam = cam / cam.max()
    cam224 = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear",
                           align_corners=False)[0, 0]
    log.info(f"alpha range [{float(alpha.min()):.2e}, {float(alpha.max()):.2e}]; "
             f"{int((alpha > 0).sum())} of 512 maps have positive weight")
    return {"acts": acts.numpy(), "grads": grads.numpy(), "alpha": alpha.numpy(),
            "weighted": weighted.numpy(), "cam7": cam.numpy(), "cam224": cam224.numpy(),
            "top_name": categories[top_idx], "top_prob": float(top3.values[0]),
            "top3": [(categories[int(i)], float(p)) for p, i in zip(top3.values, top3.indices)]}


def overlay_rgba(cam224):
    # per-pixel alpha (peak ~0.5, fading where the evidence is low): the heatmap reads as a
    # spotlight instead of a uniform tint over an already-red photo
    rgba = plt.get_cmap("jet")(cam224)
    rgba[..., 3] = 0.5 * cam224 ** 1.5
    return rgba


def anim_frame(path, title, draw, log):
    fig, ax = plt.subplots(figsize=ANIM_FIGSIZE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    draw(ax)
    ax.axis("off")
    ax.set_title(title, fontsize=8, pad=3)
    fig.savefig(path)              # fixed geometry: no click-jitter
    plt.close(fig)
    log.info(f"saved {path}")


def show_map(ax, m, cmap="viridis", sym=False):
    if sym:
        v = float(np.abs(m).max())
        ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    else:
        ax.imshow(m, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def step_figures(r, display, log):
    acts, grads, alpha = r["acts"], r["grads"], r["alpha"]
    order = np.argsort(alpha)
    pick = list(order[::-1][:3]) + list(order[:3])           # 3 heaviest +, 3 most -
    log.info("step maps (channel, alpha): " +
             ", ".join(f"({k}, {alpha[k]:+.2e})" for k in pick))

    # step 1: photo + six maps
    fig, axes = plt.subplots(1, 7, figsize=(5.5, 1.15),
                             gridspec_kw={"width_ratios": [1.25] + [1] * 6})
    fig.subplots_adjust(left=0.005, right=0.995, top=0.8, bottom=0.02, wspace=0.12)
    axes[0].imshow(display); axes[0].axis("off")
    axes[0].set_title("photo", fontsize=8, pad=2)
    for ax, k in zip(axes[1:], pick):
        show_map(ax, acts[k])
        ax.set_title(f"map {k}", fontsize=8, pad=2)
    out = FIG_DIR / "gradcam_step1_maps.pdf"
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")

    # step 2: maps (top) and their gradients (bottom), alpha under each
    fig, axes = plt.subplots(2, 6, figsize=(5.5, 2.15))
    fig.subplots_adjust(left=0.1, right=0.995, top=0.9, bottom=0.12, wspace=0.12, hspace=0.32)
    for c, k in enumerate(pick):
        show_map(axes[0, c], acts[k])
        axes[0, c].set_title(f"map {k}", fontsize=8, pad=2)
        show_map(axes[1, c], grads[k], sym=True)
        a = alpha[k] * 1e3
        axes[1, c].set_xlabel(f"α = {a:+.2f}", fontsize=8.5, labelpad=2,
                              color=GREEN if a > 0 else RED, fontweight="bold")
    axes[0, 0].set_ylabel("$A^k$", fontsize=9, rotation=0, labelpad=14, va="center")
    axes[1, 0].set_ylabel("gradient", fontsize=8, rotation=0, labelpad=22, va="center")
    fig.text(0.995, 0.005, "α in units of 10⁻³;  red cell = score rises if the cell fires more",
             ha="right", va="bottom", fontsize=7)
    out = FIG_DIR / "gradcam_step2_grads.pdf"
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")

    # step 3: alpha_k * A^k for the three heaviest maps + ... = weighted sum -> ReLU -> heatmap
    # axes boxes are 0.6 in square (w = 0.11 of 5.5 in, h = 0.6 of 1.3 in) so imshow fills them
    fig = plt.figure(figsize=(5.5, 1.3))
    w, h, y0 = 0.11, 0.6, 0.1
    boxes = {"m0": 0.0, "m1": 0.15, "m2": 0.30, "sum": 0.585, "cam": 0.875}
    for key, k in zip(["m0", "m1", "m2"], pick[:3]):
        ax = fig.add_axes([boxes[key], y0, w, h])
        show_map(ax, alpha[k] * acts[k])
        ax.set_title(f"α·A, map {k}", fontsize=7.5, pad=2)
    for x, s in [(0.13, "+"), (0.28, "+"), (0.495, "+ ...\n(512 terms)\n="),
                 (0.79, "ReLU\n(keep > 0)\n→")]:
        fig.text(x, y0 + h / 2, s, ha="center", va="center", fontsize=8, linespacing=1.1)
    ax = fig.add_axes([boxes["sum"], y0, w, h])
    show_map(ax, r["weighted"], sym=True)
    ax.set_title("weighted sum", fontsize=7.5, pad=2)
    ax = fig.add_axes([boxes["cam"], y0, w, h])
    show_map(ax, r["cam7"], cmap="jet")
    ax.set_title("7 x 7 heatmap", fontsize=7.5, pad=2)
    out = FIG_DIR / "gradcam_step3_sum.pdf"
    fig.savefig(out); plt.close(fig); log.info(f"saved {out}")

    # step 4: blocky 7x7 -> bilinear 224x224 -> overlay
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.85))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.12)
    axes[0].imshow(r["cam7"], cmap="jet", interpolation="nearest")
    axes[0].set_title("7 x 7: one value per cell", fontsize=8, pad=2)
    axes[1].imshow(r["cam224"], cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("bilinear: 224 x 224", fontsize=8, pad=2)
    axes[2].imshow(display); axes[2].imshow(overlay_rgba(r["cam224"]))
    axes[2].set_title("laid over the photo", fontsize=8, pad=2)
    for ax in axes:
        ax.axis("off")
    out = FIG_DIR / "gradcam_step4_upsample.pdf"
    fig.savefig(out, dpi=200); plt.close(fig); log.info(f"saved {out}")


def main():
    log = setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    model_in, display = load_inputs(log)
    r = grad_cam(model_in, log)

    # --- static figure: photo | overlay (slide size) ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(3.05, 1.65))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.06)
    axes[0].imshow(display)
    axes[0].set_title("input photo", fontsize=8, pad=2)
    axes[1].imshow(display)
    axes[1].imshow(overlay_rgba(r["cam224"]))
    axes[1].set_title(f"Grad-CAM, '{r['top_name']}'", fontsize=8, pad=2)
    for ax in axes:
        ax.axis("off")
    out = FIG_DIR / "gradcam.pdf"
    fig.savefig(out, dpi=200); plt.close(fig)
    log.info(f"saved {out}")

    # --- overview flip-book: photo -> coarse 7x7 -> overlay ----------------
    anim_frame(FIG_DIR / "gradcam_anim_0.pdf", "the photo (224 x 224)",
               lambda ax: ax.imshow(display), log)
    anim_frame(FIG_DIR / "gradcam_anim_1.pdf", "7 x 7 heatmap",
               lambda ax: ax.imshow(r["cam7"], cmap="jet", interpolation="nearest"), log)

    def draw_overlay(ax):
        ax.imshow(display)
        ax.imshow(overlay_rgba(r["cam224"]))

    anim_frame(FIG_DIR / "gradcam_anim_2.pdf", "upsampled, overlaid", draw_overlay, log)

    step_figures(r, display, log)
    log.info(f"done. top-3: {r['top3']}")


if __name__ == "__main__":
    main()
