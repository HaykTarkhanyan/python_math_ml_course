"""Figure for 33_color_spaces (Section 2, when HSV earns its keep).

Generates into ml/09_clustering/fig/:
  hsv_threshold.pdf  -- select "the warm/red region" of the Saryan landscape two ways,
                        by RGB distance to a reference colour and by a hue window, then
                        dim the light and re-run both. The RGB selection collapses; the
                        hue selection barely moves.

Reported as IoU between the selection before and after the lighting change, so "barely
moves" is a number and not an adjective.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_hsv_threshold.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

from color_common import ARM_BLUE, ARM_RED, ensure_fig_dir, load_saryan, setup_logging

DIM = 0.55                 # the lighting change: every pixel loses 45% of its value
RGB_RADIUS = 0.33          # RGB ball radius around the reference colour
HUE_LO, HUE_HI = 0.94, 0.09   # hue window for "warm red/orange", wrapping past 0
SAT_MIN, VAL_MIN = 0.35, 0.15


def hue_select(img):
    hsv = rgb_to_hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    in_hue = (h >= HUE_LO) | (h <= HUE_HI)      # wraps through 0
    return in_hue & (s >= SAT_MIN) & (v >= VAL_MIN)


def rgb_select(img, ref):
    return np.linalg.norm(img - ref, axis=-1) <= RGB_RADIUS


def iou(a, b):
    union = (a | b).sum()
    if union == 0:
        raise ValueError("both selections are empty; the threshold is useless here")
    return (a & b).sum() / union


def main():
    log = setup_logging("color_hsv_threshold")
    fig_dir = ensure_fig_dir()

    bright = load_saryan(320, as_float=True)
    dim = bright * DIM                       # a uniform lighting change, nothing else

    # Reference colour: the mean of the warm pixels in the bright image. This is the
    # fairest possible setup for RGB - the threshold is tuned on the image it will fail on.
    warm = hue_select(bright)
    ref = bright[warm].mean(axis=0)
    log.info(f"reference warm colour (R,G,B) = {np.round(ref*255).astype(int)}")

    sel = {
        "rgb_bright": rgb_select(bright, ref),
        "rgb_dim": rgb_select(dim, ref),
        "hsv_bright": hue_select(bright),
        "hsv_dim": hue_select(dim),
    }
    for k, m in sel.items():
        log.info(f"  {k:12s} selects {m.mean()*100:5.1f}% of pixels")

    iou_rgb = iou(sel["rgb_bright"], sel["rgb_dim"])
    iou_hsv = iou(sel["hsv_bright"], sel["hsv_dim"])
    kept_rgb = sel["rgb_dim"].sum() / max(sel["rgb_bright"].sum(), 1)
    kept_hsv = sel["hsv_dim"].sum() / max(sel["hsv_bright"].sum(), 1)
    log.info(f"after dimming to {DIM:.0%}: RGB threshold IoU {iou_rgb:.1%} "
             f"(keeps {kept_rgb:.0%} of its pixels)")
    log.info(f"after dimming to {DIM:.0%}: hue threshold IoU {iou_hsv:.1%} "
             f"(keeps {kept_hsv:.0%} of its pixels)")

    if iou_hsv <= iou_rgb:
        raise AssertionError(
            f"hue thresholding should survive a lighting change better than RGB; got "
            f"hue IoU {iou_hsv:.1%} vs RGB IoU {iou_rgb:.1%}")

    # --- figure ---------------------------------------------------------------------------
    def overlay(img, mask):
        """Grey out everything the threshold rejected."""
        grey = img.mean(axis=-1, keepdims=True).repeat(3, axis=-1) * 0.35
        return np.where(mask[..., None], img, grey)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    rows = [
        ("full light", bright, sel["rgb_bright"], sel["hsv_bright"]),
        (f"dimmed to {DIM:.0%}", dim, sel["rgb_dim"], sel["hsv_dim"]),
    ]
    for r, (label, img, m_rgb, m_hsv) in enumerate(rows):
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(label, fontsize=10)
        axes[r, 0].set_title("image" if r == 0 else "", fontsize=10.5)
        axes[r, 1].imshow(overlay(img, m_rgb))
        axes[r, 1].set_title("RGB threshold" if r == 0 else "", fontsize=10.5)
        axes[r, 2].imshow(overlay(img, m_hsv))
        axes[r, 2].set_title("hue threshold" if r == 0 else "", fontsize=10.5)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    axes[1, 1].set_xlabel(f"IoU {iou_rgb:.0%}", fontsize=11, color=ARM_RED,
                          fontweight="bold")
    axes[1, 2].set_xlabel(f"IoU {iou_hsv:.0%}", fontsize=11, color=ARM_BLUE,
                          fontweight="bold")

    fig.tight_layout()
    out = fig_dir / "hsv_threshold.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
