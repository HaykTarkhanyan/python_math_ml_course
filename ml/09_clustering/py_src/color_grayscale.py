"""Figure for 33_color_spaces (Section 3, how RGB becomes grayscale).

Generates into ml/09_clustering/fig/:
  gray_weights.pdf  -- left: what the three recipes do to pure red, green and blue.
                       middle: the Saryan landscape under each recipe.
                       right: where Rec.601 and Rec.709 actually disagree.

Also measures, and logs, which standard each library in the ma venv implements. That
measurement is the point of the slide: PIL and skimage disagree by construction.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_grayscale.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color as skcolor

from color_common import (ARM_BLUE, ARM_GREEN, ARM_RED, ensure_fig_dir, linear_to_srgb,
                          load_saryan, setup_logging, srgb_to_linear)

REC601 = np.array([0.299, 0.587, 0.114])
REC709 = np.array([0.2126, 0.7152, 0.0722])
NAIVE = np.array([1 / 3, 1 / 3, 1 / 3])


def main():
    log = setup_logging("color_grayscale")
    fig_dir = ensure_fig_dir()

    # --- what each recipe does to the primaries -------------------------------------------
    primaries = {"pure red": (255, 0, 0), "pure green": (0, 255, 0), "pure blue": (0, 0, 255)}
    recipes = {"mean": NAIVE, "Rec.601": REC601, "Rec.709": REC709}
    table = {}
    for rname, w in recipes.items():
        table[rname] = {pname: float(np.array(p) @ w) for pname, p in primaries.items()}
        log.info(f"{rname:8s} " + "  ".join(
            f"{pname} -> {table[rname][pname]:6.1f}" for pname in primaries))

    # --- which standard does each library use? --------------------------------------------
    arr = load_saryan(320)
    pil_gray = np.asarray(Image.fromarray(arr).convert("L")).astype(np.float64)
    ski_gray = skcolor.rgb2gray(arr) * 255.0
    by601 = arr @ REC601
    by709 = arr @ REC709

    gap_pil_601 = float(np.abs(pil_gray - by601).max())
    gap_ski_709 = float(np.abs(ski_gray - by709).max())
    log.info(f"PIL .convert('L')  mean {pil_gray.mean():7.3f}  "
             f"max gap vs Rec.601 = {gap_pil_601:.4f}")
    log.info(f"skimage rgb2gray   mean {ski_gray.mean():7.3f}  "
             f"max gap vs Rec.709 = {gap_ski_709:.4f}")
    if gap_pil_601 > 1.0:
        raise AssertionError(f"PIL no longer matches Rec.601 (gap {gap_pil_601:.3f}); "
                             "the slide's claim needs rechecking")
    if gap_ski_709 > 0.1:
        raise AssertionError(f"skimage no longer matches Rec.709 (gap {gap_ski_709:.3f}); "
                             "the slide's claim needs rechecking")
    log.info(f"the two libraries differ by {np.abs(by601-by709).mean():.2f} on average, "
             f"up to {np.abs(by601-by709).max():.1f} levels on this image")

    # --- luma vs luminance: the step everyone skips ----------------------------------------
    # Correct luminance: linearize, weight, re-encode. Luma: weight the stored numbers.
    lin = srgb_to_linear(arr / 255.0)
    true_lum = linear_to_srgb(lin @ REC709) * 255.0
    luma = by709
    log.info(f"luma vs correct luminance: mean gap {np.abs(luma-true_lum).mean():.1f}, "
             f"max {np.abs(luma-true_lum).max():.1f} levels")

    # --- figure ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(13.0, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.55, 1.0], wspace=0.26)

    # left: bar chart of the primaries under each recipe
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    width = 0.26
    for i, (rname, vals) in enumerate(table.items()):
        heights = [vals[p] for p in primaries]
        bars = ax.bar(x + (i - 1) * width, heights, width, label=rname,
                      color=["#999999", ARM_BLUE, ARM_RED][i])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["red", "green", "blue"], fontsize=9)
    ax.set_ylabel("grey level out")
    ax.set_ylim(0, 215)
    ax.set_title("Green carries the brightness", fontsize=10.5)
    ax.legend(fontsize=7.6, loc="upper left")
    ax.grid(alpha=0.2, axis="y")

    # middle: the image under each recipe
    ax_m = fig.add_subplot(gs[0, 1])
    ax_m.axis("off")
    ax_m.set_title("Same painting, three recipes", fontsize=10.5)
    panels = [(arr, "colour", None), (arr @ NAIVE, "mean", "gray"),
              (by601, "Rec.601\nPIL", "gray"), (by709, "Rec.709\nskimage", "gray")]
    for i, (im, t, cm) in enumerate(panels):
        sub = ax_m.inset_axes([i / 4 + 0.014, 0.04, 1 / 4 - 0.028, 0.80])
        sub.imshow(im if cm is None else im, cmap=cm, vmin=None if cm is None else 0,
                   vmax=None if cm is None else 255)
        sub.set_title(t, fontsize=8.0, linespacing=1.15)
        sub.set_xticks([]); sub.set_yticks([])

    # right: where 601 and 709 disagree
    ax_r = fig.add_subplot(gs[0, 2])
    diff = by601 - by709
    lim = float(np.abs(diff).max())
    im = ax_r.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    ax_r.set_title(f"Rec.601 minus Rec.709\n(up to {lim:.0f} levels)", fontsize=10.5)
    fig.colorbar(im, ax=ax_r, fraction=0.046, pad=0.03)

    out = fig_dir / "gray_weights.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
