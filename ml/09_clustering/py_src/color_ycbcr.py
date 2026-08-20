"""Figure for 33_color_spaces (Section 4, YCbCr and chroma subsampling).

Generates into ml/09_clustering/fig/:
  chroma_subsample.pdf  -- top: the Saryan landscape split into Y (brightness) and
                           Cb/Cr (colour). bottom: throw away three quarters of the
                           colour (4:2:0) vs throw away the same amount of brightness.
                           One is hard to see; the other wrecks the image.

That asymmetry is the whole reason JPEG and every video codec store colour at lower
resolution than brightness. Both errors are measured so the slide can quote them.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_ycbcr.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from color_common import ensure_fig_dir, load_saryan, setup_logging

# ITU-R BT.601 full-range YCbCr, the flavour JPEG uses.
KR, KG, KB = 0.299, 0.587, 0.114


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + KG * g + KB * b
    cb = 128.0 + (b - y) / (2 * (1 - KB))
    cr = 128.0 + (r - y) / (2 * (1 - KR))
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128.0, ycc[..., 2] - 128.0
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return np.stack([r, g, b], axis=-1)


CROP = 96          # side of the zoom window on the bottom row


def busiest_patch(ycc, size):
    """Top-left corner of the size x size window with the most chroma detail.

    Chroma subsampling only shows where the colour planes actually change fast, so the
    zoom picks that window from the data instead of hardcoding a guess.
    """
    detail = (np.abs(np.diff(ycc[..., 1], axis=1, prepend=0))
              + np.abs(np.diff(ycc[..., 2], axis=1, prepend=0))
              + np.abs(np.diff(ycc[..., 1], axis=0, prepend=0))
              + np.abs(np.diff(ycc[..., 2], axis=0, prepend=0)))
    # Summed-area table so every candidate window is one lookup.
    integral = detail.cumsum(axis=0).cumsum(axis=1)
    h, w = detail.shape
    best, best_rc = -np.inf, (0, 0)
    for r in range(0, h - size, 8):
        for c in range(0, w - size, 8):
            total = (integral[r + size, c + size] - integral[r, c + size]
                     - integral[r + size, c] + integral[r, c])
            if total > best:
                best, best_rc = total, (r, c)
    return best_rc


def halve_and_restore(plane):
    """Downsample a plane 2x in each direction and blow it back up - i.e. 4:2:0."""
    h, w = plane.shape
    # No mode= argument: Pillow infers "F" from float32, and passing it explicitly is
    # deprecated as of Pillow 11.
    small = Image.fromarray(plane.astype(np.float32)).resize(
        (w // 2, h // 2), Image.BOX)
    return np.asarray(small.resize((w, h), Image.NEAREST), dtype=np.float64)


def main():
    log = setup_logging("color_ycbcr")
    fig_dir = ensure_fig_dir()

    rgb = load_saryan(320).astype(np.float64)
    ycc = rgb_to_ycbcr(rgb)

    # Round-trip must be lossless before we trust anything measured below.
    rt = float(np.abs(ycbcr_to_rgb(ycc) - rgb).max())
    log.info(f"YCbCr round-trip max error = {rt:.2e}")
    assert rt < 1e-9, f"YCbCr conversion is not invertible, max error {rt:.2e}"

    # --- the two experiments ----------------------------------------------------------------
    sub_chroma = ycc.copy()
    sub_chroma[..., 1] = halve_and_restore(ycc[..., 1])
    sub_chroma[..., 2] = halve_and_restore(ycc[..., 2])
    img_chroma = np.clip(ycbcr_to_rgb(sub_chroma), 0, 255)

    sub_luma = ycc.copy()
    sub_luma[..., 0] = halve_and_restore(ycc[..., 0])
    img_luma = np.clip(ycbcr_to_rgb(sub_luma), 0, 255)

    # Both throw away exactly the same number of stored values: one full-resolution plane's
    # worth. Say so with numbers.
    err_chroma = float(np.sqrt(((img_chroma - rgb) ** 2).mean()))
    err_luma = float(np.sqrt(((img_luma - rgb) ** 2).mean()))
    log.info(f"4:2:0 on the two colour planes -> RMSE {err_chroma:.2f} levels")
    log.info(f"the same halving on the brightness plane -> RMSE {err_luma:.2f} levels")
    log.info(f"discarding colour detail is {err_luma/err_chroma:.1f}x cheaper "
             f"than discarding brightness detail")

    if err_luma <= err_chroma:
        raise AssertionError(
            "subsampling luma should hurt more than subsampling chroma; got "
            f"luma RMSE {err_luma:.2f} vs chroma RMSE {err_chroma:.2f}")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))

    top = [(rgb.astype(np.uint8), "RGB", None),
           (ycc[..., 0], "Y  brightness", "gray"),
           (ycc[..., 1], "Cb  blue-yellow", "coolwarm")]
    for ax, (im, t, cm) in zip(axes[0], top):
        ax.imshow(im if cm is None else im, cmap=cm)
        ax.set_title(t, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom row is a ZOOM. At full-page scale the two damaged images look identical;
    # the difference only shows where colour detail is fine, so crop to the busiest patch
    # rather than inviting the room to squint at a whole painting.
    r0, c0 = busiest_patch(ycc, CROP)
    log.info(f"zoom crop at row {r0}, col {c0}, size {CROP} (highest chroma detail)")
    crop = lambda im: im[r0:r0 + CROP, c0:c0 + CROP]

    bottom = [
        (crop(rgb).astype(np.uint8), "original (zoomed)", ""),
        (crop(img_chroma).astype(np.uint8),
         "colour at half resolution\n(4:2:0, what JPEG does)", f"RMSE {err_chroma:.1f}"),
        (crop(img_luma).astype(np.uint8),
         "brightness at half resolution\n(nobody does this)", f"RMSE {err_luma:.1f}"),
    ]
    for ax, (im, t, note) in zip(axes[1], bottom):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(t, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        if note:
            ax.set_xlabel(note, fontsize=10, fontweight="bold")
    # Mark the crop on the full image above it so the zoom is locatable.
    axes[0, 0].add_patch(plt.Rectangle((c0, r0), CROP, CROP, edgecolor="white",
                                       facecolor="none", lw=1.8))

    fig.tight_layout()
    out = fig_dir / "chroma_subsample.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
