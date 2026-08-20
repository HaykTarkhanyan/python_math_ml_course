"""Figure for 33_color_spaces (Section 1, a pixel is three numbers).

Generates into ml/09_clustering/fig/:
  rgb_channels.pdf  -- the Saryan landscape + its R, G, B channels, each tinted in its own
                       colour, plus a zoomed patch with the actual (R, G, B) triples
                       printed on the pixels.

Copied from ml/ch6_cnn/py_src/rgb_channels.py and re-pointed at the Saryan painting, so
the deck shows the same pixels the image-compression practical clusters.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/rgb_channels.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_common import ensure_fig_dir, load_saryan, setup_logging

ZOOM_ORIGIN = (118, 150)   # row, col of the zoom patch on the 320x320 image
ZOOM_SIZE = 4              # patch is ZOOM_SIZE x ZOOM_SIZE pixels


def main():
    log = setup_logging("color_rgb_channels")
    fig_dir = ensure_fig_dir()
    arr = load_saryan(320)
    log.info(f"image {arr.shape}, dtype {arr.dtype}, "
             f"distinct colours {len(np.unique(arr.reshape(-1, 3), axis=0)):,}")

    R = arr.copy(); R[:, :, [1, 2]] = 0
    G = arr.copy(); G[:, :, [0, 2]] = 0
    B = arr.copy(); B[:, :, [0, 1]] = 0

    r0, c0 = ZOOM_ORIGIN
    patch = arr[r0:r0 + ZOOM_SIZE, c0:c0 + ZOOM_SIZE]
    log.info(f"zoom patch at {ZOOM_ORIGIN}, top-left pixel = {tuple(patch[0, 0])}")

    fig = plt.figure(figsize=(12.6, 3.1))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.12)

    for i, (im, t) in enumerate(zip([arr, R, G, B],
                                    ["RGB image", "R channel", "G channel", "B channel"])):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
        if i == 0:
            ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), ZOOM_SIZE, ZOOM_SIZE,
                                       edgecolor="white", facecolor="none", lw=1.6))

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(patch, interpolation="nearest")
    for i in range(ZOOM_SIZE):
        for j in range(ZOOM_SIZE):
            r, g, b = patch[i, j]
            # White text on dark pixels, black on light ones, so every label stays legible.
            tone = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 128 else "black"
            ax.text(j, i, f"{r}\n{g}\n{b}", ha="center", va="center",
                    fontsize=6.4, color=tone, linespacing=1.05)
    ax.set_title(f"{ZOOM_SIZE}x{ZOOM_SIZE} pixels, up close", fontsize=11)
    ax.axis("off")

    out = fig_dir / "rgb_channels.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
