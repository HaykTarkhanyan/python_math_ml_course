"""Figure for 33_color_spaces (Section 4, colour histograms as features).

Generates into ml/09_clustering/fig/:
  color_histogram.pdf  -- what a colour histogram is, and the thing it cannot see: the
                          Saryan landscape and the same pixels shuffled have byte-for-byte
                          identical histograms.

This is the failure the photo-grouping practical (34_image_clusters) opens with: cluster
colour histograms and you group by palette, never by subject.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_histogram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_common import ARM_BLUE, ARM_GREEN, ARM_RED, SEED, ensure_fig_dir, load_saryan, setup_logging

BINS = 32


def hist3(arr, bins=BINS):
    """Per-channel histogram, normalised. The simplest colour-histogram feature there is."""
    return np.stack([np.histogram(arr[..., c], bins=bins, range=(0, 256))[0]
                     for c in range(3)]).astype(np.float64) / arr[..., 0].size


def main():
    log = setup_logging("color_histogram")
    fig_dir = ensure_fig_dir()
    rng = np.random.default_rng(SEED)

    arr = load_saryan(320)
    flat = arr.reshape(-1, 3)
    shuffled = flat[rng.permutation(len(flat))].reshape(arr.shape)

    h_orig, h_shuf = hist3(arr), hist3(shuffled)
    gap = float(np.abs(h_orig - h_shuf).max())
    log.info(f"max histogram difference, original vs shuffled = {gap:.2e}")
    assert gap < 1e-12, (
        f"shuffling pixels must not change the histogram, got {gap:.2e}")

    # The feature vector these two images hand to k-means is literally the same vector.
    log.info(f"feature vectors identical: {np.array_equal(h_orig, h_shuf)}")
    log.info(f"histogram feature length = {h_orig.size} numbers, "
             f"vs {flat.size:,} numbers in the image itself")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})

    centres = np.arange(BINS) * (256 / BINS) + (256 / BINS) / 2
    for row, (img, name) in enumerate([(arr, "the painting"),
                                       (shuffled, "its pixels, shuffled")]):
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(name, fontsize=10.5)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        ax = axes[row, 1]
        h = hist3(img)
        for c, color, lab in [(0, ARM_RED, "R"), (1, ARM_GREEN, "G"), (2, ARM_BLUE, "B")]:
            ax.plot(centres, h[c], color=color, lw=1.8, label=lab)
            ax.fill_between(centres, h[c], color=color, alpha=0.10)
        ax.set_xlim(0, 256)
        ax.set_ylim(0, max(h_orig.max(), h_shuf.max()) * 1.12)
        ax.set_ylabel("fraction of pixels", fontsize=9)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.set_title(f"colour histogram ({BINS} bins per channel)", fontsize=10.5)
            ax.legend(fontsize=8.5, loc="upper right", ncol=3)
        else:
            ax.set_xlabel("channel value")

    fig.suptitle("Identical feature vectors, and one of them is not a painting",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    out = fig_dir / "color_histogram.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
