"""Figure for 33_color_spaces (Section 2, hue is circular).

Generates into ml/09_clustering/fig/:
  hue_seam.pdf  -- left: the hue circle, showing that H = 0.01 and H = 0.99 are both red
                   but sit at opposite ends of a straight-line distance. right: k-means
                   on the Saryan pixels in naive HSV vs cone-encoded HSV, with the
                   measured red-splitting.

The overlap metric is the same one the image-compression practical (34_*) computes, so
the lecture and the notebook report the same number.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_hue_seam.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.cluster import KMeans

from color_common import ARM_RED, SEED, ensure_fig_dir, load_saryan, setup_logging

K = 8
SAT_FLOOR = 0.30      # only ask about hue where hue is meaningful
SEAM = 0.02           # how close to the 0/1 wrap counts as "on the seam"


def main():
    log = setup_logging("color_hue_seam")
    fig_dir = ensure_fig_dir()

    rgb = load_saryan(320, as_float=True)
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    pixels = rgb.reshape(-1, 3)

    # Two ways to hand HSV to a Euclidean algorithm.
    hsv_naive = np.column_stack([h, s, v])
    hsv_cone = np.column_stack([s * np.cos(2 * np.pi * h),
                                s * np.sin(2 * np.pi * h),
                                v])
    spaces = {"RGB": pixels, "HSV (naive)": hsv_naive, "HSV (cone)": hsv_cone}
    labels = {name: KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(F)
              for name, F in spaces.items()}

    # The seam test: reds just above 0 and just below 1 are the SAME colour. How often do
    # they land in the same cluster?
    lo = (h < SEAM) & (s > SAT_FLOOR)
    hi = (h > 1 - SEAM) & (s > SAT_FLOOR)
    log.info(f"seam pixels: {lo.sum():,} just above H=0, {hi.sum():,} just below H=1 "
             f"(all of them red)")
    if lo.sum() < 50 or hi.sum() < 50:
        raise ValueError(
            f"not enough seam pixels to measure ({lo.sum()} / {hi.sum()}); the seam test "
            "is meaningless on this image")

    overlap = {}
    for name, lab in labels.items():
        a = np.bincount(lab[lo], minlength=K) / lo.sum()
        b = np.bincount(lab[hi], minlength=K) / hi.sum()
        overlap[name] = float(np.minimum(a, b).sum())
        log.info(f"  {name:12s} same-cluster overlap {overlap[name]:6.1%}")

    if overlap["HSV (naive)"] > overlap["HSV (cone)"]:
        raise AssertionError(
            "the cone encoding is supposed to beat naive HSV on the seam test; got "
            f"naive {overlap['HSV (naive)']:.1%} vs cone {overlap['HSV (cone)']:.1%}")

    # --- figure ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.22)

    # left: the same two colours seen as a number line (what k-means sees) and as a
    # circle (what they are). Stacked, because the contrast between the two IS the point.
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("Hue wraps around; Euclid does not", fontsize=11)

    # -- top: H as a straight number line.
    # The two markers are filled with the colour they actually stand for, so "both are
    # red" is something the audience sees rather than something the slide claims.
    line = ax.inset_axes([0.04, 0.52, 0.92, 0.20])
    grad = hsv_to_rgb(np.stack([np.linspace(0, 1, 512),
                                np.full(512, 0.85), np.ones(512)], axis=-1))
    line.imshow(grad[None, :, :], extent=[0, 1, 0, 1], aspect="auto")
    for hv in (0.01, 0.99):
        line.plot([hv], [0.5], "o", ms=11, mec="black", mew=1.8,
                  mfc=hsv_to_rgb([hv, 0.85, 1.0]))
    line.set_xlim(0, 1); line.set_ylim(0, 1)
    line.set_xticks([]); line.set_yticks([])

    # Arrow and labels live in the PARENT axes, so nothing escapes past the title.
    ax.annotate("", xy=(0.05, 0.80), xytext=(0.95, 0.80), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color=ARM_RED, lw=2))
    ax.text(0.50, 0.83, "k-means sees these as 0.98 apart", fontsize=9, color=ARM_RED,
            ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.05, 0.48, "H = 0.01", fontsize=9, ha="center", va="top",
            transform=ax.transAxes)
    ax.text(0.95, 0.48, "H = 0.99", fontsize=9, ha="center", va="top",
            transform=ax.transAxes)

    # -- bottom: the same H as an angle, where they are neighbours
    circ = ax.inset_axes([0.22, 0.00, 0.56, 0.42])
    ang = np.linspace(0, 2 * np.pi, 512)
    r_in, r_out = 0.70, 1.0
    for i in range(len(ang) - 1):
        circ.add_patch(plt.Polygon(
            [[r_in * np.cos(ang[i]), r_in * np.sin(ang[i])],
             [r_out * np.cos(ang[i]), r_out * np.sin(ang[i])],
             [r_out * np.cos(ang[i + 1]), r_out * np.sin(ang[i + 1])],
             [r_in * np.cos(ang[i + 1]), r_in * np.sin(ang[i + 1])]],
            color=hsv_to_rgb([ang[i] / (2 * np.pi), 0.85, 1.0]), lw=0))
    for hv in (0.01, 0.99):
        x, y = 0.85 * np.cos(2 * np.pi * hv), 0.85 * np.sin(2 * np.pi * hv)
        circ.plot([x], [y], "o", ms=10, mec="black", mew=1.8,
                  mfc=hsv_to_rgb([hv, 0.85, 1.0]))
    circ.annotate("neighbours,\nboth red", xy=(1.02, 0.0), xytext=(1.30, 0.0),
                  fontsize=9, va="center", ha="left", annotation_clip=False,
                  arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    circ.set_xlim(-1.1, 1.1); circ.set_ylim(-1.1, 1.1)
    circ.set_aspect("equal"); circ.axis("off")

    # right: what that does to the clustering
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.axis("off")
    ax_r.set_title(f"k-means, k = {K}: where the seam reds land", fontsize=11)
    for i, name in enumerate(["RGB", "HSV (naive)", "HSV (cone)"]):
        sub = ax_r.inset_axes([i / 3 + 0.01, 0.16, 1 / 3 - 0.02, 0.72])
        # repaint each pixel with its cluster's mean RGB colour
        out = np.zeros_like(pixels)
        lab = labels[name]
        for c in range(K):
            m = lab == c
            out[m] = pixels[m].mean(axis=0)
        sub.imshow(out.reshape(rgb.shape))
        sub.set_title(f"{name}\nseam overlap {overlap[name]:.0%}", fontsize=8.8)
        sub.set_xticks([]); sub.set_yticks([])

    out_path = fig_dir / "hue_seam.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
