"""Classical computer vision tour for the L16 CNN Foundations deck ("Classical vision we skip").

One strip figure per method, each built from this lecture's kernels plus one idea:

Generates into ml/12_cnn/fig/:
  cv_canny.pdf     -- Canny: input, blur + Sobel |gradient|, non-max suppression, hysteresis
  cv_template.pdf  -- template matching: template, normalized cross-correlation map, matches
  cv_hough.pdf     -- Hough lines on a brick wall: Canny edges, (theta, rho) votes, lines
  cv_harris.pdf    -- Harris corners: input, response R (corner > 0, edge < 0), corners found
  cv_sift.pdf      -- SIFT: keypoints with scale and orientation, matches across two views
  cv_hog.pdf       -- HOG: input, one cell's orientation histogram, the HOG descriptor image

Images: skimage's public-domain test images (camera, coins, brick, astronaut).
OpenCV 4.11 (SIFT is in the main package since 4.4, after its patent expired in March 2020).

Run (~2 min on CPU, measured 2026-10-02: 101-152 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/classical_cv.py

Every figure is drawn at its slide size: a 5.4 x 1.65 in strip across the text width,
fonts 7-8 pt.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from skimage import color, data, exposure, feature, measure, transform

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
DIVERGING = LinearSegmentedColormap.from_list("arm_div", [ARM_BLUE, "white", ARM_RED])

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

STRIP = (5.4, 1.65)            # its size on the slide
TITLE = dict(fontsize=7.5, pad=3)


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("classical_cv")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "classical_cv.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def strip(n, width_ratios=None):
    fig, axes = plt.subplots(1, n, figsize=STRIP, gridspec_kw=dict(width_ratios=width_ratios))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.02, wspace=0.08)
    return fig, axes


def save(fig, name, log):
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)   # no dead space above/below
    plt.close(fig)
    log.info(f"saved {out}")


def non_max_suppression(mag, ang):
    """Keep a pixel only if it is the largest along its gradient direction (4 directions)."""
    q = (np.round(ang / (np.pi / 4)) % 4).astype(int)     # 0: -, 1: /, 2: |, 3: \ (gradient dir)
    offsets = {0: (0, 1), 1: (-1, 1), 2: (1, 0), 3: (1, 1)}
    out = np.zeros_like(mag)
    for d, (dy, dx) in offsets.items():
        fwd = np.roll(mag, (-dy, -dx), axis=(0, 1))
        bwd = np.roll(mag, (dy, dx), axis=(0, 1))
        keep = (q == d) & (mag >= fwd) & (mag >= bwd)
        out[keep] = mag[keep]
    return out


def canny_fig(log):
    img = data.camera()[40:296, 120:376]
    blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    gx = cv2.Sobel(blur.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
    thin = non_max_suppression(mag, ang)
    edges = cv2.Canny(blur, 50, 150)
    log.info(f"canny: {np.count_nonzero(thin > 0)} thin px, {np.count_nonzero(edges)} final edge px")
    fig, axes = strip(4)
    vmax = np.percentile(mag, 99)
    panels = [(img, "input", dict(vmin=0, vmax=255)),
              (mag, "1-2. |gradient|", dict(vmin=0, vmax=vmax)),
              (thin, "3. thinned", dict(vmin=0, vmax=vmax * 0.5)),
              (edges, "4. hysteresis", dict(vmin=0, vmax=255))]
    for ax, (a, t, kw) in zip(axes, panels):
        ax.imshow(a, cmap="gray", **kw)
        ax.set_title(t, **TITLE)
        ax.axis("off")
    save(fig, "cv_canny.pdf", log)


def template_fig(log):
    img = data.coins()
    # find the coins once, pick one in the top row as the template
    mask = img > 110
    props = [p for p in measure.regionprops(measure.label(mask)) if p.area > 800]
    top = sorted(props, key=lambda p: p.centroid[0])[:7]
    pick = sorted(top, key=lambda p: p.centroid[1])[2]
    r0, c0, r1, c1 = pick.bbox
    tpl = img[r0 - 2:r1 + 2, c0 - 2:c1 + 2]
    score = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    peaks = feature.peak_local_max(score, min_distance=12, threshold_abs=0.45)
    log.info(f"template {tpl.shape}, {len(peaks)} matches above 0.45, "
             f"scores {np.round(np.sort(score[tuple(peaks.T)])[::-1], 2).tolist()}")
    fig, axes = strip(3, width_ratios=[0.55, 1.27, 1.27])
    axes[0].imshow(tpl, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("template", **TITLE)
    im = axes[1].imshow(score, cmap="magma", vmin=-0.2, vmax=1)
    axes[1].set_title("score at every position", **TITLE)
    axes[2].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("matches with score > 0.45", **TITLE)
    h, w = tpl.shape
    for (y, x) in peaks:
        best = score[y, x] > 0.99
        axes[2].add_patch(Rectangle((x, y), w, h, fill=False, lw=1.3 if best else 0.9,
                                    ec=ARM_RED if best else ARM_ORANGE))
    for ax in axes:
        ax.axis("off")
    save(fig, "cv_template.pdf", log)


def hough_fig(log):
    gray = data.brick()[128:384, 128:384].astype(float) / 255
    edges = feature.canny(gray, sigma=2.0, mode="nearest")   # zero padding = a fake border edge
    angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, rho = transform.hough_line(edges, theta=angles)
    _, pk_t, pk_r = transform.hough_line_peaks(h, theta, rho, num_peaks=12, min_distance=9,
                                               min_angle=4, threshold=0.5 * h.max())
    log.info(f"hough: {edges.sum()} edge px, {len(pk_t)} peaks, angles "
             f"{np.round(np.rad2deg(pk_t)).astype(int).tolist()}")
    fig, axes = strip(3, width_ratios=[1, 1.45, 1])
    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title("1. edges (Canny)", **TITLE)
    axes[1].imshow(np.log1p(h), cmap="magma", aspect="auto",
                   extent=[-90, 90, rho[-1], rho[0]])
    axes[1].plot(np.rad2deg(pk_t), pk_r, "o", mfc="none", mec="#00C8FF", ms=5, mew=0.9)
    axes[1].set_title(r"2. votes in ($\theta$, $\rho$)", **TITLE)
    axes[1].set_xlabel(r"$\theta$ (deg)", fontsize=7, labelpad=1)
    axes[1].set_ylabel(r"$\rho$ (px)", fontsize=7, labelpad=1)
    axes[1].tick_params(labelsize=6.5, length=2, pad=1)
    axes[2].imshow(gray, cmap="gray")
    hgt, wid = gray.shape
    for t, r in zip(pk_t, pk_r):
        x = np.array([0, wid])
        if abs(np.sin(t)) > 1e-6:
            y = (r - x * np.cos(t)) / np.sin(t)
        else:                                   # vertical line x = r
            x, y = np.array([r, r]), np.array([0, hgt])
        axes[2].plot(x, y, color=ARM_RED, lw=0.9)
    axes[2].set_xlim(0, wid); axes[2].set_ylim(hgt, 0)
    axes[2].set_title("3. peaks = lines", **TITLE)
    axes[0].axis("off"); axes[2].axis("off")
    fig.subplots_adjust(bottom=0.2, wspace=0.22)
    save(fig, "cv_hough.pdf", log)


def harris_fig(log):
    full = data.camera().astype(float) / 255
    # computed on the whole photo, then cropped: no fake corners at the crop's border
    r_full = feature.corner_harris(full, method="k", k=0.05, sigma=1.5)
    peaks = feature.corner_peaks(r_full, min_distance=6, threshold_rel=0.02)
    y0, x0, n = 40, 120, 256
    img, r = full[y0:y0 + n, x0:x0 + n], r_full[y0:y0 + n, x0:x0 + n]
    inside = (peaks[:, 0] >= y0) & (peaks[:, 0] < y0 + n) & (peaks[:, 1] >= x0) & (peaks[:, 1] < x0 + n)
    corners = peaks[inside] - [y0, x0]
    log.info(f"harris: R in [{r.min():.4f}, {r.max():.4f}], {len(corners)} corners")
    fig, axes = strip(3)
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input", **TITLE)
    shown = np.sign(r) * np.sqrt(np.abs(r))            # signed sqrt: weak responses visible
    m = np.percentile(np.abs(shown), 99.5)
    axes[1].imshow(shown, cmap=DIVERGING, vmin=-m, vmax=m)
    axes[1].set_title("R: red corner, blue edge", **TITLE)
    axes[2].imshow(img, cmap="gray")
    axes[2].plot(corners[:, 1], corners[:, 0], "+", color=ARM_RED, ms=4, mew=0.9)
    axes[2].set_title(f"{len(corners)} corners", **TITLE)
    for ax in axes:
        ax.axis("off")
    save(fig, "cv_harris.pdf", log)


def sift_fig(log):
    img1 = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY), (256, 256))
    rot = cv2.getRotationMatrix2D((128, 128), 30, 0.7)          # rotate 30 deg, scale 0.7
    img2 = cv2.warpAffine(img1, rot, (256, 256), borderValue=255)
    img2 = np.clip(img2.astype(float) * 0.75 + 50, 0, 255).astype(np.uint8)  # other lighting
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = sorted([m for m, n in pairs if m.distance < 0.7 * n.distance],
                  key=lambda m: m.distance)
    # a match is correct if keypoint 1 lands near keypoint 2 under the known warp
    p1 = np.array([k1[m.queryIdx].pt for m in good])
    p2 = np.array([k2[m.trainIdx].pt for m in good])
    p1w = np.c_[p1, np.ones(len(p1))] @ rot.T
    correct = np.linalg.norm(p1w - p2, axis=1) < 4
    log.info(f"sift: {len(k1)} / {len(k2)} keypoints, {len(good)} matches after ratio test, "
             f"{correct.sum()} land within 4 px of the true position")
    fig, axes = strip(2, width_ratios=[1, 2])
    strongest = sorted(k1, key=lambda k: -k.response)[:40]
    axes[0].imshow(img1, cmap="gray")
    for k in strongest:
        x, y, s, a = *k.pt, k.size / 2, np.deg2rad(k.angle)
        axes[0].add_patch(plt.Circle((x, y), s, fill=False, ec=ARM_ORANGE, lw=0.7))
        axes[0].plot([x, x + s * np.cos(a)], [y, y + s * np.sin(a)], color=ARM_ORANGE, lw=0.7)
    axes[0].set_title("keypoints: where, size, angle", **TITLE)
    canvas = np.hstack([img1, np.full((256, 12), 255, np.uint8), img2])
    axes[1].imshow(canvas, cmap="gray", vmin=0, vmax=255)
    for (a, b), ok in list(zip(zip(p1, p2), correct))[:40]:
        axes[1].plot([a[0], b[0] + 268], [a[1], b[1]], lw=0.6,
                     color=ARM_BLUE if ok else ARM_RED)
    axes[1].set_title(f"rotated, shrunk, relit copy: {correct.sum()} of {len(good)} "
                      f"matches correct", **TITLE)
    for ax in axes:
        ax.axis("off")
    save(fig, "cv_sift.pdf", log)
    return int(correct[:40].sum()), min(40, len(good))


def hog_fig(log):
    img = color.rgb2gray(data.astronaut())[0:384:2, 64:448:2]       # 192 x 192, upper body
    fd, hog_img = feature.hog(img, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True)
    log.info(f"hog: {img.shape} image -> {fd.size}-dim descriptor")
    # one cell's histogram, computed by hand: gradient magnitude voted into 9 angle bins
    cy, cx = 4, 7                                             # cell row, col (16 px cells)
    patch = img[cy * 16:(cy + 1) * 16 + 2, cx * 16:(cx + 1) * 16 + 2]
    gy, gx = np.gradient(patch)
    mag = np.hypot(gx, gy)[1:-1, 1:-1]
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180)[1:-1, 1:-1]
    hist, _ = np.histogram(ang, bins=9, range=(0, 180), weights=mag)
    hist = 100 * hist / hist.sum()
    fig, axes = strip(3, width_ratios=[1, 1.35, 1])
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(Rectangle((cx * 16, cy * 16), 16, 16, fill=False, ec=ARM_RED, lw=1.2))
    axes[0].set_title("input, one 16x16 cell", **TITLE)
    bars = axes[1].bar(np.arange(9) * 20 + 10, hist, width=16, color=ARM_RED)
    axes[1].bar_label(bars, labels=[f"{v:.0f}" for v in hist], fontsize=6, padding=1)
    axes[1].set_title("its gradient angles, % of strength", **TITLE)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].tick_params(labelsize=6.5, length=2, pad=1)
    axes[1].set_xlabel("angle (deg)", fontsize=7, labelpad=1)
    axes[1].set_yticks([])
    axes[1].set_ylim(0, hist.max() * 1.25)
    for s in ("top", "right", "left"):
        axes[1].spines[s].set_visible(False)
    axes[2].imshow(exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max() * 0.35)),
                   cmap="gray")
    axes[2].set_title("every cell's histogram", **TITLE)
    axes[0].axis("off"); axes[2].axis("off")
    fig.subplots_adjust(bottom=0.2, wspace=0.2)
    save(fig, "cv_hog.pdf", log)


def main():
    log = setup_logging()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    canny_fig(log)
    template_fig(log)
    hough_fig(log)
    harris_fig(log)
    sift_fig(log)
    hog_fig(log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
