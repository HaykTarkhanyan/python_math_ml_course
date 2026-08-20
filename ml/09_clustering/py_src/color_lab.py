"""Figure for 33_color_spaces (Section 4, Lab and perceptual distance).

Generates into ml/09_clustering/fig/:
  lab_deltae.pdf  -- left: the spread of perceived difference at a FIXED RGB distance,
                     measured over random colour pairs. right: two pairs the same RGB
                     distance apart, one obvious and one nearly invisible.

The headline claim - that equal RGB distance does not mean equal perceived difference -
is measured here rather than asserted, because the effect is real but milder than the
usual telling suggests.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_lab.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color as skcolor

from color_common import ARM_BLUE, ARM_RED, SEED, ensure_fig_dir, setup_logging

N_PAIRS = 20000
STEP = 20.0          # fixed RGB distance, in 0-255 units
JND = 2.3            # just-noticeable difference, in Delta-E 1976 units


def delta_e(rgb_a, rgb_b):
    """CIE76 Delta-E between two (N,3) arrays of 0-255 RGB."""
    lab_a = skcolor.rgb2lab((rgb_a / 255.0)[:, None, :])
    lab_b = skcolor.rgb2lab((rgb_b / 255.0)[:, None, :])
    return skcolor.deltaE_cie76(lab_a, lab_b).ravel()


def main():
    log = setup_logging("color_lab")
    fig_dir = ensure_fig_dir()
    rng = np.random.default_rng(SEED)

    # Random base colours, each nudged by exactly STEP in a random direction. Clip to the
    # cube and keep only the pairs whose distance survived clipping, so STEP is honest.
    base = rng.uniform(0, 255, size=(N_PAIRS, 3))
    d = rng.normal(size=(N_PAIRS, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    other = np.clip(base + d * STEP, 0, 255)
    kept = np.abs(np.linalg.norm(other - base, axis=1) - STEP) < 1e-6
    base, other = base[kept], other[kept]
    log.info(f"{kept.sum():,} of {N_PAIRS:,} pairs kept at exactly RGB distance {STEP:.0f} "
             f"(the rest were clipped at the cube face)")

    dE = delta_e(base, other)
    p05, p50, p95 = np.percentile(dE, [5, 50, 95])
    log.info(f"Delta-E at fixed RGB distance {STEP:.0f}: "
             f"min {dE.min():.2f}, p05 {p05:.2f}, median {p50:.2f}, "
             f"p95 {p95:.2f}, max {dE.max():.2f}")
    log.info(f"p95/p05 spread = {p95/p05:.1f}x   (just-noticeable difference ~ {JND})")
    log.info(f"{(dE < JND).mean()*100:.1f}% of these pairs are below the "
             f"just-noticeable threshold, i.e. invisible")

    if p95 / p05 < 2.0:
        raise AssertionError(
            f"the whole point of the slide is that the spread is large; got {p95/p05:.2f}x")

    # Two concrete pairs to show: the least and most visible steps at this distance.
    i_quiet, i_loud = int(dE.argmin()), int(dE.argmax())
    quiet = (base[i_quiet], other[i_quiet], dE[i_quiet])
    loud = (base[i_loud], other[i_loud], dE[i_loud])
    for name, (a, b, e) in [("least visible", quiet), ("most visible", loud)]:
        log.info(f"{name}: {np.round(a).astype(int)} -> {np.round(b).astype(int)}  "
                 f"Delta-E {e:.2f}")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ax.hist(dE, bins=70, color=ARM_BLUE, alpha=0.85)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.22)          # headroom so the annotations never sit on the bars
    ax.set_xlim(0, float(dE.max()) * 1.03)   # room to the left of the JND line
    ax.axvline(JND, color=ARM_RED, lw=2, ls="--")
    # Rotated, in the empty margin left of the line: the bars are ~zero out here, and
    # this keeps the label clear of the percentile row along the top.
    ax.text(JND - 0.35, top * 0.06, f"just noticeable (Delta-E {JND})", fontsize=8.4,
            color=ARM_RED, rotation=90, ha="right", va="bottom")
    for val, lab in [(p05, "5th pct"), (p95, "95th pct")]:
        ax.axvline(val, color="#444444", lw=1.2, ls=":")
        ax.text(val + 0.3, top * 1.18, f"{lab}\n{val:.1f}", fontsize=8,
                va="top", color="#444444")
    ax.set_xlabel("perceived difference (Delta-E 1976)")
    ax.set_ylabel("pairs")
    ax.set_title(f"Every pair here is RGB distance {STEP:.0f} apart", fontsize=10.5)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Same RGB distance, different worlds", fontsize=10.5)
    for row, (name, (a, b, e)) in enumerate([("barely visible", quiet),
                                             ("obvious", loud)]):
        for col, c in enumerate([a, b]):
            sub = ax.inset_axes([0.10 + col * 0.30, 0.56 - row * 0.44, 0.28, 0.34])
            sub.imshow(np.full((10, 10, 3), c / 255.0))
            sub.set_xticks([]); sub.set_yticks([])
        ax.text(0.74, 0.73 - row * 0.44, f"{name}\nDelta-E {e:.1f}", fontsize=9.5,
                transform=ax.transAxes, va="center")

    fig.tight_layout()
    out = fig_dir / "lab_deltae.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
