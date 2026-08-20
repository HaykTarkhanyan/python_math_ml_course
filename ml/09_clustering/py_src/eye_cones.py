"""Figure for 33_color_spaces (Section 1, how the eye sees colour).

Generates into ml/09_clustering/fig/:
  eye_cones.pdf  -- approximate S/M/L cone sensitivity curves vs wavelength, plus the
                    metamerism panel: two different spectra that excite the three cone
                    types identically, so they are the same colour to a human.

Copied from ml/ch6_cnn/py_src/eye_cones.py (the CNN deck keeps its own copy) and extended
with the metamerism panel, which is the reason three numbers are enough.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/eye_cones.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_common import (ARM_BLUE, ARM_GREEN, ARM_ORANGE, ARM_RED, ensure_fig_dir,
                          setup_logging)

# Cone peaks, nm. Approximate Stockman & Sharpe values, rounded - the slide is about the
# shape (three overlapping humps), not about photometric accuracy.
PEAKS = {"S": (445.0, 22.0), "M": (540.0, 34.0), "L": (568.0, 38.0)}


def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def cone_responses(wl, spectrum):
    """Integrate a spectrum against each cone curve -> the three numbers the brain gets."""
    # np.trapz, not np.trapezoid - the ma venv is on numpy 1.26.4, where trapezoid does
    # not exist yet.
    return np.array([np.trapz(spectrum * gaussian(wl, mu, sig), wl)
                     for mu, sig in PEAKS.values()])


def main():
    log = setup_logging("color_eye_cones")
    fig_dir = ensure_fig_dir()

    wl = np.linspace(400, 700, 600)
    S = gaussian(wl, *PEAKS["S"])
    M = gaussian(wl, *PEAKS["M"])
    L = gaussian(wl, *PEAKS["L"])

    # --- metamer pair -------------------------------------------------------------------
    # A: a broad, smooth spectrum - the kind of light a real surface reflects.
    # B: three narrow spikes at a display's primaries. No broad light in it at all.
    # Solve for the three spike weights that reproduce A's three cone responses exactly.
    # Three unknowns, three equations - which is precisely why a monitor needs three
    # primaries and no more. If the responses match, the eye cannot tell them apart.
    spec_a = gaussian(wl, 575, 55) + 0.55 * gaussian(wl, 480, 45)
    primaries = {"B": 465.0, "G": 532.0, "R": 630.0}
    spikes = [gaussian(wl, mu, 9.0) for mu in primaries.values()]

    target = cone_responses(wl, spec_a)
    basis = np.array([cone_responses(wl, s) for s in spikes]).T   # (3 cones, 3 primaries)
    weights = np.linalg.solve(basis, target)
    if np.any(weights <= 0):
        raise ValueError(
            f"metamer needs positive primary weights (a display cannot emit negative "
            f"light), got {dict(zip(primaries, np.round(weights, 3)))}. Pick a target "
            f"spectrum inside the primaries' gamut.")
    spec_b = sum(w * s for w, s in zip(weights, spikes))
    resp_b = cone_responses(wl, spec_b)

    # Judge the match against the largest cone response: a mismatch of 0.03 in a channel
    # whose scale is 50 is not a colour difference, it is arithmetic noise.
    err = np.abs(resp_b - target) / target.max()
    log.info(f"primary weights: {dict(zip(primaries, np.round(weights, 3)))}")
    log.info(f"cone response A (S,M,L) = {np.round(target, 4)}")
    log.info(f"cone response B (S,M,L) = {np.round(resp_b, 4)}")
    log.info(f"mismatch, relative to the largest response = {np.round(err * 100, 4)} %")
    if err.max() > 1e-6:
        raise AssertionError(
            f"metamer should match to solver precision, got max {err.max():.2e}")

    # --- figure -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))

    ax = axes[0]
    for curve, color, lab in [(S, ARM_BLUE, "S cones (blue)"),
                              (M, ARM_GREEN, "M cones (green)"),
                              (L, ARM_RED, "L cones (red)")]:
        ax.plot(wl, curve, color=color, lw=2.2, label=lab)
        ax.fill_between(wl, curve, alpha=0.08, color=color)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("relative sensitivity")
    ax.set_title("Three cone types -> three numbers per point")
    ax.set_xlim(400, 700)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(wl, spec_a, color="#444444", lw=2.2, label="A: a real surface's light")
    ax.fill_between(wl, spec_a, alpha=0.10, color="#444444")
    ax.plot(wl, spec_b, color=ARM_ORANGE, lw=2.0, ls="--",
            label="B: a screen's three primaries")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("power")
    ax.set_title("Different light, identical cone responses\n"
                 "(this is why a screen only needs three primaries)")
    ax.set_xlim(400, 700)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = fig_dir / "eye_cones.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
