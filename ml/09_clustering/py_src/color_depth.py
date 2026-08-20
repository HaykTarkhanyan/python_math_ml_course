"""Figure for 33_color_spaces (Section 2, where depth and shading live).

Generates into ml/09_clustering/fig/:
  depth_shading.pdf  -- top: three Lambertian spheres under one light, split into H, S, V.
                        Hue and saturation are FLAT across each sphere; the entire 3D form
                        sits in V. bottom: destroy V and the depth vanishes; destroy hue and
                        the depth is untouched. Plus the honest exception - a specular
                        highlight desaturates, so gloss does show up in S.

The claim is not a metaphor. For a Lambertian surface, changing the illumination scales R,
G and B by the SAME factor, which is exactly a move along V with H and S fixed - the same
identity the HSV frame already asserts from the other direction. This script asserts it on
the rendered scene rather than restating it.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_depth.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from color_common import ARM_BLUE, ARM_RED, ensure_fig_dir, setup_logging

N = 300                                  # canvas is N x N
LIGHT = np.array([-0.45, 0.6, 0.72])     # from upper-left, towards the viewer
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT = 0.12
# (centre_x, centre_y, radius, hue) in canvas units.
# Spacing MUST exceed 2*radius: the spheres are painted in order, so any overlap means a
# later sphere covers part of an earlier one's mask and the per-sphere statistics below
# silently measure the wrong pixels. Spacing 0.31 against a diameter of 0.26.
SPHERES = [(0.19, 0.50, 0.13, 0.02),     # red
           (0.50, 0.50, 0.13, 0.33),     # green
           (0.81, 0.50, 0.13, 0.58)]     # blue
SPECULAR_ON = 2                          # index of the sphere that gets a highlight
SHININESS = 40.0


def circular_std(h):
    """Spread of hue values, in hue units, respecting the 0/1 wrap.

    A plain std() is wrong here and this script is the proof: the red sphere sits at
    H = 0.02, so its pixels straddle the seam and numpy reports a spread of 0.095 for a
    surface whose hue is in fact constant. Exactly the trap the hue-seam frame teaches.
    """
    theta = 2 * np.pi * np.asarray(h)
    r = np.abs(np.exp(1j * theta).mean())
    r = min(r, 1.0)
    return float(np.sqrt(-2.0 * np.log(r)) / (2 * np.pi)) if r > 0 else float("inf")


def render(with_specular_on=None):
    """Lambertian spheres on a flat background. Returns (rgb, shading, mask_per_sphere)."""
    yy, xx = np.mgrid[0:N, 0:N]
    x = xx / N
    y = 1.0 - yy / N
    rgb = np.full((N, N, 3), 0.90)        # light grey backdrop
    shading = np.zeros((N, N))
    masks = []

    for i, (cx, cy, r, hue) in enumerate(SPHERES):
        dx, dy = x - cx, y - cy
        rr = dx ** 2 + dy ** 2
        inside = rr <= r ** 2
        masks.append(inside)

        nz = np.sqrt(np.clip(r ** 2 - rr, 0, None)) / r
        nx, ny = dx / r, dy / r
        lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, None)
        lit = AMBIENT + (1 - AMBIENT) * lam

        base = hsv_to_rgb(np.array([hue, 0.78, 1.0]))
        # The whole point: shading multiplies the base colour, it does not shift it.
        colour = base[None, None, :] * lit[..., None]

        if with_specular_on is not None and i == with_specular_on:
            # Blinn-Phong-ish highlight: adds WHITE, which is why it desaturates.
            halfway = LIGHT + np.array([0.0, 0.0, 1.0])
            halfway /= np.linalg.norm(halfway)
            spec = np.clip(nx * halfway[0] + ny * halfway[1] + nz * halfway[2], 0, None) ** SHININESS
            colour = np.clip(colour + spec[..., None] * 0.9, 0, 1)

        rgb[inside] = colour[inside]
        shading[inside] = lit[inside]

    return np.clip(rgb, 0, 1), shading, masks


def main():
    log = setup_logging("color_depth")
    fig_dir = ensure_fig_dir()

    # Guard the layout invariant rather than trusting the constants above.
    for a in range(len(SPHERES)):
        for b in range(a + 1, len(SPHERES)):
            (xa, ya, ra, _), (xb, yb, rb, _) = SPHERES[a], SPHERES[b]
            gap = np.hypot(xa - xb, ya - yb)
            if gap <= ra + rb:
                raise ValueError(
                    f"spheres {a} and {b} overlap (centres {gap:.3f} apart, radii sum "
                    f"{ra + rb:.3f}); the per-sphere statistics would measure the wrong pixels")

    rgb, shading, masks = render()
    hsv = rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # --- the claim, measured on every sphere ---------------------------------------------
    for i, m in enumerate(masks):
        h_std, s_std = circular_std(H[m]), float(S[m].std())
        corr = float(np.corrcoef(V[m], shading[m])[0, 1])
        log.info(f"sphere {i}: over {m.sum():,} lit pixels  "
                 f"H circular std {h_std:.2e}   S std {s_std:.2e}   "
                 f"corr(V, shading) {corr:.6f}")
        if h_std > 1e-6 or s_std > 1e-6:
            raise AssertionError(
                f"sphere {i}: Lambertian shading must leave H and S fixed, got "
                f"H std {h_std:.2e}, S std {s_std:.2e}")
        if corr < 0.9999:
            raise AssertionError(
                f"sphere {i}: V should BE the shading, got correlation {corr:.6f}")
    log.info("confirmed: all the 3D form is in V; H and S are flat to machine precision")

    # --- the destroy test -----------------------------------------------------------------
    flat_v = hsv.copy()
    for m in masks:                       # flatten V per sphere, keeping hue and saturation
        flat_v[..., 2][m] = V[m].mean()
    no_depth = hsv_to_rgb(flat_v)

    grey = V.copy()                       # colour destroyed, luminance kept

    # --- the honest exception: gloss ------------------------------------------------------
    rgb_spec, _, masks_spec = render(with_specular_on=SPECULAR_ON)
    s_spec = rgb_to_hsv(rgb_spec)[..., 1]
    m = masks_spec[SPECULAR_ON]
    log.info(f"with a specular highlight, S over that sphere ranges "
             f"{s_spec[m].min():.3f} to {s_spec[m].max():.3f} "
             f"(std {s_spec[m].std():.3f}) - gloss desaturates, so it shows up in S too")
    if s_spec[m].std() < 1e-3:
        raise AssertionError("the specular sphere should show saturation variation")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.9))

    # Hue is undefined where there is no saturation, and the hsv colormap would paint the
    # grey backdrop bright red. Mask it out instead of showing a colour that means nothing.
    h_shown = np.ma.masked_where(S < 0.1, H)
    hsv_cmap = plt.get_cmap("hsv").copy()
    hsv_cmap.set_bad("#e6e6e6")

    top = [(rgb, "the scene", None),
           (h_shown, "H  hue\n(undefined on grey)", hsv_cmap),
           (S, "S  saturation", "gray"),
           (V, "V  value", "gray")]
    for ax, (im, t, cm) in zip(axes[0], top):
        ax.imshow(im, cmap=cm, vmin=None if cm is None else 0, vmax=None if cm is None else 1)
        ax.set_title(t, fontsize=10.5)
        ax.axis("off")

    bottom = [
        (rgb, "original", ""),
        (no_depth, "V flattened\n(hue and saturation kept)", "depth gone"),
        (np.dstack([grey] * 3), "colour discarded\n(only V kept)", "depth intact"),
        (rgb_spec, "gloss: the exception\n(a highlight adds white)", "S varies too"),
    ]
    for ax, (im, t, note) in zip(axes[1], bottom):
        ax.imshow(im, vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
        if note:
            colour = ARM_RED if note == "depth gone" else ARM_BLUE
            ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold", color=colour)

    fig.tight_layout(h_pad=2.2)
    out = fig_dir / "depth_shading.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
