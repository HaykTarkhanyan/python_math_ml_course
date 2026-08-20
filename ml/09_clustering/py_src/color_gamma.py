"""Figure for 33_color_spaces (Section 1, gamma / sRGB encoding).

Generates into ml/09_clustering/fig/:
  gamma_curve.pdf  -- left: the sRGB transfer curve, with the "is 128 half as bright?"
                      answer marked. right: what ignoring it costs - averaging two pixels
                      in gamma space produces a visibly darker result than averaging the
                      light they actually carry.

Every number the slide quotes is computed and asserted here, not typed in by hand.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/color_gamma.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_common import (ARM_BLUE, ARM_ORANGE, ARM_RED, ensure_fig_dir,
                          linear_to_srgb, setup_logging, srgb_to_linear)


def main():
    log = setup_logging("color_gamma")
    fig_dir = ensure_fig_dir()

    # --- the headline numbers -------------------------------------------------------------
    half_light_code = 255 * linear_to_srgb(np.array(0.5))
    light_at_128 = srgb_to_linear(np.array(128 / 255))
    log.info(f"pixel 128 carries {light_at_128*100:.1f}% of the light of pixel 255")
    log.info(f"half the light of pixel 255 is pixel {half_light_code:.1f}")
    assert abs(light_at_128 - 0.2159) < 5e-4, f"unexpected light at 128: {light_at_128}"
    assert abs(half_light_code - 187.5) < 0.5, f"unexpected half-light code: {half_light_code}"

    # round-trip sanity: encode(decode(x)) == x
    probe = np.linspace(0, 1, 1001)
    rt = float(np.abs(linear_to_srgb(srgb_to_linear(probe)) - probe).max())
    log.info(f"max sRGB round-trip error = {rt:.2e}")
    assert rt < 1e-12, f"sRGB transfer round-trip is broken, max error {rt:.2e}"

    # --- the averaging demo ---------------------------------------------------------------
    # Two pixels, black and white. Downscale them into one. What should the answer be?
    a, b = 0.0, 1.0                                   # sRGB codes, 0-1
    naive = (a + b) / 2                               # average the STORED numbers
    correct = linear_to_srgb((srgb_to_linear(np.array(a))
                              + srgb_to_linear(np.array(b))) / 2)
    log.info(f"averaging black+white: naive gives code {naive*255:.1f}, "
             f"correct gives code {float(correct)*255:.1f}")
    naive_light = float(srgb_to_linear(np.array(naive)))
    log.info(f"the naive answer carries {naive_light*100:.1f}% of the light it should "
             f"(50%), i.e. it is {(0.5-naive_light)/0.5*100:.0f}% too dark")

    # A checkerboard, downscaled both ways. This is the classic demo: the naive result is
    # not a rounding difference, it is a different grey.
    n = 64
    board = np.indices((n, n)).sum(axis=0) % 2       # 0/1 checkerboard, half black half white
    board_srgb = board.astype(np.float64)
    naive_grey = board_srgb.mean()
    correct_grey = float(linear_to_srgb(srgb_to_linear(board_srgb).mean()))
    log.info(f"checkerboard downscale: naive {naive_grey*255:.1f}, "
             f"correct {correct_grey*255:.1f}")

    # --- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))

    ax = axes[0]
    s = np.linspace(0, 1, 500)
    ax.plot(s * 255, srgb_to_linear(s) * 100, color=ARM_BLUE, lw=2.4,
            label="sRGB: what the number means")
    ax.plot(s * 255, s * 100, color="#999999", lw=1.6, ls=":", label="what people assume")
    ax.plot([128, 128], [0, light_at_128 * 100], color=ARM_RED, lw=1.4, ls="--")
    ax.plot([0, 128], [light_at_128 * 100] * 2, color=ARM_RED, lw=1.4, ls="--")
    ax.plot([128], [light_at_128 * 100], "o", color=ARM_RED, ms=7)
    ax.annotate(f"pixel 128\n= {light_at_128*100:.1f}% of the light",
                xy=(128, light_at_128 * 100), xytext=(150, 12),
                fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))
    ax.plot([half_light_code], [50], "o", color=ARM_ORANGE, ms=7)
    ax.annotate(f"half the light\nis pixel {half_light_code:.0f}",
                xy=(half_light_code, 50), xytext=(66, 62),
                fontsize=9, color="#9a6c00",
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE, lw=1.2))
    ax.set_xlabel("stored pixel value (0-255)")
    ax.set_ylabel("actual light emitted (%)")
    ax.set_title("The number is not the brightness")
    ax.set_xlim(0, 255); ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    swatches = [
        (board_srgb, "two pixels\n(black + white)"),
        (np.full((n, n), naive_grey), f"averaged as stored\ncode {naive_grey*255:.0f}"
                                      f"  ->  {naive_light*100:.0f}% light"),
        (np.full((n, n), correct_grey), f"averaged as light\ncode {correct_grey*255:.0f}"
                                        f"  ->  50% light"),
    ]
    for i, (im, title) in enumerate(swatches):
        sub = ax.inset_axes([i / 3 + 0.012, 0.12, 1 / 3 - 0.024, 0.72])
        sub.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        sub.set_title(title, fontsize=8.6)
        sub.set_xticks([]); sub.set_yticks([])
    ax.axis("off")
    ax.set_title("Shrinking an image the naive way loses light", fontsize=11)

    fig.tight_layout()
    out = fig_dir / "gamma_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
