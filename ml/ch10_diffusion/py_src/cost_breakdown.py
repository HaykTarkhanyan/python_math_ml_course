"""L31: where the saving actually comes from when you move diffusion into a latent.

Element count drops 48x. Self-attention over spatial positions drops by n^2, i.e. ~4096x.
Those are very different numbers and the slide should not blur them into one "48x".
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import save, setup_logging

log = setup_logging("cost_breakdown")

PIXEL = (512, 512, 3)
LATENT = (64, 64, 4)


def main():
    px_elems = np.prod(PIXEL)
    lt_elems = np.prod(LATENT)
    px_tokens = PIXEL[0] * PIXEL[1]
    lt_tokens = LATENT[0] * LATENT[1]
    px_attn = px_tokens**2
    lt_attn = lt_tokens**2

    log.info(f"elements   {px_elems:,} -> {lt_elems:,}   ({px_elems/lt_elems:.0f}x)")
    log.info(f"attn pairs {px_attn:,} -> {lt_attn:,}   ({px_attn/lt_attn:.0f}x)")

    groups = ["Numbers to store\n(elements)", "Self-attention cost\n(token pairs, $n^2$)"]
    pixel_vals = [px_elems, px_attn]
    latent_vals = [lt_elems, lt_attn]
    ratios = [px_elems / lt_elems, px_attn / lt_attn]

    x = np.arange(len(groups))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    b1 = ax.bar(x - w / 2, pixel_vals, w, color="#D90012", label="pixel space  512x512x3")
    b2 = ax.bar(x + w / 2, latent_vals, w, color="#0033A0", label="latent  64x64x4")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("count (log scale)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(1e3, 1e14)

    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
    # Annotate just above each group's taller bar. A fixed y collided with the legend
    # and rendered as one garbled run-on line (caught in student review).
    for xi, r, top in zip(x, ratios, pixel_vals):
        ax.text(xi, top * 12, f"{r:,.0f}x cheaper", ha="center", fontsize=10,
                weight="bold", color="#008C46")
    # Legend below the axes so it cannot overlap bars or annotations at any scale.
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)

    ax.set_title("Moving into the latent buys far more than the 48x element count suggests",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "cost_breakdown.pdf", log)


if __name__ == "__main__":
    main()
