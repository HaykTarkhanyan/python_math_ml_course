"""L31: the part L23 did not say - the latent of a latent-diffusion model is SPATIAL.

L22/L23 taught a flat code (2 numbers for MNIST). Stable Diffusion's latent is a
64x64x4 feature map: a small image, not a vector. That is why convolutions still work
in there, and it is the whole reason latent diffusion is possible.

Uses a real photograph so the "it is still an image" point is visible rather than asserted.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image

from diffusion_lib import save, setup_logging

log = setup_logging("latent_spatial")

FACTOR = 8  # 512 -> 64


def block_pool(img, k):
    """Average-pool by an integer factor. Stands in for the encoder's downsampling."""
    h, w = img.shape[0] // k * k, img.shape[1] // k * k
    img = img[:h, :w]
    return img.reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def main():
    img = load_sample_image("flower.jpg").astype(np.float32) / 255.0
    side = min(img.shape[:2])
    img = img[:side, :side]           # square crop
    img = img[: side // FACTOR * FACTOR, : side // FACTOR * FACTOR]
    log.info(f"source image {img.shape}")

    pooled = block_pool(img, FACTOR)  # (H/8, W/8, 3)
    # Four "channels" of a pretend latent: three colour projections plus a luminance
    # gradient, purely so the figure shows 4 maps like a real 64x64x4 latent.
    chans = [pooled[..., 0], pooled[..., 1], pooled[..., 2], pooled.mean(-1)]
    log.info(f"latent grid {pooled.shape[0]}x{pooled.shape[1]} x {len(chans)} channels")

    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.1, 0.5, 1, 1, 0.35, 1.6])

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    ax.set_title(f"Image\n{img.shape[0]}x{img.shape[1]}x3", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#008C46"))
    ax.text(0.5, 0.60, "encode", ha="center", fontsize=9, color="#008C46",
            transform=ax.transAxes)

    for i, c in enumerate(chans):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        ax.imshow(c, cmap="viridis")
        ax.set_title(f"ch {i}", fontsize=8)
        ax.axis("off")

    ax = fig.add_subplot(gs[:, 5])
    ax.axis("off")
    ax.text(0.0, 0.72, "The latent is still an image.", fontsize=11, weight="bold",
            color="#1E46A0", transform=ax.transAxes)
    ax.text(0.0, 0.16,
            f"{pooled.shape[0]}x{pooled.shape[1]}x4, not a flat code.\n\n"
            "Neighbouring latents are still\nneighbouring patches, so\n"
            "convolutions and attention\nstill mean something in here.\n\n"
            "The L22/L23 latent was 2 numbers.\nThis one has spatial layout.",
            fontsize=9, transform=ax.transAxes, va="bottom")

    fig.suptitle("Latent diffusion: diffuse in a small picture, not in a vector",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "latent_spatial.pdf", log)


if __name__ == "__main__":
    main()
