"""L23b: a real GAN, trained here, on the same 8x8 digits as the diffusion chapter.

The point is NOT sample quality. It is the training dynamics: the losses do not go down,
the discriminator hovers near chance, and mode coverage has to be measured because you
cannot see it. Everything the deck claims about GAN instability comes from this run.

Emits:
  gan_training.pdf  - G and D losses, and D accuracy, over training
  gan_samples.pdf   - what the generator produces at several points in training
  gan_coverage.pdf  - which digit classes the generator actually covers (mode collapse)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from gan_lib import (LATENT, SEED, Discriminator, Generator, load_data, save,
                     setup_logging)

log = setup_logging("gan_digits")

STEPS = 8000
BATCH = 128
SNAPSHOTS = [0, 500, 2000, 8000]


def train(x):
    torch.manual_seed(SEED)
    G, D = Generator(), Discriminator()
    # DCGAN's betas. beta1=0.5 rather than 0.9 is one of the few settings that reliably
    # keeps this from diverging - itself worth a sentence on the slide.
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    hist = {"g": [], "d": [], "acc": []}
    snaps = {}
    fixed_z = torch.randn(32, LATENT)

    for step in range(STEPS + 1):
        if step in SNAPSHOTS:
            with torch.no_grad():
                snaps[step] = G(fixed_z).clone()

        idx = torch.randint(0, len(x), (BATCH,))
        real = x[idx]
        z = torch.randn(BATCH, LATENT)
        fake = G(z)

        # --- discriminator: real -> 1, fake -> 0 -------------------------------
        d_real = D(real)
        d_fake = D(fake.detach())
        lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad()
        lossD.backward()
        optD.step()

        # --- generator: NON-SATURATING loss, maximize log D(G(z)) --------------
        # Not minimizing log(1 - D(G(z))): that saturates when D wins early, which is
        # the gradient-vanishing failure the deck describes.
        d_fake2 = D(fake)
        lossG = bce(d_fake2, torch.ones_like(d_fake2))
        optG.zero_grad()
        lossG.backward()
        optG.step()

        with torch.no_grad():
            acc = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()).item() / 2
        hist["g"].append(lossG.item())
        hist["d"].append(lossD.item())
        hist["acc"].append(acc)

        if step % 1000 == 0:
            log.info(f"step {step:5d}  lossD {lossD.item():.3f}  lossG {lossG.item():.3f}  "
                     f"D acc {acc:.3f}")

    return G, hist, snaps


def fig_training(hist):
    k = 100
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))

    ax = axes[0]
    ax.plot(sm(hist["d"]), color="#0033A0", lw=1.5, label="discriminator loss")
    ax.plot(sm(hist["g"]), color="#D90012", lw=1.5, label="generator loss")
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("loss", fontsize=9)
    ax.set_title("Neither loss is going anywhere", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    acc = sm(hist["acc"])
    ax.plot(acc, color="#008C46", lw=1.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.text(len(acc) * 0.40, 0.52, "0.5 = theoretical equilibrium (never reached here)",
            fontsize=8, color="gray")
    # The accuracy peak is where the generator collapsed - mark it, because the two
    # figures only make sense together.
    peak = int(np.argmax(acc))
    ax.plot(peak, acc[peak], "o", color="#D90012", ms=6)
    ax.annotate(f"discriminator dominating ({acc[peak]:.2f})\nsamples collapsed here",
                xy=(peak, acc[peak]), xytext=(peak + len(acc) * 0.18, acc[peak] + 0.02),
                fontsize=8, color="#D90012",
                arrowprops=dict(arrowstyle="->", color="#D90012", lw=1))
    log.info(f"D-accuracy peak {acc[peak]:.3f} at step ~{peak}")
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("discriminator accuracy", fontsize=9)
    ax.set_title("The only number that means anything", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    log.info(f"final smoothed D acc {sm(hist['acc'])[-1]:.3f}")
    fig.tight_layout()
    save(fig, "gan_training.pdf", log)


def fig_samples(snaps):
    fig, axes = plt.subplots(len(SNAPSHOTS), 10, figsize=(10, 4.4))
    for r, step in enumerate(SNAPSHOTS):
        imgs = snaps[step].reshape(-1, 8, 8).numpy()
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray_r", vmin=-1, vmax=1)
            ax.axis("off")
        axes[r, 0].set_ylabel(f"step {step}", fontsize=8)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("Same 10 latent vectors, followed through training", fontsize=11, y=0.99)
    fig.tight_layout()
    save(fig, "gan_samples.pdf", log)


def fig_coverage(G, x, y):
    """Mode coverage, measured rather than eyeballed.

    Train a plain classifier on the REAL digits, then use it to label 2000 generated
    samples. If the generator has dropped modes, some classes will be under-represented.
    """
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(x.numpy(), y)
    train_acc = clf.score(x.numpy(), y)
    log.info(f"class-probe accuracy on real digits: {train_acc:.3f}")

    with torch.no_grad():
        gen = G(torch.randn(2000, LATENT)).numpy()
    pred = clf.predict(gen)
    gen_counts = np.bincount(pred, minlength=10) / len(pred)
    real_counts = np.bincount(y, minlength=10) / len(y)

    for d in range(10):
        log.info(f"digit {d}: real {real_counts[d]:.3f}  generated {gen_counts[d]:.3f}")
    worst = int(np.argmin(gen_counts))
    log.info(f"least-covered class: {worst} at {gen_counts[worst]:.3f} "
             f"(real {real_counts[worst]:.3f})")
    # Total variation distance: 0 = perfect coverage, 1 = complete collapse.
    tv = 0.5 * np.abs(gen_counts - real_counts).sum()
    log.info(f"total variation distance from the real class distribution: {tv:.3f}")

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    w = 0.38
    idx = np.arange(10)
    b1 = ax.bar(idx - w / 2, real_counts, w, color="#0033A0", label="real digits")
    b2 = ax.bar(idx + w / 2, gen_counts, w, color="#D90012", label="GAN samples")
    ax.bar_label(b1, fmt="%.2f", fontsize=7, padding=1)
    ax.bar_label(b2, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(idx)
    ax.set_xlabel("digit class (as judged by a probe trained on real data)", fontsize=9)
    ax.set_ylabel("share of samples", fontsize=9)
    ax.set_title(f"Mode coverage has to be measured, not eyeballed "
                 f"(total variation = {tv:.2f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "gan_coverage.pdf", log)
    return tv


def main():
    x, y = load_data()
    log.info(f"digits: {tuple(x.shape)}, range [{x.min():.1f}, {x.max():.1f}]")
    G, hist, snaps = train(x)
    fig_training(hist)
    fig_samples(snaps)
    fig_coverage(G, x, y)


if __name__ == "__main__":
    main()
