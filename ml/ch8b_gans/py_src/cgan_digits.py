"""L23c: a conditional GAN on the 8x8 digits, plus a latent-space walk.

Two things the deck needs and cannot borrow:
  cgan_samples.pdf  - ask for a specific digit and get it (control, not just generation)
  gan_interpolation.pdf - walk a straight line in latent space; the output morphs smoothly

The second is the property that makes StyleGAN-style editing possible at all, so it is
worth showing on our own model rather than asserting it.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gan_lib import LATENT, SEED, load_data, save, setup_logging

log = setup_logging("cgan_digits")

STEPS = 9000
BATCH = 128
N_CLASSES = 10
EMB = 16


class CondGenerator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(N_CLASSES, EMB)
        self.net = nn.Sequential(
            nn.Linear(LATENT + EMB, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.Tanh(),
        )

    def forward(self, z, y):
        return self.net(torch.cat([z, self.emb(y)], dim=1))

    def forward_emb(self, z, e):
        """Generate from a raw class embedding rather than a class index, so the label
        can be interpolated continuously instead of flipping at the midpoint."""
        return self.net(torch.cat([z, e], dim=1))


class CondDiscriminator(nn.Module):
    """The label goes into the DISCRIMINATOR too - otherwise nothing forces the
    generator to respect it."""

    def __init__(self, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(N_CLASSES, EMB)
        self.net = nn.Sequential(
            nn.Linear(64 + EMB, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, self.emb(y)], dim=1))


def train(x, y):
    torch.manual_seed(SEED)
    G, D = CondGenerator(), CondDiscriminator()
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.long)

    for step in range(STEPS + 1):
        idx = torch.randint(0, len(x), (BATCH,))
        real, lab = x[idx], yt[idx]
        z = torch.randn(BATCH, LATENT)
        fake = G(z, lab)

        d_real, d_fake = D(real, lab), D(fake.detach(), lab)
        lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad(); lossD.backward(); optD.step()

        d_fake2 = D(fake, lab)
        lossG = bce(d_fake2, torch.ones_like(d_fake2))
        optG.zero_grad(); lossG.backward(); optG.step()

        if step % 1500 == 0:
            log.info(f"step {step:5d}  lossD {lossD.item():.3f}  lossG {lossG.item():.3f}")
    return G


@torch.no_grad()
def fig_conditional(G):
    """Rows = requested digit, columns = different z. Control and variety at once."""
    torch.manual_seed(SEED)
    fig, axes = plt.subplots(10, 8, figsize=(6.4, 8.0))
    for d in range(10):
        z = torch.randn(8, LATENT)
        lab = torch.full((8,), d, dtype=torch.long)
        imgs = G(z, lab).reshape(-1, 8, 8).numpy()
        for c in range(8):
            axes[d, c].imshow(imgs[c], cmap="gray_r", vmin=-1, vmax=1)
            axes[d, c].axis("off")
        axes[d, 0].axis("on"); axes[d, 0].set_xticks([]); axes[d, 0].set_yticks([])
        axes[d, 0].set_ylabel(str(d), fontsize=9, rotation=0, labelpad=8, va="center")
    fig.suptitle("Ask for a digit, get that digit\n(rows = requested class, columns = different $z$)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "cgan_samples.pdf", log)


@torch.no_grad()
def fig_interpolation(G):
    """Straight line between two latent codes, at a fixed class and across classes."""
    torch.manual_seed(SEED)
    steps = 9
    ts = np.linspace(0, 1, steps)

    fig, axes = plt.subplots(3, steps, figsize=(10.5, 3.6))
    rows = [(3, 3, "same class (3 -> 3)"),
            (1, 7, "1 -> 7"),
            (0, 8, "0 -> 8")]
    for r, (ca, cb, title) in enumerate(rows):
        za, zb = torch.randn(1, LATENT), torch.randn(1, LATENT)
        # Blend the class EMBEDDING continuously. Flipping the label index at t=0.5
        # produces a hard jump mid-row, which contradicts the caption.
        ea = G.emb(torch.tensor([ca]))
        eb = G.emb(torch.tensor([cb]))
        for i, t in enumerate(ts):
            z = (1 - t) * za + t * zb
            e = (1 - t) * ea + t * eb
            img = G.forward_emb(z, e).reshape(8, 8).numpy()
            axes[r, i].imshow(img, cmap="gray_r", vmin=-1, vmax=1)
            axes[r, i].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(title, fontsize=7.5, rotation=0, labelpad=34, va="center")
    fig.suptitle("Walking a straight line in latent space: the output moves smoothly, "
                 "and every point on the way is a digit", fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, "gan_interpolation.pdf", log)


def main():
    x, y = load_data()
    log.info(f"digits {tuple(x.shape)}, {N_CLASSES} classes")
    G = train(x, y)
    fig_conditional(G)
    fig_interpolation(G)


if __name__ == "__main__":
    main()
