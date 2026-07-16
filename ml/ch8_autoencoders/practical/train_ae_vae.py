"""Autoencoders + VAE on MNIST - trains the 3 chapter models and exports the deck figures.

This is the engine for the ch8 practical (kill 2 rabbits): the same code students run is the
source of every model-dependent figure in L22/L23. Verified as a script here, then transplanted
into ae_vae_practical.ipynb.

Compute guardrails (per AE_CHAPTER_PLAN.md): thread-capped, tiny nets, <=10 epochs, 20k subsample,
CPU, models trained sequentially. Seed 509. No downloads beyond MNIST (~11MB, torchvision, with an
openml fallback).
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -- freeze-safety + reproducibility -----------------------------------------------------------
torch.set_num_threads(4)          # do NOT peg all cores (documented lock-up risk on this laptop)
SEED = 509
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

HERE = Path(__file__).resolve().parent          # ml/ch8_autoencoders/practical
CH = HERE.parent                                 # ml/ch8_autoencoders
FIG = CH / "fig"
LOGS = CH / "logs"
DATA = HERE / "data"
for d in (FIG, LOGS, DATA):
    d.mkdir(parents=True, exist_ok=True)

LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "train_ae_vae.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ae_vae")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SUB = 20000     # train subsample (freeze-safety / speed)
EPOCHS = 8
BATCH = 256
LR = 1e-3


# -- data --------------------------------------------------------------------------------------
def load_mnist():
    """Return X_train, y_train, X_test, y_test as float tensors in [0,1], X flat (N,784)."""
    try:
        from torchvision import datasets, transforms
        tf = transforms.ToTensor()
        tr = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
        te = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
        Xtr = tr.data.float().div(255.0).view(-1, 784)
        ytr = tr.targets.clone()
        Xte = te.data.float().div(255.0).view(-1, 784)
        yte = te.targets.clone()
        log.info("MNIST loaded via torchvision")
    except Exception as e:  # loud fallback, never silent
        log.warning(f"torchvision MNIST failed ({e}); falling back to openml")
        from sklearn.datasets import fetch_openml
        mn = fetch_openml("mnist_784", version=1, as_frame=False)
        X = torch.tensor(mn.data, dtype=torch.float32).div(255.0)
        y = torch.tensor(mn.target.astype(int))
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
        log.info("MNIST loaded via openml")
    return Xtr, ytr, Xte, yte


def subsample(X, y, n):
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


# -- models ------------------------------------------------------------------------------------
class AE(nn.Module):
    """Undercomplete MLP autoencoder, 784-256-64-2 and mirror. Sigmoid output for [0,1] pixels."""

    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class VAE(nn.Module):
    """Same shape, but the encoder outputs mu and logvar; reparameterized sampling."""

    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)          # the trick: randomness in eps, gradients through mu/std
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


# -- training loops ----------------------------------------------------------------------------
def train_ae(model, X, denoise=False, noise=0.5, epochs=EPOCHS, tag="AE"):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            if denoise:
                xin = (xb + noise * torch.randn_like(xb)).clamp(0, 1)
            else:
                xin = xb
            xhat, _ = model(xin)
            loss = F.binary_cross_entropy(xhat, xb, reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        log.info(f"[{tag}] epoch {ep + 1}/{epochs}  bce={tot / len(X):.4f}")
    return model


def vae_loss(xhat, x, mu, logvar):
    recon = F.binary_cross_entropy(xhat, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return recon + kl, recon, kl


def train_vae(model, X, epochs=10, tag="VAE"):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        tr = kr = 0.0
        for (xb,) in loader:
            xhat, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(xhat, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += recon.item() * len(xb)
            kr += kl.item() * len(xb)
        log.info(f"[{tag}] epoch {ep + 1}/{epochs}  recon={tr / len(X):.2f}  kl={kr / len(X):.2f}")
    return model


# -- figures -----------------------------------------------------------------------------------
def img(ax, v):
    ax.imshow(v.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def save(fig, name):
    p = FIG / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"saved {p}")


@torch.no_grad()
def fig_recon_teaser(ae, Xte, yte):
    x = Xte[(yte < 9).nonzero(as_tuple=True)[0][0]]
    xhat, z = ae(x.unsqueeze(0))
    fig, axs = plt.subplots(1, 3, figsize=(6, 2.4))
    img(axs[0], x.numpy()); axs[0].set_title("input (784)", fontsize=10)
    axs[1].axis("off")
    axs[1].text(0.5, 0.5, f"code z =\n[{z[0,0]:.2f}, {z[0,1]:.2f}]", ha="center", va="center", fontsize=12)
    axs[1].set_title("2 numbers", fontsize=10)
    img(axs[2], xhat[0].numpy()); axs[2].set_title("reconstruction", fontsize=10)
    save(fig, "ae_recon_teaser.pdf")


@torch.no_grad()
def fig_pca_vs_ae(ae, Xn, yn):
    from sklearn.decomposition import PCA
    P = PCA(n_components=2).fit_transform(Xn.numpy())
    _, Z = ae(Xn)
    Z = Z.numpy()
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
    cmap = plt.get_cmap("tab10")
    ynn = yn.numpy()
    for ax, emb, ttl in ((axs[0], P, "PCA (linear)"), (axs[1], Z, "Autoencoder (nonlinear)")):
        for d in range(9):
            m = ynn == d
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.6, color=cmap(d), label=str(d))
        ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
    axs[1].legend(title="digit", markerscale=2.2, fontsize=7, loc="center left",
                  bbox_to_anchor=(1.01, 0.5))
    save(fig, "pca_vs_ae_mnist.pdf")


@torch.no_grad()
def fig_ae_sampling_fail(ae, Xn):
    _, Z = ae(Xn)
    Z = Z.numpy()
    g = torch.Generator().manual_seed(7)
    zs = torch.randn(6, 2, generator=g)
    dec = ae.dec(zs)
    fig = plt.figure(figsize=(9.5, 3.8))
    gs = fig.add_gridspec(2, 5, width_ratios=[3, 0.2, 1, 1, 1])
    axS = fig.add_subplot(gs[:, 0])
    axS.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.4, color="#0033A0", label="AE codes")
    axS.scatter(zs[:, 0].numpy(), zs[:, 1].numpy(), s=55, color="#D90012", marker="x", label="N(0,I) draws")
    axS.add_patch(plt.Circle((0, 0), 2.45, fill=False, color="#D90012", ls="--", lw=1.5))
    axS.set_xlabel("z1"); axS.set_ylabel("z2"); axS.set_aspect("equal"); axS.legend(fontsize=7)
    for k in range(6):
        r, c = divmod(k, 3)
        img(fig.add_subplot(gs[r, 2 + c]), dec[k].numpy())
    fig.suptitle("Plain AE: its codes (blue) never fill N(0,I) (red circle) -> N(0,I) draws decode to blurry blobs",
                 fontsize=10)
    save(fig, "ae_sampling_fail.pdf")


@torch.no_grad()
def fig_recon_pca_vs_ae(ae, Xfit, Xte, yte):
    """Reconstruct from 2 numbers: PCA(2) linear vs AE(2) nonlinear - the honest 'nonlinear wins'."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(Xfit.numpy())
    idx = torch.tensor([int((yte == d).nonzero(as_tuple=True)[0][0]) for d in range(8)])
    x = Xte[idx]
    pr = torch.tensor(pca.inverse_transform(pca.transform(x.numpy())), dtype=torch.float32).clamp(0, 1)
    ar, _ = ae(x)
    fig, axs = plt.subplots(3, 8, figsize=(10, 4))
    for k in range(8):
        img(axs[0, k], x[k].numpy())
        img(axs[1, k], pr[k].numpy())
        img(axs[2, k], ar[k].numpy())
    for r, lab in enumerate(("original", "PCA (2)", "AE (2)")):
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
        axs[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Reconstruct each digit from 2 numbers: PCA (linear) vs AE (nonlinear)", fontsize=11)
    save(fig, "recon_pca_vs_ae.pdf")


@torch.no_grad()
def fig_interp_ae_vs_vae(ae, vae, Xte, yte):
    """Same 3 -> 8 endpoints: plain AE (top, blurry) vs VAE (bottom, valid every step)."""
    i = (yte == 3).nonzero(as_tuple=True)[0][0]
    j = (yte == 8).nonzero(as_tuple=True)[0][0]
    _, zi = ae(Xte[i].unsqueeze(0)); _, zj = ae(Xte[j].unsqueeze(0))
    mi, _ = vae.encode(Xte[i].unsqueeze(0)); mj, _ = vae.encode(Xte[j].unsqueeze(0))
    ts = np.linspace(0, 1, 8)
    fig, axs = plt.subplots(2, 8, figsize=(10, 2.9))
    for k, t in enumerate(ts):
        img(axs[0, k], ae.dec((1 - t) * zi + t * zj)[0].numpy())
        img(axs[1, k], vae.dec((1 - t) * mi + t * mj)[0].numpy())
    for r, lab in enumerate(("plain AE", "VAE")):
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
        axs[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Interpolate 3 -> 8: plain AE blurs through, VAE stays a valid digit", fontsize=11)
    save(fig, "interp_ae_vs_vae.pdf")


@torch.no_grad()
def fig_sampling_ae_vs_vae(ae, vae):
    """Same z ~ N(0,I) draws decoded by AE (top) vs VAE (bottom)."""
    g = torch.Generator().manual_seed(3)
    zs = torch.randn(8, 2, generator=g)
    a = ae.dec(zs); v = vae.dec(zs)
    fig, axs = plt.subplots(2, 8, figsize=(10, 2.9))
    for k in range(8):
        img(axs[0, k], a[k].numpy())
        img(axs[1, k], v[k].numpy())
    for r, lab in enumerate(("plain AE", "VAE")):
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
        axs[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Decode the same z ~ N(0,I): AE unreliable, VAE valid digits", fontsize=11)
    save(fig, "sampling_ae_vs_vae.pdf")


@torch.no_grad()
def fig_denoising(dae, Xte, yte, noise=0.5):
    idx = torch.tensor([int((yte == d).nonzero(as_tuple=True)[0][0]) for d in range(8)])
    x = Xte[idx]
    xn = (x + noise * torch.randn_like(x)).clamp(0, 1)
    xd, _ = dae(xn)
    fig, axs = plt.subplots(3, 8, figsize=(10, 4))
    for k in range(8):
        img(axs[0, k], x[k].numpy())
        img(axs[1, k], xn[k].numpy())
        img(axs[2, k], xd[k].numpy())
    for r, lab in enumerate(("clean", "noisy", "denoised")):
        axs[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
    save(fig, "denoising_grid.pdf")


@torch.no_grad()
def fig_anomaly(ae1, Xte, yte, normal=1, anomaly=0):
    """ae1 is trained ONLY on the normal class - anomalies reconstruct badly."""
    xhat, _ = ae1(Xte)
    err = ((xhat - Xte) ** 2).mean(dim=1).numpy()
    yt = yte.numpy()
    e_norm = err[yt == normal]
    e_anom = err[yt == anomaly]
    e_rest = err[yt != normal]
    thr = float(np.quantile(e_norm, 0.95))
    log.info(f"[anomaly] mean err  normal({normal})={e_norm.mean():.4f}  "
             f"anomaly({anomaly})={e_anom.mean():.4f}  all-non-{normal}={e_rest.mean():.4f}  thr={thr:.4f}")
    fig = plt.figure(figsize=(10, 3.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[3.2, 0.9, 0.9])
    ax = fig.add_subplot(gs[:, 0])
    ax.hist(e_norm, bins=50, alpha=0.6, label=f"digit {normal} (normal)", color="#0033A0", density=True)
    ax.hist(e_anom, bins=50, alpha=0.6, label=f"digit {anomaly} (anomaly)", color="#D90012", density=True)
    ax.axvline(thr, color="k", ls="--", lw=1.5)
    ax.text(thr, ax.get_ylim()[1] * 0.92, "  threshold ->", fontsize=8, va="top")
    ax.set_xlabel("reconstruction error (MSE)"); ax.set_ylabel("density")
    ax.set_title(f"AE trained only on {normal}s: threshold the error to flag the unseen {anomaly}")
    ax.legend(loc="upper right")
    # example reconstructions: the 1s-only AE forces a 0 toward a 1-shape -> high error
    xn = Xte[yte == normal][0]; xa = Xte[yte == anomaly][0]
    rn, _ = ae1(xn.unsqueeze(0)); ra, _ = ae1(xa.unsqueeze(0))
    en = float(((rn[0] - xn) ** 2).mean()); ea = float(((ra[0] - xa) ** 2).mean())
    cells = [(gs[0, 1], xn, f"{normal} in"), (gs[0, 2], rn[0], f"out\ne={en:.3f}"),
             (gs[1, 1], xa, f"{anomaly} in"), (gs[1, 2], ra[0], f"out\ne={ea:.3f}")]
    for cell, im, ttl in cells:
        axc = fig.add_subplot(cell); img(axc, im.detach().numpy()); axc.set_title(ttl, fontsize=8)
    save(fig, "anomaly_recon_error_hist.pdf")


@torch.no_grad()
def fig_vae_grid(vae, n=18):
    from scipy.stats import norm
    ppf = norm.ppf(np.linspace(0.02, 0.98, n))
    canvas = np.zeros((28 * n, 28 * n))
    for i, yi in enumerate(ppf[::-1]):          # row 0 = highest z2 (matches origin='upper')
        for j, xi in enumerate(ppf):
            z = torch.tensor([[xi, yi]], dtype=torch.float32)
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = vae.dec(z).numpy().reshape(28, 28)
    lo, hi = float(ppf[0]), float(ppf[-1])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(canvas, cmap="gray", extent=[lo, hi, lo, hi])
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_title("VAE latent grid: every point is a valid digit")
    save(fig, "vae_latent_grid.pdf")


@torch.no_grad()
def fig_vae_samples(vae):
    zs = torch.randn(24, 2)
    dec = vae.dec(zs)
    fig, axs = plt.subplots(3, 8, figsize=(9, 3.6))
    for k, ax in enumerate(axs.flat):
        img(ax, dec[k].numpy())
    fig.suptitle("VAE: draw z ~ N(0,I), decode -> new digits", fontsize=11)
    save(fig, "vae_samples.pdf")


# -- run ---------------------------------------------------------------------------------------
def main():
    log.info("loading MNIST")
    Xtr, ytr, Xte, yte = load_mnist()
    Xtr, ytr = subsample(Xtr, ytr, N_SUB)
    Xtr08 = Xtr[ytr < 9]
    X1 = Xtr[ytr == 1]
    log.info(f"train subsample {len(Xtr)} (0-8 subset {len(Xtr08)}, 1s {len(X1)}); test {len(Xte)}")

    # small labelled subset for the scatter (speed + legibility)
    sc_idx = rng.choice((yte < 9).nonzero(as_tuple=True)[0].numpy(), size=2500, replace=False)
    Xsc, ysc = Xte[sc_idx], yte[sc_idx]

    # --- train the models (sequential, thread-capped) ---
    log.info("=== model 1/4: plain AE on digits 0-8 (latent 2) ===")
    ae = train_ae(AE(latent=2), Xtr08, tag="AE(0-8)")
    log.info("=== model 2/4: single-class AE on 1s (latent 16) for anomaly ===")
    ae1 = train_ae(AE(latent=16), X1, tag="AE(1s)")
    log.info("=== model 3/4: denoising AE (latent 32) ===")
    dae = train_ae(AE(latent=32), Xtr, denoise=True, tag="DAE")
    log.info("=== model 4/4: VAE (latent 2) on all digits ===")
    vae = train_vae(VAE(latent=2), Xtr)

    # --- figures (all models available) ---
    log.info("generating figures")
    fig_recon_teaser(ae, Xte, yte)
    fig_pca_vs_ae(ae, Xsc, ysc)
    fig_recon_pca_vs_ae(ae, Xtr08, Xte, yte)
    fig_ae_sampling_fail(ae, Xsc)
    fig_anomaly(ae1, Xte, yte, normal=1, anomaly=0)
    fig_denoising(dae, Xte, yte)
    fig_vae_grid(vae)
    fig_vae_samples(vae)
    fig_interp_ae_vs_vae(ae, vae, Xte, yte)
    fig_sampling_ae_vs_vae(ae, vae)

    log.info("DONE - all figures written to fig/")


if __name__ == "__main__":
    main()
