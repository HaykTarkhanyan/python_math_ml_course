"""L28/L29: a real DDPM, trained here, on sklearn's 8x8 handwritten digits.

Deliberately small. The point is that the derivation in L28 turns into a working generative
model in a few dozen lines and a few minutes of CPU - not that we can match Stable Diffusion.
The slides say so explicitly; do not present these samples as state of the art.

8x8 (load_digits, 1797 images) rather than 28x28 MNIST because a 28x28 run is hours on this
machine, and the teaching value is identical.

Emits:
  digits_training.pdf - the loss curve, and what the denoiser recovers at each noise level
  digits_samples.pdf  - samples generated from pure noise, and one reverse trajectory
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

from diffusion_lib import (SEED, cosine_schedule, linear_schedule, save, schedule_terms,
                           setup_logging, timestep_embedding)

log = setup_logging("digits_ddpm")

T = 1000
STEPS = 6000
BATCH = 128
T_DIM = 32


class TinyUNet(nn.Module):
    """A miniature UNet: downsample, process, upsample, with a skip connection.

    The same three ingredients as the real thing (conv stack, skip, time conditioning),
    just at 8x8 instead of 64x64. Time enters as a per-channel bias, which is the
    standard trick.
    """

    def __init__(self, ch=48):
        super().__init__()
        self.t_proj = nn.Sequential(nn.Linear(T_DIM, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.down = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1), nn.SiLU())
        self.out = nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        h = self.down(x) + self.t_proj(timestep_embedding(t, T_DIM))[:, :, None, None]
        return self.out(torch.cat([h, self.mid(h)], dim=1))   # the skip connection


def load_data():
    d = load_digits()
    x = d.images.astype(np.float32) / 16.0          # 0..1
    x = x * 2.0 - 1.0                               # -> [-1, 1], as DDPM expects
    return torch.tensor(x)[:, None, :, :], d.target


def train(x, betas):
    torch.manual_seed(SEED)
    _, abars = schedule_terms(betas)
    ab = torch.tensor(abars, dtype=torch.float32)
    model = TinyUNet()
    n_par = sum(p.numel() for p in model.parameters())
    log.info(f"TinyUNet parameters: {n_par:,}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-4)

    losses = []
    for step in range(STEPS):
        idx = torch.randint(0, len(x), (BATCH,))
        x0 = x[idx]
        t = torch.randint(0, T, (BATCH,))
        eps = torch.randn_like(x0)
        a = ab[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = ((eps - model(xt, t)) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            log.info(f"step {step:5d}  loss {loss.item():.4f}")
    return model, np.array(losses)


@torch.no_grad()
def ddpm_sample(model, betas, n=32, keep=None):
    alphas, abars = schedule_terms(betas)
    a, ab, b = (torch.tensor(v, dtype=torch.float32) for v in (alphas, abars, betas))
    torch.manual_seed(SEED)
    x = torch.randn(n, 1, 8, 8)
    snaps = {}
    for t in range(T - 1, -1, -1):
        tb = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tb)
        x = (x - b[t] / (1 - ab[t]).sqrt() * eps) / a[t].sqrt()
        if t > 0:
            x = x + b[t].sqrt() * torch.randn_like(x)
        if keep and t in keep:
            snaps[t] = x.clone()
    return x, snaps


@torch.no_grad()
def fig_training(model, losses, x, betas):
    _, abars = schedule_terms(betas)
    ab = torch.tensor(abars, dtype=torch.float32)
    show_t = [0, 150, 400, 650, 950]
    rows = 3
    x0 = x[[3, 17, 42]]

    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(rows, 2 + 2 * len(show_t), width_ratios=[6, 1.1] + [1] * (2 * len(show_t)))

    ax = fig.add_subplot(gs[:, 0])
    k = 50
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(smooth, color="#0033A0", lw=1.6)
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("MSE  $\\|\\epsilon-\\epsilon_\\theta\\|^2$", fontsize=9)
    ax.set_title("The whole derivation, optimized", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    for j, t in enumerate(show_t):
        tb = torch.full((rows,), t, dtype=torch.long)
        eps = torch.randn_like(x0)
        a = ab[t][None, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        x0_hat = (xt - (1 - a).sqrt() * model(xt, tb)) / a.sqrt()
        for i in range(rows):
            for k2, img in enumerate([xt[i, 0], x0_hat[i, 0]]):
                ax = fig.add_subplot(gs[i, 2 + 2 * j + k2])
                ax.imshow(img.numpy(), cmap="gray_r", vmin=-1, vmax=1)
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"$t={t}$\n" + ("noisy" if k2 == 0 else "$\\hat{x}_0$"),
                                 fontsize=7)
    fig.suptitle("Trained denoiser: given a noisy digit, predict the clean one "
                 "(8x8 digits, a deliberately tiny model)", fontsize=11, y=1.08)
    fig.tight_layout()
    save(fig, "digits_training.pdf", log)


def fig_samples(model, betas):
    keep = {950, 750, 500, 300, 150, 50, 0}
    samples, snaps = ddpm_sample(model, betas, n=32, keep=keep)
    log.info(f"sample range [{samples.min():.2f}, {samples.max():.2f}]")

    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(4, 15, width_ratios=[1] * 8 + [0.4] + [1] * 6)

    for i in range(32):
        ax = fig.add_subplot(gs[i // 8, i % 8])
        ax.imshow(samples[i, 0].numpy(), cmap="gray_r", vmin=-1, vmax=1)
        ax.axis("off")
    fig.text(0.25, 1.02, "32 digits generated from pure noise", ha="center", fontsize=11)

    for j, t in enumerate(sorted(keep, reverse=True)[:6]):
        for i in range(4):
            ax = fig.add_subplot(gs[i, 9 + j])
            ax.imshow(snaps[t][i, 0].numpy(), cmap="gray_r", vmin=-1, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"$t={t}$", fontsize=8)
    fig.text(0.79, 1.02, "the reverse process, step by step", ha="center", fontsize=11)
    fig.tight_layout()
    save(fig, "digits_samples.pdf", log)


@torch.no_grad()
def _sample_variant(model, betas_sample, n=16, stride=1, sigma_scale=1.0):
    """Sample with a deliberately wrong setting, to show what each bug looks like.

    betas_sample : the schedule used AT SAMPLING TIME (a bug if it differs from training)
    stride       : skip steps naively, without rescaling (the wrong way to go faster)
    sigma_scale  : multiply the injected noise (a bug if != 1)
    """
    alphas, abars = schedule_terms(betas_sample)
    a, ab, b = (torch.tensor(v, dtype=torch.float32) for v in (alphas, abars, betas_sample))
    torch.manual_seed(SEED)
    x = torch.randn(n, 1, 8, 8)
    for t in range(T - 1, -1, -stride):
        tb = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tb)
        x = (x - b[t] / (1 - ab[t]).sqrt() * eps) / a[t].sqrt()
        if t > 0:
            x = x + sigma_scale * b[t].sqrt() * torch.randn_like(x)
    return x


def fig_failures(model, betas):
    """L29: what the four classic implementation bugs actually look like.

    Every panel uses the SAME correctly-trained model. These are all sampling-time
    faults, which is the point - the model is fine and the output is still ruined.
    """
    # A cosine schedule at sampling time when the model was trained on linear: a real
    # mistake (the two are swapped freely in codebases), and a much bigger shape change
    # than merely moving beta_end.
    wrong_schedule = cosine_schedule(T)
    variants = [
        ("Correct", dict()),
        ("Schedule mismatch: SILENT\ndifferent digits, still clean", dict(betas_sample=wrong_schedule)),
        ("Naive step skipping\nevery 40th step, no rescale", dict(stride=40)),
        ("Wrong $\\sigma_t$\n3x too much noise", dict(sigma_scale=3.0)),
    ]

    fig, axes = plt.subplots(2, 8 * len(variants) // 4, figsize=(12.5, 3.4))
    fig.clf()
    gs = fig.add_gridspec(2, 4 * 5, wspace=0.08)

    for vi, (title, kw) in enumerate(variants):
        kw.setdefault("betas_sample", betas)
        s = _sample_variant(model, n=8, **kw)
        rng = float(s.max() - s.min())
        log.info(f"{title.splitlines()[0]:22s} range {s.min():+.2f}..{s.max():+.2f} "
                 f"(spread {rng:.2f})")
        for i in range(8):
            ax = fig.add_subplot(gs[i // 4, vi * 5 + i % 4])
            ax.imshow(s[i, 0].numpy(), cmap="gray_r", vmin=-1, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=8, loc="left")
    fig.suptitle("Same trained model, three sampling bugs. Two announce themselves. "
                 "One does not.", fontsize=11, y=1.06)
    save(fig, "digits_failures.pdf", log)


def load_or_train(x, betas):
    """Train once, then reuse. Iterating on figures should not cost a retrain."""
    ckpt = Path(__file__).resolve().parent / "_cache" / "digits_ddpm.pt"
    ckpt.parent.mkdir(exist_ok=True)
    if ckpt.exists():
        log.info(f"loading cached weights from {ckpt} (delete to force a retrain)")
        model = TinyUNet()
        blob = torch.load(ckpt, weights_only=True)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, np.array(blob["losses"])
    model, losses = train(x, betas)
    # losses as a plain list: a numpy array trips torch.load(weights_only=True).
    torch.save({"state_dict": model.state_dict(), "losses": losses.tolist()}, ckpt)
    log.info(f"cached weights to {ckpt}")
    return model, losses


def main():
    x, _ = load_data()
    log.info(f"digits: {tuple(x.shape)}, range [{x.min():.1f}, {x.max():.1f}]")
    betas = linear_schedule(T, 1e-4, 0.02)
    _, abars = schedule_terms(betas)
    signal_left = float(np.sqrt(abars[-1]))
    log.info(f"sqrt(abar_T) = {signal_left:.5f}")
    # If the forward process does not actually reach noise, x_T is not a sample from
    # N(0, I) and every generated image inherits a bias from its starting draw. Caught
    # this at T=400 (0.132 left); fail rather than quietly produce misleading figures.
    if signal_left > 0.02:
        raise RuntimeError(
            f"forward process does not reach noise: sqrt(abar_T)={signal_left:.4f} > 0.02. "
            f"Raise T (currently {T}) or beta_end."
        )

    model, losses = load_or_train(x, betas)
    log.info(f"final loss: {losses[-200:].mean():.4f}")
    fig_training(model, losses, x, betas)
    fig_samples(model, betas)
    fig_failures(model, betas)


if __name__ == "__main__":
    main()
