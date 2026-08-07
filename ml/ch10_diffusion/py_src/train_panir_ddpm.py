"""Class-conditional DDPM on the five ՊԱՆԻՐ letters.

The chapter's engine. Same recipe as digits_ddpm.py (T=1000, linear beta, eps-prediction),
plus a class embedding with a reserved null token so the one model serves both the conditional
and unconditional branches of classifier-free guidance - the L30 recipe, matching
diffusion_lib.train_eps_model's label-dropout scheme.

**Run this on a GPU.** Measured: 28-53 ms/step on a rented T4 against 6.7 s/step locally, a
~200x gap that turns an 11-hour run into minutes. The Colab CLI lives in WSL; mirror this repo's
layout under /content/repo so diffusion_lib's parents[3] log path still resolves. Locally it
works but is an overnight job, so it checkpoints every CHECKPOINT_EVERY steps and resumes
automatically. Never run two of these at once (diffusion_lib:23).

Config is (DATA_SIZE, CH, LEVELS, STEPS); TAG keeps experiment artifacts apart. Defaults
reproduce the shipped model - see DECISIONS.md #8 for why LEVELS is 1 and not 2.

Reads:  data/mashtots_panir_<DATA_SIZE>.npz   (run pack_mashtots.py <size> first)
Writes: data/panir_ddpm_<suf>.pt              - weights, ships with the course
        data/panir_progression_<suf>.npz      - the word sampled every PREVIEW_EVERY steps
        fig/panir_training_<suf>.pdf          - loss curve + per-class samples
        fig/panir_samples_<suf>.pdf           - samples, guidance sweep, and the word
        fig/panir_progression_<suf>.pdf       - noise -> letters, one row per snapshot
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_lib import (SEED, linear_schedule, save, schedule_terms, setup_logging,
                           timestep_embedding)

# The Windows console defaults to cp1252 and mangles the Armenian letters this logs.
# Guarded because Jupyter/Colab replace stdout with an OutStream that has no reconfigure().
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

log = setup_logging("train_panir_ddpm")

CHAPTER = Path(__file__).resolve().parent.parent

# Which packed dataset to train on, and a suffix keeping experiment outputs apart.
# Defaults reproduce the chapter's shipped names: data/panir_ddpm_24.pt etc.
DATA_SIZE = 24
TAG = ""            # e.g. "_lvl1" - every artifact of a run carries it


def paths():
    """(dataset, weights, resumable checkpoint, progression array) for the current config."""
    d = CHAPTER / "data"
    suf = f"{DATA_SIZE}{TAG}"
    return (d / f"mashtots_panir_{DATA_SIZE}.npz", d / f"panir_ddpm_{suf}.pt",
            d / f"panir_ddpm_{suf}_ckpt.pt", d / f"panir_progression_{suf}.npz")

T = 1000
STEPS = 10000
BATCH = 64
LR = 2e-3
CH = 96
# Down/up levels. ONE, not two - see DECISIONS.md #8. Halving twice (24 -> 12 -> 6) leaves the
# 1-2 px strokes sub-pixel in the deep layers, and a 7.03M two-level model produced fragments
# while this 1.50M one-level model produced legible letters at a better loss (0.0269 vs 0.038).
LEVELS = 1
CHECKPOINT_EVERY = 250
# Sampler for the final figures. "ddpm" walks all T steps and is the honest one; "ddim"
# subsamples to 50 and is ~20x cheaper. Set to "ddim" for a quick placeholder run - the
# full loop is long enough to be worth interrupting, and it has been, repeatedly.
FINAL_SAMPLER = "ddpm"
DROP_PROB = 0.15        # label dropout -> the null token; same default as train_eps_model
T_DIM = 32
PREVIEW_EVERY = 2000
# CPU locally, CUDA on a rented Colab runtime. Batch construction and every random draw stay on
# the CPU generator so a run is bit-identical regardless of device; only the model maths moves.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CondUNet(nn.Module):
    """Two-level conditional UNet: 24 -> 12 -> 6 -> 12 -> 24, with skips at both scales.

    The first attempt reused digits_ddpm's TinyUNet (one down/up level, ch=48, 266k params).
    Its loss went flat by step 800 and stayed there while samples were still malformed - the
    signature of a capacity limit rather than undertraining - and the thinnest class, Ի,
    barely rendered at all. An 8x8 digit is a much smaller thing to model than 24x24 cursive.

    Index n_classes is the reserved *null* token: training drops the real label with
    probability DROP_PROB, so at sampling time eps(x, t, null) is the unconditional
    prediction and classifier-free guidance is a difference of two forward passes.
    """

    def __init__(self, n_classes, ch=64, levels=2):
        super().__init__()
        self.n_classes = n_classes
        self.levels = levels
        self.t_proj = nn.Sequential(nn.Linear(T_DIM, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.y_emb = nn.Embedding(n_classes + 1, ch)          # +1 = null

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.SiLU())

        chans = [ch * 2 ** i for i in range(levels + 1)]       # levels=2 -> [ch, 2ch, 4ch]
        self.enc, self.downs, self.to_enc = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev = 1
        for i in range(levels):
            self.enc.append(block(prev, chans[i]))
            self.to_enc.append(nn.Linear(ch, chans[i]))
            self.downs.append(nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1), nn.SiLU()))
            prev = chans[i + 1]

        self.mid = block(chans[levels], chans[levels])
        self.to_mid = nn.Linear(ch, chans[levels])

        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(levels)):
            self.ups.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 4, stride=2, padding=1))
            self.dec.append(block(chans[i] * 2, chans[i]))     # *2 for the skip concat
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t, y):
        c = self.t_proj(timestep_embedding(t, T_DIM)) + self.y_emb(y)
        h, skips = x, []
        for i in range(self.levels):
            h = self.enc[i](h) + self.to_enc[i](c)[:, :, None, None]
            skips.append(h)
            h = self.downs[i](h)
        h = self.mid(h) + self.to_mid(c)[:, :, None, None]
        for j, i in enumerate(reversed(range(self.levels))):
            h = self.dec[j](torch.cat([self.ups[j](h), skips[i]], dim=1))
        return self.out(h)


def load_data():
    data_path = paths()[0]
    if not data_path.exists():
        raise FileNotFoundError(f"missing {data_path} - run pack_mashtots.py {DATA_SIZE} first")
    d = np.load(data_path, allow_pickle=False)
    x = d["X"].astype(np.float32) / 255.0
    x = torch.from_numpy(x * 2 - 1).unsqueeze(1)              # [-1,1], NCHW
    y = torch.from_numpy(d["y"])
    letters = [str(s) for s in d["letters"]]
    log.info("data %s  labels %s  letters %s", tuple(x.shape), np.bincount(y.numpy()).tolist(), letters)
    return x, y, letters


@torch.no_grad()
def sample(model, n_classes, betas, y, guidance=1.0, mode="ddpm", n_steps=50, size=24, seed=SEED):
    """Reverse diffusion. mode='ddim' subsamples to n_steps and is ~20x faster for previews."""
    # NB: this reseeds the *global* RNG. Training uses its own Generator, so calling this
    # for mid-training previews cannot perturb the run.
    torch.manual_seed(seed)
    n_T = len(betas)                                  # from the schedule, not the module constant
    alphas, abars = schedule_terms(betas)
    a_t = torch.tensor(alphas, dtype=torch.float32)
    ab_t = torch.tensor(abars, dtype=torch.float32)
    b_t = torch.tensor(betas, dtype=torch.float32)
    dev = next(model.parameters()).device
    a_t, ab_t, b_t = a_t.to(dev), ab_t.to(dev), b_t.to(dev)
    y = y.to(dev)
    null = torch.full_like(y, n_classes)
    x = torch.randn(len(y), 1, size, size, device=dev)

    def eps_at(x, t_b):
        e_c = model(x, t_b, y)
        if guidance == 1.0:
            return e_c
        e_u = model(x, t_b, null)                     # one call, not two
        return e_u + guidance * (e_c - e_u)

    if mode == "ddim":
        ts = np.linspace(n_T - 1, 0, n_steps).round().astype(int)
    elif mode == "ddpm":
        ts = np.arange(n_T - 1, -1, -1)
    else:
        raise ValueError(f"unknown sampler mode: {mode!r}")

    for i, t in enumerate(ts):
        t_b = torch.full((len(y),), int(t), dtype=torch.long, device=dev)
        eps = eps_at(x, t_b)
        if mode == "ddim":
            ab = ab_t[t]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            ab_prev = ab_t[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            mean = (x - b_t[t] / (1 - ab_t[t]).sqrt() * eps) / a_t[t].sqrt()
            # the noise re-injection L29 turns on; dropping it collapses samples to the mean
            x = (mean + b_t[t].sqrt() * torch.randn_like(x)) if t > 0 else mean
    return x.clamp(-1, 1).squeeze(1).cpu().numpy()


def grid(ax, imgs, title, cols):
    """Tile (N, h, w) into rows of `cols`. One long strip is unreadable past ~8 images."""
    imgs = np.asarray(imgs)
    rows = int(np.ceil(len(imgs) / cols))
    if pad := rows * cols - len(imgs):
        imgs = np.concatenate([imgs, -np.ones((pad, *imgs.shape[1:]), dtype=imgs.dtype)])
    h, w = imgs.shape[1:]
    tiled = imgs.reshape(rows, cols, h, w).transpose(0, 2, 1, 3).reshape(rows * h, cols * w)
    ax.imshow(tiled, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    data_path, weights_path, ckpt_path, prog_path = paths()
    suf = f"{DATA_SIZE}{TAG}"
    x, y, letters = load_data()
    n_classes = len(letters)
    size = x.shape[-1]

    betas = linear_schedule(T)
    _, abars = schedule_terms(betas)
    # digits_ddpm hit this exact bug at T=400: if the forward process never reaches noise,
    # x_T is not a draw from N(0,I) and every sample inherits a bias from its start.
    if abars[-1] ** 0.5 > 0.02:
        raise RuntimeError(f"sqrt(abar_T)={abars[-1] ** 0.5:.4f} too large - forward process never reaches noise")
    log.info("T=%d linear beta, sqrt(abar_T)=%.4f", T, abars[-1] ** 0.5)

    torch.manual_seed(SEED)
    model = CondUNet(n_classes, ch=CH, levels=LEVELS).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    ab = torch.tensor(abars, dtype=torch.float32)
    g = torch.Generator().manual_seed(SEED)
    log.info("CondUNet %d params, %d steps, batch %d, device %s (%s)", n_par, STEPS, BATCH, DEVICE,
             torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else f"{torch.get_num_threads()} threads")

    # Progression: the same word sampled from the same seed at fixed intervals, so the only
    # thing changing down the figure is the model. Watching ՊԱՆԻՐ emerge from noise is the
    # clearest picture of what training actually does.
    prog = []

    def snapshot(step):
        model.eval()
        s = sample(model, n_classes, betas, torch.arange(n_classes), mode="ddim",
                   n_steps=50, size=size)
        model.train()
        prog.append((step, s))
        log.info("  snapshot @%-6d per-letter ink %s", step,
                 [round(float((p > 0).mean()), 3) for p in s])

    losses, start_step = [], 1
    if ckpt_path.exists():
        c = torch.load(ckpt_path, weights_only=False)
        cfg = (c.get("ch"), c.get("levels"), c.get("n_classes"), c.get("size"), c.get("T"))
        if cfg == (CH, LEVELS, n_classes, size, T):
            model.load_state_dict(c["state_dict"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            g.set_state(c["gen_state"])
            losses, start_step = c["losses"], c["step"] + 1
            log.info("resuming from %s at step %d/%d", ckpt_path.name, start_step, STEPS)
        else:
            # Do not silently train a different model than the checkpoint holds.
            raise RuntimeError(f"checkpoint {ckpt_path} is for config {cfg}, not {(CH, LEVELS, n_classes, size, T)} - "
                               f"delete it to start fresh")
    if start_step > STEPS:
        log.info("checkpoint is already at step %d >= STEPS=%d; skipping training", start_step - 1, STEPS)

    if start_step == 1:
        snapshot(0)                      # the untrained model, for contrast

    for step in range(start_step, STEPS + 1):
        i = torch.randint(0, len(x), (BATCH,), generator=g)
        x0, y0 = x[i], y[i].clone()
        y0[torch.rand(BATCH, generator=g) < DROP_PROB] = n_classes     # -> null token
        t = torch.randint(0, T, (BATCH,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        a = ab[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        # batch built on CPU with the seeded generator, then handed to the device
        xt, t, y0, eps = xt.to(DEVICE), t.to(DEVICE), y0.to(DEVICE), eps.to(DEVICE)
        loss = ((model(xt, t, y0) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 250 == 0:
            log.info("step %5d/%d  loss %.4f (last 250: %.4f)  lr %.2e",
                     step, STEPS, losses[-1], float(np.mean(losses[-250:])), sched.get_last_lr()[0])
        if step % CHECKPOINT_EVERY == 0:
            tmp = ckpt_path.with_suffix(".tmp")          # write-then-rename: a kill mid-save can't corrupt it
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "gen_state": g.get_state(),
                        "losses": losses, "step": step, "ch": CH, "levels": LEVELS, "n_classes": n_classes,
                        "size": size, "T": T, "letters": letters}, tmp)
            tmp.replace(ckpt_path)
        if step % PREVIEW_EVERY == 0:
            snapshot(step)

    # .cpu() so the shipped weights load on a student's CPU-only laptop without map_location -
    # a CUDA-saved state_dict raises on a machine with no GPU.
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "n_classes": n_classes, "ch": CH, "levels": LEVELS,
                "letters": letters, "T": T, "size": size}, weights_path)
    log.info("wrote %s (%.2f MB)", weights_path, weights_path.stat().st_size / 1e6)

    model.eval()
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
    axes[0].plot(losses, lw=0.4, alpha=0.5, color="#0033A0")
    k = min(100, max(2, len(losses) // 10))       # stay valid for short runs too
    axes[0].plot(np.arange(k - 1, len(losses)), np.convolve(losses, np.ones(k) / k, "valid"),
                 lw=1.4, color="#D90012", label=f"running mean ({k})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("MSE on $\\epsilon$")
    axes[0].set_title(f"Training loss - final {np.mean(losses[-250:]):.4f}", fontsize=9)
    axes[0].legend(fontsize=7)
    # one batched call, not one per class - a 1000-step loop costs the same at batch 6 or 30
    y_grid = torch.arange(n_classes).repeat_interleave(6)
    reps = sample(model, n_classes, betas, y_grid, mode="ddim", n_steps=50, size=size)
    grid(axes[1], reps, "DDIM-50 samples, one row per class: " + " ".join(letters), cols=6)
    fig.tight_layout()
    save(fig, f"panir_training_{suf}.pdf", log)
    plt.close(fig)

    # height ratios track the row counts (5 / 4 / 1) so all three panels draw at the same pixel scale
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [5, 4, 1.5]})
    tag = f"DDPM-{T}" if FINAL_SAMPLER == "ddpm" else "DDIM-50"
    full = sample(model, n_classes, betas, y_grid, mode=FINAL_SAMPLER, size=size)
    grid(axes[0], full, f"{tag} samples, one row per class", cols=6)
    sweep = np.concatenate([sample(model, n_classes, betas, torch.arange(n_classes),
                                   guidance=gw, mode="ddim", n_steps=50, size=size)
                            for gw in (1.0, 2.0, 3.0, 5.0)], axis=0)
    grid(axes[1], sweep, "guidance sweep - one row per w = 1 / 2 / 3 / 5, each spelling the word", cols=n_classes)
    word = sample(model, n_classes, betas, torch.arange(n_classes), mode=FINAL_SAMPLER, size=size)
    grid(axes[2], word, f"ՊԱՆԻՐ - each letter generated independently ({tag})", cols=n_classes)
    fig.tight_layout()
    save(fig, f"panir_samples_{suf}.pdf", log)
    plt.close(fig)

    ink = [round(float((w > 0).mean()), 3) for w in word]
    log.info("word per-letter ink %s  (spread %.3f - the inconsistency the homework ends on)",
             ink, max(ink) - min(ink))

    if prog:
        steps_seen = [s for s, _ in prog]
        arr = np.stack([g for _, g in prog])                     # (n_snapshots, n_classes, H, W)
        np.savez_compressed(prog_path,
                            steps=np.array(steps_seen), samples=arr, letters=np.array(letters))
        log.info("progression: %d snapshots at steps %s", len(prog), steps_seen)

        fig, axes = plt.subplots(len(prog), 1, figsize=(5.5, 0.85 * len(prog) + 0.6))
        for ax, (st, s) in zip(np.atleast_1d(axes), prog):
            ax.imshow(np.concatenate(list(s), axis=1), cmap="gray", vmin=-1, vmax=1)
            ax.set_ylabel(f"{st:,}", fontsize=7, rotation=0, ha="right", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle("ՊԱՆԻՐ during training - same seed, same prompt, one row per step",
                     fontsize=9)
        fig.tight_layout()
        save(fig, f"panir_progression_{suf}.pdf", log)
        plt.close(fig)


if __name__ == "__main__":
    main()
