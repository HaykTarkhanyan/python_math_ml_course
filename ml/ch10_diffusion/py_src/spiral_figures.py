"""L29: the whole unconditional spiral story, from one trained model.

Emits:
  spiral_data.pdf        - the dataset, and what noising does to it (the OU random walk)
  spiral_score_field.pdf - the learned score field at three noise levels
  spiral_samplers.pdf    - DDPM vs the naive no-noise variant vs DDIM

The naive no-noise panel is the one the deck builds to: deleting the sampler's noise feeds
the denoiser inputs at a noise level it never trained on, so it over-denoises and everything
piles up near the data mean.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import (SEED, linear_schedule, make_spiral, sample, save,
                           schedule_terms, score_field, setup_logging, train_eps_model)

log = setup_logging("spiral_figures")

# The real DDPM setup (Ho et al. 2020): T=1000 with beta linear from 1e-4 to 0.02.
# Shorter schedules leave sqrt(abar_T) far from 0, so x_T is not actually noise and the
# samplers inherit that error. Keeping T=1000 also means the slide's "1000 steps" is true.
T = 1000
N_DATA = 4000
LIM = 1.35
DATA_C = "#0033A0"
GEN_C = "#D90012"
ACC_C = "#F2A800"


def panel(ax, title):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def fig_data(x, betas, rng):
    _, abars = schedule_terms(betas)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    panel(axes[0], "The data: two pixel values, one point")
    axes[0].scatter(x[:, 0], x[:, 1], s=2, color=DATA_C, alpha=0.5)

    # A handful of forward trajectories, drawn step by step so the walk is visible.
    panel(axes[1], "Noising one point = a random walk")
    starts = x[rng.choice(len(x), 4, replace=False)]
    for s in starts:
        pos, traj = s.copy(), [s.copy()]
        for t in range(300):
            pos = np.sqrt(1 - betas[t]) * pos + np.sqrt(betas[t]) * rng.normal(size=2)
            traj.append(pos.copy())
        traj = np.array(traj)
        axes[1].plot(traj[:, 0], traj[:, 1], lw=1.0, color=ACC_C, alpha=0.9)
        axes[1].scatter(*traj[0], s=26, color=DATA_C, zorder=3)
        axes[1].scatter(*traj[-1], s=26, color=GEN_C, zorder=3)
    axes[1].scatter(x[:, 0], x[:, 1], s=1, color=DATA_C, alpha=0.12)

    panel(axes[2], "Do it to every point: the data dissolves")
    t_show = 600
    noisy = np.sqrt(abars[t_show]) * x + np.sqrt(1 - abars[t_show]) * rng.normal(size=x.shape)
    axes[2].scatter(noisy[:, 0], noisy[:, 1], s=2, color=GEN_C, alpha=0.4)
    axes[2].set_xlim(-2.6, 2.6)
    axes[2].set_ylim(-2.6, 2.6)
    axes[2].set_title(f"Do it to every point ($t={t_show}$)", fontsize=10)

    fig.tight_layout()
    save(fig, "spiral_data.pdf", log)


def fig_score(model, betas, x):
    ts = [900, 450, 50]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, t in zip(axes, ts):
        XX, YY, U, V = score_field(model, betas, t, lim=LIM, grid=17)
        mag = np.hypot(U, V)
        ax.quiver(XX, YY, U / mag, V / mag, mag, cmap="viridis",
                  scale=26, width=0.005, alpha=0.9)
        ax.scatter(x[:, 0], x[:, 1], s=1.2, color=DATA_C, alpha=0.22)
        panel(ax, f"$t={t}$  (noise level {'high' if t > 600 else 'medium' if t > 200 else 'low'})")
        log.info(f"score field t={t}: mean |score| = {mag.mean():.3f}")
    fig.suptitle("What the network learned: a direction at every point, sharpening as $t\\to 0$",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    save(fig, "spiral_score_field.pdf", log)


def fig_samplers(model, betas, x):
    n = 700
    ddpm = sample(model, betas, n=n, mode="ddpm", add_noise=True)
    naive = sample(model, betas, n=n, mode="ddpm", add_noise=False)
    ddim = sample(model, betas, n=n, mode="ddim", n_steps=25)

    for name, s in [("DDPM", ddpm), ("naive (no noise)", naive), ("DDIM-25", ddim)]:
        spread = float(np.mean(np.linalg.norm(s - s.mean(0), axis=1)))
        log.info(f"{name:18s} mean radius {np.linalg.norm(s, axis=1).mean():.3f} "
                 f" spread {spread:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, (s, title) in zip(axes, [
        (ddpm, "DDPM: keep the noise step\n1000 steps, stochastic"),
        (naive, "Naive: just delete the noise\ncollapses to the mean"),
        (ddim, "DDIM: rescale the step instead\n25 steps, deterministic"),
    ]):
        ax.scatter(x[:, 0], x[:, 1], s=1.2, color=DATA_C, alpha=0.18)
        ax.scatter(s[:, 0], s[:, 1], s=4, color=GEN_C, alpha=0.75)
        panel(ax, title)
    axes[1].scatter([0], [0], s=90, marker="x", color="black", zorder=5, lw=2)
    fig.tight_layout()
    save(fig, "spiral_samplers.pdf", log)


def main():
    rng = np.random.default_rng(SEED)
    betas = linear_schedule(T)
    x = make_spiral(N_DATA, rng=rng)
    log.info(f"spiral: {x.shape}, T={T}")

    fig_data(x, betas, rng)
    model, losses = train_eps_model(x, betas, steps=14000, log=log)
    log.info(f"final loss (mean of last 200 steps): {losses[-200:].mean():.4f}")
    fig_score(model, betas, x)
    fig_samplers(model, betas, x)


if __name__ == "__main__":
    main()
