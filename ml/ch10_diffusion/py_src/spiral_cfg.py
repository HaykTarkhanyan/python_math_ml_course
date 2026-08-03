"""L30: classifier-free guidance on the spiral.

One model is trained class-conditionally with the label dropped 15% of the time, so the
same weights give both eps_theta(x,t,c) and eps_theta(x,t). Guidance then pushes along
their difference:

    eps~ = eps_theta(x,t) + alpha * ( eps_theta(x,t,c) - eps_theta(x,t) )

alpha = 1 is exactly the plain conditional model, i.e. no guidance. This is the diffusers
convention and the one written on the slide.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import (SEED, linear_schedule, make_spiral, sample, save,
                           score_field, setup_logging, spiral_class_labels, train_eps_model)

log = setup_logging("spiral_cfg")

T = 1000
N_DATA = 4000
LIM = 1.35
CLASS_NAMES = ["person (inner)", "dog (middle)", "cat (outer)"]
CLASS_COLS = ["#0033A0", "#F2A800", "#D90012"]
ALPHAS = [1.0, 3.0, 7.5]


def panel(ax, title):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)


def fig_classes(x, y):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for c in range(3):
        m = y == c
        ax.scatter(x[m, 0], x[m, 1], s=3, color=CLASS_COLS[c], alpha=0.65,
                   label=CLASS_NAMES[c])
    panel(ax, "Label the spiral by region")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "spiral_classes.pdf", log)


def fig_fields(model, betas):
    """Unconditional vs conditional field, and their difference - the guidance direction."""
    t = 300
    XX, YY, Uu, Vu = score_field(model, betas, t, lim=LIM, grid=15, y=None, n_classes=3)
    XX, YY, Uc, Vc = score_field(model, betas, t, lim=LIM, grid=15, y=2, n_classes=3)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (U, V, title, col) in zip(axes, [
        (Uu, Vu, "No class: $f(x,t)$\npoints at the data in general", "gray"),
        (Uc, Vc, "Cat class: $f(x,t,\\mathrm{cat})$\npoints at the cat region", "#D90012"),
        (Uc - Uu, Vc - Vu, "The difference\n$f(x,t,\\mathrm{cat}) - f(x,t)$", "#008C46"),
    ]):
        mag = np.hypot(U, V) + 1e-9
        ax.quiver(XX, YY, U / mag, V / mag, scale=22, width=0.005, color=col, alpha=0.9)
        panel(ax, title)
    fig.suptitle("Guidance amplifies what the class adds, after removing what is generic",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "spiral_cfg_fields.pdf", log)


def fig_guidance_sweep(model, betas, x):
    n = 500
    y = np.full(n, 2)  # the cat class
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(11, 3.8))
    for ax, a in zip(axes, ALPHAS):
        s = sample(model, betas, n=n, mode="ddim", n_steps=50,
                   y=y, guidance=a, n_classes=3)
        ax.scatter(x[:, 0], x[:, 1], s=1.2, color="#999999", alpha=0.25)
        ax.scatter(s[:, 0], s[:, 1], s=5, color="#D90012", alpha=0.8)
        r = np.linalg.norm(s, axis=1)
        panel(ax, f"$\\alpha={a}$" + ("  (no guidance)" if a == 1.0 else ""))
        log.info(f"alpha={a}: mean radius {r.mean():.3f}  sd {r.std():.3f}")
    # Honest title: on a toy this separable, conditioning alone already works. What the
    # sweep actually shows is the COST curve - moderate guidance sharpens the fit, and
    # too much throws samples off the data entirely (measured: mean radius 0.91 -> 1.10,
    # sd 0.07 -> 0.17 going from alpha=1 to alpha=7.5).
    fig.suptitle("Cat class only: moderate $\\alpha$ sharpens the fit, "
                 "too much $\\alpha$ leaves the data behind", fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "spiral_cfg_sweep.pdf", log)


def main():
    rng = np.random.default_rng(SEED)
    betas = linear_schedule(T)
    x = make_spiral(N_DATA, rng=rng)
    y = spiral_class_labels(x)
    log.info(f"class counts: {np.bincount(y)}")

    fig_classes(x, y)
    model, losses = train_eps_model(x, betas, y_data=y, n_classes=3,
                                    steps=14000, drop_prob=0.15, log=log)
    log.info(f"final loss: {losses[-200:].mean():.4f}")
    fig_fields(model, betas)
    fig_guidance_sweep(model, betas, x)


if __name__ == "__main__":
    main()
