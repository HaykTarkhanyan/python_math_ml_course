"""L27: a bimodal data distribution flattening into N(0,1) under the VP forward process.

Fully analytic - no sampling. Under x_t = sqrt(abar) x_0 + sqrt(1-abar) eps, a Gaussian
component N(mu, s^2) maps to N(sqrt(abar) mu, abar s^2 + (1-abar)), so the whole mixture
stays a mixture and we can draw the exact density at any t.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import SEED, cosine_schedule, save, schedule_terms, setup_logging

log = setup_logging("forward_1d")

# A clearly bimodal "data distribution": two bumps of unequal mass and width.
COMPONENTS = [(-1.6, 0.28, 0.4), (1.1, 0.45, 0.6)]  # (mu, sigma, weight)
T = 200


def mixture_density(grid, abar):
    d = np.zeros_like(grid)
    for mu, sigma, w in COMPONENTS:
        m = np.sqrt(abar) * mu
        v = abar * sigma**2 + (1.0 - abar)
        d += w * np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    return d


def main():
    rng = np.random.default_rng(SEED)  # noqa: F841 - kept so the seed is on record
    _, abars = schedule_terms(cosine_schedule(T))
    grid = np.linspace(-4.5, 4.5, 600)
    shown = [0, 20, 60, 199]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.5), sharey=True)
    for ax, t in zip(axes, shown):
        abar = abars[t]
        ax.fill_between(grid, mixture_density(grid, abar), color="#0033A0", alpha=0.30)
        ax.plot(grid, mixture_density(grid, abar), color="#0033A0", lw=1.8)
        ax.plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
                color="#D90012", lw=1.2, ls="--")
        ax.set_title(rf"$t={t}$,  $\bar\alpha_t={abar:.3f}$", fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("$x$", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        log.info(f"t={t:3d} abar={abar:.4f}")

    axes[0].set_ylabel("density", fontsize=9)
    axes[-1].legend(["data at $t$", "$\\mathcal{N}(0,1)$"], fontsize=8,
                    frameon=False, loc="upper right")
    fig.suptitle("The forward process erases structure: any $p(x)$ becomes $\\mathcal{N}(0,1)$",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    save(fig, "forward_1d.pdf", log)


if __name__ == "__main__":
    main()
