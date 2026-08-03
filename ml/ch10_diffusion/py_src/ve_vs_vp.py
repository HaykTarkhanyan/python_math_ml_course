"""L27: why the naive forward process fails - variance exploding vs variance preserving.

VE: x_t = x_0 + sqrt(t beta) eps      -> mean coefficient stays 1, variance grows without bound
VP: x_t = sqrt(abar_t) x_0 + ...      -> mean coefficient -> 0, variance -> 1

The point of the figure is that only VP lands on a distribution we can actually sample from.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import save, setup_logging

log = setup_logging("ve_vs_vp")

BETA = 0.02
T = 400


def main():
    t = np.arange(T + 1)
    ve_mean = np.ones_like(t, dtype=float)
    ve_var = t * BETA
    vp_mean = np.sqrt((1 - BETA) ** t)
    vp_var = 1 - (1 - BETA) ** t

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    ax = axes[0]
    ax.plot(t, ve_mean, color="#D90012", lw=2, label="mean coefficient")
    ax.plot(t, ve_var, color="#0033A0", lw=2, label="variance")
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.set_title("Variance exploding: $x_t = x_0 + \\sqrt{t\\beta}\\,\\epsilon$", fontsize=10)
    ax.set_xlabel("step $t$", fontsize=9)
    ax.set_ylim(-0.3, 8.5)
    ax.text(200, 6.6, "variance runs away", color="#0033A0", fontsize=9)
    ax.text(200, 1.35, "mean never moves off $x_0$", color="#D90012", fontsize=9)

    ax = axes[1]
    ax.plot(t, vp_mean, color="#D90012", lw=2, label="mean coefficient $\\sqrt{\\bar\\alpha_t}$")
    ax.plot(t, vp_var, color="#0033A0", lw=2, label="variance $1-\\bar\\alpha_t$")
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_title("Variance preserving: $x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$",
                 fontsize=10)
    ax.set_xlabel("step $t$", fontsize=9)
    ax.set_ylim(-0.3, 8.5)
    ax.text(150, 2.2, "lands exactly on $\\mathcal{N}(0,\\mathbf{I})$", color="#008C46", fontsize=9)

    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="center right")
        ax.spines[["top", "right"]].set_visible(False)

    log.info(f"VE at t={T}: mean_coef={ve_mean[-1]:.3f} var={ve_var[-1]:.2f}")
    log.info(f"VP at t={T}: mean_coef={vp_mean[-1]:.5f} var={vp_var[-1]:.5f}")
    fig.tight_layout()
    save(fig, "ve_vs_vp.pdf", log)


if __name__ == "__main__":
    main()
