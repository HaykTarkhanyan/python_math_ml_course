"""L28: why a KL divergence between two Gaussians of equal, fixed variance is just a
squared distance between their means.

Left panel  - the KL between two distributions, as an overlap picture.
Right panel - KL against mean separation, with the quadratic overlaid, so the students can
see the two curves coincide once the variance is held fixed.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import save, setup_logging

log = setup_logging("kl_gaussians")

SIGMA = 1.0


def main():
    grid = np.linspace(-5, 7, 700)

    def pdf(x, mu, s=SIGMA):
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))

    ax = axes[0]
    for mu, col, lab in [(0.0, "#0033A0", r"$q(x_{t-1}\mid x_t, x_0)$  (true posterior)"),
                         (2.2, "#D90012", r"$p_\theta(x_{t-1}\mid x_t)$  (our model)")]:
        ax.fill_between(grid, pdf(grid, mu), color=col, alpha=0.25)
        ax.plot(grid, pdf(grid, mu), color=col, lw=2, label=lab)
        ax.axvline(mu, color=col, ls=":", lw=1.2)
    ax.annotate("", xy=(2.2, 0.44), xytext=(0.0, 0.44),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.6, color="#008C46"))
    ax.text(1.1, 0.46, r"$\tilde\mu_t - \mu_\theta$", ha="center",
            fontsize=11, color="#008C46")
    ax.set_ylim(0, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("$x_{t-1}$", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Same fixed width, different centre", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = axes[1]
    d = np.linspace(0, 3.2, 200)
    kl = d**2 / (2 * SIGMA**2)
    ax.plot(d, kl, color="#0033A0", lw=2.5, label=r"$D_{\mathrm{KL}}$ (computed)")
    ax.plot(d, d**2 / (2 * SIGMA**2), color="#F2A800", lw=1.6, ls="--",
            label=r"$\|\tilde\mu_t-\mu_\theta\|^2 / 2\sigma_t^2$")
    ax.set_xlabel(r"distance between means  $\|\tilde\mu_t - \mu_\theta\|$", fontsize=9)
    ax.set_ylabel("divergence", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("They are the same curve", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    log.info(f"KL at separation 2.2 = {2.2**2 / (2 * SIGMA**2):.4f}")

    fig.tight_layout()
    save(fig, "kl_gaussians.pdf", log)


if __name__ == "__main__":
    main()
