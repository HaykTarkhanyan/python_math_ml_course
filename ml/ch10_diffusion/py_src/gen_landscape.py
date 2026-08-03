"""L31: the generative landscape as a trade-off plot, not another comparison table.

L23_vae.tex:454 already ships a VAE/GAN/diffusion table. Repeating it wastes a frame, so
this positions the families on the axis that actually explains the field's history:
sample quality against how many network evaluations one sample costs.

Positions are qualitative and deliberately labelled as such on the slide.
"""

import matplotlib.pyplot as plt

from diffusion_lib import save, setup_logging

log = setup_logging("gen_landscape")

# (name, network evals per sample, qualitative sample quality 0-10, colour, note)
FAMILIES = [
    ("VAE",              1, 4.0, "#0033A0", "one pass, blurry"),
    ("GAN",              1, 7.8, "#F2A800", "one pass, sharp,\nhard to train"),
    ("Diffusion (DDPM)", 1000, 9.3, "#D90012", "1000 passes"),
    ("Diffusion (DDIM)", 25, 8.9, "#D90012", "25 passes,\nsame model"),
    ("Flow matching",    6, 9.0, "#7832A0", "straighter paths,\nfewer steps"),
]


def main():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    for name, evals, quality, colour, note in FAMILIES:
        ax.scatter(evals, quality, s=190, color=colour, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}\n{note}", (evals, quality),
                    textcoords="offset points", xytext=(0, -46),
                    ha="center", fontsize=8.5, color="#333333")
        log.info(f"{name:20s} evals={evals:5d} quality={quality}")

    # The arrow that tells the story: DDIM moved diffusion left at almost no cost.
    ax.annotate("", xy=(25, 8.9), xytext=(1000, 9.3),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#008C46",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(150, 9.75, "DDIM: 40x fewer steps,\nno retraining", fontsize=9,
            color="#008C46", ha="center", weight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.55, 3000)
    ax.set_ylim(2.2, 10.6)
    ax.set_xlabel("Network evaluations per sample (log scale)  ->  slower", fontsize=10)
    ax.set_ylabel("Sample quality (qualitative)", fontsize=10)
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.set_title("The trade the field made: quality bought with compute", fontsize=12)

    fig.tight_layout()
    save(fig, "gen_landscape.pdf", log)


if __name__ == "__main__":
    main()
