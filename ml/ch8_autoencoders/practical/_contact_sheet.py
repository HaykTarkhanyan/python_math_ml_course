"""Lay the 9 ch8 figures onto one contact sheet for quick review."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SCRATCH = Path(r"C:\Users\hayk_\AppData\Local\Temp\claude\C--Users-hayk--OneDrive-Desktop-01-python-math-ml-course\357b309a-48ba-4887-b868-e1fe5405b06b\scratchpad")
OUT = SCRATCH / "ch8_contact_sheet.png"

panels = [
    ("ae_recon_teaser", "1. Recon teaser (784->2->784)"),
    ("recon_pca_vs_ae", "2. Recon from 2 numbers: PCA vs AE"),
    ("pca_vs_ae_mnist", "3. Latent scatter: PCA vs AE"),
    ("denoising_grid", "4. Denoising"),
    ("anomaly_recon_error_hist", "5. Anomaly: threshold + why"),
    ("ae_sampling_fail", "6. AE sampling-fail (cliffhanger)"),
    ("vae_latent_grid", "7. VAE latent grid (payoff)"),
    ("vae_samples", "8. VAE samples z~N(0,I)"),
    ("interp_ae_vs_vae", "9. Interpolate 3->8: AE vs VAE"),
    ("sampling_ae_vs_vae", "10. Sample N(0,I): AE vs VAE"),
]

fig, axs = plt.subplots(4, 3, figsize=(22, 18))
for ax, (name, title) in zip(axs.flat, panels):
    ax.imshow(mpimg.imread(SCRATCH / f"{name}.png"))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
for ax in axs.flat[len(panels):]:
    ax.axis("off")
fig.suptitle("ch8 Autoencoders + VAE - all 10 MNIST figures (v2, improved)", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved {OUT}")
