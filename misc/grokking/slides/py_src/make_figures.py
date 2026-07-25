"""Generate native explanatory figures for the grokking slide deck."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "fig"
OUT.mkdir(exist_ok=True)


def save_fourier_intro() -> None:
    x = np.linspace(0, 2 * np.pi, 600)
    wave_1 = np.sin(x)
    wave_3 = 0.62 * np.sin(3 * x + 0.4)
    signal = wave_1 + wave_3
    fig, (ax_signal, ax_spectrum) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax_signal.plot(x, signal, color="#222222", lw=3, label="combined signal")
    ax_signal.plot(x, wave_1, color="#3465a4", lw=2, alpha=.9, label="frequency 1")
    ax_signal.plot(x, wave_3, color="#d1495b", lw=2, alpha=.9, label="frequency 3")
    ax_signal.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax_signal.set_xlabel("position around a circle")
    ax_signal.set_ylabel("value")
    ax_signal.set_title("A signal can be a sum of simple waves")
    ax_signal.grid(alpha=.2)
    ax_signal.legend(frameon=False, loc="upper right")
    frequencies = np.arange(0, 7)
    magnitudes = np.array([0, 1.0, 0, .62, 0, 0, 0])
    colors = ["#b9c7dc"] * len(frequencies)
    colors[1], colors[3] = "#3465a4", "#d1495b"
    ax_spectrum.bar(frequencies, magnitudes, color=colors)
    ax_spectrum.set_ylim(0, 1.15)
    ax_spectrum.set_xlabel("frequency")
    ax_spectrum.set_ylabel("magnitude")
    ax_spectrum.set_title("Fourier transform: which waves are present?")
    ax_spectrum.set_xticks(frequencies)
    ax_spectrum.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "fourier_intro.png", dpi=180, transparent=False)
    plt.close(fig)


def save_fourier_token_bridge() -> None:
    """Show exactly what the DFT receives from a token embedding coordinate."""
    modulus = 7
    residues = np.arange(modulus)
    values = np.cos(2 * np.pi * residues / modulus)
    spectrum = np.abs(np.fft.fft(values))
    spectrum /= spectrum.max()

    fig, (ax_values, ax_spectrum) = plt.subplots(1, 2, figsize=(12, 4.2))
    markerline, stemlines, baseline = ax_values.stem(residues, values, basefmt=" ")
    plt.setp(markerline, color="#3465a4", markersize=8)
    plt.setp(stemlines, color="#3465a4", linewidth=2.5)
    ax_values.plot(residues, values, color="#3465a4", alpha=.45, lw=2)
    ax_values.set_xticks(residues)
    ax_values.set_ylim(-1.25, 1.25)
    ax_values.set_xlabel(r"residue $n$")
    ax_values.set_ylabel(r"one embedding coordinate $e_j(n)$")
    ax_values.set_title(r"Raw view: one number for each token $n$")
    ax_values.grid(alpha=.2)

    colors = ["#c7c7c7"] * modulus
    colors[1] = "#d1495b"
    colors[-1] = "#d1495b"
    ax_spectrum.bar(residues, spectrum, color=colors)
    ax_spectrum.set_xticks(residues)
    ax_spectrum.set_ylim(0, 1.15)
    ax_spectrum.set_xlabel(r"DFT frequency $k$")
    ax_spectrum.set_ylabel(r"normalized magnitude $|\hat e_j(k)|$")
    ax_spectrum.set_title(r"Fourier view: this coordinate is periodic")
    ax_spectrum.annotate("sine/cosine pair", xy=(1, 1.0), xytext=(2.4, .88),
                         arrowprops={"arrowstyle": "->", "color": "#333333"},
                         fontsize=11)
    ax_spectrum.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "fourier_token_bridge.png", dpi=180)
    plt.close(fig)


def rounded_box(ax, xy, text, color, width=2.6, height=1.0):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.05,rounding_size=0.12", ec=color, fc=color + "20", lw=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=13, weight="bold")
    return xy[0] + width, xy[1] + height / 2


def save_attention_mechanics() -> None:
    fig = plt.figure(figsize=(10.5, 4.5))
    ax = fig.add_axes([.015, .10, .69, .84])
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 5.7); ax.axis("off")
    right = rounded_box(ax, (.25, 3.55), "token vectors\n$2, 3, =$", "#3465a4", 2.75, 1.15)
    q = rounded_box(ax, (3.55, 4.25), "queries\n$Q=XW_Q$", "#8b5cf6", 2.25, 1.0)
    k = rounded_box(ax, (3.55, 2.82), "keys\n$K=XW_K$", "#d97706", 2.25, 1.0)
    v = rounded_box(ax, (3.55, 1.39), "values\n$V=XW_V$", "#059669", 2.25, 1.0)
    for target in [q, k, v]:
        ax.add_patch(FancyArrowPatch(right, (3.45, target[1]), arrowstyle="->", mutation_scale=16, lw=1.8, color="#333333", zorder=0))
    score = rounded_box(ax, (6.45, 3.22), "attention\nscores\n$QK^{T}/\sqrt{d_h}$", "#d1495b", 2.45, 1.35)
    weighted = rounded_box(ax, (9.32, 3.22), "weighted\nvalues\n$\mathrm{softmax}(\cdot)V$", "#3465a4", 2.45, 1.35)
    ax.add_patch(FancyArrowPatch(q, (6.40, 4.20), arrowstyle="->", mutation_scale=16, lw=1.8, color="#333333", zorder=0))
    ax.add_patch(FancyArrowPatch(k, (6.40, 3.55), arrowstyle="->", mutation_scale=16, lw=1.8, color="#333333", zorder=0))
    ax.add_patch(FancyArrowPatch((8.94, 3.895), (9.28, 3.895), arrowstyle="->", mutation_scale=16, lw=1.8, color="#333333", zorder=0))
    ax.add_patch(FancyArrowPatch(v, (9.28, 3.45), arrowstyle="->", mutation_scale=16, lw=1.8, color="#333333", zorder=0))
    ax.text(6.0, .48, "At the '=' position, one head chooses which token information\nshould be written into the residual stream.", ha="center", va="center", fontsize=13, color="#333333")
    matrix = fig.add_axes([.77, .22, .18, .59])
    weights = np.array([[.55, .30, .15], [.22, .58, .20], [.43, .43, .14]])
    image = matrix.imshow(weights, cmap="Blues", vmin=0, vmax=.6)
    matrix.set_xticks(range(3), ["2", "3", "="])
    matrix.set_yticks(range(3), ["2", "3", "="])
    matrix.set_xlabel("key/value token")
    matrix.set_ylabel("query token")
    matrix.set_title("one attention head")
    for row in range(3):
        for col in range(3):
            matrix.text(col, row, f"{weights[row,col]:.2f}", ha="center", va="center", fontsize=11)
    fig.colorbar(image, ax=matrix, shrink=.78, label="attention weight")
    fig.savefig(OUT / "attention_mechanics.png", dpi=180)
    plt.close(fig)


def save_mlp_mechanics() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.8); ax.axis("off")
    stages = [
        ("periodic features", r"$[\cos\theta_a,\sin\theta_a,\cos\theta_b,\sin\theta_b]$", "#3465a4"),
        ("MLP hidden units", r"$\mathrm{ReLU}(W_1x+b_1)$", "#059669"),
        ("products of waves", r"$\cos\theta_a\cos\theta_b,\;\sin\theta_a\sin\theta_b$", "#d97706"),
        ("logit for $c$", r"$\cos(\theta_a+\theta_b-\theta_c)$", "#d1495b"),
    ]
    x_positions = [0.35, 3.25, 6.15, 9.05]
    for (title, formula, color), x in zip(stages, x_positions):
        patch = FancyBboxPatch((x, 1.35), 2.55, 2.0, boxstyle="round,pad=.08,rounding_size=.12", ec=color, fc=color + "18", lw=2.2)
        ax.add_patch(patch)
        ax.text(x + 1.275, 2.82, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
        ax.text(x + 1.275, 2.0, formula, ha="center", va="center", fontsize=12)
    for x in [2.93, 5.83, 8.73]:
        ax.add_patch(FancyArrowPatch((x, 2.35), (x + .28, 2.35), arrowstyle="->", mutation_scale=18, lw=2, color="#333333"))
    ax.text(6, .48, r"The MLP is not ``adding'' symbols directly. It builds nonlinear products that implement the trig addition identity.", ha="center", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "mlp_trig_mechanics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    save_fourier_intro()
    save_fourier_token_bridge()
    save_attention_mechanics()
    save_mlp_mechanics()
