"""Mechanistic diagnostics for modular-addition grokking.

The script looks for Fourier features in the number embeddings and asks whether
the answer-position residual stream is linearly explained by sin/cos waves of
the correct modular sum.  It does not claim a full circuit reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import ModularAdditionTransformer

ARTIFACTS = Path("artifacts")


def harmonic_basis(values: torch.Tensor, modulus: int, max_frequency: int | None = None) -> tuple[torch.Tensor, list[str]]:
    """Constant, sine, and cosine columns evaluated at modular values."""
    max_frequency = max_frequency or (modulus - 1) // 2
    columns, names = [torch.ones_like(values, dtype=torch.float32)], ["constant"]
    for frequency in range(1, max_frequency + 1):
        angle = 2 * torch.pi * frequency * values.float() / modulus
        columns.extend((torch.sin(angle), torch.cos(angle)))
        names.extend((f"sin({frequency})", f"cos({frequency})"))
    return torch.stack(columns, dim=1), names


def r_squared(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    residual = (actual - predicted).square().sum(dim=0)
    total = (actual - actual.mean(dim=0)).square().sum(dim=0).clamp_min(1e-12)
    return 1 - residual / total


def save_embedding_diagnostics(embeddings: torch.Tensor, modulus: int) -> dict:
    values = torch.arange(modulus)
    max_frequency = (modulus - 1) // 2
    frequency_fits, frequency_predictions = [], []
    for frequency in range(1, max_frequency + 1):
        angle = 2 * torch.pi * frequency * values.float() / modulus
        basis = torch.stack((torch.ones_like(values, dtype=torch.float32), torch.sin(angle), torch.cos(angle)), dim=1)
        fitted = basis @ torch.linalg.lstsq(basis, embeddings).solution
        frequency_fits.append(r_squared(embeddings, fitted))
        frequency_predictions.append(fitted)
    frequency_fits = torch.stack(frequency_fits)
    fits, best_frequencies = frequency_fits.max(dim=0)
    spectrum = torch.fft.rfft(embeddings, dim=0).abs().mean(dim=1)
    frequencies = torch.arange(len(spectrum))

    top = torch.topk(spectrum[1:], k=min(8, len(spectrum) - 1))
    print("Strongest non-constant embedding frequencies:")
    for magnitude, offset in zip(top.values, top.indices):
        print(f"  frequency {offset.item() + 1:>2}: mean magnitude {magnitude.item():.3f}")
    print(f"Best single-frequency embedding fit: R²={fits.max().item():.3f}")

    plt.figure(figsize=(8, 4.5))
    plt.bar(frequencies[1:].numpy(), spectrum[1:].numpy(), color="#3b82f6")
    plt.xlabel("discrete Fourier frequency")
    plt.ylabel("mean magnitude across embedding dimensions")
    plt.title("Fourier spectrum of learned number-token embeddings")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "embedding_fourier_spectrum.png", dpi=160)
    plt.close()

    dimensions = torch.topk(fits, k=min(6, embeddings.shape[1])).indices.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for axis, dimension in zip(axes.flat, dimensions):
        frequency = int(best_frequencies[dimension].item())
        fitted = frequency_predictions[frequency - 1]
        axis.plot(values, embeddings[:, dimension], "o", ms=3, label="embedding")
        axis.plot(values, fitted[:, dimension], lw=2, label="sin/cos fit")
        axis.set_title(f"dimension {dimension}, freq {frequency}, R²={fits[dimension]:.2f}")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("number token")
    fig.supylabel("feature value")
    fig.suptitle("Embedding dimensions fitted by Fourier waves")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "embedding_sine_cosine_fits.png", dpi=160)
    plt.close(fig)
    return {"top_embedding_frequency": int(top.indices[0].item() + 1), "best_embedding_single_frequency_r2": float(fits.max())}


def save_answer_stream_diagnostics(model: ModularAdditionTransformer, modulus: int) -> dict:
    values = torch.arange(modulus)
    x, y = torch.meshgrid(values, values, indexing="ij")
    tokens = torch.stack((x.flatten(), y.flatten(), torch.full((modulus * modulus,), modulus)), dim=1)
    with torch.no_grad():
        _, cache = model.forward_with_cache(tokens)
    answer_residual = cache["post_layer"][:, -1]
    sums = ((x + y) % modulus).flatten()
    basis, names = harmonic_basis(sums, modulus)
    fit = basis @ torch.linalg.lstsq(basis, answer_residual).solution
    fit_r2 = r_squared(answer_residual, fit)
    print(f"Answer-residual sin/cos probe: mean R²={fit_r2.mean().item():.3f}, best feature R²={fit_r2.max().item():.3f}")

    surfaces = answer_residual.reshape(modulus, modulus, -1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(surfaces, dim=(0, 1)).abs().mean(dim=2))
    plt.figure(figsize=(6, 5))
    plt.imshow(spectrum.numpy(), cmap="magma")
    plt.colorbar(label="mean Fourier magnitude")
    plt.xlabel("frequency of y")
    plt.ylabel("frequency of x")
    plt.title("2D Fourier spectrum of answer-position residual stream")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "answer_residual_2d_fourier.png", dpi=160)
    plt.close()

    best = int(fit_r2.argmax())
    plt.figure(figsize=(6, 5))
    plt.scatter(answer_residual[:, best], fit[:, best], s=10, alpha=0.7)
    limits = torch.stack((answer_residual[:, best], fit[:, best])).flatten()
    low, high = limits.min().item(), limits.max().item()
    plt.plot((low, high), (low, high), "--", color="black", lw=1)
    plt.xlabel("actual answer-residual feature")
    plt.ylabel("sin/cos probe prediction")
    plt.title(f"Best answer-stream harmonic probe (feature {best}, R²={fit_r2[best]:.3f})")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "answer_residual_harmonic_probe.png", dpi=160)
    plt.close()
    return {"answer_residual_mean_harmonic_r2": float(fit_r2.mean()), "answer_residual_best_harmonic_r2": float(fit_r2.max()), "best_answer_residual_feature": best, "harmonic_columns": names}


def save_ablation_and_neuron_diagnostics(model: ModularAdditionTransformer, modulus: int) -> dict:
    """Measure component necessity and render the most structured MLP neurons."""
    values = torch.arange(modulus)
    x, y = torch.meshgrid(values, values, indexing="ij")
    answers = ((x + y) % modulus).flatten()
    tokens = torch.stack((x.flatten(), y.flatten(), torch.full((modulus * modulus,), modulus)), dim=1)
    with torch.no_grad():
        full_logits, cache = model.forward_with_cache(tokens)
        attention_ablated = model(tokens, ablate="attention")
        mlp_ablated = model(tokens, ablate="mlp")

    def metrics(logits: torch.Tensor) -> tuple[float, float]:
        return ((logits.argmax(dim=-1) == answers).float().mean().item(), torch.nn.functional.cross_entropy(logits, answers).item())

    ablations = {"full": metrics(full_logits), "attention_ablated": metrics(attention_ablated), "mlp_ablated": metrics(mlp_ablated)}
    print("Full / attention-ablated / MLP-ablated accuracy:", *(f"{ablations[key][0]:.3f}" for key in ablations))
    labels = list(ablations)
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, [ablations[label][0] for label in labels], color=["#2563eb", "#f97316", "#dc2626"])
    plt.ylim(0, 1.05)
    plt.ylabel("accuracy over all modular-addition equations")
    plt.title("Targeted component ablations")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "component_ablations.png", dpi=160)
    plt.close()

    hidden = cache["mlp_hidden"][:, -1].reshape(modulus, modulus, -1)
    neuron_ids = torch.topk(hidden.var(dim=(0, 1)), k=min(9, hidden.shape[-1])).indices.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)
    for axis, neuron in zip(axes.flat, neuron_ids):
        image = axis.imshow(hidden[:, :, neuron].numpy(), cmap="coolwarm", origin="lower")
        axis.set_title(f"MLP neuron {neuron}")
        axis.set_xlabel("y")
        axis.set_ylabel("x")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Most variable MLP-neuron output surfaces")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "mlp_neuron_surfaces.png", dpi=160)
    plt.close(fig)
    return {f"{name}_accuracy": accuracy for name, (accuracy, _) in ablations.items()} | {f"{name}_loss": loss for name, (_, loss) in ablations.items()}


def main() -> None:
    checkpoint = torch.load(ARTIFACTS / "checkpoint.pt", map_location="cpu", weights_only=False)
    modulus, config = checkpoint["modulus"], checkpoint["config"]
    model = ModularAdditionTransformer(modulus, config.get("d_model", 128), config.get("n_heads", 4), config.get("d_mlp", 512))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    summary = {
        "modulus": modulus,
        **save_embedding_diagnostics(model.token_embedding.weight[:modulus].detach(), modulus),
        **save_answer_stream_diagnostics(model, modulus),
        **save_ablation_and_neuron_diagnostics(model, modulus),
    }
    (ARTIFACTS / "mechanistic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved Fourier, probes, neuron surfaces, and component ablations in artifacts/.")


if __name__ == "__main__":
    main()
