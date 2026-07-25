"""Paper-aligned Fourier analysis for the modular-addition transformer.

This implements the central measurements from Nanda et al. (2023): Fourier
spectra of the neuron-logit map and logits, degree-2 single-frequency neuron
fits, and restricted/excluded logit losses.  It is intended for checkpoints
trained with ``train.py --paper-mode --save-snapshots``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model import ModularAdditionTransformer
from train import build_dataset

ARTIFACTS = Path("artifacts")


def load_model(checkpoint: dict) -> ModularAdditionTransformer:
    config, modulus = checkpoint["config"], checkpoint["modulus"]
    model = ModularAdditionTransformer(
        modulus, config.get("d_model", 128), config.get("n_heads", 4), config.get("d_mlp", 512),
        use_layer_norm=not config.get("paper_mode", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def all_inputs(modulus: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(modulus)
    a, b = torch.meshgrid(values, values, indexing="ij")
    answers = ((a + b) % modulus).flatten()
    tokens = torch.stack((a.flatten(), b.flatten(), torch.full((modulus * modulus,), modulus)), dim=1)
    return a, b, tokens, answers


def key_frequencies(model: ModularAdditionTransformer) -> tuple[torch.Tensor, torch.Tensor]:
    layer = model.transformer.layers[0]
    # W_L maps each MLP neuron to each output logit, as in the paper.
    neuron_logit = model.unembed.weight @ layer.linear2.weight
    spectrum = torch.fft.rfft(neuron_logit, dim=0).abs().norm(dim=1)
    keys = torch.topk(spectrum[1:], k=min(5, len(spectrum) - 1)).indices + 1
    return keys, spectrum


def logit_frequency_projection(logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Keep only the key Fourier frequencies in the logits.

    This is an orthogonal projection onto the sin/cos waves of ``a+b`` for
    every output logit. It is the real-valued equivalent of the paper's 2D DFT
    frequency ablation.
    """
    modulus = logits.shape[-1]
    columns = [torch.ones_like(a, dtype=torch.float32).flatten()]
    for frequency in keys.tolist():
        angle = 2 * torch.pi * frequency * (a + b).float() / modulus
        columns.extend((torch.sin(angle).flatten(), torch.cos(angle).flatten()))
    basis = torch.stack(columns, dim=1)
    return basis @ torch.linalg.lstsq(basis, logits.reshape(-1, modulus)).solution


def degree_two_features(a: torch.Tensor, b: torch.Tensor, modulus: int, frequency: int) -> torch.Tensor:
    angle_a, angle_b = 2 * torch.pi * frequency * a.float() / modulus, 2 * torch.pi * frequency * b.float() / modulus
    linear = [torch.sin(angle_a), torch.cos(angle_a), torch.sin(angle_b), torch.cos(angle_b)]
    features = [torch.ones_like(a, dtype=torch.float32), *linear]
    features.extend(linear[i] * linear[j] for i in range(len(linear)) for j in range(i, len(linear)))
    return torch.stack([feature.flatten() for feature in features], dim=1)


def r_squared(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return 1 - (actual - predicted).square().sum(0) / (actual - actual.mean(0)).square().sum(0).clamp_min(1e-12)


def analyze_checkpoint(checkpoint: dict, output: Path) -> dict:
    model, modulus, config = load_model(checkpoint), checkpoint["modulus"], checkpoint["config"]
    a, b, tokens, answers = all_inputs(modulus)
    with torch.no_grad():
        logits, cache = model.forward_with_cache(tokens)
    logits_3d = logits.reshape(modulus, modulus, modulus)
    keys, wl_spectrum = key_frequencies(model)
    restricted = logit_frequency_projection(logits, a, b, keys)
    excluded = logits - restricted
    train_tokens, train_answers, test_tokens, test_answers = build_dataset(modulus, config["train_fraction"], config["seed"])
    train_ids = train_tokens[:, 0] * modulus + train_tokens[:, 1]
    test_ids = test_tokens[:, 0] * modulus + test_tokens[:, 1]
    measures = {
        "full_train_loss": float(F.cross_entropy(logits[train_ids], train_answers)),
        "full_test_loss": float(F.cross_entropy(logits[test_ids], test_answers)),
        "restricted_train_loss": float(F.cross_entropy(restricted[train_ids], train_answers)),
        "excluded_train_loss": float(F.cross_entropy(excluded[train_ids], train_answers)),
        "logit_fourier_fve": float(1 - (logits - restricted).square().sum() / (logits - logits.mean()).square().sum()),
        "key_frequencies": keys.tolist(),
    }
    hidden = cache["mlp_hidden"][:, -1]
    all_fits = []
    for frequency in keys.tolist():
        basis = degree_two_features(a, b, modulus, frequency)
        all_fits.append(r_squared(hidden, basis @ torch.linalg.lstsq(basis, hidden).solution))
    neuron_fits = torch.stack(all_fits)
    measures["best_degree2_neuron_r2"] = float(neuron_fits.max())
    measures["fraction_neurons_r2_over_085"] = float((neuron_fits.max(0).values > .85).float().mean())

    plt.figure(figsize=(8, 4.5))
    plt.bar(torch.arange(1, len(wl_spectrum)), wl_spectrum[1:].detach().numpy(), color="#3b82f6")
    plt.xlabel("Fourier frequency over output-logit axis")
    plt.ylabel("norm across MLP neurons")
    plt.title("Fourier components of neuron-logit map $W_L$")
    plt.tight_layout(); plt.savefig(output / "paper_neuron_logit_fourier.png", dpi=160); plt.close()
    logits_spectrum = torch.fft.fftshift(torch.fft.fft2(logits_3d, dim=(0, 1)).abs().norm(dim=2))
    plt.figure(figsize=(6, 5)); plt.imshow(logits_spectrum.numpy(), cmap="magma"); plt.colorbar(label="norm over logits")
    plt.title("2D Fourier components of logits"); plt.xlabel("frequency of b"); plt.ylabel("frequency of a")
    plt.tight_layout(); plt.savefig(output / "paper_logits_2d_fourier.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4.5)); plt.hist(neuron_fits.max(0).values.numpy(), bins=25, color="#8b5cf6")
    plt.xlabel("best degree-2 single-frequency R² per MLP neuron"); plt.ylabel("number of neurons")
    plt.title("Degree-2 Fourier fit of MLP neurons")
    plt.tight_layout(); plt.savefig(output / "paper_neuron_degree2_fits.png", dpi=160); plt.close()
    return measures


def save_progress_from_snapshots(final_checkpoint: dict) -> None:
    """Plot paper-style train/test/restricted/excluded loss over saved epochs.

    Key frequencies are selected from the final model, then held fixed for each
    earlier checkpoint, matching the paper's progress-measure procedure.
    """
    snapshot_paths = sorted((ARTIFACTS / "snapshots").glob("checkpoint_*.pt"))
    if not snapshot_paths:
        print("No snapshots found; rerun training with --save-snapshots for progress curves.")
        return
    final_model = load_model(final_checkpoint)
    keys, _ = key_frequencies(final_model)
    modulus, config = final_checkpoint["modulus"], final_checkpoint["config"]
    a, b, tokens, _ = all_inputs(modulus)
    train_tokens, train_answers, test_tokens, test_answers = build_dataset(modulus, config["train_fraction"], config["seed"])
    train_ids, test_ids = train_tokens[:, 0] * modulus + train_tokens[:, 1], test_tokens[:, 0] * modulus + test_tokens[:, 1]
    rows = []
    for path in snapshot_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = load_model(checkpoint)
        with torch.no_grad():
            logits = model(tokens)
        restricted = logit_frequency_projection(logits, a, b, keys)
        excluded = logits - restricted
        rows.append({
            "step": checkpoint["history"][-1]["step"],
            "train_loss": float(F.cross_entropy(logits[train_ids], train_answers)),
            "test_loss": float(F.cross_entropy(logits[test_ids], test_answers)),
            "restricted_loss": float(F.cross_entropy(restricted[train_ids], train_answers)),
            "excluded_loss": float(F.cross_entropy(excluded[train_ids], train_answers)),
        })
    plt.figure(figsize=(9, 5))
    for name, label in (("train_loss", "train"), ("test_loss", "test"), ("restricted_loss", "restricted"), ("excluded_loss", "excluded")):
        plt.semilogy([row["step"] for row in rows], [row[name] for row in rows], label=label)
    plt.xlabel("full-batch training epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("Paper-style Fourier progress measures")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(ARTIFACTS / "paper_progress_measures.png", dpi=160); plt.close()
    (ARTIFACTS / "paper_progress_measures.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "checkpoint.pt")
    parser.add_argument("--progress", action="store_true", help="Compute restricted/excluded-loss curves from artifacts/snapshots")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = analyze_checkpoint(checkpoint, ARTIFACTS)
    (ARTIFACTS / "paper_mechanistic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.progress:
        save_progress_from_snapshots(checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
