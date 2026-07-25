"""Train a tiny transformer until it memorizes, then (hopefully) groks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from model import ModularAdditionTransformer

ARTIFACTS = Path("artifacts")


def build_dataset(modulus: int, train_fraction: float, seed: int):
    """Make all p² equations, then split them once into train and test sets."""
    values = torch.arange(modulus)
    x, y = torch.meshgrid(values, values, indexing="ij")
    answers = ((x + y) % modulus).reshape(-1)
    equals = torch.full_like(answers, modulus)
    tokens = torch.stack((x.reshape(-1), y.reshape(-1), equals), dim=1)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(tokens), generator=generator)
    cutoff = int(train_fraction * len(tokens))
    train_ids, test_ids = permutation[:cutoff], permutation[cutoff:]
    return tokens[train_ids], answers[train_ids], tokens[test_ids], answers[test_ids]


@torch.no_grad()
def accuracy(model: nn.Module, tokens: torch.Tensor, answers: torch.Tensor) -> float:
    return (model(tokens).argmax(dim=-1) == answers).float().mean().item()


@torch.no_grad()
def cross_entropy(model: nn.Module, tokens: torch.Tensor, answers: torch.Tensor) -> float:
    """Loss on withheld equations; the progress metric used alongside accuracy."""
    return nn.functional.cross_entropy(model(tokens), answers).item()


def plot_metrics(history: list[dict[str, float]]) -> None:
    steps = [row["step"] for row in history]
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    loss_ax.plot(steps, [row["loss"] for row in history], color="#ef4444", label="batch loss")
    if "held_out_loss" in history[0]:
        loss_ax.plot(steps, [row["held_out_loss"] for row in history], color="#7c3aed", label="held-out loss")
    loss_ax.set_xlabel("optimizer steps")
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title("Training loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()
    accuracy_ax.plot(steps, [row["train_accuracy"] for row in history], label="training accuracy")
    accuracy_ax.plot(steps, [row["test_accuracy"] for row in history], label="test accuracy")
    accuracy_ax.set_xlabel("optimizer steps")
    accuracy_ax.set_ylabel("accuracy")
    accuracy_ax.set_ylim(-0.02, 1.02)
    accuracy_ax.set_title("Memorization, then possible grokking")
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "metrics.png", dpi=160)
    plt.close()


def save_artifacts(
    model: nn.Module, optimizer: torch.optim.Optimizer, modulus: int,
    history: list[dict[str, float]], config: dict, save_snapshot: bool = False,
) -> None:
    """Persist the latest curve and state so long runs remain inspectable."""
    plot_metrics(history)
    model_to_save = getattr(model, "_orig_mod", model)
    payload = {"model_state": model_to_save.state_dict(), "optimizer_state": optimizer.state_dict(), "modulus": modulus,
               "history": history, "config": config}
    torch.save(payload, ARTIFACTS / "checkpoint.pt")
    if save_snapshot:
        snapshots = ARTIFACTS / "snapshots"
        snapshots.mkdir(exist_ok=True)
        torch.save(payload, snapshots / f"checkpoint_{history[-1]['step']:06d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, default=67, help="p in x + y mod p")
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128, help="Embedding width; 32 is a faster demo")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; one is fastest for this tiny model")
    parser.add_argument("--device", default="cpu", help="Use cuda if it is available")
    parser.add_argument("--resume", action="store_true", help="Continue from artifacts/checkpoint.pt in this run directory")
    parser.add_argument("--paper-mode", action="store_true", help="Use Nanda et al.'s mainline P=113, full-batch, no-LayerNorm setup")
    parser.add_argument("--full-batch", action="store_true", help="Use every training equation once per optimizer step")
    parser.add_argument("--save-snapshots", action="store_true", help="Save an inspectable checkpoint at every evaluation")
    args = parser.parse_args()
    if args.paper_mode:
        args.modulus, args.train_fraction, args.steps = 113, 0.3, 40_000
        args.d_model, args.n_heads, args.d_mlp = 128, 4, 512
        args.learning_rate, args.weight_decay, args.full_batch = 1e-3, 1.0, True
        args.eval_every = 200
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        # Inter-op threads may already be fixed by an embedding application.
        pass
    device = torch.device(args.device)
    ARTIFACTS.mkdir(exist_ok=True)
    train_tokens, train_answers, test_tokens, test_answers = build_dataset(
        args.modulus, args.train_fraction, args.seed
    )
    train_tokens, train_answers = train_tokens.to(device), train_answers.to(device)
    test_tokens, test_answers = test_tokens.to(device), test_answers.to(device)

    model = ModularAdditionTransformer(
        args.modulus, d_model=args.d_model, n_heads=args.n_heads, d_mlp=args.d_mlp,
        use_layer_norm=not args.paper_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history: list[dict[str, float]] = []
    start_step = 0
    if args.resume:
        checkpoint = torch.load(ARTIFACTS / "checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print("Checkpoint has no optimizer state; continuing with fresh AdamW moments.")
        history = checkpoint["history"]
        start_step = history[-1]["step"]
        if args.paper_mode:
            # Paper mode's 40k is a total epoch budget, not an extra 40k.
            args.steps = max(0, 40_000 - start_step)
        print(f"Resuming from step {start_step} for {args.steps} additional updates.")

    target_step = start_step + args.steps
    for step in range(start_step + 1, target_step + 1):
        if args.full_batch:
            batch_tokens, batch_answers = train_tokens, train_answers
        else:
            batch_ids = torch.randint(len(train_tokens), (args.batch_size,), generator=generator, device=device)
            batch_tokens, batch_answers = train_tokens[batch_ids], train_answers[batch_ids]
        loss = loss_fn(model(batch_tokens), batch_answers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == start_step + 1 or step % args.eval_every == 0 or step == target_step:
            model.eval()
            row = {
                "step": step,
                "loss": loss.item(),
                "train_accuracy": accuracy(model, train_tokens, train_answers),
                "test_accuracy": accuracy(model, test_tokens, test_answers),
                "held_out_loss": cross_entropy(model, test_tokens, test_answers),
            }
            history.append(row)
            print(
                f"step {step:>6} | loss {row['loss']:.4f} | "
                f"train {row['train_accuracy']:.3f} | test {row['test_accuracy']:.3f} | "
                f"held-out loss {row['held_out_loss']:.4f}"
            )
            save_artifacts(model, optimizer, args.modulus, history, vars(args), args.save_snapshots)
            model.train()

    print(f"Saved {ARTIFACTS / 'metrics.png'} and {ARTIFACTS / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
