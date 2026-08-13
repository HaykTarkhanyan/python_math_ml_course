"""Figures for L45 - Opening the box.

Four figures, one model load:

* ``logit_lens.pdf``      - the prediction being built layer by layer, with the jump at layer 9;
* ``probe_by_layer.pdf``  - probe accuracy for "is a name repeated?", against a chance baseline;
* ``attention_heads.pdf`` - what a few real heads look at on the running example;
* ``induction_scores.pdf``- induction score for all 144 heads on repeated random tokens.

The logit-lens figure is the important one. It shows the logit difference climbing from ~0 to
+3.6 with a visible step at layer 9 - which is exactly where L46 will find the name-mover heads.
The two decks agree on where the work happens without either asserting it.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ioi_core import (
    DATA_PATH,
    FIG_DIR,
    SEED,
    load_model,
    load_prompts,
    logit_diff,
    logit_diff_directions,
    save_results,
    setup_logging,
)

# Armenian flag palette, per the repo's charting rule.
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

N_PROBE = 128
N_INDUCTION_SEQ = 8
INDUCTION_SEQ_LEN = 40


def fig_logit_lens(model, data, directions, log) -> dict:
    """Logit difference as a function of how many layers we let run."""
    _, cache = model.run_with_cache(data["clean"])

    accumulated, labels = cache.accumulated_resid(
        layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
    )
    scaled = cache.apply_ln_to_stack(accumulated, layer=-1, pos_slice=-1)
    per_layer = torch.einsum("cbd,bd->c", scaled, directions) / directions.shape[0]
    values = per_layer.tolist()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(values))
    ax.plot(x, values, marker="o", color=BLUE, lw=2, zorder=3)
    ax.axhline(0, color="0.6", lw=1)

    jump = int(np.argmax(np.diff(values))) + 1
    ax.annotate(
        f"biggest single jump\nat layer {jump - 1}",
        xy=(jump, values[jump]),
        xytext=(jump - 4.2, values[jump] + 0.9),
        color=RED,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("logit difference\n(correct name - wrong name)")
    ax.set_xlabel("residual stream, after this many layers")
    ax.set_title("The answer is built gradually, then decided in one place")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logit_lens.pdf")
    plt.close(fig)
    log.info(f"logit lens: final {values[-1]:+.3f}, biggest jump entering layer {jump - 1}")
    return {"labels": labels, "values": values, "biggest_jump_layer": jump - 1}


def fig_probe_by_layer(model, log) -> dict:
    """Can a linear probe read 'is a name repeated?' off the residual stream?"""
    probe_samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))["probe"]

    by_length: dict[int, list[dict]] = {}
    for s in probe_samples:
        by_length.setdefault(len(model.to_tokens(s["text"])[0]), []).append(s)
    best_len = max(by_length, key=lambda k: len(by_length[k]))
    group = by_length[best_len][:N_PROBE]

    labels = np.array([s["has_duplicate"] for s in group], dtype=int)
    if not 0.4 < labels.mean() < 0.6:
        raise RuntimeError(
            f"probe labels are unbalanced after length filtering ({labels.mean():.2f} positive); "
            f"the accuracy figure would be meaningless. Fix make_ioi_dataset.py."
        )
    log.info(f"probe: {len(group)} prompts at length {best_len}, {labels.mean():.0%} positive")

    tokens = model.to_tokens([s["text"] for s in group])
    _, cache = model.run_with_cache(tokens)

    accuracies = []
    for layer in range(model.cfg.n_layers):
        activations = cache["resid_post", layer][:, -1, :].numpy()
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        scores = cross_val_score(clf, activations, labels, cv=5, scoring="accuracy")
        accuracies.append(float(scores.mean()))
        log.info(f"  layer {layer:2d}: probe accuracy {scores.mean():.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(model.cfg.n_layers)
    bars = ax.bar(x, accuracies, color=BLUE, zorder=3)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(0.5, color=RED, ls="--", lw=1.5, label="chance (balanced classes)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xlabel("layer the probe reads from")
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title('Probing for "is one of these names repeated?"')
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "probe_by_layer.pdf")
    plt.close(fig)
    return {"accuracies": accuracies, "n_prompts": len(group)}


def fig_attention_heads(model, data, log) -> dict:
    """Attention patterns for a few named heads on one running-example sentence."""
    heads = [(9, 9), (9, 6), (9, 8), (10, 7)]
    tokens = data["clean"][:1]
    str_tokens = model.to_str_tokens(tokens[0])
    _, cache = model.run_with_cache(tokens)

    fig, axes = plt.subplots(1, len(heads), figsize=(13.5, 3.9))
    for ax, (layer, head) in zip(axes, heads):
        pattern = cache["pattern", layer][0, head].numpy()
        im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"L{layer}H{head}", fontsize=11)
        ax.set_xticks(range(len(str_tokens)))
        ax.set_xticklabels([t.strip() or "_" for t in str_tokens], rotation=90, fontsize=6)
        ax.set_yticks(range(len(str_tokens)))
        ax.set_yticklabels([t.strip() or "_" for t in str_tokens], fontsize=6)
    axes[0].set_ylabel("query position (from)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="attention weight")
    fig.suptitle("Four real heads, same sentence", y=0.99)
    fig.savefig(FIG_DIR / "attention_heads.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"attention patterns drawn for {heads}")
    return {"heads": [f"L{l}H{h}" for l, h in heads], "tokens": str_tokens}


def fig_induction_scores(model, log) -> dict:
    """Induction score per head: attention to the token after a previous copy of yourself."""
    generator = torch.Generator().manual_seed(SEED)
    random_tokens = torch.randint(
        1000, 10000, (N_INDUCTION_SEQ, INDUCTION_SEQ_LEN), generator=generator
    )
    bos = torch.full((N_INDUCTION_SEQ, 1), model.tokenizer.bos_token_id)
    tokens = torch.cat([bos, random_tokens, random_tokens], dim=1)

    _, cache = model.run_with_cache(tokens)

    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer]  # [batch, head, query, key]
        # A repeated-sequence induction head attends from position i in the second copy back to
        # the token that FOLLOWED the earlier occurrence: offset seq_len - 1 on the diagonal.
        diagonal = pattern.diagonal(offset=-(INDUCTION_SEQ_LEN - 1), dim1=-2, dim2=-1)
        scores[layer] = diagonal.mean(dim=(0, 2))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(scores.numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(model.cfg.n_heads))
    ax.set_yticks(range(model.cfg.n_layers))
    ax.set_title("Induction score: attending to what came next last time")
    fig.colorbar(im, ax=ax, label="mean attention on the induction stripe")

    flat = scores.flatten()
    top = torch.topk(flat, 3)
    found = []
    for i, value in zip(top.indices, top.values):
        layer, head = int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads
        found.append({"head": f"L{layer}H{head}", "score": value.item()})
        ax.text(head, layer, "*", ha="center", va="center", color=ORANGE, fontsize=16, fontweight="bold")
        log.info(f"  induction head L{layer}H{head}: {value.item():.3f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "induction_scores.pdf")
    plt.close(fig)
    return {"top_heads": found, "seq_len": INDUCTION_SEQ_LEN}


def main() -> None:
    log = setup_logging("l45_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(log)
    data = load_prompts(model, log, n=64)
    directions = logit_diff_directions(model, data["answer_tokens"])

    logits = model(data["clean"])
    log.info(f"baseline logit diff on this batch: {logit_diff(logits, data['answer_tokens']):.3f}")

    results = {
        "logit_lens": fig_logit_lens(model, data, directions, log),
        "attention_heads": fig_attention_heads(model, data, log),
        "induction": fig_induction_scores(model, log),
        "probe": fig_probe_by_layer(model, log),
    }
    save_results("l45_figs", results, log)
    log.info(f"4 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
