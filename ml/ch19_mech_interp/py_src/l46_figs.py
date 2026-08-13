"""Figures for L46 - Does it actually do that?

Five figures:

* ``cold_open_three_heads.pdf`` - L9H9, L9H6, L9H8 side by side, causal effect deliberately
  hidden. The room is asked to pick the head that does nothing, and cannot;
* ``dla_per_head.pdf``          - direct logit attribution, every head, sorted;
* ``ablation_compare.pdf``      - zero vs mean vs resample ablation, same heads, three answers;
* ``patching_heatmap.pdf``      - the canonical layer x position causal-tracing grid;
* ``self_repair.pdf``           - built from ``results/self_repair.json``, not recomputed.

The patching grid is the expensive one: ``n_layers x seq_len`` forward passes. It runs on a
deliberately small prompt batch because the heatmap is an average over prompts and 16 is already
enough to make the structure unambiguous.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from ioi_core import (
    FIG_DIR,
    RESULTS_DIR,
    SEED,
    head_label,
    load_model,
    load_prompts,
    logit_diff,
    logit_diff_directions,
    per_head_dla,
    save_results,
    setup_logging,
)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

N_PROMPTS = 64
N_PATCHING = 16
COLD_OPEN_HEADS = [(9, 9), (9, 6), (9, 8)]
ABLATION_HEADS = [(9, 9), (9, 6), (10, 0)]


def fig_cold_open(model, data, log) -> dict:
    """Three heads in the same layer, all looking at the answer. One does nothing."""
    tokens = data["clean"][:1]
    str_tokens = [t.strip() or "_" for t in model.to_str_tokens(tokens[0])]
    _, cache = model.run_with_cache(tokens)

    # Drop the BOS column. Every GPT-2 head parks a large share of its attention on the first
    # token - the "attention sink" - and leaving it in makes the colour scale about the sink
    # rather than about the sentence. The deck says this out loud on the frame; it is not a
    # cosmetic crop.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (layer, head) in zip(axes, COLD_OPEN_HEADS):
        pattern = cache["pattern", layer][0, head].numpy()[1:, 1:]
        ax.imshow(pattern, cmap="Blues", vmin=0, vmax=pattern.max(), aspect="auto")
        ax.set_title(head_label(layer, head), fontsize=13)
        ax.set_xticks(range(len(str_tokens) - 1))
        ax.set_xticklabels(str_tokens[1:], rotation=90, fontsize=6)
        ax.set_yticks(range(len(str_tokens) - 1))
        ax.set_yticklabels(str_tokens[1:] if ax is axes[0] else [], fontsize=6)
    fig.suptitle(
        "Three heads, same layer, same sentence. One contributes nothing.\n"
        "(attention sink on the first token removed; each panel scaled to its own maximum)",
        y=1.02,
        fontsize=11,
    )
    fig.savefig(FIG_DIR / "cold_open_three_heads.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("cold open figure drawn (causal effect deliberately not shown)")
    return {"heads": [head_label(l, h) for l, h in COLD_OPEN_HEADS]}


def fig_dla(model, data, directions, log) -> dict:
    _, cache = model.run_with_cache(data["clean"])
    dla = per_head_dla(model, cache, directions)

    flat = dla.flatten().numpy()
    order = np.argsort(flat)[::-1]
    labels = [head_label(int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads) for i in order]

    n_show = 12
    idx = list(range(n_show)) + list(range(len(order) - 4, len(order)))
    values = [flat[order[i]] for i in idx]
    names = [labels[i] for i in idx]
    colors = [BLUE if v > 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(range(len(values)), values, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%+.2f", fontsize=7, padding=2)
    ax.axhline(0, color="0.3", lw=1)
    ax.axvline(n_show - 0.5, color="0.6", ls=":", lw=1.5)
    ax.text(n_show - 0.4, max(values) * 0.85, "  ...the other 128 heads...", fontsize=8, color="0.4")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("direct logit attribution")
    ax.set_title("Which heads push toward the correct name - and which push away")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dla_per_head.pdf")
    plt.close(fig)

    top = [{"head": labels[i], "dla": float(flat[order[i]])} for i in range(5)]
    log.info(f"top DLA heads: {[(t['head'], round(t['dla'], 2)) for t in top]}")
    return {"top": top, "dla": dla.tolist()}


def fig_ablation_compare(model, data, log) -> dict:
    """Zero, mean and resample ablation of the same heads give three different answers."""
    answer_tokens = data["answer_tokens"]
    baseline = logit_diff(model(data["clean"]), answer_tokens).item()

    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data["clean"].shape[0], generator=generator)

    def run(mode: str) -> float:
        by_layer: dict[int, list[int]] = {}
        for layer, head in ABLATION_HEADS:
            by_layer.setdefault(layer, []).append(head)

        def make_hook(head_idxs):
            def hook(z, hook):
                for h in head_idxs:
                    if mode == "zero":
                        z[:, :, h, :] = 0.0
                    elif mode == "mean":
                        z[:, :, h, :] = z[:, :, h, :].mean(dim=0, keepdim=True)
                    elif mode == "resample":
                        z[:, :, h, :] = z[perm][:, :, h, :]
                    else:
                        raise ValueError(f"unknown ablation mode {mode!r}")
                return z

            return hook

        hooks = [(f"blocks.{l}.attn.hook_z", make_hook(hs)) for l, hs in by_layer.items()]
        with model.hooks(fwd_hooks=hooks):
            return logit_diff(model(data["clean"]), answer_tokens).item()

    results = {mode: run(mode) for mode in ("zero", "mean", "resample")}
    for mode, value in results.items():
        log.info(f"  {mode:9s} ablation -> logit diff {value:+.3f} (drop {baseline - value:+.3f})")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = ["baseline", "zero", "mean", "resample"]
    values = [baseline] + [results[m] for m in ("zero", "mean", "resample")]
    bars = ax.bar(names, values, color=["0.55", RED, BLUE, ORANGE], zorder=3)
    ax.bar_label(bars, fmt="%+.2f", fontsize=10, padding=3)
    ax.axhline(baseline, color="0.55", ls="--", lw=1.2)
    ax.set_ylabel("logit difference after ablating\nL9H9, L9H6, L10H0")
    ax.set_title("Same three heads deleted. Three different stories.")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_compare.pdf")
    plt.close(fig)
    return {"baseline": baseline, **results}


def fig_patching_heatmap(model, data, log) -> dict:
    """Causal tracing: patch clean residual stream into the corrupted run, position by position."""
    clean = data["clean"][:N_PATCHING]
    corrupt = data["corrupt_swap"][:N_PATCHING]
    answer_tokens = data["answer_tokens"][:N_PATCHING]

    clean_ld = logit_diff(model(clean), answer_tokens).item()
    corrupt_ld = logit_diff(model(corrupt), answer_tokens).item()
    span = clean_ld - corrupt_ld
    log.info(f"patching endpoints: clean {clean_ld:+.3f}, corrupt {corrupt_ld:+.3f}, span {span:.3f}")
    if abs(span) < 1.0:
        raise RuntimeError(
            f"clean and corrupted runs differ by only {span:.3f} logits; the patching heatmap "
            f"would be noise. Check the corruption in make_ioi_dataset.py."
        )

    _, clean_cache = model.run_with_cache(clean)
    seq_len = clean.shape[1]
    grid = np.zeros((model.cfg.n_layers, seq_len))

    for layer in range(model.cfg.n_layers):
        for pos in range(seq_len):
            def hook(resid, hook, layer=layer, pos=pos):
                resid[:, pos, :] = clean_cache["resid_pre", layer][:, pos, :]
                return resid

            with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)]):
                patched = logit_diff(model(corrupt), answer_tokens).item()
            grid[layer, pos] = (patched - corrupt_ld) / span
        log.info(f"  layer {layer:2d} patched, max recovery {grid[layer].max():.2f}")

    str_tokens = [t.strip() or "_" for t in model.to_str_tokens(clean[0])]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(str_tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(model.cfg.n_layers))
    ax.set_ylabel("layer patched")
    ax.set_xlabel("position patched")
    ax.set_title("Activation patching: how much of the correct behaviour comes back")
    fig.colorbar(im, ax=ax, label="fraction of the gap recovered")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "patching_heatmap.pdf")
    plt.close(fig)

    best = np.unravel_index(np.argmax(grid), grid.shape)
    log.info(f"strongest patch: layer {best[0]}, position {best[1]} ({str_tokens[best[1]]!r}) "
             f"recovering {grid[best]:.2f}")
    return {
        "clean_logit_diff": clean_ld,
        "corrupt_logit_diff": corrupt_ld,
        "grid": grid.tolist(),
        "tokens": str_tokens,
        "best": {"layer": int(best[0]), "pos": int(best[1]), "recovery": float(grid[best])},
    }


def fig_self_repair(log) -> dict:
    """Rebuilt from the saved gate result - the experiment is not re-run here."""
    path = RESULTS_DIR / "self_repair.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run self_repair.py first.")
    r = json.loads(path.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    names = ["baseline", "DLA says\nit should be", "what it\nactually is"]
    values = [
        r["baseline_logit_diff"],
        r["baseline_logit_diff"] - r["dla_predicted_drop"],
        r["ablated_logit_diff"],
    ]
    bars = ax1.bar(names, values, color=["0.55", RED, BLUE], zorder=3)
    ax1.bar_label(bars, fmt="%+.2f", fontsize=10, padding=3)
    ax1.axhline(0, color="0.3", lw=1)
    ax1.set_ylabel("logit difference")
    ax1.set_title("Delete the three heads that do the job")
    ax1.grid(axis="y", alpha=0.3)

    backups = r["backups"][:5]
    labels = [b["head"] for b in backups]
    before = [b["baseline_dla"] for b in backups]
    after = [b["ablated_dla"] for b in backups]
    x = np.arange(len(labels))
    b1 = ax2.bar(x - 0.2, before, 0.4, label="before ablation", color="0.65", zorder=3)
    b2 = ax2.bar(x + 0.2, after, 0.4, label="after ablation", color=ORANGE, zorder=3)
    ax2.bar_label(b1, fmt="%+.2f", fontsize=6, padding=2)
    ax2.bar_label(b2, fmt="%+.2f", fontsize=6, padding=2)
    ax2.axhline(0, color="0.3", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("direct logit attribution")
    ax2.set_title("Who picked up the slack")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "self_repair.pdf")
    plt.close(fig)
    log.info("self-repair figure rebuilt from saved results")
    return {"source": str(path.name)}


def main() -> None:
    log = setup_logging("l46_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(log)
    data = load_prompts(model, log, n=N_PROMPTS)
    directions = logit_diff_directions(model, data["answer_tokens"])

    results = {
        "cold_open": fig_cold_open(model, data, log),
        "dla": fig_dla(model, data, directions, log),
        "ablation_compare": fig_ablation_compare(model, data, log),
        "self_repair": fig_self_repair(log),
        "patching": fig_patching_heatmap(model, data, log),
    }
    save_results("l46_figs", {k: v for k, v in results.items() if k != "dla"} | {"dla_top": results["dla"]["top"]}, log)
    log.info(f"5 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
