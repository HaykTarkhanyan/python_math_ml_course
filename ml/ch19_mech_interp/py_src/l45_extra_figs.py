"""Extra L45 figures for the pedagogical slow-down pass.

The first draft of L45 *asserted* the two ideas the whole chapter rests on. This script measures
them instead, so the slides can show rather than claim:

* ``logit_decomposition.pdf`` - every component's contribution to one sentence's logit
  difference, and the fact that they **add up to the real number**. Linearity stops being a
  claim about the architecture and becomes an arithmetic check the room can watch.
* ``probe_vs_lens.pdf``       - probe accuracy and logit difference on one shared layer axis,
  so the six-layer gap between "readable" and "used" is visible instead of tabulated.

It also prints the raw logits for one sentence, for the by-hand worked-numbers frame.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from ioi_core import (
    FIG_DIR,
    RESULTS_DIR,
    load_model,
    load_prompts,
    logit_diff,
    logit_diff_directions,
    save_results,
    setup_logging,
)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
N_TOP = 6


def worked_numbers(model, data, log) -> dict:
    """The actual logits for one sentence - the by-hand frame."""
    sample = data["samples"][0]
    tokens = data["clean"][:1]
    logits = model(tokens)[0, -1, :]

    io_tok = model.to_single_token(sample["answer"])
    s_tok = model.to_single_token(sample["wrong_answer"])

    top = torch.topk(logits, 5)
    top_tokens = [(model.to_string(i.item()), v.item()) for i, v in zip(top.indices, top.values)]

    result = {
        "prompt": sample["clean"],
        "io": sample["answer"],
        "s": sample["wrong_answer"],
        "logit_io": logits[io_tok].item(),
        "logit_s": logits[s_tok].item(),
        "logit_diff": logits[io_tok].item() - logits[s_tok].item(),
        "top5": top_tokens,
    }
    log.info(f"worked example: {result['prompt']!r}")
    log.info(f"  logit({result['io']!r}) = {result['logit_io']:.3f}")
    log.info(f"  logit({result['s']!r})  = {result['logit_s']:.3f}")
    log.info(f"  difference           = {result['logit_diff']:.3f}")
    log.info(f"  top-5 predictions: {[(t, round(v, 2)) for t, v in top_tokens]}")
    return result


def fig_decomposition(model, data, directions, log) -> dict:
    """Break the logit difference into one term per component, and check the sum."""
    tokens = data["clean"]
    logits, cache = model.run_with_cache(tokens)
    true_ld = logit_diff(logits, data["answer_tokens"]).item()

    stack, labels = cache.get_full_resid_decomposition(
        layer=-1, pos_slice=-1, expand_neurons=False, return_labels=True
    )
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)
    contributions = torch.einsum("cbd,bd->c", stack, directions) / directions.shape[0]

    total = contributions.sum().item()

    # GPT-2's unembedding carries a per-token bias (max |b_U| ~ 7.0). It shifts the logit
    # difference by a constant that depends only on WHICH two names are being compared - the
    # model computes none of it. Leave it out and the components appear to under-explain the
    # answer by ~7%; put it in and the sum is exact. The slide shows it as its own bar.
    answer_tokens = data["answer_tokens"]
    bias_term = (model.b_U[answer_tokens[:, 0]] - model.b_U[answer_tokens[:, 1]]).mean().item()

    log.info(f"sum of {len(labels)} component contributions: {total:+.4f}")
    log.info(f"unembedding bias term (not computed)        : {bias_term:+.4f}")
    log.info(f"components + bias                           : {total + bias_term:+.4f}")
    log.info(f"actual logit difference                     : {true_ld:+.4f}")

    gap = abs(total + bias_term - true_ld)
    if gap > 0.01:
        raise RuntimeError(
            f"decomposition does not reconstruct the logit difference (gap {gap:.4f}). "
            f"The slide claims these add up exactly, so this must not ship as an approximation."
        )
    log.info(f"reconstruction gap: {gap:.5f} - exact")

    values = contributions.numpy()
    order = np.argsort(np.abs(values))[::-1]
    top_idx = order[:N_TOP]
    rest = float(values.sum() - values[top_idx].sum())

    n_other = len(labels) - N_TOP
    names = [labels[i] for i in top_idx] + [f"all {n_other}\nothers", "unembed\nbias"]
    heights = list(values[top_idx]) + [rest, bias_term]
    colors = [BLUE if v > 0 else RED for v in heights[:-1]] + ["0.6"]

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    bars = ax.bar(range(len(heights)), heights, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=2)

    ax.bar([len(heights) + 0.6], [true_ld], color=ORANGE, zorder=3, width=0.8)
    ax.text(len(heights) + 0.6, true_ld + 0.12, f"{true_ld:+.2f}", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(len(heights) + 0.6, -0.62, "all of these,\nadded up", ha="center", fontsize=9,
            color="0.25")

    ax.axhline(0, color="0.3", lw=1)
    ax.axvline(len(heights) - 0.3, color="0.6", ls=":", lw=1.5)
    ax.set_xticks(list(range(len(heights))) + [len(heights) + 0.6])
    ax.set_xticklabels(names + ["actual\nlogit diff"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("contribution to the logit difference")
    ax.set_title("Every component's share - and they add up to the real number")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logit_decomposition.pdf")
    plt.close(fig)

    return {
        "true_logit_diff": true_ld,
        "sum_of_parts": total,
        "unembed_bias": bias_term,
        "gap": gap,
        "n_components": len(labels),
        "top": [{"component": labels[i], "contribution": float(values[i])} for i in top_idx],
        "rest": rest,
    }


def fig_probe_vs_lens(log) -> dict:
    """The six-layer gap, on one shared x axis."""
    path = RESULTS_DIR / "l45_figs.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run l45_figs.py first.")
    saved = json.loads(path.read_text(encoding="utf-8"))

    accuracies = saved["probe"]["accuracies"]
    lens_values = saved["logit_lens"]["values"]
    # accumulated_resid returns one entry per layer boundary plus the embedding; align the tail
    # of that series with layers 0..n-1 so both panels share an x axis.
    lens_by_layer = lens_values[-len(accuracies):]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    x = np.arange(len(accuracies))

    ax1.plot(x, accuracies, marker="o", color=BLUE, lw=2, zorder=3)
    ax1.axhline(0.5, color="0.6", ls="--", lw=1.2)
    ax1.text(0.15, 0.53, "chance", fontsize=8, color="0.4")
    ax1.set_ylim(0.4, 1.08)
    ax1.set_ylabel("probe accuracy\n(is a name repeated?)")
    ax1.set_title("The information is there long before it is used")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, lens_by_layer, marker="o", color=ORANGE, lw=2, zorder=3)
    ax2.axhline(0, color="0.6", lw=1)
    ax2.set_ylabel("logit difference\n(what it would answer)")
    ax2.set_xlabel("layer")
    ax2.set_xticks(x)
    ax2.grid(axis="y", alpha=0.3)

    first_perfect = next(i for i, a in enumerate(accuracies) if a >= 0.999)
    for ax in (ax1, ax2):
        ax.axvspan(first_perfect, 9, color=RED, alpha=0.08, zorder=0)
    ax1.annotate(
        f"readable from layer {first_perfect}",
        xy=(first_perfect, accuracies[first_perfect]),
        xytext=(first_perfect + 0.4, 0.62),
        fontsize=9, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    ax2.annotate(
        "used at layer 9",
        xy=(9, lens_by_layer[9]),
        xytext=(4.4, lens_by_layer[9] + 0.6),
        fontsize=9, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    mid = (first_perfect + 9) / 2
    ax2.text(mid, min(lens_by_layer) - 0.15, f"{9 - first_perfect} layers, unused",
             ha="center", fontsize=9, color=RED, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "probe_vs_lens.pdf")
    plt.close(fig)
    log.info(f"probe readable from layer {first_perfect}; answer appears at layer 9")
    return {"first_perfect_layer": first_perfect, "gap_layers": 9 - first_perfect}


def main() -> None:
    log = setup_logging("l45_extra_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(log)
    data = load_prompts(model, log, n=64)
    directions = logit_diff_directions(model, data["answer_tokens"])

    results = {
        "worked_numbers": worked_numbers(model, data, log),
        "decomposition": fig_decomposition(model, data, directions, log),
        "probe_vs_lens": fig_probe_vs_lens(log),
    }
    save_results("l45_extra_figs", results, log)


if __name__ == "__main__":
    main()
