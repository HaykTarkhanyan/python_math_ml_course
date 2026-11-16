"""Extra figures for "Opening the box" (02_opening_the_box.tex), formerly L45.

The first draft of L45 *asserted* the two ideas the whole chapter rests on. This script measures
them instead, so the slides can show rather than claim:

* ``logit_decomposition.pdf`` - every component's contribution to the logit difference, and the
  fact that they **add up to the real number**. Linearity stops being a claim about the
  architecture and becomes an arithmetic check the room can watch.
* ``probe_vs_lens.pdf``       - probe accuracy and logit difference on one shared layer axis,
  so the six-layer gap between "readable" and "used" is visible instead of tabulated.

It also records the raw logits for one sentence, for the by-hand worked-numbers frame.

Measurements go to ``results/l45_extra_figs.json``; the figures are drawn from that file (and from
``results/l45_figs.json`` for the probe/lens pair). Figures are drawn at their slot size.

Usage (about 30 s on CPU with GPT-2 small cached; --plot-only takes seconds):
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_extra_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_extra_figs.py --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mi_common import FIG_DIR, load_results, save_results, setup_logging

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"
N_TOP = 6

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
SAVE = dict(bbox_inches="tight", metadata={"CreationDate": None})


# ---------------------------------------------------------------- measurement (needs the model)

def worked_numbers(model, data, log) -> dict:
    """The actual logits for one sentence - the by-hand frame."""
    import torch
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


def measure_decomposition(model, data, directions, log) -> dict:
    """Break the logit difference into one term per component, and check the sum."""
    import torch
    from ioi_core import logit_diff
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
    top_idx = np.argsort(np.abs(values))[::-1][:N_TOP]
    return {
        "true_logit_diff": true_ld,
        "sum_of_parts": total,
        "unembed_bias": bias_term,
        "gap": gap,
        "n_components": len(labels),
        "top": [{"component": labels[i], "contribution": float(values[i])} for i in top_idx],
        "rest": float(values.sum() - values[top_idx].sum()),
    }


def run(log) -> dict:
    from ioi_core import load_model, load_prompts, logit_diff_directions
    model = load_model(log)
    data = load_prompts(model, log, n=64)
    directions = logit_diff_directions(model, data["answer_tokens"])
    return {
        "worked_numbers": worked_numbers(model, data, log),
        "decomposition": measure_decomposition(model, data, directions, log),
    }


# ---------------------------------------------------------------- figures (JSON only)

def fig_decomposition(res: dict, log) -> None:
    d = res["decomposition"]
    n_other = d["n_components"] - len(d["top"])
    names = [t["component"] for t in d["top"]] + [f"{n_other}\nothers", "unembed\nbias"]
    heights = [t["contribution"] for t in d["top"]] + [d["rest"], d["unembed_bias"]]
    if abs(sum(heights) - d["true_logit_diff"]) > 0.01:
        raise RuntimeError("the bars do not add up to the logit difference the figure claims")
    colors = [BLUE if v > 0 else RED for v in heights[:-1]] + ["0.6"]

    fig, ax = plt.subplots(figsize=(5.0, 2.15))
    x = np.arange(len(heights))
    bars = ax.bar(x, heights, color=colors, zorder=3, width=0.72)
    ax.bar_label(bars, labels=[f"{h:+.2f}" for h in heights], fontsize=7, padding=1.5)
    xa = len(heights) + 0.55
    ax.bar([xa], [d["true_logit_diff"]], color=ORANGE, zorder=3, width=0.72)
    ax.text(xa, d["true_logit_diff"] + 0.12, f"{d['true_logit_diff']:+.2f}", ha="center",
            va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvline(len(heights) - 0.2, color="0.6", ls=":", lw=1.2)
    ax.set_xticks(list(x) + [xa])
    ax.set_xticklabels(names + ["actual\nlogit diff"], fontsize=7)
    ax.set_ylabel("share of the\nlogit difference", fontsize=8.5)
    ax.set_ylim(min(heights) - 0.5, max(max(heights), d["true_logit_diff"]) + 0.6)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "logit_decomposition.pdf", **SAVE)
    plt.close(fig)
    log.info(f"decomposition: {d['n_components']} components, bars sum {sum(heights):+.4f}, "
             f"actual {d['true_logit_diff']:+.4f}")


def fig_probe_vs_lens(log) -> None:
    """The six-layer gap, on one shared x axis: x = the stream after layer L, both panels."""
    saved = load_results("l45_figs")
    accuracies = saved["probe"]["accuracies"]
    lens_values = saved["logit_lens"]["values"]
    labels = saved["logit_lens"]["labels"]
    # accumulated_resid has one more point than there are layers (the embedding first); drop it so
    # both panels read "the stream after layer L" for L = 0..11.
    if len(lens_values) != len(accuracies) + 1 or labels[0] != "0_pre":
        raise RuntimeError(f"unexpected logit-lens layout: {labels}")
    lens_by_layer = lens_values[1:]

    first_perfect = next(i for i, a in enumerate(accuracies) if a >= 0.999)
    used = int(np.argmax(np.diff([lens_values[0]] + lens_by_layer)))
    if used != 9:
        raise RuntimeError(f"the slides say the answer is written in layer 9; the data says {used}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.7, 2.55), sharex=True)
    x = np.arange(len(accuracies))
    for ax in (ax1, ax2):
        ax.axvspan(first_perfect - 0.15, used - 0.85, color=RED, alpha=0.08, zorder=0)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=7.5)
    ax1.plot(x, accuracies, marker="o", ms=3, color=BLUE, lw=1.6, zorder=3)
    ax1.axhline(0.5, color="0.6", ls="--", lw=0.9)
    ax1.text(10.9, 0.44, "chance", fontsize=7, color="0.4", va="center", ha="center")
    ax1.set_ylim(0.4, 1.1)
    ax1.set_ylabel("probe\naccuracy", fontsize=8)
    mid = (first_perfect + used - 1) / 2
    ax1.text(mid, 0.66, f"readable from layer {first_perfect}", ha="center", fontsize=8, color=RED)

    ax2.plot(x, lens_by_layer, marker="o", ms=3, color=ORANGE, lw=1.6, zorder=3)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_ylabel("logit\ndifference", fontsize=8)
    ax2.set_xlabel("residual stream after layer", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.text(mid, 2.2, f"{used - first_perfect} layers: not used yet", ha="center", fontsize=8,
             color=RED)
    ax2.annotate(f"used in layer {used}", xy=(used - 0.45, 2.6), xytext=(used - 3.6, 3.7),
                 fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(FIG_DIR / "probe_vs_lens.pdf", **SAVE)
    plt.close(fig)
    log.info(f"probe readable from layer {first_perfect}; answer written in layer {used}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("l45_extra_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        save_results("l45_extra_figs", run(log), log)
    fig_decomposition(load_results("l45_extra_figs"), log)
    fig_probe_vs_lens(log)


if __name__ == "__main__":
    main()
