"""Gate experiment: is there a GPT-2-small head that *looks* important on IOI and is not?

L46's cold open wants a concrete trap - a head whose attention pattern tells a convincing story
("it looks right at the answer!") and whose causal effect is nil. That frame is the argument for
the whole deck: attention patterns are suggestive, never conclusive.

The hunt has three filters, in order:

1. **looks important** - high attention from the final position to the indirect-object name;
2. **contributes nothing** - direct logit attribution near zero;
3. **causally inert** - mean-ablating it alone barely moves the logit difference.

A head passing all three is the cold open. If none does, the cold open falls back to the
self-repair result and this script records that it was checked.
"""

from __future__ import annotations

import torch

from ioi_core import (
    head_label,
    load_model,
    load_prompts,
    logit_diff,
    logit_diff_directions,
    per_head_dla,
    save_results,
    setup_logging,
)

N_PROMPTS = 64
N_CANDIDATES = 8

# What counts as "looks important" and "does nothing". Stated up front so the verdict is not
# tuned after seeing the numbers.
MIN_ATTENTION = 0.15
MAX_ABS_DLA = 0.15
MAX_ABS_ABLATION_EFFECT = 0.15


def io_positions(model, data) -> torch.Tensor:
    """Position of the indirect-object name token in each prompt.

    The template order differs between BABA and ABBA prompts, so this is per-prompt, not a
    constant. Found by matching the answer token id rather than by counting words.
    """
    tokens, answers = data["clean"], data["answer_tokens"][:, 0]
    positions = []
    for row, answer in zip(tokens, answers):
        hits = (row == answer).nonzero(as_tuple=True)[0]
        if len(hits) != 1:
            raise RuntimeError(
                f"expected the IO name exactly once per prompt, found {len(hits)} occurrences. "
                f"The dataset generator should guarantee IO != S and IO != C."
            )
        positions.append(hits[0].item())
    return torch.tensor(positions)


def ablate_one_head(model, tokens, layer, head, answer_tokens) -> float:
    def hook(z, hook):
        z[:, :, head, :] = z[:, :, head, :].mean(dim=0, keepdim=True)
        return z

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]):
        logits = model(tokens)
    return logit_diff(logits, answer_tokens).item()


def main() -> None:
    log = setup_logging("decoy_head_hunt")
    model = load_model(log)
    data = load_prompts(model, log, n=N_PROMPTS)

    answer_tokens = data["answer_tokens"]
    directions = logit_diff_directions(model, answer_tokens)
    io_pos = io_positions(model, data)
    log.info(f"IO token positions in batch: {sorted(set(io_pos.tolist()))}")

    logits, cache = model.run_with_cache(data["clean"])
    baseline_ld = logit_diff(logits, answer_tokens).item()
    dla = per_head_dla(model, cache, directions)
    log.info(f"baseline logit diff: {baseline_ld:.3f}")

    # Attention from the final position to the IO name, per head, averaged over prompts.
    batch = torch.arange(len(io_pos))
    attn_to_io = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer]  # [batch, head, query, key]
        attn_to_io[layer] = pattern[batch, :, -1, io_pos].mean(dim=0)

    log.info("--- heads that LOOK most important (attention to the IO name) ---")
    top = torch.topk(attn_to_io.flatten(), N_CANDIDATES)
    candidates = []
    for i, attn in zip(top.indices, top.values):
        layer, head = int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads
        candidates.append((layer, head, attn.item(), dla[layer, head].item()))
        log.info(
            f"  {head_label(layer, head):8s} attn->IO {attn.item():.3f}   "
            f"DLA {dla[layer, head].item():+.3f}"
        )

    log.info("--- causal check on each candidate (ablate alone) ---")
    records = []
    for layer, head, attn, head_dla in candidates:
        ablated_ld = ablate_one_head(model, data["clean"], layer, head, answer_tokens)
        effect = baseline_ld - ablated_ld
        is_decoy = (
            attn >= MIN_ATTENTION
            and abs(head_dla) <= MAX_ABS_DLA
            and abs(effect) <= MAX_ABS_ABLATION_EFFECT
        )
        log.info(
            f"  {head_label(layer, head):8s} attn {attn:.3f}  DLA {head_dla:+.3f}  "
            f"ablation effect {effect:+.3f}  {'DECOY' if is_decoy else ''}"
        )
        records.append(
            {
                "head": head_label(layer, head),
                "layer": layer,
                "head_idx": head,
                "attn_to_io": attn,
                "dla": head_dla,
                "ablation_effect": effect,
                "is_decoy": is_decoy,
            }
        )

    decoys = [r for r in records if r["is_decoy"]]

    save_results(
        "decoy_head_hunt",
        {
            "model": "gpt2-small",
            "n_prompts": N_PROMPTS,
            "baseline_logit_diff": baseline_ld,
            "thresholds": {
                "min_attention": MIN_ATTENTION,
                "max_abs_dla": MAX_ABS_DLA,
                "max_abs_ablation_effect": MAX_ABS_ABLATION_EFFECT,
            },
            "candidates": records,
            "decoys": decoys,
        },
        log,
    )

    log.info("=" * 70)
    if decoys:
        best = max(decoys, key=lambda d: d["attn_to_io"])
        log.info(
            f"GATE PASSED: {best['head']} attends {best['attn_to_io']:.2f} to the answer and "
            f"moves the logit difference by {best['ablation_effect']:+.3f}. Use it for the cold open."
        )
    else:
        log.info("GATE FAILED: no head clears all three filters. Cold open falls back to self-repair.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
