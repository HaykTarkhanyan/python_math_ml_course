"""Gate experiment: does the Hydra effect reproduce in GPT-2-small on IOI?

L46 section 6 is built on one claim: **ablate the name-mover heads and performance barely
drops, because backup name-mover heads take over.** If that does not reproduce at this scale,
that section needs a different spine, and the chapter should find out now rather than after the
deck is written.

The experiment:

1. baseline logit difference on clean IOI prompts;
2. per-head direct logit attribution -> the name movers are the top positive heads;
3. mean-ablate the top-k name movers, measure the drop;
4. **the actual test** - recompute per-head DLA under ablation and compare to baseline. Self-repair
   shows up as *other* heads' attribution going **up** once the name movers are gone.

Step 4 is the one that matters. A drop smaller than DLA predicted is suggestive; heads visibly
increasing their contribution is the mechanism.

Raw numbers go to ``results/self_repair.json``; the L46 figure is built from that file.
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

N_NAME_MOVERS = 3
N_PROMPTS = 128


def mean_ablate_heads(model, tokens, heads, directions, answer_tokens):
    """Mean-ablate the given (layer, head) pairs and return (logit_diff, per-head DLA).

    Mean over the batch, not zero. Zero is not a neutral value - it is an activation the model
    has never seen, so a zero-ablation result tells you about the corruption rather than about
    the head. This is the distinction L46 section 1 spends a frame on, so the code that
    generates its figures had better honour it.
    """
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(head_idx):
        def hook(z, hook):  # z: [batch, pos, head, d_head]
            z[:, :, head_idx, :] = z[:, :, head_idx, :].mean(dim=0, keepdim=True)
            return z

        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(hs)) for layer, hs in by_layer.items()]

    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens)

    return logit_diff(logits, answer_tokens).item(), per_head_dla(model, cache, directions)


def main() -> None:
    log = setup_logging("self_repair")
    model = load_model(log)
    data = load_prompts(model, log, n=N_PROMPTS)

    answer_tokens = data["answer_tokens"]
    directions = logit_diff_directions(model, answer_tokens)

    log.info("--- baseline ---")
    logits, cache = model.run_with_cache(data["clean"])
    baseline_ld = logit_diff(logits, answer_tokens).item()
    baseline_dla = per_head_dla(model, cache, directions)
    log.info(f"baseline logit diff: {baseline_ld:.3f}")

    flat = baseline_dla.flatten()
    top = torch.topk(flat, N_NAME_MOVERS)
    name_movers = [(int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads) for i in top.indices]
    log.info("top heads by direct logit attribution (the name movers):")
    for (layer, head), value in zip(name_movers, top.values):
        log.info(f"  {head_label(layer, head):8s} DLA {value.item():+.3f}")

    bottom = torch.topk(-flat, 3)
    log.info("most negative heads (the negative name movers):")
    for i, value in zip(bottom.indices, bottom.values):
        layer, head = int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads
        log.info(f"  {head_label(layer, head):8s} DLA {-value.item():+.3f}")

    log.info(f"--- ablating {N_NAME_MOVERS} name movers (mean-ablation) ---")
    ablated_ld, ablated_dla = mean_ablate_heads(
        model, data["clean"], name_movers, directions, answer_tokens
    )

    predicted_drop = sum(baseline_dla[l, h].item() for l, h in name_movers)
    actual_drop = baseline_ld - ablated_ld
    log.info(f"baseline logit diff : {baseline_ld:+.3f}")
    log.info(f"ablated  logit diff : {ablated_ld:+.3f}")
    log.info(f"DLA predicted drop  : {predicted_drop:+.3f}")
    log.info(f"actual drop         : {actual_drop:+.3f}")
    log.info(f"recovered by backups: {predicted_drop - actual_drop:+.3f} "
             f"({100 * (predicted_drop - actual_drop) / predicted_drop:.0f}% of the prediction)")

    log.info("--- which heads compensated? (DLA change, name movers excluded) ---")
    delta = ablated_dla - baseline_dla
    for layer, head in name_movers:
        delta[layer, head] = 0.0

    top_backup = torch.topk(delta.flatten(), 6)
    backups = []
    for i, value in zip(top_backup.indices, top_backup.values):
        layer, head = int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads
        backups.append(
            {
                "head": head_label(layer, head),
                "layer": layer,
                "head_idx": head,
                "baseline_dla": baseline_dla[layer, head].item(),
                "ablated_dla": ablated_dla[layer, head].item(),
                "delta": value.item(),
            }
        )
        log.info(
            f"  {head_label(layer, head):8s} "
            f"{baseline_dla[layer, head].item():+.3f} -> {ablated_dla[layer, head].item():+.3f} "
            f"(delta {value.item():+.3f})"
        )

    total_compensation = delta[delta > 0].sum().item()
    log.info(f"total positive DLA increase across all other heads: {total_compensation:+.3f}")

    save_results(
        "self_repair",
        {
            "model": "gpt2-small",
            "n_prompts": N_PROMPTS,
            "ablation": "mean",
            "baseline_logit_diff": baseline_ld,
            "ablated_logit_diff": ablated_ld,
            "dla_predicted_drop": predicted_drop,
            "actual_drop": actual_drop,
            "compensation": predicted_drop - actual_drop,
            "name_movers": [
                {"head": head_label(l, h), "dla": baseline_dla[l, h].item()} for l, h in name_movers
            ],
            "backups": backups,
            "total_positive_dla_increase": total_compensation,
            "baseline_dla": baseline_dla.tolist(),
            "ablated_dla": ablated_dla.tolist(),
        },
        log,
    )

    log.info("=" * 70)
    if actual_drop < predicted_drop * 0.85:
        log.info("GATE PASSED: ablation under-delivers vs DLA - self-repair is present.")
    else:
        log.info("GATE FAILED: ablation drop matches the DLA prediction - no self-repair here.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
