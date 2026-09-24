"""Gate G7 + figures for the optional deck "Personas and model diffing" (xx_personas_and_diffing.tex).

A model organism with a KNOWN change: fine-tune GPT-2 small on positive movie-review sentences
only (SST-2), then ask the diffing questions as if we did not know what we did.

Runs on a GPU when present (committed results: Colab T4, DECISIONS #41), else CPU with 2 threads.
Deliberately small: one pass over 640 sentences of at most 32 tokens (~80 steps).

Measured, base vs tuned:
  1. behaviour: how positive is the next word on NEUTRAL prompts - about movies (the training
     domain) and about everything else (weather, a phone, a neighbour...)? Metric: log-sum-exp over
     five positive words minus five negative words ("positivity gap").
  2. a sentiment direction computed on the BASE model alone (mean residual of positive minus
     negative SST-2 validation sentences, per layer) - found independently of the fine-tune;
  3. weights: relative change of each block's weights;
  4. activations: mean shift of the residual stream on the neutral prompts, per layer, and its
     cosine with the sentiment direction (and with random directions, as a control);
  5. causality: steer the BASE model along the sentiment direction (strength as a multiple of the
     residual norm), and remove the direction from the TUNED model - how much of its positivity is
     that one direction?

The tuned checkpoint goes to ml/ch19_mech_interp/checkpoints/ (git-ignored, ~500 MB).
Results -> results/diffing_finetune.json; figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/diffing_finetune.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/diffing_finetune.py --plot-only
"""

from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import (BLUE, CHAPTER_DIR, FIG_DIR, GREY, ORANGE, RED, SEED, load_results, pick_device,
                       save_results, setup_logging)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

N_TRAIN, MAX_LEN, BATCH, LR = 640, 32, 8, 5e-5
N_DIR = 200                         # SST-2 validation sentences per label for the sentiment direction
POS_WORDS = [" great", " good", " wonderful", " amazing", " excellent"]
NEG_WORDS = [" bad", " terrible", " awful", " horrible", " poor"]
PROMPTS = {
    "movies": ["The movie was", "The film we saw last night was", "The acting in this movie was",
               "The plot of the film was", "The new episode was"],
    "weather": ["The weather today is", "The weather this weekend will be", "The rain this morning was"],
    "things": ["The new phone is", "My old car is", "The hotel room was", "The food at the restaurant was"],
    "people": ["My neighbor is", "My boss is", "The teacher in our school is"],
    "daily life": ["The traffic this morning was", "My day at work was", "The bus ride home was"],
}
STEER_LAYER = 6
STEER_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
CKPT = CHAPTER_DIR / "checkpoints" / "gpt2_sst2_positive.pt"


def positivity(model, tok, prompts, hook=None) -> list:
    """log-sum-exp over positive words minus over negative words, for each prompt's next token."""
    import torch
    pos = [tok.encode(w)[0] for w in POS_WORDS]
    neg = [tok.encode(w)[0] for w in NEG_WORDS]
    out = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=model.device)
        handle = model.transformer.h[STEER_LAYER].register_forward_hook(hook) if hook else None
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        if handle:
            handle.remove()
        lp = logits.log_softmax(-1)
        out.append(float(torch.logsumexp(lp[pos], 0) - torch.logsumexp(lp[neg], 0)))
    return out


def last_resid(model, tok, sentences) -> np.ndarray:
    """Residual stream at the last token, after every block: (n_layers+1, n, d)."""
    import torch
    rows = []
    for s in sentences:
        ids = torch.tensor([tok.encode(s)[:MAX_LEN]], device=model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        rows.append(torch.stack([h[0, -1] for h in hs]).cpu().numpy())
    return np.stack(rows, 1)


def mean_resid(model, tok, sentences) -> np.ndarray:
    """Residual stream averaged over all tokens of the sentence, after every block: (n_layers+1, n, d)."""
    import torch
    rows = []
    for s in sentences:
        ids = torch.tensor([tok.encode(s)[:MAX_LEN]], device=model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        rows.append(torch.stack([h[0].mean(0) for h in hs]).cpu().numpy())
    return np.stack(rows, 1)


def heldout_accuracy(rp: np.ndarray, rn: np.ndarray) -> list:
    """Difference-of-means direction from the first half of each class, scored on the second half."""
    half = rp.shape[1] // 2
    acc = []
    for layer in range(rp.shape[0]):
        d = rp[layer, :half].mean(0) - rn[layer, :half].mean(0)
        d = d / np.linalg.norm(d)
        thr = (rp[layer, :half] @ d).mean() / 2 + (rn[layer, :half] @ d).mean() / 2
        acc.append(float((np.mean(rp[layer, half:] @ d > thr) + np.mean(rn[layer, half:] @ d <= thr)) / 2))
    return acc


PIN_LAYERS = [6, 9, 11]


def run(log) -> dict:
    import copy
    import torch
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    dev = pick_device()
    log.info(f"device: {dev}")
    base = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)
    for w in POS_WORDS + NEG_WORDS:
        if len(tok.encode(w)) != 1:
            raise ValueError(f"{w!r} is not a single token")

    sst = load_dataset("stanfordnlp/sst2")
    train_pos = [s.strip() for s, lab in zip(sst["train"]["sentence"], sst["train"]["label"])
                 if lab == 1 and len(s.split()) >= 6][:N_TRAIN]
    val = sst["validation"]
    val_pos = [s.strip() for s, lab in zip(val["sentence"], val["label"]) if lab == 1][:N_DIR]
    val_neg = [s.strip() for s, lab in zip(val["sentence"], val["label"]) if lab == 0][:N_DIR]
    log.info(f"training on {len(train_pos)} positive SST-2 sentences; direction from {len(val_pos)}+{len(val_neg)}")
    out: dict = {"n_train": len(train_pos), "example_train": train_pos[:5], "device": dev}

    # ------------------------------------------------------------------ 2. sentiment direction, base model
    rp, rn = last_resid(base, tok, val_pos), last_resid(base, tok, val_neg)
    half = N_DIR // 2
    dirs, probe_acc = [], []
    for layer in range(rp.shape[0]):
        d = rp[layer, :half].mean(0) - rn[layer, :half].mean(0)
        d = d / np.linalg.norm(d)
        dirs.append(d)
        thr = (rp[layer, :half] @ d).mean() / 2 + (rn[layer, :half] @ d).mean() / 2
        acc = (np.mean(rp[layer, half:] @ d > thr) + np.mean(rn[layer, half:] @ d <= thr)) / 2
        probe_acc.append(float(acc))
    log.info(f"sentiment direction, held-out accuracy by layer: {np.round(probe_acc, 2).tolist()}")
    resid_norm = float(np.linalg.norm(rp[STEER_LAYER + 1], axis=1).mean())
    pooled_acc = heldout_accuracy(mean_resid(base, tok, val_pos), mean_resid(base, tok, val_neg))
    log.info(f"same, averaged over all tokens instead of the last one: {np.round(pooled_acc, 2).tolist()}")
    out["direction"] = {"heldout_accuracy_by_layer": probe_acc, "steer_layer": STEER_LAYER,
                        "heldout_accuracy_pooled_by_layer": pooled_acc,
                        "resid_norm_at_steer_layer": resid_norm}

    before = {dom: positivity(base, tok, ps) for dom, ps in PROMPTS.items()}

    # ------------------------------------------------------------------ fine-tune (or reuse the checkpoint)
    tuned = copy.deepcopy(base)
    if CKPT.exists():
        tuned.load_state_dict(torch.load(CKPT, map_location=dev))
        log.info(f"loaded fine-tuned checkpoint {CKPT}")
        out["train_log"] = load_results("diffing_finetune").get("train_log") if (CHAPTER_DIR / "results" / "diffing_finetune.json").exists() else None
    else:
        tuned.train()
        opt = torch.optim.AdamW(tuned.parameters(), lr=LR)
        losses = []
        g = torch.Generator().manual_seed(SEED)
        order = torch.randperm(len(train_pos), generator=g).tolist()
        for step, i in enumerate(range(0, len(order), BATCH)):
            batch = [train_pos[j] for j in order[i:i + BATCH]]
            ids = [tok.encode(s)[:MAX_LEN] for s in batch]
            L = max(len(x) for x in ids)
            inp = torch.full((len(ids), L), tok.eos_token_id)
            mask = torch.zeros((len(ids), L), dtype=torch.long)
            for r, x in enumerate(ids):
                inp[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
            inp, mask = inp.to(dev), mask.to(dev)
            labels = inp.masked_fill(mask == 0, -100)
            loss = tuned(inp, attention_mask=mask, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
            if step % 10 == 0:
                log.info(f"  step {step}: loss {float(loss):.3f}")
        tuned.eval()
        CKPT.parent.mkdir(exist_ok=True)
        (CKPT.parent / ".gitignore").write_text("*\n", encoding="utf-8")
        torch.save(tuned.state_dict(), CKPT)
        out["train_log"] = {"losses": losses, "steps": len(losses)}
        log.info(f"fine-tuned {len(losses)} steps; loss {losses[0]:.3f} -> {np.mean(losses[-5:]):.3f}; saved {CKPT}")

    after = {dom: positivity(tuned, tok, ps) for dom, ps in PROMPTS.items()}
    out["behaviour"] = {dom: {"before": before[dom], "after": after[dom]} for dom in PROMPTS}
    for dom in PROMPTS:
        log.info(f"  positivity {dom:10s}: {np.mean(before[dom]):+.2f} -> {np.mean(after[dom]):+.2f}")

    # ------------------------------------------------------------------ 3. weights
    wchange = []
    for layer in range(len(base.transformer.h)):
        num = sum(float((pt - pb).norm() ** 2) for pb, pt in zip(base.transformer.h[layer].parameters(),
                                                                tuned.transformer.h[layer].parameters()))
        den = sum(float(pb.norm() ** 2) for pb in base.transformer.h[layer].parameters())
        wchange.append((num / den) ** 0.5)
    out["weight_change_by_block"] = wchange
    log.info(f"relative weight change by block: {np.round(np.array(wchange) * 1e3, 2).tolist()} (x1e-3)")

    # ------------------------------------------------------------------ 4. activations
    neutral = [p for dom, ps in PROMPTS.items() if dom != "movies" for p in ps]
    ab, at = last_resid(base, tok, neutral), last_resid(tuned, tok, neutral)
    diff = (at - ab).mean(1)                                   # (n_layers+1, d)
    g = np.random.default_rng(SEED)
    cos, cos_rand, shift_norm = [], [], []
    for layer in range(diff.shape[0]):
        v = diff[layer]
        n = np.linalg.norm(v)
        shift_norm.append(float(n / np.linalg.norm(ab[layer], axis=1).mean()))
        cos.append(float(v @ dirs[layer] / (n + 1e-9)))
        r = g.standard_normal((200, v.shape[0])); r /= np.linalg.norm(r, axis=1, keepdims=True)
        cos_rand.append(float(np.abs(r @ v / (n + 1e-9)).mean()))
    out["activation_diff"] = {"cos_with_sentiment": cos, "mean_abs_cos_random": cos_rand,
                              "relative_shift": shift_norm}
    log.info(f"cos(activation shift, sentiment direction) by layer: {np.round(cos, 2).tolist()}")
    log.info(f"random-direction control: {np.round(cos_rand, 3).tolist()}")

    # ------------------------------------------------------------------ 5. causality: steer / remove
    d = torch.tensor(dirs[STEER_LAYER + 1], dtype=torch.float32, device=dev)

    def steer(alpha):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * resid_norm * d
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def remove(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h - (h @ d)[..., None] * d[None, None, :] + (float(np.mean(ab[STEER_LAYER + 1] @ dirs[STEER_LAYER + 1]))) * d
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    steer_rows = []
    for a in STEER_ALPHAS:
        s = positivity(base, tok, neutral, hook=steer(a) if a else None)
        steer_rows.append({"alpha": a, "positivity": float(np.mean(s))})
        log.info(f"  steer base by {a} x |resid|: positivity {np.mean(s):+.2f}")
    removed = positivity(tuned, tok, neutral, hook=remove)

    # the same pinning at later layers, where the shift lines up better with the direction
    pinned = {}
    for L in PIN_LAYERS:
        dL = torch.tensor(dirs[L + 1], dtype=torch.float32, device=dev)
        base_proj = float(np.mean(ab[L + 1] @ dirs[L + 1]))

        def pin(module, inp, output, dL=dL, base_proj=base_proj):
            h = output[0] if isinstance(output, tuple) else output
            h = h - (h @ dL)[..., None] * dL[None, None, :] + base_proj * dL
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        handle = tuned.transformer.h[L].register_forward_hook(pin)
        try:
            pinned[str(L)] = float(np.mean(positivity(tuned, tok, neutral)))
        finally:
            handle.remove()
        log.info(f"  tuned, direction pinned at layer {L}: positivity {pinned[str(L)]:+.2f}")
    out["steering"] = {"rows": steer_rows,
                       "tuned_positivity": float(np.mean([x for dom in PROMPTS if dom != "movies" for x in after[dom]])),
                       "base_positivity": float(np.mean([x for dom in PROMPTS if dom != "movies" for x in before[dom]])),
                       "tuned_with_direction_removed": float(np.mean(removed)),
                       "tuned_pinned_by_layer": pinned}
    log.info(f"  tuned {out['steering']['tuned_positivity']:+.2f}, tuned with the direction pinned to its base "
             f"value {out['steering']['tuned_with_direction_removed']:+.2f}, base {out['steering']['base_positivity']:+.2f}")
    return out


# --------------------------------------------------------------------------- figures
def fig_behaviour(res: dict) -> None:
    b = res["behaviour"]
    doms = list(b)
    before = [np.mean(b[d]["before"]) for d in doms]
    after = [np.mean(b[d]["after"]) for d in doms]
    # Figures are drawn at the size they get on the slide (text width ~5.5in), so fonts stay 7-9pt.
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    x = np.arange(len(doms))
    b1 = ax.bar(x - 0.2, before, 0.4, color=GREY, label="before fine-tuning")
    b2 = ax.bar(x + 0.2, after, 0.4, color=RED, label="after (positive movie reviews only)")
    ax.bar_label(b1, labels=[f"{v:+.1f}" for v in before], fontsize=7, padding=1)
    ax.bar_label(b2, labels=[f"{v:+.1f}" for v in after], fontsize=7, padding=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("positivity gap\n(log-odds, next word)", fontsize=8)
    ax.set_ylim(min(0, min(before)) - 0.3, max(after) * 1.35)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diffing_behaviour.pdf", bbox_inches="tight"); plt.close(fig)


def fig_layers(res: dict) -> None:
    w = res["weight_change_by_block"]
    a = res["activation_diff"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.1))
    bars = axes[0].bar(range(len(w)), np.array(w) * 1e3, color=ORANGE)
    axes[0].bar_label(bars, labels=[f"{v * 1e3:.1f}" for v in w], fontsize=6, padding=1, rotation=90)
    axes[0].set_xlabel("block", fontsize=8)
    axes[0].set_ylabel("relative weight change\n(x 0.001)", fontsize=8)
    axes[0].set_xticks(range(0, len(w), 2)); axes[0].set_ylim(0, max(w) * 1e3 * 1.25)
    layers = np.arange(len(a["cos_with_sentiment"]))
    axes[1].plot(layers, a["cos_with_sentiment"], "-o", color=RED, ms=2.5, label="sentiment direction")
    axes[1].plot(layers, a["mean_abs_cos_random"], "--", color=GREY, label="random directions")
    axes[1].set_xlabel("layer (12 = final)", fontsize=8)
    axes[1].set_ylabel("cosine with the\nactivation shift", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diffing_layers.pdf", bbox_inches="tight"); plt.close(fig)


def fig_steer(res: dict) -> None:
    s = res["steering"]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    al = [r["alpha"] for r in s["rows"]]
    ax.plot(al, [r["positivity"] for r in s["rows"]], "-o", color=BLUE, ms=3,
            label=f"original, steered at layer {res['direction']['steer_layer']}")
    ax.axhline(s["tuned_positivity"], color=RED, ls="--", lw=1.2, label="fine-tuned")
    ax.set_xlabel("steering strength\n(x the stream's own size)", fontsize=7.5)
    ax.set_ylabel("positivity gap", fontsize=8)
    ax.set_title("adding the direction", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    pins = s["tuned_pinned_by_layer"]
    labels = ["original", "fine-\ntuned"] + [f"pinned\nat L{k}" for k in pins]
    vals = [s["base_positivity"], s["tuned_positivity"]] + [pins[k] for k in pins]
    colors = [GREY, RED] + [ORANGE] * len(pins)
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.1f}" for v in vals], fontsize=7, padding=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, max(vals) * 1.2); ax.tick_params(axis="y", labelsize=7)
    ax.set_title("pinning it, fine-tuned model", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diffing_steer.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("diffing_finetune")
    if not args.plot_only:
        save_results("diffing_finetune", run(log), log)
    res = load_results("diffing_finetune")
    fig_behaviour(res); fig_layers(res); fig_steer(res)
    log.info("wrote diffing_behaviour / layers / steer")


if __name__ == "__main__":
    main()
