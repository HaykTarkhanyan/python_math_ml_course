"""Figures for "Opening the box" (02_opening_the_box.tex), formerly L45.

One model load measures everything into ``results/l45_figs.json``; the figures are drawn from that
file and nothing else (``--plot-only`` redraws them without loading GPT-2):

* ``logit_lens.pdf``         - the prediction being built layer by layer, with the jump in layer 9;
* ``probe_by_layer.pdf``     - probe accuracy for "is a name repeated?", against a chance baseline;
* ``attention_pattern.pdf``  - one head's full attention pattern, to teach how to read one;
* ``attention_last_row.pdf`` - four heads, only the row that predicts the answer;
* ``induction_scores.pdf``   - induction score for all 144 heads on repeated random tokens.

The logit-lens figure is the important one. It shows the logit difference sitting near 0 for nine
layers and jumping in layer 9 - which is exactly where the next deck finds the name-mover heads.
The two decks agree on where the work happens without either asserting it.

Every figure is drawn at the size it occupies on the slide (16:9 text width ~5.5 in), so the
font sizes here are the font sizes students see (_learnings/2026-10-01-2317).

Usage (about 1 minute on CPU with GPT-2 small cached; --plot-only takes seconds):
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_figs.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mi_common import FIG_DIR, load_results, save_results, setup_logging, stream_ticks

# Armenian flag palette, per the repo's charting rule.
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

N_PROBE = 128
N_INDUCTION_SEQ = 8
INDUCTION_SEQ_LEN = 40
HEADS = [(9, 9), (9, 6), (9, 8), (10, 7)]
PATTERN_HEAD = "L9H9"

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
SAVE = dict(bbox_inches="tight", metadata={"CreationDate": None})


# ---------------------------------------------------------------- measurement (needs the model)

def measure_logit_lens(model, data, directions, log) -> dict:
    """Logit difference as a function of how many layers we let run."""
    import torch
    _, cache = model.run_with_cache(data["clean"])
    accumulated, labels = cache.accumulated_resid(
        layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
    )
    scaled = cache.apply_ln_to_stack(accumulated, layer=-1, pos_slice=-1)
    values = (torch.einsum("cbd,bd->c", scaled, directions) / directions.shape[0]).tolist()
    jump = int(np.argmax(np.diff(values))) + 1
    log.info(f"logit lens: final {values[-1]:+.3f}, biggest jump entering {labels[jump]}")
    return {"labels": labels, "values": values, "biggest_jump_layer": jump - 1}


def measure_probe(model, log) -> dict:
    """Can a linear probe read 'is a name repeated?' off the residual stream?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from ioi_core import DATA_PATH, SEED

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

    _, cache = model.run_with_cache(model.to_tokens([s["text"] for s in group]))
    accuracies = []
    t0 = time.time()
    for layer in range(model.cfg.n_layers):
        activations = cache["resid_post", layer][:, -1, :].numpy()
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        scores = cross_val_score(clf, activations, labels, cv=5, scoring="accuracy")
        accuracies.append(float(scores.mean()))
        elapsed = time.time() - t0
        log.info(f"  layer {layer:2d}: probe accuracy {scores.mean():.3f} "
                 f"({elapsed:.0f}s elapsed, ~{elapsed / (layer + 1) * (11 - layer):.0f}s left)")
    return {"accuracies": accuracies, "n_prompts": len(group)}


def measure_attention(model, data, directions, log) -> dict:
    """Attention patterns of four named heads on the running-example sentence, plus each head's
    direct logit attribution on that same sentence (so a caption can put what a head LOOKS at
    next to what it WRITES, measured on one input)."""
    import torch
    tokens = data["clean"][:1]
    str_tokens = model.to_str_tokens(tokens[0])
    _, cache = model.run_with_cache(tokens)

    heads = {}
    stack, head_labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)
    dla = torch.einsum("cbd,bd->c", stack, directions[:1])
    for layer, head in HEADS:
        name = f"L{layer}H{head}"
        pattern = cache["pattern", layer][0, head].numpy()
        heads[name] = {
            "pattern": pattern.round(5).tolist(),
            "dla_this_sentence": float(dla[head_labels.index(name)]),
        }
        log.info(f"  {name}: last-row max on {str_tokens[int(pattern[-1].argmax())]!r} "
                 f"({pattern[-1].max():.2f}), DLA on this sentence {heads[name]['dla_this_sentence']:+.2f}")
    return {"tokens": str_tokens, "heads": heads}


def measure_induction(model, log) -> dict:
    """Induction score per head: attention to the token after a previous copy of yourself."""
    import torch
    from ioi_core import SEED
    generator = torch.Generator().manual_seed(SEED)
    random_tokens = torch.randint(1000, 10000, (N_INDUCTION_SEQ, INDUCTION_SEQ_LEN), generator=generator)
    bos = torch.full((N_INDUCTION_SEQ, 1), model.tokenizer.bos_token_id)
    _, cache = model.run_with_cache(torch.cat([bos, random_tokens, random_tokens], dim=1))

    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer]  # [batch, head, query, key]
        # A repeated-sequence induction head attends from position i in the second copy back to
        # the token that FOLLOWED the earlier occurrence: offset seq_len - 1 on the diagonal.
        diagonal = pattern.diagonal(offset=-(INDUCTION_SEQ_LEN - 1), dim1=-2, dim2=-1)
        scores[layer] = diagonal.mean(dim=(0, 2))

    top = torch.topk(scores.flatten(), 3)
    found = []
    for i, value in zip(top.indices, top.values):
        layer, head = int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads
        found.append({"head": f"L{layer}H{head}", "score": value.item()})
        log.info(f"  induction head L{layer}H{head}: {value.item():.3f}")
    return {"top_heads": found, "seq_len": INDUCTION_SEQ_LEN, "n_seqs": N_INDUCTION_SEQ,
            "scores": scores.numpy().round(5).tolist()}


def run(log) -> dict:
    from ioi_core import load_model, load_prompts, logit_diff, logit_diff_directions
    model = load_model(log)
    data = load_prompts(model, log, n=64)
    directions = logit_diff_directions(model, data["answer_tokens"])
    log.info(f"baseline logit diff on this batch: {logit_diff(model(data['clean']), data['answer_tokens']):.3f}")
    return {
        "logit_lens": measure_logit_lens(model, data, directions, log),
        "attention_heads": measure_attention(model, data, directions, log),
        "induction": measure_induction(model, log),
        "probe": measure_probe(model, log),
    }


# ---------------------------------------------------------------- figures (JSON only)

def fig_logit_lens(res: dict, fig_dir, log) -> None:
    lens = res["logit_lens"]
    values = np.array(lens["values"])
    ticks = stream_ticks(lens["labels"])
    x = np.arange(len(values))
    # values[k] is the stream after layer k-1 (index 0 = embedding only).
    before, after = values[ticks.index("8")], values[ticks.index("9")]
    jump = int(np.argmax(np.diff(values))) + 1
    if ticks[jump] != "9":
        raise RuntimeError(f"the slide says the jump is in layer 9; the data says {ticks[jump]}")

    fig, ax = plt.subplots(figsize=(4.8, 1.95))
    ax.plot(x, values, marker="o", ms=3.5, color=BLUE, lw=1.8, zorder=3)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.annotate(f"layer 9 alone: {after - before:+.1f}", xy=(jump - 0.5, (before + after) / 2),
                xytext=(jump - 4.6, (before + after) / 2 + 0.6), color=RED, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("", xy=(0, before + 0.25), xytext=(jump - 1, before + 0.25),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
    ax.text((jump - 1) / 2, before + 0.38, f"layers 0-8 together: {before - values[0]:+.1f}",
            ha="center", va="bottom", fontsize=8, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_xlabel("residual stream after layer", fontsize=8.5)
    ax.set_ylabel("logit difference\n(right name - wrong name)", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(fig_dir / "logit_lens.pdf", **SAVE)
    plt.close(fig)
    log.info(f"logit lens: layers 0-8 {before - values[0]:+.3f}, layer 9 {after - before:+.3f}, "
             f"after layer 9 {after:+.3f}, final {values[-1]:+.3f}")


def fig_probe_by_layer(res: dict, fig_dir, log) -> None:
    acc = res["probe"]["accuracies"]
    fig, ax = plt.subplots(figsize=(4.6, 2.15))
    x = np.arange(len(acc))
    bars = ax.bar(x, acc, color=BLUE, zorder=3, width=0.7)
    ax.bar_label(bars, labels=[f"{a:.2f}" for a in acc], fontsize=7, padding=1.5)
    ax.axhline(0.5, color=RED, ls="--", lw=1.1, label="chance")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("layer the probe reads (the stream after that layer)", fontsize=8.5)
    ax.set_ylabel("held-out accuracy", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.02))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(fig_dir / "probe_by_layer.pdf", **SAVE)
    plt.close(fig)
    first = next(i for i, a in enumerate(acc) if a >= 0.999)
    log.info(f"probe: first 100% at layer {first}; min after that {min(acc[first:]):.3f}")


def _short(tokens: list[str]) -> list[str]:
    return ["BOS" if t == "<|endoftext|>" else (t.strip() or "_") for t in tokens]


def fig_attention_pattern(res: dict, fig_dir, log) -> None:
    """One head, the whole pattern: how to read an attention heatmap."""
    att = res["attention_heads"]
    tokens = _short(att["tokens"])
    pattern = np.array(att["heads"][PATTERN_HEAD]["pattern"])
    fig, ax = plt.subplots(figsize=(3.05, 2.75))
    im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("looks at (key)", fontsize=8)
    ax.set_ylabel("from (query)", fontsize=8)
    ax.set_title(PATTERN_HEAD, fontsize=9)
    # Frame the last row (the position that predicts the answer), drawn just inside the cells so
    # the axes edge does not hide its bottom and sides.
    ax.add_patch(plt.Rectangle((-0.4, len(tokens) - 1.42), len(tokens) - 0.2, 0.84, fill=False,
                               edgecolor=RED, lw=1.3, clip_on=False, zorder=5))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(fig_dir / "attention_pattern.pdf", **SAVE)
    plt.close(fig)
    log.info(f"attention pattern drawn for {PATTERN_HEAD}")


def fig_attention_last_row(res: dict, fig_dir, log) -> None:
    """Four heads, one row each: where the final position ("to") looks."""
    att = res["attention_heads"]
    tokens = _short(att["tokens"])
    names = list(att["heads"])
    rows = np.array([att["heads"][n]["pattern"][-1] for n in names])
    fig, ax = plt.subplots(figsize=(5.3, 1.5))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            if rows[i, j] >= 0.05:
                ax.text(j, i, f"{rows[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if rows[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(fig_dir / "attention_last_row.pdf", **SAVE)
    plt.close(fig)
    io = att["tokens"].index(" William")
    for n, r in zip(names, rows):
        log.info(f"  {n}: {r[io]:.0%} on ' William', DLA {att['heads'][n]['dla_this_sentence']:+.2f}")


def fig_induction_scores(res: dict, fig_dir, log) -> None:
    ind = res["induction"]
    scores = np.array(ind["scores"])
    fig, ax = plt.subplots(figsize=(4.3, 2.15))
    im = ax.imshow(scores, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xlabel("head", fontsize=8.5)
    ax.set_ylabel("layer", fontsize=8.5)
    ax.set_xticks(range(scores.shape[1]))
    ax.set_yticks(range(scores.shape[0]))
    ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("induction score", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for t in ind["top_heads"]:
        layer, head = (int(v) for v in t["head"][1:].split("H"))
        ax.text(head, layer, "*", ha="center", va="center", color=ORANGE, fontsize=13,
                fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(fig_dir / "induction_scores.pdf", **SAVE)
    plt.close(fig)
    log.info(f"induction: top {[(t['head'], round(t['score'], 3)) for t in ind['top_heads']]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("l45_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.plot_only:
        save_results("l45_figs", run(log), log)
    res = load_results("l45_figs")
    for fig_fn in (fig_logit_lens, fig_probe_by_layer, fig_attention_pattern,
                   fig_attention_last_row, fig_induction_scores):
        fig_fn(res, FIG_DIR, log)
    log.info(f"5 figures written to {FIG_DIR} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
