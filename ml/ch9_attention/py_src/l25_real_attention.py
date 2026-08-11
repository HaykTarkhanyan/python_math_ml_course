"""Real attention patterns from a real transformer, for L25's multi-head section.

Rather than borrow a published attention picture, this pulls the actual attention weights out
of a small pretrained encoder and plots them. Everything on the resulting slides is therefore
reproducible from this repo.

The sentence is the 3Blue1Brown running example L24 already uses, so the chapter keeps one
example from the single-head derivation through to multi-head.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch9_attention/py_src/l25_real_attention.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
SENTENCE = "a fluffy blue creature roamed the verdant forest"

ARM_BLUE = "#0033A0"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l25_real_attention.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 10, "figure.dpi": 140})


def head_entropy(a):
    """Low entropy = this head concentrates on a few tokens; high = it spreads out."""
    p = np.clip(a, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=-1).mean())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    enc = tok(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    pretty = [t.replace("▁", "") or t for t in tokens]
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    log.info("%s: %d layers x %d heads, %d tokens: %s",
             MODEL_NAME, n_layers, n_heads, len(tokens), pretty)

    # Pick a middle layer - early layers are mostly positional, late ones task-specific.
    layer = n_layers // 2
    attn = out.attentions[layer][0].numpy()          # (heads, n, n)

    ent = [head_entropy(attn[h]) for h in range(n_heads)]
    order = np.argsort(ent)
    sharp, diffuse = int(order[0]), int(order[-1])
    log.info("layer %d: sharpest head %d (entropy %.3f), most diffuse head %d (entropy %.3f)",
             layer, sharp, ent[sharp], diffuse, ent[diffuse])
    for h in range(n_heads):
        # Where does each head send most of its attention, on average?
        off = float(np.mean([np.argmax(attn[h][i]) - i for i in range(len(tokens))]))
        log.info("  head %2d entropy %.3f  mean offset of its strongest key %+.2f", h, ent[h], off)

    if ent[sharp] >= ent[diffuse]:
        raise ValueError("entropy ordering is broken")

    show = [sharp, diffuse]
    fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.0))
    for ax, h in zip(np.atleast_1d(axes), show):
        im = ax.imshow(attn[h], cmap="Blues", vmin=0, vmax=attn[h].max())
        ax.set_xticks(range(len(pretty))); ax.set_yticks(range(len(pretty)))
        ax.set_xticklabels(pretty, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(pretty, fontsize=7.5)
        kind = "concentrates" if h == sharp else "spreads out"
        ax.set_title(f"layer {layer}, head {h} - {kind}\n(entropy {ent[h]:.2f})", fontsize=9.5)
        ax.set_xlabel("attends to", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    np.atleast_1d(axes)[0].set_ylabel("token doing the attending", fontsize=8.5)
    fig.suptitle(f'Two heads in the same layer, same sentence - real weights from {MODEL_NAME}',
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    out_path = FIG / "l25_real_heads.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    log.info("wrote %s", out_path)

    # A compact per-head entropy chart: heads genuinely differ, and that is the whole point.
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.bar(range(n_heads), ent, color=[ARM_BLUE if h not in show else "#D90012"
                                              for h in range(n_heads)])
    ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
    ax.set_xlabel(f"head index (layer {layer})")
    ax.set_ylabel("attention entropy\n(low = focused)")
    ax.set_title("The heads in one layer do measurably different things", fontsize=10.5)
    ax.set_xticks(range(n_heads))
    ax.set_ylim(0, max(ent) * 1.2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / "l25_head_entropy.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", FIG / "l25_head_entropy.pdf")
    log.info("SLIDE NUMBERS: layer %d, sharpest head %d entropy %.2f, "
             "most diffuse head %d entropy %.2f, %d heads total",
             layer, sharp, ent[sharp], diffuse, ent[diffuse], n_heads)


if __name__ == "__main__":
    main()
