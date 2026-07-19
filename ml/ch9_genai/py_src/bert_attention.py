"""DistilBERT self-attention on the chapter demo sentence, for the GenAI attention decks.

Generates into ml/ch9_genai/fig/:
  bert_attention_donkey.pdf  -- one clear head, CONTENT TOKENS ONLY (tokens x tokens heatmap)
  bert_heads_grid.pdf        -- content-token attention of every head in a mid layer (for L25)

BERT heads dump most of their mass on [SEP]/[CLS] (a well-documented artifact), which hides the
linguistic pattern -- so we drop those tokens and renormalise each query row over content tokens.

Real attention weights from distilbert-base-uncased (one-time ~260 MB download, CPU inference).
Run: ./ma/Scripts/python.exe ml/ch9_genai/py_src/bert_attention.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
SENT = "Donkey don't die, spring is coming"

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
LOGS = ROOT / "logs"


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("bert_attention")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS / "bert_attention.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


def main():
    log = setup_logging()
    FIG.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModel

    torch.manual_seed(SEED)
    name = "distilbert-base-uncased"
    log.info(f"loading {name} (one-time download if not cached)")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_attentions=True)
    model.eval()

    enc = tok(SENT, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    atts = out.attentions  # tuple(n_layers) of (1, heads, seq, seq)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    n_layers = len(atts)
    n_heads = atts[0].shape[1]
    seq = len(toks)
    log.info(f"tokens: {toks}")
    log.info(f"{n_layers} layers x {n_heads} heads, seq={seq}")

    # drop [CLS] (0) and [SEP] (last); renormalise each query row over the remaining content tokens
    content = list(range(1, seq - 1))
    ctoks = [toks[i] for i in content]

    def content_mat(M):
        C = M[np.ix_(content, content)].astype(float).copy()
        C = C / C.sum(1, keepdims=True)
        return C

    # pick the head whose content attention most clearly points each word at ANOTHER word
    best = None
    for L in range(n_layers):
        A = atts[L][0].numpy()
        for H in range(n_heads):
            C = content_mat(A[H])
            off = C.copy()
            np.fill_diagonal(off, 0.0)
            score = float(off.max(1).mean())  # avg strongest off-diagonal weight per query
            if best is None or score > best[0]:
                best = (score, L, H, C)
    _, L, H, C = best
    log.info(f"chosen: layer {L + 1}, head {H + 1} (content tokens only)")

    # ---- single clear head ----
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(C, cmap="Blues", vmin=0.0, vmax=float(C.max()))
    ax.set_xticks(range(len(ctoks)))
    ax.set_xticklabels(ctoks, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(ctoks)))
    ax.set_yticklabels(ctoks, fontsize=10)
    ax.set_xlabel("attended-to token", fontsize=10)
    ax.set_ylabel("query token", fontsize=10)
    for i in range(len(ctoks)):
        for j in range(len(ctoks)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if C[i, j] > 0.33 else "#555")
    ax.set_title(f"DistilBERT self-attention  (layer {L + 1}, head {H + 1})", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Real model, our sentence  (special tokens dropped)', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "bert_attention_donkey.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")

    # ---- all heads of a mid layer (content tokens), for L25 ----
    Lg = min(4, n_layers - 1)
    A = atts[Lg][0].numpy()
    ncol = 4
    nrow = int(np.ceil(n_heads / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.7 * nrow))
    for h in range(n_heads):
        ax = axes.flat[h]
        ax.imshow(content_mat(A[h]), cmap="Blues")
        ax.set_title(f"head {h + 1}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for h in range(n_heads, nrow * ncol):
        axes.flat[h].axis("off")
    fig.suptitle(f"DistilBERT layer {Lg + 1}: every head attends differently", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out2 = FIG / "bert_heads_grid.pdf"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out2}")
    log.info("done")


if __name__ == "__main__":
    main()
