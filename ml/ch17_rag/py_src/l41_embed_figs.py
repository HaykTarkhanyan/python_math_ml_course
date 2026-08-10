"""Embedding figures for L41 (RAG retrieval): figures 08-10 and 13.

Everything here is MEASURED, not drawn:

  08  real intfloat/multilingual-e5-small embeddings of the cheese-factory corpus, PCA to 2-D
  09  a real InfoNCE training run on toy data (tiny MLP, CPU, seconds) - before vs after
  10  a real in-batch similarity matrix from e5-small
  13  real BM25 ranks vs real dense ranks for the three deck queries

e5-small is used because it is multilingual (so the Armenian section can reuse it) and it
carries the query:/passage: prefix convention the deck teaches.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_embed_figs.py

USE_TF=0 is required: TensorFlow is present in the `ma` venv and transformers refuses to
import its TF integration under Keras 3.
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
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS, CHUNK_LABELS, QUERIES, tokenize

SEED = 509
MODEL_NAME = "intfloat/multilingual-e5-small"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "l41_embed_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


# --- BM25 over the actual 14-chunk corpus ----------------------------------------------
# The deck derives Robertson's raw idf. With only 14 documents that form goes negative for
# common terms, so corpus-wide scoring here uses the standard Lucene variant
# ln(1 + (N-n+0.5)/(n+0.5)), which is the form real implementations ship. Noted on the slide.
def bm25_ranks(query, docs_tokens, k1=1.5, b=0.75):
    n_docs = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / n_docs
    scores = []
    for toks in docs_tokens:
        dl = len(toks)
        norm = (1 - b) + b * dl / avgdl
        s = 0.0
        for term in tokenize(query):
            tf = toks.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in docs_tokens if term in d)
            idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            s += tf / (k1 * norm + tf) * idf
        scores.append(s)
    return np.array(scores)


def rank_of(scores, target):
    """1-based rank of `target` when sorting scores descending."""
    return int((np.argsort(-scores) == target).nonzero()[0][0]) + 1


# --- figure 08 -------------------------------------------------------------------------
def fig_embedding_space(model):
    """Real e5-small embeddings of all 14 chunks plus the 3 queries, PCA to 2-D."""
    chunk_vecs = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True)
    q_texts = [q["text"] for q in QUERIES.values()]
    q_vecs = model.encode([f"query: {t}" for t in q_texts], normalize_embeddings=True)

    allv = np.vstack([chunk_vecs, q_vecs])
    centred = allv - allv.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt[:2].T
    cp, qp = proj[:len(CHUNKS)], proj[len(CHUNKS):]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(cp[:, 0], cp[:, 1], s=70, color=ARM_BLUE, alpha=0.75, zorder=3,
               label="indexed chunks")

    # Label only the chunks the story needs. Labelling all 18 produced three separate
    # collisions (flagged in student review, 2026-08-10) and taught nothing extra: the
    # figure's job is to show queries landing near their answers, not to enumerate a corpus.
    LABELLED = {
        0: (0, 10),      # press 2.5 bar   - answer to "direct"
        9: (0, -16),     # ripening cellar - answer to "paraphrase"
        8: (0, 10),      # PRS-400 model   - answer to "identifier"
        2: (-6, 10),     # brining
        11: (4, -16),    # vacuum packing
    }
    for ix, (dx, dy) in LABELLED.items():
        ax.annotate(CHUNK_LABELS[ix], (cp[ix, 0], cp[ix, 1]), fontsize=8,
                    color="#333333", fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points", ha="center")

    ax.scatter(qp[:, 0], qp[:, 1], s=150, color=ARM_RED, marker="*", zorder=4,
               label="queries")
    for (x, y), key in zip(qp, QUERIES):
        ax.annotate(key, (x, y), fontsize=8.5, color=ARM_RED, fontweight="bold",
                    xytext=(0, -14), textcoords="offset points", ha="center")
        tgt = QUERIES[key]["target"]
        ax.plot([x, cp[tgt, 0]], [y, cp[tgt, 1]], color=ARM_RED, lw=0.9, ls="--",
                alpha=0.5, zorder=2)

    ax.set_title(f"The corpus as vectors ({MODEL_NAME}, PCA to 2-D)", fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=9, loc="best")
    save(fig, "08_embedding_space")


# --- figure 09 -------------------------------------------------------------------------
class ToyEncoder(nn.Module):
    def __init__(self, d_in=8, d_out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def fig_contrastive(n_items=12, steps=600, temperature=0.1):
    """A real InfoNCE run: pairs start scattered, end colocated. Tiny MLP, CPU, seconds."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    latent = rng.normal(size=(n_items, 8))
    noise = 0.55
    queries = torch.tensor(latent + noise * rng.normal(size=latent.shape), dtype=torch.float32)
    passages = torch.tensor(latent + noise * rng.normal(size=latent.shape), dtype=torch.float32)

    enc = ToyEncoder()
    with torch.no_grad():
        before_q, before_p = enc(queries).numpy(), enc(passages).numpy()

    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    labels = torch.arange(n_items)
    for step in range(steps):
        opt.zero_grad()
        zq, zp = enc(queries), enc(passages)
        logits = zq @ zp.T / temperature
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            log.info("  InfoNCE step %3d  loss %.4f", step, loss.item())
    log.info("  InfoNCE final loss %.4f", loss.item())

    with torch.no_grad():
        after_q, after_p = enc(queries).numpy(), enc(passages).numpy()

    def pair_gap(zq, zp):
        sim = zq @ zp.T
        pos = np.diag(sim).mean()
        neg = (sim.sum() - np.trace(sim)) / (len(sim) ** 2 - len(sim))
        return pos, neg

    pb, nb = pair_gap(before_q, before_p)
    pa, na = pair_gap(after_q, after_p)
    log.info("  before: positive %.3f  negative %.3f  gap %.3f", pb, nb, pb - nb)
    log.info("  after : positive %.3f  negative %.3f  gap %.3f", pa, na, pa - na)
    if (pa - na) <= (pb - nb):
        raise ValueError("contrastive demo failed: training did not widen the positive/negative gap")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, (zq, zp), title, gap in [
        (axes[0], (before_q, before_p), "Before training", pb - nb),
        (axes[1], (after_q, after_p), "After InfoNCE", pa - na),
    ]:
        for i in range(n_items):
            ax.plot([zq[i, 0], zp[i, 0]], [zq[i, 1], zp[i, 1]], color=GREY, lw=0.8, alpha=0.55, zorder=1)
        ax.scatter(zq[:, 0], zq[:, 1], s=55, color=ARM_RED, zorder=3, label="query")
        ax.scatter(zp[:, 0], zp[:, 1], s=55, color=ARM_BLUE, marker="s", zorder=3, label="its passage")
        circ = plt.Circle((0, 0), 1.0, fill=False, color=GREY, lw=0.8, ls=":")
        ax.add_patch(circ)
        ax.set_title(f"{title}\npositive - negative similarity = {gap:+.2f}", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "09_contrastive_before_after")


# --- figure 10 -------------------------------------------------------------------------
def fig_in_batch_negatives(model, n=8):
    """Real e5-small similarity matrix for a batch of (query, passage) pairs."""
    pairs = [
        ("what pressure for the Lori press?", CHUNKS[0]),
        ("how hot is pasteurisation?", CHUNKS[3]),
        ("when is the starter culture added?", CHUNKS[4]),
        ("how long does rennet take?", CHUNKS[5]),
        ("how big are the curd cubes?", CHUNKS[6]),
        ("what temperature is the ripening cellar?", CHUNKS[9]),
        ("what protective equipment is required?", CHUNKS[10]),
        ("how are the wheels packed?", CHUNKS[11]),
    ][:n]

    qv = model.encode([f"query: {q}" for q, _ in pairs], normalize_embeddings=True)
    pv = model.encode([f"passage: {p}" for _, p in pairs], normalize_embeddings=True)
    sim = qv @ pv.T

    diag = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    correct = int((sim.argmax(axis=1) == np.arange(n)).sum())
    log.info("  in-batch: diagonal mean %.3f, off-diagonal mean %.3f, argmax correct %d/%d",
             diag.mean(), off.mean(), correct, n)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(sim, cmap="Blues", vmin=off.min(), vmax=diag.max())
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if sim[i, j] > (off.min() + diag.max()) / 2 else "#333333",
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=ARM_RED, lw=2.0))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"P{i+1}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"Q{i+1}" for i in range(n)], fontsize=9)
    ax.set_xlabel("passages in the batch")
    ax.set_ylabel("queries in the batch")
    ax.set_title("In-batch negatives: the diagonal is the only correct answer\n"
                 f"(real {MODEL_NAME} similarities)", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "10_in_batch_negatives")
    return diag.mean(), off.mean(), correct


# --- figure 13 -------------------------------------------------------------------------
def fig_where_each_dies(model):
    """Rank AND score of the correct chunk under BM25 vs dense, for the deck's queries.

    Reports the score, not just the rank: BM25's rank is meaningless when its score is 0,
    because that is arbitrary tie-breaking rather than retrieval. The first version of this
    figure hid exactly that, and told a story the numbers did not support.
    """
    docs_tokens = [tokenize(c) for c in CHUNKS]
    chunk_vecs = model.encode([f"passage: {c}" for c in CHUNKS], normalize_embeddings=True)

    keys, bm_ranks, dn_ranks, bm_scores = [], [], [], []
    for key, q in QUERIES.items():
        bs = bm25_ranks(q["text"], docs_tokens)
        qv = model.encode([f"query: {q['text']}"], normalize_embeddings=True)[0]
        ds = chunk_vecs @ qv
        keys.append(key)
        bm_ranks.append(rank_of(bs, q["target"]))
        dn_ranks.append(rank_of(ds, q["target"]))
        bm_scores.append(bs[q["target"]])
        log.info("  %-11s BM25 rank %2d (score %.3f) | dense rank %2d (score %.3f) | %s",
                 key, bm_ranks[-1], bs[q["target"]], dn_ranks[-1], ds[q["target"]], q["note"])

    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    b1 = ax.bar(x - w / 2, bm_ranks, w, color=ARM_ORANGE, label="BM25 (keywords)")
    b2 = ax.bar(x + w / 2, dn_ranks, w, color=ARM_BLUE, label="dense (embeddings)")
    ax.bar_label(b1, fmt="%d", fontsize=10, fontweight="bold", padding=2)
    ax.bar_label(b2, fmt="%d", fontsize=10, fontweight="bold", padding=2)

    # Mark the structural failure: a BM25 score of zero is not a bad rank, it is no signal.
    for xi, (r, s) in enumerate(zip(bm_ranks, bm_scores)):
        if s < 1e-9:
            ax.annotate("BM25 score = 0.00\nno signal at all",
                        xy=(xi - w / 2, r), xytext=(xi - w / 2 - 0.32, r + 3.0),
                        fontsize=8.5, color=ARM_RED, fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))

    ax.axhline(1, color=GREY, lw=0.9, ls=":")
    ax.text(len(keys) - 0.45, 1.3, "perfect", fontsize=8.5, color=GREY, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "direct question\n(words overlap)",
        "paraphrase\n(no words shared)",
        'exact identifier\n"PRS-400"',
    ], fontsize=9)
    ax.set_ylabel("rank of the correct chunk\n(1 = found it, lower is better)")
    ax.set_ylim(0, max(bm_ranks + dn_ranks) * 1.42)
    ax.set_title("BM25 does not degrade - it stops working entirely", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "13_where_each_dies")
    return list(zip(keys, bm_ranks, dn_ranks, bm_scores))


def main():
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("  dim=%d", model.get_sentence_embedding_dimension())

    fig_embedding_space(model)
    fig_contrastive()
    fig_in_batch_negatives(model)
    fig_where_each_dies(model)
    log.info("done: 4 figures")


if __name__ == "__main__":
    main()
