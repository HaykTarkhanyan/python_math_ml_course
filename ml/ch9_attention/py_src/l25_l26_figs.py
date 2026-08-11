"""Figures for L25 (the Transformer block) and L26 (the family, training, why it won).

REAL (computed, and the numbers on the slides come from this script's log):
  l25_permutation   self-attention really is permutation-equivariant. The same tokens in two
                    orders give byte-for-byte identical per-token outputs WITHOUT positional
                    encoding, and different ones with it. Asserted, not asserted-at.
  l25_pe_heatmap    the sinusoidal positional encoding matrix, computed from the formula.
  l25_pe_dotprod    why sinusoids work: PE similarity depends on offset, not absolute position.
  l25_causal_mask   the causal mask before and after softmax.
  l26_quadratic     attention cost vs feed-forward cost against sequence length, with the
                    crossover computed rather than eyeballed.

ILLUSTRATIVE (schematic, labelled as such on the figure):
  l25_block         the Transformer block, as a data-flow diagram.
  l26_three_arch    encoder-only / decoder-only / encoder-decoder.

Run:  ./ma/Scripts/python.exe ml/ch9_attention/py_src/l25_l26_figs.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l25_l26_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(E, Wq, Wk, Wv, mask=None):
    """One head of scaled dot-product self-attention. E is (n_tokens, d_model)."""
    Q, K, V = E @ Wq, E @ Wk, E @ Wv
    scores = Q @ K.T / np.sqrt(Wk.shape[1])
    if mask is not None:
        scores = scores + mask
    return softmax(scores) @ V, softmax(scores)


def positional_encoding(n_pos, d_model):
    """The sinusoidal encoding from Vaswani et al. (2017), computed from the formula."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# --- L25: permutation equivariance -----------------------------------------------------
def fig_permutation():
    """The predict-first payoff: without position, word order is invisible to attention."""
    rng = np.random.default_rng(SEED)
    d_model, d_k = 16, 8
    words = ["dog", "bites", "man"]
    E = {w: rng.normal(size=d_model) for w in words}
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) * 0.4 for _ in range(3))

    order_a = ["dog", "bites", "man"]
    order_b = ["man", "bites", "dog"]
    Ea = np.stack([E[w] for w in order_a])
    Eb = np.stack([E[w] for w in order_b])

    out_a, _ = self_attention(Ea, Wq, Wk, Wv)
    out_b, _ = self_attention(Eb, Wq, Wk, Wv)

    # "dog" is row 0 in order A and row 2 in order B. Compare its output vector.
    dog_a, dog_b = out_a[0], out_b[2]
    gap_nope = float(np.abs(dog_a - dog_b).max())
    log.info("WITHOUT positional encoding: max |out(dog) difference| = %.3e", gap_nope)

    pe = positional_encoding(3, d_model)
    out_a_pe, _ = self_attention(Ea + pe, Wq, Wk, Wv)
    out_b_pe, _ = self_attention(Eb + pe, Wq, Wk, Wv)
    gap_pe = float(np.abs(out_a_pe[0] - out_b_pe[2]).max())
    log.info("WITH positional encoding:    max |out(dog) difference| = %.4f", gap_pe)

    if gap_nope > 1e-9:
        raise ValueError(f"attention should be permutation-equivariant, got {gap_nope}")
    if gap_pe < 1e-3:
        raise ValueError("positional encoding failed to break the symmetry")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    x = np.arange(d_k)
    w = 0.38
    for ax, (a, b, title, gap) in zip(axes, [
        (dog_a, dog_b, "No positional encoding", gap_nope),
        (out_a_pe[0], out_b_pe[2], "With positional encoding", gap_pe),
    ]):
        ax.bar(x - w / 2, a, w, color=ARM_BLUE, label='"dog bites man"')
        ax.bar(x + w / 2, b, w, color=ARM_ORANGE, label='"man bites dog"')
        ok = gap < 1e-9
        ax.set_title(f"{title}\nmax difference = {gap:.1e}" if ok
                     else f"{title}\nmax difference = {gap:.3f}",
                     fontsize=10, color=(ARM_RED if ok else GREEN))
        ax.set_xlabel("output dimension for the word \"dog\"")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("value")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "l25_permutation")
    return gap_nope, gap_pe


# --- L25: positional encoding ----------------------------------------------------------
def fig_pe_heatmap(n_pos=60, d_model=64):
    pe = positional_encoding(n_pos, d_model)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    im = ax.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("embedding dimension")
    ax.set_title("Sinusoidal positional encoding, computed from the formula", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save(fig, "l25_pe_heatmap")


def fig_pe_dotprod(n_pos=60, d_model=64):
    """Why sinusoids: the similarity between two positions depends only on their distance."""
    pe = positional_encoding(n_pos, d_model)
    pen = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    sim = pen @ pen.T

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for p in (10, 25, 40):
        ax.plot(np.arange(n_pos) - p, sim[p], lw=2.0, label=f"position {p}")
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("offset from that position")
    ax.set_ylabel("cosine similarity\nbetween encodings")
    ax.set_title("The pattern depends on the gap, not on where you are", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)

    curves = np.stack([sim[p][max(0, p - 8):p + 9] for p in (10, 25, 40)])
    spread = float(np.abs(curves - curves.mean(axis=0)).max())
    log.info("PE similarity curves for positions 10/25/40 agree to within %.4f", spread)
    save(fig, "l25_pe_dotprod")


# --- L25: causal mask ------------------------------------------------------------------
def fig_causal_mask(n=6):
    rng = np.random.default_rng(SEED)
    scores = rng.normal(size=(n, n))
    mask = np.triu(np.full((n, n), -np.inf), k=1)
    masked = scores + mask
    attn = softmax(masked)
    leak = float(np.triu(attn, k=1).max())
    log.info("after masking, max attention paid to a future token = %.3e", leak)
    if leak > 1e-12:
        raise ValueError("causal mask leaked attention to the future")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (m, title, cmap) in zip(axes, [
        (scores, "raw scores $QK^T/\\sqrt{d_k}$", "RdBu_r"),
        (np.where(np.isinf(masked), np.nan, masked), "after adding $-\\infty$ above the diagonal", "RdBu_r"),
        (attn, "after softmax: rows sum to 1", "Blues"),
    ]):
        im = ax.imshow(m, cmap=cmap)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("key (token attended to)")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("query (token doing\nthe attending)", fontsize=9)
    fig.tight_layout()
    save(fig, "l25_causal_mask")


# --- L25: the block --------------------------------------------------------------------
def fig_block():
    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    ax.set_xlim(0, 5); ax.set_ylim(0, 5.6); ax.axis("off")

    def blk(y, label, color, h=0.62):
        ax.add_patch(FancyBboxPatch((1.15, y), 2.7, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="none"))
        ax.text(2.5, y + h / 2, label, ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")

    blk(0.25, "input tokens", GREY)
    blk(1.35, "Multi-head attention", ARM_BLUE)
    blk(2.30, "Add \\& Norm", ARM_ORANGE, h=0.5)
    blk(3.15, "Feed-forward (per token)", GREEN)
    blk(4.10, "Add \\& Norm", ARM_ORANGE, h=0.5)

    for y0, y1 in [(0.87, 1.35), (1.97, 2.30), (2.80, 3.15), (3.77, 4.10)]:
        ax.add_patch(FancyArrowPatch((2.5, y0), (2.5, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=GREY, lw=1.4))

    # the residual stream
    for y0, y1, ytarget in [(0.6, 2.55, 2.55), (2.6, 4.35, 4.35)]:
        ax.add_patch(FancyArrowPatch((1.05, y0), (1.05, y1), arrowstyle="-",
                                     color=ARM_RED, lw=1.6, linestyle="--"))
        ax.add_patch(FancyArrowPatch((1.05, ytarget), (1.13, ytarget), arrowstyle="-|>",
                                     mutation_scale=11, color=ARM_RED, lw=1.6))
    ax.text(0.42, 2.6, "residual stream", rotation=90, va="center", ha="center",
            fontsize=8.5, color=ARM_RED, fontweight="bold")

    ax.text(4.2, 2.6, r"$\times N$", fontsize=14, color=GREY, fontweight="bold")
    ax.text(2.5, 5.25, "One Transformer block", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(4.95, 0.02, "schematic", ha="right", fontsize=7, color=GREY, style="italic")
    save(fig, "l25_block")


# --- L26: the quadratic wall -----------------------------------------------------------
def fig_quadratic(d_model=4096, d_ff_mult=4):
    """Attention is O(n^2 d); the feed-forward is O(n d^2). The crossover is at n = d."""
    n = np.logspace(1, 5.7, 200)
    attn = 2 * n ** 2 * d_model          # QK^T and the weighted sum
    ffn = 2 * n * d_model ** 2 * d_ff_mult
    crossover = d_model * d_ff_mult      # solve 2 n^2 d = 2 n d^2 f  ->  n = d f
    log.info("d_model=%d, ffn multiplier=%d -> attention overtakes the feed-forward at n=%d tokens",
             d_model, d_ff_mult, crossover)

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.loglog(n, attn, color=ARM_RED, lw=2.4, label=r"attention  $\propto n^2 d$")
    ax.loglog(n, ffn, color=ARM_BLUE, lw=2.4, label=r"feed-forward  $\propto n d^2$")
    ax.axvline(crossover, color=GREY, ls="--", lw=1.3)
    ax.text(crossover * 1.15, ffn[0] * 4, f"crossover\nn = {crossover:,}", fontsize=8.5,
            color=GREY, fontweight="bold")

    for ctx, lab in [(2048, "GPT-3"), (128000, "modern long context")]:
        if ctx <= n.max():
            ax.axvline(ctx, color=GREEN, ls=":", lw=1.2)
            ax.text(ctx * 1.1, attn.max() * 0.02, lab, rotation=90, fontsize=7.5, color=GREEN)

    ax.set_xlabel("sequence length $n$ (tokens)")
    ax.set_ylabel("floating-point operations")
    ax.set_title(f"Where attention starts to dominate ($d_{{model}}$ = {d_model:,})", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "l26_quadratic")
    return crossover


# --- L26: three architectures ----------------------------------------------------------
def fig_three_arch():
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.3); ax.axis("off")

    specs = [
        (0.25, "Encoder-only", "BERT", "every token sees\nevery other token", ARM_BLUE,
         "understanding:\nclassify, tag, embed"),
        (3.45, "Decoder-only", "GPT", "each token sees\nonly the past", ARM_RED,
         "generation:\nthe dominant design"),
        (6.65, "Encoder-decoder", "T5", "encoder reads all,\ndecoder attends to it", GREEN,
         "sequence to sequence:\ntranslation"),
    ]
    for x, name, model, how, color, use in specs:
        ax.add_patch(FancyBboxPatch((x, 1.55), 2.75, 0.62, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 1.37, 1.86, f"{name}  ({model})", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(x + 1.37, 1.15, how, ha="center", va="center", fontsize=8.5, color="#333333")
        ax.text(x + 1.37, 2.55, use, ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold")

    ax.text(4.8, 0.35, "same block, three ways of deciding who may look at whom",
            ha="center", fontsize=9, color=GREY, style="italic")
    ax.text(9.5, 0.02, "schematic", ha="right", fontsize=7, color=GREY, style="italic")
    save(fig, "l26_three_arch")


def main():
    gap_nope, gap_pe = fig_permutation()
    fig_pe_heatmap()
    fig_pe_dotprod()
    fig_causal_mask()
    fig_block()
    crossover = fig_quadratic()
    fig_three_arch()
    log.info("KEY NUMBERS FOR THE SLIDES: permutation gap without PE = %.1e, with PE = %.3f; "
             "quadratic crossover = %d tokens", gap_nope, gap_pe, crossover)
    log.info("done: 7 figures")


if __name__ == "__main__":
    main()


# --- L26: cross-attention, worked ------------------------------------------------------
def fig_cross_attention():
    """A decoder token's query against the ENCODER's keys and values, with real numbers.

    Added after the L26 student review: cross-attention was the only genuinely new mechanism
    in the L24-L26 arc that never got the worked-numbers treatment self-attention had in L24.
    """
    rng = np.random.default_rng(SEED)
    d_model, d_k = 8, 4
    source = ["le", "chat", "noir"]          # what the encoder read (French)
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) * 0.5 for _ in range(3))

    enc_out = rng.normal(size=(len(source), d_model))     # the encoder's output
    dec_tok = rng.normal(size=(1, d_model))               # the decoder writing "the ..."

    # The whole trick: Q comes from the DECODER, K and V come from the ENCODER.
    Q = dec_tok @ Wq
    K, V = enc_out @ Wk, enc_out @ Wv
    scores = (Q @ K.T / np.sqrt(d_k))[0]
    weights = softmax(scores)
    out = weights @ V

    log.info("cross-attention worked example (d_k=%d):", d_k)
    for w, s, a in zip(source, scores, weights):
        log.info("  source token %-5s score %+.3f -> weight %.3f", w, s, a)
    log.info("  weights sum to %.6f; output vector = %s",
             weights.sum(), np.array2string(out, precision=3))
    if abs(weights.sum() - 1) > 1e-9:
        raise ValueError("cross-attention weights must sum to 1")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.add_patch(FancyBboxPatch((0.3, 3.6), 4.2, 1.9, boxstyle="round,pad=0.02",
                                facecolor="#F2F6FB", edgecolor="#C8D6E8", lw=1.2))
    ax.text(2.4, 5.15, "ENCODER output", ha="center", fontsize=8.5,
            color=ARM_BLUE, fontweight="bold")
    for i, wtok in enumerate(source):
        ax.add_patch(FancyBboxPatch((0.55 + i * 1.32, 3.9), 1.15, 0.75,
                                    boxstyle="round,pad=0.02", facecolor=ARM_BLUE,
                                    edgecolor="none"))
        ax.text(1.12 + i * 1.32, 4.28, wtok, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.text(2.4, 3.35, "supplies K and V", ha="center", fontsize=8, color=ARM_BLUE)

    ax.add_patch(FancyBboxPatch((6.2, 3.6), 3.2, 1.9, boxstyle="round,pad=0.02",
                                facecolor="#FDF3F3", edgecolor="#EFCFCF", lw=1.2))
    ax.text(7.8, 5.15, "DECODER, writing now", ha="center", fontsize=8.5,
            color=ARM_RED, fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.05, 3.9), 1.5, 0.75, boxstyle="round,pad=0.02",
                                facecolor=ARM_RED, edgecolor="none"))
    ax.text(7.8, 4.28, "the", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
    ax.text(7.8, 3.35, "supplies Q", ha="center", fontsize=8, color=ARM_RED)

    for i, a in enumerate(weights):
        ax.add_patch(FancyArrowPatch((7.05, 4.28), (1.7 + i * 1.32, 4.28),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=GREEN, lw=0.6 + 5.5 * a,
                                     connectionstyle="arc3,rad=-0.30"))
    ax.text(4.9, 1.9, "arrow thickness = attention weight",
            ha="center", fontsize=8, color=GREEN, style="italic")

    ax = axes[1]
    bars = ax.bar(source, weights, color=[GREEN if a == weights.max() else ARM_BLUE
                                          for a in weights])
    ax.bar_label(bars, fmt="%.3f", fontsize=9.5, fontweight="bold", padding=2)
    ax.set_ylim(0, weights.max() * 1.3)
    ax.set_ylabel("attention weight")
    ax.set_title("softmax over the encoder tokens", fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, "l26_cross_attention")
    return source, scores, weights
