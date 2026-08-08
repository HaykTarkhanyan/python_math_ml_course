"""Concept figures for L34 (Vision-Language Models II - how a model draws).

NO NEURAL NETWORK IS TRAINED anywhere in this chapter (instructor decision,
VLM_CHAPTER_PLAN.md 2026-08-07).

`vq_quantization` DOES fit a codebook, with k-means rather than a learned VQ-VAE encoder.
That is deliberate and is labelled as such on the slide: k-means on raw patches is the
simplest honest stand-in for vector quantization, it needs no training loop, and it answers
the chapter's own question - what does discretizing 1-2 px strokes actually cost? A learned
VQ-VAE would do better; this is therefore a LOWER bound on quality, which is the safe
direction for the claim the slide makes.

Generates into ml/ch12_vlm/fig/:
  vq_quantization.pdf -- our letters through codebooks of increasing size, plus the error curve
  raster_order.pdf    -- next-token prediction over a token grid, in raster order

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch12_vlm/py_src/l34_drawing_figs.py
"""

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"
DATA = REPO_ROOT / "ml" / "ch10_diffusion" / "data" / "mashtots_panir_24.npz"

PATCH = 4
CODEBOOK_SIZES = [8, 32, 128, 512]
FIT_PATCHES = 40_000       # subsample for k-means; the full set is ~161k and adds nothing


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "l34_drawing_figs.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


log = build_logger()


def load_dataset():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found - ch10's packed dataset is required")
    d = np.load(DATA, allow_pickle=False)
    X = d["X"].astype(np.float32) / 255.0
    log.info(f"loaded {X.shape[0]} letters at {X.shape[1]}x{X.shape[2]}, "
             f"classes {list(d['letters'])}")
    return X, d["y"], list(d["letters"])


def to_patches(imgs, patch=PATCH):
    """(n, s, s) -> (n, n_patch, patch*patch) in raster order."""
    n, s, _ = imgs.shape
    if s % patch:
        raise ValueError(f"patch {patch} does not divide image size {s}")
    k = s // patch
    return (imgs.reshape(n, k, patch, k, patch)
                .transpose(0, 1, 3, 2, 4)
                .reshape(n, k * k, patch * patch))


def from_patches(patches, size, patch=PATCH):
    """Inverse of to_patches."""
    n = patches.shape[0]
    k = size // patch
    return (patches.reshape(n, k, k, patch, patch)
                   .transpose(0, 1, 3, 2, 4)
                   .reshape(n, size, size))


# ---------------------------------------------------------------------------------------
def fig_vq_quantization(X, y, letters):
    """What does turning an image into a fixed vocabulary of patches actually cost?"""
    size = X.shape[1]
    all_patches = to_patches(X)
    flat = all_patches.reshape(-1, PATCH * PATCH)
    n_patch = all_patches.shape[1]
    log.info(f"{flat.shape[0]:,} patches of dim {flat.shape[1]} "
             f"({n_patch} per image, {PATCH}x{PATCH} each)")

    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(flat.shape[0], size=min(FIT_PATCHES, flat.shape[0]), replace=False)
    fit_set = flat[fit_idx]

    # One clean example of each letter, most ink first, for the visual row.
    show_idx = [int(np.flatnonzero(y == c)[
        np.argmax(X[y == c].mean(axis=(1, 2)))]) for c in range(len(letters))]
    show = X[show_idx]
    show_patches = to_patches(show)

    recons, errors, bits = {}, [], []
    for k in CODEBOOK_SIZES:
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5,
                             batch_size=2048, max_iter=200).fit(fit_set)
        # Reconstruction error over the WHOLE dataset, not just the shown examples.
        codes_all = km.predict(flat)
        recon_all = km.cluster_centers_[codes_all]
        mse = float(np.mean((recon_all - flat) ** 2))
        errors.append(mse)
        bits.append(n_patch * np.log2(k))
        recons[k] = from_patches(
            km.cluster_centers_[km.predict(show_patches.reshape(-1, PATCH * PATCH))]
              .reshape(len(show_idx), n_patch, PATCH * PATCH), size)
        log.info(f"codebook {k:4d}: MSE {mse:.5f}, {n_patch * np.log2(k):.0f} bits/image, "
                 f"fit {time.perf_counter() - t0:.1f}s")

    raw_bits = size * size * 8
    log.info(f"raw image = {raw_bits} bits; codebook 512 = {bits[-1]:.0f} bits "
             f"({raw_bits / bits[-1]:.1f}x compression)")

    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)

    # Left: the letters, original on top, then each codebook size.
    gs_l = gs[0].subgridspec(len(CODEBOOK_SIZES) + 1, len(letters), hspace=0.12, wspace=0.08)
    rows = [("original", show)] + [(f"K = {k}", recons[k]) for k in CODEBOOK_SIZES]
    for r, (name, imgs) in enumerate(rows):
        for c in range(len(letters)):
            ax = fig.add_subplot(gs_l[r, c])
            ax.imshow(imgs[c], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center",
                              labelpad=26,
                              color="black" if r == 0 else BLUE,
                              fontweight="bold" if r == 0 else "normal")
            if r == 0:
                ax.set_title(letters[c], fontsize=13)

    # Right: the cost curve.
    ax = fig.add_subplot(gs[1])
    ax.plot(CODEBOOK_SIZES, errors, "o-", color=RED, lw=2.0, ms=7)
    for k, e in zip(CODEBOOK_SIZES, errors):
        ax.annotate(f"{e:.4f}", (k, e), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CODEBOOK_SIZES)
    ax.set_xticklabels([str(k) for k in CODEBOOK_SIZES])
    ax.set_xlabel("codebook size K (the visual vocabulary)", fontsize=10)
    ax.set_ylabel("reconstruction MSE over all 4,481 letters", fontsize=10)
    ax.set_title(f"Bigger vocabulary, better image\n"
                 f"K=512: {n_patch} tokens = {bits[-1]:.0f} bits vs {raw_bits} raw "
                 f"({raw_bits / bits[-1]:.1f}x)", fontsize=10.5)
    ax.set_ylim(0, max(errors) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    fig.suptitle("Turning letters into a vocabulary of patches "
                 f"(k-means on {PATCH}x{PATCH} patches, a stand-in for a learned VQ-VAE)",
                 fontsize=12, y=1.0)
    out = FIG / "vq_quantization.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")
    return errors, bits


# ---------------------------------------------------------------------------------------
def fig_raster_order(k=6, generated=20):
    """Once an image is tokens, generating it is next-token prediction. Nothing else changes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                            gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            if idx < generated:
                face, tc = BLUE, "white"
            elif idx == generated:
                face, tc = RED, "white"
            else:
                face, tc = "white", "0.55"
            # White edges throughout, so adjacent filled cells still read as separate tokens.
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=face,
                                       edgecolor="white" if idx <= generated else "0.75",
                                       lw=1.6 if idx <= generated else 1.0))
            ax.text(j + 0.5, -i + 0.5, str(idx + 1), ha="center", va="center",
                    fontsize=8.5, color=tc,
                    fontweight="bold" if idx == generated else "normal")
    # One scan-direction cue above the grid - never drawn through the numbers.
    ax.annotate("", xy=(k, 1.28), xytext=(0, 1.28),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    ax.text(k / 2, 1.42, "left to right, then down", fontsize=9, color=ORANGE,
            ha="center", fontweight="bold")
    ax.set_xlim(-0.25, k + 0.25); ax.set_ylim(-k + 0.15, 1.95)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"The token grid\n{generated} generated (blue), next one in red",
                 fontsize=10.5)

    ax = axes[1]
    total = k * k
    w = 0.82
    for idx in range(total):
        if idx < generated:
            face, edge = BLUE, "white"
        elif idx == generated:
            face, edge = RED, "white"
        else:
            face, edge = "white", "0.75"
        ax.add_patch(plt.Rectangle((idx, 0), w, 1.15, facecolor=face, edgecolor=edge, lw=0.9))
    # Brace-style span over exactly the generated region, ending at the token being predicted.
    ax.annotate("", xy=(generated - 0.1, 1.55), xytext=(0, 1.55),
                arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color=BLUE, lw=1.4))
    ax.text(generated / 2, 1.75, f"context: tokens 1..{generated}",
            fontsize=9, color=BLUE, ha="center", fontweight="bold")
    ax.annotate("", xy=(generated + w / 2, 1.35), xytext=(generated + w / 2, 2.35),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(generated + w / 2, 2.5, f"predict token {generated + 1}",
            fontsize=9.5, color=RED, ha="center", fontweight="bold")
    ax.set_xlim(-1, total + 1); ax.set_ylim(-0.5, 3.0)
    ax.axis("off")
    ax.set_title(f"Flattened: a sequence of {total} tokens\n"
                 "identical to predicting the next word", fontsize=10.5)

    fig.tight_layout()
    out = FIG / "raster_order.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
# Explanatory diagrams. Added 2026-08-08 (instructor: more frames and illustrations,
# explanatory rather than experimental). Nothing below measures anything.

from matplotlib.patches import FancyBboxPatch, Rectangle    # noqa: E402


def _panel(ax, x, y, w, h, text, color, fs=8, alpha=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, color="0.4", lw=1.3, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls))


def fig_straight_through():
    """The forward path goes through the quantizer; the backward path walks around it."""
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    _panel(ax, 0.3, 1.5, 1.7, 0.8, "encoder", BLUE)
    _panel(ax, 2.9, 1.5, 2.2, 0.8, "snap to nearest\ncodebook entry", ORANGE)
    _panel(ax, 6.0, 1.5, 1.7, 0.8, "decoder", BLUE)

    _arrow(ax, 2.0, 1.9, 2.9, 1.9)
    _arrow(ax, 5.1, 1.9, 6.0, 1.9)
    ax.text(4.0, 2.55, "FORWARD: use the codebook entry", fontsize=9, ha="center",
            color="0.25")

    # backward path detours around the quantizer
    _arrow(ax, 6.0, 1.2, 2.0, 1.2, color=RED, lw=1.6)
    ax.plot([6.0, 6.0], [1.5, 1.2], color=RED, lw=1.6)
    ax.plot([2.0, 2.0], [1.2, 1.5], color=RED, lw=1.6)
    ax.text(4.0, 1.05, "BACKWARD: copy the gradient straight across,\n"
                       "as if the quantizer were not there", fontsize=9, ha="center",
            va="top", color=RED)

    ax.add_patch(Rectangle((2.85, 1.05), 2.3, 1.75, fill=False, ec=RED, lw=1.0, ls=":"))
    ax.text(4.0, 0.3, "The derivative of \"pick the nearest\" is zero almost everywhere. "
                      "So we route around it and check that training works.",
            fontsize=8.5, ha="center", style="italic", color="0.3")
    out = FIG / "straight_through.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_codebook_collapse():
    """Codebook collapse, drawn. Schematic usage histograms, not a measurement."""
    rng = np.random.default_rng(SEED)
    k = 64
    healthy = rng.gamma(6.0, 1.0, size=k)
    healthy = healthy / healthy.sum()
    collapsed = np.zeros(k)
    live = rng.choice(k, size=9, replace=False)
    collapsed[live] = rng.gamma(6.0, 1.0, size=9)
    collapsed = collapsed / collapsed.sum()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 2.9), sharey=True)
    for ax, data, title, color in [
            (axes[0], healthy, "healthy: every entry earns its keep", "#008C46"),
            (axes[1], collapsed, f"collapsed: {int((collapsed > 0).sum())} of {k} entries "
                                 f"do all the work", RED)]:
        ax.bar(range(k), data, color=color, alpha=0.85, width=0.9)
        ax.set_title(title, fontsize=9.5, color=color)
        ax.set_xlabel("codebook entry", fontsize=8.5)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("how often it is chosen", fontsize=8.5)
    fig.suptitle("Schematic. An entry that is never the nearest to anything gets no gradient, "
                 "so it stays where it was.", fontsize=8.5, y=0.02)
    fig.tight_layout()
    out = FIG / "codebook_collapse.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_three_paradigms():
    """Chameleon vs Transfusion vs Janus - the chapter's central disagreement, drawn."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    specs = [
        ("1. fully discrete", BLUE, "Chameleon",
         [("image -> codebook indices", ORANGE),
          ("ONE sequence of\ndiscrete tokens", BLUE),
          ("one transformer,\none softmax", "#7832A0"),
          ("cross-entropy\n(one loss)", "#008C46")],
         "Clean. Pays the\nquantization cost."),
        ("2. hybrid", RED, "Transfusion",
         [("image -> continuous latents", ORANGE),
          ("text tokens AND\nimage latents", RED),
          ("one transformer,\ntwo heads", "#7832A0"),
          ("cross-entropy\n+ diffusion loss", "#008C46")],
         "No quantization loss.\nTwo objectives to balance."),
        ("3. decoupled", ORANGE, "Janus",
         [("TWO image encoders:\nunderstand vs generate", ORANGE),
          ("separate paths in", ORANGE),
          ("one transformer\nbody", "#7832A0"),
          ("task-specific\nheads", "#008C46")],
         "Understanding and drawing\nwant different features."),
    ]
    for ax, (title, color, who, boxes, note) in zip(axes, specs):
        ax.set_xlim(0, 3); ax.set_ylim(0, 4.8); ax.axis("off")
        ax.set_title(f"{title}\n{who}", fontsize=10, color=color)
        for i, (label, c) in enumerate(boxes):
            y = 3.75 - i * 0.85
            _panel(ax, 0.25, y, 2.5, 0.6, label, c, fs=7.5)
            if i < len(boxes) - 1:
                _arrow(ax, 1.5, y, 1.5, y - 0.25)
        ax.text(1.5, 0.15, note, fontsize=8, ha="center", color="0.25", style="italic")
    fig.suptitle("All three are trying to reconcile ONE thing: a language model emits "
                 "discrete symbols, an image is continuous.", fontsize=9.5, y=1.0)
    fig.tight_layout()
    out = FIG / "three_paradigms.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    X, y, letters = load_dataset()
    fig_vq_quantization(X, y, letters)
    fig_raster_order()
    fig_straight_through()
    fig_codebook_collapse()
    fig_three_paradigms()
    log.info("done - 5 figures")
