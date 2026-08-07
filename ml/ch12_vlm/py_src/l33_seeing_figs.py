"""Concept figures for L33 (Vision-Language Models I - how a model sees).

NO MODEL IS TRAINED anywhere in this chapter (instructor decision, VLM_CHAPTER_PLAN.md
2026-08-07). Everything here is exact arithmetic, a real dataset, or a clearly-labelled
schematic drawn from a specified distribution.

Generates into ml/ch12_vlm/fig/:
  patchify_grid.pdf   -- a real letter cut into patches, and the sequence of vectors it becomes
  clip_contrastive.pdf-- the NxN image-text similarity matrix (schematic, specified structure)
  token_budget.pdf    -- exact ViT token counts vs resolution, against a page of text

Reuses ml/ch10_diffusion/data/mashtots_panir_24.npz so the chapter needs no new download and
the patchify frame is the same letter students generated in ch10.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch12_vlm/py_src/l33_seeing_figs.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"
DATA = REPO_ROOT / "ml" / "ch10_diffusion" / "data" / "mashtots_panir_24.npz"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "l33_seeing_figs.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


log = build_logger()


def load_letter(letter="Պ"):
    """One clean example of `letter` from the ch10 dataset. Fails loud if it is missing."""
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found - ch10's packed dataset is required")
    d = np.load(DATA, allow_pickle=False)
    letters = list(d["letters"])
    if letter not in letters:
        raise ValueError(f"{letter!r} not in {letters}")
    idx = np.flatnonzero(d["y"] == letters.index(letter))
    if idx.size == 0:
        raise RuntimeError(f"no samples for {letter!r} in {DATA}")
    # Pick the example with the most ink: the clearest one to show patchified.
    imgs = d["X"][idx].astype(np.float32) / 255.0
    pick = idx[int(np.argmax(imgs.mean(axis=(1, 2))))]
    log.info(f"letter {letter}: {idx.size} samples, using index {pick}")
    return d["X"][pick].astype(np.float32) / 255.0


# ---------------------------------------------------------------------------------------
def fig_patchify(img, patch=4):
    """An image is not a sequence. Cut it into patches and it becomes one."""
    size = img.shape[0]
    if size % patch:
        raise ValueError(f"patch {patch} does not divide image size {size}")
    n_side = size // patch
    n_patch = n_side ** 2
    # Raster order: top-to-bottom, left-to-right. Same order the transformer will see.
    patches = (img.reshape(n_side, patch, n_side, patch)
                  .transpose(0, 2, 1, 3)
                  .reshape(n_patch, patch * patch))
    log.info(f"patchify: {size}x{size} / {patch} -> {n_patch} patches of dim {patch * patch}")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]})

    ax = axes[0]
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    for k in range(1, n_side):
        ax.axhline(k * patch - 0.5, color=BLUE, lw=0.9)
        ax.axvline(k * patch - 0.5, color=BLUE, lw=0.9)
    ax.set_title(f"1. The image\n{size}x{size} pixels", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # The patches, pulled apart so the grid reads as separate objects.
    ax = axes[1]
    gap = 1.15
    for i in range(n_side):
        for j in range(n_side):
            p = patches[i * n_side + j].reshape(patch, patch)
            x0, y0 = j * patch * gap, -i * patch * gap - patch
            ax.imshow(p, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest",
                      extent=(x0, x0 + patch, y0, y0 + patch))
            # Outline every patch, so blank ones still read as tokens rather than empty space.
            ax.add_patch(Rectangle((x0, y0), patch, patch, fill=False,
                                   edgecolor=BLUE, lw=0.7, alpha=0.55))
    ax.set_xlim(-1, n_side * patch * gap)
    ax.set_ylim(-n_side * patch * gap, 1)
    ax.set_aspect("equal")
    ax.set_title(f"2. Cut into patches\n{n_side}x{n_side} = {n_patch} patches of {patch}x{patch}",
                 fontsize=10)
    ax.axis("off")

    # Each patch flattened to a vector: the sequence the transformer actually receives.
    ax = axes[2]
    im = ax.imshow(patches, cmap="gray_r", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    ax.set_title(f"3. Flatten each patch\n{n_patch} vectors of dim {patch * patch}",
                 fontsize=10)
    ax.set_xlabel("vector component", fontsize=9)
    ax.set_ylabel("patch index (raster order)", fontsize=9)
    ax.set_xticks([0, patch * patch - 1]); ax.set_xticklabels(["1", str(patch * patch)])
    ax.set_yticks([0, n_patch - 1]); ax.set_yticklabels(["1", str(n_patch)])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("pixel value", fontsize=8)

    fig.suptitle("An image becomes a sequence of vectors - and a transformer can read it",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "patchify_grid.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_clip_contrastive(n=8):
    """CLIP's batch: N images, N captions, N correct pairs and N*N-N free negatives.

    Schematic - the numbers come from a specified distribution, not a trained CLIP. The
    frame's point is the STRUCTURE of the matrix, which is exact.
    """
    rng = np.random.default_rng(SEED)
    sim = rng.normal(0.10, 0.07, size=(n, n))          # mismatched pairs: low similarity
    sim[np.arange(n), np.arange(n)] = rng.normal(0.75, 0.06, size=n)   # matched pairs: high
    sim = np.clip(sim, -0.1, 1.0)
    off = sim[~np.eye(n, dtype=bool)]
    log.info(f"clip matrix {n}x{n}: diagonal mean {np.diag(sim).mean():.3f}, "
             f"off-diagonal mean {off.mean():.3f}, "
             f"{n} positives vs {n * n - n} negatives")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(sim, cmap="Blues", vmin=-0.1, vmax=1.0)
    for k in range(n):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.2))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{k + 1}" for k in range(n)], fontsize=9)
    ax.set_yticklabels([f"I{k + 1}" for k in range(n)], fontsize=9)
    ax.set_xlabel("text embeddings in the batch", fontsize=10)
    ax.set_ylabel("image embeddings in the batch", fontsize=10)
    ax.set_title(f"Every batch supplies its own wrong answers\n"
                 f"{n} correct pairs (red) against {n * n - n} negatives - for free",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("cosine similarity", fontsize=9)
    fig.tight_layout()
    out = FIG / "clip_contrastive.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_token_budget(patch=14):
    """Exact ViT token counts. This is the constraint that shapes every VLM design."""
    res = [224, 336, 448, 672, 1024]
    tokens = [(r // patch) ** 2 for r in res]
    labels = [f"{r}x{r}" for r in res]
    page_of_text = 500      # ~400 words of English at roughly 1.3 tokens/word
    log.info("token budget: " + ", ".join(f"{l}={t}" for l, t in zip(labels, tokens)))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [BLUE if r != 336 else RED for r in res]
    bars = ax.bar(labels, tokens, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10, fontweight="bold")
    ax.axhline(page_of_text, color=ORANGE, lw=2.0, ls="--")
    ax.text(len(res) - 0.42, page_of_text * 1.10,
            f"a full page of text ~ {page_of_text} tokens",
            color=ORANGE, fontsize=9.5, ha="right", fontweight="bold")
    ax.set_ylabel(f"visual tokens (patch {patch}x{patch})", fontsize=10)
    ax.set_xlabel("input resolution", fontsize=10)
    ax.set_title("One image costs more than a page of text\n"
                 "LLaVA-1.5 sits at 336x336 = 576 tokens (red)", fontsize=11)
    ax.set_ylim(0, max(tokens) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "token_budget.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    letter = load_letter("Պ")
    fig_patchify(letter)
    fig_clip_contrastive()
    fig_token_budget()
    log.info("done - 3 figures")
