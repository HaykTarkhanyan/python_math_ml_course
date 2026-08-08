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
    # Every resolution here is an exact multiple of the patch size, so (s/p)^2 holds
    # literally. 1024 is NOT (1024/14 = 73.1) and quoting it would contradict the formula
    # the slide states.
    res = [224, 336, 448, 672, 896]
    for r in res:
        if r % patch:
            raise ValueError(f"{r} is not a multiple of patch {patch}; (s/p)^2 would be a lie")
    tokens = [(r // patch) ** 2 for r in res]
    labels = [f"{r}x{r}" for r in res]
    page_of_text = 500      # ~400 words of English at roughly 1.3 tokens/word
    log.info("token budget: " + ", ".join(f"{l}={t}" for l, t in zip(labels, tokens)))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [BLUE if r != 336 else RED for r in res]
    bars = ax.bar(labels, tokens, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10, fontweight="bold")
    ax.axhline(page_of_text, color=ORANGE, lw=2.0, ls="--")
    # Annotate into the empty upper-left region and point at the line. Placing the label ON
    # the line struck it through both the line and the neighbouring bar label.
    ax.annotate(f"a full page of text\n~ {page_of_text} tokens",
                xy=(0.62, page_of_text), xytext=(0.12, max(tokens) * 0.52),
                color=ORANGE, fontsize=9.5, ha="left", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
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


# ---------------------------------------------------------------------------------------
# CLIP, in detail. Added 2026-08-08: the original three CLIP frames stated the objective
# without ever showing how a similarity number is produced, what the temperature does, or what
# one training step actually computes. These three figures back the expanded frames.

def fig_clip_pipeline():
    """Two towers to one number. The dimensions are CLIP ViT-B/32's real ones."""
    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.5):
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=0.18, lw=1.5))
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y, x1, label=""):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.3))
        if label:
            ax.text((x0 + x1) / 2, y + 0.13, label, ha="center", fontsize=7.5, color="0.35")

    # image tower
    box(0.1, 2.15, 1.35, 0.75, "image", ORANGE)
    arrow(1.45, 2.52, 1.95)
    box(1.95, 2.15, 1.7, 0.75, "ViT-B/32", BLUE)
    arrow(3.65, 2.52, 4.15, "768-d")
    box(4.15, 2.15, 1.8, 0.75, "linear\nprojection", BLUE)
    arrow(5.95, 2.52, 6.45, "512-d")
    box(6.45, 2.15, 1.7, 0.75, "L2 normalise", "#008C46")

    # text tower
    box(0.1, 0.5, 1.35, 0.75, '"a photo of\na cat"', ORANGE)
    arrow(1.45, 0.87, 1.95)
    box(1.95, 0.5, 1.7, 0.75, "text\ntransformer", RED)
    arrow(3.65, 0.87, 4.15, "512-d")
    box(4.15, 0.5, 1.8, 0.75, "linear\nprojection", RED)
    arrow(5.95, 0.87, 6.45, "512-d")
    box(6.45, 0.5, 1.7, 0.75, "L2 normalise", "#008C46")

    # join
    ax.annotate("", xy=(8.75, 1.7), xytext=(8.15, 2.52),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.3))
    ax.annotate("", xy=(8.75, 1.7), xytext=(8.15, 0.87),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.3))
    box(8.75, 1.32, 1.35, 0.78, "dot\nproduct", "#7832A0")
    ax.text(9.42, 1.05, "one number in $[-1, 1]$", ha="center", fontsize=8, color="0.3")
    ax.text(5.1, 3.25, "Both towers must end in the SAME dimension, or there is nothing to "
                       "compare", ha="center", fontsize=9.5)
    ax.text(7.3, 1.72, "after normalising, the dot\nproduct IS the cosine",
            ha="center", fontsize=7.5, color="#008C46", style="italic")
    fig.tight_layout()
    out = FIG / "clip_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_clip_temperature():
    """What the learned temperature actually controls: how peaked the softmax is."""
    sims = np.array([0.31, 0.24, 0.28, 0.22])       # one row of cosine similarities
    taus = [1.0, 0.2, 0.07, 0.01]
    fig, axes = plt.subplots(1, len(taus), figsize=(10.4, 2.7), sharey=True)
    for ax, tau in zip(axes, taus):
        p = np.exp(sims / tau) / np.exp(sims / tau).sum()
        colors = [RED if k == 0 else "0.65" for k in range(len(p))]
        bars = ax.bar(range(len(p)), p, color=colors, alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
        ax.set_title(rf"$\tau = {tau}$", fontsize=10)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels([f"T{k+1}" for k in range(len(p))], fontsize=8)
        ax.set_ylim(0, 1.18)
        ax.grid(alpha=0.2, axis="y")
        log.info(f"temperature tau={tau}: softmax {np.round(p, 3)}")
    axes[0].set_ylabel("probability")
    fig.suptitle("Same four similarities (0.31, 0.24, 0.28, 0.22), four temperatures. "
                 r"CLIP initialises $\tau$ at 0.07 and learns it.", fontsize=9.5)
    fig.tight_layout()
    out = FIG / "clip_temperature.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_clip_worked_step():
    """One training step on a batch of 4, with every number computable by hand on the slide."""
    sim = np.array([
        [0.31, 0.24, 0.28, 0.22],
        [0.19, 0.35, 0.21, 0.26],
        [0.25, 0.20, 0.33, 0.18],
        [0.23, 0.27, 0.20, 0.30],
    ])
    tau = 0.07
    logits = sim / tau
    row_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    col_p = np.exp(logits.T) / np.exp(logits.T).sum(axis=1, keepdims=True)
    loss_i = -np.log(np.diag(row_p)).mean()
    loss_t = -np.log(np.diag(col_p)).mean()
    total = (loss_i + loss_t) / 2
    log.info(f"worked CLIP step: image->text loss {loss_i:.4f}, text->image {loss_t:.4f}, "
             f"total {total:.4f}; row-1 softmax {np.round(row_p[0], 3)}")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})

    for ax, data, title, fmt in [
            (axes[0], sim, r"1. cosine similarities $s_{ij}$", "{:.2f}"),
            (axes[1], logits, r"2. divide by $\tau = 0.07$", "{:.1f}")]:
        ax.imshow(data, cmap="Blues", vmin=data.min(), vmax=data.max())
        for i in range(4):
            for j in range(4):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > data.mean() else "black")
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.0))
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"T{k+1}" for k in range(4)], fontsize=8)
        ax.set_yticklabels([f"I{k+1}" for k in range(4)], fontsize=8)
        ax.set_title(title, fontsize=9.5)

    ax = axes[2]
    bars = ax.bar(range(4), row_p[0], color=[RED, "0.65", "0.65", "0.65"], alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"T{k+1}" for k in range(4)], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("3. softmax of row I1\n"
                 rf"loss for this row $= -\ln {row_p[0,0]:.3f} = {-np.log(row_p[0,0]):.2f}$",
                 fontsize=9.5)
    ax.grid(alpha=0.2, axis="y")
    fig.suptitle(f"Average over all 4 rows, then over all 4 columns, and halve: "
                 f"batch loss = {total:.2f}", fontsize=9.5, y=0.02)
    fig.tight_layout()
    out = FIG / "clip_worked_step.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")
    return sim, row_p, loss_i, loss_t, total


# ---------------------------------------------------------------------------------------
# Explanatory diagrams. Added 2026-08-08 (instructor: more frames, more illustrations,
# explanatory rather than experimental). Nothing here measures anything - these are teaching
# diagrams for frames that previously carried their whole argument in prose.

def _panel(ax, x, y, w, h, text, color, fs=7.5, alpha=0.16):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, color="0.4", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig_vit_pipeline(letter):
    """The Vision Transformer end to end. The frame previously had no picture at all."""
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.1], wspace=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(letter, cmap="gray_r")
    for k in range(1, 6):
        ax0.axhline(k * 4 - 0.5, color=BLUE, lw=0.8)
        ax0.axvline(k * 4 - 0.5, color=BLUE, lw=0.8)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title("1. patchify\n$24\\times24$, $4\\times4$ patches", fontsize=8.5)

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    for i in range(5):
        _panel(ax, 0.15 + i * 0.62, 2.5, 0.5, 0.5, "", ORANGE)
    ax.text(1.7, 2.15, "36 flattened patches", fontsize=7.5, ha="center", color="0.35")
    ax.text(1.7, 3.2, "2. one linear projection", fontsize=8.5, ha="center")

    _arrow(ax, 3.35, 2.75, 3.9, 2.75)

    for i in range(5):
        _panel(ax, 4.0 + i * 0.62, 2.5, 0.5, 0.5, f"+{i+1}", "#7832A0", fs=7)
    ax.text(5.55, 3.2, "3. add position embeddings", fontsize=8.5, ha="center")
    ax.text(5.55, 2.15, "now each patch knows where it was", fontsize=7.5,
            ha="center", color="0.35")

    _panel(ax, 0.15, 1.05, 0.95, 0.55, "CLS", "#008C46", fs=8)
    for i in range(5):
        _panel(ax, 1.25 + i * 0.62, 1.05, 0.5, 0.55, "", "#7832A0")
    ax.text(1.9, 0.72, "4. prepend a classification token", fontsize=8.5, ha="left")

    _arrow(ax, 4.5, 1.32, 5.05, 1.32)
    _panel(ax, 5.15, 0.85, 2.1, 0.95, "$N$ ordinary\ntransformer blocks", BLUE, fs=8.5)
    _arrow(ax, 7.3, 1.32, 7.85, 1.32)
    _panel(ax, 7.95, 0.85, 1.9, 0.95, "one vector\nper patch\n(+ the CLS vector)",
           "#008C46", fs=7.5)
    ax.text(5.0, 0.25, "Nothing on this row is vision-specific. Step 4 onward is ch9, "
                       "unchanged.", fontsize=8.5, ha="center", style="italic", color="0.3")
    out = FIG / "vit_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_position_embeddings(letter):
    """Why position embeddings are not optional: shuffled patches are the same multiset."""
    rng = np.random.default_rng(SEED)
    p = 4
    n = letter.shape[0] // p
    patches = (letter.reshape(n, p, n, p).transpose(0, 2, 1, 3).reshape(n * n, p, p))
    order = rng.permutation(n * n)
    shuffled = patches[order].reshape(n, n, p, p).transpose(0, 2, 1, 3).reshape(n * p, n * p)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
    for ax, img, title in [(axes[0], letter, "the image"),
                           (axes[1], shuffled, "its patches, shuffled")]:
        ax.imshow(img, cmap="gray_r")
        for k in range(1, n):
            ax.axhline(k * p - 0.5, color=BLUE, lw=0.7)
            ax.axvline(k * p - 0.5, color=BLUE, lw=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.5, 0.78, "To self-attention,\nwithout position embeddings,", ha="center",
            fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.52, "these two are\nTHE SAME INPUT", ha="center", fontsize=13,
            color=RED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.24, "Attention is permutation-equivariant:\nit sees a SET of patches, "
                       "not a grid.", ha="center", fontsize=8.5, color="0.35",
            transform=ax.transAxes)
    fig.tight_layout()
    out = FIG / "position_embeddings.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_shared_space():
    """What 'a shared embedding space' means, as a picture. Schematic, and labelled so."""
    rng = np.random.default_rng(SEED)
    concepts = ["cat", "dog", "car", "tree"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    img_pts = rng.uniform(-1, 1, size=(4, 2))
    txt_pts = rng.uniform(-1, 1, size=(4, 2))
    axes[0].set_title("before training: two unrelated spaces", fontsize=10)
    for k, c in enumerate(concepts):
        axes[0].scatter(*img_pts[k], s=90, color=BLUE, marker="o", zorder=3)
        axes[0].scatter(*txt_pts[k], s=90, color=RED, marker="s", zorder=3)
        axes[0].annotate(c, img_pts[k], fontsize=7.5, xytext=(4, 4),
                         textcoords="offset points")
        axes[0].annotate(f'"{c}"', txt_pts[k], fontsize=7.5, xytext=(4, 4),
                         textcoords="offset points")

    anchors = np.array([[-0.65, 0.6], [0.7, 0.55], [-0.6, -0.65], [0.65, -0.6]])
    axes[1].set_title("after training: paired things point the same way", fontsize=10)
    for k, c in enumerate(concepts):
        i_pt = anchors[k] + rng.normal(0, 0.06, 2)
        t_pt = anchors[k] + rng.normal(0, 0.06, 2)
        axes[1].plot([i_pt[0], t_pt[0]], [i_pt[1], t_pt[1]], color="0.6", lw=1, zorder=1)
        axes[1].scatter(*i_pt, s=90, color=BLUE, marker="o", zorder=3)
        axes[1].scatter(*t_pt, s=90, color=RED, marker="s", zorder=3)
        axes[1].annotate(c, i_pt, fontsize=7.5, xytext=(5, 4), textcoords="offset points")

    for ax in axes:
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].scatter([], [], color=BLUE, marker="o", label="image embedding")
    axes[0].scatter([], [], color=RED, marker="s", label="text embedding")
    axes[0].legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Schematic - a real space has 512 dimensions, not 2", fontsize=8.5, y=0.02)
    fig.tight_layout()
    out = FIG / "shared_space.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_zero_shot():
    """Zero-shot classification as a picture: one image against N sentences.

    Two clearly separated rows - image on top, class sentences below - because an earlier
    version overlapped them and was unreadable.
    """
    classes = ["cat", "dog", "marshrutka"]
    sims = [0.31, 0.12, 0.09]

    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.4); ax.axis("off")

    # top row: the image
    _panel(ax, 0.15, 2.35, 1.15, 0.75, "image", ORANGE, fs=9)
    _arrow(ax, 1.35, 2.72, 1.75, 2.72)
    _panel(ax, 1.8, 2.35, 1.45, 0.75, "image\nencoder", BLUE, fs=8.5)
    _arrow(ax, 3.3, 2.72, 3.7, 2.72)
    _panel(ax, 3.75, 2.42, 1.35, 0.6, "one vector", "#008C46", fs=8.5)

    # bottom row: the class sentences
    for k, c in enumerate(classes):
        _panel(ax, 0.15, 1.35 - k * 0.5, 1.9, 0.4, f'"a photo of a {c}"', "0.5", fs=6.8)
    ax.text(1.1, 1.92, "class list, written at inference time", fontsize=7.5, ha="center")
    _arrow(ax, 2.1, 1.15, 1.75 + 0.05, 1.15)
    _panel(ax, 2.2, 0.85, 1.45, 0.7, "text\nencoder", RED, fs=8.5)
    _arrow(ax, 3.7, 1.2, 4.1, 1.2)
    _panel(ax, 4.15, 0.9, 1.35, 0.6, "three vectors", "#008C46", fs=8.5)

    # join
    _arrow(ax, 5.15, 2.72, 5.95, 2.2)
    _arrow(ax, 5.55, 1.2, 5.95, 1.75)
    ax.text(6.35, 3.02, "cosine similarity of each pair", fontsize=8, ha="left", color="0.35")
    for k, s in enumerate(sims):
        y = 2.35 - k * 0.62
        color = RED if k == 0 else "0.62"
        ax.barh(y, s * 7.0, left=7.15, height=0.36, color=color, alpha=0.88)
        ax.text(7.15 + s * 7.0 + 0.1, y, f"{s:.2f}", va="center", fontsize=8.5)
        ax.text(7.05, y, classes[k], ha="right", va="center", fontsize=8)
    ax.annotate("pick the largest", xy=(9.3, 2.35), xytext=(8.4, 0.75),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3),
                fontsize=8.5, color=RED, fontweight="bold", ha="center")
    # Kept clear of the class boxes on the left, which it used to sit on top of.
    ax.text(7.6, 0.28, "No classifier head. No fine-tuning. No training run.",
            fontsize=9, ha="center", style="italic", color="0.3")
    out = FIG / "zero_shot.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_three_designs():
    """Projector vs resampler vs early fusion, side by side. The chapter's key comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    specs = [
        ("A - projector", BLUE,
         [("image", ORANGE), ("frozen vision\nencoder", BLUE),
          ("linear / MLP\nprojector", "#008C46"), ("LLM", "#7832A0")],
         "576 tokens in,\n576 tokens out", "LLaVA"),
        ("B - resampler", RED,
         [("image", ORANGE), ("frozen vision\nencoder", RED),
          ("Q-Former\n$k$ learned queries", "#008C46"), ("LLM", "#7832A0")],
         "576 tokens in,\n$k$ = 32 or 64 out", "BLIP-2, Flamingo"),
        ("C - early fusion", ORANGE,
         [("image", ORANGE), ("patch\nprojection", ORANGE),
          ("(no separate\nencoder)", "0.6"), ("one transformer\nfor both", "#7832A0")],
         "trained on both\nfrom scratch", "Fuyu, Chameleon"),
    ]
    for ax, (title, color, boxes, cost, who) in zip(axes, specs):
        ax.set_xlim(0, 3); ax.set_ylim(0, 4.6); ax.axis("off")
        ax.set_title(title, fontsize=10.5, color=color)
        for i, (label, c) in enumerate(boxes):
            y = 3.6 - i * 0.85
            _panel(ax, 0.35, y, 2.3, 0.62, label, c, fs=8)
            if i < len(boxes) - 1:
                _arrow(ax, 1.5, y, 1.5, y - 0.23)
        ax.text(1.5, 0.5, cost, fontsize=8.5, ha="center", color="0.25")
        ax.text(1.5, 0.05, who, fontsize=8, ha="center", style="italic", color=color)
    fig.suptitle("Same problem, three answers: how many tokens does the language model see, "
                 "and who decides?", fontsize=9.5, y=1.0)
    fig.tight_layout()
    out = FIG / "three_designs.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def _market():
    """A real photograph, for the frames where a 24x24 letter is too abstract to convince.

    A Yerevan spice market: dozens of near-identical bowls (counting), handwritten Armenian
    price labels (text in images, and a minority script), and fine detail that a 336x336
    resize destroys. One image carries three different frames.
    """
    from PIL import Image
    path = FIG / "img" / "yerevan_market.jpg"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - re-download it before running this script")
    return Image.open(path).convert("RGB")


def fig_patchify_real():
    """Patchify, on a real photo at the resolution a real model actually uses."""
    from PIL import Image
    im = _market().resize((224, 224), Image.LANCZOS)
    arr = np.asarray(im)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    axes[0].imshow(arr)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("$224\\times224$, the standard input", fontsize=9.5)

    axes[1].imshow(arr)
    for k in range(1, 14):
        axes[1].axhline(k * 16 - 0.5, color="white", lw=0.7, alpha=0.9)
        axes[1].axvline(k * 16 - 0.5, color="white", lw=0.7, alpha=0.9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("cut into $16\\times16$ patches\n$14\\times14 = 196$ tokens",
                      fontsize=9.5)
    fig.suptitle("Each square becomes ONE vector. The transformer never sees a pixel grid "
                 "again.", fontsize=9.5, y=0.03)
    fig.tight_layout()
    out = FIG / "patchify_real.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    log.info(f"wrote {out}")


def fig_resolution_detail():
    """What a 336x336 resize actually destroys, shown on real handwriting."""
    from PIL import Image
    im = _market()
    box = (620, 330, 940, 570)          # the handwritten Armenian price labels
    native = im.crop(box)

    # What the same region looks like after the WHOLE image is squeezed to 336x336 - which is
    # exactly what a fixed-resolution vision encoder does before it sees anything.
    small = im.resize((336, 336), Image.LANCZOS)
    sx, sy = 336 / im.width, 336 / im.height
    sbox = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
    degraded = small.crop(sbox).resize(native.size, Image.NEAREST)
    log.info(f"resolution figure: crop {native.size} vs the same region at "
             f"{sbox[2]-sbox[0]}x{sbox[3]-sbox[1]} after a 336x336 resize")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, img, title, color in [
            (axes[0], native, "the labels, at the photo's own resolution", "#008C46"),
            (axes[1], degraded, "the same labels after a $336\\times336$ resize", RED)]:
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, color=color)
    fig.suptitle("576 tokens is not the constraint that hurts here - the RESIZE is. "
                 "The writing is gone before the model starts.", fontsize=9, y=0.04)
    fig.tight_layout()
    out = FIG / "resolution_detail.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    log.info(f"wrote {out}")


def fig_anyres():
    """Why resizing to a square is lossy, and what tiling does instead - on the real photo."""
    from PIL import Image
    im = _market()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    axes[0].imshow(im)
    axes[0].set_title("the original\n$960\\times720$", fontsize=9)

    axes[1].imshow(im.resize((336, 336), Image.LANCZOS))
    axes[1].set_title("squashed to $336\\times336$\n576 tokens, and distorted", fontsize=9,
                      color=RED)

    axes[2].imshow(im)
    w, h = im.size
    for i in range(2):
        for j in range(2):
            axes[2].add_patch(Rectangle((j * w / 2, i * h / 2), w / 2 - 2, h / 2 - 2,
                                        fill=False, ec=BLUE, lw=2.4))
    axes[2].add_patch(Rectangle((w * 0.72, h * 0.03), w * 0.25, h * 0.25,
                                fc="white", ec=ORANGE, lw=2.0, alpha=0.92))
    axes[2].text(w * 0.845, h * 0.155, "thumbnail\n(context)", ha="center", va="center",
                 fontsize=7)
    axes[2].set_title("4 crops at full detail + 1 thumbnail\n$5\\times576 = 2{,}880$ tokens",
                      fontsize=9, color="#008C46")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = FIG / "anyres_tiling.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=170); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    letter = load_letter("Պ")
    fig_patchify(letter)
    fig_clip_contrastive()
    fig_token_budget()
    fig_clip_pipeline()
    fig_clip_temperature()
    fig_clip_worked_step()
    fig_vit_pipeline(letter)
    fig_position_embeddings(letter)
    fig_shared_space()
    fig_zero_shot()
    fig_three_designs()
    fig_patchify_real()
    fig_resolution_detail()
    fig_anyres()
    log.info("done - 14 figures")
