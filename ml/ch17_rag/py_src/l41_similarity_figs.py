"""Similarity figures for L41: cosine refresher (16, 17) and semantic chunking (18).

  16  REAL - exact 2-D arithmetic. Two documents pointing the same way but with very
           different magnitudes get the SAME cosine and very different Euclidean distances.
  17  REAL - the same four vectors after normalisation, all on the unit circle.
  18  REAL - intfloat/multilingual-e5-small similarity between consecutive sentences of a
           real multi-topic passage. The split points are wherever similarity dips, and the
           script checks the dips actually land on the true topic boundaries.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_similarity_figs.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l41_similarity_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})

# A passage with three clearly different topics. True boundaries are AFTER index 2 and 5.
SENTENCES = [
    "The Lori press operates at 2.5 bar during the first pressing stage.",
    "After twenty minutes the pressure is raised to 3.2 bar.",
    "Operators must record both press readings in the shift log.",
    "Brining follows pressing.",
    "The brine tank is held at 12 degrees.",
    "Wheels stay in the brine for eight hours.",
    "Ripening cellars are kept at 10 degrees.",
    "Humidity in the cellar must stay near 85 percent.",
]
TRUE_BOUNDARIES = {2, 5}  # index i means "a split belongs between sentence i and i+1"
SHORT = ["press 2.5 bar", "raised to 3.2", "log the reading",
         "brining follows", "brine 12 deg", "eight hours", "cellar 10 deg", "humidity 85%"]

# Four 2-D vectors for the cosine refresher. d_long is exactly 3x d_short.
VECS = {
    "query":            np.array([1.0, 0.35]),
    "short document":   np.array([0.9, 0.45]),
    "long document":    np.array([2.7, 1.35]),
    "other topic":      np.array([-0.35, 1.1]),
}
VCOLORS = {"query": ARM_RED, "short document": ARM_BLUE,
           "long document": "#7FA8DC", "other topic": ARM_ORANGE}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- figures 16 and 17 -----------------------------------------------------------------
def fig_cosine_geometry():
    q = VECS["query"]
    log.info("cosine refresher:")
    for name, v in VECS.items():
        if name == "query":
            continue
        log.info("  %-16s cos=%.4f  euclidean=%.4f  |v|=%.3f",
                 name, cosine(q, v), float(np.linalg.norm(q - v)), float(np.linalg.norm(v)))

    if abs(cosine(q, VECS["short document"]) - cosine(q, VECS["long document"])) > 1e-9:
        raise ValueError("short and long document must have identical cosine for the lesson")

    # Hand-tuned label offsets: query and "short document" are nearly collinear, so their
    # labels collide if both are placed at the arrow tip.
    label_offset = {
        "query": (0.10, -0.20),
        "short document": (-0.30, 0.16),
        "long document": (-0.15, 0.14),
        "other topic": (-0.42, 0.10),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, v in VECS.items():
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=VCOLORS[name], lw=2.4))
        dx, dy = label_offset[name]
        ax.text(v[0] + dx, v[1] + dy, name, fontsize=9,
                color=VCOLORS[name], fontweight="bold")

    for name in ("short document", "long document"):
        v = VECS[name]
        ax.plot([0, v[0]], [0, v[1]], color=VCOLORS[name], lw=0.8, ls=":", alpha=0.6)

    c_same = cosine(q, VECS["short document"])
    c_other = cosine(q, VECS["other topic"])
    ax.text(1.55, -0.32,
            f"cos(query, short) = cos(query, long) = {c_same:.3f}\n"
            f"cos(query, other topic) = {c_other:.3f}",
            fontsize=9.5, color="#333333")

    ax.set_xlim(-0.8, 3.3)
    ax.set_ylim(-0.55, 1.75)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axvline(0, color=GREY, lw=0.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("The long document is 3x the short one, and scores identically",
                 fontsize=10.5)
    save(fig, "16_cosine_geometry")


def fig_cosine_normalised():
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    circ = plt.Circle((0, 0), 1.0, fill=False, color=GREY, lw=1.0, ls=":")
    ax.add_patch(circ)
    for name, v in VECS.items():
        u = v / np.linalg.norm(v)
        ax.annotate("", xy=u, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=VCOLORS[name], lw=2.4))
        ax.scatter(*u, color=VCOLORS[name], s=45, zorder=4)
    ax.text(0.72, 0.30, "short and long\nland on the\nsame point", fontsize=8.5,
            color=ARM_BLUE, fontweight="bold")
    ax.set_xlim(-1.25, 1.45); ax.set_ylim(-0.35, 1.3)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.7); ax.axvline(0, color=GREY, lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Normalise, and only the angle is left", fontsize=10.5)
    save(fig, "17_cosine_normalised")


# --- figure 18 -------------------------------------------------------------------------
def fig_semantic_chunking(model):
    vecs = model.encode([f"passage: {s}" for s in SENTENCES],
                        normalize_embeddings=True, show_progress_bar=False)
    sims = np.array([float(vecs[i] @ vecs[i + 1]) for i in range(len(SENTENCES) - 1)])

    threshold = sims.mean() - 0.5 * sims.std()
    detected = {i for i, s in enumerate(sims) if s < threshold}
    log.info("semantic chunking (threshold %.4f = mean - 0.5 sd):", threshold)
    for i, s in enumerate(sims):
        log.info("  between %d and %d: %.4f%s", i + 1, i + 2, s,
                 "   <- SPLIT" if i in detected else "")
    log.info("  detected %s vs true %s", sorted(detected), sorted(TRUE_BOUNDARIES))

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(sims))
    colors = [ARM_RED if i in detected else ARM_BLUE for i in x]
    bars = ax.bar(x, sims, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)

    ax.axhline(threshold, color=ARM_ORANGE, ls="--", lw=1.4)
    # Placed in the gap left of the last bar, on white: it previously sat on top of a dark
    # blue bar and was unreadable on a projector (student review, 2026-08-10).
    ax.text(3.5, threshold + 0.005, "split below here", fontsize=8.5,
            color="#B07A00", ha="center", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # Mark where a human would actually put the topic boundaries.
    lo = min(sims) - 0.03
    for i in TRUE_BOUNDARIES:
        ax.scatter(i, lo + 0.006, marker="^", s=110, color="#2E8B57", zorder=5)
    ax.text(0.02, 0.965,
            "green triangle = real topic change" +
            f"   |   red bar = algorithm split ({len(detected)} of them)",
            transform=ax.transAxes, fontsize=8.5, color="#333333", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[i]}\n|\n{SHORT[i+1]}" for i in x], fontsize=7.2)
    ax.set_ylabel("similarity between\nneighbouring sentences")
    ax.set_ylim(lo, max(sims) + 0.03)
    ax.set_title("Semantic chunking finds the real breaks, and some imaginary ones",
                 fontsize=11)
    ax.tick_params(axis="x", length=0)
    save(fig, "18_semantic_chunking")
    return detected


def main():
    fig_cosine_geometry()
    fig_cosine_normalised()
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    detected = fig_semantic_chunking(model)
    if detected == TRUE_BOUNDARIES:
        log.info("RESULT: detected boundaries match the true topic boundaries exactly")
    else:
        log.warning("RESULT: detected %s but true boundaries are %s - the slide must say so",
                    sorted(detected), sorted(TRUE_BOUNDARIES))
    log.info("done: 3 figures")


if __name__ == "__main__":
    main()
