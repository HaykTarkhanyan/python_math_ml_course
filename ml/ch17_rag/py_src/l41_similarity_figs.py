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
import time
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

plt.rcParams.update({
    "font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
    "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})

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
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error("could not write %s: %s", out, e)
                raise
            log.warning("errno 22 on %s, retry %d/4", out.name, attempt)
            time.sleep(1)
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
        "short document": (0.45, -0.03),
        "long document": (-1.05, 0.12),
        "other topic": (-0.42, 0.25),
    }
    fig, ax = plt.subplots(figsize=(2.87, 1.75))   # drawn at slot size: 0.52 column
    z = {"long document": 2, "other topic": 3, "short document": 4, "query": 5}
    for name, v in VECS.items():
        ax.annotate("", xy=v, xytext=(0, 0), zorder=z[name],
                    arrowprops=dict(arrowstyle="-|>", color=VCOLORS[name], lw=1.5,
                                    mutation_scale=8))
        dx, dy = label_offset[name]
        ax.annotate(name, xy=v, xytext=(v[0] + dx, v[1] + dy), fontsize=6.5,
                    color=VCOLORS[name], fontweight="bold", va="center" if name == "short document" else "baseline",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=VCOLORS[name])
                    if name == "short document" else None)

    for name in ("short document", "long document"):
        v = VECS[name]
        ax.plot([0, v[0]], [0, v[1]], color=VCOLORS[name], lw=0.8, ls=":", alpha=0.6)

    c_same = cosine(q, VECS["short document"])
    c_other = cosine(q, VECS["other topic"])
    ax.text(0.1, -0.08,
            f"cos(query, short) = cos(query, long) = {c_same:.3f}\n"
            f"cos(query, other topic) = {c_other:.3f}",
            fontsize=6.5, color="#333333", va="top")

    ax.set_xlim(-0.8, 3.3)
    ax.set_ylim(-0.55, 1.75)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axvline(0, color=GREY, lw=0.7, ymax=0.8)   # stops below the "other topic" label
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("3x as long, the same cosine", fontsize=7.5)
    save(fig, "16_cosine_geometry")


def fig_cosine_normalised():
    fig, ax = plt.subplots(figsize=(2.31, 1.6))   # drawn at slot size: 0.42 column
    circ = plt.Circle((0, 0), 1.0, fill=False, color=GREY, lw=1.0, ls=":")
    ax.add_patch(circ)
    for name, v in VECS.items():
        u = v / np.linalg.norm(v)
        ax.annotate("", xy=u, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=VCOLORS[name], lw=1.5,
                                    mutation_scale=8))
        ax.scatter(*u, color=VCOLORS[name], s=14, zorder=4)
    u_short = VECS["short document"] / np.linalg.norm(VECS["short document"])
    ax.annotate("short and long:\nsame point", xy=u_short, xytext=(1.08, 0.72), fontsize=6.5,
                color=ARM_BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=ARM_BLUE))
    ax.text(1.06, 0.16, "query", fontsize=6.5, color=ARM_RED, fontweight="bold")
    ax.set_xlim(-1.15, 1.95); ax.set_ylim(-0.35, 1.3)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.7); ax.axvline(0, color=GREY, lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Normalise, and only the angle is left", fontsize=7.5)
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
    # The slides quote the band (0.819 to 0.876) and "both real breaks, plus two invented".
    got = (round(float(sims.min()), 3), round(float(sims.max()), 3), sorted(detected))
    if got != (0.819, 0.876, [1, 2, 3, 5]):
        raise ValueError(f"semantic-chunking numbers changed: {got}, slides quote (0.819, 0.876, [1, 2, 3, 5])")

    fig, ax = plt.subplots(figsize=(4.96, 1.9))   # drawn at slot size: 0.90 width
    x = np.arange(len(sims))
    colors = [ARM_RED if i in detected else ARM_BLUE for i in x]
    bars = ax.bar(x, sims, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=1.5)

    ax.axhline(threshold, color=ARM_ORANGE, ls="--", lw=1.0)
    # Placed in the gap left of the last bar, on white: it previously sat on top of a dark
    # blue bar and was unreadable on a projector (student review, 2026-08-10).
    ax.text(1.005, threshold, "split\nbelow", fontsize=6.5, color="#B07A00", va="center",
            fontweight="bold", transform=ax.get_yaxis_transform())

    # Mark where a human would actually put the topic boundaries.
    lo = 0.0   # from zero: the next slide's point is how narrow the band is (student review)
    for i in TRUE_BOUNDARIES:
        ax.scatter(i, lo + 0.04, marker="^", s=30, color="#2E8B57", zorder=5)
    ax.text(0.02, 0.965,
            "green triangle = real topic change" +
            f"   |   red bar = algorithm split ({len(detected)} of them)",
            transform=ax.transAxes, fontsize=6.5, color="#333333", va="top")

    ax.set_xticks(np.arange(len(SENTENCES)) - 0.5)
    ax.set_xticklabels([name.replace(" ", "\n", 1) for name in SHORT], fontsize=6)
    ax.set_xlim(-0.75, len(sims) - 0.25)
    ax.set_ylabel("similarity between\nneighbouring sentences")
    ax.set_ylim(lo, 1.08)
    ax.set_title("Semantic chunking finds the real breaks, and some imaginary ones")
    ax.tick_params(axis="x", length=3)
    ax.set_xlabel("each bar: similarity of the two sentences either side of it")
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
