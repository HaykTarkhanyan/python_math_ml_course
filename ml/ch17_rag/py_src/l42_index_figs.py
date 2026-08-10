"""L42 figures 14-17: how the search itself is made fast.

  14  REAL          - brute-force scan latency, measured on this laptop, 5k to 250k vectors
  15  ILLUSTRATIVE  - HNSW drawn as a skip list: sparse top layer, dense bottom layer
  16  REAL          - recall@10 against fraction of the index scanned (k-means / IVF style)
  17  REAL          - memory and recall@10 after int8 and binary quantization

The vectors in 14, 16 and 17 are synthetic (clustered Gaussians, seed 509), not the
18-chunk corpus: the question is how search behaves at 10^4-10^5 vectors, and 18 real ones
cannot answer it. The *behaviour* being measured - scan time, recall loss - is real.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_index_figs.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"
SEED = 509
DIM = 384

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_index_figs.log", encoding="utf-8")],
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


def clustered_vectors(n, dim=128, n_clusters=40, seed=SEED):
    """Synthetic embeddings with cluster structure, unit-normalised like real ones."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    which = rng.integers(0, n_clusters, size=n)
    x = centres[which] + 0.6 * rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


# --- figure 14 -------------------------------------------------------------------------
def fig_brute_force(repeats=5):
    sizes = [4_000, 10_000, 25_000, 50_000, 100_000, 250_000]
    rng = np.random.default_rng(SEED)
    times = []
    for n in sizes:
        mat = rng.normal(size=(n, DIM)).astype(np.float32)
        vec = rng.normal(size=DIM).astype(np.float32)
        mat @ vec
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            scores = mat @ vec
            np.argpartition(-scores, 10)[:10]
            ts.append((time.perf_counter() - t0) * 1e3)
        times.append(float(np.median(ts)))
        log.info("14 %7d vectors x %d dims: %.2f ms  (%.0f MB)",
                 n, DIM, times[-1], mat.nbytes / 1e6)
        del mat

    per_vec = times[-1] / sizes[-1]
    est_1m = per_vec * 1_000_000
    log.info("14 -> 1,000,000 vectors would take about %.0f ms (linear extrapolation)", est_1m)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(sizes, times, "o-", color=ARM_BLUE, lw=2, ms=6, label="measured")
    ax.plot([sizes[-1], 1_000_000], [times[-1], est_1m], "--", color=GREY, lw=1.6,
            label="linear extrapolation")
    ax.scatter([1_000_000], [est_1m], s=70, color=GREY, zorder=4)
    ax.annotate(f"1M vectors\n{est_1m:.0f} ms", (1_000_000, est_1m), fontsize=9.5,
                color=GREY, fontweight="bold", xytext=(-8, -26),
                textcoords="offset points", ha="right")
    ax.annotate(f"our 4,000 chunks\n{times[0]:.2f} ms", (sizes[0], times[0]), fontsize=9.5,
                color=GREEN, fontweight="bold", xytext=(8, -8), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"vectors in the index ({DIM} dimensions, float32)")
    ax.set_ylabel("one exhaustive search (ms)")
    ax.set_title("Comparing against every vector, measured on this laptop CPU", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    save(fig, "l42_14_brute_force")
    return est_1m


# --- figure 15 -------------------------------------------------------------------------
def fig_hnsw():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")

    layers = [(3.55, 4, "layer 2: a few nodes, long hops"),
              (2.15, 9, "layer 1: more nodes, shorter hops"),
              (0.75, 20, "layer 0: every node, local links")]
    positions = {}
    for li, (y, n, label) in enumerate(layers):
        xs = np.linspace(0.7, 9.3, n) + rng.normal(scale=0.06, size=n)
        positions[li] = xs
        ax.add_patch(FancyBboxPatch((0.35, y - 0.42), 9.4, 0.84,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#F4F6FA", edgecolor="#D8DFEA", lw=1.0))
        ax.text(9.95, y, label, fontsize=10, color=GREY, va="center", ha="left",
                rotation=0)
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [y, y], color="#B9C4D6", lw=1.0, zorder=1)
        ax.scatter(xs, np.full(n, y), s=60, color=ARM_BLUE, zorder=3,
                   edgecolor="white", lw=1.0)

    # greedy descent: enter at the top left, hop across, drop a layer, refine
    path = [(positions[0][0], layers[0][0]), (positions[0][2], layers[0][0]),
            (positions[1][5], layers[1][0]), (positions[1][6], layers[1][0]),
            (positions[2][13], layers[2][0]), (positions[2][14], layers[2][0])]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ARM_RED, lw=2.0))
    ax.scatter([path[-1][0]], [path[-1][1]], s=150, color=ARM_RED, marker="*", zorder=5)
    ax.text(path[0][0] - 0.35, path[0][1] + 0.45, "enter here", fontsize=10.5,
            color=ARM_RED, fontweight="bold")
    ax.text(path[-1][0] + 0.25, path[-1][1] - 0.48, "nearest neighbour", fontsize=10.5,
            color=ARM_RED, fontweight="bold")

    ax.set_title("HNSW: a skip list, but the nodes are vectors and the hops are "
                 "in 384 dimensions", fontsize=12.5)
    fig.text(0.5, -0.01, "Schematic - a real graph has 384-dimensional nodes and no "
             "left-to-right order.", ha="center", fontsize=9.5, color=GREY, style="italic")
    save(fig, "l42_15_hnsw")


# --- figure 16 -------------------------------------------------------------------------
def fig_ann_recall(n=20_000, dim=128, n_lists=100, n_queries=200, k=10):
    x = clustered_vectors(n, dim=dim)
    rng = np.random.default_rng(SEED + 1)
    q = x[rng.choice(n, size=n_queries, replace=False)] + 0.15 * rng.normal(size=(n_queries, dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q.astype(np.float32)

    exact = np.argsort(-(q @ x.T), axis=1)[:, :k]

    km = KMeans(n_clusters=n_lists, random_state=SEED, n_init=3).fit(x)
    assign = km.labels_
    centres = km.cluster_centers_.astype(np.float32)
    buckets = [np.flatnonzero(assign == c) for c in range(n_lists)]

    probes = [1, 2, 3, 5, 10, 20, 50, 100]
    recalls, scanned = [], []
    for p in probes:
        order = np.argsort(-(q @ centres.T), axis=1)[:, :p]
        hit, seen = 0, 0
        for qi in range(n_queries):
            idx = np.concatenate([buckets[c] for c in order[qi]])
            seen += len(idx)
            sub = idx[np.argsort(-(x[idx] @ q[qi]))[:k]]
            hit += len(set(sub.tolist()) & set(exact[qi].tolist()))
        recalls.append(hit / (n_queries * k))
        scanned.append(seen / (n_queries * n))
        log.info("16 nprobe %3d: scanned %5.1f%% of the index, recall@%d = %.3f",
                 p, 100 * scanned[-1], k, recalls[-1])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot([100 * s for s in scanned], [100 * r for r in recalls], "o-",
            color=ARM_BLUE, lw=2, ms=6)
    lines = [f"{p:>3} cluster{'s' if p > 1 else ' '} -> {100 * s:5.1f}% scanned, "
             f"{100 * r:5.1f}% recall"
             for p, s, r in zip(probes, scanned, recalls) if p in (1, 2, 3, 5, 100)]
    ax.text(2.0, 12, "\n".join(lines), fontsize=9.5, color="#333333", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6FA", edgecolor="#D8DFEA"))
    ax.axhline(100, color=GREEN, ls=":", lw=1.2)
    ax.text(1.05, 102, "exhaustive search", fontsize=9, color=GREEN)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlim(0.9, 130)
    ax.set_xlabel("share of the index actually compared (%, log scale)")
    ax.set_ylabel(f"recall@{k} (% of the true top {k} found)")
    ax.set_ylim(0, 112)
    ax.set_title(f"Approximate search: {n:,} vectors in {n_lists} clusters", fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    save(fig, "l42_16_ann_recall")
    return probes, recalls, scanned


# --- figure 17 -------------------------------------------------------------------------
def fig_quantization(n=20_000, dim=128, n_queries=200, k=10):
    x = clustered_vectors(n, dim=dim)
    rng = np.random.default_rng(SEED + 2)
    q = x[rng.choice(n, size=n_queries, replace=False)] + 0.15 * rng.normal(size=(n_queries, dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q.astype(np.float32)
    exact = np.argsort(-(q @ x.T), axis=1)[:, :k]

    scale = np.abs(x).max() / 127.0
    x_i8 = np.round(x / scale).astype(np.int8)
    approx_i8 = np.argsort(-(q @ x_i8.T.astype(np.float32)), axis=1)[:, :k]

    x_bin = np.sign(x).astype(np.int8)
    bin_scores = np.sign(q).astype(np.float32) @ x_bin.T.astype(np.float32)
    approx_bin = np.argsort(-bin_scores, axis=1)[:, :k]

    # What production actually does: shortlist with the cheap binary codes, then re-score
    # that shortlist with the full float32 vectors.
    shortlist = np.argsort(-bin_scores, axis=1)[:, :100]
    approx_rescore = np.stack([
        shortlist[i][np.argsort(-(x[shortlist[i]] @ q[i]))[:k]] for i in range(n_queries)])

    def recall(approx):
        return float(np.mean([len(set(a.tolist()) & set(e.tolist())) / k
                              for a, e in zip(approx, exact)]))

    rows = [
        ("float32\n(original)", dim * 4, 1.0),
        ("int8\n(1 byte/dim)", dim * 1, recall(approx_i8)),
        ("binary\n(1 bit/dim)", dim / 8, recall(approx_bin)),
        ("binary, then rescore\nthe top 100", dim / 8, recall(approx_rescore)),
    ]
    for name, byts, rec in rows:
        log.info("17 %-18s %6.0f bytes/vector  recall@%d %.3f  (%.1f GB for 10M vectors)",
                 name.replace("\n", " "), byts, k, rec, byts * 1e7 / 1e9)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    names = [r[0] for r in rows]
    mem = [r[1] for r in rows]
    rec = [100 * r[2] for r in rows]
    colours = [ARM_BLUE, ARM_ORANGE, ARM_RED, GREEN]

    bars = axes[0].bar(names, mem, color=colours, width=0.55)
    axes[0].bar_label(bars, labels=[f"{m:.0f} B" for m in mem], fontsize=10.5,
                      fontweight="bold", padding=3)
    axes[0].set_ylabel("bytes per vector")
    axes[0].set_ylim(0, max(mem) * 1.2)
    axes[0].set_title(f"memory ({dim} dimensions)", fontsize=10)

    bars = axes[1].bar(names, rec, color=colours, width=0.55)
    axes[1].bar_label(bars, fmt="%.0f%%", fontsize=10.5, fontweight="bold", padding=3)
    axes[1].set_ylabel(f"recall@{k} vs exact float32 (%)")
    axes[1].set_ylim(0, 118)
    axes[1].set_title("how many of the true top 10 survive", fontsize=10)

    for ax in axes:
        ax.tick_params(axis="x", length=0, labelsize=8.5)
    fig.suptitle("Quantization: shrink the vectors, then check what it cost you",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    save(fig, "l42_17_quantization")
    return rows


def main():
    fig_brute_force()
    fig_hnsw()
    probes, recalls, scanned = fig_ann_recall()
    if recalls[-1] < 0.99:
        raise ValueError("scanning every cluster did not reproduce exact search "
                         f"(recall {recalls[-1]:.3f}) - the IVF implementation is wrong")
    fig_quantization()
    log.info("done: 4 figures")


if __name__ == "__main__":
    main()
