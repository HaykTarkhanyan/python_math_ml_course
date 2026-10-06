"""L42 figures 14-17: how the search itself is made fast.

  14  REAL          - brute-force scan latency, measured on this laptop, 5k to 250k vectors
  15  ILLUSTRATIVE  - HNSW drawn as a skip list: sparse top layer, dense bottom layer
  16  REAL          - recall@10 against fraction of the index scanned (k-means / IVF style)
  17  REAL          - memory and recall@10 after int8 and binary quantization

The vectors in 14, 16 and 17 are synthetic (clustered Gaussians, seed 509), not the
18-chunk corpus: the question is how search behaves at 10^4-10^5 vectors, and 18 real ones
cannot answer it. The *behaviour* being measured - scan time, recall loss - is real.

14's timings are measured once and kept in results/l42_brute_force_timings.json, so redrawing
the figure does not re-time it (the deck quotes these numbers). Delete the file to re-measure.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_index_figs.py   (~20 s)
"""

import json
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
TIMINGS = Path(__file__).resolve().parent.parent / "results" / "l42_brute_force_timings.json"
# Numbers the slides quote; a rerun that disagrees must fail, not silently redraw.
QUOTED = {"recall_3_clusters": 0.988, "recall_1_cluster": 0.575, "scanned_3": 0.032,
          "scanned_1": 0.012, "int8_recall": 0.969, "brute_4000_ms": 0.34, "brute_1m_ms": 100}
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
    "font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
    "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


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


def clustered_vectors(n, dim=128, n_clusters=40, seed=SEED):
    """Synthetic embeddings with cluster structure, unit-normalised like real ones."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    which = rng.integers(0, n_clusters, size=n)
    x = centres[which] + 0.6 * rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


# --- figure 14 -------------------------------------------------------------------------
def measure_brute_force(sizes, repeats):
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
        log.info("14 measured %7d vectors (%.0f MB)", n, mat.nbytes / 1e6)
        del mat
    return times


def fig_brute_force(repeats=5):
    sizes = [4_000, 10_000, 25_000, 50_000, 100_000, 250_000]
    if TIMINGS.exists():
        data = json.loads(TIMINGS.read_text(encoding="utf-8"))
        if data["sizes"] != sizes or data["dim"] != DIM:
            raise ValueError(f"{TIMINGS} was measured for other sizes or dim - delete it to re-measure")
        times = data["median_ms"]
        log.info("14 timings read from %s (%s)", TIMINGS.name, data["source"])
    else:
        times = measure_brute_force(sizes, repeats)
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        TIMINGS.write_text(json.dumps({
            "source": f"measured by l42_index_figs.py on {time.strftime('%Y-%m-%d %H:%M')}",
            "dim": DIM, "repeats": repeats, "sizes": sizes,
            "median_ms": [round(t, 2) for t in times]}, indent=2), encoding="utf-8")
        log.info("14 timings written to %s", TIMINGS)
    for n, t in zip(sizes, times):
        log.info("14 %7d vectors x %d dims: %.2f ms", n, DIM, t)

    per_vec = times[-1] / sizes[-1]
    est_1m = per_vec * 1_000_000
    log.info("14 -> 1,000,000 vectors would take about %.0f ms (linear extrapolation)", est_1m)

    fig, ax = plt.subplots(figsize=(3.0, 1.85))   # drawn at slot size: 0.54 width
    ax.plot(sizes, times, "o-", color=ARM_BLUE, ms=3.5, label="measured")
    ax.plot([sizes[-1], 1_000_000], [times[-1], est_1m], "--", color=GREY, lw=1.1,
            label="linear extrapolation")
    ax.scatter([1_000_000], [est_1m], s=20, color=GREY, zorder=4)
    ax.annotate(f"1M vectors\n{est_1m:.0f} ms", (1_000_000, est_1m), fontsize=6.5,
                color=GREY, fontweight="bold", xytext=(-2, -36),
                textcoords="offset points", ha="right", va="top")
    ax.annotate(f"our 4,000 chunks\n{times[0]:.2f} ms", (sizes[0], times[0]), xytext=(4_300, 2.5),
                fontsize=6.5, color=GREEN, fontweight="bold", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREEN))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"vectors in the index ({DIM} dimensions, float32)")
    ax.set_ylabel("one exhaustive search (ms)")
    ax.set_title("Exhaustive search, measured on this laptop CPU")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    save(fig, "l42_14_brute_force")
    return times, est_1m


# --- figure 15 -------------------------------------------------------------------------
def fig_hnsw():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(4.3, 1.75))   # drawn at slot size: 0.78 width
    ax.set_xlim(0, 10.8); ax.set_ylim(-0.1, 4.45); ax.axis("off")

    layers = [(3.55, 4, "layer 2: a few nodes,\nlong hops"),
              (2.15, 9, "layer 1: more nodes,\nshorter hops"),
              (0.75, 20, "layer 0: every node,\nlocal links")]
    positions = {}
    for li, (y, n, label) in enumerate(layers):
        xs = np.linspace(0.7, 7.3, n) + rng.normal(scale=0.05, size=n)
        positions[li] = xs
        ax.add_patch(FancyBboxPatch((0.35, y - 0.42), 7.3, 0.84,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#F4F6FA", edgecolor="#D8DFEA", lw=0.7))
        ax.text(7.9, y, label, fontsize=6.5, color=GREY, va="center", ha="left")
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [y, y], color="#B9C4D6", lw=0.7, zorder=1)
        ax.scatter(xs, np.full(n, y), s=16, color=ARM_BLUE, zorder=3,
                   edgecolor="white", lw=0.6)

    # greedy descent: enter at the top left, hop across, drop a layer, refine
    path = [(positions[0][0], layers[0][0]), (positions[0][2], layers[0][0]),
            (positions[1][5], layers[1][0]), (positions[1][6], layers[1][0]),
            (positions[2][13], layers[2][0]), (positions[2][14], layers[2][0])]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ARM_RED, lw=1.2, mutation_scale=8))
    ax.scatter([path[-1][0]], [path[-1][1]], s=45, color=ARM_RED, marker="*", zorder=5)
    ax.text(path[0][0] - 0.3, path[0][1] + 0.5, "enter here", fontsize=7,
            color=ARM_RED, fontweight="bold")
    ax.text(path[-1][0] + 0.2, path[-1][1] - 0.62, "nearest neighbour", fontsize=7,
            color=ARM_RED, fontweight="bold")

    ax.set_title("HNSW: a skip list whose nodes are vectors")
    ax.text(5.4, -0.2, "Schematic - a real graph has 384-dimensional nodes and no "
            "left-to-right order.", ha="center", va="top", fontsize=6.5, color=GREY, style="italic")
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

    fig, ax = plt.subplots(figsize=(3.0, 1.85))   # drawn at slot size: 0.54 width
    ax.plot([100 * s for s in scanned], [100 * r for r in recalls], "o-",
            color=ARM_BLUE, ms=3.5)
    lines = [f"{p:>3} cluster{'s' if p > 1 else ' '}: {100 * s:5.1f}% scanned, "
             f"{100 * r:5.1f}% recall"
             for p, s, r in zip(probes, scanned, recalls) if p in (1, 2, 3, 5, 100)]
    ax.text(1.0, 8, "\n".join(lines), fontsize=5.8, color="#333333", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6FA", edgecolor="#D8DFEA"))
    ax.axhline(100, color=GREEN, ls=":", lw=1.0)
    ax.text(12, 104, "exhaustive search", fontsize=6.5, color=GREEN)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlim(0.9, 130)
    ax.set_xlabel("share of the index actually compared (%, log scale)")
    ax.set_ylabel(f"recall@{k} (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"Approximate search: {n:,} vectors in {n_lists} clusters")
    ax.grid(alpha=0.25, lw=0.4)
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
        ("float32", dim * 4, 1.0),
        ("int8", dim * 1, recall(approx_i8)),
        ("binary", dim / 8, recall(approx_bin)),
        ("binary +\nrescore top 100", dim / 8, recall(approx_rescore)),
    ]
    for name, byts, rec in rows:
        log.info("17 %-18s %6.0f bytes/vector  recall@%d %.3f  (%.1f GB for 10M vectors)",
                 name.replace("\n", " "), byts, k, rec, byts * 1e7 / 1e9)

    # Horizontal bars sharing one name column: four names under each of two vertical-bar
    # panels need about 3 in of label width, and the slot is 3.4 in for both panels.
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.45), sharey=True)   # slot: 0.62 width
    names = [r[0] for r in rows][::-1]
    mem = [r[1] for r in rows][::-1]
    rec = [100 * r[2] for r in rows][::-1]
    colours = [ARM_BLUE, ARM_ORANGE, ARM_RED, GREEN][::-1]

    bars = axes[0].barh(names, mem, color=colours, height=0.6)
    axes[0].bar_label(bars, labels=[f"{m:.0f} B" for m in mem], fontsize=6.5,
                      fontweight="bold", padding=2)
    axes[0].set_xlim(0, max(mem) * 1.55)
    axes[0].set_xticks([])
    axes[0].set_title("bytes per vector")

    bars = axes[1].barh(names, rec, color=colours, height=0.6)
    axes[1].bar_label(bars, fmt="%.0f%%", fontsize=6.5, fontweight="bold", padding=2)
    axes[1].set_xlim(0, 135)
    axes[1].set_xticks([])
    axes[1].set_title(f"recall@{k}")

    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_visible(False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "l42_17_quantization")
    return rows


def check_quoted(got):
    bad = {key: (QUOTED[key], val) for key, val in got.items() if round(val, 3) != QUOTED[key]}
    if bad:
        raise ValueError(f"numbers the slides quote changed (quoted, now): {bad}")
    log.info("all %d quoted numbers match the slides", len(got))


def main():
    times, est_1m = fig_brute_force()
    fig_hnsw()
    probes, recalls, scanned = fig_ann_recall()
    if recalls[-1] < 0.99:
        raise ValueError("scanning every cluster did not reproduce exact search "
                         f"(recall {recalls[-1]:.3f}) - the IVF implementation is wrong")
    rows = fig_quantization()
    check_quoted({
        "recall_3_clusters": recalls[probes.index(3)], "recall_1_cluster": recalls[probes.index(1)],
        "scanned_3": round(scanned[probes.index(3)], 3), "scanned_1": round(scanned[probes.index(1)], 3),
        "int8_recall": rows[1][2], "brute_4000_ms": times[0], "brute_1m_ms": round(est_1m),
    })
    log.info("done: 4 figures")


if __name__ == "__main__":
    main()
