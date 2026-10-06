"""L42 figures 10-13: metadata filtering, the cascade, late interaction, contradictions.

  10  ILLUSTRATIVE - filter first vs search first, and what each returns
  11  REAL latency  - the retrieve-then-rerank funnel, with measured milliseconds
  12  REAL          - token-level MaxSim heatmap (query tokens x document tokens)
  13  REAL          - two contradictory revisions get near-identical similarity scores

Figure 11's timings are measured on this laptop's CPU with intfloat/multilingual-e5-small
(12 layers, 384 dim). It stands in for a cross-encoder of the same size: a cross-encoder is
the same transformer reading query and passage concatenated, so one forward pass per pair
is the same work. The stand-in is stated on the slide.

The timings are measured once and kept in results/l42_cascade_timings.json, so redrawing the
figure does not re-time it: two August runs differed by 40%, and the slides quote one of them.
Delete the file to re-measure.

Figure 12 uses the same model's token vectors. It is NOT a trained ColBERT model - it shows
the MaxSim mechanism computed for real on real token vectors, which is what the frame
claims and nothing more.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_rerank_figs.py   (~30 s)
"""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS
from l42_data import REVISION_QUERY, REVISIONS

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"
MODEL_NAME = "intfloat/multilingual-e5-small"

FIG = Path(__file__).resolve().parent.parent / "fig"
TIMINGS = Path(__file__).resolve().parent.parent / "results" / "l42_cascade_timings.json"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_rerank_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 7,  # layout pass 2026-10-05: figures drawn at slot size
                     "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 140})


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


def box(ax, x, y, w, h, label, color, fontsize=6.5, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, color=color, lw=1.0))


# --- figure 10 -------------------------------------------------------------------------
def fig_metadata_filter():
    fig, axes = plt.subplots(1, 2, figsize=(5.07, 1.75))   # drawn at slot size: 0.92 width

    for ax, title, subtitle in [
        (axes[0], "Filter first, then search", "search a smaller index"),
        (axes[1], "Search first, then filter", "the top-k may empty out"),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0.6, 5); ax.axis("off")
        ax.text(5, 4.62, title, ha="center", fontsize=7.5, fontweight="bold")
        ax.text(5, 4.05, subtitle, ha="center", fontsize=6.5, color=GREY, style="italic")

    ax = axes[0]
    box(ax, 0.0, 2.5, 2.8, 1.1, "1,000,000\nchunks", ARM_BLUE, fontsize=6)
    arrow(ax, 2.85, 3.05, 3.3, 3.05)
    box(ax, 3.35, 2.5, 3.6, 1.1, "access OK and\nrev = current", ARM_ORANGE,
        fontsize=6, text_color="#3A2A00")
    arrow(ax, 7.0, 3.05, 7.4, 3.05)
    box(ax, 7.45, 2.5, 2.55, 1.1, "80,000\nchunks", ARM_BLUE, fontsize=6)
    arrow(ax, 8.72, 2.45, 8.72, 1.85)
    box(ax, 5.6, 0.7, 4.4, 1.1, "search these\n-> 10 valid hits", GREEN, fontsize=6)

    ax = axes[1]
    box(ax, 0.0, 2.5, 2.8, 1.1, "1,000,000\nchunks", ARM_BLUE, fontsize=6)
    arrow(ax, 2.85, 3.05, 3.3, 3.05)
    box(ax, 3.35, 2.5, 3.6, 1.1, "search all\n-> top 10", ARM_BLUE, fontsize=6)
    arrow(ax, 7.0, 3.05, 7.4, 3.05)
    box(ax, 7.45, 2.5, 2.55, 1.1, "then\nfilter", ARM_ORANGE, fontsize=6, text_color="#3A2A00")
    arrow(ax, 8.72, 2.45, 8.72, 1.85)
    box(ax, 5.6, 0.7, 4.4, 1.1, "2 survive\n-> thin answer", ARM_RED, fontsize=6)

    fig.suptitle("Where the filter goes changes what the user gets", fontsize=7.5, y=1.02)
    fig.text(0.5, 0.08, "Illustrative counts - the ordering is the point, not the numbers.",
             ha="center", va="top", fontsize=6.5, color=GREY, style="italic")
    save(fig, "l42_10_metadata_filter")


# --- figure 11 -------------------------------------------------------------------------
def measure_latency(model, n_pairs=100, n_index=100_000, dim=384, repeats=5):
    """Encoder latency for one query, brute-force scan cost, and per-pair scoring cost."""
    rng = np.random.default_rng(509)
    query = "What pressure should the Lori press run at for Lori cheese?"

    model.encode([f"query: {query}"], normalize_embeddings=True)  # warm up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.encode([f"query: {query}"], normalize_embeddings=True)
        ts.append((time.perf_counter() - t0) * 1e3)
    t_encode = float(np.median(ts))

    mat = rng.normal(size=(n_index, dim)).astype(np.float32)
    vec = rng.normal(size=dim).astype(np.float32)
    mat @ vec
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        scores = mat @ vec
        np.argpartition(-scores, 100)[:100]
        ts.append((time.perf_counter() - t0) * 1e3)
    t_scan = float(np.median(ts))
    del mat

    # A cross-encoder reads query and passage together: one forward pass per pair.
    pairs = [f"query: {query} [SEP] passage: {CHUNKS[i % len(CHUNKS)]}" for i in range(n_pairs)]
    model.encode(pairs[:8], batch_size=8)  # warm up
    t0 = time.perf_counter()
    model.encode(pairs, batch_size=8, show_progress_bar=False)
    t_pairs = (time.perf_counter() - t0) * 1e3

    log.info("11 measured: encode 1 query %.1f ms | scan %d vectors %.1f ms | "
             "%d pairs through the encoder %.0f ms (%.2f ms/pair)",
             t_encode, n_index, t_scan, n_pairs, t_pairs, t_pairs / n_pairs)
    return t_encode, t_scan, n_index, t_pairs, n_pairs


def cascade_timings(model):
    if TIMINGS.exists():
        d = json.loads(TIMINGS.read_text(encoding="utf-8"))
        log.info("11 timings read from %s (%s)", TIMINGS.name, d["source"])
        return d["encode_ms"], d["scan_ms"], d["n_index"], d["pairs_ms"], d["n_pairs"]
    t_encode, t_scan, n_index, t_pairs, n_pairs = measure_latency(model)
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps({
        "source": f"measured by l42_rerank_figs.py on {time.strftime('%Y-%m-%d %H:%M')}",
        "encode_ms": round(t_encode, 1), "scan_ms": round(t_scan, 1), "n_index": n_index,
        "pairs_ms": round(t_pairs), "n_pairs": n_pairs}, indent=2), encoding="utf-8")
    log.info("11 timings written to %s", TIMINGS)
    return t_encode, t_scan, n_index, t_pairs, n_pairs


def fig_cascade(model):
    t_encode, t_scan, n_index, t_pairs, n_pairs = cascade_timings(model)
    scan_1m = t_scan * (1_000_000 / n_index)
    per_pair = t_pairs / n_pairs
    all_pairs_h = per_pair * 1_000_000 / 1000 / 3600
    log.info("11 derived: scan 1M %.0f ms | cross-encode 1M pairs %.1f hours", scan_1m, all_pairs_h)
    # The slides quote: scan 82 ms measured, about 3 seconds for 100 pairs, about 9 hours.
    got = (round(scan_1m), round(t_pairs / 1000), round(all_pairs_h))
    if got != (82, 3, 9):
        raise ValueError(f"cascade numbers changed: {got}, slides quote (82, 3, 9)")

    stages = [
        ("1,000,000 chunks", "the whole index", "", ARM_BLUE),
        ("100 candidates", "bi-encoder search",
         f"{t_encode:.0f} ms encode + {scan_1m:.0f} ms scan", ARM_BLUE),
        ("10 survivors", "cross-encoder rerank",
         f"{t_pairs:.0f} ms to score the 100", ARM_ORANGE),
        ("3 in the prompt", "token budget", "free", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(4.96, 1.75))   # drawn at slot size: 0.90 width
    ax.set_xlim(0, 10.2); ax.set_ylim(-0.35, 4.45); ax.axis("off")

    widths = [8.4, 5.6, 3.4, 2.6]
    ytop, h, gap = 4.35, 0.86, 0.26
    for i, ((title, what, cost, colour), w) in enumerate(zip(stages, widths)):
        y = ytop - i * (h + gap) - h
        x = (10.2 - w) / 2
        if i:
            nxt = (10.2 - widths[i]) / 2
            prev = (10.2 - widths[i - 1]) / 2
            ax.add_patch(Polygon([[prev, y + h + gap], [prev + widths[i - 1], y + h + gap],
                                  [nxt + w, y + h], [nxt, y + h]],
                                 facecolor="#EDEFF3", edgecolor="none", zorder=0))
        box(ax, x, y, w, h, title, colour, fontsize=7)
        ax.text(x - 0.15, y + h / 2, what, ha="right", va="center", fontsize=6.5,
                color="#333333")
        if cost:
            ax.text(x + w + 0.15, y + h / 2, cost, ha="left", va="center", fontsize=6.5,
                    color=ARM_RED, fontweight="bold")

    ax.text(5.1, -0.25,
            f"Cross-encoding all 1,000,000 chunks instead: {all_pairs_h:.0f} hours per query.",
            ha="center", va="top", fontsize=7, color=ARM_RED, fontweight="bold")
    ax.set_title("Retrieve wide with the cheap model, re-score the survivors with the "
                 "expensive one")
    ax.text(5.1, -0.85,
            f"Measured on this laptop CPU with e5-small; the 1M scan is the "
            f"{n_index:,}-vector timing scaled linearly.",
            ha="center", va="top", fontsize=6, color=GREY, style="italic")
    save(fig, "l42_11_cascade")


# --- figure 12 -------------------------------------------------------------------------
def token_vectors(model, text):
    tok = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model[0].auto_model(**tok).last_hidden_state[0]
    ids = tok["input_ids"][0]
    pieces = model.tokenizer.convert_ids_to_tokens(ids)
    keep = [i for i, p in enumerate(pieces) if p not in ("<s>", "</s>", "<pad>")]
    vecs = torch.nn.functional.normalize(out[keep], dim=-1).numpy()
    labels = [pieces[i].replace("▁", "") for i in keep]
    return labels, vecs


def fig_maxsim(model):
    query = "What pressure for the Lori press?"
    doc = CHUNKS[0]
    qlab, qv = token_vectors(model, query)
    dlab, dv = token_vectors(model, doc)
    sim = qv @ dv.T
    best = sim.argmax(axis=1)
    maxsim = float(sim.max(axis=1).sum())
    for i, lab in enumerate(qlab):
        log.info("12 query token %-10s best match %-12s cos %.3f", lab, dlab[best[i]],
                 sim[i, best[i]])
    log.info("12 MaxSim total = %.3f over %d query tokens", maxsim, len(qlab))
    best_cos = {lab: round(float(sim[i, best[i]]), 2) for i, lab in enumerate(qlab)}
    got = (best_cos["Lor"], best_cos["press"], best_cos["pressure"], dlab[best[qlab.index("pressure")]])
    if got != (0.96, 0.93, 0.87, "press"):
        raise ValueError(f"MaxSim numbers changed: {got}, the frame quotes (0.96, 0.93, 0.87, 'press')")

    fig, ax = plt.subplots(figsize=(4.41, 1.82))   # drawn at slot size: 0.80 width
    im = ax.imshow(sim, cmap="Blues", vmin=float(sim.min()), vmax=1.0, aspect="auto")
    for i in range(len(qlab)):
        ax.add_patch(plt.Rectangle((best[i] - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=ARM_RED, lw=1.0))
        ax.text(best[i], i, f"{sim[i, best[i]]:.2f}", ha="center", va="center",
                fontsize=5.8, color="white", fontweight="bold")

    ax.set_xticks(range(len(dlab)))
    ax.set_xticklabels(dlab, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(qlab)))
    ax.set_yticklabels(qlab, fontsize=6.5)
    ax.set_xlabel("document tokens")
    ax.set_ylabel("query tokens")
    ax.set_title(f"Late interaction: every query token keeps its best match "
                 f"(MaxSim total {maxsim:.2f})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015, label="cosine")
    cbar.ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "l42_12_maxsim")


# --- figure 13 -------------------------------------------------------------------------
def fig_revisions(model):
    vecs = model.encode([f"passage: {r['text']}" for r in REVISIONS], normalize_embeddings=True)
    qv = model.encode([f"query: {REVISION_QUERY}"], normalize_embeddings=True)[0]
    sims = vecs @ qv
    gap = abs(float(sims[0] - sims[1]))
    for r, s in zip(REVISIONS, sims):
        log.info("13 %s (%s, %s) cosine %.4f", r["id"], r["date"], r["status"], s)
    log.info("13 gap %.4f; the %s revision scores higher",
             gap, REVISIONS[int(np.argmax(sims))]["status"])

    if round(gap, 3) != 0.008 or REVISIONS[int(np.argmax(sims))]["status"] != "superseded":
        raise ValueError(f"revision numbers changed (gap {gap:.4f}); the trapbox quotes 0.008, superseded higher")

    labels = [f"{r['id']}\n{r['date']}\n({r['status']})" for r in REVISIONS]
    colours = [ARM_RED if r["status"] == "superseded" else GREEN for r in REVISIONS]

    fig, ax = plt.subplots(figsize=(2.42, 1.55))   # drawn at slot size: 0.44 width
    bars = ax.bar(labels, sims, color=colours, width=0.4)
    ax.bar_label(bars, fmt="%.4f", fontsize=6.5, fontweight="bold", padding=2)
    ax.set_ylim(0.80, 0.90)
    ax.set_ylabel("cosine to the question")
    ax.set_title("Both revisions answer the question.\nOne of them is wrong now.", fontsize=7)
    ax.annotate(f"gap: {gap:.4f}", xy=(0.5, 0.884), ha="center", va="bottom",
                fontsize=6.5, color=GREY, fontweight="bold")
    ax.tick_params(axis="x", length=0, labelsize=6.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # No footnote: the trapbox under the figure says it ("No threshold separates those two").
    save(fig, "l42_13_revisions")
    return gap


def main():
    fig_metadata_filter()
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    fig_cascade(model)
    fig_maxsim(model)
    gap = fig_revisions(model)
    if gap > 0.05:
        raise ValueError(f"the two revisions are no longer near-identical (gap {gap:.3f}); "
                         "the frame claiming the embedder cannot separate them must change")
    log.info("done: 4 figures")


if __name__ == "__main__":
    main()
