"""Figures for L44 (deck 11: RAG beyond the standard pipeline).

REAL figures read their numbers from results/l44_probe.json (written by l44_probe_claims.py)
or compute them here from l44_data.py; nothing is typed in by hand. ILLUSTRATIVE figures are
schematics with no data.

  01  REAL  coverage: right and wrong suppliers in the top-k prompt, BM25 vs dense, with the
            graph traversal's result as a point on the same axes
  02  REAL  the hand-written knowledge graph, laid out in columns, traversal paths highlighted
  03  REAL  Louvain communities of that graph
  04  REAL  Personalized PageRank from the recall nodes vs plain PageRank, per supplier
  05  REAL  PPR by hand: the 7-node toy graph (the table on the slide is asserted below)
  06  REAL  contextual retrieval: the two contract clauses, bare vs with context
  07  ILLUSTRATIVE  global search as map-reduce over community summaries
  08  ILLUSTRATIVE  late chunking vs naive chunking
  09  ILLUSTRATIVE  RAPTOR's tree of summaries
  10  ILLUSTRATIVE  Corrective RAG's three branches

Run (~5 s):  ./ma/.venv/Scripts/python.exe llms/04_rag/py_src/l44_figs.py
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np

from l44_data import NODE_TYPE, RECALL_ANSWER, SUPPLIERS, TRIPLES

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#8a8a8a"
LIGHT = "#f2f2f2"

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "fig"
PROBE = HERE.parent / "results" / "l44_probe.json"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l44_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
})
METADATA = {"CreationDate": None}   # stable PDFs across redraws (see _learnings)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02, metadata=METADATA)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error("could not write %s: %s", out, e)
                raise
            log.warning("errno 22 on %s, retry %d/4", out.name, attempt)
            time.sleep(1)
    plt.close(fig)
    log.info("wrote %s", out)


def load_probe():
    if not PROBE.exists():
        raise FileNotFoundError(f"{PROBE} missing - run l44_probe_claims.py first")
    return json.loads(PROBE.read_text(encoding="utf-8"))


def graph():
    g = nx.MultiDiGraph()
    for h, r, t, src in TRIPLES:
        g.add_edge(h, t, relation=r, source=src)
    return g


# ------------------------------------------------------------------------------- 01
def fig_coverage(probe):
    """The star is GraphRAG-style local search on the graph a real LLM extracted (raw names,
    no resolution), read from results/l44_llm_graph.json - not the hand-written graph."""
    cov = probe["coverage"]
    llm_path = PROBE.parent / "l44_llm_graph.json"
    if not llm_path.exists():
        raise FileNotFoundError(f"{llm_path} missing - run l44_llm_graph.py first")
    llm = json.loads(llm_path.read_text(encoding="utf-8"))
    ls = {m: v["raw"]["local_search"] for m, v in llm.items()}
    shapes = {(v["n_chunks"], len(v["right"]), len(v["wrong"])) for v in ls.values()}
    if len(shapes) != 1:
        raise ValueError(f"extractors disagree, draw one star per model: {shapes}")
    n_chunks, right, wrong = shapes.pop()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharex=True)
    styles = {"BM25": dict(color=ARM_ORANGE, marker="o"), "dense": dict(color=ARM_BLUE, marker="s")}
    for name, st in styles.items():
        rows = cov[name]["rows"]
        k = [r["k"] for r in rows]
        axes[0].plot(k, [len(r["right_suppliers"]) for r in rows], ms=3, lw=1.3, label=name, **st)
        axes[1].plot(k, [len(r["distractors"]) for r in rows], ms=3, lw=1.3, label=name, **st)
    for ax, val in ((axes[0], right), (axes[1], wrong)):
        ax.plot([n_chunks], [val], marker="*", ms=11, color=GREEN, ls="none",
                label=f"graph, LLM-built ({n_chunks} chunks)")
    axes[0].set_title("right suppliers named in the prompt (of 4)")
    axes[1].set_title("wrong suppliers named in the prompt (of 4)")
    for ax in axes:
        ax.set_xlabel("chunks in the prompt (top $k$)")
        ax.set_ylim(-0.3, 4.4)
        ax.set_yticks(range(5))
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "l44_01_coverage")


# ------------------------------------------------------------------------------- 02
COLS = {"recall": 0.0, "incident": 1.0, "supplier": 2.15, "thing": 3.3}
TYPE_COLOR = {"recall": ARM_RED, "incident": ARM_ORANGE, "supplier": ARM_BLUE, "thing": GREY}


def layered_positions(g):
    order = {
        "recall": ["RC-2019-02", "RC-2023-07"],
        "incident": ["INC-031", "INC-034", "INC-112", "INC-118", "INC-047", "INC-063",
                     "INC-120", "INC-127"],
        "supplier": ["Kotayk Hydraulics", "Sevan Farms", "Ararat Salt", "Gyumri Packaging",
                     "Masis Logistics", "Dilijan Cultures", "Aragats Dairy", "Lori Milk Co-op"],
        "thing": ["PRS-220 press", "PRS-400 press", "antibiotic test", "raw milk",
                  "brine tank 2", "vacuum film", "brined cheese", "Lori cheese",
                  "starter culture"],
    }
    pos = {}
    for typ, names in order.items():
        n = len(names)
        ys = np.linspace(1, 0, n) if n > 2 else np.array([0.8, 0.45])
        for name, y in zip(names, ys):
            pos[name] = (COLS[typ], y)
    missing = set(g.nodes) - set(pos)
    if missing:
        raise ValueError(f"nodes without a position: {missing}")
    return pos


def fig_graph(probe):
    g = graph()
    pos = layered_positions(g)
    path_edges = set()
    for p in probe["traversal"]["paths"]:
        path_edges.add((p[0], p[1]))
        path_edges.add((p[1], p[2]))
    fig, ax = plt.subplots(figsize=(5.4, 2.75))
    for h, t, d in g.edges(data=True):
        on = (h, t) in path_edges
        (x1, y1), (x2, y2) = pos[h], pos[t]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6 if on else 0.5,
                                    color=GREEN if on else "#bbbbbb", shrinkA=9, shrinkB=9,
                                    mutation_scale=7))
    for n, (x, y) in pos.items():
        typ = NODE_TYPE[n]
        ax.scatter([x], [y], s=46, color=TYPE_COLOR[typ], zorder=3,
                   edgecolor="black" if n in RECALL_ANSWER else "none", linewidth=1.0)
        ha = "right" if typ == "recall" else "left"
        dx = -0.06 if typ == "recall" else 0.05
        ax.text(x + dx, y, n, fontsize=6.3, va="center", ha=ha, zorder=4,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
    for typ, x in COLS.items():
        ax.text(x, 1.09, {"recall": "recalls", "incident": "incidents", "supplier": "suppliers",
                          "thing": "other entities"}[typ], ha="center", fontsize=7.5,
                fontweight="bold", color=TYPE_COLOR[typ])
    ax.set_xlim(-0.75, 4.05)
    ax.set_ylim(-0.06, 1.15)
    ax.axis("off")
    save(fig, "l44_02_graph")


# ------------------------------------------------------------------------------- 03
def fig_communities(probe):
    comms = probe["communities"]["communities"]
    ug = nx.Graph(graph().to_undirected())
    palette = [ARM_BLUE, ARM_RED, ARM_ORANGE, GREEN, "#7a3fa0"]
    names = ["milk intake", "presses + 2019 recall", "packaging + 2023 recall",
             "brine and salt", "starter cultures"]
    if len(comms) != len(names):
        raise ValueError(f"expected {len(names)} communities, probe found {len(comms)}: "
                         "relabel the legend")
    # Hand-placed positions (a layout, not data); the COLOURS are the measured communities.
    pos = {
        "Lori Milk Co-op": (0.1, 3.0), "Sevan Farms": (1.0, 3.6), "raw milk": (1.6, 2.5),
        "antibiotic test": (2.7, 3.3), "INC-034": (2.3, 4.4), "Masis Logistics": (0.8, 1.4),
        "Aragats Dairy": (2.6, 1.45), "INC-047": (0.3, 0.4), "INC-120": (2.9, 0.4),
        "RC-2019-02": (4.0, 4.0), "Lori cheese": (3.7, 4.9), "INC-031": (5.1, 4.4),
        "PRS-220 press": (4.45, 2.85), "Kotayk Hydraulics": (5.95, 3.55), "PRS-400 press": (5.0, 1.8),
        "INC-127": (5.6, 0.8),
        "RC-2023-07": (7.1, 4.2), "INC-118": (8.4, 4.6), "brined cheese": (7.7, 3.2),
        "Gyumri Packaging": (9.3, 3.9), "vacuum film": (9.5, 2.9),
        "INC-112": (6.9, 1.95), "Ararat Salt": (6.9, 0.55), "brine tank 2": (7.95, 1.25),
        "Dilijan Cultures": (9.2, 1.7), "INC-063": (9.7, 0.9), "starter culture": (8.9, 0.4),
    }
    if set(pos) != set(ug.nodes):
        raise ValueError(f"position table out of date: {set(ug.nodes) ^ set(pos)}")
    fig, ax = plt.subplots(figsize=(5.4, 2.55))
    for c, color in zip(comms, palette):
        xs = [pos[n][0] for n in c]
        ys = [pos[n][1] for n in c]
        ax.add_patch(FancyBboxPatch((min(xs) - 0.45, min(ys) - 0.42),
                                    max(xs) - min(xs) + 0.9, max(ys) - min(ys) + 0.75,
                                    boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=color, alpha=0.08, ec=color, lw=0.8))
    nx.draw_networkx_edges(ug, pos, ax=ax, edge_color="#aaaaaa", width=0.7)
    for c, color, nm in zip(comms, palette, names):
        nx.draw_networkx_nodes(ug, pos, nodelist=c, node_color=color, node_size=28, ax=ax,
                               label=f"{nm} ({len(c)})")
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.14, n, fontsize=6.6, ha="center", va="top",
                bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.85))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
              fontsize=7.0, markerscale=1.3, columnspacing=1.2,
              title=f"Louvain communities, modularity {probe['communities']['modularity']:.2f}",
              title_fontsize=7.0)
    ax.set_xlim(-0.4, 10.3)
    ax.set_ylim(-0.25, 5.25)
    ax.axis("off")
    save(fig, "l44_03_communities")


# ------------------------------------------------------------------------------- 04
def fig_ppr(probe):
    scores = probe["ppr"]["scores"]
    ug = nx.Graph(graph().to_undirected())
    plain = nx.pagerank(ug, alpha=probe["ppr"]["alpha"])
    order = probe["ppr"]["supplier_order"][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
    for ax, vals, title in ((axes[0], scores, "Personalized PageRank\n(restart at the two recalls)"),
                            (axes[1], plain, "plain PageRank\n(restart anywhere)")):
        v = [vals[s] for s in order]
        colors = [ARM_BLUE if s in RECALL_ANSWER else GREY for s in order]
        bars = ax.barh(order, v, color=colors)
        ax.bar_label(bars, labels=[f"{x:.3f}" for x in v], padding=2, fontsize=6.3)
        ax.set_title(title)
        ax.set_xlim(0, max(v) * 1.35)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].text(0.98, 0.02, "blue = right answer", transform=axes[0].transAxes,
                 ha="right", fontsize=6.5, color=ARM_BLUE)
    fig.tight_layout(w_pad=1.0)
    save(fig, "l44_04_ppr")


# ------------------------------------------------------------------------------- 05
TOY_EDGES = [("R", "I1"), ("R", "I2"), ("I1", "S1"), ("I2", "S2"), ("S2", "M"), ("S3", "M")]
TOY_ALPHA = 0.5   # probability of following an edge; 1 - alpha = restart at R


def toy_ppr_steps(n_steps):
    g = nx.Graph(TOY_EDGES)
    nodes = ["R", "I1", "I2", "S1", "S2", "M", "S3"]
    p = {n: 0.0 for n in nodes}
    p["R"] = 1.0
    steps = [dict(p)]
    for _ in range(n_steps):
        new = {n: 0.0 for n in nodes}
        new["R"] += 1 - TOY_ALPHA
        for n in nodes:
            deg = g.degree(n)
            for m in g.neighbors(n):
                new[m] += TOY_ALPHA * p[n] / deg
        p = new
        steps.append(dict(p))
    return nodes, steps


def check_toy_table():
    """The slide prints steps 1 and 2 and the converged values: assert them here."""
    nodes, steps = toy_ppr_steps(200)
    s1, s2, conv = steps[1], steps[2], steps[-1]
    expect1 = {"R": 0.5, "I1": 0.25, "I2": 0.25}
    expect2 = {"R": 0.625, "I1": 0.125, "I2": 0.125, "S1": 0.0625, "S2": 0.0625}
    for n in nodes:
        if abs(s1[n] - expect1.get(n, 0.0)) > 1e-12:
            raise AssertionError(f"step 1 {n}: {s1[n]} != {expect1.get(n, 0.0)}")
        if abs(s2[n] - expect2.get(n, 0.0)) > 1e-12:
            raise AssertionError(f"step 2 {n}: {s2[n]} != {expect2.get(n, 0.0)}")
    nxp = nx.pagerank(nx.Graph(TOY_EDGES), alpha=TOY_ALPHA,
                      personalization={n: float(n == "R") for n in nodes}, tol=1e-12)
    for n in nodes:
        if abs(nxp[n] - conv[n]) > 1e-6:
            raise AssertionError(f"converged {n}: hand loop {conv[n]} vs networkx {nxp[n]}")
    log.info("toy PPR step 1 %s", {k: round(v, 4) for k, v in s1.items()})
    log.info("toy PPR step 2 %s", {k: round(v, 4) for k, v in s2.items()})
    log.info("toy PPR converged %s", {k: round(v, 3) for k, v in conv.items()})
    return conv


def fig_toy(conv):
    pos = {"R": (0, 0.5), "I1": (1, 0.85), "I2": (1, 0.15), "S1": (2, 0.85), "S2": (2, 0.15),
           "M": (3, 0.15), "S3": (4, 0.15)}
    names = {"R": "recall", "I1": "incident 1", "I2": "incident 2", "S1": "supplier 1",
             "S2": "supplier 2", "M": "raw milk", "S3": "supplier 3"}
    color = {"R": ARM_RED, "I1": ARM_ORANGE, "I2": ARM_ORANGE, "S1": ARM_BLUE, "S2": ARM_BLUE,
             "M": GREY, "S3": GREY}
    fig, ax = plt.subplots(figsize=(3.2, 1.25))
    for a, b in TOY_EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color="#999999", lw=0.9, zorder=1)
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=150, color=color[n], zorder=2)
        ax.text(x, y, n, ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", zorder=3)
        ax.text(x, y - 0.24 if y < 0.5 else y + 0.2, names[n], ha="center", fontsize=6,
                va="center")
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-0.25, 1.15)
    ax.axis("off")
    save(fig, "l44_05_toy_graph")


# ------------------------------------------------------------------------------- 06
def fig_contextual(probe):
    ctx = probe["contextual"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.9))
    width = 0.36
    for ax, retr in zip(axes, ("BM25", "dense")):
        x = np.arange(2)
        gold = [ctx[v][retr]["clause_scores"]["Sevan Farms"] for v in ("bare", "with context")]
        other = [ctx[v][retr]["clause_scores"]["Aragats Dairy"] for v in ("bare", "with context")]
        b1 = ax.bar(x - width / 2, gold, width, color=ARM_BLUE, label="Sevan Farms clause (right)")
        b2 = ax.bar(x + width / 2, other, width, color=GREY, label="Aragats Dairy clause")
        fmt = "{:.2f}" if retr == "BM25" else "{:.3f}"
        for bars in (b1, b2):
            ax.bar_label(bars, labels=[fmt.format(b.get_height()) for b in bars], padding=1.5,
                         fontsize=6.3)
        ax.set_xticks(x, ["chunk alone", "context + chunk"])
        ax.set_title(f"{retr} score")
        lo = 0 if retr == "BM25" else 0.75
        hi = max(gold + other)
        ax.set_ylim(lo, hi + (hi - lo) * 0.2)
        if retr == "dense":
            ax.text(0.02, 0.97, "axis starts at 0.75", transform=ax.transAxes, fontsize=6.5,
                    color="#555555", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "l44_06_contextual")


# ------------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, ec=None, fs=7, color="black", weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc=fc, ec=ec or fc, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def fig_mapreduce():
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    box(ax, 0.5, 1.72, 2.6, 0.26, "\"What kinds of problems keep coming up?\"", "#e8eef8",
        ARM_BLUE, fs=7.5)
    comm = ["milk intake", "presses", "packaging", "brine, salt", "cultures"]
    xs = np.linspace(-1.6, 2.6, 5)
    for x, c in zip(xs, comm):
        box(ax, x, 1.12, 0.9, 0.34, f"summary:\n{c}", LIGHT, GREY, fs=6.3)
        arrow(ax, 0.5, 1.58, x, 1.30, color="#999999", lw=0.7)
        box(ax, x, 0.55, 0.9, 0.34, "partial answer\n+ score 0-100", "#fdf3dc", ARM_ORANGE,
            fs=6.0)
        arrow(ax, x, 0.94, x, 0.73)
        arrow(ax, x, 0.37, 0.5, 0.12, color="#999999", lw=0.7)
    box(ax, 0.5, -0.02, 3.0, 0.26, "one answer from the best-scored partials", "#e3f2e8", GREEN,
        fs=7.2)
    ax.text(-2.25, 1.12, "MAP", fontsize=7.5, weight="bold", color=ARM_ORANGE, ha="right",
            va="center")
    ax.text(-2.25, 0.55, "one LLM call\nper summary", fontsize=6.5, color="#555555", ha="right",
            va="center")
    ax.text(-2.25, -0.02, "REDUCE", fontsize=7.5, weight="bold", color=GREEN, ha="right",
            va="center")
    ax.set_xlim(-2.9, 3.1)
    ax.set_ylim(-0.2, 1.9)
    ax.axis("off")
    save(fig, "l44_07_mapreduce")


def fig_late_chunking():
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 1.8))
    words = ["The", "Lori", "press", "...", "|", "It", "is", "raised", "to", "3.2", "bar"]
    for ax, title, late in ((axes[0], "naive chunking: cut first, then embed each chunk", False),
                            (axes[1], "late chunking: embed the whole document, then cut", True)):
        xs = np.linspace(0, 10, len(words))
        for x, w in zip(xs, words):
            if w == "|":
                lo = 0.82 if late else -0.1     # late chunking cuts only after encoding
                ax.plot([x, x], [lo, 1.25], color=ARM_RED, lw=1.2, ls="--")
                continue
            box(ax, x, 0.0, 0.78, 0.32, w, LIGHT, GREY, fs=6.5)
        if late:
            box(ax, 5, 0.62, 10.8, 0.3, "one encoder pass over the whole document "
                "(every token sees 'Lori')", "#e8eef8", ARM_BLUE, fs=6.5)
            for x, w in zip(xs, words):
                if w != "|":
                    arrow(ax, x, 0.17, x, 0.46, color="#999999", lw=0.6)
            ax.text(1.5, 1.02, "chunk 1 = mean of its tokens", ha="center", fontsize=6.3,
                    color=ARM_BLUE)
            ax.text(7.5, 1.02, "chunk 2 = mean of its tokens (still knows 'Lori')",
                    ha="center", fontsize=6.3, color=ARM_BLUE)
        else:
            box(ax, 1.5, 0.62, 3.8, 0.3, "encoder pass on chunk 1", "#e8eef8", ARM_BLUE, fs=6.5)
            box(ax, 7.5, 0.62, 5.8, 0.3, "encoder pass on chunk 2 (never saw 'Lori')", "#fbe3e5",
                ARM_RED, fs=6.5)
            for x, w in zip(xs, words):
                if w != "|":
                    arrow(ax, x, 0.17, x, 0.46, color="#999999", lw=0.6)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlim(-0.7, 10.9)
        ax.set_ylim(-0.25, 1.25)
        ax.axis("off")
    fig.tight_layout(h_pad=0.4)
    save(fig, "l44_08_late_chunking")


def fig_raptor():
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    leaves = ["press\n2.5 bar", "press\nmaint.", "PRS-400", "PRS-220\nrecall", "brine\n12 deg",
              "INC-112\nsalt", "vacuum\npacker", "INC-118\nfilm"]
    xs = np.linspace(0, 7, len(leaves))
    for x, t in zip(xs, leaves):
        box(ax, x, 0, 0.88, 0.42, t, LIGHT, GREY, fs=6.3)
    mids = [(0.75, "summary:\npress operation"), (3.0, "summary:\npress models"),
            (5.25, "summary:\nbrine, packaging")]
    groups = [(0, 1), (2, 3), (4, 5, 6, 7)]
    for (mx, t), grp in zip(mids, groups):
        box(ax, mx, 1.0, 1.35, 0.42, t, "#fdf3dc", ARM_ORANGE, fs=6.3)
        for i in grp:
            arrow(ax, xs[i], 0.22, mx, 0.78, color="#999999", lw=0.6)
    box(ax, 3.0, 2.0, 2.4, 0.42, "summary: production line\nand its failures", "#e8eef8",
        ARM_BLUE, fs=6.5)
    for mx, _ in mids:
        arrow(ax, mx, 1.22, 3.0, 1.78, color="#999999", lw=0.6)
    ax.text(7.75, 0, "chunks", fontsize=6.5, va="center", color="#555555")
    ax.text(7.75, 1.0, "level 1:\ncluster, summarise", fontsize=6.5, va="center", color="#555555")
    ax.text(7.75, 2.0, "level 2:\nrepeat", fontsize=6.5, va="center", color="#555555")
    ax.set_xlim(-0.5, 9.0)
    ax.set_ylim(-0.3, 2.3)
    ax.axis("off")
    save(fig, "l44_09_raptor")


def fig_crag():
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    box(ax, -0.1, 0.9, 0.9, 0.34, "retrieve\ntop-$k$", LIGHT, GREY, fs=6.5)
    box(ax, 1.4, 0.9, 1.6, 0.4, "evaluator grades\nthe chunks", "#e8eef8", ARM_BLUE, fs=6.5)
    arrow(ax, 0.35, 0.9, 0.6, 0.9)
    rows = [(1.55, "Correct", GREEN, "keep the chunks, strip\nthe irrelevant sentences"),
            (0.9, "Ambiguous", ARM_ORANGE, "keep them AND\nsearch the web"),
            (0.25, "Incorrect", ARM_RED, "throw them away,\nsearch the web")]
    for y, name, col, act in rows:
        box(ax, 3.35, y, 1.3, 0.3, name, "white", col, fs=6.5, color=col, weight="bold")
        arrow(ax, 2.2, 0.9, 2.7, y, color="#999999", lw=0.7)
        box(ax, 5.05, y, 1.85, 0.42, act, LIGHT, GREY, fs=6.2)
        arrow(ax, 4.0, y, 4.12, y, color="#999999", lw=0.7)
        arrow(ax, 5.98, y, 6.35, 0.9, color="#999999", lw=0.7)
    box(ax, 6.85, 0.9, 0.9, 0.34, "generate", "#e3f2e8", GREEN, fs=6.5)
    ax.set_xlim(-0.6, 7.4)
    ax.set_ylim(0.0, 1.8)
    ax.axis("off")
    save(fig, "l44_10_crag")


# ------------------------------------------------------------------------------- 11, 12
# Transcribed numbers (REAL, from the sources named; checked 2026-11-27).
# GraphRAG-Bench, Xiang et al. 2025, arXiv 2506.05690 (ICLR 2026), accuracy, GPT-4o-mini;
# "RAG" = vanilla RAG with reranking, "graph" = best graph method for that task and corpus.
BENCH = {
    "novels": [("facts", 60.92, 60.14, "HippoRAG 2"),
               ("reasoning\nacross passages", 42.93, 53.38, "HippoRAG 2"),
               ("summaries", 51.30, 64.40, "MS GraphRAG")],
    "medical guidelines": [("facts", 64.73, 66.28, "HippoRAG 2"),
                           ("reasoning\nacross passages", 58.64, 61.98, "HippoRAG 2"),
                           ("summaries", 65.75, 67.88, "Fast-GraphRAG")],
}
# Anthropic, "Introducing Contextual Retrieval", 19 Sep 2024: top-20 retrieval failure rate.
ANTHROPIC = [("chunks\nas they are", 5.7), ("+ context in\nembeddings", 3.7),
             ("+ context in\nBM25 too", 2.9), ("+ reranking", 1.9)]


def fig_bench():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.95), sharey=True)
    w = 0.38
    for ax, (corpus, rows) in zip(axes, BENCH.items()):
        x = np.arange(len(rows))
        rag = [r[1] for r in rows]
        gr = [r[2] for r in rows]
        b1 = ax.bar(x - w / 2, rag, w, color=GREY, label="plain RAG + reranker")
        b2 = ax.bar(x + w / 2, gr, w, color=ARM_BLUE, label="best graph method")
        for bars in (b1, b2):
            ax.bar_label(bars, labels=[f"{b.get_height():.1f}" for b in bars], padding=1.5,
                         fontsize=6.8)
        for xi, r in zip(x, rows):
            ax.text(xi + w / 2, 4, r[3], rotation=90, ha="center", va="bottom", fontsize=6.4,
                    color="white")
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_title(corpus)
        ax.set_ylim(0, 80)
    axes[0].set_ylabel("accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "l44_11_bench")


def fig_anthropic():
    fig, ax = plt.subplots(figsize=(4.0, 1.75))
    labels = [r[0] for r in ANTHROPIC]
    vals = [r[1] for r in ANTHROPIC]
    colors = [GREY, ARM_BLUE, ARM_BLUE, GREEN]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    base = vals[0]
    ax.bar_label(bars, labels=[f"{v:.1f}%" + ("" if i == 0 else f"\n(-{(1 - v / base) * 100:.0f}%)")
                               for i, v in enumerate(vals)], padding=1.5, fontsize=6.5)
    ax.set_ylabel("needed chunk not in top 20")
    ax.set_ylim(0, 7.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    save(fig, "l44_12_contextual_anthropic")


def main():
    t0 = time.time()
    probe = load_probe()
    fig_bench()
    fig_anthropic()
    fig_coverage(probe)
    fig_graph(probe)
    fig_communities(probe)
    fig_ppr(probe)
    fig_toy(check_toy_table())
    fig_contextual(probe)
    fig_mapreduce()
    fig_late_chunking()
    fig_raptor()
    fig_crag()
    log.info("done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
