"""Explanatory figures added to 08_rag_retrieval.tex after the instructor's read-through (2026-11-18).

e01_embed_limits     how much text one embedding model reads, against an LLM's context window
e02_corpus_eda       illustrative length distribution of the 4,000-document factory archive (look before chunking)
e03_chunks_arith     the real sample text, coloured chunk by chunk: fixed size, fixed size + overlap
e04_chunks_struct    the same text: by sentence, by paragraph, recursive
e05_recursive_tree   recursive splitting step by step (a real run of the algorithm on a 4-paragraph text)
e06_bow_counts       counting query words per chunk (the "dumbest possible thing")
e07_idf_curve        idf against how many chunks contain the word, with the deck's three words marked
e08_tfidf_stacked    where each chunk's tf-idf score comes from, word by word (the mirror of 06_bm25_worked)
e09_bm25_journey     one word ("press") through tf-idf and BM25, in D1 (once) and D2 (three times)
e10_points_toy       a 2-D toy: chunks as points, a question as a point, nearest wins
e11_infonce_steps    InfoNCE as a multiple-choice test: similarities -> /tau -> softmax -> -log p(correct)

Numbers: chunk text, chunking functions, BM25 constants and similarities come from the deck's own scripts
(l41_chunking_fig.py, l41_data.py, the slide "The same loss, by hand"); the script asserts the values the slides
print. e02 and e10 are illustrative (stated on the slides). e01's limits: model cards / provider docs, checked
2026-11-18 (gemini-embedding-001: 2,048 tokens, Google developers blog).

Run (repo root, ~5 s):  ./ma/Scripts/python.exe llms/04_rag/py_src/l41_explain_figs.py
"""
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
FIG = HERE.parents[1] / "fig"
LOGS = HERE.parents[3] / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOGS / "l41_explain_figs.log", encoding="utf-8")])
log = logging.getLogger("l41_explain_figs")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from l41_chunking_fig import ANSWER, PARAGRAPHS, TEXT, fixed_size, paragraphs, recursive, sentences  # noqa: E402
from l41_data import AVGDL, B, CHUNKS, CORPUS_N, DOC_FREQ, K1, WORKED_DOCS, WORKED_QUERY_TERMS, tokenize  # noqa: E402

RED, BLUE, ORANGE, GREEN, PURPLE, GREY = "#D90012", "#0033A0", "#F2A800", "#2E8B57", "#7A4FBF", "#666666"
TERM_COLOR = {"lori": RED, "press": BLUE, "bar": ORANGE}          # as in 06_bm25_worked
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    out = FIG / f"{name}.pdf"
    for attempt in range(1, 6):                     # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            break
        except OSError as e:
            if attempt == 5:
                log.error("could not write %s: %s", out, e)
                raise
            time.sleep(1)
    plt.close(fig)
    log.info("wrote %s", out.name)


# ------------------------------------------------------------------ e01
def embed_limits():
    rows = [("all-MiniLM-L6-v2 (2021)", 256), ("multilingual-e5-large (2023)", 512),
            ("gemini-embedding-001 (2025)", 2048), ("text-embedding-3-small (2024)", 8191),
            ("BGE-M3 (2024)", 8192), ("Qwen3-Embedding (2025)", 32768),
            ("an LLM's context window (2026)", 1_000_000)]
    fig, ax = plt.subplots(figsize=(3.9, 1.75))       # slot: 0.70 of the text width
    y = np.arange(len(rows))[::-1]
    for yi, (name, n) in zip(y, rows):
        llm = name.startswith("an LLM")
        ax.barh(yi, n, color=GREY if llm else BLUE, height=0.62)
        ax.text(n * 1.12, yi, f"{n:,} tokens", va="center", fontsize=7, color=GREY if llm else "black")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(100, 2.6e7)
    ax.set_xlabel("most tokens one input can have (log scale)", fontsize=7)
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_title("Embedders read thousands of tokens, not millions", fontsize=7.5)
    save(fig, "e01_embed_limits")


# ------------------------------------------------------------------ e02
def corpus_eda():
    rng = np.random.default_rng(509)
    kinds = [("maintenance logs", 1600, 300, BLUE), ("incident reports", 1100, 800, ORANGE),
             ("procedures", 1000, 2500, GREEN), ("equipment manuals", 300, 40000, RED)]
    assert sum(k[1] for k in kinds) == 4000
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    bins = np.logspace(1.3, 5.3, 40)
    data = [rng.lognormal(np.log(med), 0.55, n) for _, n, med, _ in kinds]
    ax.hist(data, bins=bins, stacked=True, color=[k[3] for k in kinds], label=[k[0] for k in kinds],
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(512, color="black", lw=1, ls="--")
    top = ax.get_ylim()[1] * 1.45
    ax.set_ylim(0, top)
    ax.text(560, top * 0.97, "512 tokens: what a\ntypical embedder reads", fontsize=6.5, va="top")
    ax.set_xlabel("document length in tokens (log scale)", fontsize=7)
    ax.set_ylabel("documents", fontsize=7)
    ax.legend(fontsize=6.3, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax.set_title("Illustrative: the 4,000-document archive", fontsize=7.5)
    save(fig, "e02_corpus_eda")


# ------------------------------------------------------------------ e03 / e04: coloured text
def text_panels(name, panels, width_in=4.9, cols=78):
    """Draw TEXT once per strategy, each character on a background of its chunk's colour."""
    fs = width_in * 72 / (0.6 * cols)                 # monospace advance is 0.6 em: one char = one column
    lines, start = [], 0                              # greedy word wrap that keeps character indices
    while start < len(TEXT):
        end = min(start + cols, len(TEXT))
        if end < len(TEXT):
            cut = TEXT.rfind(" ", start, end)
            end = cut + 1 if cut > start else end
        lines.append((start, end))
        start = end
    a0 = TEXT.index(ANSWER)
    a1 = a0 + len(ANSWER)
    line_h = 1.55
    rows_per = len(lines) * line_h + 1.6
    fig_h = rows_per * len(panels) * fs * 1.1 / 72
    fig = plt.figure(figsize=(width_in, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, cols)
    ax.set_ylim(-rows_per * len(panels), 0)
    ax.axis("off")
    for p, (label, spans, color) in enumerate(panels):
        top = -p * rows_per
        intact = any(s <= a0 and e >= a1 for s, e in spans)
        ax.text(0, top - 0.55, label, fontsize=fs * 1.05, fontweight="bold", va="center")
        ax.text(cols, top - 0.55, f"{len(spans)} chunks, " + ("answer intact" if intact else "answer split"),
                fontsize=fs * 1.0, fontweight="bold", va="center", ha="right", color=GREEN if intact else RED)
        for li, (s, e) in enumerate(lines):
            y = top - 1.6 - li * line_h
            for col, ci in enumerate(range(s, e)):
                owners = [k for k, (cs, ce) in enumerate(spans) if cs <= ci < ce]
                if owners:
                    k = owners[0]
                    alpha = 0.95 if len(owners) > 1 else (0.55 if k % 2 == 0 else 0.25)
                    ax.add_patch(Rectangle((col, y - 0.62), 1, 1.24, facecolor=color, alpha=alpha, lw=0))
                ch = TEXT[ci]
                ax.text(col + 0.5, y, "¶" if ch == "\n" else ch, fontsize=fs, family="DejaVu Sans Mono",
                        ha="center", va="center", color="white" if owners and len(owners) > 1 else "black")
                if a0 <= ci < a1:
                    ax.plot([col, col + 1], [y - 0.72, y - 0.72], color="black", lw=1.0)
    save(fig, name)


def chunk_figures():
    arith = [("Fixed size: a cut every 60 characters", fixed_size(TEXT, 60), RED),
             ("Fixed size + 15 overlap (darkest = in two chunks)", fixed_size(TEXT, 60, 15), ORANGE)]
    struct = [("By sentence", sentences(TEXT), BLUE), ("By paragraph", paragraphs(TEXT), PURPLE),
              ("Recursive, up to 150 characters", recursive(TEXT, 150), GREEN)]
    assert len(arith[0][1]) == 5 and len(arith[1][1]) == 7 and len(struct[0][1]) == 4, "counts differ from the slide's table"
    text_panels("e03_chunks_arith", arith)
    text_panels("e04_chunks_struct", struct)


# ------------------------------------------------------------------ e05
DEMO = PARAGRAPHS + ["Clean the gauge every week.", "Log each cleaning."]
CAP = 110


def recursive_trace(text, seps, cap, level, trace):
    """LangChain-style recursive splitting with a trace for the figure: split on the first separator,
    recurse into pieces that are still too big with the next one, glue small neighbours up to the cap."""
    sep, rest = seps[0], seps[1:]
    parts = [p for p in (re.split(r"(?<=[.!?]) " if sep == ". " else re.escape(sep), text)) if p]
    out, glue = [], []

    def flush():
        if glue:
            joined = (" " if sep == ". " else "\n\n").join(glue)
            out.append(joined)
            if len(glue) > 1:
                trace.append(("merge", level, list(glue), joined))
            glue.clear()

    for part in parts:
        if len(part) <= cap:
            trace.append(("fits", level, part, None))
            if glue and len((" ".join(glue + [part]))) > cap:
                flush()
            glue.append(part)
        else:
            trace.append(("big", level, part, None))
            flush()
            out.extend(recursive_trace(part, rest, cap, level + 1, trace))
    flush()
    return out


def recursive_tree():
    trace = []
    chunks = recursive_trace("\n\n".join(DEMO), ["\n\n", ". "], CAP, 1, trace)
    log.info("recursive demo: %d chunks %s", len(chunks), [len(c) for c in chunks])
    assert len(chunks) == 5 and any(t[0] == "merge" for t in trace), "the demo should split twice and merge once"

    def short(s, n=3, max_chars=12):
        words, out = s.replace("\n", " ").split()[:n], []
        for w in words:
            if out and len(" ".join(out + [w])) > max_chars:
                break
            out.append(w)
        return " ".join(out) + " ..."

    fig, ax = plt.subplots(figsize=(5.3, 2.05))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.35, 3.45)
    ax.axis("off")

    def box(x, y, label, n, ok, w=1.85, merged=False):
        ec = GREEN if ok else RED
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=ec, alpha=0.10 if not merged else 0.22, lw=0))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc="none", ec=ec, lw=1.0))
        ax.text(x, y + 0.08, label, ha="center", va="center", fontsize=6.2)
        ax.text(x, y - 0.13, f"{n} chars" + ("" if ok else ": too big"), ha="center", va="center", fontsize=6.0,
                color=ec, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.8))

    whole = "\n\n".join(DEMO)
    box(5.3, 3.12, "the whole section", len(whole), False, w=2.3)
    paras = [t for t in trace if t[1] == 1 and t[0] in ("big", "fits")]
    xs1 = [1.55, 4.6, 7.45, 9.55]
    for x, (kind, _, text, _) in zip(xs1, paras):
        box(x, 2.12, short(text, 2), len(text), kind == "fits", w=1.75)
        arrow(5.3, 2.86, x, 2.38)
    sents = [t for t in trace if t[1] == 2 and t[0] == "fits"]
    xs2 = [0.8, 2.45, 4.1, 5.75]
    parents = [xs1[0], xs1[0], xs1[1], xs1[1]]
    for x, px, (_, _, text, _) in zip(xs2, parents, sents):
        box(x, 0.95, short(text, 2), len(text), True, w=1.5)
        arrow(px, 1.86, x, 1.21)
    merged = [t for t in trace if t[0] == "merge"][0]
    box(8.5, 0.95, "Clean ... + Log ...", len(merged[3]), True, w=2.0, merged=True)
    arrow(xs1[2], 1.86, 8.2, 1.21)
    arrow(xs1[3], 1.86, 8.8, 1.21)
    ax.text(10.6, 3.12, f"cap = {CAP} characters", ha="right", va="center", fontsize=6.6, fontweight="bold")
    save(fig, "e05_recursive_tree")


# ------------------------------------------------------------------ BM25 / tf-idf helpers (same formulas as l41_bm25_figs)
def idf(term):
    n = DOC_FREQ[term]
    return float(np.log((CORPUS_N - n + 0.5) / (n + 0.5)))


def length_norm(dl):
    return (1 - B) + B * dl / AVGDL


def worked():
    rows = []
    for tag, ix in zip(["D1", "D2", "D3"], WORKED_DOCS):
        toks = tokenize(CHUNKS[ix])
        tf = {t: toks.count(t) for t in WORKED_QUERY_TERMS}
        rows.append({"tag": tag, "dl": len(toks), "tf": tf,
                     "tfidf": {t: tf[t] * idf(t) for t in WORKED_QUERY_TERMS},
                     "bm25": {t: (tf[t] / (K1 * length_norm(len(toks)) + tf[t]) * idf(t) if tf[t] else 0.0)
                              for t in WORKED_QUERY_TERMS}})
    tot = [round(sum(r["tfidf"].values()), 2) for r in rows]
    bm = [round(sum(r["bm25"].values()), 2) for r in rows]
    assert tot == [8.62, 9.65, 3.47] and bm == [4.20, 2.28, 1.49], (tot, bm)   # the slides' numbers
    return rows


def bow_counts(rows):
    desc = {"D1": "the press runs at 2.5 bar", "D2": "press maintenance schedule", "D3": "cheese brining"}
    fig, ax = plt.subplots(figsize=(3.6, 1.35))
    terms = WORKED_QUERY_TERMS
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, t in enumerate(terms):
            c = r["tf"][t]
            ax.add_patch(Rectangle((j, y), 0.95, 0.9, facecolor=TERM_COLOR[t], alpha=0.12 + 0.28 * c, lw=0))
            ax.text(j + 0.475, y + 0.45, str(c), ha="center", va="center", fontsize=8.5, fontweight="bold")
        total = sum(r["tf"].values())
        ax.text(3.35, y + 0.45, f"= {total}", va="center", fontsize=8.5, fontweight="bold",
                color=RED if total == 4 else "black")
        ax.text(-0.15, y + 0.45, f"{r['tag']}  {desc[r['tag']]}", ha="right", va="center", fontsize=7)
    for j, t in enumerate(terms):
        ax.text(j + 0.475, 3.12, t, ha="center", fontsize=7.5, family="DejaVu Sans Mono", color=TERM_COLOR[t])
    ax.text(3.35, 3.12, "total", fontsize=7.5)
    ax.text(4.05, 1.45, "D2 wins\n(wrong)", fontsize=7, color=RED, va="center")
    ax.set_xlim(-3.2, 4.9)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    save(fig, "e06_bow_counts")


def idf_curve():
    n = np.logspace(0, np.log10(CORPUS_N - 1), 300)
    fig, ax = plt.subplots(figsize=(3.0, 1.85))
    ax.plot(n, np.log((CORPUS_N - n + 0.5) / (n + 0.5)), color=GREY, lw=1.3)
    ax.axhline(0, color="black", lw=0.6)
    for t in WORKED_QUERY_TERMS:
        ax.plot(DOC_FREQ[t], idf(t), "o", color=TERM_COLOR[t], ms=5)
        off = {"lori": (-92, 6), "bar": (6, 9), "press": (7, -11)}[t]
        ax.annotate(f"{t}: {DOC_FREQ[t]} chunks, idf {idf(t):.2f}", (DOC_FREQ[t], idf(t)),
                    xytext=off, textcoords="offset points", fontsize=6.3, color=TERM_COLOR[t])
    the_n = 3900
    ax.plot(the_n, np.log((CORPUS_N - the_n + 0.5) / (the_n + 0.5)), "o", color="black", ms=4)
    ax.annotate('"the": in almost every chunk', (the_n, -3.66), xytext=(-110, 2), textcoords="offset points",
                fontsize=6.3, va="center")
    ax.set_xscale("log")
    ax.set_xlabel(f"chunks that contain the word (of N = {CORPUS_N:,})", fontsize=6.8)
    ax.set_ylabel("idf", fontsize=7)
    ax.set_title("Rare word, big weight", fontsize=7.5)
    save(fig, "e07_idf_curve")


def tfidf_stacked(rows):
    fig, ax = plt.subplots(figsize=(2.8, 1.7))        # slot: one column, about half the text width
    for i, r in enumerate(rows):
        bottom = 0.0
        for t in WORKED_QUERY_TERMS:
            v = r["tfidf"][t]
            if v:
                ax.bar(i, v, bottom=bottom, color=TERM_COLOR[t], width=0.6, edgecolor="white", linewidth=0.6)
                lab = f"{r['tf'][t]} x {idf(t):.2f}" if r["tf"][t] > 1 else f"{v:.2f}"
                ax.text(i, bottom + v / 2, lab, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
                bottom += v
        ax.text(i, bottom + 0.25, f"{bottom:.2f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(3), [f"{r['tag']}\n({r['dl']} tokens)" for r in rows], fontsize=6.8)
    ax.set_ylabel("tf-idf score", fontsize=7)
    ax.set_ylim(0, 13.5)
    for t in WORKED_QUERY_TERMS:
        ax.bar(0, 0, color=TERM_COLOR[t], label=f'"{t}"  idf {idf(t):.2f}')
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", ncol=3, handlelength=0.8, columnspacing=0.8)
    ax.set_title("tf-idf: three presses, three times the weight", fontsize=7)
    save(fig, "e08_tfidf_stacked")


def bm25_journey(rows):
    d1, d2 = rows[0], rows[1]
    t = "press"
    stages = [
        ("1. raw\ncount", d1["tf"][t], d2["tf"][t], "{:.0f}"),
        ("2. tf-idf:\ncount x idf", d1["tfidf"][t], d2["tfidf"][t], "{:.2f}"),
        ("3. BM25:\ndamped count", d1["tf"][t] / (K1 * length_norm(d1["dl"]) + d1["tf"][t]),
         d2["tf"][t] / (K1 * length_norm(d2["dl"]) + d2["tf"][t]), "{:.2f}"),
        ("4. BM25:\ndamped x idf", d1["bm25"][t], d2["bm25"][t], "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(5.3, 1.85), gridspec_kw={"wspace": 0.35})
    for ax, (title, a, b, fmt) in zip(axes, stages):
        ax.bar([0, 1], [a, b], color=[BLUE, "#7F99CF"], width=0.6)
        for x, v in [(0, a), (1, b)]:
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks([0, 1], ["D1\n1 press", "D2\n3 presses"], fontsize=6.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(a, b) * 1.3)
        ax.set_title(title, fontsize=6.7)
        ax.text(0.5, max(a, b) * 1.18, f"D2 / D1 = {b / a:.1f}x", ha="center", fontsize=6.6,
                color=RED if b / a > 2 else GREEN, fontweight="bold")
    save(fig, "e09_bm25_journey")


# ------------------------------------------------------------------ e10
def points_toy():
    pts = {"press runs at 2.5 bar": (0.82, 0.62), "press maintenance": (0.95, 0.30), "brining at 12 degrees": (0.15, 0.80),
           "ripening cellar": (0.05, 0.55), "pasteurisation": (0.35, 0.95), "vacuum packing": (0.60, 0.10)}
    q = (0.74, 0.74)
    fig, ax = plt.subplots(figsize=(3.2, 2.05))
    for name, (x, y) in pts.items():
        ax.plot(x, y, "o", color=BLUE, ms=6)
        ax.annotate(name, (x, y), xytext=(5, -9 if name == "press maintenance" else 4), textcoords="offset points",
                    fontsize=6.6)
    near = min(pts, key=lambda k: np.hypot(pts[k][0] - q[0], pts[k][1] - q[1]))
    ax.plot(*q, "*", color=RED, ms=12)
    ax.annotate('question: "What pressure\nfor the Lori press?"', q, xytext=(9, 9), textcoords="offset points",
                fontsize=6.6, color=RED)
    ax.plot([q[0], pts[near][0]], [q[1], pts[near][1]], color=RED, ls="--", lw=1)
    ax.add_patch(plt.Circle(q, np.hypot(pts[near][0] - q[0], pts[near][1] - q[1]) * 1.02, fill=False, ec=RED, lw=0.6,
                            ls=":"))
    ax.set_xlim(-0.05, 1.25)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Toy, 2 numbers per text: the nearest dot is the answer", fontsize=7.3)
    save(fig, "e10_points_toy")


# ------------------------------------------------------------------ e11
def infonce_steps():
    sims = np.array([0.834, 0.779, 0.745])            # the slide "The same loss, by hand"
    names = [r"$d^+$", r"$d_1^-$", r"$d_2^-$"]
    colors = [GREEN, GREY, GREY]

    def probs(tau):
        z = np.exp(sims / tau)
        return z / z.sum()

    # the slide's table was computed from the unrounded similarities; the rounded ones above reproduce it to 0.001
    p1, p05 = np.array([0.350, 0.331, 0.320]), np.array([0.667, 0.221, 0.113])
    assert np.abs(probs(1.0) - p1).max() <= 0.0015 and np.abs(probs(0.05) - p05).max() <= 0.0015
    panels = [("1. similarity\nto the query", sims, "{:.3f}", None),
              ("2-3. divide by $\\tau=1$,\nthen softmax", p1, "{:.3f}", 1.051),
              ("2-3. divide by $\\tau=0.05$,\nthen softmax", p05, "{:.3f}", 0.405)]
    fig, axes = plt.subplots(1, 3, figsize=(5.3, 1.95), gridspec_kw={"wspace": 0.3})
    for ax, (title, vals, fmt, loss) in zip(axes, panels):
        ax.bar(range(3), vals, color=colors, width=0.6)
        for x, v in enumerate(vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(range(3), names, fontsize=7.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.32 if loss is None else 0.85)
        ax.set_title(title, fontsize=6.9)
        if loss is not None:
            ax.set_xlabel(f"4. loss = -log {vals[0]:.3f}\n= {loss:.3f}", fontsize=6.8, fontweight="bold",
                          color=GREEN if loss < 0.6 else RED)
    save(fig, "e11_infonce_steps")


def main():
    t0 = time.perf_counter()
    FIG.mkdir(exist_ok=True)
    embed_limits()
    corpus_eda()
    chunk_figures()
    recursive_tree()
    rows = worked()
    bow_counts(rows)
    idf_curve()
    tfidf_stacked(rows)
    bm25_journey(rows)
    points_toy()
    infonce_steps()
    log.info("done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
