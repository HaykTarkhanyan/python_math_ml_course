"""Explanatory figures for the keyword section of 09_rag_hybrid_rerank.tex (instructor's read-through, 2026-11-19).

k01_noise_words     BM25 on the rambling question: which words make up the score of the winning (wrong) chunk and of
                    the answer chunk
k02_rake_split      RAKE step 1 on the same question: stopwords and punctuation are the cuts, what is left are phrases
k03_rake_degree     RAKE step 2: word co-occurrence inside phrases; degree = row sum, score = degree / frequency
k04_keybert_toy     KeyBERT as a picture: the message and its candidate phrases as points, closest phrases win
                    (illustrative coordinates; the real run is l42_04_keybert.pdf)

Real numbers: k01-k03 run the deck's own code (l42_data.bm25_scores' formula, l42_keywords.rake) on l42_data.RAMBLING.
Run (repo root, ~3 s):  ./ma/Scripts/python.exe llms/05_rag/py_src/l42_explain_figs.py
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
                              logging.FileHandler(LOGS / "l42_explain_figs.log", encoding="utf-8")])
log = logging.getLogger("l42_explain_figs")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from l41_data import B, CHUNKS, K1, tokenize  # noqa: E402
from l42_data import RAMBLING, RAMBLING_TARGET, bm25_scores, corpus_tokens, ranks_from_scores  # noqa: E402
from l42_keywords import STOPWORDS, rake  # noqa: E402

RED, BLUE, ORANGE, GREEN, GREY = "#D90012", "#0033A0", "#F2A800", "#2E8B57", "#666666"
QUESTION_PART = "what pressure is the Lori press actually supposed to run at"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    out = FIG / f"{name}.pdf"
    for attempt in range(1, 6):
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


# ------------------------------------------------------------------ k01
def per_term(query, docs, i, k1=K1, b=B):
    """Per-query-word BM25 contributions to document i; same formula as l42_data.bm25_scores."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    norm = (1 - b) + b * len(docs[i]) / avgdl
    out = {}
    for term in tokenize(query):                       # a repeated query word counts each time, as in bm25_scores
        tf = docs[i].count(term)
        if tf == 0:
            continue
        df = sum(1 for d in docs if term in d)
        out[term] = out.get(term, 0.0) + tf / (k1 * norm + tf) * np.log(1 + (n - df + 0.5) / (df + 0.5))
    return out


def noise_words():
    docs = corpus_tokens()
    scores = bm25_scores(RAMBLING, docs)
    ranks = ranks_from_scores(scores)
    top = int(np.argmax(scores))
    assert ranks[RAMBLING_TARGET] == 2, ranks[RAMBLING_TARGET]          # the slide: "already reached rank 2"
    question_words = {t for t in tokenize(QUESTION_PART) if t not in STOPWORDS}
    cols = [(top, "rank 1\n(wrong): press\nmaintenance chunk"), (RAMBLING_TARGET, "rank 2\n(right): the\nanswer chunk")]
    group_color = {"question": RED, "story": "#9A9A9A", "filler": "#DCDCDC"}
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    for x, (i, label) in enumerate(cols):
        contrib = per_term(RAMBLING, docs, i)
        assert abs(sum(contrib.values()) - scores[i]) < 1e-9
        group = {t: "question" if t in question_words else "filler" if t in STOPWORDS else "story" for t in contrib}
        order = sorted(contrib, key=lambda t: (["question", "story", "filler"].index(group[t]), -contrib[t]))
        bottom = 0.0
        for t in order:
            v = contrib[t]
            ax.bar(x, v, bottom=bottom, color=group_color[group[t]], edgecolor="white", width=0.62, linewidth=0.6)
            ax.text(x, bottom + v / 2, f"{t} {v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if group[t] == "question" else "black", style="italic" if group[t] == "filler" else "normal")
            bottom += v
        ax.text(x, bottom + 0.08, f"{scores[i]:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        log.info("%s  score %.3f  %s", label.replace(chr(10), " "), scores[i], {t: round(v, 2) for t, v in contrib.items()})
    ax.set_xticks(range(2), [c[1] for c in cols], fontsize=6.6)
    ax.set_ylabel("BM25 score", fontsize=7)
    ax.set_xlim(-0.55, 2.6)
    for k, (g, c) in enumerate(group_color.items()):
        ax.add_patch(Rectangle((1.55, 5.6 - k * 0.55), 0.18, 0.35, facecolor=c, lw=0))
        ax.text(1.8, 5.77 - k * 0.55, {"question": "a word of the\nactual question", "story": "a word of the story",
                                        "filler": "filler (the, in, at)"}[g], fontsize=6.3, va="center")
    save(fig, "k01_noise_words")


# ------------------------------------------------------------------ k02
def rake_spans(text):
    """RAKE's phrase split (l42_keywords.rake) with character positions, for drawing."""
    phrases = []
    for sm in re.finditer(r"[^.!?]+[.!?]*", text):
        cur = None
        for tm in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]*|[,;:]", sm.group(0)):
            s, e = sm.start() + tm.start(), sm.start() + tm.end()
            tok = tm.group(0)
            if tok in {",", ";", ":"} or tok.lower() in STOPWORDS:
                if cur:
                    phrases.append(cur)
                    cur = None
            else:
                cur = (cur[0], e) if cur else (s, e)
        if cur:
            phrases.append(cur)
    return phrases


def rake_split():
    spans = rake_spans(RAMBLING)
    _, _, degfreq = rake(RAMBLING, top_k=100)
    words_drawn = {w.lower() for s, e in spans for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", RAMBLING[s:e])}
    assert words_drawn == set(degfreq), words_drawn ^ set(degfreq)
    width_in, cols = 5.0, 74
    fs = width_in * 72 / (0.6 * cols)
    lines, start = [], 0
    while start < len(RAMBLING):
        end = min(start + cols, len(RAMBLING))
        if end < len(RAMBLING):
            end = RAMBLING.rfind(" ", start, end) + 1
        lines.append((start, end))
        start = end
    lh = 1.7
    fig = plt.figure(figsize=(width_in, (len(lines) * lh + 0.4) * fs * 1.1 / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, cols)
    ax.set_ylim(-len(lines) * lh - 0.2, 0.2)
    ax.axis("off")
    colors = [BLUE, ORANGE]
    for li, (s, e) in enumerate(lines):
        y = -li * lh - 0.8
        for col, ci in enumerate(range(s, e)):
            k = next((k for k, (ps, pe) in enumerate(spans) if ps <= ci < pe), None)
            if k is not None:
                ax.add_patch(Rectangle((col, y - 0.65), 1, 1.3, facecolor=colors[k % 2], alpha=0.30, lw=0))
            ax.text(col + 0.5, y, RAMBLING[ci], fontsize=fs, family="DejaVu Sans Mono", ha="center", va="center",
                    color="black" if k is not None else "#9A9A9A")
    save(fig, "k02_rake_split")
    return spans


# ------------------------------------------------------------------ k03
def rake_degree():
    words = ["second", "pressing", "stage", "hydraulic", "gauge", "lori", "press", "pressure"]
    spans = rake_spans(RAMBLING)
    phrases = [[w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", RAMBLING[s:e])] for s, e in spans]
    _, _, degfreq = rake(RAMBLING, top_k=100)
    n = len(words)
    m = np.zeros((n, n), dtype=int)
    for ph in phrases:
        for i, a in enumerate(words):
            for j, b_ in enumerate(words):
                if a in ph and b_ in ph:
                    m[i, j] += 1
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            ax.add_patch(Rectangle((j, n - 1 - i), 0.95, 0.95, facecolor=BLUE if v else "white",
                                   alpha=0.15 + 0.45 * (i == j) if v else 1, edgecolor="#DDDDDD", lw=0.4))
            if v:
                ax.text(j + 0.475, n - 1 - i + 0.47, str(v), ha="center", va="center", fontsize=7)
        deg, freq = degfreq[words[i]]
        full_deg = deg                                   # degree counts every word in the phrase, matrix shows a subset
        ax.text(n + 0.25, n - 1 - i + 0.47, f"{full_deg}", va="center", fontsize=7.5, fontweight="bold")
        ax.text(n + 1.25, n - 1 - i + 0.47, f"{freq}", va="center", fontsize=7.5)
        ax.text(n + 2.2, n - 1 - i + 0.47, f"{deg / freq:.0f}", va="center", fontsize=7.5, fontweight="bold",
                color=RED if words[i] in ("lori", "press", "pressure") else BLUE)
    for j, w in enumerate(words):
        ax.text(j + 0.475, n + 0.15, w, rotation=60, ha="left", va="bottom", fontsize=6.5)
        ax.text(-0.15, n - 1 - j + 0.47, w, ha="right", va="center", fontsize=6.8,
                color=RED if w in ("lori", "press", "pressure") else "black")
    for x, lab in [(n + 0.25, "degree"), (n + 1.25, "freq"), (n + 2.2, "score")]:
        ax.text(x, n + 0.15, lab, rotation=60, ha="left", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xlim(-2.4, n + 3.0)
    ax.set_ylim(-0.1, n + 2.2)
    ax.axis("off")
    save(fig, "k03_rake_degree")


# ------------------------------------------------------------------ k04
def keybert_toy():
    msg = (0.55, 0.55)
    pts = {"second pressing stage": (0.62, 0.66), "pressure": (0.47, 0.64), "lori press": (0.66, 0.47),
           "hydraulic gauge": (0.24, 0.40), "maintenance report": (0.80, 0.25), "last week": (0.18, 0.88),
           "sorry": (0.10, 0.15)}
    dist = {k: np.hypot(x - msg[0], y - msg[1]) for k, (x, y) in pts.items()}
    top3 = sorted(dist, key=dist.get)[:3]
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    for k, (x, y) in pts.items():
        ax.plot(x, y, "o", color=GREEN if k in top3 else GREY, ms=6)
        ax.annotate(k, (x, y), xytext=(-38, 4) if k == "pressure" else (5, 3), textcoords="offset points", fontsize=6.6,
                    color=GREEN if k in top3 else "black", fontweight="bold" if k in top3 else "normal")
    ax.plot(*msg, "*", color=RED, ms=12)
    ax.annotate("the whole message", msg, xytext=(-82, -3), textcoords="offset points", fontsize=6.6, color=RED)
    ax.add_patch(plt.Circle(msg, max(dist[k] for k in top3) * 1.08, fill=False, ec=GREEN, lw=0.7, ls=":"))
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0.05, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Illustrative: the 3 phrases closest to the message win", fontsize=7.2)
    save(fig, "k04_keybert_toy")


# ------------------------------------------------------------------ k05
def multiquery_flow():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(5.3, 1.9))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=6.6, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, alpha=0.13, lw=0))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="none", ec=color, lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.8))

    box(0.9, 1.8, 1.6, 0.9, "the user's\nquestion", RED, bold=True)
    box(2.65, 1.8, 1.2, 0.9, "LLM:\n2 rewrites", BLUE)
    arrow(1.7, 1.8, 2.05, 1.8)
    queries = ["original: what pressure is the\nLori press supposed to run at?", "Lori press operating\npressure in bar",
               "pressing stage settings\nfor Lori cheese"]
    ys = [3.0, 1.8, 0.6]
    for y, q in zip(ys, queries):
        box(5.0, y, 3.3, 0.9, q, GREY, fs=6.2)
        arrow(3.25, 1.8, 3.35, y)
        box(7.4, y, 0.95, 0.6, "search", GREEN, fs=6.4)
        arrow(6.65, y, 6.92, y)
        arrow(7.88, y, 8.62, 1.8)
    box(9.55, 1.8, 1.85, 1.1, "fuse the\nlists (RRF)\n-> top k", ORANGE, bold=True)
    ax.text(0.9, 0.55, "the original always\nstays in the set", ha="center", fontsize=6.2, color=GREY, style="italic")
    save(fig, "k05_multiquery_flow")


# ------------------------------------------------------------------ k06
def two_spaces():
    texts = ["press at 2.5 bar", "press maintenance", "brining", "ripening cellar"]
    a = [(0.62, 0.72), (0.45, 0.42), (0.10, 0.80), (0.12, 0.18)]
    b = [(0.12, 0.22), (0.12, 0.80), (0.80, 0.45), (0.62, 0.93)]
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.3), gridspec_kw={"wspace": 0.3})   # slot: 0.65 of the text width
    for ax, pts, title, c in [(axes[0], a, "model A (old index)", BLUE), (axes[1], b, "model B (new model)", GREEN)]:
        for t, (x, y) in zip(texts, pts):
            ax.plot(x, y, "o", color=c, ms=5)
            ax.annotate(t, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
        ax.set_xlim(0, 1.35)
        ax.set_ylim(0.05, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7.2, color=c)
    fig.text(0.5, -0.04, "same texts, two maps: a cosine across the maps means nothing", ha="center", va="top",
             fontsize=6.8, color=RED, fontweight="bold")
    save(fig, "k06_two_spaces")


def main():
    t0 = time.perf_counter()
    noise_words()
    rake_split()
    rake_degree()
    keybert_toy()
    multiquery_flow()
    two_spaces()
    log.info("done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
