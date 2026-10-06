"""BM25 figures for L41 (RAG retrieval): figures 04-07.

Figures 04 and 05 plot the BM25 component functions exactly - pure maths, nothing assumed.
Figures 06 and 07 rank three real chunks from the cheese-factory corpus. The corpus
statistics they use (N=4000, document frequencies, average length) describe a hypothetical
factory archive and are stated in l41_data.py; the BM25 arithmetic on top of them is exact
and is logged so the slide can quote verified numbers.

Run:  ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_bm25_figs.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from l41_data import (
    AVGDL, B, CHUNKS, CORPUS_N, DOC_FREQ, K1, WORKED_DOCS, WORKED_QUERY_TERMS, tokenize,
)

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "l41_bm25_figs.log", encoding="utf-8")],
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

DOC_TAGS = ["D1", "D2", "D3"]


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


# --- BM25 itself -----------------------------------------------------------------------
def idf(term):
    """Robertson-Sparck Jones weight with no relevance information, eq (3.2) reduced form."""
    n = DOC_FREQ[term]
    return np.log((CORPUS_N - n + 0.5) / (n + 0.5))


def length_norm(doc_len):
    """B = (1-b) + b * dl/avdl   -- Robertson & Zaragoza eq (3.12)."""
    return (1 - B) + B * doc_len / AVGDL


def term_score(tf, doc_len, term):
    """One term's BM25 contribution -- eq (3.15)."""
    if tf == 0:
        return 0.0
    return tf / (K1 * length_norm(doc_len) + tf) * idf(term)


def score_worked_docs():
    """Score the three worked-example chunks, returning per-term contributions."""
    rows = []
    for tag, ix in zip(DOC_TAGS, WORKED_DOCS):
        toks = tokenize(CHUNKS[ix])
        dl = len(toks)
        contribs = {t: term_score(toks.count(t), dl, t) for t in WORKED_QUERY_TERMS}
        raw = sum(toks.count(t) for t in WORKED_QUERY_TERMS)
        # Plain tf-idf: rarity weighting only. No saturation, no length correction.
        tfidf = sum(toks.count(t) * idf(t) for t in WORKED_QUERY_TERMS)
        rows.append({
            "tag": tag, "dl": dl, "B": length_norm(dl), "raw": raw, "tfidf": tfidf,
            "tf": {t: toks.count(t) for t in WORKED_QUERY_TERMS},
            "contribs": contribs, "total": sum(contribs.values()),
        })
    return rows


# --- figure 04 -------------------------------------------------------------------------
def fig_saturation():
    """tf saturation for several k1, against the linear 'more is proportionally better' line."""
    tf = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(2.85, 1.38))

    ax.plot(tf, tf / tf.max() * 1.0, color=GREY, ls=":", lw=1.1,
            label="raw count (scaled)")
    for k1, color in zip([0.5, 1.5, 3.0], [ARM_RED, ARM_BLUE, ARM_ORANGE]):
        ax.plot(tf, tf / (k1 + tf), color=color, lw=1.4, label=f"$k_1$ = {k1}")

    ax.axhline(1.0, color=GREY, lw=0.8, alpha=0.5)
    ax.text(0.3, 1.02, "ceiling", ha="left", fontsize=6.5, color=GREY)
    ax.set_xlabel("term frequency in the document  ($tf$)")
    ax.set_ylabel(r"saturation  $tf\,/\,(k_1 + tf)$")
    ax.set_title("The 10th mention adds almost nothing")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", frameon=True, framealpha=1, facecolor="white", edgecolor="none")
    save(fig, "04_bm25_saturation")


# --- figure 05 -------------------------------------------------------------------------
def fig_length_norm():
    """The B factor against relative document length, for several b."""
    ratio = np.linspace(0, 3, 400)
    fig, ax = plt.subplots(figsize=(2.62, 1.38))

    for b, color, style in zip([0.0, 0.5, 0.75, 1.0],
                               [GREY, ARM_ORANGE, ARM_BLUE, ARM_RED],
                               ["--", "-", "-", "-"]):
        ax.plot(ratio, (1 - b) + b * ratio, color=color, lw=1.4, ls=style,
                label=f"$b$ = {b}" + ("  (off)" if b == 0 else "  (full)" if b == 1 else ""))

    ax.plot([1.0, 1.0], [0, 1.0], color=GREY, lw=0.8, alpha=0.6)   # up to the crossing, clear of the legend
    ax.text(1.06, 0.08, "average-length\ndocument", fontsize=6.5, color=GREY, va="bottom")
    ax.set_xlabel(r"relative length  $dl\,/\,avdl$")
    ax.set_ylabel("penalty factor  $B$")
    ax.set_title("Longer documents are discounted")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "05_bm25_length_norm")


# --- figure 06 -------------------------------------------------------------------------
def fig_worked_contributions(rows):
    """Per-term BM25 contributions stacked, for the three worked chunks."""
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    colors = {"lori": ARM_RED, "press": ARM_BLUE, "bar": ARM_ORANGE}

    bottoms = np.zeros(len(rows))
    for term in WORKED_QUERY_TERMS:
        vals = np.array([r["contribs"][term] for r in rows])
        bars = ax.bar([r["tag"] for r in rows], vals, bottom=bottoms, color=colors[term],
                      edgecolor="white", linewidth=0.8,
                      label=f'"{term}"  (idf {idf(term):.2f})')
        for bar, v in zip(bars, vals):
            if v > 0.25:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2,
                        f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
        bottoms += vals

    for x, r in enumerate(rows):
        ax.text(x, r["total"] + 0.12, f'{r["total"]:.2f}', ha="center",
                fontsize=7, fontweight="bold")
        ax.text(x, -0.12, f'{r["dl"]} tokens\nB = {r["B"]:.2f}', ha="center", va="top",
                fontsize=6.5, color=GREY)

    ax.set_ylabel("BM25 score")
    ax.set_title('Query: "lori press bar" - who wins, and why')
    ax.set_ylim(0, max(r["total"] for r in rows) * 1.22)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.tick_params(axis="x", length=0, pad=21)
    save(fig, "06_bm25_worked")


# --- figure 07 -------------------------------------------------------------------------
def fig_count_vs_bm25(rows):
    """Three scorers side by side: counting and tf-idf both rank D2 first. Only BM25 is right."""
    fig, axes = plt.subplots(1, 3, figsize=(4.4, 1.55))
    tags = [r["tag"] for r in rows]

    panels = [
        ("raw", "%d", "Count the query words", "query-term occurrences"),
        ("tfidf", "%.2f", "tf-idf\n(rare words weighted)", "tf-idf score"),
        ("total", "%.2f", "BM25\n(+ saturation, + length)", "BM25 score"),
    ]
    for ax, (key, fmt, title, ylabel) in zip(axes, panels):
        vals = [r[key] for r in rows]
        win = int(np.argmax(vals))
        correct = tags[win] == "D1"
        color = ARM_BLUE if correct else ARM_RED
        bars = ax.bar(tags, vals, color=[color if i == win else "#BBBBBB"
                                         for i in range(len(rows))])
        ax.bar_label(bars, fmt=fmt, fontsize=6.5, fontweight="bold", padding=1)
        verdict = "gets it right" if correct else "gets it wrong"
        ax.set_title(f"{title}\n({verdict})", fontsize=6.5,
                     color=("#2E8B57" if correct else ARM_RED))
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.set_ylim(0, max(vals) * 1.32)
        ax.tick_params(axis="x", length=0)

    fig.text(0.5, -0.06,
             "D1 is the chunk that actually answers the question.",
             ha="center", fontsize=6.5, color=GREY)
    fig.tight_layout()
    save(fig, "07_count_vs_bm25")


def main():
    rows = score_worked_docs()

    log.info("BM25 parameters: k1=%.2f b=%.2f avdl=%.1f N=%d", K1, B, AVGDL, CORPUS_N)
    for term in WORKED_QUERY_TERMS:
        log.info("  idf(%-6s) df=%4d -> %.4f", term, DOC_FREQ[term], idf(term))
    for r in rows:
        log.info("  %s dl=%2d B=%.4f tf=%s raw=%d tfidf=%.4f bm25=%.4f",
                 r["tag"], r["dl"], r["B"], r["tf"], r["raw"], r["tfidf"], r["total"])

    raw_rank = [r["tag"] for r in sorted(rows, key=lambda r: -r["raw"])]
    tfidf_rank = [r["tag"] for r in sorted(rows, key=lambda r: -r["tfidf"])]
    bm_rank = [r["tag"] for r in sorted(rows, key=lambda r: -r["total"])]
    log.info("ranking by raw count : %s", " > ".join(raw_rank))
    log.info("ranking by tf-idf    : %s", " > ".join(tfidf_rank))
    log.info("ranking by BM25      : %s", " > ".join(bm_rank))
    if bm_rank[0] != "D1":
        raise ValueError(f"worked example broken: BM25 ranks {bm_rank[0]} first, expected D1")
    if raw_rank[0] == "D1":
        raise ValueError("worked example pointless: raw counting already ranks D1 first")

    # "One chunk, scored by hand" prints these, rounded as on the slide. Check the printed
    # arithmetic too: 0.4878 x the 3-decimal idf must round to the printed contribution.
    d1 = rows[0]
    sat = 1 / (K1 * length_norm(d1["dl"]) + 1)
    printed = {"lori": 1.694, "press": 1.099, "bar": 1.410}
    got = {"B": round(length_norm(d1["dl"]), 2), "sat": round(sat, 4), "total": round(d1["total"], 3)}
    if got != {"B": 0.70, "sat": 0.4878, "total": 4.203} or d1["dl"] != 12:
        raise ValueError(f"by-hand frame numbers changed: {got}, dl {d1['dl']}")
    for term, value in printed.items():
        if d1["tf"][term] != 1:
            raise ValueError(f"by-hand frame assumes tf = 1 for {term}")
        if round(sat * idf(term), 3) != value or round(0.4878 * round(idf(term), 3), 3) != value:
            raise ValueError(f"by-hand frame: {term} contribution no longer prints as {value}")
    if round(sum(printed.values()), 3) != 4.203:
        raise ValueError("by-hand frame: the printed contributions do not add up to 4.203")
    log.info("by-hand frame checked: B 0.70, saturation 0.4878, D1 = 4.203")

    fig_saturation()
    fig_length_norm()
    fig_worked_contributions(rows)
    fig_count_vs_bm25(rows)
    log.info("done: 4 figures")


if __name__ == "__main__":
    main()
