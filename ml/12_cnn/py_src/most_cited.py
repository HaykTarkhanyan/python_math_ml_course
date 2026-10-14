"""Nature's 20 most-cited papers of the 21st century, for the L17 deck (after "ResNet: the numbers").

Generates into ml/12_cnn/fig/:
  most_cited_top20.pdf -- one bar per paper, Nature's rank order (1 at top): the median citation
                          count across the databases that index it, a whisker from the lowest to
                          the highest database, machine-learning papers in blue.

Data: Nature's own supplementary spreadsheet for Pearson, Ledford, Hutson & Van Noorden,
"Exclusive: the most-cited papers of the twenty-first century", Nature 640, 588-592 (2025),
https://www.nature.com/magazine-assets/d41586-025-01125-9/50860378 (downloaded 2026-10-02 to
ml/12_cnn/data/nature_2025_most_cited_supplementary.xlsx). Counts are from March 2025.
Nature ranks by the MEDIAN RANK across five databases (Dimensions, Scopus, OpenAlex, Web of
Science, Google Scholar), so the bars are not sorted by count. A database Nature marked "[low]"
for a paper (it under-records it) is left out of that paper's median and range, as Nature did.

Run (~10 s on CPU, guess):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/most_cited.py
"""

import logging
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
XLSX = CH_DIR / "data" / "nature_2025_most_cited_supplementary.xlsx"

DATABASES = ["Dimensions", "Scopus", "OpenAlex", "WoS", "Google Scholar"]
TOP = 20

# Short labels by Nature's final rank (the titles in the sheet are too long for a slide).
# ml = machine learning, drawn in blue.
LABELS = {
    1: ("ResNet: deep residual learning, 2016", True),
    2: ("Real-time PCR data analysis, 2001", False),
    3: ("Thematic analysis in psychology, 2006", False),
    4: ("DSM-5 (psychiatry manual), 2013", False),
    5: ("SHELX (crystallography software), 2008", False),
    6: ("Random forests, 2001", True),
    7: ("Attention is all you need, 2017", True),
    8: ("AlexNet: ImageNet classification, 2012", True),
    9: ("Global cancer statistics 2020", False),
    10: ("Global cancer statistics 2018", False),
    11: ("PRISMA: systematic reviews, 2009", False),
    12: ("U-Net, 2015", True),
    13: ("Graphene: electric field effect, 2004", False),
    14: ("lme4: linear mixed-effects models, 2015", False),
    15: ("Scikit-learn, 2011", True),
    16: ("Deep learning (LeCun, Bengio, Hinton), 2015", True),
    17: ("Common method biases (psychology), 2003", False),
    18: ("DESeq2: RNA-seq analysis, 2014", False),
    19: ("Hallmarks of cancer, 2011", False),
    20: ("Measuring inconsistency in meta-analyses, 2003", False),
}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("most_cited")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "most_cited.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def same_paper(a: str, b: str) -> bool:
    """Databases word titles slightly differently ("Electric field in..." vs "Electric Field
    Effect in..."): call it the same paper if most words are shared."""
    wa, wb = (set(re.findall(r"[a-z0-9]+", str(t).lower())) for t in (a, b))
    return len(wa & wb) / len(wa | wb) >= 0.6


def load(log):
    """Per paper (Nature rank 1..TOP): title and its citation count in each database it counts in."""
    med = pd.read_excel(XLSX, sheet_name="Median rankings", header=None, engine="openpyxl")
    five = pd.read_excel(XLSX, sheet_name="Five databases", header=None, engine="openpyxl")
    # each database is a 4-column block (rank, title, year, count), data from row 2
    blocks = {}
    for d, db in enumerate(DATABASES):
        sub = five.iloc[2:, 4 * d:4 * d + 4].dropna(how="all")
        sub.columns = ["rank", "title", "year", "count"]
        # rows below each list hold notes, not papers; a rank we need that is missing here
        # raises a KeyError below, so nothing is skipped silently
        sub = sub[pd.to_numeric(sub["rank"], errors="coerce").notna()
                  & pd.to_numeric(sub["count"], errors="coerce").notna()]
        blocks[db] = {int(r): (t, int(c)) for r, t, _, c in sub.itertuples(index=False)}
    papers = []
    for _, row in med.iloc[2:].iterrows():
        final = row[8]
        if not isinstance(final, (int, float)) or pd.isna(final) or int(final) > TOP:
            continue
        title, counts = str(row[0]), {}
        for d, db in enumerate(DATABASES):
            rank = row[1 + d]
            if pd.isna(rank) or str(rank).strip() == "[low]":
                continue                    # not indexed, or flagged low: not in Nature's median
            t, c = blocks[db][int(rank)]
            if not same_paper(t, title):
                raise ValueError(f"title mismatch in {db} rank {rank}: {t!r} vs {title!r}")
            counts[db] = c
        if len(counts) != int(row[6]):                 # Nature's "observations for median"
            raise ValueError(f"{title!r}: {len(counts)} databases used, Nature used {row[6]}")
        papers.append((int(final), title, counts))
    papers.sort()
    if [p[0] for p in papers] != list(range(1, TOP + 1)):
        raise ValueError(f"expected ranks 1..{TOP}, got {[p[0] for p in papers]}")
    for rank, title, counts in papers:
        log.info(f"#{rank:2d} {title[:55]:55s} n={len(counts)} median={np.median(list(counts.values())):>9,.0f} "
                 f"range={min(counts.values()):,}-{max(counts.values()):,}")
    return papers


def plot(papers, log):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))   # plot area; labels add ~2 in (tight bbox)
    y = np.arange(len(papers))
    for i, (rank, _, counts) in enumerate(papers):
        label, ml = LABELS[rank]
        vals = np.array(list(counts.values())) / 1000
        m = np.median(vals)
        color = ARM_BLUE if ml else "#a8a8a8"
        ax.barh(i, m, height=0.68, color=color, zorder=2)
        ax.plot([vals.min(), vals.max()], [i, i], color="#333333", lw=0.7, zorder=3)
        ax.text(vals.max() + 4, i, f"{m:.0f}k", va="center", fontsize=6.5,
                color=ARM_BLUE if ml else "#444444", fontweight="bold" if ml else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}. {LABELS[r][0]}" for r, _, _ in papers], fontsize=6.5)
    for tick, (r, _, _) in zip(ax.get_yticklabels(), papers):
        if LABELS[r][1]:
            tick.set_color(ARM_BLUE); tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 400)
    ax.set_xlabel("citations, thousands\n(bar: median; line: range across databases)",
                  fontsize=6.5, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = FIG_DIR / "most_cited_top20.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    plot(load(log), log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
