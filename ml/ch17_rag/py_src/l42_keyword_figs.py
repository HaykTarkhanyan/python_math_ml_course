"""L42 figures 02, 03, 04 - keyword extraction on the same rambling question.

  02  RAKE, with its arithmetic visible: degree / frequency per word, summed per phrase
  03  YAKE, the five statistical features and the phrases they rank first (lower = better)
  04  KeyBERT, cosine similarity of every candidate phrase to the whole question

All REAL: produced by the implementations in l42_keywords.py, which are ours (no yake /
rake-nltk / keybert package is installed in the `ma` venv). See that module's docstring
for what is faithful and what is simplified.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l42_keyword_figs.py
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

from l42_data import RAMBLING
from l42_keywords import keybert, rake, yake

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"
MODEL_NAME = "intfloat/multilingual-e5-small"

# The three tokens that identify the chunk which answers the question. Used ONLY to colour
# bars, never to change a score, and the rule is objective (whole-token membership) rather
# than a judgement call about what is "on topic".
ANSWER_TOKENS = {"pressure", "press", "lori"}
COLOUR_NOTE = ("red = the phrase contains \"pressure\", \"press\" or \"Lori\" - "
               "the words that point at the answer")

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l42_keyword_figs.log", encoding="utf-8")],
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


def colour_for(phrase):
    return ARM_RED if set(phrase.split()) & ANSWER_TOKENS else ARM_BLUE


# --- figure 02 -------------------------------------------------------------------------
def fig_rake():
    ranked, word_table, degfreq = rake(RAMBLING, top_k=6)
    log.info("02 RAKE phrases: %s", [(p, round(s, 2)) for p, s in ranked])
    log.info("02 RAKE words  : %s",
             [(w, degfreq[w][0], degfreq[w][1], round(s, 2)) for w, s in word_table[:10]])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                            gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    # Top six by score, plus the two words the user actually cared about, wherever they
    # landed. Showing only the winners would hide the whole point.
    shown = [w for w, _ in word_table[:6]]
    for w in ("pressure", "press"):
        if w in degfreq and w not in shown:
            shown.append(w)
    names = shown[::-1]
    vals = [dict(word_table)[w] for w in names]
    bars = ax.barh(names, vals, color=[colour_for(n) for n in names], height=0.62)
    ax.bar_label(bars, labels=[f"{degfreq[w][0]}/{degfreq[w][1]} = {v:.1f}"
                               for w, v in zip(names, vals)],
                 fontsize=8.5, padding=3)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("word score = degree / frequency")
    ax.set_title("step 1: score each word", fontsize=10.5)
    ax.tick_params(axis="y", length=0, labelsize=9)

    ax = axes[1]
    names = [p for p, _ in ranked][::-1]
    vals = [s for _, s in ranked][::-1]
    bars = ax.barh(names, vals, color=[colour_for(n) for n in names], height=0.62)
    ax.bar_label(bars, fmt="%.1f", fontsize=9.5, fontweight="bold", padding=3)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("phrase score = sum of its word scores")
    ax.set_title("step 2: add them up per phrase", fontsize=10.5)
    ax.tick_params(axis="y", length=0, labelsize=9)

    fig.suptitle("RAKE: cut the text at every stopword, then reward long rare phrases",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.text(0.5, -0.03, COLOUR_NOTE, ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "l42_02_rake")


# --- figure 03 -------------------------------------------------------------------------
def fig_yake():
    ranked, term_scores, parts = yake(RAMBLING, top_k=6)
    log.info("03 YAKE phrases (lower better): %s", [(p, round(s, 4)) for p, s in ranked])
    shown = ["pressure", "press", "gauge", "maintenance", "week"]
    for w in shown:
        if w in term_scores:
            p = parts[w]
            log.info("03 YAKE term %-12s S=%.3f  case %.2f pos %.2f freq %.2f rel %.2f diff %.2f",
                     w, term_scores[w], p["case"], p["pos"], p["freq"], p["rel"], p["diff"])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                            gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    words = [w for w in shown if w in term_scores]
    vals = [term_scores[w] for w in words][::-1]
    names = words[::-1]
    bars = ax.barh(names, vals, color=[colour_for(n) for n in names], height=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=9.5, padding=3)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xlabel("single-term score (lower = more important)")
    ax.set_title("step 1: five features per term", fontsize=10.5)
    ax.tick_params(axis="y", length=0, labelsize=9)

    ax = axes[1]
    names = [p for p, _ in ranked][::-1]
    vals = [s for _, s in ranked][::-1]
    bars = ax.barh(names, vals, color=[colour_for(n) for n in names], height=0.62)
    ax.bar_label(bars, fmt="%.3f", fontsize=9.5, fontweight="bold", padding=3)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel("phrase score (lower = better)")
    ax.set_title("step 2: combine over the phrase", fontsize=10.5)
    ax.tick_params(axis="y", length=0, labelsize=9)

    fig.suptitle("YAKE: casing, position, frequency, context, spread - no training",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.text(0.5, -0.03, COLOUR_NOTE, ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "l42_03_yake")


# --- figure 04 -------------------------------------------------------------------------
def fig_keybert(model):
    top, allc = keybert(RAMBLING, model, top_k=8)
    log.info("04 KeyBERT top-8: %s", [(p, round(s, 3)) for p, s in top])
    worst = sorted(allc, key=lambda kv: kv[1])[:3]
    log.info("04 KeyBERT weakest candidates: %s", [(p, round(float(s), 3)) for p, s in worst])

    names = [p for p, _ in top][::-1]
    vals = [s for _, s in top][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.barh(names, vals, color=[colour_for(n) for n in names], height=0.64)
    ax.bar_label(bars, fmt="%.3f", fontsize=9.5, fontweight="bold", padding=3)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("cosine similarity between the phrase and the whole question")
    ax.set_title(f"KeyBERT: embed every candidate phrase, keep the closest\n"
                 f"({len(allc)} candidates, {MODEL_NAME})", fontsize=10.5)
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    fig.text(0.5, -0.03, COLOUR_NOTE, ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "l42_04_keybert")


def main():
    log.info("question: %r", RAMBLING)
    fig_rake()
    fig_yake()
    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    fig_keybert(model)
    log.info("done: 3 figures")


if __name__ == "__main__":
    main()
