"""Text-modality figures for Interpretability Deck 3 (24_shap_lime).

Same two methods the deck already teaches on tables, now on documents:

  text_shap_tokens.pdf     : token-level SHAP on ONE short post (headers stripped).
                             base value + sum of per-word SHAP = this document's score,
                             i.e. the deck's identity with words as the features.
  text_lime_newsgroups.pdf : LIME on the SAME classifier trained WITH the newsgroup
                             headers left in -> the clever-Hans reproduction of
                             Ribeiro et al. (2016): the top words are mail-header
                             artifacts, and stripping the headers costs real accuracy.

Model: TF-IDF -> logistic regression (the course's own classifier from ch.3), on the
alt.atheism vs soc.religion.christian subset of 20 newsgroups. Deliberately a "glass
box" model class: at ~16k vocabulary columns (headers stripped) you still cannot read
it per document, which is the point the deck makes.

Measured, seed 509: headers kept -> test acc 0.9066; headers/footers/quotes stripped
-> 0.7908. The 11.6-point gap is the artifact the deck quotes.

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_text_figs.py
"""
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_footer, strip_newsgroup_header, strip_newsgroup_quoting)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEED = 509
CATS = ["alt.atheism", "soc.religion.christian"]
SHORT = ["atheism", "christian"]          # short labels for the plots
RED, BLUE = "#D90012", "#0033A0"          # class 0 (atheism) / class 1 (christian)

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
# Read the two categories straight out of the extracted archive. fetch_20newsgroups() would
# re-read all 18846 files to build its cache, which takes tens of minutes on this filesystem;
# we only need ~1800 of them.
NEWS_HOME = Path.home() / "scikit_learn_data" / "20news_home"

log = logging.getLogger("interp_text")


class Split:
    """Minimal stand-in for the sklearn Bunch: .data (list of str) and .target (array)."""

    def __init__(self, data, target):
        self.data = data
        self.target = np.asarray(target)


def load_split(split, remove):
    """Load one bydate split for our two categories, applying sklearn's own strip functions."""
    base = NEWS_HOME / f"20news-bydate-{split}"
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} is missing. Fetch it once with\n"
            f"  python -c \"from sklearn.datasets import fetch_20newsgroups as f; "
            f"f(subset='train', categories={CATS})\"")
    texts, targets = [], []
    for label, cat in enumerate(CATS):
        paths = sorted((base / cat).iterdir())
        if not paths:
            raise FileNotFoundError(f"{base / cat} is empty")
        for fp in paths:
            t = fp.read_text(encoding="latin-1")
            if "headers" in remove:
                t = strip_newsgroup_header(t)
            if "footers" in remove:
                t = strip_newsgroup_footer(t)
            if "quotes" in remove:
                t = strip_newsgroup_quoting(t)
            texts.append(t)
            targets.append(label)
    order = np.random.RandomState(SEED).permutation(len(texts))
    return Split([texts[k] for k in order], [targets[k] for k in order])


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOGS_DIR / "interp_text_figs.log", mode="w", encoding="utf-8")],
    )


def fit(remove):
    """TF-IDF + logistic regression on the 2-class newsgroup subset.

    remove=() keeps the mail headers/footers/quotes; remove=('headers',...) strips them.
    Returns (pipeline, test_accuracy, train_split, test_split, vocab_size).
    """
    tr = load_split("train", remove)
    te = load_split("test", remove)
    pipe = make_pipeline(
        TfidfVectorizer(lowercase=True, sublinear_tf=True),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    pipe.fit(tr.data, tr.target)
    acc = pipe.score(te.data, te.target)
    vocab = len(pipe.named_steps["tfidfvectorizer"].vocabulary_)
    log.info(f"remove={remove or '()'}: train={len(tr.data)} test={len(te.data)} "
             f"vocab={vocab} test_acc={acc:.4f}")
    return pipe, acc, tr, te, vocab


def token_skew(train_split, token):
    """Share of training posts containing `token`, and how many of those are the atheism class."""
    hits = [t for d, t in zip(train_split.data, train_split.target)
            if re.search(rf"\b{re.escape(token)}\b", d, flags=re.I)]
    share = len(hits) / len(train_split.data)
    atheism = float(np.mean([t == 0 for t in hits])) if hits else float("nan")
    return share, atheism


def posting_stat(train_split):
    """Ribeiro's headline artifact: how often does the token 'posting' appear, and in which class?"""
    share, atheism = token_skew(train_split, "posting")
    log.info(f"token 'posting': in {share:.1%} of training posts; {atheism:.1%} of those are atheism")
    return share, atheism


# ---------------------------------------------------------------- figure 1: SHAP on text
def pick_content_doc(pipe, tr, te):
    """A short post whose prediction is driven by ordinary WORDS, not usernames or hostnames.

    SHAP on this corpus otherwise lands on a signature block (`bobbe@`, `ICO.`, `TEK.`), which
    demonstrates nothing about reading token attributions. We rank candidates cheaply with the
    linear model's own per-token contributions (w_j * x_j - no SHAP needed) and require the top
    contributors to be common words: alphabetic, >= 4 chars, and present in >= MIN_DF training
    posts, which is what rules out the rare proper nouns.
    """
    MIN_DF = 25
    vec = pipe.named_steps["tfidfvectorizer"]
    clf = pipe.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    doc_freq = np.asarray((vec.transform(tr.data) > 0).sum(0)).ravel()
    common = np.array([t.isalpha() and len(t) >= 4 for t in vocab]) & (doc_freq >= MIN_DF)

    n_tok = np.array([len(re.findall(r"\w+", d)) for d in te.data])
    proba = pipe.predict_proba(te.data)[:, 1]
    ok = pipe.predict(te.data) == te.target
    cand = np.where((n_tok >= 30) & (n_tok <= 90) & ok & (np.abs(proba - 0.5) > 0.25))[0]
    if len(cand) == 0:
        raise RuntimeError("no short, confident, correctly-classified test post found")

    contrib = vec.transform([te.data[k] for k in cand]).multiply(clf.coef_[0]).toarray()
    best, best_score = None, -1
    for row, k in zip(contrib, cand):
        top = np.argsort(-np.abs(row))[:8]
        score = int(common[top].sum())
        if score > best_score:
            best, best_score = int(k), score
    log.info(f"SHAP doc search: {len(cand)} candidates, best has {best_score}/8 common words")
    return best


def fig_shap_tokens(pipe_clean, tr_clean, te_clean):
    """Token-level SHAP for one short post, drawn as a signed bar + the sum identity."""
    def f(texts):
        return pipe_clean.predict_proba(list(texts))[:, 1]     # P(christian)

    i = pick_content_doc(pipe_clean, tr_clean, te_clean)
    doc = te_clean.data[i]
    proba_i = float(pipe_clean.predict_proba([doc])[0, 1])
    truth = SHORT[int(te_clean.target[i])]
    n_words = len(re.findall(r"\w+", doc))
    log.info(f"SHAP doc idx={i} tokens={n_words} true={truth} P(christian)={proba_i:.3f}")
    log.info(f"SHAP doc text: {doc!r}")

    explainer = shap.Explainer(f, shap.maskers.Text(r"\W+"), seed=SEED)
    sv = explainer([doc])[0]
    base, total = float(sv.base_values), float(sv.values.sum())
    log.info(f"base={base:.4f} sum(phi)={total:.4f} base+sum={base + total:.4f} f(x)={proba_i:.4f}")

    order = np.argsort(-np.abs(sv.values))[:11]
    order = order[np.argsort(sv.values[order])]
    words = [str(sv.data[k]).strip().strip(".,;:!?\"'()") or "_" for k in order]
    vals = sv.values[order]
    for w, v in zip(reversed(words), reversed(vals)):
        log.info(f"    {w:>14s} {v:+.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.barh(range(len(vals)), vals, color=[BLUE if v > 0 else RED for v in vals],
            height=0.72, alpha=0.9)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(words, fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("SHAP value  $\\phi_j$   (effect on $P(\\mathrm{christian})$)", fontsize=10)
    ax.set_title("One post, one SHAP value per word", fontsize=13.5)
    pad = 0.05 * max(abs(vals).max(), 1e-9)
    for k, v in enumerate(vals):
        ax.text(v + (pad if v > 0 else -pad), k, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10.5, color="0.25")
    # zero must be on the axis or the bar lengths cannot be read as magnitudes
    ax.set_xlim(min(0.0, vals.min()) - 2 * pad, max(0.0, vals.max()) + 6 * pad)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # the identity goes UNDER the axes: anywhere inside them collides with a bar
    fig.text(0.5, -0.135,
             f"base value {base:.2f}   $+$   $\\sum_j \\phi_j$ = {total:+.2f}   $=$   "
             f"{base + total:.2f}   $=$   $\\hat f(x)$   $\\rightarrow$   {truth}",
             ha="center", fontsize=12.5, color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5fa", ec="0.7", lw=0.9))
    fig.savefig(FIG_DIR / "text_shap_tokens.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote text_shap_tokens.pdf")


# ------------------------------------------------- figure 2: LIME on the header-leaking model
def fig_lime_newsgroups(pipe_dirty, tr_dirty, te_dirty, acc_dirty, acc_clean):
    """LIME word weights on the header-keeping model, next to the accuracy it buys."""
    proba = pipe_dirty.predict_proba(te_dirty.data)[:, 1]
    n_tok = np.array([len(re.findall(r"\w+", d)) for d in te_dirty.data])
    ok = pipe_dirty.predict(te_dirty.data) == te_dirty.target
    cand = np.where((n_tok >= 80) & (n_tok <= 400) & ok & (np.abs(proba - 0.5) > 0.35))[0]
    if len(cand) == 0:
        raise RuntimeError("no suitable test post found for the LIME explanation")

    header_words = {"nntp", "posting", "host", "re", "subject", "lines", "organization",
                    "distribution", "reply", "keywords", "edu", "writes", "article", "in"}
    explainer = LimeTextExplainer(class_names=SHORT, random_state=SEED)

    # prefer a post whose explanation is actually dominated by header artifacts
    best = None
    for i in cand[:25]:
        exp = explainer.explain_instance(te_dirty.data[i], pipe_dirty.predict_proba,
                                         num_features=10, num_samples=2000)
        pairs = exp.as_list()
        n_art = sum(w.lower() in header_words for w, _ in pairs)
        log.info(f"  LIME probe idx={i}: {n_art}/10 header artifacts in top-10")
        if best is None or n_art > best[0]:
            best = (n_art, int(i), pairs)
        if n_art >= 4:
            break
    n_art, i, pairs = best
    log.info(f"LIME doc idx={i} tokens={n_tok[i]} true={SHORT[te_dirty.target[i]]} "
             f"P(christian)={proba[i]:.3f}; {n_art}/10 top words are header artifacts")
    # class skew of each top word: an artifact sits almost entirely in one newsgroup
    for w, v in pairs:
        share, skew = token_skew(tr_dirty, w)
        log.info(f"    {w:>18s} {v:+.4f}  in {share:6.1%} of train posts, {skew:5.1%} atheism"
                 f"{'   <-- header field' if w.lower() in header_words else ''}")

    pairs = sorted(pairs, key=lambda t: t[1])
    words = [w for w, _ in pairs]
    vals = np.array([v for _, v in pairs])
    is_art = np.array([w.lower() in header_words for w in words])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [2.15, 1]})

    ax = axes[0]
    bars = ax.barh(range(len(vals)), vals, height=0.72,
                   color=[BLUE if v > 0 else RED for v in vals])
    # matplotlib takes scalar alpha/edgecolor per bar call, so style the artifacts per patch
    for patch, artifact in zip(bars, is_art):
        patch.set_alpha(0.95 if artifact else 0.32)
        if artifact:
            patch.set_edgecolor("0.15")
            patch.set_linewidth(1.1)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([f"{w} *" if a else w for w, a in zip(words, is_art)], fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    for tick, a in zip(ax.get_yticklabels(), is_art):
        tick.set_fontweight("bold" if a else "normal")
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("LIME weight   (toward christian $\\rightarrow$)", fontsize=10)
    ax.set_title("Why this post? Header fields (*), hostnames, usernames", fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar([0, 1], [acc_dirty, acc_clean], width=0.6,
                  color=[RED, BLUE], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=12, padding=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["headers\nkept", "headers\nremoved"], fontsize=10)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("test accuracy", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(f"{acc_dirty - acc_clean:+.3f} of it was the headers", fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "text_lime_newsgroups.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote text_lime_newsgroups.pdf")


def main():
    setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    pipe_dirty, acc_dirty, tr_dirty, te_dirty, vocab_dirty = fit(remove=())
    pipe_clean, acc_clean, tr_clean, te_clean, vocab_clean = fit(
        remove=("headers", "footers", "quotes"))
    posting_stat(tr_dirty)

    fig_shap_tokens(pipe_clean, tr_clean, te_clean)
    fig_lime_newsgroups(pipe_dirty, tr_dirty, te_dirty, acc_dirty, acc_clean)

    log.info(f"SUMMARY  acc headers-kept={acc_dirty:.4f}  headers-removed={acc_clean:.4f}  "
             f"drop={acc_dirty - acc_clean:+.4f}  vocab={vocab_dirty}/{vocab_clean}")


if __name__ == "__main__":
    main()
