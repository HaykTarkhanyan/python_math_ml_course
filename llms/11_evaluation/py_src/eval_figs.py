"""Figures for LLM-11 "Evaluation and hallucination" (llms/11_evaluation/LLM11_evaluation.tex).

Real numbers only from what is local or published:
  --gpt2   the cached GPT-2 (HF_HUB_OFFLINE=1, no download) on LLM-4's two prompts, "Lionel Messi
           plays the sport of" and "Henrikh Mkhitaryan plays the sport of": the exact next-token
           distribution (top 5, entropy) and 20 samples of 3 tokens at temperature 1 (seed 509)
           -> results/eval_gpt2.json (~30 s)
  (default) the figures and the by-hand checks (~5 s)
Published: the Armenian corpus contamination rates (Avetisyan et al. 2026, arXiv 2609.03350).
Toys (labelled): the word-overlap scores, the guess-vs-abstain expected scores.

Generates into llms/11_evaluation/fig/:
  eval_samples.pdf        -- 20 samples at T = 1: the first word, Messi vs Mkhitaryan (real)
  eval_guess.pdf          -- expected score of guessing vs abstaining, 0/1 grading vs a penalty (toy)
  eval_contamination.pdf  -- share of each Armenian corpus's documents overlapping evaluation sets

Run:
  ./ma/Scripts/python.exe llms/11_evaluation/py_src/eval_figs.py --gpt2   # once (~30 s)
  ./ma/Scripts/python.exe llms/11_evaluation/py_src/eval_figs.py          # figures (~5 s)
"""
import json
import logging
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("error", message=".*missing from.*font.*")

SEED = 509
HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "eval_gpt2.json"

PROMPTS = {"Messi": "Lionel Messi plays the sport of", "Mkhitaryan": "Henrikh Mkhitaryan plays the sport of"}
N_SAMPLES = 20
REFERENCE = "the cat sat on the mat"
CANDIDATES = {"paraphrase": "a feline rested upon the rug", "close": "the cat is on the mat",
              "nonsense": "the mat sat on the cat"}
CONTAMINATION = {"FineWeb-2 (hy)": 17.4, "HPLT v2 (hy)": 10.9, "CulturaX (hy)": 7.9, "ArmWeb": 3.3}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("eval_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "eval_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden |= {id(tick.label1), id(tick.label2)}
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            raise ValueError(f"{name}: text {t.get_text()!r} leaves the figure")


def save(fig, name, log):
    assert_texts_inside(fig, name)
    out = FIG / name
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, metadata={"CreationDate": None})   # same figure -> same bytes
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


def check(name, got, want):
    if got != want:
        raise ValueError(f"{name}: got {got!r}, the slide prints {want!r}")


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "sampling": f"{N_SAMPLES} samples, 3 new tokens, temperature 1, seed {SEED}"}
    for k, (name, prompt) in enumerate(PROMPTS.items()):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            p = torch.softmax(model(ids).logits[0, -1].double(), dim=-1)
        ent = float(-(p * torch.log(p + 1e-30)).sum())
        top = torch.topk(p, 5)
        torch.manual_seed(SEED + k)
        with torch.no_grad():
            out = model.generate(ids.repeat(N_SAMPLES, 1), do_sample=True, temperature=1.0, top_k=0,
                                 max_new_tokens=3, pad_token_id=tok.eos_token_id)
        texts = [tok.decode(o[ids.shape[1]:]) for o in out]
        first = [t.strip().split()[0].strip(".,;:!?\"'").lower() if t.strip() else "" for t in texts]
        res[name] = {"prompt": prompt, "entropy_nats": ent,
                     "top5": [[tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)],
                     "samples": texts, "first_words": first}
        log.info(f"{name}: entropy {ent:.2f} nats; top5 {res[name]['top5']}; first words {Counter(first)}")
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {GPT2_JSON.name} in {time.time() - t0:.0f} s")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- by hand
def ngram_precision(cand, ref, n):
    c, r = cand.split(), ref.split()
    cg = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
    rg = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    hit = sum(min(v, rg[g]) for g, v in cg.items())
    return hit, sum(cg.values())


def overlap_table():
    out = {k: {"uni": ngram_precision(c, REFERENCE, 1), "bi": ngram_precision(c, REFERENCE, 2)}
           for k, c in CANDIDATES.items()}
    check("word overlap (hits, total)", {k: (v["uni"], v["bi"]) for k, v in out.items()},
          {"paraphrase": ((1, 6), (0, 5)), "close": ((5, 6), (3, 5)), "nonsense": ((6, 6), (4, 5))})
    return out


def guess_table():
    rows = {}
    for p in (0.2, 0.5, 0.8):
        rows[p] = {"binary_guess": p, "penalty3_guess": round(p * 1 - 3 * (1 - p), 2), "abstain": 0}
    check("expected scores, penalty 3", [rows[p]["penalty3_guess"] for p in rows], [-2.2, -1.0, 0.2])
    check("threshold t for penalty t/(1-t) = 3", 3 / (1 + 3), 0.75)
    return rows


# ----------------------------------------------------------------------------- figures
MEANINGS = {"Messi": [("football", ("football", "soccer", "footballing"))],
            "Mkhitaryan": [("hockey", ("hockey", "ice")), ("football", ("football", "soccer"))]}


def fig_samples(G, log):
    """20 samples per prompt, grouped by meaning (the step semantic entropy takes)."""
    fm = Counter(G["Messi"]["first_words"])
    fk = Counter(G["Mkhitaryan"]["first_words"])
    check("Messi: distinct first words", len(fm), 12)
    check("Messi: meaning football (football, soccer, footballing)", fm["football"] + fm["soccer"] + fm["footballing"], 11)
    check("Mkhitaryan: distinct first words", len(fk), 16)
    check("Mkhitaryan: meaning hockey (hockey, ice hockey)", fk["hockey"] + fk["ice"], 5)
    check("Mkhitaryan: meaning football (football, soccer)", fk["football"] + fk["soccer"], 3)
    check("both 'ice' samples say ice hockey",
          [t.strip().startswith("ice hockey") for t, w in zip(G["Mkhitaryan"]["samples"], G["Mkhitaryan"]["first_words"]) if w == "ice"],
          [True, True])
    check("next-token entropy (nats)", [round(G["Messi"]["entropy_nats"], 2), round(G["Mkhitaryan"]["entropy_nats"], 2)],
          [4.46, 6.57])
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.75), sharex=True)
    out = {}
    for ax, name, color in ((axes[0], "Messi", ARMBLUE), (axes[1], "Mkhitaryan", ARMRED)):
        cnt = Counter(G[name]["first_words"])
        labels, vals, used = [], [], set()
        for meaning, words in MEANINGS[name]:
            parts = [w if w != 'ice' else 'ice hockey' for w in words if cnt[w]]
            labels.append(meaning + "\n(" + ", ".join(parts) + ")")
            vals.append(sum(cnt[w] for w in words))
            used |= set(words)
        rest = {w: c for w, c in cnt.items() if w not in used}
        labels.append(f"{len(rest)} other\nanswers")
        vals.append(sum(rest.values()))
        y = np.arange(len(labels))[::-1]
        bars = ax.barh(y, vals, color=[color] * (len(labels) - 1) + [LIGHT], height=0.6)
        ax.bar_label(bars, fontsize=FS_TINY, padding=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FS_TINY)
        ax.set_title(G[name]["prompt"].split(" plays")[0], fontsize=FS_SMALL)
        ax.set_xlim(0, N_SAMPLES * 0.8)
        ax.set_xlabel(f"of {N_SAMPLES} samples")
        ax.tick_params(axis="y", length=0)
        clean(ax)
        out[name] = dict(zip(labels, vals))
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "eval_samples.pdf", log)
    return out


def fig_guess(log):
    p = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ax.plot(p, p, color=ARMBLUE, lw=1.4, label="guess, graded 0/1")
    ax.plot(p, p - 3 * (1 - p), color=ARMRED, lw=1.4, label="guess, a wrong answer costs 3")
    ax.axhline(0, color="#2CA02C", lw=1.8, label="say \"I don't know\"")
    ax.axvline(0.75, color=GREY, ls=":", lw=0.9)
    ax.text(0.75, 1.13, "t = 0.75", fontsize=FS_TINY, color=GREY, ha="center", va="bottom")
    ax.set_xlabel("chance the guess is right, p")
    ax.set_ylabel("expected score")
    ax.set_ylim(-3.1, 1.35)
    ax.legend(frameon=False, fontsize=FS_TINY, loc="lower right")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "eval_guess.pdf", log)


def fig_contamination(log):
    names = list(CONTAMINATION)
    vals = [CONTAMINATION[n] for n in names]
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=[ARMRED, ARMORANGE, ARMORANGE, GREEN], height=0.6)
    ax.bar_label(bars, labels=[f"{v}%" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 21)
    ax.set_xlabel("% of documents sharing a 13-gram\nwith ten Armenian test sets")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "eval_contamination.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    G = load_gpt2()
    R = {"date": time.strftime("%Y-%m-%d"), "overlap": overlap_table(), "guess": guess_table()}
    R["samples"] = fig_samples(G, log)
    R["entropy"] = {k: G[k]["entropy_nats"] for k in PROMPTS}
    fig_guess(log)
    fig_contamination(log)
    (RESULTS / "eval_figs.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    log.info(f"{R}")
    log.info(f"done: 3 figures in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
