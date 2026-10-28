"""Figures for LLM-7 "Pretraining: from the internet to a base model" (llms/07_pretraining/LLM7_pretraining.tex).

Outline: llms/07_pretraining/LLM7_pretraining_OUTLINE.md. Toy numbers by default; real numbers only for
claims about real models and data:
  --gpt2   one run of the cached GPT-2 small (HF_HUB_OFFLINE=1, no download) -> results/pretraining_gpt2.json
           (~1 min): greedy continuations of five much-quoted openings (memorisation), a bare question
           vs the same question as a Q/A document (base model), and GPT-2's loss on one short window
  (default) the figures, from that JSON plus published numbers (sources below) (~10 s)

Published numbers (checked 2026-10-06; the outline lists the sources): FineWeb's funnel 36T -> 20T -> 15T
tokens; Common Crawl CC-MAIN-2026-39 language shares; Llama 3's data mix.

Generates into llms/07_pretraining/fig/:
  pre_recite.pdf        -- words recited verbatim, five openings (real, GPT-2)
  pre_objectives.pdf    -- five ways to hide text, one row each (bigram, word2vec, GPT, BERT, T5)
  pre_w2v_pairs.pdf     -- skip-gram's training pairs for one word, window 2 (by hand)
  pre_packing.pdf       -- three toy documents -> one stream with <eot> -> windows of 8
  pre_window_loss.pdf   -- one window, seven guesses: GPT-2's p(right token) and -ln p (real)
  pre_bpc.pdf           -- per-token perplexity vs bits per character, one text, two tokenizers (toy)
  pre_funnel.pdf        -- FineWeb's funnel (published)
  pre_c4_page_q.pdf     -- the toy web page alone (the predict-first frame)
  pre_c4_page.pdf       -- four C4 rules applied to it (by hand, rules coded)
  pre_languages.pdf     -- Common Crawl page shares: English, Russian, Armenian (published)
  pre_mixture.pdf       -- Llama 3's data mix (published)
  pre_base_model.pdf    -- a bare question vs a Q/A document, continued by GPT-2 (real)

Run:
  ./ma/Scripts/python.exe llms/07_pretraining/py_src/pretraining_figs.py --gpt2   # once (~1 min)
  ./ma/Scripts/python.exe llms/07_pretraining/py_src/pretraining_figs.py          # figures (~10 s)
"""
import json
import logging
import math
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("error", message=".*missing from.*font.*")

SEED = 509
HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "pretraining_gpt2.json"

# Five much-quoted openings: (prompt, the real continuation). Whitespace is collapsed before words
# are compared, so a newline where the original has a space is not counted as a mistake.
RECITE = {
    "MIT licence": ("Permission is hereby granted, free of charge, to any person obtaining a copy",
                    " of this software and associated documentation files (the \"Software\"), to deal in the"
                    " Software without restriction, including without limitation the rights to use, copy,"
                    " modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,"
                    " and to permit persons to whom the Software is furnished to do so, subject to the"
                    " following conditions:"),
    "Apache licence": ("Licensed under the Apache License, Version 2.0 (the \"License\");",
                       " you may not use this file except in compliance with the License. You may obtain a"
                       " copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by"
                       " applicable law or agreed to in writing, software distributed under the License is"
                       " distributed on an \"AS IS\" BASIS,"),
    "Gettysburg Address": ("Four score and seven years ago our fathers brought forth",
                           " on this continent, a new nation, conceived in Liberty, and dedicated to the"
                           " proposition that all men are created equal. Now we are engaged in a great civil"
                           " war, testing whether that nation, or any nation so conceived and so dedicated,"
                           " can long endure."),
    "Declaration of Independence": ("When in the Course of human events, it becomes necessary for one people",
                                    " to dissolve the political bands which have connected them with another,"
                                    " and to assume among the powers of the earth, the separate and equal"
                                    " station to which the Laws of Nature and of Nature's God entitle them"),
    "Lorem ipsum": ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do",
                    " eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,"
                    " quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."),
}
N_NEW = 48
BARE_Q = "What is the capital of Armenia?"
QA_DOC = "Q: What is the capital of France?\nA: Paris\n\nQ: What is the capital of Armenia?\nA:"
WINDOW_TEXT = "The cat sat on the mat."           # preceded by <|endoftext|>, as a document start

# Published numbers (outline: sources).
FUNNEL = [("96 Common Crawl snapshots", None), ("after URL, text, language and quality filters", 36),
          ("after near-duplicate removal (MinHash)", 20), ("after C4 and custom filters: FineWeb", 15)]
CC_SHARE = {"English": 41.8620, "Russian": 6.6954, "Armenian": 0.0315}       # CC-MAIN-2026-39, % of pages
LLAMA3_MIX = {"general knowledge": 50, "math and reasoning": 25, "code": 17, "multilingual": 8}
# Toy, for the bits-per-character frame: one Armenian sentence of 45 characters, two tokenizers.
# 45 characters = 38 Armenian letters (2 bytes each in UTF-8) + 7 spaces = 83 bytes; byte-level pieces
# merge a few, so 80 tokens. Tokens, total bits:
BPC_TOY = {"byte-level (GPT-2 style)": (80, 70.0), "Armenian-aware": (12, 60.0)}
BPC_CHARS = 45

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"
TINT_RED, TINT_BLUE, TINT_ORANGE, TINT_GREEN = "#F9D2D5", "#CFD8EE", "#FCE9B8", "#D5EFDB"
MONO = ["DejaVu Sans Mono"]

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("pretraining_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "pretraining_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


# ----------------------------------------------------------------------------- helpers
def canvas(h, w=W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=0.6, r=0.04, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color=GREY, lw=0.8, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    """Every drawn text must sit inside the figure (tick labels outside the axis range are skipped)."""
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


def pct(p):
    """A probability as a percentage with two significant digits (0.0001 must not print as 0.000)."""
    v = 100 * p
    return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%" if v >= 1 else f"{v:.2f}%" if v >= 0.1 else f"{v:.3f}%"


def words_matched(gen, truth):
    """Words of `gen` that match `truth` before the first difference (whitespace collapsed)."""
    g, t = gen.split(), truth.split()
    n = 0
    while n < min(len(g), len(t)) and g[n] == t[n]:
        n += 1
    return n, len(g)


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    def greedy(prompt, n):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:])

    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "decoding": f"greedy, {N_NEW} new tokens", "recite": {}}
    for k, (name, (prompt, truth)) in enumerate(RECITE.items(), 1):
        gen = greedy(prompt, N_NEW)
        m, n_gen = words_matched(gen, truth)
        res["recite"][name] = {"prompt": prompt, "generated": gen, "words_matched": m, "words_generated": n_gen}
        log.info(f"{k}/{len(RECITE)} {name}: {m} of {n_gen} words verbatim; {time.time() - t0:.0f}s")
    res["base_model"] = {"bare": {"prompt": BARE_Q, "generated": greedy(BARE_Q, 30)},
                         "qa": {"prompt": QA_DOC, "generated": greedy(QA_DOC, 12)}}
    ids = [tok.eos_token_id] + tok.encode(WINDOW_TEXT)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    logp = torch.log_softmax(logits.double(), dim=-1)
    rows = []
    for i in range(1, len(ids)):
        lp = float(logp[i - 1, ids[i]])
        rows.append({"context": tok.decode(ids[1:i]) or "<eot>", "target": tok.decode([ids[i]]),
                     "p": math.exp(lp), "nll": -lp})
    res["window"] = {"text": WINDOW_TEXT, "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"base model: bare -> {res['base_model']['bare']['generated']!r}")
    log.info(f"base model: Q/A -> {res['base_model']['qa']['generated']!r}")
    log.info(f"window: {[(r['target'], round(r['p'], 4)) for r in rows]}")
    log.info(f"wrote {GPT2_JSON.name} in {time.time() - t0:.0f} s")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- figures
def fig_recite(G, log):
    names = list(RECITE)
    m = [G["recite"][k]["words_matched"] for k in names]
    n = [G["recite"][k]["words_generated"] for k in names]
    fig, ax = plt.subplots(figsize=(3.75, 1.3))     # the slot: 0.68 of the text width
    y = np.arange(len(names))[::-1]
    ax.barh(y, n, color="#EEEEEE", height=0.62, label="words GPT-2 wrote")
    bars = ax.barh(y, m, color=[GREEN if v >= 20 else ARMRED for v in m], height=0.62,
                   label="before its first mistake")
    ax.bar_label(bars, labels=[f"{a} of {b}" for a, b in zip(m, n)], fontsize=FS_TINY, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS_SMALL)
    ax.set_xlim(0, max(n) * 1.25)
    ax.set_xlabel("words right, before the first mistake")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "pre_recite.pdf", log)
    return dict(zip(names, zip(m, n)))


def fig_objectives(log):
    words = ["the", "cat", "sat", "on", "the", "mat"]
    # per row: (name, how, {position: role}); roles: "ctx" seen, "tgt" guessed, "mask" hidden and guessed
    rows = [
        ("bigram (47)", "one token back", {2: "ctx", 3: "tgt"}),
        ("word2vec", "a word guesses its neighbours", {0: "tgt", 1: "tgt", 2: "ctx", 3: "tgt", 4: "tgt"}),
        ("GPT", "everything to the left", {0: "ctx", 1: "ctx", 2: "ctx", 3: "tgt"}),
        ("BERT", "15% hidden, both sides seen", {0: "ctx", 1: "ctx", 2: "mask", 3: "ctx", 4: "ctx", 5: "ctx"}),
        ("T5", "a span hidden, written back", {0: "ctx", 1: "ctx", 2: "mask", 3: "mask", 4: "ctx", 5: "ctx"}),
    ]
    style = {"ctx": (TINT_BLUE, ARMBLUE, INK), "tgt": (TINT_RED, ARMRED, ARMRED), "mask": (TINT_RED, ARMRED, ARMRED),
             None: ("white", "#DDDDDD", LIGHT)}
    fig, ax = canvas(1.85)
    x0, cw, rh, top = 0.95, 0.42, 0.27, 1.6
    for i, (name, how, roles) in enumerate(rows):
        y = top - (i + 1) * rh
        ax.text(0.05, y + rh / 2, name, va="center", fontsize=FS_SMALL, fontweight="bold", color=INK)
        for j, w in enumerate(words):
            role = roles.get(j)
            fc, ec, tc = style[role]
            if name == "T5" and role == "mask":
                if j == 2:                       # the span "sat on" becomes one sentinel
                    rbox(ax, x0 + j * cw + 0.02, y + 0.03, 2 * cw - 0.04, rh - 0.06, fc, ec, lw=0.6)
                    ax.text(x0 + (j + 1) * cw, y + rh / 2, "<X>", ha="center", va="center",
                            fontsize=FS_SMALL, color=tc, family=MONO, fontweight="bold")
                continue
            rbox(ax, x0 + j * cw + 0.02, y + 0.03, cw - 0.04, rh - 0.06, fc, ec, lw=0.6)
            label = "?" if role in ("tgt", "mask") else w
            ax.text(x0 + (j + 0.5) * cw, y + rh / 2, label, ha="center", va="center", fontsize=FS_SMALL,
                    color=tc, family=MONO, fontweight="bold" if role in ("tgt", "mask") else "normal")
        ax.text(x0 + 6 * cw + 0.1, y + rh / 2, how, va="center", fontsize=FS_TINY, color=GREY)
    ax.text(x0, 0.06, "blue = seen", fontsize=FS_TINY, color=ARMBLUE)
    ax.text(x0 + 0.8, 0.06, "red ? = guessed", fontsize=FS_TINY, color=ARMRED)
    ax.text(x0 + 1.85, 0.06, "white = not used in this example", fontsize=FS_TINY, color=GREY)
    save(fig, "pre_objectives.pdf", log)


def w2v_pairs(words, centre, window):
    return [(words[centre], words[j]) for j in range(max(0, centre - window), min(len(words), centre + window + 1))
            if j != centre]


def fig_w2v_pairs(log):
    words = ["the", "cat", "sat", "on", "the", "mat"]
    pairs = w2v_pairs(words, 2, 2)
    check("skip-gram pairs for 'sat'", pairs, [("sat", "the"), ("sat", "cat"), ("sat", "on"), ("sat", "the")])
    n_all = sum(len(w2v_pairs(words, c, 2)) for c in range(len(words)))
    check("skip-gram pairs, whole sentence", n_all, 18)
    fig, ax = canvas(1.25, 4.6)
    x0, cw = 0.35, 0.65
    for j, w in enumerate(words):
        inwin = 0 <= j <= 4
        fc, ec = ((TINT_ORANGE, ARMORANGE) if j == 2 else (TINT_BLUE, ARMBLUE) if inwin else ("white", "#DDDDDD"))
        rbox(ax, x0 + j * cw + 0.03, 0.72, cw - 0.06, 0.3, fc, ec)
        ax.text(x0 + (j + 0.5) * cw, 0.87, w, ha="center", va="center", fontsize=FS, family=MONO,
                color=INK if inwin else LIGHT)
    for j in (0, 1, 3, 4):
        arrow(ax, (x0 + 2.5 * cw, 1.02), (x0 + (j + 0.5) * cw, 1.02), color=ARMORANGE, lw=0.7,
              rad=-0.45 if j < 2 else 0.45)
    ax.text(x0 + 2.5 * cw, 0.5, "pairs: (sat, the)  (sat, cat)  (sat, on)  (sat, the)", ha="center",
            fontsize=FS_SMALL, family=MONO, color=INK)
    ax.text(x0 + 2.5 * cw, 0.2, "window 2: two words each side; every word takes a turn in the middle",
            ha="center", fontsize=FS_TINY, color=GREY)
    save(fig, "pre_w2v_pairs.pdf", log)
    return n_all


def fig_packing(log):
    docs = [["Ani", "is", "old", "."], ["Cats", "sleep", "a", "lot", "."], ["2", "+", "2", "=", "4"]]
    stream = []
    for d in docs:
        stream += d + ["<eot>"]
    check("stream length", len(stream), 17)
    win = 8
    windows = [stream[i:i + win] for i in range(0, len(stream), win)]
    check("windows", [len(w) for w in windows], [8, 8, 1])
    fig, ax = canvas(1.55)
    ax.text(0.05, 1.38, "three documents", fontsize=FS_TINY, color=GREY)
    x = 0.05
    for d, c in zip(docs, (TINT_BLUE, TINT_GREEN, TINT_ORANGE)):
        w = 0.1 + 0.3 * len(d)
        rbox(ax, x, 1.07, w, 0.24, c, LIGHT)
        ax.text(x + w / 2, 1.19, " ".join(d), ha="center", va="center", fontsize=FS_SMALL)
        x += w + 0.12
    ax.text(0.05, 0.86, "one stream, <eot> between documents (LLM-1), cut into windows of 8", fontsize=FS_TINY, color=GREY)
    cw = 0.315
    doc_of = []
    for k, d in enumerate(docs):
        doc_of += [k] * (len(d) + 1)
    for i, t in enumerate(stream):
        c = (TINT_BLUE, TINT_GREEN, TINT_ORANGE)[doc_of[i]] if t != "<eot>" else "#E8E8E8"
        rbox(ax, 0.05 + i * cw + 0.01, 0.48, cw - 0.02, 0.28, c, LIGHT, lw=0.4)
        ax.text(0.05 + (i + 0.5) * cw, 0.62, t, ha="center", va="center",
                fontsize=FS_TINY - 1.0 if t == "<eot>" else FS_TINY, family=MONO, color=INK)
    for k, w in enumerate(windows):
        xa, xb = 0.05 + k * win * cw, 0.05 + (k * win + len(w)) * cw
        ax.plot([xa + 0.02, xa + 0.02, xb - 0.02, xb - 0.02], [0.4, 0.33, 0.33, 0.4], color=ARMRED, lw=0.8)
        ax.text((xa + xb) / 2, 0.16, f"window {k + 1}" if len(w) == win else "...", ha="center",
                fontsize=FS_TINY, color=ARMRED)
    save(fig, "pre_packing.pdf", log)


def fig_window_loss(G, log):
    rows = G["window"]["rows"]
    check("guesses in the window", len(rows), 7)
    fig, ax = plt.subplots(figsize=(4.4, 1.6))
    names = [r["target"].replace(" ", "␣") for r in rows]
    nll = [r["nll"] for r in rows]
    bars = ax.bar(range(len(rows)), nll, color=ARMBLUE, width=0.6)
    ax.bar_label(bars, labels=[f"p={pct(r['p'])}\n-ln p={r['nll']:.2f}" for r in rows], fontsize=FS_TINY - 0.5,
                 padding=1.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, family=MONO, fontsize=FS_SMALL)
    ax.set_ylim(0, max(nll) * 1.45)
    ax.set_ylabel("-ln p(right token)")
    ax.set_xlabel("the token to guess (each from everything before it)")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "pre_window_loss.pdf", log)
    printed = [round(v, 2) for v in nll]
    check("printed -ln p", printed, [3.28, 9.22, 6.84, 1.49, 0.87, 5.76, 2.89])
    check("printed p", [pct(r["p"]) for r in rows], ["3.8%", "0.010%", "0.11%", "22%", "42%", "0.31%", "5.6%"])
    check("printed sum", round(sum(printed), 2), 30.35)
    check("printed mean", round(30.35 / 7, 2), 4.34)
    check("printed perplexity", round(math.exp(4.34)), 77)
    check("unrounded perplexity", round(math.exp(sum(nll) / len(nll))), 76)
    check("share of the loss from ' cat'", round(100 * 9.22 / 30.35), 30)
    mean = sum(nll) / len(nll)
    return {"nll": [round(v, 2) for v in nll], "p": [round(r["p"], 3) for r in rows], "sum": sum(nll),
            "mean": mean, "ppl": math.exp(mean)}


def bpc_numbers():
    out = {}
    for name, (n_tok, bits) in BPC_TOY.items():
        out[name] = {"tokens": n_tok, "bits": bits, "bits_per_token": bits / n_tok,
                     "ppl_per_token": 2 ** (bits / n_tok), "bits_per_char": bits / BPC_CHARS}
    return out


def fig_bpc(log):
    B = bpc_numbers()
    a, b = B["byte-level (GPT-2 style)"], B["Armenian-aware"]
    check("bpc printed", [round(a["ppl_per_token"], 1), round(b["ppl_per_token"], 1), round(a["bits_per_char"], 2),
                          round(b["bits_per_char"], 2), round(b["ppl_per_token"] / a["ppl_per_token"])],
          [1.8, 32.0, 1.56, 1.33, 17])
    check("bits per token, byte-level", a["bits_per_token"], 0.875)
    check("UTF-8 bytes of the toy sentence", 38 * 2 + 7, 83)
    names = list(B)
    fig, axes = plt.subplots(1, 2, figsize=(3.53, 1.2))     # the slot: 0.64 of the text width
    colors = [ARMBLUE, GREEN]
    for ax, key, title, fmt in ((axes[0], "ppl_per_token", "perplexity per token\n(lower looks better)", "{:.1f}"),
                                (axes[1], "bits_per_char", "bits per character\n(lower is better)", "{:.2f}")):
        vals = [B[n][key] for n in names]
        bars = ax.bar([0, 1], vals, color=colors, width=0.55)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=FS_TINY, padding=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["byte-level", "Armenian-aware"], fontsize=FS_TINY)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_yticks([])
        ax.set_title(title, fontsize=FS_TINY)
        clean(ax)
        ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    save(fig, "pre_bpc.pdf", log)
    return B


def fig_funnel(log):
    fig, ax = canvas(1.45, 5.5)
    vals = [v for _, v in FUNNEL if v is not None]
    check("funnel", vals, [36, 20, 15])
    y = 1.12
    scale = 2.6 / 36
    for label, v in FUNNEL:
        if v is None:
            rbox(ax, 0.05, y, 2.6, 0.22, "#EEEEEE", LIGHT)
            ax.text(1.35, y + 0.11, "raw web pages", ha="center", va="center", fontsize=FS_TINY, color=GREY)
        else:
            w = v * scale
            rbox(ax, 0.05 + (2.6 - w) / 2, y, w, 0.22, TINT_BLUE if v > 15 else ARMBLUE, ARMBLUE)
            ax.text(1.35, y + 0.11, f"{v}T tokens", ha="center", va="center", fontsize=FS_SMALL,
                    color="white" if v == 15 else INK, fontweight="bold")
        ax.text(2.8, y + 0.11, label, va="center", fontsize=FS_TINY, color=INK)
        y -= 0.33
    save(fig, "pre_funnel.pdf", log)


# C4's rules (T5 paper, section 2.2), the four used on the toy page
def c4_line_verdict(line):
    if "javascript" in line.lower():
        return "drop: mentions Javascript"
    if not line.rstrip().endswith((".", "!", "?", '"')):
        return "drop: no end punctuation"
    if len(line.split()) < 5:
        return "drop: fewer than 5 words"
    return "keep"


PAGE = ["Home | News | Contact", "Please enable Javascript to view this page.",
        "Yerevan is one of the oldest cities in the world.", "It was founded in 782 BC as Erebuni.",
        "Read more.", "© 2026 All rights reserved"]


def fig_c4_page_question(log):
    fig, ax = canvas(1.55, 3.3)
    rbox(ax, 0.05, 0.1, 3.05, 1.4, "white", LIGHT)
    for i, line in enumerate(PAGE):
        ax.text(0.15, 1.32 - i * 0.2, line, va="center", fontsize=FS_SMALL, color=INK)
    save(fig, "pre_c4_page_q.pdf", log)


def fig_c4_page(log):
    verdicts = [c4_line_verdict(l) for l in PAGE]
    kept = [l for l, v in zip(PAGE, verdicts) if v == "keep"]
    n_sent = len(re.findall(r"[.!?](\s|$)", " ".join(kept)))
    check("C4 verdicts", verdicts, ["drop: no end punctuation", "drop: mentions Javascript", "keep", "keep",
                                    "drop: fewer than 5 words", "drop: no end punctuation"])
    check("sentences left", n_sent, 2)
    fig, ax = canvas(1.75)
    rbox(ax, 0.05, 0.3, 3.05, 1.4, "white", LIGHT)
    for i, (line, v) in enumerate(zip(PAGE, verdicts)):
        y = 1.52 - i * 0.2
        keep = v == "keep"
        t = ax.text(0.15, y, line, va="center", fontsize=FS_SMALL, color=INK if keep else GREY)
        if not keep:
            fig.canvas.draw()
            bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
            ax.plot([bb.x0, bb.x1], [y, y], color=ARMRED, lw=0.8)
        ax.text(3.22, y, v, va="center", fontsize=FS_TINY, color=GREEN if keep else ARMRED)
    ax.text(0.05, 0.1, f"{len(kept)} lines survive = {n_sent} sentences; a page needs 3, "
            "so the whole page goes", fontsize=FS_SMALL, color=ARMRED, fontweight="bold")
    save(fig, "pre_c4_page.pdf", log)


def fig_languages(log):
    names = list(CC_SHARE)
    vals = [CC_SHARE[n] for n in names]
    ratio = CC_SHARE["English"] / CC_SHARE["Armenian"]
    check("English pages per Armenian page", round(ratio, -1), 1330.0)
    fig, ax = plt.subplots(figsize=(4.0, 1.25))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=[ARMBLUE, ARMBLUE, ARMRED], height=0.6)
    ax.bar_label(bars, labels=[f"{v:.2f}%" if v > 1 else f"{v:.4f}%" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 400)
    ax.set_xticks([0.01, 0.1, 1, 10, 100])          # plain labels: 10^-2's exponent printed at 5 pt
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("share of pages in the crawl (log scale)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "pre_languages.pdf", log)
    return ratio


def fig_mixture(log):
    names = list(LLAMA3_MIX)
    vals = [LLAMA3_MIX[n] for n in names]
    check("mix sums to 100", sum(vals), 100)
    fig, ax = plt.subplots(figsize=(3.6, 1.3))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=[ARMBLUE, ARMBLUE, ARMBLUE, ARMRED], height=0.6)
    ax.bar_label(bars, labels=[f"{v}%" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 60)
    ax.set_xlabel("share of Llama 3's ~15.6T training tokens")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "pre_mixture.pdf", log)


def fig_base_model(G, log):
    bm = G["base_model"]
    fig, ax = canvas(1.85)
    for k, (key, title, color) in enumerate((("bare", "a bare question", ARMRED),
                                              ("qa", "a Q/A document with one worked example", GREEN))):
        x = 0.05 + k * 2.75
        rbox(ax, x, 0.15, 2.6, 1.65, "#FAFAFA", LIGHT)
        ax.text(x + 0.1, 1.68, title, fontsize=FS_SMALL, fontweight="bold", color=color, va="center")
        prompt, gen = bm[key]["prompt"], bm[key]["generated"]
        # one row per line of prompt + output; the prompt part in black bold, GPT-2's part in colour
        rows, cut, pos = [], len(prompt), 0
        for line in (prompt + gen).split("\n"):
            a = max(0, min(len(line), cut - pos))
            if a == 0 and len(line) > 40:      # an output-only line too long for the box
                rows += [("", part) for part in textwrap.wrap(line, 40)]
            else:
                rows.append((line[:a], line[a:]))
            pos += len(line) + 1
        if len(rows) > 10:
            raise ValueError(f"{key}: {len(rows)} lines do not fit the box")
        y = 1.48
        for p_part, g_part in rows:
            xs = x + 0.1
            if p_part:
                t = ax.text(xs, y, p_part, fontsize=FS_SMALL, family=MONO, color=INK, va="center",
                            fontweight="bold")
                fig.canvas.draw()
                xs = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1
            if g_part:
                ax.text(xs, y, g_part, fontsize=FS_SMALL, family=MONO, color=color, va="center")
            y -= 0.15
    ax.text(W / 2, 0.0, "black = the prompt; colour = what GPT-2 wrote (greedy)", ha="center", va="bottom",
            fontsize=FS_TINY, color=GREY)
    save(fig, "pre_base_model.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    G = load_gpt2()
    R = {"date": time.strftime("%Y-%m-%d")}
    R["recite"] = fig_recite(G, log)
    check("recitation", R["recite"], {"MIT licence": (34, 34), "Apache licence": (30, 30),
                                      "Gettysburg Address": (0, 42), "Declaration of Independence": (1, 42),
                                      "Lorem ipsum": (0, 18)})
    check("Gettysburg continuation", G["recite"]["Gettysburg Address"]["generated"][:33],
          " the first of the great and great")
    check("base model, Q/A", G["base_model"]["qa"]["generated"], " Armenia\n\nQ: What is the capital of Azerbaijan?")
    fig_objectives(log)
    R["w2v_pairs_sentence"] = fig_w2v_pairs(log)
    fig_packing(log)
    R["window"] = fig_window_loss(G, log)
    R["bpc"] = fig_bpc(log)
    fig_funnel(log)
    fig_c4_page_question(log)
    fig_c4_page(log)
    R["english_per_armenian"] = fig_languages(log)
    fig_mixture(log)
    fig_base_model(G, log)
    (RESULTS / "pretraining_figs.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    log.info(f"recite: {R['recite']}")
    log.info(f"window: {R['window']}")
    log.info(f"bpc: {R['bpc']}")
    log.info(f"done: 12 figures in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
