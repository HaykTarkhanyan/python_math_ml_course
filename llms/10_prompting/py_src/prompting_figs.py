"""Figures for LLM-10 "Prompting: the prompt is the context" (llms/10_prompting/LLM10_prompting.tex).

Real measurement (the queue's recommended option, "measure few-shot fragility on the cached GPT-2"):
  --gpt2   the cached GPT-2 small (HF_HUB_OFFLINE=1, no download) classifies 48 short reviews
           (written for this deck, 24 positive, 24 negative) with a 4-example few-shot prompt, once
           for each of the 24 orderings of the same 4 examples, plus zero-shot. The label is the one
           of " positive" / " negative" with the higher logit after "Sentiment:".
           24 x 48 + 48 = 1,200 forward passes of ~80 tokens -> results/prompting_gpt2.json
           (~1-3 min on 4 CPU threads; progress logged)
  (default) the figures (~5 s)
Toy: the chain-of-thought step count (GPT-2's 12 layers per generated token).

Generates into llms/10_prompting/fig/:
  prompt_orderings.pdf   -- accuracy for each of the 24 orderings, sorted; zero-shot and 50% marked
  prompt_recency.pdf     -- share predicted positive, grouped by the label of the last example
  prompt_cot_steps.pdf   -- serial layer passes: answer at once vs after 40 tokens of reasoning (toy)

Run:
  ./ma/Scripts/python.exe llms/10_prompting/py_src/prompting_figs.py --gpt2   # ~1-3 min, background
  ./ma/Scripts/python.exe llms/10_prompting/py_src/prompting_figs.py          # figures (~5 s)
"""
import itertools
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "prompting_gpt2.json"

DEMOS = [("A wonderful film with brilliant acting.", "positive"),
         ("The plot was dull and far too long.", "negative"),
         ("I loved every minute of it.", "positive"),
         ("A complete waste of time.", "negative")]
TEST_POS = ["An absolute joy from start to finish.", "The best movie I have seen this year.",
            "Funny, warm and beautifully shot.", "I would happily watch it again.",
            "A clever script and a great cast.", "It made me laugh out loud.",
            "Touching, honest and well made.", "The music alone is worth the ticket.",
            "A delightful surprise.", "Gripping until the very last scene.",
            "Smart, fresh and full of heart.", "The actors are fantastic.",
            "A moving story told with great care.", "Pure fun for the whole family.",
            "One of the finest films of the decade.", "I left the cinema smiling.",
            "Excellent pacing and a strong ending.", "A real treat for fans of the genre.",
            "Charming characters and sharp dialogue.", "Worth every minute.",
            "The director did a superb job.", "A beautiful and inspiring film.",
            "Great fun, highly recommended.", "It is even better the second time."]
TEST_NEG = ["A boring mess with no story.", "The worst film I have seen this year.",
            "Flat jokes and wooden acting.", "I wanted to leave after twenty minutes.",
            "A lazy script and a wasted cast.", "Not a single scene made me laugh.",
            "Clumsy, dull and badly edited.", "The music was loud and annoying.",
            "A disappointing letdown.", "I nearly fell asleep halfway through.",
            "Predictable, tired and empty.", "The actors seemed bored.",
            "A confusing story told without care.", "Painful to sit through.",
            "One of the weakest films of the decade.", "I left the cinema angry.",
            "Terrible pacing and a silly ending.", "Even fans of the genre will hate it.",
            "Annoying characters and awful dialogue.", "Not worth your money.",
            "The director has no idea what he is doing.", "An ugly and depressing film.",
            "Avoid it at all costs.", "It is even worse the second time."]
LABELS = [" positive", " negative"]
CORRECT_ORDER = (0, 1, 2, 3)

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
    lg = logging.getLogger("prompting_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "prompting_figs.log", encoding="utf-8")):
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


def prompt_for(order, review):
    shots = "".join(f"Review: {DEMOS[i][0]}\nSentiment: {DEMOS[i][1]}\n\n" for i in order)
    return f"{shots}Review: {review}\nSentiment:"


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)                             # one light process; leave cores free
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    label_ids = [tok.encode(l) for l in LABELS]
    if any(len(i) != 1 for i in label_ids):
        raise ValueError(f"labels are not single tokens: {label_ids}")
    label_ids = [i[0] for i in label_ids]
    tests = [(t, "positive") for t in TEST_POS] + [(t, "negative") for t in TEST_NEG]

    def run(order):
        preds = []
        for review, _ in tests:
            ids = tok(prompt_for(order, review), return_tensors="pt").input_ids
            with torch.no_grad():
                z = model(ids).logits[0, -1, label_ids]
            preds.append("positive" if float(z[0]) > float(z[1]) else "negative")
        acc = sum(p == y for p, (_, y) in zip(preds, tests)) / len(tests)
        return acc, preds.count("positive") / len(preds)

    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "n_test": len(tests), "demos": DEMOS, "orderings": []}
    acc0, pos0 = run(())
    res["zero_shot"] = {"accuracy": acc0, "share_positive": pos0}
    log.info(f"zero-shot: accuracy {acc0:.3f}, predicted positive {pos0:.3f} ({time.time() - t0:.0f} s)")
    perms = list(itertools.permutations(range(4)))
    for k, order in enumerate(perms, 1):
        acc, pos = run(order)
        res["orderings"].append({"order": list(order), "labels": [DEMOS[i][1] for i in order],
                                 "accuracy": acc, "share_positive": pos})
        el = time.time() - t0
        log.info(f"{k}/{len(perms)} order {order}: accuracy {acc:.3f}, positive {pos:.3f}; "
                 f"{el:.0f} s elapsed, ~{el / (k + 1) * (len(perms) - k):.0f} s left")
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    accs = [o["accuracy"] for o in res["orderings"]]
    log.info(f"orderings: min {min(accs):.3f}, max {max(accs):.3f}, mean {np.mean(accs):.3f}; "
             f"wrote {GPT2_JSON.name} in {time.time() - t0:.0f} s")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- figures
def fig_orderings(G, log):
    accs = sorted(100 * o["accuracy"] for o in G["orderings"])
    fig, ax = plt.subplots(figsize=(4.4, 1.6))
    bars = ax.bar(range(len(accs)), accs, color=ARMBLUE, width=0.7)
    ax.bar_label(bars, labels=[f"{a:.0f}" for a in accs], fontsize=FS_TINY, padding=1)
    ax.axhline(50, color=GREY, ls=":", lw=0.9)
    ax.axhline(100 * G["zero_shot"]["accuracy"], color=ARMORANGE, ls="--", lw=0.9)
    ax.text(len(accs) - 0.2, 100 * G["zero_shot"]["accuracy"], "zero-shot", ha="left", va="center",
            fontsize=FS_TINY, color=ARMORANGE)
    ax.text(len(accs) - 0.2, 50, "coin flip", ha="left", va="center", fontsize=FS_TINY, color=GREY)
    ax.set_xlim(-0.7, len(accs) + 2.6)
    ax.set_ylim(0, 112)
    ax.set_xticks([])
    ax.set_xlabel("the 24 orderings of the same 4 examples, sorted by accuracy")
    ax.set_ylabel("accuracy (%)")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "prompt_orderings.pdf", log)
    printed = [round(min(accs)), round(max(accs)), round(float(np.median(accs))),
               round(100 * G["zero_shot"]["accuracy"]), sum(a == 50 for a in accs)]
    check("orderings: min, max, median, zero-shot, how many at 50", printed, [50, 85, 70, 69, 6])
    check("orderings tied with zero-shot", sum(a == 100 * G["zero_shot"]["accuracy"] for a in accs), 2)
    check("orderings better / worse than zero-shot", [sum(a > 100 * G["zero_shot"]["accuracy"] for a in accs), sum(a < 100 * G["zero_shot"]["accuracy"] for a in accs)], [12, 10])
    return {"min": min(accs), "max": max(accs), "median": float(np.median(accs))}


def fig_recency(G, log):
    groups = {"positive": [], "negative": []}
    for o in G["orderings"]:
        groups[o["labels"][-1]].append(100 * o["share_positive"])
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    fig, ax = plt.subplots(figsize=(3.2, 1.3))
    names = ["zero-shot\n(no examples)", "last example\npositive", "last example\nnegative"]
    vals = [100 * G["zero_shot"]["share_positive"], means["positive"], means["negative"]]
    bars = ax.barh([2, 1, 0], vals, color=[GREY, GREEN, ARMRED], height=0.6)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], fontsize=FS_TINY, padding=2)
    ax.axvline(50, color=GREY, ls=":", lw=0.9)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(names)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of the 48 reviews called positive\n(mean of 12 orderings each)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "prompt_recency.pdf", log)
    all_neg = [o for o in G["orderings"] if o["share_positive"] == 0]
    check("orderings calling every review negative, and how many end with a positive example",
          (len(all_neg), sum(o["labels"][-1] == "positive" for o in all_neg)), (6, 6))
    check("most positive calls in any ordering (%)", round(100 * max(o["share_positive"] for o in G["orderings"])), 38)
    best = max(G["orderings"], key=lambda o: o["accuracy"])
    check("the best ordering's last example", best["labels"][-1], "positive")
    check("share called positive: last example positive / negative; zero-shot",
          [round(means["positive"]), round(means["negative"]), round(100 * G["zero_shot"]["share_positive"])],
          [9, 25, 81])
    return means


def fig_cot_steps(log):
    n_layers, n_reason = 12, 40
    direct, cot = n_layers * 1, n_layers * (n_reason + 1)
    check("serial layer passes", (direct, cot), (12, 492))
    fig, ax = plt.subplots(figsize=(3.2, 1.2))
    bars = ax.barh([1, 0], [direct, cot], color=[GREY, ARMBLUE], height=0.55)
    ax.bar_label(bars, labels=[f"{direct}", f"{cot}"], fontsize=FS_TINY, padding=2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["answer at once\n(1 token)", "40 tokens of reasoning,\nthen the answer"])
    ax.set_xlim(0, 600)
    ax.set_xlabel("layers passed through in a row\n(GPT-2: 12 per token)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "prompt_cot_steps.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    G = load_gpt2()
    R = {"date": time.strftime("%Y-%m-%d"), "zero_shot": G["zero_shot"]}
    R["orderings"] = fig_orderings(G, log)
    R["recency"] = fig_recency(G, log)
    fig_cot_steps(log)
    (RESULTS / "prompting_figs.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    log.info(f"{R}")
    log.info(f"done: 3 figures in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
