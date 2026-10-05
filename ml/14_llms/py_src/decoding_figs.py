"""Figures for LLM-3 "Decoding: from probabilities to text" (ml/14_llms/LLM3_decoding.tex).

Toy, demonstrative numbers by default (ml/SLIDE_STYLE.md, 2026-10-05). Two real inputs only:
  - the cold open: GPT-2's greedy continuation of COLD_PROMPT (the cached GPT-2, no download),
    produced once with --gpt2 and stored in results/decoding_gpt2.json;
  - the map's real top-5 after "The cat sat on the" (results/roadmap_gpt2.json, from
    llm_roadmap.py --gpt2).
Numbers the slides quote are asserted (QUOTED), so a changed input fails loudly.

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text, and
save() refuses any text outside its figure.

Two steps, so torch is not imported for every redraw:
  ./ma/Scripts/python.exe ml/14_llms/py_src/decoding_figs.py --gpt2   # greedy run -> JSON (~2 min, mostly importing torch)
  ./ma/Scripts/python.exe ml/14_llms/py_src/decoding_figs.py          # the figures (~15 s)

Generates into ml/14_llms/fig/: dec_*.pdf.
"""
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "decoding_gpt2.json"
MAP_JSON = RESULTS / "roadmap_gpt2.json"

COLD_PROMPT = "The cat sat on the"     # the map's sentence; all 6 prompts tried looped under greedy
N_NEW = 50

# Toy numbers (all demonstrative) and the values the slides quote.
TREE = {"on": (0.5, {"the": 0.3, "a": 0.3, "my": 0.2}), "in": (0.4, {"a": 0.9, "the": 0.1}),
        "by": (0.1, {"the": 0.5, "a": 0.5})}
PEAKED = [0.88, 0.04, 0.02, 0.01, 0.01] + [0.04 / 15] * 15
FLAT = list(np.linspace(0.08, 0.02, 20))
QUOTED = {"top5_sum": 29.6, "rest": 70.4, "topp_cum": [0.076, 0.141, 0.195, 0.247], "topp_keep": 4,
          "topp_renorm": [0.31, 0.26, 0.22, 0.21],
          "temp_T1": [0.67, 0.24, 0.09], "temp_T10": [0.37, 0.33, 0.30],
          "greedy_path": 0.15, "best_path": 0.36, "tail": 0.25,
          "peaked_topp": 2, "flat_topp": 16, "peaked_minp": 1, "flat_minp": 20,
          "spec_accept": 0.5,
          # student-review additions (2026-10-05): all toys
          "spec_toy": {"small": [0.6, 0.3, 0.1], "big": [0.3, 0.5, 0.2], "accepted": [0.30, 0.30, 0.10],
                       "reject": 0.30, "residual": [0.0, 0.67, 0.33], "final": [0.30, 0.50, 0.20]},
          "tail_T": {"0.5": 0.0, "1": 0.25, "2": 0.98},
          "order_toy": {"p": [0.5, 0.3, 0.15, 0.05], "T": 2, "top_p": 0.8, "temp_first_keeps": 3,
                        "cut_first_keeps": 2, "after_T": [0.38, 0.29, 0.21, 0.12]}}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY = "#222222", "#777777"
TINT_RED, TINT_BLUE, TINT_ORANGE, TINT_GREEN = "#F9D2D5", "#CFD8EE", "#FCE9B8", "#D5EFDB"

plt.rcParams.update({"font.family": SANS, "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("decoding_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "decoding_figs.log", encoding="utf-8")):
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


def arrow(ax, p0, p1, color=GREY, lw=0.8, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 zorder=z, shrinkA=0, shrinkB=0))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    """Every drawn text must sit inside the figure (draw first: axis labels are placed at draw time;
    skip tick labels outside the axis range, which exist but are never drawn)."""
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
            fig.savefig(out)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids = tok(COLD_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, pad_token_id=50256)
    new = out[0, ids.shape[1]:].tolist()
    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "prompt": COLD_PROMPT, "max_new_tokens": N_NEW, "decoding": "greedy",
           "new_ids": new, "new_text": tok.decode(new)}
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"greedy: {res['new_text']!r}")


def loop_start(text):
    """Character offset where the first repeated sentence comes back (quotes and spaces ignored),
    and that sentence; (None, None) if nothing repeats."""
    import re
    seen = set()
    for m in re.finditer(r"[^.]+\.?", text):
        key = m.group().strip(' "\n')
        if not key:
            continue
        if key in seen:
            return m.start() + (len(m.group()) - len(m.group().lstrip(' "\n'))), key
        seen.add(key)
    return None, None


# ----------------------------------------------------------------------------- figures
def fig_greedy_loop(R, log):
    g = json.loads(GPT2_JSON.read_text(encoding="utf-8"))
    assert g["prompt"] == COLD_PROMPT, "rerun with --gpt2 after changing COLD_PROMPT"
    full = (COLD_PROMPT + g["new_text"]).replace("\n", " ")
    start, rep = loop_start(full[len(COLD_PROMPT):])
    if start is None:
        raise ValueError(f"no repeated sentence in GPT-2's greedy output: {g['new_text']!r}")
    start += len(COLD_PROMPT)
    R["greedy_loop"] = {"prompt": COLD_PROMPT, "text": g["new_text"], "repeated_sentence": rep,
                        "loop_starts_at_char": start}
    log.info(f"loop: {rep!r} from char {start}")
    words = full.split(" ")
    fig, ax = canvas(0.62)
    rbox(ax, 0.02, 0.04, W - 0.04, 0.54, "white", "#BBBBBB", lw=0.6, r=0.05, z=1)
    x, y, n_prompt = 0.14, 0.42, len(COLD_PROMPT.split(" "))
    r = fig.canvas.get_renderer()
    char = 0
    for k, w in enumerate(words):
        color = ARMBLUE if k < n_prompt else (ARMRED if char >= start else INK)
        t = ax.text(x, y, w, fontsize=FS, color=color, va="center", ha="left",
                    fontweight="bold" if k < n_prompt else "normal")
        wd = t.get_window_extent(r).width / fig.dpi + 0.045
        if x + wd > W - 0.12:
            t.remove()
            x, y = 0.14, y - 0.19
            t = ax.text(x, y, w, fontsize=FS, color=color, va="center", ha="left",
                        fontweight="bold" if k < n_prompt else "normal")
        x += wd
        char += len(w) + 1
    if y < 0.15:
        raise ValueError("greedy text does not fit its two-line box")
    save(fig, "dec_greedy_loop.pdf", log)


def fig_map_top5(R, log):
    m = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    top = [(t.strip(), p) for t, p in m["top5"]]
    s = sum(p for _, p in top)
    assert round(100 * s, 1) == QUOTED["top5_sum"] and round(100 * (1 - s), 1) == QUOTED["rest"], s
    # The by-hand frame adds the probabilities as printed (3 decimals), so assert that arithmetic,
    # and separately that the unrounded values keep the same number of tokens.
    shown = [round(p, 3) for _, p in top]
    cum = [round(float(c), 3) for c in np.cumsum(shown)]
    assert cum[:4] == QUOTED["topp_cum"], cum
    real_cum = np.cumsum([p for _, p in top])
    assert sum(c < 0.2 for c in cum) + 1 == sum(c < 0.2 for c in real_cum) + 1 == QUOTED["topp_keep"]
    kept = shown[:QUOTED["topp_keep"]]
    renorm = [round(p / sum(kept), 2) for p in kept]
    assert renorm == QUOTED["topp_renorm"], renorm
    R["map_top5"] = {"top5": top, "sum": s, "cumulative": cum}
    fig = plt.figure(figsize=(3.1, 1.55))
    ax = fig.add_axes([0.40, 0.08, 0.44, 0.84])
    labels = [t for t, _ in top] + ["the other 50,252"]
    vals = [100 * p for _, p in top] + [100 * (1 - s)]
    ys = np.arange(len(vals))[::-1]
    bars = ax.barh(ys, vals, color=[ARMRED] * 5 + ["#BBBBBB"], height=0.62)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=FS_SMALL, padding=2)
    ax.set_yticks(ys, labels, fontsize=FS_SMALL)
    ax.set_xlim(0, 88)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    save(fig, "dec_map_top5.pdf", log)


def fig_ayb_bytes(R, log):
    fig, ax = canvas(0.62, w=3.0)
    ax.text(0.10, 0.31, "ա", fontsize=14, color=ARMRED, va="center")
    ax.text(0.38, 0.31, "=", fontsize=FS + 1, va="center")
    for k, (b, ok) in enumerate((("D5", True), ("A1", False))):
        rbox(ax, 0.62 + k * 0.52, 0.16, 0.44, 0.30, TINT_BLUE if ok else "white",
             ARMBLUE if ok else "#BBBBBB", lw=0.8, z=2)
        ax.text(0.84 + k * 0.52, 0.31, b, ha="center", va="center", fontsize=FS, family=MONO,
                color=INK if ok else "#AAAAAA", zorder=3)
    ax.plot([1.62, 1.62], [0.06, 0.56], color=ARMRED, lw=1.2, ls=(0, (3, 2)))
    ax.text(1.70, 0.43, "stop here:", fontsize=FS_SMALL, color=ARMRED, va="center")
    ax.text(1.70, 0.20, "D5 alone is not text", fontsize=FS_SMALL, color=ARMRED, va="center")
    save(fig, "dec_ayb_bytes.pdf", log)


def tree_paths():
    out = {}
    for w1, (p1, nxt) in TREE.items():
        for w2, p2 in nxt.items():
            out[(w1, w2)] = p1 * p2
    return out


def fig_tree(R, log, mode):
    paths = tree_paths()
    greedy1 = max(TREE, key=lambda w: TREE[w][0])
    greedy2 = max(TREE[greedy1][1], key=TREE[greedy1][1].get)
    best = max(paths, key=paths.get)
    assert round(paths[(greedy1, greedy2)], 2) == QUOTED["greedy_path"], paths
    assert round(paths[best], 2) == QUOTED["best_path"] and best == ("in", "a"), best
    beam1 = sorted(TREE, key=lambda w: -TREE[w][0])[:2]
    beam2 = sorted([k for k in paths if k[0] in beam1], key=lambda k: -paths[k])[:2]
    assert best in beam2, beam2
    R["tree"] = {"paths": {f"{a} {b}": round(v, 3) for (a, b), v in paths.items()},
                 "greedy": [greedy1, greedy2], "best": list(best), "beam_step1": beam1,
                 "beam_step2": [list(k) for k in beam2]}
    fig, ax = canvas(1.95)
    root = (0.45, 0.88)
    rbox(ax, root[0] - 0.22, root[1] - 0.12, 0.44, 0.24, TINT_BLUE, ARMBLUE, z=3)
    ax.text(*root, "sat", ha="center", va="center", fontsize=FS, zorder=4)
    x1, x2, x3 = 1.45, 2.80, 4.05
    y1 = {"on": 1.40, "in": 0.80, "by": 0.25}
    leaves = {"on": [("the", 1.62), ("a", 1.38), ("my", 1.14)], "in": [("a", 0.86), ("the", 0.62)],
              "by": [("the", 0.34), ("a", 0.10)]}
    for w1, (p1, nxt) in TREE.items():
        on_greedy = w1 == greedy1
        kept1 = w1 in beam1
        if mode == "greedy":
            col1 = ARMRED if on_greedy else "#BBBBBB"
        else:
            col1 = GREEN if kept1 else "#BBBBBB"
        arrow(ax, (root[0] + 0.24, root[1]), (x1 - 0.30, y1[w1]), color=col1, lw=1.0)
        rbox(ax, x1 - 0.28, y1[w1] - 0.11, 0.56, 0.22, "white", col1, lw=1.0, z=3)
        ax.text(x1, y1[w1], f"{w1}  {p1:.1f}", ha="center", va="center", fontsize=FS, zorder=4,
                color=INK if col1 != "#BBBBBB" else "#999999")
        if w1 == "by" and mode == "beam":
            ax.text(x1 + 0.34, y1[w1], "dropped", fontsize=FS_TINY, color="#999999", va="center")
            continue
        for w2, yy in leaves[w1]:
            p = paths[(w1, w2)]
            if mode == "greedy":
                col = ARMRED if (w1, w2) == (greedy1, greedy2) else (GREEN if (w1, w2) == best else "#BBBBBB")
            else:
                col = GREEN if (w1, w2) in beam2 else "#BBBBBB"
            if mode == "greedy" and w1 == "by":
                col = "#BBBBBB"
            arrow(ax, (x1 + 0.30, y1[w1]), (x2 - 0.32, yy), color=col, lw=0.9)
            rbox(ax, x2 - 0.30, yy - 0.10, 0.60, 0.20, "white", col, lw=1.0, z=3)
            ax.text(x2, yy, f"{w2}  {TREE[w1][1][w2]:.1f}", ha="center", va="center", fontsize=FS_SMALL,
                    zorder=4, color=INK if col != "#BBBBBB" else "#999999")
            ax.text(x2 + 0.36, yy, f"= {p:.2f}", fontsize=FS_SMALL, va="center",
                    color=col if col != "#BBBBBB" else "#999999", fontweight="bold" if col != "#BBBBBB" else "normal")
    if mode == "greedy":
        ax.text(x3 + 0.12, 1.45, "greedy: on, then the", fontsize=FS_SMALL, color=ARMRED, ha="left")
        ax.text(x3 + 0.12, 1.30, "0.5 x 0.3 = 0.15", fontsize=FS_SMALL, color=ARMRED, ha="left")
        ax.text(x3 + 0.12, 0.95, "best: in, then a", fontsize=FS_SMALL, color=GREEN, ha="left")
        ax.text(x3 + 0.12, 0.80, "0.4 x 0.9 = 0.36", fontsize=FS_SMALL, color=GREEN, ha="left")
        name = "dec_greedy_tree.pdf"
    else:
        ax.text(x3 + 0.12, 1.45, "keep the best 2", fontsize=FS_SMALL, color=GREEN, ha="left")
        ax.text(x3 + 0.12, 1.30, "at every step", fontsize=FS_SMALL, color=GREEN, ha="left")
        ax.text(x3 + 0.12, 0.95, "winner: in a = 0.36", fontsize=FS_SMALL, color=GREEN, ha="left",
                fontweight="bold")
        name = "dec_beam_tree.pdf"
    ax.text(x1, 1.86, "step 1", ha="center", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(x2, 1.86, "step 2 (path probability)", ha="center", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, name, log)


def fig_prob_trap(R, log):
    """Schematic, after Holtzman et al. (2020): demonstrative curves, not data."""
    rng = np.random.default_rng(509)
    n = 60
    human = np.clip(rng.beta(1.2, 1.6, n), 0.01, 0.98)
    beam = np.clip(0.86 + 0.06 * rng.standard_normal(n), 0.6, 0.99)
    fig = plt.figure(figsize=(W, 1.55))
    ax = fig.add_axes([0.13, 0.22, 0.85, 0.66])
    ax.plot(np.arange(n), human, color=ARMBLUE, lw=1.0, label="text a person wrote")
    ax.plot(np.arange(n), beam, color=ARMRED, lw=1.2, label="beam search output")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("position in the text", labelpad=1)
    ax.set_ylabel("probability the model\ngave the actual\nnext token", labelpad=1, fontsize=FS_TINY)
    ax.set_xticks([])
    ax.legend(fontsize=FS_TINY, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.text(0.99, 0.02, "schematic, after Holtzman et al. (2020) - illustrative values", fontsize=FS_TINY,
             color=GREY, ha="right", va="bottom")
    clean(ax)
    save(fig, "dec_prob_trap.pdf", log)


def fig_long_tail(R, log):
    head = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
    tail_n, tail_p = 50000, 0.000005
    assert round(sum(head), 2) == 0.75 and round(tail_n * tail_p, 2) == QUOTED["tail"]
    fig = plt.figure(figsize=(W, 1.45))
    ax = fig.add_axes([0.08, 0.16, 0.90, 0.70])
    ax.bar(np.arange(10), head, color=ARMBLUE, width=0.7)
    ax.bar(np.arange(10, 60), [tail_p * 400] * 50, color="#BBBBBB", width=1.0)
    ax.text(35, 0.06, f"... 50,000 more tokens at 0.000005 each: together 25%", ha="center",
            fontsize=FS_SMALL, color=ARMRED)
    ax.text(2.5, 0.205, "10 sensible tokens: 75%", ha="left", fontsize=FS_SMALL, color=ARMBLUE)
    ax.set_xticks([])
    ax.set_ylabel("probability", labelpad=1)
    ax.set_ylim(0, 0.25)
    ax.set_title("toy distribution (tail bars not to scale)", fontsize=FS_TINY, color=GREY, loc="left")
    clean(ax)
    save(fig, "dec_long_tail.pdf", log)


def temperature(p, T):
    q = np.array(p) ** (1 / T)
    return q / q.sum()


def fig_temperature(R, log):
    z = np.array([2.0, 1.0, 0.0])
    t1 = np.exp(z) / np.exp(z).sum()
    t10 = np.exp(z / 10) / np.exp(z / 10).sum()
    assert [round(v, 2) for v in t1] == QUOTED["temp_T1"], t1
    assert [round(v, 2) for v in t10] == QUOTED["temp_T10"], t10
    m = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    top = [(t.strip(), p) for t, p in m["top5"]]
    fig = plt.figure(figsize=(W, 1.55))
    out = {}
    for k, T in enumerate((0.5, 1.0, 2.0)):
        q = temperature([p for _, p in top], T)
        out[str(T)] = q.round(3).tolist()
        ax = fig.add_axes([0.06 + k * 0.33, 0.30, 0.27, 0.50])
        bars = ax.bar(range(5), q, color=ARMRED if T == 0.5 else (ARMORANGE if T == 1 else ARMBLUE), width=0.7)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in q], fontsize=FS_TINY, padding=1)
        ax.set_xticks(range(5), [t for t, _ in top], fontsize=FS_TINY, rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.set_yticks([])
        ax.set_ylim(0, 0.62)
        ax.set_title(f"T = {T}", fontsize=FS_SMALL)
        clean(ax)
        ax.spines["left"].set_visible(False)
    R["temperature"] = {"toy_T1": t1.round(4).tolist(), "toy_T10": t10.round(4).tolist(), "map_top5": out}
    fig.text(0.01, 0.97, "the map's top five, renormalised among themselves", fontsize=FS_TINY,
             color=GREY, va="top")
    save(fig, "dec_temperature.pdf", log)


def keep_counts(p):
    p = np.sort(np.array(p))[::-1]
    topp = int(np.searchsorted(np.cumsum(p), 0.9 - 1e-12)) + 1
    minp = int((p >= 0.1 * p[0]).sum())
    return topp, minp


def fig_topk_topp(R, log):
    assert abs(sum(PEAKED) - 1) < 1e-9 and abs(sum(FLAT) - 1) < 1e-9
    (pk_p, pk_m), (fl_p, fl_m) = keep_counts(PEAKED), keep_counts(FLAT)
    assert (pk_p, fl_p, pk_m, fl_m) == (QUOTED["peaked_topp"], QUOTED["flat_topp"],
                                        QUOTED["peaked_minp"], QUOTED["flat_minp"]), (pk_p, fl_p, pk_m, fl_m)
    R["topk_topp"] = {"peaked": {"top_p_0.9_keeps": pk_p, "min_p_0.1_keeps": pk_m},
                      "flat": {"top_p_0.9_keeps": fl_p, "min_p_0.1_keeps": fl_m}}
    fig = plt.figure(figsize=(W, 1.75))
    for k, (name, p, keep) in enumerate((("peaked: \"The capital of Armenia is\"", PEAKED, pk_p),
                                         ("flat: \"My favourite food is\"", FLAT, fl_p))):
        ax = fig.add_axes([0.05 + k * 0.50, 0.14, 0.43, 0.62])
        cols = [ARMBLUE if i < keep else "#CCCCCC" for i in range(20)]
        ax.bar(range(20), p, color=cols, width=0.75)
        ax.axvline(4.5, color=ARMRED, lw=1.0, ls=(0, (3, 2)))
        ax.set_ylim(0, max(p) * 1.32)
        ax.text(4.7, max(p) * 1.18, "top-k = 5 cuts here", fontsize=FS_TINY, color=ARMRED)
        ax.set_title(name, fontsize=FS_SMALL, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(19.5, max(p) * 1.05, f"top-p 0.9 keeps {keep} (blue)", fontsize=FS_TINY, color=ARMBLUE,
                ha="right")
        clean(ax)
        ax.spines["left"].set_visible(False)
    fig.text(0.01, 0.97, "toy distributions over 20 tokens", fontsize=FS_TINY, color=GREY, va="top")
    save(fig, "dec_topk_vs_topp.pdf", log)


def fig_knob_pipeline(R, log):
    steps = [("logits", "#EEEEEE"), ("penalties", TINT_ORANGE), ("temperature", TINT_RED),
             ("softmax", "#EEEEEE"), ("top-k", TINT_GREEN), ("top-p", TINT_GREEN), ("min-p", TINT_GREEN),
             ("sample", TINT_BLUE)]
    fig, ax = canvas(0.95)
    r = fig.canvas.get_renderer()
    widths = []
    for s, _ in steps:
        tt = fig.text(0, 0, s, fontsize=FS_SMALL)
        widths.append(tt.get_window_extent(r).width / fig.dpi + 0.12)
        tt.remove()
    gap = (W - 0.08 - sum(widths)) / (len(steps) - 1)
    if gap < 0.10:
        raise ValueError(f"knob pipeline too wide: gap {gap:.2f} in")
    x = 0.04
    for k, ((s, c), bw) in enumerate(zip(steps, widths)):
        rbox(ax, x, 0.42, bw, 0.30, c, "#888888", z=2)
        ax.text(x + bw / 2, 0.57, s, ha="center", va="center", fontsize=FS_SMALL, zorder=3)
        if k < len(steps) - 1:
            arrow(ax, (x + bw + 0.01, 0.57), (x + bw + gap - 0.01, 0.57), lw=0.8)
        x += bw + gap
    ax.text(0.04, 0.18, "Hugging Face's order (transformers 5.15). llama.cpp's default chain moves temperature "
            "to the end, after the cut-offs.", fontsize=FS_TINY, color=GREY)
    save(fig, "dec_knob_pipeline.pdf", log)


def fig_json_mask(R, log):
    fig, ax = canvas(1.05)
    prefix = '{"city": "Yerevan", "year": '
    rbox(ax, 0.20, 0.66, 3.15, 0.28, "#F4F4F4", "#BBBBBB", z=1)
    ax.text(0.30, 0.80, prefix, fontsize=FS, family=MONO, va="center")
    ax.text(3.45, 0.80, "next token?", fontsize=FS_SMALL, color=GREY, va="center")
    x = 0.20
    for t, ok in (("19", True), ("20", True), ("4", True), ('"', False), ("{", False), ("[", False),
                  ("null", False), (" The", False)):
        w = 0.32 + 0.07 * len(t)
        rbox(ax, x, 0.16, w, 0.28, TINT_GREEN if ok else "#F2F2F2", GREEN if ok else "#BBBBBB", lw=0.8, z=2)
        ax.text(x + w / 2, 0.30, t.replace(" ", "␣"), ha="center", va="center", fontsize=FS,
                family=MONO, color=INK if ok else "#AAAAAA", zorder=3)
        if not ok:
            ax.plot([x + 0.05, x + w - 0.05], [0.20, 0.40], color=ARMRED, lw=1.0, zorder=4)
        x += w + 0.08
    ax.text(0.20, 0.52, "allowed (keep a number going)", fontsize=FS_TINY, color=GREEN)
    ax.text(2.25, 0.52, "masked: probability set to 0 (would break the schema)", fontsize=FS_TINY,
            color=ARMRED)
    save(fig, "dec_json_mask.pdf", log)


def fig_speculative(R, log):
    p_big, p_small = 0.3, 0.6
    assert min(1, p_big / p_small) == QUOTED["spec_accept"]
    fig, ax = canvas(1.52)
    ax.text(0.04, 1.40, "greedy version: \"ok\" means the drafted token equals the big model's own top token at "
            "that position", fontsize=FS_TINY, color=GREY, va="center")
    words = ["the", "cat", "sat", "on", "moon"]
    ok = [True, True, True, True, False]
    ax.text(0.04, 1.08, "small model drafts 5:", fontsize=FS_SMALL, color=ARMBLUE, va="center")
    ax.text(0.04, 0.62, "big model checks all 5\nin one pass:", fontsize=FS_SMALL, color=INK, va="center",
            linespacing=1.15)
    for k, (w, good) in enumerate(zip(words, ok)):
        x = 1.62 + k * 0.58
        rbox(ax, x, 0.96, 0.52, 0.24, TINT_BLUE, ARMBLUE, z=2)
        ax.text(x + 0.26, 1.08, w, ha="center", va="center", fontsize=FS, zorder=3)
        arrow(ax, (x + 0.26, 0.95), (x + 0.26, 0.76), lw=0.7)
        rbox(ax, x, 0.50, 0.52, 0.24, TINT_GREEN if good else TINT_RED, GREEN if good else ARMRED, z=2)
        ax.text(x + 0.26, 0.62, "ok" if good else "no", ha="center", va="center", fontsize=FS,
                color=GREEN if good else ARMRED, fontweight="bold", zorder=3)
    ax.text(1.62 + 4 * 0.58 + 0.60, 0.62, "-> big model's\n   own token: \"the\"", fontsize=FS_SMALL,
            color=ARMRED, va="center", linespacing=1.15)
    ax.plot([1.62, 1.62 + 4 * 0.58 - 0.08], [0.36, 0.36], color=GREEN, lw=1.2)
    ax.text(1.62 + 2 * 0.58 - 0.05, 0.20, "4 tokens accepted + 1 corrected = 5 tokens for one big pass",
            ha="center", fontsize=FS_SMALL, color=GREEN)
    save(fig, "dec_speculative.pdf", log)


def review_toys(R, log):
    """Worked examples added after the student review; every printed number asserted."""
    S = QUOTED["spec_toy"]
    small, big = np.array(S["small"]), np.array(S["big"])
    acc_prob = np.minimum(1, big / small)
    accepted = small * acc_prob
    reject = 1 - accepted.sum()
    residual = np.maximum(0, big - small)
    residual = residual / residual.sum()
    final = accepted + reject * residual
    assert [round(v, 2) for v in accepted] == S["accepted"] and round(reject, 2) == S["reject"]
    assert [round(v, 2) for v in residual] == S["residual"], residual
    assert np.allclose(final, big) and [round(v, 2) for v in final] == S["final"], final

    head = np.array([0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02])
    tail_n, tail_p = 50000, 0.000005
    tail_share = {}
    for T in (0.5, 1, 2):
        h, tl = (head ** (1 / T)).sum(), tail_n * tail_p ** (1 / T)
        tail_share[str(T)] = tl / (h + tl)
    assert {k: round(v, 2) for k, v in tail_share.items()} == QUOTED["tail_T"], tail_share

    O = QUOTED["order_toy"]
    po = np.array(O["p"])
    after_T = po ** (1 / O["T"])
    after_T /= after_T.sum()

    def nucleus(q):
        return int(np.searchsorted(np.cumsum(np.sort(q)[::-1]), O["top_p"] - 1e-12)) + 1
    assert [round(v, 2) for v in after_T] == O["after_T"], after_T
    assert nucleus(after_T) == O["temp_first_keeps"] and nucleus(po) == O["cut_first_keeps"]
    R["review_toys"] = {"speculative": {"accepted": accepted.tolist(), "reject": reject,
                                        "residual": residual.tolist(), "final": final.tolist()},
                        "tail_share_by_T": tail_share, "order": {"after_T": after_T.tolist()}}
    log.info(f"review toys: {R['review_toys']}")


def toy_checks(R, log):
    small = np.float32(np.finfo(np.float32).smallest_subnormal)
    prod = np.float32(1.0)
    for _ in range(100):
        prod = np.float32(prod * np.float32(0.1))
    a, b = np.float32(1e8), np.float32(1.0)
    left, right = (a + b) - a, (a - a) + b
    assert prod == 0 and small < 1.5e-45 and left == 0 and right == 1, (prod, small, left, right)
    R["toys"] = {"float32_product_0.1_x100": float(prod), "float32_smallest": float(small),
                 "log_sum": 100 * math.log(0.1), "fp_order": [float(left), float(right)]}
    log.info(f"toys: {R['toys']}")


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        log.info(f"done in {time.time() - t0:.0f} s")
        return
    for p in (GPT2_JSON, MAP_JSON):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run with --gpt2 (and llm_roadmap.py --gpt2) first")
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"date": time.strftime("%Y-%m-%d")}
    jobs = [fig_greedy_loop, fig_map_top5, fig_ayb_bytes, lambda R, log: fig_tree(R, log, "greedy"),
            lambda R, log: fig_tree(R, log, "beam"), fig_prob_trap, fig_long_tail, fig_temperature,
            fig_topk_topp, fig_knob_pipeline, fig_json_mask, fig_speculative, toy_checks, review_toys]
    for k, job in enumerate(jobs, 1):
        job(R, log)
        el = time.time() - t0
        log.info(f"{k}/{len(jobs)}, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left")
    (RESULTS / "decoding_figs.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
