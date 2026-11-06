"""Figures for LLM-6 "Decoding: from probabilities to text" (llms/06_decoding/LLM6_decoding.tex).

Toy, demonstrative numbers by default (ml/SLIDE_STYLE.md, 2026-10-05). Two real inputs only:
  - the cold open: GPT-2's greedy continuation of COLD_PROMPT (the cached GPT-2, no download),
    produced once with --gpt2 and stored in results/decoding_gpt2.json;
  - the map's real top-5 after "The cat sat on the" (results/roadmap_gpt2.json, from
    llm_roadmap.py --gpt2).
Numbers the slides quote are asserted (QUOTED), so a changed input fails loudly.

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text, and
save() refuses any text outside its figure.

Two steps, so torch is not imported for every redraw:
  ./ma/Scripts/python.exe llms/06_decoding/py_src/decoding_figs.py --gpt2   # greedy run -> JSON (~2 min, mostly importing torch)
  ./ma/Scripts/python.exe llms/06_decoding/py_src/decoding_figs.py          # the figures (~15 s)

Generates into llms/06_decoding/fig/: dec_*.pdf.
"""
import json
import logging
import math
import re
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
MAP_JSON = ROOT / "llms" / "00_shared" / "results" / "roadmap_gpt2.json"   # shared map

COLD_PROMPT = "The cat sat on the"     # the map's sentence; all 6 prompts tried looped under greedy
N_NEW = 50

# Toy numbers (all demonstrative) and the values the slides quote.
TREE = {"on": (0.5, {"the": 0.3, "a": 0.25, "my": 0.2}), "in": (0.4, {"a": 0.9, "the": 0.1}),
        "by": (0.1, {"the": 0.5, "a": 0.5})}
# Beam search, step 3 (2026-11-06): only the two beams kept after step 2 get children.
TREE3 = {("in", "a"): {"box": 0.5, "hat": 0.3}, ("on", "the"): {"mat": 0.6, "floor": 0.3}}
BEAM_B = 2
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
                        "cut_first_keeps": 2, "after_T": [0.38, 0.29, 0.21, 0.12]},
          # second student review (2026-10-20): beam's log scores, all four knobs in order
          "beam_logs": {"best": -1.02, "second": -1.90},
          "chain_toy": {"p": [0.30, 0.20, 0.16, 0.12, 0.10, 0.07, 0.05], "T": 0.5, "k": 5,
                        "top_p": 0.9, "min_p": 0.2, "squares_sum": 0.1874,
                        "after_T": [0.48, 0.21, 0.14, 0.08, 0.05, 0.03, 0.01],
                        "after_k": [0.50, 0.22, 0.14, 0.08, 0.06],
                        "running": [0.50, 0.72, 0.86, 0.94], "after_p": [0.53, 0.24, 0.15, 0.08],
                        "bar": 0.106, "after_minp": [0.58, 0.26, 0.16]},
          # instructor review (2026-11-06): beam search step by step, logs, temperature extremes
          "beam3": {"step1": [["on", -0.69], ["in", -0.92], ["by", -2.30]],
                    "step2": [["in a", -1.02], ["on the", -1.90], ["on a", -2.08], ["on my", -2.30],
                              ["in the", -3.22]],
                    "step3": [["in a box", -1.71], ["in a hat", -2.23], ["on the mat", -2.41],
                              ["on the floor", -3.10]],
                    "winner": 0.18, "greedy3": 0.09},
          "float32_zero_at": 46,
          "temp_table": {"0.1": [1.0, 0.0, 0.0], "0.5": [0.87, 0.12, 0.02], "1": [0.67, 0.24, 0.09],
                         "2": [0.51, 0.31, 0.19], "10": [0.37, 0.33, 0.30], "100": [0.34, 0.33, 0.33]},
          "temp_map": {"0.1": 0.79, "10_spread": 0.01},
          "penalty": {"before": [0.54, 0.13, 0.11, 0.09, 0.07, 0.05], "mult": [0.39, 0.19, 0.11, 0.13, 0.11, 0.07],
                      "sub": [0.17, 0.30, 0.06, 0.20, 0.17, 0.11]}}

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


def tree_paths():
    out = {}
    for w1, (p1, nxt) in TREE.items():
        for w2, p2 in nxt.items():
            out[(w1, w2)] = p1 * p2
    return out


def fig_tree(R, log, mode="greedy"):
    """The greedy tree (2 steps). The beam version became fig_beam_steps (3 frames) on 2026-11-06."""
    assert mode == "greedy", mode
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
        col1 = ARMRED if w1 == greedy1 else "#BBBBBB"
        arrow(ax, (root[0] + 0.24, root[1]), (x1 - 0.30, y1[w1]), color=col1, lw=1.0)
        rbox(ax, x1 - 0.28, y1[w1] - 0.11, 0.56, 0.22, "white", col1, lw=1.0, z=3)
        ax.text(x1, y1[w1], f"{w1}  {p1:g}", ha="center", va="center", fontsize=FS, zorder=4,
                color=INK if col1 != "#BBBBBB" else "#999999")
        for w2, yy in leaves[w1]:
            p = paths[(w1, w2)]
            col = ARMRED if (w1, w2) == (greedy1, greedy2) else (GREEN if (w1, w2) == best else "#BBBBBB")
            arrow(ax, (x1 + 0.30, y1[w1]), (x2 - 0.32, yy), color=col, lw=0.9)
            rbox(ax, x2 - 0.30, yy - 0.10, 0.60, 0.20, "white", col, lw=1.0, z=3)
            ax.text(x2, yy, f"{w2}  {TREE[w1][1][w2]:g}", ha="center", va="center", fontsize=FS_SMALL,
                    zorder=4, color=INK if col != "#BBBBBB" else "#999999")
            ax.text(x2 + 0.36, yy, f"= {p:.3g}", fontsize=FS_SMALL, va="center",
                    color=col if col != "#BBBBBB" else "#999999", fontweight="bold" if col != "#BBBBBB" else "normal")
    ax.text(x3 + 0.12, 1.45, "greedy: on, then the", fontsize=FS_SMALL, color=ARMRED, ha="left")
    ax.text(x3 + 0.12, 1.30, "0.5 x 0.3 = 0.15", fontsize=FS_SMALL, color=ARMRED, ha="left")
    ax.text(x3 + 0.12, 0.95, "best: in, then a", fontsize=FS_SMALL, color=GREEN, ha="left")
    ax.text(x3 + 0.12, 0.80, "0.4 x 0.9 = 0.36", fontsize=FS_SMALL, color=GREEN, ha="left")
    ax.text(x1, 1.86, "step 1", ha="center", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(x2, 1.86, "step 2 (path probability)", ha="center", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, "dec_greedy_tree.pdf", log)


def beam_children(path):
    if len(path) == 0:
        return {w: p for w, (p, _) in TREE.items()}
    if len(path) == 1:
        return TREE[path[0]][1]
    return TREE3.get(path, {})


def beam_search_toy():
    """The toy beam search, B = 2, three steps. Score = total log-probability."""
    beams, steps = [((), 0.0)], []
    for _ in range(3):
        cands = [(path + (w,), lp + math.log(p)) for path, lp in beams for w, p in beam_children(path).items()]
        cands.sort(key=lambda c: -c[1])
        steps.append({"cands": cands, "kept": [c[0] for c in cands[:BEAM_B]]})
        beams = cands[:BEAM_B]
    return steps


def fig_beam_steps(R, log):
    """Beam search with B = 2, one figure per step (2026-11-06, instructor: 'explain and show beam
    search better'). The newest column's candidates are ranked; green = kept, grey = dropped."""
    steps = beam_search_toy()
    B = QUOTED["beam3"]
    for k, key in enumerate(("step1", "step2", "step3")):
        got = [[" ".join(p), round(lp, 2)] for p, lp in steps[k]["cands"]]
        assert got == B[key], (key, got)
    winner = steps[2]["kept"][0]
    assert round(math.exp(steps[2]["cands"][0][1]), 2) == B["winner"] and winner == ("in", "a", "box")
    g = ()                                                      # greedy over the same toy, 3 steps
    gp = 1.0
    for _ in range(3):
        w, p = max(beam_children(g).items(), key=lambda kv: kv[1])
        g, gp = g + (w,), gp * p
    assert g == ("on", "the", "mat") and round(gp, 2) == B["greedy3"], (g, gp)
    R["beam3"] = {"steps": [{"cands": [[" ".join(p), lp] for p, lp in s["cands"]],
                             "kept": [" ".join(p) for p in s["kept"]]} for s in steps],
                  "greedy": " ".join(g), "greedy_p": gp}

    X = [1.25, 2.45, 3.65]
    Y = {("on",): 1.50, ("in",): 0.95, ("by",): 0.38,
         ("on", "the"): 1.86, ("on", "a"): 1.60, ("on", "my"): 1.34, ("in", "a"): 1.03, ("in", "the"): 0.77,
         ("on", "the", "mat"): 1.98, ("on", "the", "floor"): 1.72, ("in", "a", "box"): 1.16, ("in", "a", "hat"): 0.90}
    lp = {p: s for st in steps for p, s in st["cands"]}
    final_beams = {}
    for upto in range(3):
        fig, ax = canvas(2.25)
        root = (0.40, 1.10)
        rbox(ax, root[0] - 0.22, root[1] - 0.11, 0.44, 0.22, TINT_BLUE, ARMBLUE, z=3)
        ax.text(*root, "sat", ha="center", va="center", fontsize=FS, zorder=4)
        alive = set(steps[upto]["kept"])
        for s in range(upto + 1):
            newest = s == upto
            for path, _ in steps[s]["cands"]:
                x, y = X[s], Y[path]
                parent = path[:-1]
                px, py = (root[0] + 0.23, root[1]) if not parent else (X[s - 1] + 0.32, Y[parent])
                if newest:
                    kept = path in alive
                    ec, fc, tc = (GREEN, TINT_GREEN, INK) if kept else ("#BBBBBB", "white", "#999999")
                else:
                    on_beam = any(a[:len(path)] == path for a in alive)
                    ec, fc, tc = (ARMBLUE, TINT_BLUE, INK) if on_beam else ("#CCCCCC", "white", "#AAAAAA")
                arrow(ax, (px, py), (x - 0.32, y), color=ec, lw=0.8)
                rbox(ax, x - 0.30, y - 0.10, 0.60, 0.20, fc, ec, lw=0.9, z=3)
                ax.text(x, y, f"{path[-1]}  {beam_children(parent)[path[-1]]:g}", ha="center", va="center",
                        fontsize=FS_SMALL, color=tc, zorder=4)
                if newest:
                    ax.text(x + 0.36, y, f"{math.exp(lp[path]):.3g}  (log {lp[path]:.2f})", fontsize=FS_TINY,
                            va="center", color=GREEN if path in alive else "#999999",
                            fontweight="bold" if path in alive else "normal")
            ax.text(X[s], 2.17, f"step {s + 1}", ha="center", va="center", fontsize=FS_TINY, color=GREY)
        n_c, n_b = len(steps[upto]["cands"]), BEAM_B
        notes = [f"{n_c} candidates, keep the best {n_b}",
                 "(green: kept, grey: dropped; number = path probability)"]
        if upto == 2:
            notes = ["both beams now start with \"in a\": \"on the\" fell out",
                     "greedy would have written \"on the mat\": 0.09"]
            final_beams = {"winner": "in a box", "p": 0.18}
        for j, t in enumerate(notes):
            ax.text(5.42, 0.36 - 0.16 * j, t, fontsize=FS_TINY, ha="right", va="center",
                    color=(ARMRED if (upto == 2 and j == 1) else GREY))
        save(fig, f"dec_beam_step{upto + 1}.pdf", log)
    R["beam3"]["final"] = final_beams


PEN_TOKENS = ["sorry", "not", ".", "so", "afraid", "here"]
PEN_LOGITS = [3.0, 1.6, 1.4, 1.2, 1.0, 0.6]
PEN_COUNTS = [3, 0, 2, 0, 0, 0]                 # times each token already appeared in the text


def fig_penalty(R, log):
    """Repetition penalties, seen (instructor 2026-11-06: 'more visual stuff ... for the penalty one').
    Toy next-token distribution after GPT-2's loop "I'm sorry. I'm sorry. I'm"; both definitions."""
    z, c = np.array(PEN_LOGITS), np.array(PEN_COUNTS)

    def softmax(v):
        return np.exp(v) / np.exp(v).sum()
    before = softmax(z)
    mult = softmax(np.where(c > 0, np.where(z > 0, z / 1.3, z * 1.3), z))
    sub = softmax(z - c * 0.5 - (c > 0) * 0.5)
    Q = QUOTED["penalty"]
    for name, v in (("before", before), ("mult", mult), ("sub", sub)):
        assert [round(float(x), 2) for x in v] == Q[name], (name, v)
    R["penalty"] = {"before": before.tolist(), "mult": mult.tolist(), "sub": sub.tolist()}
    fig = plt.figure(figsize=(W, 1.85))
    x = np.arange(len(PEN_TOKENS))
    panels = (("repetition_penalty = 1.3\n(Hugging Face, llama.cpp): divide the logit", mult, ARMBLUE),
              ("presence 0.5 + frequency 0.5 per use\n(OpenAI): subtract, more for each use", sub, ARMORANGE))
    for k, (title, after, col) in enumerate(panels):
        ax = fig.add_axes([0.05 + k * 0.50, 0.24, 0.43, 0.52])
        b1 = ax.bar(x - 0.19, before, width=0.36, color="#CCCCCC", label="before")
        b2 = ax.bar(x + 0.19, after, width=0.36, color=col, label="after")
        ax.bar_label(b1, labels=[f".{round(v * 100):02d}" for v in before], fontsize=5.5, padding=1, color=GREY)
        ax.bar_label(b2, labels=[f".{round(v * 100):02d}" for v in after], fontsize=5.5, padding=1, color=col)
        ax.set_xticks(x, [f"{t}\n(used {n}x)" if n else t for t, n in zip(PEN_TOKENS, PEN_COUNTS)], fontsize=FS_TINY)
        for lab, n in zip(ax.get_xticklabels(), PEN_COUNTS):
            if n:
                lab.set_color(ARMRED)
        ax.set_yticks([])
        ax.set_ylim(0, 0.66)
        ax.set_title(title, fontsize=FS_TINY, loc="left")
        ax.legend(fontsize=FS_TINY, frameon=False, loc="upper right")
        clean(ax)
        ax.spines["left"].set_visible(False)
    fig.text(0.01, 0.97, "toy: the next token after \"... I'm sorry. I'm sorry. I'm\"", fontsize=FS_TINY,
             color=GREY, va="top")
    save(fig, "dec_penalty.pdf", log)


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


def fig_logprob(R, log):
    """Left: multiplying 100 probabilities of 0.1 in float32 hits exactly 0. Right: adding their logs
    is a straight line (2026-11-06, instructor: 'explain better why work with logprob')."""
    n = 100
    prod, run = np.float32(1.0), []
    for _ in range(n):
        prod = np.float32(prod * np.float32(0.1))
        run.append(float(prod))
    zero_at = next(i for i, v in enumerate(run, 1) if v == 0.0)
    assert zero_at == QUOTED["float32_zero_at"], zero_at
    logs = np.cumsum([math.log(0.1)] * n)
    assert round(logs[-1], 1) == -230.3
    R["logprob"] = {"float32_zero_at": zero_at, "last_nonzero": run[zero_at - 2], "log_sum_100": logs[-1]}
    fig = plt.figure(figsize=(W, 1.75))
    ax = fig.add_axes([0.10, 0.22, 0.36, 0.62])
    xs = np.arange(1, n + 1)
    ax.semilogy(xs, [10.0 ** -k for k in xs], color="#BBBBBB", lw=1.0, ls=(0, (3, 2)), label="true value")
    nz = [(x, v) for x, v in zip(xs, run) if v > 0]
    ax.semilogy([x for x, _ in nz], [v for _, v in nz], color=ARMRED, lw=1.4, label="float32")
    ax.axvline(zero_at, color=ARMRED, lw=0.6, ls=":")
    ax.text(zero_at + 3, 1e-20, f"token {zero_at}:\nfloat32 says 0", fontsize=FS_TINY, color=ARMRED, va="center")
    ax.set_ylim(1e-100, 10)
    ax.set_yticks([1, 1e-25, 1e-50, 1e-75, 1e-100])
    ax.set_yticklabels(["1", "$10^{-25}$", "$10^{-50}$", "$10^{-75}$", "$10^{-100}$"], fontsize=FS)  # exponents >= 5.5 pt
    ax.set_xlabel("tokens so far", labelpad=1)
    ax.set_title("multiply the probabilities", fontsize=FS_SMALL)
    ax.legend(fontsize=FS_TINY, frameon=False, loc="lower left")
    clean(ax)
    ax2 = fig.add_axes([0.60, 0.22, 0.36, 0.62])
    ax2.plot(xs, logs, color=ARMBLUE, lw=1.4)
    ax2.set_ylim(-250, 5)
    ax2.set_xlabel("tokens so far", labelpad=1)
    ax2.set_title("add the log-probabilities", fontsize=FS_SMALL)
    ax2.text(98, -45, "after 100 tokens: -230.3,\nstill an ordinary number", fontsize=FS_TINY, color=ARMBLUE,
             ha="right", va="center")
    clean(ax2)
    ax2.text(3, -215, "toy: every token has\nprobability 0.1", fontsize=FS_TINY, color=GREY, ha="left", va="center")
    save(fig, "dec_logprob.pdf", log)


def temperature(p, T):
    q = np.array(p) ** (1 / T)
    return q / q.sum()


TEMPS = (0.1, 0.5, 1.0, 2.0, 10.0)
TEMP_COLORS = ("#7A0009", ARMRED, ARMORANGE, ARMBLUE, "#001A55")


def fig_temperature(R, log):
    z = np.array([2.0, 1.0, 0.0])
    t1 = np.exp(z) / np.exp(z).sum()
    t10 = np.exp(z / 10) / np.exp(z / 10).sum()
    assert [round(v, 2) for v in t1] == QUOTED["temp_T1"], t1
    assert [round(v, 2) for v in t10] == QUOTED["temp_T10"], t10
    table = {}
    for T in (0.1, 0.5, 1, 2, 10, 100):                          # the by-hand table, extremes included
        q = np.exp(z / T) / np.exp(z / T).sum()
        table[f"{T:g}"] = q.tolist()
        assert [round(float(v), 2) for v in q] == QUOTED["temp_table"][f"{T:g}"], (T, q)
    m = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    top = [(t.strip(), p) for t, p in m["top5"]]
    fig = plt.figure(figsize=(W, 1.60))
    out = {}
    for k, (T, col) in enumerate(zip(TEMPS, TEMP_COLORS)):
        q = temperature([p for _, p in top], T)
        out[f"{T:g}"] = q.round(3).tolist()
        ax = fig.add_axes([0.03 + k * 0.197, 0.30, 0.165, 0.50])
        bars = ax.bar(range(5), q, color=col, width=0.7)
        ax.bar_label(bars, labels=[f"{v:.2f}" if i in (0, 4) else "" for i, v in enumerate(q)], fontsize=5.5,
                     padding=1)                                 # first and last only: level bars collide
        ax.set_xticks(range(5), [t for t, _ in top], fontsize=5.5, rotation=40, ha="right",
                      rotation_mode="anchor")
        ax.set_yticks([])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"T = {T:g}", fontsize=FS_SMALL)
        clean(ax)
        ax.spines["left"].set_visible(False)
    assert round(out["0.1"][0], 2) == QUOTED["temp_map"]["0.1"], out
    assert max(out["10"]) - min(out["10"]) < QUOTED["temp_map"]["10_spread"], out
    R["temperature"] = {"toy_T1": t1.round(4).tolist(), "toy_T10": t10.round(4).tolist(), "toy_table": table,
                        "map_top5": out}
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


# A toy grammar with one integer field, {"year": <digits>}, and a toy vocabulary of 9 tokens.
GRAMMAR_KEY = '{"year":'
GRAMMAR_TOKENS = ['"', '"year', '":', " 19", "99", "}", '"}', " The", "<end>"]
GRAMMAR_STATES = ["{", '{"year', '{"year":', '{"year": 19', '{"year": 1999}']
GRAMMAR_QUOTED = ["110000000", "101000000", "000110000", "000011000", "000000001"]


def grammar_complete(text):
    return re.fullmatch(r'\{"year": ?[0-9]+\}', text) is not None


def grammar_viable(text):
    """Can `text` still grow into {"year": <digits>}?"""
    k = GRAMMAR_KEY
    if len(text) <= len(k):
        return k.startswith(text)
    if not text.startswith(k):
        return False
    m = re.fullmatch(r" ?([0-9]*)(\}?)", text[len(k):])
    return bool(m) and (m.group(1) != "" or m.group(2) == "")


def grammar_row(state):
    """The whole token's text is checked against the grammar, never just its first character."""
    return "".join("1" if (grammar_complete(state) if t == "<end>" else
                           not grammar_complete(state) and grammar_viable(state + t)) else "0"
                   for t in GRAMMAR_TOKENS)


def fig_grammar_table(R, log):
    rows = [grammar_row(s) for s in GRAMMAR_STATES]
    if rows != GRAMMAR_QUOTED:
        raise ValueError(f"grammar table changed: {rows}; the slide reads {GRAMMAR_QUOTED}")
    if grammar_row('{"year": 1999') != grammar_row('{"year": 19'):     # the keybox: one state, one row
        raise ValueError("texts in the same grammar state should share a row")
    R["grammar_table"] = dict(zip(GRAMMAR_STATES, rows))
    log.info(f"grammar table: {R['grammar_table']}")

    fig, ax = canvas(1.5)
    x0, cw, rh, top = 1.62, 0.425, 0.215, 1.25
    ax.text(x0 - 0.1, top + 0.05, "text so far", ha="right", va="bottom", fontsize=FS_TINY,
            color=GREY, style="italic")
    for j, t in enumerate(GRAMMAR_TOKENS):
        ax.text(x0 + (j + 0.5) * cw, top + 0.05, t.replace(" ", "␣"), ha="center", va="bottom",
                fontsize=FS_SMALL, family=MONO if t != "<end>" else SANS,
                color=INK if t != "<end>" else GREY)
    for i, (state, row) in enumerate(zip(GRAMMAR_STATES, rows)):
        y = top - (i + 1) * rh
        ax.text(0.05, y + rh / 2, f"{i + 1}", ha="left", va="center", fontsize=FS_TINY, color=GREY)
        ax.text(x0 - 0.1, y + rh / 2, state, ha="right", va="center", fontsize=FS_SMALL, family=MONO)
        for j, ok in enumerate(row):
            ok = ok == "1"
            ax.add_patch(Rectangle((x0 + j * cw, y), cw, rh, facecolor=TINT_GREEN if ok else "#F4F4F4",
                                   edgecolor="white", lw=1.2))
            ax.text(x0 + (j + 0.5) * cw, y + rh / 2, "✓" if ok else "✗", ha="center",
                    va="center", fontsize=FS, color=GREEN if ok else "#BBBBBB",
                    fontweight="bold" if ok else "normal")
    ax.text(W / 2, 0.03, "toy grammar {\"year\": <digits>}, toy vocabulary of 9 tokens; "
            "✓ = legal next token", ha="center", fontsize=FS_TINY, color=GREY)
    save(fig, "dec_grammar_table.pdf", log)


def fig_speculative(R, log):
    """Overview, greedy version. Same example as fig_spec_verify (student review 2026-11-06: the two
    frames used different sentences and different token counts)."""
    p_big, p_small = 0.3, 0.6
    assert min(1, p_big / p_small) == QUOTED["spec_accept"]
    status = []
    for j, d in enumerate(SPEC_DRAFT):
        guess = SPEC_BIG[len(SPEC_PROMPT) - 1 + j]
        if "no" in status:
            status.append("skip")
        else:
            status.append("ok" if guess == d else "no")
    n_ok = status.count("ok")
    fix = SPEC_BIG[len(SPEC_PROMPT) - 1 + n_ok]
    assert (n_ok, fix) == (2, "floor"), (status, fix)
    fig, ax = canvas(1.52)
    ax.text(0.04, 1.40, "greedy version: \"ok\" means the drafted token equals the big model's own top token at "
            "that position", fontsize=FS_TINY, color=GREY, va="center")
    ax.text(0.04, 1.08, "after \"The cat sat\",\nsmall model drafts 4:", fontsize=FS_SMALL, color=ARMBLUE,
            va="center", linespacing=1.15)
    ax.text(0.04, 0.62, "big model checks all 4\nin one pass:", fontsize=FS_SMALL, color=INK, va="center",
            linespacing=1.15)
    style = {"ok": (TINT_GREEN, GREEN, "ok"), "no": (TINT_RED, ARMRED, "no"), "skip": ("white", "#BBBBBB", "-")}
    for k, (w, st) in enumerate(zip(SPEC_DRAFT, status)):
        x = 1.62 + k * 0.58
        rbox(ax, x, 0.96, 0.52, 0.24, TINT_ORANGE, ARMORANGE, z=2)
        ax.text(x + 0.26, 1.08, w, ha="center", va="center", fontsize=FS, zorder=3)
        arrow(ax, (x + 0.26, 0.95), (x + 0.26, 0.76), lw=0.7)
        fc, ec, lab = style[st]
        rbox(ax, x, 0.50, 0.52, 0.24, fc, ec, z=2)
        ax.text(x + 0.26, 0.62, lab, ha="center", va="center", fontsize=FS, color=ec, fontweight="bold", zorder=3)
    ax.text(1.62 + 4 * 0.58 + 0.08, 0.62, f"-> big model's own\n   token: \"{fix}\";\n   \".\" is dropped",
            fontsize=FS_SMALL, color=ARMRED, va="center", linespacing=1.15)
    ax.plot([1.62, 1.62 + 2 * 0.58 - 0.08], [0.36, 0.36], color=GREEN, lw=1.2)
    ax.text(1.62, 0.20, f"{n_ok} tokens accepted + 1 corrected = {n_ok + 1} tokens for one big pass",
            ha="left", fontsize=FS_SMALL, color=GREEN)
    save(fig, "dec_speculative.pdf", log)


SPEC_PROMPT = ["The", "cat", "sat"]
SPEC_DRAFT = ["on", "the", "mat", "."]
SPEC_BIG = {2: "on", 3: "the", 4: "floor", 5: ".", 6: "It"}    # big model's top token after position i (toy)


def fig_spec_verify(R, log):
    """One forward pass of the big model over prompt + draft gives a next-token guess at every
    position; compare each guess with the drafted token one column to its right (greedy, toy)."""
    seq = SPEC_PROMPT + SPEC_DRAFT
    accepted, fix = [], None
    for j, d in enumerate(SPEC_DRAFT):
        guess = SPEC_BIG[len(SPEC_PROMPT) - 1 + j]
        if guess == d:
            accepted.append(d)
        else:
            fix = guess
            break
    assert accepted == ["on", "the"] and fix == "floor", (accepted, fix)
    R["spec_verify"] = {"accepted": accepted, "fix": fix, "out": accepted + [fix]}
    fig, ax = canvas(1.90)                      # drawn at full slot width (5.5 in) so text stays >= 6.5 pt
    x0, dx, bw = 0.95, 0.56, 0.48
    xs = [x0 + i * dx for i in range(len(seq) + 1)]
    ax.text(0.04, 0.33, "input:", fontsize=FS_SMALL, color=INK, va="center")
    ax.text(0.04, 1.21, "big model's\nguess:", fontsize=FS_SMALL, color=INK, va="center", linespacing=1.1)
    ax.text(0.04, 1.58, "check:", fontsize=FS_SMALL, color=INK, va="center")
    ax.text(0.04, 1.40, "made at:", fontsize=FS_TINY, color=GREY, va="center")
    for i, w in enumerate(seq):
        draft = i >= len(SPEC_PROMPT)
        rbox(ax, xs[i], 0.22, bw, 0.22, TINT_ORANGE if draft else TINT_BLUE, ARMORANGE if draft else ARMBLUE, z=2)
        ax.text(xs[i] + bw / 2, 0.33, w, ha="center", va="center", fontsize=FS_SMALL, zorder=3)
        arrow(ax, (xs[i] + bw / 2, 0.46), (xs[i] + bw / 2, 0.58), lw=0.6)
    ax.text(xs[3], 0.10, "drafted by the small model", fontsize=FS_TINY, color=ARMORANGE, va="center")
    ax.text(xs[0], 0.10, "prompt", fontsize=FS_TINY, color=ARMBLUE, va="center")
    rbox(ax, xs[0] - 0.05, 0.60, xs[len(seq) - 1] + bw - xs[0] + 0.10, 0.28, "#F4F4F4", GREY, z=1)
    ax.text((xs[0] + xs[len(seq) - 1] + bw) / 2, 0.74, "big model: ONE forward pass over all 7 tokens",
            ha="center", va="center", fontsize=FS_SMALL, color=INK, zorder=3)
    for i in range(len(seq)):
        src = (xs[i] + bw / 2, 0.89)
        if i not in SPEC_BIG:                                   # guesses inside the prompt: not needed
            continue
        tgt_col = i + 1
        j = tgt_col - len(SPEC_PROMPT)
        status = "ok" if j < len(accepted) else ("fix" if j == len(accepted) else "gone")
        col = {"ok": GREEN, "fix": ARMRED, "gone": "#BBBBBB"}[status]
        arrow(ax, src, (xs[tgt_col] + bw / 2, 1.08), color=col, lw=0.7)
        rbox(ax, xs[tgt_col], 1.10, bw, 0.22, "white", col, lw=0.9, z=2)
        ax.text(xs[tgt_col] + bw / 2, 1.21, SPEC_BIG[i], ha="center", va="center", fontsize=FS_SMALL,
                color=INK if status != "gone" else "#AAAAAA", zorder=3)
        ax.text(xs[tgt_col] + bw / 2, 1.40, f"\"{seq[i]}\"", ha="center", va="center", fontsize=FS_TINY,
                color=GREY if status != "gone" else "#BBBBBB")      # which position produced this guess
        mark = {"ok": "same", "fix": "differs", "gone": "dropped"}[status]
        ax.text(xs[tgt_col] + bw / 2, 1.58, mark, ha="center", va="center", fontsize=FS_TINY, color=col,
                fontweight="bold" if status != "gone" else "normal")
    ax.text(5.45, 1.80, "a guess made at position i sees only tokens 1..i (how: LLM-3), so it equals what i separate passes would give",
            fontsize=FS_TINY, color=GREY, ha="right", va="center")
    save(fig, "dec_spec_verify.pdf", log)


SPEC_SMALL_COST, SPEC_K = 0.1, 4
SPEC_ROUNDS = [2, 4, 3]                                         # drafted tokens accepted per round (toy)


def fig_spec_timeline(R, log):
    """Where the speed comes from: time measured in big-model passes (toy costs)."""
    round_cost = SPEC_K * SPEC_SMALL_COST + 1.0
    horizon = round_cost * len(SPEC_ROUNDS)
    spec_tokens = sum(a + 1 for a in SPEC_ROUNDS)
    plain_tokens = int(horizon)                                 # finished big passes only
    assert math.isclose(round_cost, 1.4) and math.isclose(horizon, 4.2) and spec_tokens == 12 and plain_tokens == 4
    R["spec_timeline"] = {"round_cost": round_cost, "horizon": horizon, "spec_tokens": spec_tokens,
                          "plain_tokens": plain_tokens, "speedup": spec_tokens / plain_tokens}
    fig, ax = canvas(1.55)
    x0, unit = 1.15, 1.0
    for row, (y, name) in enumerate(((1.05, "plain decoding"), (0.48, "speculative"))):
        ax.text(0.04, y + 0.11, name, fontsize=FS_SMALL, color=INK, va="center")
    t, k = 0.0, 0
    while t + 1.0 <= horizon + 1e-9:
        rbox(ax, x0 + t * unit + 0.01, 1.05, unit - 0.02, 0.22, TINT_RED, ARMRED, z=2)
        k += 1
        ax.text(x0 + (t + 0.5) * unit, 1.16, f"big pass: token {k}", ha="center", va="center", fontsize=FS_TINY)
        t += 1.0
    rbox(ax, x0 + t * unit + 0.01, 1.05, (horizon - t) * unit - 0.02, 0.22, TINT_RED, ARMRED, z=2)
    total = 0
    t = 0.0
    for r, acc in enumerate(SPEC_ROUNDS):
        for s in range(SPEC_K):
            rbox(ax, x0 + t * unit + 0.005, 0.48, SPEC_SMALL_COST * unit - 0.01, 0.22, TINT_ORANGE, ARMORANGE, lw=0.4, z=2)
            t += SPEC_SMALL_COST
        rbox(ax, x0 + t * unit + 0.01, 0.48, unit - 0.02, 0.22, TINT_GREEN, GREEN, z=2)
        total += acc + 1
        ax.text(x0 + (t + 0.5) * unit, 0.59, f"check: +{acc + 1} tokens", ha="center", va="center", fontsize=FS_TINY)
        t += 1.0
    ax.text(x0 + horizon * unit, 1.38, f"{plain_tokens} tokens", ha="right", fontsize=FS_SMALL, color=ARMRED)
    ax.text(x0 + horizon * unit, 0.81, f"{spec_tokens} tokens", ha="right", fontsize=FS_SMALL, color=GREEN,
            fontweight="bold")
    ax.annotate("", (x0 + horizon * unit, 0.24), (x0, 0.24), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6))
    ax.text(x0, 0.10, "time in big-model passes (toy: a draft step costs 0.1; 2, 4, 3 of 4 drafts accepted)",
            fontsize=FS_TINY, color=GREY, va="center")
    ax.text(x0 + 0.01, 0.76, "4 small draft steps", fontsize=FS_TINY, color=ARMORANGE, va="center")
    save(fig, "dec_spec_timeline.pdf", log)


def fig_spec_heads(R, log):
    """Draft heads on the big model itself (Medusa-style), schematic."""
    fig, ax = canvas(1.80)
    cy = 1.01
    rbox(ax, 0.10, cy - 0.40, 1.25, 0.80, TINT_BLUE, ARMBLUE, z=2)
    ax.text(0.725, cy + 0.09, "big model", ha="center", va="center", fontsize=FS, fontweight="bold", zorder=3)
    ax.text(0.725, cy - 0.11, "all layers,\nrun once", ha="center", va="center", fontsize=FS_TINY, zorder=3,
            linespacing=1.1)
    rbox(ax, 1.55, cy - 0.13, 0.36, 0.26, "white", INK, z=2)
    ax.text(1.73, cy, "h", ha="center", va="center", fontsize=FS, style="italic", zorder=3)
    arrow(ax, (1.36, cy), (1.54, cy))
    heads = [("LM head", "token t+1", "on", ARMBLUE),
             ("head 2", "token t+2", "the", ARMORANGE),
             ("head 3", "token t+3", "mat", ARMORANGE),
             ("head 4", "token t+4", ".", ARMORANGE)]
    for j, (name, what, word, col) in enumerate(heads):
        y = 1.55 - j * 0.36
        arrow(ax, (1.92, cy), (2.30, y), color=col, lw=0.7)
        rbox(ax, 2.32, y - 0.12, 1.55, 0.24, "white", col, z=2)
        ax.text(3.095, y, f"{name}: {what}", ha="center", va="center", fontsize=FS_TINY, zorder=3)
        arrow(ax, (3.88, y), (4.10, y), color=col, lw=0.7)
        rbox(ax, 4.12, y - 0.11, 0.50, 0.22, TINT_ORANGE if j else TINT_BLUE, col, z=2)
        ax.text(4.37, y, word, ha="center", va="center", fontsize=FS_SMALL, zorder=3)
    ax.text(4.75, cy, "= a draft\nof 4 tokens,\nchecked in\nthe next pass", fontsize=FS_TINY, color=INK,
            va="center", linespacing=1.15)
    ax.text(0.10, 0.10, "each extra head is one small layer trained on top of the frozen big model; "
            "no second model to run", fontsize=FS_TINY, color=GREY, va="center")
    save(fig, "dec_spec_heads.pdf", log)


def fig_prompt_cache(R, log):
    """Three requests that share (or do not share) a long beginning. Schematic token counts."""
    fig, ax = canvas(1.75)
    x0, scale = 1.55, 3.2 / 3300
    rows = [("request 1", [("instructions + lecture notes: 3,300 tokens", 3300, "full"), ("Q1", 60, "full")],
             "pays full price; the provider keeps its work on the prefix"),
            ("request 2\n(a minute later)", [("same 3,300 tokens: read from the cache", 3300, "cached"),
                                             ("Q2", 60, "full")], "same beginning: cheap and faster"),
            ("request 3", [("\"Today is 7 Nov.\" + the same notes", 3300, "miss"), ("Q3", 60, "full")],
             "first token differs: no cache hit at all")]
    style = {"full": (TINT_RED, ARMRED), "cached": (TINT_GREEN, GREEN), "miss": (TINT_RED, ARMRED)}
    for r, (name, parts, note) in enumerate(rows):
        y = 1.32 - r * 0.50
        ax.text(0.04, y + 0.11, name, fontsize=FS_SMALL, va="center", linespacing=1.05)
        x = x0
        for label, n, kind in parts:
            w = max(n * scale, 0.22)
            fc, ec = style[kind]
            rbox(ax, x, y, w - 0.02, 0.22, fc, ec, z=2)
            ax.text(x + (w - 0.02) / 2, y + 0.11, label, ha="center", va="center", fontsize=FS_TINY, zorder=3)
            x += w
        ax.text(x0, y - 0.09, note, fontsize=FS_TINY, color=GREEN if r == 1 else GREY, va="center")
    ax.text(5.45, 0.04, "schematic", fontsize=FS_TINY, color=GREY, ha="right", va="bottom")
    save(fig, "dec_prompt_cache.pdf", log)


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
    # "Why it works for any two models": min(p_s, p_b) + max(0, p_b - p_s) = p_b for random pairs
    rng = np.random.default_rng(509)
    for _ in range(1000):
        ps, pb = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        kept = ps * np.minimum(1, pb / ps)
        rej = 1 - kept.sum()
        short = np.maximum(0, pb - ps)
        assert np.isclose(rej, short.sum()) and np.allclose(kept + rej * short / short.sum(), pb)

    B = QUOTED["beam_logs"]
    assert round(math.log(QUOTED["best_path"]), 2) == B["best"], math.log(QUOTED["best_path"])
    assert round(math.log(QUOTED["greedy_path"]), 2) == B["second"], math.log(QUOTED["greedy_path"])

    C = QUOTED["chain_toy"]
    sq = np.array(C["p"]) ** (1 / C["T"])
    assert np.isclose(sum(C["p"]), 1) and round(sq.sum(), 4) == C["squares_sum"], sq.sum()
    cols = {"after_T": sq / sq.sum()}
    kk = sq[:C["k"]]
    cols["after_k"] = kk / kk.sum()
    run = np.cumsum(cols["after_k"])
    n_p = int(np.searchsorted(run, C["top_p"] - 1e-12)) + 1
    pp = sq[:n_p]
    cols["after_p"] = pp / pp.sum()
    bar = C["min_p"] * cols["after_p"].max()
    mm = pp[cols["after_p"] >= bar]
    cols["after_minp"] = mm / mm.sum()
    for name, v in cols.items():                          # every column, and its rounded sum
        assert [round(float(x), 2) for x in v] == C[name], (name, v)
        assert round(sum(C[name]), 2) == 1.0, (name, C[name])
    assert [round(float(x), 2) for x in run[:n_p]] == C["running"] and round(bar, 3) == C["bar"], (run, bar)
    assert len(C["after_k"]) < len(C["p"]) and n_p < C["k"] and len(mm) < n_p   # each knob cuts

    R["review_toys"] = {"speculative": {"accepted": accepted.tolist(), "reject": reject,
                                        "residual": residual.tolist(), "final": final.tolist()},
                        "tail_share_by_T": tail_share, "order": {"after_T": after_T.tolist()},
                        "chain": {k: v.tolist() for k, v in cols.items()} | {"bar": bar}}
    log.info(f"review toys: {R['review_toys']}")


def toy_checks(R, log):
    small = np.float32(np.finfo(np.float32).smallest_subnormal)
    prod = np.float32(1.0)
    for _ in range(100):
        prod = np.float32(prod * np.float32(0.1))
    a, b = np.float32(1e8), np.float32(1.0)
    left, right = (a + b) - a, (a - a) + b
    assert prod == 0 and small < 1.5e-45 and left == 0 and right == 1, (prod, small, left, right)
    assert round(10 * math.log10(50257)) == 47              # "50,257^10 ~ 10^47" on "Beam search: the idea"
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
    jobs = [fig_greedy_loop, fig_map_top5, fig_tree, fig_beam_steps, fig_logprob, fig_penalty,
            fig_long_tail, fig_temperature, fig_topk_topp, fig_knob_pipeline, fig_json_mask, fig_grammar_table,
            fig_speculative, fig_spec_verify, fig_spec_timeline, fig_spec_heads, fig_prompt_cache, toy_checks,
            review_toys]
    for k, job in enumerate(jobs, 1):
        job(R, log)
        el = time.time() - t0
        log.info(f"{k}/{len(jobs)}, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left")
    (RESULTS / "decoding_figs.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
