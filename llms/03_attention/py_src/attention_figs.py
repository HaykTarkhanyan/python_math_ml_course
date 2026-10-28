"""Figures for LLM-3 "Attention: how tokens share meaning" (llms/03_attention/LLM3_attention.tex).

Toy, demonstrative numbers (ml/SLIDE_STYLE.md, 2026-10-05). The by-hand frames use three toy
vectors read as [adjective, fluffy, blue, noun]; every number the slides quote is asserted
(QUOTED), so a changed toy fails loudly. The only real numbers are GPT-2's sizes and the GPT-2
tokens of the instructor's July sentence; they are constants here, checked against the local
caches with --verify (HF cache config.json + safetensors header, tiktoken's gpt2 encoding; no
download, no model run).

Drawn at the size each figure gets on the slide (5.5 in = full text width), 6.5-8 pt text, and
save() refuses any text outside its figure.

Generates into llms/03_attention/fig/:
  att_raw_grid.pdf      -- raw vectors as their own query/key/value: scores and weights, creature's row
  att_learned_grid.pdf  -- one-number queries and keys: score = q x k, weights, creature's row
  att_shapes.pdf        -- the matrix shapes on "Donkey don't die, spring is coming" (9 tokens)
  att_dot_product.pdf   -- q . k for four keys: aligned, 45 degrees, perpendicular, opposite
  att_sqrt_dk.pdf       -- softmax of raw scores (d_k = 64), divided by sqrt(64) = 8, and by 64 itself
  att_softmax_why.pdf   -- scores, divided by their sum (a negative weight), softmax (all positive)
  Added 2026-10-28 (rework: show, not tell):
  att_three_moves.pdf   -- wall 1: score bars -> softmax weights -> the context as a weighted blend
  att_toy_chips.pdf     -- the three toy vectors as 4-cell chips [adjective, fluffy, blue, noun]
  att_soft_lookup.pdf   -- a Python dict (KeyError) vs a soft lookup that blends every value
  att_qk_axis.pdf       -- w_Q reads the noun cell, w_K the adjective cell: each word's query and key
  att_argmax_softmax.pdf -- weight on one key as its score slides: argmax steps, softmax has a slope
  att_dk_spread.pdf     -- q . k for d_k = 4, 16, 64 (spread 2, 4, 8), and after dividing by sqrt(d_k)
  att_saturation.pdf    -- softmax of (1,0,0) x 1, 2, 4, 8: from soft to one-hot
  att_value_chips.pdf   -- W_V, the weighted sum of value chips, the residual add, as chip arithmetic
                           (replaces att_value_update.pdf, the before/after table)
  att_gpt2_cattn.pdf    -- GPT-2's 768 x 2,304 c_attn: Q | K | V, each 12 heads of 64
  The matrix toy (d_k = 2, real matrix products; the slide's 10-line code is run on it and its
  printout asserted):
  att_mat_qk.pdf        -- q = W_Q e for creature, k = W_K e for fluffy, each row's dot product written out
  att_mat_scores.pdf    -- 2-number queries and keys: score grid q.k / sqrt(2) and the weights
  att_mat_pipeline.pdf  -- X -> Q, K, V -> QK^T / sqrt(2) -> softmax -> A V, creature's row in red

Run (~8 s):
  ./ma/Scripts/python.exe llms/03_attention/py_src/attention_figs.py            # the figures
  ./ma/Scripts/python.exe llms/03_attention/py_src/attention_figs.py --verify   # also check GPT2 against the caches
"""
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
from matplotlib.patches import FancyArrowPatch, Rectangle

warnings.filterwarnings("error", message=".*missing from.*font.*")

SEED = 509
HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"

# ----------------------------------------------------------------------------- the toy
TOKENS = ["fluffy", "blue", "creature"]
X = np.array([[1, 1, 0, 0],      # fluffy
              [1, 0, 1, 0],      # blue
              [0, 0, 0, 2]],     # creature
             dtype=float)
W_Q1 = np.array([0, 0, 0, 1.0])  # one-number query: reads "noun"
W_K1 = np.array([1.0, 0, 0, 0])  # one-number key: reads "adjective"
W_V = np.diag([0, 1.0, 1.0, 1.0])  # hands over fluffy, blue, noun; drops the "adjective" tag

QUOTED = {
    "raw_scores_creature": [0, 0, 2], "raw_weights_creature": [0.11, 0.11, 0.79],
    "raw_delta": [0.21, 0.11, 0.11, 1.57], "raw_new": [0.21, 0.11, 0.11, 3.57],
    "q": [0, 0, 2], "k": [1, 1, 0], "learned_scores_creature": [2, 2, 0],
    "learned_weights_creature": [0.47, 0.47, 0.06], "learned_weights_fluffy": [0.33, 0.33, 0.33],
    "learned_delta": [0, 0.47, 0.47, 0.13], "learned_new": [0, 0.47, 0.47, 2.13],
    "raw_weights3": [0.107, 0.107, 0.787], "learned_weights3": [0.468, 0.468, 0.063],
    "softmax_220": {"e2": 7.39, "denom": 15.78},     # (e^2, e^2, e^0) / (2 e^2 + 1), shown by hand
    "dot": {"aligned": 5.0, "45": 3.6, "perpendicular": 0.0, "opposite": -4.6},
    "sqrt_dk": 8,
    # why divide by sqrt(d_k): a losing score's weight before and after raising it by 1
    "saturation": {"big": [0.0003, 0.0009], "small": [0.21, 0.42]},   # scores (8,0,0) / (1,0,0)
    # why softmax and not "divide by the sum": att_softmax_why.pdf
    "softmax_why": {"scores": [3, 2, -1], "by_sum": [0.75, 0.5, -0.25], "softmax": [0.72, 0.27, 0.01]},
    # added 2026-10-28
    "three_moves": {"scores": [0.5, 2.6, 0.5], "weights": [0.1, 0.8, 0.1]},     # "The fix" diagram's .1/.8/.1
    "soft_lookup": {"scores": [2.6, 1.6, 0.0], "weights": [0.69, 0.26, 0.05]},   # kitten vs cat, dog, car
    "saturation_steps": {"scale": [1, 2, 4, 8], "top": [0.58, 0.79, 0.96, 1.0], "loser_8": 0.0003},
    "dk_spread": {"dk": [4, 16, 64], "spread": [2, 4, 8]},                       # measured, to +-0.3
    # student review 2026-10-28: the raw softmax shown by hand, softmax's slope w(1 - w) for the
    # losing weight, and the d_k = 64 divisor drawn (att_sqrt_dk.pdf third panel: nearly flat)
    "softmax_002": {"e2": 7.389, "denom": 9.389},     # (e^0, e^0, e^2) / (2 + e^2), the raw toy
    "slope": {"big": 0.0003, "small": 0.17},          # w(1 - w) of the loser at (8,0,0) / (1,0,0)
    "by_dk_range": 0.15,                              # max - min weight after dividing by 64
    # the matrix toy (instructor 2026-10-28, after offers on 10-05, 10-20, 10-27): d_k = 2, real
    # matrix products on the same three vectors; W_V as in the one-number toy
    "mat": {"q": {"fluffy": [0.5, 0.5], "blue": [0.5, 0.5], "creature": [2, 1]},
            "k": {"fluffy": [1, 0.5], "blue": [1, 0.5], "creature": [0, -1]},
            "scores_creature": [2.5, 2.5, -1], "scaled_creature": [1.77, 1.77, -0.71],
            "weights_creature": [0.48, 0.48, 0.04], "weights_fluffy": [0.41, 0.41, 0.17],
            "out_creature": [0, 0.48, 0.48, 0.08], "out_fluffy": [0, 0.41, 0.41, 0.34],
            "printout": "[[0.   0.41 0.41 0.34]\n [0.   0.41 0.41 0.34]\n [0.   0.48 0.48 0.08]]"},
}
W_Q2 = np.array([[0.5, 0, 0, 1.0],      # the matrix toy's W_Q (d_k x d = 2 x 4): rows are questions
                 [0, 0.5, 0.5, 0.5]])
W_K2 = np.array([[1.0, 0, 0, 0],
                 [0, 0.5, 0.5, -0.5]])
# GPT-2 small (checked with --verify against config.json, the safetensors header and tiktoken, 2026-10-05)
GPT2 = {"d": 768, "heads": 12, "d_k": 64, "layers": 12, "n_ctx": 1024, "c_attn": [768, 2304],
        "c_proj": [768, 768],
        "sentence": "Donkey don't die, spring is coming",
        "tokens": ["Don", "key", " don", "'t", " die", ",", " spring", " is", " coming"]}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"
TINT_RED = "#F9D2D5"
GREEN_D = "#1E7B45"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("attention_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "attention_figs.log", encoding="utf-8")):
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


def arrow(ax, p0, p1, color=GREY, lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


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


def softmax_rows(S):
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def r2(v):
    return [round(float(x), 2) for x in np.ravel(v)]


def check(name, got, want):
    if r2(got) != r2(want):
        raise AssertionError(f"{name}: computed {r2(got)}, slides quote {want}")


def num(v):
    """0.50 -> '.5', 2.0 -> '2', 0.11 -> '.11': short labels for small cells."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    s = s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)
    return s or "0"


def grid(ax, x0, y_top, cell_w, cell_h, M, vmax, row_labels=None, col_labels=None, hl_row=None,
         title=None, fmt=num, cmap="Blues", col_rot=0):
    """A matrix as coloured cells with its numbers; row 0 at the top. Returns (x_right, y_bottom)."""
    cm = plt.get_cmap(cmap)
    n_r, n_c = M.shape
    for i in range(n_r):
        for j in range(n_c):
            v = M[i, j]
            x, y = x0 + j * cell_w, y_top - (i + 1) * cell_h
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, facecolor=cm(0.08 + 0.5 * max(v, 0) / vmax),
                                   edgecolor="white", linewidth=0.8))
            if fmt is not None:
                ax.text(x + cell_w / 2, y + cell_h / 2, fmt(v), ha="center", va="center",
                        fontsize=FS_TINY, color=INK)
    if hl_row is not None:
        ax.add_patch(Rectangle((x0, y_top - (hl_row + 1) * cell_h), n_c * cell_w, cell_h,
                               fill=False, edgecolor=ARMRED, linewidth=1.3, zorder=5))
    if row_labels:
        for i, lab in enumerate(row_labels):
            ax.text(x0 - 0.05, y_top - (i + 0.5) * cell_h, lab, ha="right", va="center",
                    fontsize=FS_TINY, color=ARMRED if i == hl_row else INK)
    if col_labels:
        for j, lab in enumerate(col_labels):
            ax.text(x0 + (j + 0.5) * cell_w, y_top + 0.04, lab, ha="left" if col_rot else "center",
                    va="bottom", fontsize=FS_TINY, rotation=col_rot, rotation_mode="anchor")
    if title:
        ax.text(x0 + n_c * cell_w / 2, y_top + (0.42 if col_rot else 0.17), title, ha="center",
                va="bottom", fontsize=FS_SMALL, color=INK)
    return x0 + n_c * cell_w, y_top - n_r * cell_h


SLOTS = ["adj.", "fluffy", "blue", "noun"]
SLOT_COLORS = ["#8C8C8C", ARMORANGE, ARMBLUE, "#2E8B57"]


def chip(ax, x, y, v, cw=0.26, ch=0.22, vmax=2.0, label=None, label_color=INK, numbers=True):
    """A toy vector as 4 cells [adjective, fluffy, blue, noun], each in its slot's colour, darker the
    bigger the number (alpha capped at 0.6 so the numbers stay readable). (x, y) is the bottom-left
    corner; returns the right edge."""
    for j, val in enumerate(v):
        a = 0.06 + 0.54 * np.sqrt(min(abs(float(val)) / vmax, 1.0))
        ax.add_patch(Rectangle((x + j * cw, y), cw, ch, facecolor=SLOT_COLORS[j], alpha=a, lw=0))
        ax.add_patch(Rectangle((x + j * cw, y), cw, ch, fill=False, edgecolor="white", lw=0.8))
        if numbers:
            ax.text(x + (j + 0.5) * cw, y + ch / 2, num(val), ha="center", va="center", fontsize=FS_TINY,
                    color=INK)
    ax.add_patch(Rectangle((x, y), 4 * cw, ch, fill=False, edgecolor="#999999", lw=0.5))
    if label:
        ax.text(x - 0.05, y + ch / 2, label, ha="right", va="center", fontsize=FS_TINY, color=label_color)
    return x + 4 * cw


def slot_header(ax, x, y, cw):
    for j, (s, c) in enumerate(zip(SLOTS, SLOT_COLORS)):
        ax.text(x + (j + 0.5) * cw, y, s, ha="center", va="bottom", fontsize=FS_TINY, color=c)


# ----------------------------------------------------------------------------- the toy, computed
def toy(R, log):
    S_raw = X @ X.T / np.sqrt(X.shape[1])
    A_raw = softmax_rows(S_raw)
    d_raw = A_raw[2] @ X                     # raw: the vectors are their own values
    q, k = X @ W_Q1, X @ W_K1
    S_l = np.outer(q, k) / np.sqrt(1)        # a query/key space of one number
    A_l = softmax_rows(S_l)
    d_l = A_l[2] @ (X @ W_V.T)
    check("raw_scores_creature", S_raw[2], QUOTED["raw_scores_creature"])
    check("raw_weights_creature", A_raw[2], QUOTED["raw_weights_creature"])
    check("raw_delta", d_raw, QUOTED["raw_delta"])
    check("raw_new", X[2] + d_raw, QUOTED["raw_new"])
    check("q", q, QUOTED["q"])
    check("k", k, QUOTED["k"])
    check("learned_scores_creature", S_l[2], QUOTED["learned_scores_creature"])
    check("learned_weights_creature", A_l[2], QUOTED["learned_weights_creature"])
    check("learned_weights_fluffy", A_l[0], QUOTED["learned_weights_fluffy"])
    check("learned_delta", d_l, QUOTED["learned_delta"])
    check("learned_new", X[2] + d_l, QUOTED["learned_new"])
    # The by-hand lines print 3-decimal weights; their arithmetic must give the printed 2-decimal
    # results (with the 2-decimal weights it would not: 0.11 + 0.11 = 0.22, the slide says 0.21).
    for name, A_row, vals, want in (("raw", A_raw[2], X, "raw_delta"),
                                    ("learned", A_l[2], X @ W_V.T, "learned_delta")):
        w3 = np.round(A_row, 3)
        if list(w3) != QUOTED[f"{name}_weights3"]:
            raise AssertionError(f"{name}: 3-decimal weights {list(w3)} != {QUOTED[name + '_weights3']}")
        check(f"{name}: printed arithmetic", w3 @ vals, QUOTED[want])
    sm = QUOTED["softmax_220"]                 # the printed exponentials give the printed weights
    if round(np.e ** 2, 2) != sm["e2"] or round(2 * np.e ** 2 + 1, 2) != sm["denom"]:
        raise AssertionError(f"e^2 = {np.e ** 2:.3f}, 2e^2 + 1 = {2 * np.e ** 2 + 1:.3f}, slides quote {sm}")
    s0 = QUOTED["softmax_002"]                 # the raw toy by hand: (1, 1, e^2) / (2 + e^2)
    if round(2 + np.e ** 2, 3) != s0["denom"] or [round(1 / s0["denom"], 3), round(s0["e2"] / s0["denom"], 3)] \
            != [QUOTED["raw_weights3"][0], QUOTED["raw_weights3"][2]]:
        raise AssertionError(f"2 + e^2 = {2 + np.e ** 2:.3f}; slides quote {s0} -> {QUOTED['raw_weights3']}")
    check("softmax by hand", [sm["e2"] / sm["denom"], sm["e2"] / sm["denom"], 1 / sm["denom"]],
          QUOTED["learned_weights_creature"])
    if not np.allclose(S_raw, S_raw.T):
        raise AssertionError("the raw score grid should be symmetric")
    if np.isclose(S_l[2, 0], S_l[0, 2]):
        raise AssertionError("the learned score grid should not be symmetric")
    R["toy"] = {"S_raw": S_raw.tolist(), "A_raw": A_raw.tolist(), "S_learned": S_l.tolist(),
                "A_learned": A_l.tolist(), "delta_raw": d_raw.tolist(), "delta_learned": d_l.tolist()}
    log.info(f"toy: raw creature weights {r2(A_raw[2])}, learned {r2(A_l[2])}")
    return S_raw, A_raw, S_l, A_l, d_raw, d_l, q, k


# ----------------------------------------------------------------------------- figures
def fig_raw_grid(T, log):
    S_raw, A_raw = T[0], T[1]
    fig, ax = canvas(1.75, 2.75)
    cw, ch, top = 0.30, 0.28, 1.1
    x0 = 0.58
    xr, yb = grid(ax, x0, top, cw, ch, S_raw, 2.0, TOKENS, TOKENS, hl_row=2, col_rot=50)
    ax.text(x0 + 1.5 * cw, yb - 0.06, "score = q$\\cdot$k / $\\sqrt{4}$", ha="center", va="top",
            fontsize=FS_SMALL)
    grid(ax, xr + 0.12, top, cw, ch, A_raw, 1.0, None, TOKENS, hl_row=2, col_rot=50)
    ax.text(xr + 0.12 + 1.5 * cw, yb - 0.06, "weights", ha="center", va="top", fontsize=FS_SMALL)
    save(fig, "att_raw_grid.pdf", log)


def fig_learned_grid(T, log):
    S_l, A_l, q, k = T[2], T[3], T[6], T[7]
    fig, ax = canvas(1.75, 2.75)
    cw, ch, top = 0.27, 0.28, 1.1
    x0 = 0.9
    xr, yb = grid(ax, x0, top, cw, ch, S_l, 2.0, None, None, hl_row=2, title=None)
    for i, (lab, qi) in enumerate(zip(TOKENS, q)):          # rows: token and its query
        y = top - (i + 0.5) * ch
        ax.text(x0 - 0.36, y, lab, ha="right", va="center", fontsize=FS_TINY,
                color=ARMRED if i == 2 else INK)
        ax.text(x0 - 0.05, y, f"q={num(qi)}", ha="right", va="center", fontsize=FS_TINY, color=ARMBLUE)
    for j, (lab, kj) in enumerate(zip(TOKENS, k)):          # columns: key, then the token
        x = x0 + (j + 0.5) * cw
        ax.text(x, top + 0.04, f"k={num(kj)}", ha="center", va="bottom", fontsize=FS_TINY, color=ARMORANGE)
        ax.text(x - 0.06, top + 0.2, lab, ha="left", va="bottom", fontsize=FS_TINY, rotation=50,
                rotation_mode="anchor")
    ax.text(x0 + 1.5 * cw, yb - 0.06, "score = q $\\times$ k", ha="center", va="top", fontsize=FS_SMALL)
    grid(ax, xr + 0.12, top, cw, ch, A_l, 1.0, None, None, hl_row=2, title=None)
    for j, lab in enumerate(TOKENS):
        ax.text(xr + 0.12 + (j + 0.5) * cw - 0.06, top + 0.2, lab, ha="left", va="bottom",
                fontsize=FS_TINY, rotation=50, rotation_mode="anchor")
    ax.text(xr + 0.12 + 1.5 * cw, yb - 0.06, "weights", ha="center", va="top", fontsize=FS_SMALL)
    save(fig, "att_learned_grid.pdf", log)


def fig_shapes(log):
    toks = [t.replace(" ", "␣") for t in GPT2["tokens"]]
    n = len(toks)
    fig, ax = canvas(2.15)

    def mat(x, y_top, w, h, label, dims, fc):
        ax.add_patch(Rectangle((x, y_top - h), w, h, facecolor=fc, edgecolor=INK, linewidth=0.6))
        ax.text(x + w / 2, y_top - h / 2, label, ha="center", va="center", fontsize=FS_SMALL)
        ax.text(x + w / 2, y_top - h - 0.04, dims, ha="center", va="top", fontsize=FS_TINY, color=GREY)

    rows, big, small = 0.3, 0.95, 0.28            # 9 rows, 768, 64 - not to scale
    # row 1: X W_Q = Q
    y1 = 2.0
    mat(0.05, y1, big, rows, "X", f"{n} x {GPT2['d']}", "#E8E8E8")
    ax.text(1.09, y1 - rows / 2, "$\\times$", ha="center", va="center", fontsize=FS)
    mat(1.2, y1, small, big, "$W_Q$", f"{GPT2['d']} x {GPT2['d_k']}", "#CFD8EE")
    ax.text(1.6, y1 - rows / 2, "=", ha="center", va="center", fontsize=FS)
    mat(1.72, y1, small, rows, "Q", f"{n} x {GPT2['d_k']}", "#CFD8EE")
    ax.text(2.12, y1 - rows / 2, "and the same with\n$W_K \\to$ K,  $W_V \\to$ V",
            ha="left", va="center", fontsize=FS_TINY, color=GREY)
    # row 2: Q K^T = scores
    y2 = 0.68
    mat(0.05, y2, small, rows, "Q", f"{n} x {GPT2['d_k']}", "#CFD8EE")
    ax.text(0.45, y2 - rows / 2, "$\\times$", ha="center", va="center", fontsize=FS)
    mat(0.56, y2, rows, small, "$K^\\top$", f"{GPT2['d_k']} x {n}", "#FCE9B8")
    ax.text(0.98, y2 - rows / 2, "=", ha="center", va="center", fontsize=FS)
    mat(1.1, y2, rows, rows, "", f"{n} x {n}", "#F9D2D5")
    arrow(ax, (1.45, y2 - rows / 2), (3.15, y2 - rows / 2 + 0.05), color=ARMRED, lw=0.7)
    ax.text(2.3, y2 - 0.02, "one score per pair", ha="center", va="bottom", fontsize=FS_TINY, color=ARMRED)
    # the 9 x 9 grid with the real tokens
    cell, gx, gtop = 0.13, 3.9, 1.47
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((gx + j * cell, gtop - (i + 1) * cell), cell, cell,
                                   facecolor="#F9D2D5" if i == 4 else "#FBEFF0",
                                   edgecolor="white", linewidth=0.6))
    ax.add_patch(Rectangle((gx, gtop - 5 * cell), n * cell, cell, fill=False, edgecolor=ARMRED,
                           linewidth=1.0, zorder=5))
    for i, t in enumerate(toks):
        ax.text(gx - 0.04, gtop - (i + 0.5) * cell, t, ha="right", va="center", fontsize=FS_TINY,
                color=ARMRED if i == 4 else INK)
        ax.text(gx + (i + 0.5) * cell - 0.03, gtop + 0.03, t, ha="left", va="bottom", fontsize=FS_TINY,
                rotation=60, rotation_mode="anchor")
    ax.text(gx + n * cell / 2, gtop - n * cell - 0.05, "row = who asks, column = who answers;\n"
            "the red row: what ␣die looks at", ha="center", va="top", fontsize=FS_TINY, color=GREY)
    ax.text(0.05, 0.03, f"softmax each row, then multiply by V ({n} x {GPT2['d_k']}):\n"
            f"the output is {n} x {GPT2['d_k']}.  Boxes not to scale.", fontsize=FS_TINY, color=GREY)
    save(fig, "att_shapes.pdf", log)


def fig_dot_product(R, log):
    q = np.array([2.4, 0.0])
    keys = [("aligned", np.array([2.1, 0.35]), ARMBLUE, "aligned"),
            ("45°", np.array([1.5, 1.5]), ARMORANGE, "45"),
            ("perpendicular", np.array([0.0, 2.1]), GREY, "perpendicular"),
            ("opposite", np.array([-1.9, 0.5]), ARMRED, "opposite")]
    fig = plt.figure(figsize=(2.65, 1.75))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axhline(0, color="#E5E5E5", lw=0.6)
    ax.axvline(0, color="#E5E5E5", lw=0.6)
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK))
    ax.text(q[0] - 0.05, -0.12, "query q", ha="right", va="top", fontsize=FS_SMALL, fontweight="bold")
    got = {}
    for name, kv, col, key in keys:
        ax.annotate("", xy=kv, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=col))
        d = float(q @ kv)
        got[key] = round(d, 1)
        ha = "left" if kv[0] >= 0 else "center"
        ax.text(kv[0] + (0.08 if kv[0] >= 0 else 0), kv[1] + 0.06, f"{name}\nq$\\cdot$k = {d:.1f}",
                ha=ha, va="bottom", fontsize=FS_TINY, color=col)
    if got != QUOTED["dot"]:
        raise AssertionError(f"dot products {got} != quoted {QUOTED['dot']}")
    ax.set_xlim(-3.0, 3.6)
    ax.set_ylim(-0.45, 3.05)
    ax.set_aspect("equal")
    ax.axis("off")
    R["dot"] = got
    save(fig, "att_dot_product.pdf", log)


def fig_sqrt_dk(R, log):
    rng = np.random.default_rng(SEED)
    dk = 64
    pairs = rng.standard_normal((10_000, 2, dk))           # how big is q . k for random entries?
    spread = float((pairs[:, 0] * pairs[:, 1]).sum(1).std())
    if abs(spread - np.sqrt(dk)) > 0.3 or np.sqrt(dk) != QUOTED["sqrt_dk"]:
        raise AssertionError(f"spread of q.k over {dk} numbers: {spread:.2f}, expected about 8")
    q = rng.standard_normal(dk)
    K = rng.standard_normal((5, dk))
    raw = K @ q
    w_raw = softmax_rows(raw[None])[0]
    w_s = softmax_rows(raw[None] / np.sqrt(dk))[0]
    w_d = softmax_rows(raw[None] / dk)[0]                  # divided by d_k itself: nearly flat
    if w_d.max() - w_d.min() > QUOTED["by_dk_range"]:
        raise AssertionError(f"dividing by {dk} should flatten the weights: {r2(w_d)}")
    R["sqrt_dk"] = {"spread_10k_pairs": round(spread, 2), "raw_scores": r2(raw),
                    "w_raw": r2(w_raw), "w_scaled": r2(w_s), "w_by_dk": r2(w_d)}
    log.info(f"sqrt_dk: measured spread {spread:.2f} over 10,000 random pairs; raw {r2(raw)}; "
             f"weights raw {r2(w_raw)} scaled {r2(w_s)} by d_k {r2(w_d)}")
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.5), sharey=True)    # full width since 2026-10-28
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.14, wspace=0.12)
    labels = [f"k{i + 1}" for i in range(5)]
    for ax, w, col, title in ((axes[0], w_raw, ARMRED, "softmax(q$\\cdot$k): one key takes all"),
                              (axes[1], w_s, ARMBLUE, "softmax(q$\\cdot$k / 8): all keep a share"),
                              (axes[2], w_d, GREY, "softmax(q$\\cdot$k / 64): nearly flat")):
        bars = ax.bar(labels, w, color=col, width=0.7)
        ax.bar_label(bars, labels=[num(v) for v in w], fontsize=FS_TINY, padding=1)
        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontsize=FS_SMALL, pad=3)
        ax.tick_params(axis="x", labelsize=FS_TINY, length=0)
        ax.tick_params(axis="y", labelsize=FS_TINY)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("weight", fontsize=FS_TINY)
    save(fig, "att_sqrt_dk.pdf", log)


def saturation(R, log):
    """The sqrt(d_k) frame in numbers: raise one losing score by 1 and see how far its weight moves,
    at a gap of 8 (unscaled, d_k = 64) and at a gap of 1 (scaled)."""
    got = {}
    for name, top in (("big", 8.0), ("small", 1.0)):
        before = softmax_rows(np.array([[top, 0.0, 0.0]]))[0, 1]
        after = softmax_rows(np.array([[top, 1.0, 0.0]]))[0, 1]
        got[name] = [before, after]
    shown = {"big": [round(float(v), 4) for v in got["big"]], "small": r2(got["small"])}
    if shown != QUOTED["saturation"]:
        raise AssertionError(f"saturation: computed {shown}, slides quote {QUOTED['saturation']}")
    slope = {k: float(v[0] * (1 - v[0])) for k, v in got.items()}       # d w / d s for the loser
    if round(slope["big"], 4) != QUOTED["slope"]["big"] or round(slope["small"], 2) != QUOTED["slope"]["small"]:
        raise AssertionError(f"slopes w(1 - w): {slope}, slides quote {QUOTED['slope']}")
    R["saturation"] = {k: [float(v) for v in vs] for k, vs in got.items()}
    log.info(f"saturation: losing weight before/after a +1 nudge {R['saturation']}")


def fig_softmax_why(R, log):
    """Three bar panels: raw scores, divided by their sum (a negative weight), softmax (all > 0)."""
    Q = QUOTED["softmax_why"]
    s = np.array(Q["scores"], dtype=float)
    by_sum = s / s.sum()
    sm = softmax_rows(s[None])[0]
    check("divide by the sum", by_sum, Q["by_sum"])
    check("softmax", sm, Q["softmax"])
    if not np.isclose(by_sum.sum(), 1) or by_sum.min() >= 0 or sm.min() <= 0:
        raise AssertionError("the figure's point: both sum to 1, only softmax stays positive")
    R["softmax_why"] = {"scores": s.tolist(), "by_sum": by_sum.tolist(), "softmax": sm.tolist()}
    fig, axes = plt.subplots(1, 3, figsize=(2.65, 1.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.2, wspace=0.18)
    panels = ((s, "scores", None), (by_sum, "$\\div$ their sum", "a negative weight"),
              (sm, "softmax", "all > 0, sum 1"))
    for ax, (v, title, note) in zip(axes, panels):
        cols = [ARMRED if x < 0 else ARMBLUE for x in v]
        bars = ax.bar(["k1", "k2", "k3"], v, color=cols, width=0.7)
        ax.bar_label(bars, labels=[num(x).replace("-", "\u2212") for x in v], fontsize=FS_TINY, padding=1)
        ax.set_ylim(-0.5 * v.max(), 1.28 * v.max())     # zero line at the same height in all three
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_title(title, fontsize=FS_SMALL, pad=3)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=FS_TINY, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        if note:
            ax.set_xlabel(note, fontsize=FS_TINY, color=ARMRED if v.min() < 0 else ARMBLUE, labelpad=2)
    save(fig, "att_softmax_why.pdf", log)


# ----------------------------------------------------------------------------- added 2026-10-28
def fig_three_moves(R, log):
    """Wall 1's three moves on "The fix" diagram's step: scores -> weights .1/.8/.1 -> a blend."""
    Q3 = QUOTED["three_moves"]
    s = np.array(Q3["scores"], dtype=float)
    w = softmax_rows(s[None])[0]
    check("three moves: weights", w, Q3["weights"])
    R["three_moves"] = {"scores": s.tolist(), "weights": w.tolist()}
    words, cols = ["The", "cat", "sat"], ["#9A9A9A", ARMBLUE, ARMORANGE]
    fig, ax = canvas(1.5)
    base = 0.3

    def head(xc, title, sub):
        ax.text(xc, 1.46, title, ha="center", va="top", fontsize=FS, fontweight="bold")
        ax.text(xc, 1.28, sub, ha="center", va="top", fontsize=FS_TINY, color=GREY)

    def bars(x0, vals, scale, fmt):
        for i, (v, wd, c) in enumerate(zip(vals, words, cols)):
            x = x0 + i * 0.38
            ax.add_patch(Rectangle((x, base), 0.28, v * scale, facecolor=c, lw=0))
            ax.text(x + 0.14, base + v * scale + 0.03, fmt(v), ha="center", va="bottom", fontsize=FS_TINY)
            ax.text(x + 0.14, base - 0.04, wd, ha="center", va="top", fontsize=FS_TINY)
        ax.plot([x0 - 0.06, x0 + 1.08], [base, base], color=GREY, lw=0.5)

    head(0.72, "1. score", "how relevant is each word?")
    bars(0.15, s, 0.27, lambda v: f"{v:.1f}")
    arrow(ax, (1.32, 0.62), (1.62, 0.62))
    head(2.35, "2. softmax", "positive weights that sum to 1")
    bars(1.85, w, 0.75, num)
    arrow(ax, (3.02, 0.62), (3.32, 0.62))
    head(4.4, "3. weighted sum", "this step's own context")
    x, bw = 3.55, 1.7                                       # the blend: one bar, cut by the weights
    for wi, wd, c in zip(w, words, cols):
        ax.add_patch(Rectangle((x, 0.5), bw * wi, 0.3, facecolor=c, edgecolor="white", lw=0.8))
        ax.text(x + bw * wi / 2, 0.44, f"{num(wi)} {wd}", ha="center", va="top", fontsize=FS_TINY,
                color=c if wd != "The" else GREY)
        x += bw * wi
    ax.text(4.4, 0.88, "context = .1 h(The) + .8 h(cat) + .1 h(sat)", ha="center", va="bottom",
            fontsize=FS_TINY)
    ax.text(4.4, 0.12, "mostly cat", ha="center", va="center", fontsize=FS_SMALL, color=ARMBLUE)
    save(fig, "att_three_moves.pdf", log)


def fig_toy_chips(log):
    fig, ax = canvas(1.05, 2.6)
    cw, ch, x0 = 0.4, 0.22, 0.85
    slot_header(ax, x0, 0.86, cw)
    for i, (t, v) in enumerate(zip(TOKENS, X)):
        chip(ax, x0, 0.58 - i * 0.28, v, cw=cw, ch=ch, label=t)
    save(fig, "att_toy_chips.pdf", log)


def fig_soft_lookup(R, log):
    Qs = QUOTED["soft_lookup"]
    s = np.array(Qs["scores"], dtype=float)
    w = softmax_rows(s[None])[0]
    check("soft lookup: weights", w, Qs["weights"])
    R["soft_lookup"] = {"scores": s.tolist(), "weights": w.tolist()}
    fig, ax = canvas(1.6)
    # left: the hard lookup
    ax.text(1.15, 1.55, "hard lookup: a Python dict", ha="center", va="top", fontsize=FS, fontweight="bold")
    mono = dict(fontsize=FS_TINY, family="DejaVu Sans Mono", va="center")
    ax.text(0.1, 1.15, 'd = {"cat": v1, "dog": v2, "car": v3}', **mono)
    ax.text(0.1, 0.82, 'd["cat"]', **mono)
    ax.text(1.0, 0.82, "-> v1", color=GREEN_D, **mono)
    ax.text(0.1, 0.55, 'd["kitten"]', **mono)
    ax.text(1.0, 0.55, "-> KeyError", color=ARMRED, **mono)
    ax.text(0.1, 0.2, "one key matches exactly,\nor nothing does", fontsize=FS_TINY, color=GREY, va="center")
    ax.plot([2.4, 2.4], [0.1, 1.5], color=LIGHT, lw=0.6)
    # right: the soft lookup
    ax.text(3.95, 1.55, "soft lookup: attention", ha="center", va="top", fontsize=FS, fontweight="bold")
    keys, cols = ["cat", "dog", "car"], [ARMBLUE, ARMORANGE, "#9A9A9A"]
    ax.add_patch(Rectangle((2.55, 0.68), 0.62, 0.26, facecolor="#EEF1F8", edgecolor=ARMBLUE, lw=0.7))
    ax.text(2.86, 0.81, "kitten", ha="center", va="center", fontsize=FS_SMALL)
    ax.text(2.86, 0.6, "query", ha="center", va="top", fontsize=FS_TINY, color=GREY)
    ax.text(3.45, 1.28, "key", ha="center", va="bottom", fontsize=FS_TINY, color=GREY)
    ax.text(3.88, 1.28, "score", ha="center", va="bottom", fontsize=FS_TINY, color=GREY)
    ax.text(4.25, 1.28, "weight (softmax)", ha="left", va="bottom", fontsize=FS_TINY, color=GREY)
    for i, (k, sc, wi, c) in enumerate(zip(keys, s, w, cols)):
        y = 1.12 - i * 0.24
        ax.plot([3.17, 3.3], [0.81, y], color=LIGHT, lw=0.6)
        ax.text(3.45, y, k, ha="center", va="center", fontsize=FS_SMALL)
        ax.text(3.88, y, f"{sc:.1f}", ha="center", va="center", fontsize=FS_TINY)
        ax.add_patch(Rectangle((4.25, y - 0.07), 0.95 * wi, 0.14, facecolor=c, lw=0))
        ax.text(4.25 + 0.95 * wi + 0.04, y, num(wi), ha="left", va="center", fontsize=FS_TINY)
    x, bw = 3.05, 1.45
    for wi, k, c in zip(w, keys, cols):
        ax.add_patch(Rectangle((x, 0.1), bw * wi, 0.2, facecolor=c, edgecolor="white", lw=0.8))
        x += bw * wi
    ax.text(3.0, 0.2, "result:", ha="right", va="center", fontsize=FS_TINY)
    ax.text(4.56, 0.2, "a blend of\nall three values", ha="left", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, "att_soft_lookup.pdf", log)


def fig_qk_axis(T, log):
    q, k = T[6], T[7]
    fig, ax = canvas(1.5, 2.75)
    cw, ch, x0 = 0.27, 0.22, 0.78
    slot_header(ax, x0, 1.23, cw)
    ax.text(2.12, 1.23, "query", ha="center", va="bottom", fontsize=FS_TINY, color=ARMBLUE)
    ax.text(2.55, 1.23, "key", ha="center", va="bottom", fontsize=FS_TINY, color=ARMORANGE)
    for i, (t, v) in enumerate(zip(TOKENS, X)):
        y = 0.96 - i * 0.28
        chip(ax, x0, y, v, cw=cw, ch=ch, label=t, label_color=ARMRED if i == 2 else INK)
        ax.add_patch(Rectangle((x0 + 3 * cw, y), cw, ch, fill=False, edgecolor=ARMBLUE, lw=1.3, zorder=5))
        ax.add_patch(Rectangle((x0, y), cw, ch, fill=False, edgecolor=ARMORANGE, lw=1.3, zorder=5))
        arrow(ax, (x0 + 4 * cw + 0.04, y + ch / 2), (1.98, y + ch / 2), color=LIGHT, lw=0.6)
        ax.text(2.12, y + ch / 2, num(q[i]), ha="center", va="center", fontsize=FS_SMALL, color=ARMBLUE,
                fontweight="bold" if q[i] else "normal")
        ax.text(2.55, y + ch / 2, num(k[i]), ha="center", va="center", fontsize=FS_SMALL, color=ARMORANGE,
                fontweight="bold" if k[i] else "normal")
    ax.text(0.05, 0.1, "blue box: $w_Q$ reads it.   orange box: $w_K$ reads it.", ha="left", va="center",
            fontsize=FS_TINY, color=GREY)
    save(fig, "att_qk_axis.pdf", log)


def fig_argmax_softmax(R, log):
    """Weight on key 2 as its score slides, the other two scores fixed at 1 and 0."""
    s = np.linspace(-3, 4, 401)
    soft = np.exp(s) / (np.exp(s) + np.e + 1)
    hard = (s > 1).astype(float)
    fig, ax = plt.subplots(figsize=(2.65, 1.6))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.25)
    ax.plot(s, hard, color=ARMRED, lw=1.4, drawstyle="steps-post", label="argmax: pick the top one")
    ax.plot(s, soft, color=ARMBLUE, lw=1.4, label="softmax")
    ax.text(-2.9, 0.22, "flat: a small nudge\nchanges nothing", fontsize=FS_TINY, color=ARMRED, va="bottom")
    ax.text(1.9, 0.3, "a slope\neverywhere:\nit can learn", fontsize=FS_TINY, color=ARMBLUE, va="center")
    ax.set_xlabel("score of key 2 (the others: 1 and 0)", fontsize=FS_TINY, labelpad=2)
    ax.set_ylabel("weight on key 2", fontsize=FS_TINY)
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=FS_TINY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_TINY, loc="upper left", handlelength=1.2, ncol=2,
              columnspacing=0.8)
    save(fig, "att_argmax_softmax.pdf", log)


def fig_dk_spread(R, log):
    """How big q . k gets for random entries of spread 1, for d_k = 64, 4, 16 (64 first, with the same
    draw as fig_sqrt_dk, so its spread is the 8.05 quoted there)."""
    rng = np.random.default_rng(SEED)
    dots = {}
    for dk in (64, 4, 16):
        p = rng.standard_normal((10_000, 2, dk))
        dots[dk] = (p[:, 0] * p[:, 1]).sum(1)
    spread = {dk: float(dots[dk].std()) for dk in dots}
    for dk, want in zip(QUOTED["dk_spread"]["dk"], QUOTED["dk_spread"]["spread"]):
        if abs(spread[dk] - want) > 0.3:
            raise AssertionError(f"spread of q.k at d_k={dk}: {spread[dk]:.2f}, slides say about {want}")
    R["dk_spread"] = {str(dk): round(v, 2) for dk, v in spread.items()}
    log.info(f"dk_spread: {R['dk_spread']}")
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.45))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2, wspace=0.08)
    cols = {4: ARMORANGE, 16: ARMBLUE, 64: ARMRED}
    for dk in (4, 16, 64):
        axes[0].hist(dots[dk], bins=np.linspace(-32, 32, 81), histtype="step", color=cols[dk], lw=1.1,
                     density=True, label=f"$d_k$ = {dk}: spread {spread[dk]:.1f}")
        axes[1].hist(dots[dk] / np.sqrt(dk), bins=np.linspace(-4.5, 4.5, 46), histtype="step",
                     color=cols[dk], lw=1.1, density=True)
    axes[0].set_title("q$\\cdot$k: more numbers, wider scores", fontsize=FS_SMALL, pad=3)
    axes[1].set_title("q$\\cdot$k / $\\sqrt{d_k}$: all three spread about 1", fontsize=FS_SMALL, pad=3)
    axes[0].legend(frameon=False, fontsize=FS_TINY, loc="upper left", handlelength=1.0)
    axes[0].set_ylim(0, 0.32)
    for ax in axes:
        ax.set_yticks([])
        ax.tick_params(labelsize=FS_TINY)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_xlabel("score", fontsize=FS_TINY, labelpad=1)
    axes[1].set_xlabel("score, scaled", fontsize=FS_TINY, labelpad=1)
    save(fig, "att_dk_spread.pdf", log)


def fig_saturation(R, log):
    Qs = QUOTED["saturation_steps"]
    fig, axes = plt.subplots(1, 4, figsize=(2.75, 1.5), sharey=True)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.12, wspace=0.12)
    tops = []
    for ax, c in zip(axes, Qs["scale"]):
        wv = softmax_rows(np.array([[c, 0.0, 0.0]]))[0]
        tops.append(wv[0])
        bars = ax.bar(["k1", "k2", "k3"], wv, color=[ARMBLUE, LIGHT, LIGHT], width=0.7)
        ax.bar_label(bars, labels=[num(wv[0]), "", ""], fontsize=FS_TINY, padding=1)
        ax.set_title(f"({c}, 0, 0)", fontsize=FS_TINY, pad=2)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=FS_TINY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if c == 8 and round(float(wv[1]), 4) != Qs["loser_8"]:
            raise AssertionError(f"loser weight at (8,0,0): {wv[1]:.5f}")
    check("saturation steps", tops, Qs["top"])
    R["saturation_steps"] = [float(t) for t in tops]
    save(fig, "att_saturation.pdf", log)


def fig_value_chips(T, log):
    """W_V drops the adjective tag; creature's weights times the value chips; the residual add."""
    A_l, d_l = T[3], T[5]
    V = X @ W_V.T
    w = np.round(A_l[2], 2)
    check("value chips: weights", w, QUOTED["learned_weights_creature"])
    new = X[2] + d_l
    fig, ax = canvas(1.8)
    xl = 1.12
    for j, (s, c) in enumerate(zip(["adjective", "fluffy", "blue", "noun"], SLOT_COLORS)):
        ax.add_patch(Rectangle((xl + j * 0.85, 1.65), 0.1, 0.1, facecolor=c, alpha=0.6, lw=0))
        ax.text(xl + j * 0.85 + 0.14, 1.7, s, ha="left", va="center", fontsize=FS_TINY, color=c)
    lab = dict(ha="left", va="center", fontsize=FS_TINY, fontweight="bold")
    # 1. W_V: word -> its value
    ax.text(0.05, 1.27, "1. $W_V$ makes\n    each value", **lab)
    x = xl
    for t, v_in, v_out in zip(TOKENS, X, V):
        ax.text(x + 0.62, 1.4, t, ha="center", va="bottom", fontsize=FS_TINY)
        r = chip(ax, x, 1.18, v_in, cw=0.14, ch=0.18)
        arrow(ax, (r + 0.03, 1.27), (r + 0.18, 1.27))
        chip(ax, r + 0.21, 1.18, v_out, cw=0.14, ch=0.18)
        x += 1.43
    ax.text(5.45, 1.03, "the adjective cell is dropped: found by it, but not handed over",
            ha="right", va="center", fontsize=FS_TINY, color=GREY)
    # 2. the weighted sum
    ax.text(0.05, 0.66, "2. weigh and\n    add them", **lab)
    x = xl
    for i, (wi, v) in enumerate(zip(w, V)):
        ax.text(x, 0.66, ("" if i == 0 else "+ ") + f"{num(wi)} $\\times$", ha="left", va="center",
                fontsize=FS_TINY)
        x = chip(ax, x + (0.36 if i == 0 else 0.48), 0.57, v, cw=0.14, ch=0.18) + 0.06
    ax.text(x, 0.66, "=", ha="left", va="center", fontsize=FS_SMALL)
    chip(ax, x + 0.13, 0.57, d_l, cw=0.22, ch=0.18)
    # 3. the residual add
    ax.text(0.05, 0.22, "3. add to\n    creature", **lab)
    r = chip(ax, xl, 0.13, X[2], cw=0.22, ch=0.18)
    ax.text(r + 0.06, 0.22, "+", ha="left", va="center", fontsize=FS_SMALL)
    r = chip(ax, r + 0.2, 0.13, d_l, cw=0.22, ch=0.18)
    ax.text(r + 0.06, 0.22, "=", ha="left", va="center", fontsize=FS_SMALL)
    r = chip(ax, r + 0.2, 0.13, new, cw=0.22, ch=0.18)
    ax.text(r + 0.1, 0.22, "a fluffy blue creature", ha="left", va="center", fontsize=FS_SMALL,
            color=GREEN_D, fontweight="bold")
    save(fig, "att_value_chips.pdf", log)


def fig_gpt2_cattn(log):
    g = GPT2
    fig, ax = canvas(1.45, 4.6)                              # drawn at its slot: 0.84 of the text width
    x0, y0, wd, ht = 0.4, 0.32, 2.85, 0.72
    third = wd / 3
    names = [("$W_Q$, all 12 heads", "#CFD8EE", ARMBLUE), ("$W_K$, all 12 heads", "#FCE9B8", ARMORANGE),
             ("$W_V$, all 12 heads", "#D5EBDD", "#2E8B57")]
    for t, (name, fc, ec) in enumerate(names):
        xt = x0 + t * third
        ax.add_patch(Rectangle((xt, y0), third, ht, facecolor=fc, edgecolor=INK, lw=0.6))
        for h in range(1, g["heads"]):
            ax.plot([xt + h * third / g["heads"]] * 2, [y0, y0 + ht], color="white", lw=0.6)
        ax.add_patch(Rectangle((xt, y0), third / g["heads"], ht, facecolor=ec, alpha=0.75, lw=0))
        ax.text(xt + third / 2, y0 + ht + 0.04, name, ha="center", va="bottom", fontsize=FS_TINY, color=ec)
        ax.text(xt + third / 2, y0 - 0.04, f"{g['heads']} heads $\\times$ {g['d_k']}", ha="center",
                va="top", fontsize=FS_TINY, color=GREY)
    ax.text(x0 + wd / 2, 1.43, f"GPT-2's c_attn, one per layer: {g['c_attn'][0]} $\\times$ {g['c_attn'][1]:,}",
            ha="center", va="top", fontsize=FS_SMALL, fontweight="bold")
    ax.text(x0 - 0.08, y0 + ht / 2, f"{g['d']} rows", rotation=90, ha="center", va="center", fontsize=FS_TINY)
    ax.text(x0 + third / g["heads"] / 2, 0.1, "dark strip: head 1, 768 x 64", ha="left", va="center",
            fontsize=FS_TINY, color=INK)
    # one head's score grid
    gx, gs = 3.6, 0.72
    ax.add_patch(Rectangle((gx, y0), gs, gs, facecolor=TINT_RED, edgecolor=ARMRED, lw=0.6))
    ax.text(gx + gs / 2, y0 + gs / 2, f"{g['n_ctx']:,}\n$\\times$\n{g['n_ctx']:,}", ha="center", va="center",
            fontsize=FS_TINY, color=ARMRED)
    ax.text(gx + gs / 2, y0 + ht + 0.04, "one head's score grid", ha="center", va="bottom", fontsize=FS_TINY,
            color=ARMRED)
    ax.text(gx + gs / 2, y0 - 0.04, f"{g['n_ctx'] ** 2:,} scores", ha="center", va="top", fontsize=FS_TINY,
            color=GREY)
    save(fig, "att_gpt2_cattn.pdf", log)


# ----------------------------------------------------------------------------- the matrix toy (2026-10-28)
def attention_slide(X_, Wq, Wk, Wv):
    """The deck's "Attention in about 10 lines", line for line, so its printout on the toy is checked."""
    Q, K, V = X_ @ Wq, X_ @ Wk, X_ @ Wv
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores -= scores.max(1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(1, keepdims=True)
    return weights @ V


def mat_toy(R, log):
    """d_k = 2: q = W_Q e as a real matrix product, then the whole line on X, and the slide's code."""
    M = QUOTED["mat"]
    Q, K, V = X @ W_Q2.T, X @ W_K2.T, X @ W_V.T
    for i, t in enumerate(TOKENS):
        check(f"matrix toy q {t}", Q[i], M["q"][t])
        check(f"matrix toy k {t}", K[i], M["k"][t])
    S = Q @ K.T
    Ss = S / np.sqrt(Q.shape[1])
    A = softmax_rows(Ss)
    O = A @ V
    check("matrix toy scores, creature", S[2], M["scores_creature"])
    check("matrix toy scaled, creature", Ss[2], M["scaled_creature"])
    check("matrix toy weights, creature", A[2], M["weights_creature"])
    check("matrix toy weights, fluffy", A[0], M["weights_fluffy"])
    check("matrix toy output, creature", O[2], M["out_creature"])
    check("matrix toy output, fluffy", O[0], M["out_fluffy"])
    W_V_slide = np.diag([0, 1, 1, 1])                      # built as the slide builds it (ints)
    out = attention_slide(X.astype(int), W_Q2.T, W_K2.T, W_V_slide.T)
    printed = str(np.round(out, 2))
    if printed != M["printout"]:
        raise AssertionError(f"the slide prints\n{M['printout']}\nbut the code prints\n{printed}")
    if not np.allclose(out, O):
        raise AssertionError("the 10-line function and the step-by-step matrix toy disagree")
    R["mat"] = {"Q": Q.tolist(), "K": K.tolist(), "scaled": Ss.tolist(), "A": A.tolist(), "O": O.tolist(),
                "printout": printed}
    log.info(f"matrix toy: creature weights {r2(A[2])}, output {r2(O[2])}")
    return Q, K, V, Ss, A, O


def fig_mat_qk(log):
    """q = W_Q e for creature and k = W_K e for fluffy, each row's dot product written out."""
    fig, ax = canvas(1.8)
    cw, ch, top = 0.26, 0.21, 1.3
    dot = "$\\cdot$"

    def product(x0, W, e, out, cmap, title, ename, oname, oc):
        ax.text(x0 + 0.95, 1.78, title, ha="center", va="top", fontsize=FS_SMALL, fontweight="bold")
        xr, _ = grid(ax, x0, top, cw, ch, W, 1.0, ["row 1", "row 2"], SLOTS, cmap=cmap, col_rot=50)
        ax.text(xr + 0.09, top - ch, "$\\times$", ha="center", va="center", fontsize=FS)
        xe = xr + 0.19
        grid(ax, xe, top, cw, ch, e.reshape(4, 1), 2.0, cmap="Greys")
        ax.text(xe + cw / 2, top - 4 * ch - 0.03, ename, ha="center", va="top", fontsize=FS_TINY)
        ax.text(xe + cw + 0.09, top - ch, "=", ha="center", va="center", fontsize=FS)
        xo = xe + cw + 0.19
        grid(ax, xo, top, cw, ch, out.reshape(2, 1), 2.0, cmap=cmap)
        ax.text(xo + cw / 2, top - 2 * ch - 0.03, oname, ha="center", va="top", fontsize=FS_SMALL, color=oc,
                fontweight="bold")
        for r in range(2):
            terms = " + ".join(f"{num(w).replace('-', '')}{dot}{num(x)}" if w >= 0 else
                               f"({num(w)}){dot}{num(x)}" for w, x in zip(W[r], e))
            ax.text(x0 + 0.95, 0.24 - r * 0.15, f"row {r + 1}: {terms} = {num(out[r])}", ha="center",
                    va="center", fontsize=FS_TINY, color=oc)

    q, k = W_Q2 @ X[2], W_K2 @ X[0]
    product(0.45, W_Q2, X[2], q, "Blues", "query of creature:  $\\mathbf{q} = W_Q\\,\\mathbf{e}$",
            "e (creature)", "q", ARMBLUE)
    product(3.2, W_K2, X[0], k, "Oranges", "key of fluffy:  $\\mathbf{k} = W_K\\,\\mathbf{e}$",
            "e (fluffy)", "k", ARMORANGE)
    save(fig, "att_mat_qk.pdf", log)


def fig_mat_scores(Q, K, Ss, A, log):
    fig, ax = canvas(1.6)
    cw, ch, top = 0.48, 0.25, 1.12
    x1 = 1.55
    sfmt = lambda v: f"{v:.2f}".replace("-", "−")
    xr, yb = grid(ax, x1, top, cw, ch, Ss, 1.8, None, None, hl_row=2, fmt=sfmt)
    for i, t in enumerate(TOKENS):
        ax.text(x1 - 0.05, top - (i + 0.5) * ch, f"{t}   q = ({num(Q[i, 0])}, {num(Q[i, 1])})", ha="right",
                va="center", fontsize=FS_TINY, color=ARMRED if i == 2 else INK)
    x2 = xr + 0.3
    grid(ax, x2, top, cw, ch, A, 1.0, None, None, hl_row=2)
    for j, t in enumerate(TOKENS):
        for x0 in (x1, x2):
            ax.text(x0 + (j + 0.5) * cw, top + 0.2, t, ha="center", va="bottom", fontsize=FS_TINY)
        ax.text(x1 + (j + 0.5) * cw, top + 0.04, f"({num(K[j, 0])}, {num(K[j, 1])})".replace("-", "−"),
                ha="center", va="bottom", fontsize=FS_TINY, color=ARMORANGE)
    ax.text(x1 - 0.05, top + 0.04, "k =", ha="right", va="bottom", fontsize=FS_TINY, color=ARMORANGE)
    ax.text(x1 + 1.5 * cw, yb - 0.06, "score = q$\\cdot$k / $\\sqrt{2}$", ha="center", va="top", fontsize=FS_SMALL)
    ax.text(x2 + 1.5 * cw, yb - 0.06, "weights (softmax by rows)", ha="center", va="top", fontsize=FS_SMALL)
    save(fig, "att_mat_scores.pdf", log)


def fig_mat_pipeline(Q, K, V, Ss, A, O, log):
    fig, ax = canvas(2.2)
    ch = 0.18
    sfmt = lambda v: f"{v:.2f}".replace("-", "−")
    xX, tX = 0.6, 1.45
    grid(ax, xX, tX, 0.2, ch, X, 2.0, TOKENS, None, cmap="Greys")
    ax.text(xX + 0.4, tX + 0.06, "X  (3 x 4)", ha="center", va="bottom", fontsize=FS_SMALL)
    xm = 1.98
    blocks = [(Q, tX + 0.55, 0.25, "Blues", "$Q = X W_Q^\\top$"),
              (K, tX - 0.15, 0.25, "Oranges", "$K = X W_K^\\top$"),
              (V, tX - 0.85, 0.2, "Greens", "$V = X W_V^\\top$")]
    for Mx, top, cw, cmap, lab in blocks:
        grid(ax, xm, top, cw, ch, Mx, 2.0, None, None, cmap=cmap)
        ax.text(xm, top + 0.03, lab, ha="left", va="bottom", fontsize=FS_TINY)
        arrow(ax, (xX + 0.83, tX - 1.5 * ch), (xm - 0.05, top - 1.5 * ch))
    xS, tS = 3.1, 1.62
    grid(ax, xS, tS, 0.32, ch, Ss, 1.8, None, None, hl_row=2, fmt=sfmt)
    ax.text(xS + 0.48, tS + 0.06, "$QK^\\top / \\sqrt{2}$", ha="center", va="bottom", fontsize=FS_SMALL)
    arrow(ax, (xm + 0.53, tX + 0.55 - 1.5 * ch), (xS - 0.04, tS - 1.2 * ch))
    arrow(ax, (xm + 0.53, tX - 0.15 - 1.5 * ch), (xS - 0.04, tS - 1.8 * ch))
    xA = 4.32
    grid(ax, xA, tS, 0.3, ch, A, 1.0, None, None, hl_row=2)
    ax.text(xA + 0.45, tS + 0.06, "softmax by rows", ha="center", va="bottom", fontsize=FS_SMALL)
    arrow(ax, (xS + 0.99, tS - 1.5 * ch), (xA - 0.04, tS - 1.5 * ch))
    xO, tO = 4.2, 0.62
    grid(ax, xO, tO, 0.26, ch, O, 1.0, None, None, hl_row=2)
    ax.text(xO + 0.52, tO + 0.06, "output = A V", ha="center", va="bottom", fontsize=FS_SMALL)
    arrow(ax, (xA + 0.45, tS - 3 * ch - 0.03), (xA + 0.45, tO + 0.22))
    yV = tX - 0.85 - 1.5 * ch
    arrow(ax, (xm + 0.83, yV), (xO - 0.05, yV))
    ax.text((xm + 0.83 + xO) / 2, yV + 0.05, "the values, blended by A", ha="center", va="bottom",
            fontsize=FS_TINY, color=GREY)
    save(fig, "att_mat_pipeline.pdf", log)


# ----------------------------------------------------------------------------- --verify
def verify_gpt2(log):
    """Check the GPT2 constants against what is already on disk (no download, no model run)."""
    import os
    import struct
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots").glob("*/config.json"))
    if not snaps:
        raise FileNotFoundError("no cached GPT-2 config.json; --verify needs the local HF cache")
    cfg = json.loads(snaps[0].read_text(encoding="utf-8"))
    got = {"d": cfg["n_embd"], "heads": cfg["n_head"], "d_k": cfg["n_embd"] // cfg["n_head"],
           "layers": cfg["n_layer"], "n_ctx": cfg["n_ctx"]}
    with open(snaps[0].parent / "model.safetensors", "rb") as f:
        hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    got["c_attn"] = hdr["h.0.attn.c_attn.weight"]["shape"]
    got["c_proj"] = hdr["h.0.attn.c_proj.weight"]["shape"]
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    got["tokens"] = [enc.decode([i]) for i in enc.encode(GPT2["sentence"])]
    for key, v in got.items():
        if v != GPT2[key]:
            raise AssertionError(f"GPT2[{key!r}] = {GPT2[key]}, the local files say {v}")
    log.info(f"--verify: GPT-2 sizes and the sentence's {len(got['tokens'])} tokens match the local files")


def gpt2_arithmetic(R, log):
    g = GPT2
    per_head = 3 * g["d"] * g["d_k"]
    if g["heads"] * 3 * g["d_k"] != g["c_attn"][1] or per_head * g["heads"] != g["c_attn"][0] * g["c_attn"][1]:
        raise AssertionError("c_attn should hold W_Q, W_K, W_V of all heads")
    grid_n = g["n_ctx"] ** 2
    R["gpt2"] = {"per_head_qkv": per_head, "scores_per_head_layer": grid_n,
                 "scores_full_window": grid_n * g["heads"] * g["layers"]}
    expect = {"per_head_qkv": 147_456, "scores_per_head_layer": 1_048_576, "scores_full_window": 150_994_944}
    if R["gpt2"] != expect:
        raise AssertionError(f"GPT-2 arithmetic {R['gpt2']} != quoted {expect}")
    log.info(f"GPT-2 arithmetic: {R['gpt2']}")


def main():
    log = setup_logging()
    t0 = time.time()
    if "--verify" in sys.argv:
        verify_gpt2(log)
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"date": time.strftime("%Y-%m-%d"), "quoted": QUOTED, "gpt2": None}
    T = toy(R, log)
    gpt2_arithmetic(R, log)
    saturation(R, log)
    MT = mat_toy(R, log)
    jobs = [lambda: fig_raw_grid(T, log), lambda: fig_learned_grid(T, log), lambda: fig_shapes(log),
            lambda: fig_mat_qk(log), lambda: fig_mat_scores(MT[0], MT[1], MT[3], MT[4], log),
            lambda: fig_mat_pipeline(*MT, log),
            lambda: fig_dot_product(R, log), lambda: fig_sqrt_dk(R, log),
            lambda: fig_softmax_why(R, log),
            lambda: fig_three_moves(R, log), lambda: fig_toy_chips(log), lambda: fig_soft_lookup(R, log),
            lambda: fig_qk_axis(T, log), lambda: fig_argmax_softmax(R, log), lambda: fig_dk_spread(R, log),
            lambda: fig_saturation(R, log), lambda: fig_value_chips(T, log), lambda: fig_gpt2_cattn(log)]
    for i, job in enumerate(jobs, 1):
        job()
        el = time.time() - t0
        log.info(f"{i}/{len(jobs)}, {el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "attention_figs.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
