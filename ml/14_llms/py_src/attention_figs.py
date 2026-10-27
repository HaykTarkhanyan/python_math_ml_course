"""Figures for LLM-3 "Attention: how tokens share meaning" (ml/14_llms/LLM3_attention.tex).

Toy, demonstrative numbers (ml/SLIDE_STYLE.md, 2026-10-05). The by-hand frames use three toy
vectors read as [adjective, fluffy, blue, noun]; every number the slides quote is asserted
(QUOTED), so a changed toy fails loudly. The only real numbers are GPT-2's sizes and the GPT-2
tokens of the instructor's July sentence; they are constants here, checked against the local
caches with --verify (HF cache config.json + safetensors header, tiktoken's gpt2 encoding; no
download, no model run).

Drawn at the size each figure gets on the slide (5.5 in = full text width), 6.5-8 pt text, and
save() refuses any text outside its figure.

Generates into ml/14_llms/fig/:
  att_raw_grid.pdf      -- raw vectors as their own query/key/value: scores and weights, creature's row
  att_learned_grid.pdf  -- one-number queries and keys: score = q x k, weights, creature's row
  att_value_update.pdf  -- creature's four numbers: before, after (raw), after (learned W_Q, W_K, W_V)
  att_shapes.pdf        -- the matrix shapes on "Donkey don't die, spring is coming" (9 tokens)
  att_dot_product.pdf   -- q . k for four keys: aligned, 45 degrees, perpendicular, opposite
  att_sqrt_dk.pdf       -- softmax of raw scores (d_k = 64) vs scores divided by sqrt(64) = 8
  att_softmax_why.pdf   -- scores, divided by their sum (a negative weight), softmax (all positive)

Run (~5 s):
  ./ma/Scripts/python.exe ml/14_llms/py_src/attention_figs.py            # the figures
  ./ma/Scripts/python.exe ml/14_llms/py_src/attention_figs.py --verify   # also check GPT2 against the caches
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
FEATURES = ["adjective", "fluffy", "blue", "noun"]
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
}
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


def fig_value_update(T, log):
    d_raw, d_l = T[4], T[5]
    rows = np.vstack([X[2], X[2] + d_raw, X[2] + d_l])
    labels = ["creature, before", "after, raw vectors", "after, learned W"]
    fig, ax = canvas(1.05, 2.75)
    cw, ch, top = 0.4, 0.27, 0.86
    x0 = 1.08
    grid(ax, x0, top, cw, ch, rows, 3.6, labels, FEATURES, hl_row=2, col_rot=0)
    save(fig, "att_value_update.pdf", log)


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
    R["sqrt_dk"] = {"spread_10k_pairs": round(spread, 2), "raw_scores": r2(raw),
                    "w_raw": r2(w_raw), "w_scaled": r2(w_s)}
    log.info(f"sqrt_dk: measured spread {spread:.2f} over 10,000 random pairs; raw {r2(raw)}; "
             f"weights raw {r2(w_raw)} scaled {r2(w_s)}")
    fig, axes = plt.subplots(1, 2, figsize=(2.75, 1.7), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.8, bottom=0.14, wspace=0.12)
    labels = [f"k{i + 1}" for i in range(5)]
    for ax, w, col, title in ((axes[0], w_raw, ARMRED, "softmax(q$\\cdot$k)"),
                              (axes[1], w_s, ARMBLUE, "softmax(q$\\cdot$k / 8)")):
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
    jobs = [lambda: fig_raw_grid(T, log), lambda: fig_learned_grid(T, log),
            lambda: fig_value_update(T, log), lambda: fig_shapes(log),
            lambda: fig_dot_product(R, log), lambda: fig_sqrt_dk(R, log),
            lambda: fig_softmax_why(R, log)]
    for i, job in enumerate(jobs, 1):
        job()
        el = time.time() - t0
        log.info(f"{i}/{len(jobs)}, {el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "attention_figs.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
