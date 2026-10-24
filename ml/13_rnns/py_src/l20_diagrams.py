"""Diagrams for the [53] RNN Foundations deck, redrawn from TikZ at slot size (2026-10-15).

The instructor review of 2026-10-15 found several [53] TikZ diagrams broken (overlapping boxes on
"What vanishing means in practice", gradient arcs piled on the nodes in "Backprop through time")
and most of the rest drawn tiny with half the frame empty. SLIDE_STYLE.md asks for essential
figures in matplotlib, drawn at the width they get on the slide, so they live here now.

Generates into ml/13_rnns/fig/ (slot width on the slide in brackets):
  cold_open.pdf          reviews of 3 / 11 / 42 words vs an MLP input of fixed size (0.48 col)
  seq_zoo.pdf            text, speech and music: three sequences whose length varies (0.58 col)
  onehot_columns.pdf     a 5-word vocabulary, four words as one-hot columns (0.64 col)
  onehot_cube.pdf        good / great / plastic as the corners of a cube, all sqrt(2) apart (0.58)
  one_new_arrow.pdf      a dense hidden layer, then the same layer plus the V arrow (0.48 col)
  folded_unrolled.pdf    the folded RNN (a loop) = the unrolled chain (full width, new frame)
  param_sharing.pdf      one W, one V, reused at every step (0.5 col)
  pad_effect.pdf         the toy net's score after 0..4 zero <pad> tokens (0.5 col, new frame)
  pad_mask.pdf           a padded batch of 3 reviews and its 0/1 mask (full width, new frame)
  bptt.pdf               backprop through time: gradients flow back, V's gradient is a sum (0.54)
  clipping.pdf           gradient clipping: same direction, length capped at h (0.5 col)
  vanish_practice.pdf    the backward signal fading along the chain + the Goris sentence shaded
                         by how much gradient reaches each word from "were" (full width)
  eigen_bridge.pdf       a 2x2 V with eigenvalues 0.8 and 1.2: the two parts of a vector over
                         29 multiplications, one fades, one grows (0.5 col, added 2026-10-20)

The toy net in pad_effect.pdf is the LOCKED net of forward_pass_anim.py (W, V, b, U, c copied
from there, its forward scores asserted below), so the 0.38 on the pad frame is the same 0.38 the
forward-pass animation ends on.

Run with the project venv (about 5 s):
    ./ma/Scripts/python.exe ml/13_rnns/py_src/l20_diagrams.py
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

# A missing glyph must crash the script, never ship a tofu box to a slide.
warnings.filterwarnings("error", message=".*missing from current font.*")

SEED = 509
BLUE, RED, ORANGE = "#0033A0", "#D90012", "#F2A800"
GREEN, GRAY, LIGHT = "#008C46", "#888888", "#DDDDDD"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# The LOCKED toy net of forward_pass_anim.py (rows of W = the 3 Armenian words).
W_TOY = np.array([[0.5, -0.5], [1.0, 0.0], [0.0, 1.0]])
V_TOY = np.array([[0.5, -0.5], [1.0, 0.0]])
B_TOY = np.zeros(2)
U_TOY = np.array([1.0, -1.0])
C_TOY = 0.0


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_diagrams")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l20_diagrams.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# --- drawing helpers: every canvas is in inches, axes span the whole figure ---------------
def canvas(w, h):
    """Figure of w x h inches whose data units are inches (see _learnings/2026-10-06-1315)."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text_w(s, fs, family=None):
    """Ink width of s at fs points, in inches - box widths come from this, not a guess."""
    prop = FontProperties(family=family) if family else None
    return TextPath((0, 0), s, size=fs, prop=prop).get_extents().width / 72


def arrow(ax, p, q, color="black", lw=1.0, style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def rbox(ax, cx, cy, w, h, text="", fc="white", ec="black", lw=0.9, fs=8, tc="black", **kw):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, **kw)


def node(ax, cx, cy, r, text="", fc="white", ec="black", lw=0.9, fs=8, tc="black"):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc)


def save(fig, name, log):
    out = FIG_DIR / name
    # Writing many PDFs back to back can hit a transient Windows lock (errno 22); see
    # _learnings/2026-09-28-2313_errno22-on-write-is-a-transient-lock.md. Bounded retry.
    for attempt in range(1, 6):
        try:
            fig.savefig(out, metadata={"CreationDate": None})
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out} after {attempt} attempts: {e}")
                raise
            log.warning(f"write of {out} failed (errno 22), retry {attempt}/4 in 1 s")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {out}")


Z = r"$\mathbf{z}^{[%s]}$"
X = r"$\mathbf{x}^{[%s]}$"


# --- figures --------------------------------------------------------------------------------
def fig_cold_open(log):
    """Reviews as strips of word squares (true counts) vs a fixed 8-number input layer."""
    w, h = 2.75, 1.95
    fig, ax = canvas(w, h)
    step, sq, x0 = 0.06, 0.048, 0.08
    rows = [(3, '3 words: "Fresh and sweet!"'), (11, "11 words, naming Yerevan"),
            (42, "42 words, the verdict at the very end")]
    y = h - 0.32
    for n, label in rows:
        ax.text(x0, y + 0.09, label, ha="left", va="bottom", fontsize=7)
        for k in range(n):
            last = (n == 42 and k == n - 1)
            ax.add_patch(Rectangle((x0 + k * step, y - sq / 2), sq, sq, lw=0.5,
                                   ec=RED if last else BLUE,
                                   fc=(RED + "AA") if last else (BLUE + "33")))
        y -= 0.36
    # fixed input layer: 8 neurons in a box
    n_in, r, gap = 8, 0.055, 0.155
    cx, cy = w / 2, 0.30
    left = cx - (n_in - 1) * gap / 2
    rbox(ax, cx, cy, n_in * gap + 0.1, 0.26, fc=ORANGE + "22", ec=ORANGE, lw=1.0)
    for k in range(n_in):
        node(ax, left + k * gap, cy, r, fc="white", ec="black", lw=0.7)
    ax.text(cx, cy - 0.21, "MLP input layer: exactly 8 numbers", ha="center", va="top",
            fontsize=7)
    arrow(ax, (cx, y + 0.24), (cx, cy + 0.17), color=GRAY, lw=1.6, ms=10)
    ax.text(cx + 0.08, (y + 0.24 + cy + 0.17) / 2, "all three, into\nthe same 8 slots?",
            ha="left", va="center", fontsize=6.5, color=GRAY, linespacing=1.0)
    ax.text(cx + n_in * gap / 2 + 0.17, cy + 0.02, "?", ha="center", va="center", fontsize=16,
            color=RED, fontweight="bold")
    save(fig, "cold_open.pdf", log)


def fig_seq_zoo(log):
    """Text, speech, music: one strip each, every one a sequence whose length varies."""
    rng = np.random.default_rng(SEED)
    w, h = 3.2, 2.3
    fig, ax = canvas(w, h)
    lx, x0, x1 = 0.04, 0.62, w - 0.05
    band = 0.70
    tops = [h - 0.06, h - 0.06 - band - 0.04, h - 0.06 - 2 * (band + 0.04)]

    # Text: token chips
    t = tops[0]
    ax.text(lx, t - 0.12, "Text", ha="left", va="top", fontsize=8.5, fontweight="bold",
            color=BLUE)
    ax.text(lx, t - 0.30, "3 to 3,000\nwords", ha="left", va="top", fontsize=6.5, color=GRAY,
            linespacing=1.0)
    words = "The pomegranates arrived fresh and sweet".split()
    cx = x0
    cy = t - 0.30
    for wd in words:
        bw = max(text_w(wd, 7), 0.05) + 0.08
        rbox(ax, cx + bw / 2, cy, bw, 0.20, wd, fc=BLUE + "18", ec=BLUE, lw=0.7, fs=7)
        cx += bw + 0.03
    if cx > w:
        raise ValueError(f"text chips run to {cx:.2f} in, canvas is {w} in")
    ax.text(x0, cy - 0.20, "order matters: the verdict can sit in the last word", ha="left",
            va="top", fontsize=6.5, color=GRAY, style="italic")

    # Speech: a synthetic waveform, three "syllables" of different length
    t = tops[1]
    ax.text(lx, t - 0.12, "Speech", ha="left", va="top", fontsize=8.5, fontweight="bold",
            color=RED)
    ax.text(lx, t - 0.30, "0.5 s to\nan hour", ha="left", va="top", fontsize=6.5, color=GRAY,
            linespacing=1.0)
    n = 1400
    tt = np.linspace(0, 1, n)
    env = np.zeros(n)
    for c, wdt in [(0.13, 0.07), (0.38, 0.11), (0.66, 0.08), (0.86, 0.06)]:
        env += np.exp(-((tt - c) / wdt) ** 2)
    sig = env * (np.sin(2 * np.pi * 55 * tt) + 0.6 * np.sin(2 * np.pi * 131 * tt)
                 + 0.35 * rng.standard_normal(n))
    sig = sig / np.abs(sig).max()
    mid = t - 0.32
    ax.plot(x0 + tt * (x1 - x0), mid + 0.2 * sig, color=RED, lw=0.45)
    ax.plot([x0, x1], [mid, mid], color=LIGHT, lw=0.5, zorder=0)
    ax.text(x0, mid - 0.25, "about 16,000 numbers per second", ha="left", va="top",
            fontsize=6.5, color=GRAY, style="italic")

    # Music: a melody as a piano roll
    t = tops[2]
    ax.text(lx, t - 0.12, "Music", ha="left", va="top", fontsize=8.5, fontweight="bold",
            color=GREEN)
    ax.text(lx, t - 0.30, "8 to 8,000\nnotes", ha="left", va="top", fontsize=6.5, color=GRAY,
            linespacing=1.0)
    pitches = [0, 2, 4, 2, 5, 4, 2, 0, 4, 7, 5, 4, 2, 0]
    durs = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2]
    total = sum(durs)
    unit = (x1 - x0) / total
    base, ph = t - 0.52, 0.042
    for k in range(9):
        ax.plot([x0, x1], [base + k * ph, base + k * ph], color="#F0F0F0", lw=0.4, zorder=0)
    pos = 0
    for p, d in zip(pitches, durs):
        ax.add_patch(Rectangle((x0 + pos * unit + 0.005, base + p * ph - ph * 0.4),
                               d * unit - 0.01, ph * 0.8, fc=GREEN + "BB", ec=GREEN, lw=0.4))
        pos += d
    ax.text(x0, base - 0.08, "a note means something only next to its neighbours",
            ha="left", va="top", fontsize=6.5, color=GRAY, style="italic")
    save(fig, "seq_zoo.pdf", log)


def fig_onehot_columns(log):
    """Vocabulary of 5 on the left; four words of the review as one-hot columns."""
    vocab = ["The", "pomegranates", "arrived", "fresh", "sweet"]
    words = ["The", "pomegranates", "arrived", "fresh"]
    w, h = 3.5, 2.05
    fig, ax = canvas(w, h)
    cell = 0.19
    row_y = [1.08 - k * cell for k in range(5)]
    ax.text(0.05, row_y[0] + 0.2, "vocabulary", ha="left", va="bottom", fontsize=7.5,
            fontweight="bold")
    for k, v in enumerate(vocab):
        ax.text(0.05, row_y[k], f"{k}", ha="left", va="center", fontsize=7, color=GRAY,
                family="monospace")
        ax.text(0.18, row_y[k], v, ha="left", va="center", fontsize=7.5)
    col_x = [1.42, 2.02, 2.62, 3.22]
    chip_y = [1.86, 1.56, 1.86, 1.56]          # staggered, so wide chips never touch
    for j, (wd, cx) in enumerate(zip(words, col_x)):
        bw = max(0.36, text_w(wd, 7.5) + 0.1)
        rbox(ax, cx, chip_y[j], bw, 0.22, wd, fc=BLUE + "18", ec=BLUE, lw=0.8, fs=7.5)
        arrow(ax, (cx, chip_y[j] - 0.12), (cx, row_y[0] + cell / 2 + 0.03), color=BLUE,
              lw=0.8)
        hot = vocab.index(wd)
        for k in range(5):
            on = (k == hot)
            ax.add_patch(Rectangle((cx - cell / 2, row_y[k] - cell / 2), cell, cell, lw=0.7,
                                   ec="black", fc=(GREEN + "55") if on else "white"))
            ax.text(cx, row_y[k], "1" if on else "0", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if on else "normal")
    for k in range(5):
        ax.plot([1.0, col_x[0] - cell / 2 - 0.02], [row_y[k], row_y[k]], color=LIGHT, lw=0.5,
                ls=":", zorder=0)
    ax.text(w - 0.03, 0.03, "each word = a 5-d vector with a single 1, at its own index",
            ha="right", va="bottom", fontsize=7, color=GRAY, style="italic")
    save(fig, "onehot_columns.pdf", log)


def fig_onehot_cube(log):
    """good / great / plastic = the 3 corners (1,0,0) (0,1,0) (0,0,1): all sqrt(2) apart."""
    w, h = 3.2, 2.35
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([-0.06, -0.08, 1.12, 1.16], projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=35)
    # unit cube wireframe
    for a in (0, 1):
        for b in (0, 1):
            ax.plot([0, 1], [a, a], [b, b], color=LIGHT, lw=0.6)
            ax.plot([a, a], [0, 1], [b, b], color=LIGHT, lw=0.6)
            ax.plot([a, a], [b, b], [0, 1], color=LIGHT, lw=0.6)
    for v in [(1.2, 0, 0), (0, 1.2, 0), (0, 0, 1.2)]:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=GRAY, lw=0.7)
    pts = {"good": (1, 0, 0), "great": (0, 1, 0), "plastic": (0, 0, 1)}
    cols = {"good": GREEN, "great": BLUE, "plastic": RED}
    names = list(pts)
    for i in range(3):
        for j in range(i + 1, 3):
            p, q = np.array(pts[names[i]]), np.array(pts[names[j]])
            ax.plot(*zip(p, q), color=RED, lw=1.3)
            m = (p + q) / 2
            ax.text(*(m + np.array([0.04, 0.04, 0.04])), r"$\sqrt{2}$", fontsize=8, color=RED)
    coords = {"good": "(1, 0, 0)", "great": "(0, 1, 0)", "plastic": "(0, 0, 1)"}
    for nm, p in pts.items():
        ax.scatter(*p, s=26, color=cols[nm], depthshade=False, zorder=5)
    # labels in 2-D figure coordinates, placed by eye next to each projected corner
    for nm, (fx, fy, ha) in {"good": (0.05, 0.12, "left"), "great": (0.97, 0.12, "right"),
                             "plastic": (0.55, 0.93, "left")}.items():
        fig.text(fx, fy, f'"{nm}" {coords[nm]}', fontsize=7.5, color=cols[nm],
                 fontweight="bold", ha=ha)
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.2); ax.set_zlim(0, 1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    d = [np.linalg.norm(np.array(pts[a]) - np.array(pts[b])) for a, b in
         [("good", "great"), ("good", "plastic"), ("great", "plastic")]]
    assert np.allclose(d, np.sqrt(2)), d
    log.info(f"one-hot distances: {np.round(d, 4).tolist()}")
    save(fig, "onehot_cube.pdf", log)


def fig_one_new_arrow(log):
    """Top: a dense hidden layer x -> z -> f. Bottom: the same, plus z[t-1] -> z in red."""
    w, h = 2.7, 2.25
    fig, ax = canvas(w, h)
    r = 0.2
    xs = [0.45, 1.35, 2.25]

    def row(y, sup, title):
        ax.text(0.05, y + 0.36, title, ha="left", va="center", fontsize=7.5, fontweight="bold")
        node(ax, xs[0], y, r, X % sup if sup else r"$\mathbf{x}$", fc=BLUE + "22", ec=BLUE, fs=8.5)
        node(ax, xs[1], y, r, Z % sup if sup else r"$\mathbf{z}$", fc=GREEN + "22", ec=GREEN,
             fs=8.5)
        node(ax, xs[2], y, r, r"$f^{[%s]}$" % sup if sup else r"$f$", fc=ORANGE + "33",
             ec=ORANGE, fs=8.5)
        arrow(ax, (xs[0] + r, y), (xs[1] - r, y), color=BLUE, lw=1.0)
        ax.text((xs[0] + xs[1]) / 2, y + 0.05, r"$\mathbf{W}$", ha="center", va="bottom",
                fontsize=8, color=BLUE)
        arrow(ax, (xs[1] + r, y), (xs[2] - r, y), color=ORANGE, lw=1.0)
        ax.text((xs[1] + xs[2]) / 2, y + 0.05, r"$\mathbf{U}$", ha="center", va="bottom",
                fontsize=8, color=ORANGE)

    row(1.78, None, "a hidden layer, [41_1]")
    ax.plot([0.05, w - 0.05], [1.3, 1.3], color=LIGHT, lw=0.6)
    yb = 0.82
    row(yb, "t", "the same layer + one arrow = an RNN")
    zp = (xs[1] - 0.6, yb - 0.5)
    node(ax, *zp, r * 1.3, Z % "t-1", fc=GREEN + "12", ec=GREEN, fs=8.5)
    arrow(ax, (zp[0] + r * 0.95, zp[1] + r * 0.95), (xs[1] - r * 0.72, yb - r * 0.72), color=RED,
          lw=1.8)
    ax.text(zp[0] + 0.44, zp[1] + 0.14, r"$\mathbf{V}$", ha="left", va="center", fontsize=9,
            color=RED, fontweight="bold")
    ax.text(1.42, zp[1] - 0.04, "hidden to hidden: the\nprevious word's state",
            ha="left", va="center", fontsize=6.5, color=RED, linespacing=1.05)
    save(fig, "one_new_arrow.pdf", log)


def draw_unrolled(ax, x_left, y_state, n, gap, r, labels_t=None, show_f="last", fs=8.5):
    """An unrolled chain of n steps; returns state x positions."""
    xs = [x_left + k * gap for k in range(n)]
    labels_t = labels_t or [str(k + 1) for k in range(n)]
    for k, (x, t) in enumerate(zip(xs, labels_t)):
        node(ax, x, y_state, r, Z % t, fc=GREEN + "22", ec=GREEN, fs=fs)
        rbox(ax, x, y_state - 0.62, 0.36, 0.22, X % t, fc=BLUE + "18", ec=BLUE, lw=0.8, fs=fs)
        arrow(ax, (x, y_state - 0.50), (x, y_state - r), color=BLUE, lw=1.0)
        ax.text(x + 0.04, y_state - 0.37, r"$\mathbf{W}$", ha="left", va="center", fontsize=7,
                color=BLUE)
        if show_f == "all" or (show_f == "last" and k == n - 1):
            node(ax, x, y_state + 0.58, r * 0.85, r"$f^{[%s]}$" % t, fc=ORANGE + "33", ec=ORANGE,
                 fs=fs)
            arrow(ax, (x, y_state + r), (x, y_state + 0.58 - r * 0.85), color=ORANGE, lw=1.0)
            ax.text(x + 0.04, y_state + 0.30, r"$\mathbf{U}$", ha="left", va="center",
                    fontsize=7, color=ORANGE)
        if k > 0:
            arrow(ax, (xs[k - 1] + r, y_state), (x - r, y_state), color=RED, lw=1.4)
            ax.text((xs[k - 1] + x) / 2, y_state + 0.05, r"$\mathbf{V}$", ha="center",
                    va="bottom", fontsize=7.5, color=RED)
    return xs


def fig_folded_unrolled(log):
    """Left: the folded RNN, a cell with a loop. Right: the same net unrolled over 4 words."""
    w, h = 5.5, 1.95
    fig, ax = canvas(w, h)
    # folded
    cx, cy, r = 0.75, 1.02, 0.26
    node(ax, cx, cy, r, Z % "t", fc=GREEN + "22", ec=GREEN, fs=8.5)
    rbox(ax, cx, cy - 0.66, 0.40, 0.22, X % "t", fc=BLUE + "18", ec=BLUE, lw=0.8, fs=8.5)
    arrow(ax, (cx, cy - 0.54), (cx, cy - r), color=BLUE, lw=1.0)
    ax.text(cx + 0.05, cy - 0.40, r"$\mathbf{W}$", ha="left", va="center", fontsize=7.5,
            color=BLUE)
    node(ax, cx, cy + 0.62, r * 0.8, r"$f^{[t]}$", fc=ORANGE + "33", ec=ORANGE, fs=8.5)
    arrow(ax, (cx, cy + r), (cx, cy + 0.62 - r * 0.8), color=ORANGE, lw=1.0)
    ax.text(cx + 0.05, cy + 0.36, r"$\mathbf{U}$", ha="left", va="center", fontsize=7.5,
            color=ORANGE)
    arrow(ax, (cx + r * 0.85, cy + r * 0.5), (cx + r * 0.85, cy - r * 0.5), color=RED, lw=1.6,
          connectionstyle="arc3,rad=-2.4")
    ax.text(cx + 0.62, cy, r"$\mathbf{V}$", ha="left", va="center", fontsize=9, color=RED,
            fontweight="bold")
    ax.text(cx, 0.06, "folded: one cell, a loop", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold")
    ax.text(1.72, cy, "=", ha="center", va="center", fontsize=18, color=GRAY)
    ax.text(1.72, cy - 0.30, "unroll\nin time", ha="center", va="top", fontsize=6.5,
            color=GRAY, linespacing=1.0)
    # unrolled
    xs = draw_unrolled(ax, 2.35, cy, 4, 0.92, 0.22, show_f="last")
    ax.text((xs[0] + xs[-1]) / 2, 0.06, "unrolled: one copy per word, the same W, V, U in every copy",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    arrow(ax, (xs[0] - 0.2, cy + 0.62), (xs[-2] + 0.25, cy + 0.62), color=GRAY, lw=0.7)
    ax.text((xs[0] + xs[-2]) / 2, cy + 0.66, "time", ha="center", va="bottom", fontsize=6.5,
            color=GRAY)
    save(fig, "folded_unrolled.pdf", log)


def fig_param_sharing(log):
    """One W and one V as two little matrices on top, dotted to every place they are used."""
    w, h = 2.75, 2.2
    fig, ax = canvas(w, h)
    cy = 1.08
    xs = draw_unrolled(ax, 0.45, cy, 3, 0.9, 0.2, show_f="none")
    # the parameter store
    top = 1.98
    for (label, color, cx) in [(r"$\mathbf{W}$", BLUE, 0.85), (r"$\mathbf{V}$", RED, 1.9)]:
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((cx - 0.1 + j * 0.1, top - i * 0.1), 0.1, 0.1,
                                       fc=color + "44", ec=color, lw=0.5))
        ax.text(cx + 0.16, top, label, ha="left", va="center", fontsize=8.5, color=color)
    ax.text(0.85, top - 0.18, "used 3 times", ha="center", va="top", fontsize=6.5, color=BLUE)
    ax.text(1.9, top - 0.18, "used 2 times", ha="center", va="top", fontsize=6.5, color=RED)
    ax.text(w / 2, 0.02, "one copy of each matrix: every blue arrow is the same W,\n"
            "every red arrow the same V", ha="center", va="bottom", fontsize=6.5, color=GRAY,
            style="italic", linespacing=1.05)
    save(fig, "param_sharing.pdf", log)


def toy_scores_with_pads(n_pads):
    """The locked toy net reads its 3 words, then n zero <pad> vectors; score after each."""
    z = np.zeros(2)
    for i in range(3):
        z = np.tanh(V_TOY.T @ z + W_TOY[i] + B_TOY)
    sig = lambda a: 1 / (1 + np.exp(-a))
    scores = [float(sig(U_TOY @ z + C_TOY))]
    states = [z.copy()]
    for _ in range(n_pads):
        z = np.tanh(V_TOY.T @ z + np.zeros(2) + B_TOY)   # a pad's one-hot is all zeros
        scores.append(float(sig(U_TOY @ z + C_TOY)))
        states.append(z.copy())
    return scores, states


def fig_pad_effect(log):
    """Bar chart: the toy net's score after reading 0..4 extra pads."""
    scores, states = toy_scores_with_pads(4)
    # locked: matches forward_pass_anim.py (0.38) and the numbers printed on the frame
    assert round(scores[0], 2) == 0.38, scores
    assert round(scores[1], 2) == 0.65, scores
    assert [round(v, 2) for v in states[1]] == [0.56, -0.05], states[1]
    log.info(f"scores after 0..4 pads: {[round(s, 3) for s in scores]}")
    log.info(f"state after 1 pad: {np.round(states[1], 3).tolist()}")
    fig, ax = plt.subplots(figsize=(2.8, 2.0))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.2, top=0.86)
    ks = np.arange(len(scores))
    cols = [BLUE] + [RED] * (len(scores) - 1)
    bars = ax.bar(ks, scores, color=cols, width=0.62)
    ax.bar_label(bars, labels=[f"{s:.2f}" for s in scores], fontsize=7.5, padding=2)
    ax.axhline(0.5, color=GRAY, lw=0.8, ls="--")
    ax.text(4.42, 0.86, "dashed: 0.5, where the verdict flips", ha="right", va="top",
            fontsize=6.5, color=GRAY)
    ax.set_xticks(ks, ["none"] + [str(k) for k in ks[1:]], fontsize=7.5)
    ax.set_xlabel("<pad> tokens read after the 3 words", fontsize=7.5)
    ax.set_ylabel("score", fontsize=7.5)
    ax.set_ylim(0, 0.88)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Same 3 words, same net, only pads added", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "pad_effect.pdf", log)


def fig_pad_mask(log):
    """A batch of 3 reviews padded to the longest one, the mask beside it, and where each
    review's state must be read."""
    reviews = ["Fresh and sweet!", "The pomegranates arrived fresh and sweet.",
               "Arrived bruised and dry."]
    toks = [r.split() for r in reviews]
    T = max(len(t) for t in toks)
    mono, char_in = 7, 0.6 * 7 / 72
    col_w = []
    for j in range(T):
        cands = [t[j] if j < len(t) else "<pad>" for t in toks]
        col_w.append(max(char_in * len(c) for c in cands) + 0.09)
    gap, x0 = 0.03, 0.62
    lefts, x = [], x0
    for cw in col_w:
        lefts.append(x)
        x += cw + gap
    grid_right = x - gap
    w, h = 5.5, 1.75
    fig, ax = canvas(w, h)
    row_h, ys = 0.24, [1.18, 0.86, 0.54]
    ax.text(x0, 1.50, "the batch: 3 reviews, padded to the longest (6)", ha="left",
            va="bottom", fontsize=7.5, fontweight="bold")
    for i, (t, y) in enumerate(zip(toks, ys)):
        ax.text(x0 - 0.08, y, f"len {len(t)}", ha="right", va="center", fontsize=7, color=GRAY)
        for j in range(T):
            real = j < len(t)
            ax.add_patch(Rectangle((lefts[j], y - row_h / 2), col_w[j], row_h, lw=0.8,
                                   ec=GREEN if real else "#AAAAAA",
                                   fc=(GREEN + "1C") if real else "#EEEEEE",
                                   ls="-" if real else "--"))
            ax.text(lefts[j] + col_w[j] / 2, y, t[j] if real else "<pad>", ha="center",
                    va="center", fontsize=mono, family="monospace",
                    color="black" if real else "#999999")
        j = len(t) - 1
        ax.add_patch(Rectangle((lefts[j] - 0.025, y - row_h / 2 - 0.025), col_w[j] + 0.05,
                               row_h + 0.05, fill=False, ec=ORANGE, lw=1.6))
    ax.text(x0, 0.30, "orange: each review's last REAL word - read its state there",
            ha="left", va="top", fontsize=7, color="#B07800")
    # mask grid
    mx0, cell = grid_right + 0.42, 0.2
    ax.text(mx0, 1.50, "the mask", ha="left", va="bottom", fontsize=7.5, fontweight="bold")
    ax.text(mx0, 0.30, "1 = word, 0 = pad", ha="left", va="top", fontsize=7, color=GRAY)
    for i, (t, y) in enumerate(zip(toks, ys)):
        for j in range(T):
            real = j < len(t)
            ax.add_patch(Rectangle((mx0 + j * cell, y - cell / 2), cell, cell, lw=0.6, ec="black",
                                   fc=(GREEN + "55") if real else "white"))
            ax.text(mx0 + j * cell + cell / 2, y, "1" if real else "0", ha="center",
                    va="center", fontsize=7.5)
    arrow(ax, (grid_right + 0.06, ys[1]), (mx0 - 0.06, ys[1]), color=GRAY, lw=0.8)
    if mx0 + T * cell > w:
        raise ValueError(f"mask grid runs to {mx0 + T * cell:.2f} in, canvas is {w} in")
    log.info(f"pad_mask: lengths {[len(t) for t in toks]}, T={T}, grid ends {grid_right:.2f} in")
    save(fig, "pad_mask.pdf", log)


def fig_bptt(log):
    """Forward arrows on top; the gradient flowing back below; V's gradient = a sum over steps."""
    w, h = 3.0, 1.65
    fig, ax = canvas(w, h)
    r, cy = 0.19, 0.75
    xs = [0.32, 0.92, 1.52, 2.12]
    for k, x in enumerate(xs):
        node(ax, x, cy, r, Z % (k + 1), fc=GREEN + "22", ec=GREEN, fs=8.5)
        if k:
            arrow(ax, (xs[k - 1] + r, cy), (x - r, cy), color="black", lw=0.9)
            ax.text((xs[k - 1] + x) / 2, cy - 0.05, r"$\mathbf{V}$", ha="center", va="top",
                    fontsize=7, color="black")
    rbox(ax, 2.70, cy, 0.34, 0.26, r"$L$", fc=RED + "22", ec=RED, fs=9)
    arrow(ax, (xs[-1] + r, cy), (2.53, cy), color="black", lw=0.9)
    # tied weight, on top: each use of V (one per arrow between states) adds a share
    sy = 1.42
    label = r"$\partial L / \partial \mathbf{V}$ = the sum of 3 shares, one per use of V"
    bw = text_w(label, 7) + 0.16
    if bw > w - 0.1:
        raise ValueError(f"sum box needs {bw:.2f} in, canvas is {w} in")
    rbox(ax, w / 2, sy, bw, 0.30, label, fc=RED + "10", ec=RED, lw=0.8, fs=7)
    for k in range(3):
        mx = (xs[k] + xs[k + 1]) / 2
        ax.plot([mx, mx], [cy + 0.04, sy - 0.16], color=RED, lw=0.7, ls=":")
        ax.text(mx + 0.03, (cy + sy) / 2, "+", ha="left", va="center", fontsize=8, color=RED)
    # backward: below the nodes, fading
    alphas = [0.35, 0.55, 0.75, 1.0]
    yb = cy - 0.42
    arrow(ax, (2.70, cy - 0.15), (2.70, yb), color=RED, lw=1.4, style="-")
    arrow(ax, (2.70, yb), (xs[-1] + 0.05, yb), color=RED, lw=1.4)
    for k in range(3, 0, -1):
        arrow(ax, (xs[k] - 0.05, yb), (xs[k - 1] + 0.05, yb), color=RED, lw=1.4,
              alpha=alphas[k - 1])
    ax.text(1.5, yb - 0.08, "backward: the gradient flows right to left,\n"
            "through every V on the way", ha="center", va="top", fontsize=7, color=RED,
            linespacing=1.05)
    save(fig, "bptt.pdf", log)


def fig_clipping(log):
    """A gradient longer than h, and the clipped one: same direction, length exactly h."""
    w, h = 2.75, 2.0
    fig, ax = canvas(w, h)
    o = np.array([0.95, 0.85])
    hr = 0.62
    ax.add_patch(Circle(o, hr, fill=False, ec=GRAY, lw=0.9, ls="--"))
    g = np.array([1.55, 0.88])
    gn = np.linalg.norm(g)
    clipped = g * hr / gn
    arrow(ax, tuple(o), tuple(o + g), color=RED, lw=1.8, ms=9)
    arrow(ax, tuple(o), tuple(o + clipped), color=GREEN, lw=2.4, ms=9)
    ang = np.degrees(np.arctan2(g[1], g[0]))
    ax.text(*(o + g * 0.80 + np.array([-0.06, 0.1])), r"$\nabla$, too long", rotation=ang,
            ha="center", va="bottom", fontsize=7.5, color=RED, rotation_mode="anchor")
    ax.text(*(o + clipped * 0.5 + np.array([0.08, -0.1])), "clipped", rotation=ang,
            ha="center", va="top", fontsize=7.5, color=GREEN, rotation_mode="anchor")
    ax.plot([o[0], o[0]], [o[1], o[1] - hr], color=GRAY, lw=0.7)
    ax.text(o[0] + 0.04, o[1] - hr / 2, r"$h$", ha="left", va="center", fontsize=8,
            color=GRAY)
    ax.text(o[0], o[1] - hr - 0.1, "threshold: a circle of radius h", ha="center", va="top",
            fontsize=7, color=GRAY)
    ax.text(w - 0.12, 0.05, "same direction,\nlength cut to h", ha="right", va="bottom",
            fontsize=7, color=GREEN, linespacing=1.05, multialignment="right")
    save(fig, "clipping.pdf", log)


def fig_vanish_practice(log):
    """Top: the chain and its backward signal, fading. Bottom: the Goris sentence ([53]'s
    fixed-window frame, [54]'s gate story) with each word shaded by how much gradient reaches it
    from "were" (0.8 per step back). 2026-10-15 student review: it used "The cat ... was", a
    second sentence for the same idea, and h where the rest of the deck says z."""
    w, h = 5.5, 2.05
    fig, ax = canvas(w, h)
    r, cy = 0.235, 1.62
    xs = [0.45, 1.15, 1.85, 3.40, 4.10]
    labels = [r"$\mathbf{z}^{[1]}$", r"$\mathbf{z}^{[2]}$", r"$\mathbf{z}^{[3]}$",
              r"$\mathbf{z}^{[T-1]}$", r"$\mathbf{z}^{[T]}$"]
    for k, (x, lab) in enumerate(zip(xs, labels)):
        node(ax, x, cy, r, lab, fc=GREEN + "22", ec=GREEN, fs=8.5)
    for a, b in [(0, 1), (1, 2), (3, 4)]:
        arrow(ax, (xs[a] + r, cy), (xs[b] - r, cy), color="black", lw=0.9)
    ax.text(2.62, cy, r"$\cdots$", ha="center", va="center", fontsize=11)
    arrow(ax, (xs[2] + r, cy), (2.38, cy), color="black", lw=0.9)
    arrow(ax, (2.86, cy), (xs[3] - r, cy), color="black", lw=0.9)
    rbox(ax, 4.85, cy, 0.62, 0.28, r"loss $\mathcal{L}$", fc=RED + "22", ec=RED, fs=8)
    arrow(ax, (xs[4] + r, cy), (4.53, cy), color="black", lw=0.9)
    yb = cy - 0.40
    segs = [((4.85, yb), (xs[4], yb), 1.0, 2.6, "large"), ((xs[4], yb), (xs[3], yb), 0.8, 2.1, ""),
            ((xs[2] + 0.5, yb), (xs[2], yb), 0.35, 1.2, "tiny"),
            ((xs[2], yb), (xs[1], yb), 0.25, 0.9, ""), ((xs[1], yb), (xs[0], yb), 0.15, 0.6, "~0")]
    arrow(ax, (4.85, cy - 0.15), (4.85, yb), color=RED, lw=2.6, style="-")
    for (p, q, al, lw, lab) in segs:
        arrow(ax, p, q, color=RED, lw=lw, alpha=al)
        if lab:
            ax.text((p[0] + q[0]) / 2, yb - 0.07, lab, ha="center", va="top", fontsize=7,
                    color=RED, alpha=max(al, 0.5))
    ax.text(2.62, yb, r"$\cdots$", ha="center", va="center", fontsize=11, color=RED, alpha=0.5)
    # the sentence, shaded by gradient reach
    sent = ("The pomegranates that my grandmother sent us from Goris last autumn were "
            "sweet").split()
    iv = sent.index("were")
    ic = sent.index("pomegranates")
    assert iv - ic == 10, iv - ic          # the fixed-window frame says "10 words back"
    log.info(f'"pomegranates" sits {iv - ic} words before "were"')
    sy = 0.42
    char = 0.062
    widths = [char * len(t) + 0.07 for t in sent]
    total = sum(widths) + 0.025 * (len(sent) - 1)
    x = (w - total) / 2
    cxs = []
    for k, (t, bw) in enumerate(zip(sent, widths)):
        reach = 0.8 ** abs(iv - k) if k <= iv else 0.0
        fc = RED if k == iv else RED
        alpha = 0.85 if k == iv else 0.08 + 0.6 * reach
        ax.add_patch(FancyBboxPatch((x, sy - 0.11), bw, 0.22, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, alpha=alpha if k <= iv else 0.04, ec="none"))
        ax.text(x + bw / 2, sy, t, ha="center", va="center", fontsize=7.5,
                color="white" if k == iv else "black",
                fontweight="bold" if k in (ic, iv) else "normal")
        cxs.append(x + bw / 2)
        x += bw + 0.025
    ax.annotate("", xy=(cxs[ic], sy + 0.13), xytext=(cxs[iv], sy + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=0.1", mutation_scale=7))
    # no superscript here: at 7 pt it would print the exponent at ~5 pt
    ax.text((cxs[ic] + cxs[iv]) / 2, sy + 0.42, f'"were" needs "pomegranates", {iv - ic} words '
            f'back: x0.8 per step leaves {0.8 ** (iv - ic):.2f} of the signal',
            ha="center", va="bottom", fontsize=7.5, color=RED)
    ax.text(w / 2, 0.06, "shading = how much gradient from \"were\" reaches each word "
            "(illustrative: 0.8 per step)", ha="center", va="bottom", fontsize=6.5,
            color=GRAY, style="italic")
    save(fig, "vanish_practice.pdf", log)


def fig_eigen_bridge(log):
    """2026-10-20 (student review): the bridge from the scalar 0.8^29 to "the largest eigenvalue
    decides". V = [[1, -0.2], [-0.2, 1]]: eigenvalue 0.8 along (1,1), 1.2 along (1,-1). The
    vector (1, 0) = 0.5 (1,1) + 0.5 (1,-1); V only rescales each part."""
    V = np.array([[1.0, -0.2], [-0.2, 1.0]])
    lam = np.linalg.eigvalsh(V)
    assert np.allclose(lam, [0.8, 1.2]), lam
    g = np.array([1.0, 0.0])
    ks = np.arange(30)
    total = np.array([np.linalg.norm(np.linalg.matrix_power(V, k) @ g) for k in ks])
    part_08 = 0.5 * np.sqrt(2) * 0.8 ** ks           # length of the (1,1) part
    part_12 = 0.5 * np.sqrt(2) * 1.2 ** ks           # length of the (1,-1) part
    assert np.allclose(total, np.sqrt(part_08 ** 2 + part_12 ** 2))
    assert round(total[29]) == 140, total[29]
    log.info(f"eigen bridge: |V^29 g| = {total[29]:.1f}, parts {part_08[29]:.5f} / "
             f"{part_12[29]:.1f}")
    fig, ax = plt.subplots(figsize=(2.85, 2.1))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.88)
    ax.semilogy(ks, part_12, color=RED, lw=1.6)
    ax.semilogy(ks, part_08, color=BLUE, lw=1.6)
    ax.semilogy(ks, total, color="black", lw=0.9, ls="--")
    # labels in the empty corners, never on a line
    ax.text(0.5, 40, "part along (1,-1):\nx1.2 per step", ha="left", va="center", fontsize=7,
            color=RED, linespacing=1.0)
    ax.text(0.5, 0.003, "part along (1,1):\nx0.8 per step", ha="left", va="center", fontsize=7,
            color=BLUE, linespacing=1.0)
    ax.set_xlabel("multiplications by V", fontsize=7.5)
    ax.set_ylabel("length (log scale)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8.5)        # log axis: exponents print at 70%
    ax.set_title("Two directions, two fates", fontsize=8)     # dashed line: named on the slide
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "eigen_bridge.pdf", log)


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fig_cold_open, fig_seq_zoo, fig_onehot_columns, fig_onehot_cube,
              fig_one_new_arrow, fig_folded_unrolled, fig_param_sharing, fig_pad_effect,
              fig_pad_mask, fig_bptt, fig_clipping, fig_vanish_practice, fig_eigen_bridge):
        f(log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
