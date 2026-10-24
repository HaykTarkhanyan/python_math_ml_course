"""Diagrams for the [54] Road to Attention deck, redrawn or new at slot size (2026-10-15).

The instructor review of 2026-10-15 found the [54] TikZ diagrams broken or unreadable: the LSTM
anatomy was a tangle of input wires with labels under the circles, the encoder-decoder frame
showed a thumbnail with no words in it, and the cliffhanger's look-back arrows merged into a grey
smear. SLIDE_STYLE.md asks for essential figures in matplotlib, drawn at the width they get on
the slide, so they live here now. The gate story is new (instructor-approved the same day).

Generates into ml/13_rnns/fig/ (slot width on the slide in brackets):
  lstm_anatomy.pdf       one LSTM step: the cell highway on top, four gates fed by one input
                         bus, the output path (full width)
  gate_mask.pdf          a forget gate [0.9, 0.1] times a cell [2.0, -1.0] = [1.8, -0.1]: one
                         slot kept, one wiped (0.5 col, answer of a predict-first frame)
  gate_story_0..4.pdf    click-through: one memory slot of an LSTM reads "The pomegranates
                         that my grandmother sent us from Goris last autumn were sweet" -
                         input gate writes "plural", forget gate holds it, output gate releases
                         it at the verb, forget gate wipes it after (full width). ILLUSTRATIVE
                         gate values, but the cell values are computed with the real update
                         c = f * c_prev + i * candidate and asserted below
  lstm_timeline.pdf      1991 diagnosis -> 1997 LSTM -> 2015 voice search -> 2016 Translate
                         (full width)
  seq2seq.pdf            encoder reads "I watch all of this in silence", hands one vector C to
                         the decoder, which writes the first 6 words of the chapter's Armenian
                         line (full width)
  bottleneck.pdf         the 42-word review poured through a funnel into one fixed-size C,
                         early words faded (full width)
  cliffhanger.pdf        the decoder about to write word 4, dashed "?" lines to every encoder
                         state - the question, not the answer (full width)

Armenian text is read from py_src/data/armenian_line.txt (never retyped) and set in Segoe UI;
a missing glyph is promoted to an error. The 42-word review is imported from pad_truncate.py,
its one home.

Run with the project venv (about 5 s):
    ./ma/Scripts/python.exe ml/13_rnns/py_src/l21_diagrams.py
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath

# A missing glyph must crash the script, never ship a tofu box to a slide.
warnings.filterwarnings("error", message=".*missing from current font.*")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from pad_truncate import REVIEWS  # noqa: E402  (the 42-word review lives there)

SEED = 509
BLUE, RED, ORANGE = "#0033A0", "#D90012", "#F2A800"
GREEN, GRAY, LIGHT = "#008C46", "#888888", "#DDDDDD"
ARM_FONT = "Segoe UI"

CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
DATA_FILE = HERE.parent / "data" / "armenian_line.txt"
GLOSS_HEAD = "I watch all of this in silence".split()   # gloss of the line's first 6 words


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l21_diagrams")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l21_diagrams.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# --- drawing helpers: every canvas is in inches, axes span the whole figure ---------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text_w(s, fs, family=None):
    """Ink width of s at fs points, in inches."""
    prop = FontProperties(family=family) if family else None
    return TextPath((0, 0), s, size=fs, prop=prop).get_extents().width / 72


def arrow(ax, p, q, color="black", lw=1.0, style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def rbox(ax, cx, cy, w, h, text="", fc="white", ec="black", lw=0.9, fs=8, tc="black", **kw):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, **kw)


def node(ax, cx, cy, r, text="", fc="white", ec="black", lw=0.9, fs=8, tc="black"):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc)


def save(fig, name, log):
    out = FIG_DIR / name
    # Bounded retry on the transient Windows lock (errno 22), see
    # _learnings/2026-09-28-2313_errno22-on-write-is-a-transient-lock.md.
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


def armenian_head(log, n=6):
    text = DATA_FILE.read_text(encoding="utf-8").strip()
    words = [w.strip(",.!?;:") for w in text.split()]
    if len(words) < n:
        raise ValueError(f"expected >= {n} words in {DATA_FILE}, got {len(words)}")
    log.info(f"first {n} words of the Armenian line: {words[:n]}")
    return words[:n]


# --- LSTM anatomy ------------------------------------------------------------------------
def fig_lstm_anatomy(log):
    w, h = 5.4, 2.25
    fig, ax = canvas(w, h)
    hy, gy, by = 1.92, 0.98, 0.42           # highway, gates, input bus
    gw, gh = 0.66, 0.40
    # cell highway
    arrow(ax, (0.30, hy), (5.25, hy), color=BLUE, lw=2.4, ms=10)
    ax.text(0.30, hy + 0.08, r"$\mathbf{c}^{[t-1]}$", ha="left", va="bottom", fontsize=9,
            color=BLUE)
    ax.text(5.25, hy + 0.08, r"$\mathbf{c}^{[t]}$", ha="right", va="bottom", fontsize=9,
            color=BLUE)
    ax.text(3.72, hy + 0.08, r"cell highway: only $\odot$ and + touch it", ha="center",
            va="bottom", fontsize=7, color=BLUE)
    fx, ix, cx_, ox, tx = 1.25, 2.30, 3.22, 4.12, 4.78
    for x in (fx, 2.76):                       # opaque discs, so the highway does not show through
        node(ax, x, hy, 0.12, fc="white", ec="none")
    node(ax, fx, hy, 0.12, r"$\odot$", fc=RED + "22", ec=RED, fs=9)
    node(ax, 2.76, hy, 0.12, "+", fc=GREEN + "22", ec=GREEN, fs=10)
    # gates
    gates = [(fx, "forget", r"$\mathbf{f}$: $\sigma$", RED),
             (ix, "input", r"$\mathbf{i}$: $\sigma$", GREEN),
             (cx_, "candidate", r"$\tilde{\mathbf{c}}$: tanh", GREEN),
             (ox, "output", r"$\mathbf{o}$: $\sigma$", ORANGE)]
    for x, name, lab, col in gates:
        rbox(ax, x, gy, gw, gh, "", fc=col + "1C", ec=col, lw=1.0)
        ax.text(x, gy + 0.08, lab, ha="center", va="center", fontsize=8.5)
        ax.text(x, gy - 0.10, name, ha="center", va="center", fontsize=7, color=col)
        arrow(ax, (x, by), (x, gy - gh / 2), color=GRAY, lw=0.9)          # tap from the bus
    # forget path
    arrow(ax, (fx, gy + gh / 2), (fx, hy - 0.12), color=RED, lw=1.2)
    # input path: i x candidate -> +
    node(ax, 2.76, 1.47, 0.11, r"$\odot$", fc=GREEN + "18", ec=GREEN, fs=8.5)
    arrow(ax, (ix + 0.18, gy + gh / 2), (2.76 - 0.08, 1.47 - 0.08), color=GREEN, lw=1.1)
    arrow(ax, (cx_ - 0.18, gy + gh / 2), (2.76 + 0.08, 1.47 - 0.08), color=GREEN, lw=1.1)
    arrow(ax, (2.76, 1.58), (2.76, hy - 0.12), color=GREEN, lw=1.2)
    # output path: z = o x tanh(c)
    rbox(ax, tx, 1.47, 0.52, 0.24, r"tanh", fc=BLUE + "14", ec=BLUE, lw=0.8, fs=8)
    arrow(ax, (tx, hy), (tx, 1.59), color=BLUE, lw=1.0)
    node(ax, tx, gy, 0.11, r"$\odot$", fc=ORANGE + "33", ec=ORANGE, fs=8.5)
    arrow(ax, (tx, 1.35), (tx, gy + 0.11), color=BLUE, lw=1.0)
    arrow(ax, (ox + gw / 2, gy), (tx - 0.11, gy), color=ORANGE, lw=1.1)
    rbox(ax, tx, 0.42, 0.56, 0.28, r"$\mathbf{z}^{[t]}$", fc=ORANGE + "22", ec=ORANGE, fs=9)
    arrow(ax, (tx, gy - 0.11), (tx, 0.56), color=ORANGE, lw=1.2)
    ax.text(tx + 0.33, 0.42, "out", ha="left", va="center", fontsize=7, color=ORANGE)
    # input bus: z[t-1] from the left, x[t] from below
    rbox(ax, 0.42, by, 0.62, 0.28, r"$\mathbf{z}^{[t-1]}$", fc=GREEN + "18", ec=GREEN, fs=9)
    ax.plot([0.73, ox], [by, by], color=GRAY, lw=1.6)
    rbox(ax, 1.75, 0.14, 0.50, 0.24, r"$\mathbf{x}^{[t]}$", fc=BLUE + "18", ec=BLUE, fs=9)
    ax.plot([1.75, 1.75], [0.26, by], color=GRAY, lw=1.6)
    ax.text(2.05, 0.30, "both feed every gate", ha="left", va="center", fontsize=6.5,
            color=GRAY, style="italic")
    save(fig, "lstm_anatomy.pdf", log)


# --- a gate is a soft switch -------------------------------------------------------------
def fig_gate_mask(log):
    """[2.0, -1.0] (cell) times [0.9, 0.1] (forget gate), element-wise = [1.8, -0.1]."""
    c, f = np.array([2.0, -1.0]), np.array([0.9, 0.1])
    out = c * f
    assert np.allclose(out, [1.8, -0.1]), out
    w, h = 2.7, 1.55
    fig, ax = canvas(w, h)
    cols = [(0.35, r"$\mathbf{c}^{[t-1]}$", c, BLUE, "{:+.1f}"),
            (1.12, r"$\mathbf{f}$ (gate)", f, RED, "{:.1f}"),
            (1.95, "result", out, GREEN, "{:+.1f}")]
    bw, bh = 0.5, 0.32
    for x, title, vals, col, fmt in cols:
        ax.text(x, 1.36, title, ha="center", va="center", fontsize=8.5, color=col)
        for j, v in enumerate(vals):
            y = 0.98 - j * 0.42
            alpha = (0.15 + 0.7 * v) if col == RED else 0.15
            ax.add_patch(Rectangle((x - bw / 2, y - bh / 2), bw, bh, fc=col, alpha=alpha,
                                   ec="none"))
            ax.add_patch(Rectangle((x - bw / 2, y - bh / 2), bw, bh, fill=False, ec=col, lw=0.9))
            ax.text(x, y, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if (col == RED and v > 0.6) else "black")
    ax.text(0.735, 0.77, r"$\odot$", ha="center", va="center", fontsize=12)
    ax.text(1.535, 0.77, "=", ha="center", va="center", fontsize=12)
    ax.text(2.26, 0.98, "kept", ha="left", va="center", fontsize=7, color=GREEN)
    ax.text(2.26, 0.56, "wiped", ha="left", va="center", fontsize=7, color=RED)
    ax.text(w / 2, 0.08, "memory slot 1 above, slot 2 below", ha="center", va="center",
            fontsize=6.5, color=GRAY, style="italic")
    save(fig, "gate_mask.pdf", log)


# --- gate story --------------------------------------------------------------------------
STORY_WORDS = ("The pomegranates that my grandmother sent us from Goris last autumn were "
               "sweet").split()
# ILLUSTRATIVE gates for one memory slot ("is the subject plural?"); the cell is computed.
F = [0.50, 0.10, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.10, 0.50]
I = [0.10, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10]
CAND = [0, +1, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0]     # +1 plural noun, -1 singular noun
O = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.95, 0.10, 0.10]


def story_cells():
    c, cs, zs = 0.0, [], []
    for f, i, cand, o in zip(F, I, CAND, O):
        c = f * c + i * cand
        cs.append(c)
        zs.append(o * np.tanh(c))
    return np.array(cs), np.array(zs)


def fig_gate_story(log):
    cs, zs = story_cells()
    k_pom, k_grand, k_aut, k_were = (STORY_WORDS.index(w) for w in
                                     ("pomegranates", "grandmother", "autumn", "were"))
    # the numbers the headers print, asserted
    assert round(cs[k_pom], 2) == 0.95, cs
    assert (round(cs[k_grand - 1], 2), round(cs[k_grand], 2)) == (0.89, 0.82), cs
    assert round(cs[k_aut], 2) == 0.58 and round(zs[k_aut], 2) == 0.5, (cs, zs)
    assert round(cs[k_were], 2) == 0.06, cs
    vanilla_aut = 0.95 * 0.8 ** (k_aut - k_pom)
    assert round(vanilla_aut, 2) == 0.13, vanilla_aut
    log.info(f"gate story cell: {np.round(cs, 3).tolist()}")
    log.info(f"gate story z: {np.round(zs, 3).tolist()}")

    n = len(STORY_WORDS)
    w, h = 5.5, 2.53
    x0, dx, cw = 1.02, 0.355, 0.31
    xs = [x0 + k * dx for k in range(n)]
    yc0, cscale = 1.40, 0.55
    rows = [("forget f", F, RED, 1.16), ("input i", I, GREEN, 0.94), ("output o", O, ORANGE, 0.72)]
    frames = [
        (-1, None, ["One memory slot of an LSTM, one job: is the subject plural?",
                    "Click by click: what each gate does to it."]),
        (k_pom, k_pom, ['"pomegranates": the input gate opens (0.95) and writes +1, "plural",',
                        f"into the cell: c = {cs[k_pom]:.2f}."]),
        (k_aut - 1, k_grand, ['"grandmother" is singular, but the input gate stays shut (0.05): c only '
                              f"goes {cs[k_grand - 1]:.2f} -> {cs[k_grand]:.2f}.",
                              "The forget gate holds the memory at 0.97 per word."]),
        (k_aut, k_aut, ['"autumn": the next word is the verb, so the output gate opens (0.95):',
                        f"z = 0.95 x tanh({cs[k_aut]:.2f}) = {zs[k_aut]:.2f} reaches the readout, "
                        'which picks "were", not "was".']),
        (n - 1, k_were, ['"were": the subject is used up - the forget gate drops to 0.1 and wipes '
                         f"the slot (c = {cs[k_were]:.2f}).",
                         f'Dashed: a vanilla RNN fading 0.8 per word keeps {vanilla_aut:.2f} by '
                         f'"autumn", not {cs[k_aut]:.2f}.']),
    ]
    for fi, (upto, focus, header) in enumerate(frames):
        fig, ax = canvas(w, h)
        for li, line in enumerate(header):
            ax.text(0.05, h - 0.12 - 0.17 * li, line, ha="left", va="center", fontsize=7.5,
                    fontweight="bold" if li == 0 else "normal")
        ax.text(0.05, 0.40, "illustrative\ngate values", ha="left", va="top", fontsize=6.5,
                color=GRAY, style="italic", linespacing=1.0)
        if focus is not None:
            ax.add_patch(Rectangle((xs[focus] - dx / 2, 0.60), dx, 1.40, fc="#FFF4CC",
                                   ec="none", zorder=0))
        # cell bars
        ax.text(0.05, yc0 + 0.22, "cell c", ha="left", va="center", fontsize=7.5, color=BLUE,
                fontweight="bold")
        ax.plot([x0 - dx / 2, xs[-1] + dx / 2], [yc0, yc0], color=LIGHT, lw=0.6)
        for k in range(n):
            if k > upto:
                continue
            hgt = cs[k] * cscale
            ax.add_patch(Rectangle((xs[k] - 0.11, yc0), 0.22, hgt, fc=BLUE + "AA", ec=BLUE,
                                   lw=0.5))
            ax.text(xs[k], yc0 + max(hgt, 0) + 0.03, f"{cs[k]:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=BLUE)
        if fi == len(frames) - 1:
            gx = [xs[k] for k in range(k_pom, k_aut + 1)]
            gy = [yc0 + cscale * 0.95 * 0.8 ** (k - k_pom) for k in range(k_pom, k_aut + 1)]
            ax.plot(gx, gy, color=GRAY, lw=1.0, ls="--")   # named in the header line
        # gate rows
        for name, vals, col, y in rows:
            ax.text(0.05, y, name, ha="left", va="center", fontsize=7, color=col,
                    fontweight="bold")
            for k in range(n):
                shown = k <= upto
                ax.add_patch(Rectangle((xs[k] - cw / 2, y - 0.1), cw, 0.2, lw=0.5,
                                       ec=col if shown else LIGHT,
                                       fc=col if shown else "white",
                                       alpha=(0.15 + 0.75 * vals[k]) if shown else 1.0))
                if shown:
                    ax.text(xs[k], y, f"{vals[k]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if vals[k] > 0.6 else "black")
        # words
        for k, wd in enumerate(STORY_WORDS):
            shown = k <= upto
            ax.text(xs[k] + 0.12, 0.58, wd, ha="right", va="top", rotation=30,
                    rotation_mode="anchor", fontsize=7.5,
                    color="black" if shown else "#BBBBBB",
                    fontweight="bold" if k == focus else "normal")
        save(fig, f"gate_story_{fi}.pdf", log)


# --- LSTM history: 1991 -> 2016 ----------------------------------------------------------
def fig_lstm_timeline(log):
    """Same four facts as the July TikZ frame (web-verified 2026-09-06, see [54] provenance)."""
    # (year, side, colour, label, label anchor): edge labels grow inward, never off the canvas
    events = [
        (1991, -1, RED, "the disease diagnosed:\nHochreiter's diploma thesis", "left"),
        (1997, +1, BLUE, "the cure published: Hochreiter &\nSchmidhuber, Neural Computation.\n"
                         "The field mostly shrugs - neural\nnets are out of fashion.", "center"),
        (2015, -1, GREEN, "Google voice search ships\nLSTM acoustic models", "center"),
        (2016, +1, GREEN, "Google Translate rebuilt on\nLSTMs (Wu et al.: 8 layers\n"
                          "each way); Siri and Alexa too", "right"),
    ]
    shift = {"left": -1.2, "center": 0.0, "right": 1.2}
    fig, ax = plt.subplots(figsize=(5.4, 1.85))
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    lo, hi = 1988.5, 2019.5
    ax.plot([lo, hi], [0, 0], color="#CCCCCC", lw=1.6, zorder=0)
    for year, side, col, label, ha in events:
        ax.plot([year], [0], marker="o", markersize=6, color=col, markeredgecolor="black",
                markeredgewidth=0.4, zorder=3)
        ax.plot([year, year], [0, side * 0.22], color=col, lw=0.8)
        ax.text(year, side * 0.26, str(year), ha="center", va="bottom" if side > 0 else "top",
                fontsize=8, fontweight="bold", color=col)
        ax.text(year + shift[ha], side * 0.56, label, ha=ha, va="bottom" if side > 0 else "top",
                fontsize=7, linespacing=1.05, multialignment=ha)
    ax.annotate("", xy=(2015, -0.12), xytext=(1997, -0.12),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text(2006, -0.2, "18 years", ha="center", va="top", fontsize=7.5, color=GRAY)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.35, 1.6)
    ax.axis("off")
    save(fig, "lstm_timeline.pdf", log)


# --- seq2seq -----------------------------------------------------------------------------
def staggered_words(ax, xs, words, y_hi, y_lo, fs, family=None, color="black"):
    for k, (x, wd) in enumerate(zip(xs, words)):
        ax.text(x, y_hi if k % 2 == 0 else y_lo, wd, ha="center", va="center", fontsize=fs,
                fontfamily=family, color=color)


def fig_seq2seq(log):
    arm = armenian_head(log)
    w, h = 5.5, 1.75
    fig, ax = canvas(w, h)
    sy, r = 1.0, 0.13
    xe = [0.30 + k * 0.36 for k in range(len(GLOSS_HEAD))]
    xd = [3.38 + k * 0.36 for k in range(len(arm))]
    cx = (xe[-1] + xd[0]) / 2
    for k, x in enumerate(xe):
        node(ax, x, sy, r, fc=GREEN + "22", ec=GREEN, lw=0.8)
        arrow(ax, (x, 0.70), (x, sy - r), color=BLUE, lw=0.8)
        if k:
            arrow(ax, (xe[k - 1] + r, sy), (x - r, sy), color="black", lw=0.8)
    staggered_words(ax, xe, GLOSS_HEAD, 0.58, 0.40, 7.5)
    rbox(ax, cx, sy, 0.34, 0.42, "C", fc=RED + "22", ec=RED, lw=1.2, fs=11, fontweight="bold")
    arrow(ax, (xe[-1] + r, sy), (cx - 0.17, sy), color=RED, lw=1.4)
    arrow(ax, (cx + 0.17, sy), (xd[0] - r, sy), color=RED, lw=1.4)
    ax.text(cx, sy + 0.30, "context\nvector", ha="center", va="bottom", fontsize=7, color=RED,
            linespacing=1.0)
    for k, x in enumerate(xd):
        node(ax, x, sy, r, fc=ORANGE + "33", ec=ORANGE, lw=0.8)
        arrow(ax, (x, sy + r), (x, 1.30), color=ORANGE, lw=0.8)
        if k:
            arrow(ax, (xd[k - 1] + r, sy), (x - r, sy), color="black", lw=0.8)
    staggered_words(ax, xd, arm, 1.42, 1.62, 8, family=ARM_FONT)
    # 2026-10-20 (student review): C also reaches every later decoder step (Cho et al. 2014's
    # design), matching "every step gets the same C" two frames later. Drawn under the chain.
    for x in xd[1:]:
        ax.add_patch(FancyArrowPatch((cx + 0.06, sy - 0.21), (x, sy - r), arrowstyle="-|>",
                                     mutation_scale=6, color=RED, lw=0.7, ls="--", alpha=0.7,
                                     shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.25"))
    ax.text(xd[3], 0.42, "the same C, again at every step", ha="center", va="center",
            fontsize=7, color=RED)
    ax.text((xe[0] + xe[-1]) / 2, 0.14, "encoder: reads English, word by word",
            ha="center", va="center", fontsize=7, color=GREEN, fontweight="bold")
    ax.text((xd[0] + xd[-1]) / 2, 0.14, "decoder: writes Armenian, word by word",
            ha="center", va="center", fontsize=7, color="#B07800", fontweight="bold")
    if xd[-1] + 0.3 > w:
        raise ValueError(f"decoder runs to {xd[-1]:.2f} in, canvas is {w} in")
    save(fig, "seq2seq.pdf", log)


# --- the bottleneck ----------------------------------------------------------------------
def fig_bottleneck(log):
    words = REVIEWS["long"].split()
    assert len(words) == 42, len(words)
    w, h = 5.5, 1.62
    fig, ax = canvas(w, h)
    fs, gap = 7, 0.045
    lines, cur, cur_w, max_w = [], [], 0.0, 5.0
    for wd in words:
        ww = text_w(wd, fs) + gap
        if cur and cur_w + ww > max_w:
            lines.append(cur); cur, cur_w = [], 0.0
        cur.append(wd); cur_w += ww
    lines.append(cur)
    k, y = 0, h - 0.12
    for line in lines:
        lw_ = sum(text_w(t, fs) + gap for t in line) - gap
        x = (w - lw_) / 2
        for t in line:
            alpha = 0.22 + 0.78 * (k / (len(words) - 1)) ** 1.5
            ax.text(x, y, t, ha="left", va="center", fontsize=fs, color="black", alpha=alpha,
                    fontweight="bold" if t == "disappointing" else "normal")
            x += text_w(t, fs) + gap
            k += 1
        y -= 0.16
    top = y + 0.06
    ax.add_patch(Polygon([(0.35, top), (5.15, top), (2.95, 0.50), (2.55, 0.50)], closed=True,
                         fc=BLUE + "14", ec=BLUE, lw=0.9))
    ax.text(0.35, 0.42, "faded = read early,\nthen overwritten", ha="left", va="center",
            fontsize=6.5, color=GRAY, style="italic", linespacing=1.0)
    n_c, cell = 10, 0.11
    left = 2.75 - n_c * cell / 2
    for j in range(n_c):
        ax.add_patch(Rectangle((left + j * cell, 0.24), cell, cell * 1.6, fc=RED + "44", ec=RED,
                               lw=0.6))
    arrow(ax, (2.75, 0.50), (2.75, 0.44), color=BLUE, lw=0.8)
    ax.text(left + n_c * cell + 0.1, 0.33, "C: one fixed-size vector,\nthe same size for 5 "
            "words or 500", ha="left", va="center", fontsize=7, color=RED, linespacing=1.05)
    log.info(f"bottleneck: {len(words)} words in {len(lines)} lines")
    save(fig, "bottleneck.pdf", log)


# --- the cliffhanger ---------------------------------------------------------------------
def fig_cliffhanger(log):
    arm = armenian_head(log)
    w, h = 5.5, 1.85
    fig, ax = canvas(w, h)
    sy, r = 0.95, 0.13
    xe = [0.32 + k * 0.37 for k in range(len(GLOSS_HEAD))]
    xd = [3.45 + k * 0.42 for k in range(4)]
    cx = (xe[-1] + xd[0]) / 2
    # the look-back lines first, so the nodes sit on top of them
    for x in xe:
        ax.add_patch(FancyArrowPatch((x, sy + r), (xd[3] - 0.04, sy + r), arrowstyle="-|>",
                                     mutation_scale=6, color=GRAY, lw=0.7, ls="--",
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={-0.12 - 0.02 * (xd[3] - x)}"))
    ax.text(xd[3] - 0.25, 1.62, "?", ha="center", va="center", fontsize=16, color=RED,
            fontweight="bold")
    for k, x in enumerate(xe):
        node(ax, x, sy, r, fc=GREEN + "22", ec=GREEN, lw=0.8)
        arrow(ax, (x, 0.65), (x, sy - r), color=BLUE, lw=0.8)
        if k:
            arrow(ax, (xe[k - 1] + r, sy), (x - r, sy), color="black", lw=0.8)
    staggered_words(ax, xe, GLOSS_HEAD, 0.53, 0.35, 7.5)
    rbox(ax, cx, sy, 0.30, 0.38, "C", fc=RED + "18", ec=RED, lw=1.0, fs=10)
    arrow(ax, (xe[-1] + r, sy), (cx - 0.15, sy), color=RED, lw=1.2)
    arrow(ax, (cx + 0.15, sy), (xd[0] - r, sy), color=RED, lw=1.2)
    for k, x in enumerate(xd):
        now = (k == 3)
        node(ax, x, sy, r * (1.25 if now else 1.0), fc=(ORANGE + "88") if now else (ORANGE + "33"),
             ec=ORANGE, lw=1.4 if now else 0.8)
        if k:
            arrow(ax, (xd[k - 1] + r, sy), (x - r * (1.25 if now else 1.0), sy), color="black",
                  lw=0.8)
        arrow(ax, (x, sy - r * (1.25 if now else 1.0)), (x, 0.66), color=ORANGE, lw=0.8)
    staggered_words(ax, xd[:3], arm[:3], 0.53, 0.35, 8, family=ARM_FONT)
    ax.text(xd[3], 0.35, "word 4: ?", ha="center", va="center", fontsize=7.5, color=RED,
            fontweight="bold")
    ax.text((xe[0] + xe[-1]) / 2, 0.10, "encoder states, one per English word", ha="center",
            va="center", fontsize=7, color=GREEN)
    ax.text((xd[0] + xd[-1]) / 2 + 0.1, 0.10, "decoder, about to write word 4", ha="center",
            va="center", fontsize=7, color="#B07800")
    save(fig, "cliffhanger.pdf", log)


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fig_lstm_anatomy, fig_gate_mask, fig_gate_story, fig_lstm_timeline, fig_seq2seq,
              fig_bottleneck, fig_cliffhanger):
        f(log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
