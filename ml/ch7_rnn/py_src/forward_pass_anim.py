"""Mandatory ANIM flip-book for the L20 RNN Foundations deck ("A complete forward pass,
in real numbers").

Generates into ml/ch7_rnn/fig/:
  forward_pass_0.pdf .. forward_pass_6.pdf -- seven flip-book frames shown via \\only<n>
                     on one beamer frame. The FIRST THREE WORDS of the chapter's
                     Armenian line (py_src/data/armenian_line.txt) are fed one at a time
                     through a toy 2-d RNN cell, with the FULL vector arithmetic shown at
                     every step: one-hot -> W^T x (a column selection) -> + V^T z[t-1] ->
                     + b -> tanh -> new state. After the third word, the U/sigmoid readout
                     turns the final state into a single score in (0,1).

Toy net (LOCKED, shared with unroll_anim.py -- if you change these matrices here, change
them there too, or the two ANIMs will disagree):
  vocab (3 words, in the order they appear in the Armenian line): word0, word1, word2
  x[t]      in R^3, one-hot
  W (3x2)   -- rows = vocab words, so W^T x[t] selects ROW i of W (= column i of W^T)
  z[t-1]    in R^2
  V (2x2)
  b in R^2, tanh activation -> z[t] in R^2
  after the last word: U (2x1), sigmoid -> score in (0,1), c = 0

  W = [[ 0.5, -0.5],       V = [[ 0.5, -0.5],      b = [0, 0]
       [ 1.0,  0.0],            [ 1.0,  0.0]]      U = [1, -1],  c = 0
       [ 0.0,  1.0]]

LOCKED computed values (rounded 2dp, asserted below -- do not hand-edit without
re-running this script and updating the "Forward pass" static frame to match):
  z[1] = [0.46, -0.46]
  z[2] = [0.65, -0.23]
  z[3] = [0.10,  0.59]
  raw  = -0.49,  score = sigmoid(raw) = 0.38

Font: Armenian words are set in Segoe UI (ships with Windows 11, has Armenian coverage;
Noto Sans Armenian was checked and is not installed on this machine). Any matplotlib
"Glyph ... missing from current font" warning is promoted to a hard error below, so a
missing-glyph run crashes instead of silently shipping a tofu box to a slide.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/forward_pass_anim.py
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
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch

# FAIL LOUDLY: a missing Armenian glyph must crash the script, never render as tofu.
warnings.filterwarnings("error", message=".*missing from current font.*")

ARM_FONT = "Segoe UI"
BLUE, GREEN, GRAY, RED, ORANGE = "#0033A0", "#008C46", "#999999", "#D90012", "#F2A800"

SEED = 509

# --- toy 2-d net (LOCKED, see module docstring) -----------------------------------
W = np.array([[0.5, -0.5], [1.0, 0.0], [0.0, 1.0]])   # 3x2, rows = vocab words
V = np.array([[0.5, -0.5], [1.0, 0.0]])               # 2x2
B = np.array([0.0, 0.0])
U = np.array([1.0, -1.0])                             # 2x1, flattened
C = 0.0

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
DATA_FILE = HERE.parent / "data" / "armenian_line.txt"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_forward_pass_anim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    # Log lines carry Armenian text; the Windows console's default codepage (cp1252)
    # cannot encode it, so reconfigure stdout to utf-8 and force utf-8 on the file too --
    # otherwise those lines silently vanish from both handlers instead of failing loudly.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "forward_pass_anim.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def first_three_words(log) -> list[str]:
    """Read the chapter's canonical Armenian line and return its first 3 words.

    Never retype the line by hand -- always read it from the data file.
    """
    text = DATA_FILE.read_text(encoding="utf-8").strip()
    words = [w.strip(",.!?;:") for w in text.split()]
    if len(words) < 3:
        raise ValueError(f"expected >= 3 words in {DATA_FILE}, got {len(words)}")
    first3 = words[:3]
    log.info(f"first three words of the Armenian line: {first3}")
    return first3


def onehot(i: int, n: int = 3) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def run_forward(vocab, log):
    """Feed the 3 words forward through the toy net; return per-step dicts + readout."""
    z = np.zeros(2)
    steps = []
    for i, word in enumerate(vocab):
        x = onehot(i)
        w_term = W.T @ x
        v_term = V.T @ z
        pre = v_term + w_term + B
        z_new = np.tanh(pre)
        steps.append({
            "word": word, "idx": i, "x": x, "z_prev": z.copy(),
            "w_term": w_term, "v_term": v_term, "pre": pre, "z": z_new.copy(),
        })
        z = z_new
    raw = float(U @ z + C)
    score = float(sigmoid(raw))
    log.info(f"steps: {[(s['word'], np.round(s['z'], 4).tolist()) for s in steps]}")
    log.info(f"raw={raw:.4f} score={score:.4f}")
    return steps, raw, score


def r2(v):
    """Round to 2dp for display -- a scalar or array."""
    return np.round(v, 2)


# --- drawing -----------------------------------------------------------------------
# Drawn at the size the figure gets on the slide (L20 embeds it at 0.82\linewidth, about
# 4.5 in), with the axes spanning the whole figure and data units = inches. Every font
# size below is therefore its true size on the slide (style guide: 7-10 pt). Redrawn
# 2026-10-03: the old 12 in canvas was shrunk to ~38% on the slide, leaving ~3 pt text.
FIG_W, FIG_H = 4.5, 2.45
SLOT_X = [0.42, 1.27, 2.12]   # x of each word slot
STATE_Y = 1.18                 # y of the state circles
WORD_Y = 0.30                  # y of the word boxes
R_STATE = 0.24                 # state circle radius
LEG_X, LEG_Y = 2.72, 0.80      # top-left of the numeric legend
ARROW = dict(arrowstyle="-|>", mutation_scale=7)


def draw_legend(ax, active_idx, show_readout):
    """Persistent numeric legend: W (rows = vocab words), V, b and, once the readout has
    appeared, U and c. The active word's row of W is highlighted - the "one-hot selects a
    row of W" visual."""
    ax.text(LEG_X, LEG_Y, "W (rows = words)", fontsize=7, fontweight="bold", va="center")
    for i, word in enumerate(VOCAB):
        y = LEG_Y - 0.14 * (i + 1)
        active = (i == active_idx)
        if active:
            ax.add_patch(Rectangle((LEG_X - 0.03, y - 0.065), 1.42, 0.13,
                                    fc=GREEN + "30", ec=GREEN, lw=0.8))
        col = "black" if active else GRAY
        ax.text(LEG_X, y, word, fontsize=7, fontfamily=ARM_FONT, color=col, va="center")
        ax.text(LEG_X + 0.52, y, f"[{W[i, 0]:+.2f}, {W[i, 1]:+.2f}]", fontsize=7,
                family="monospace", color=col, va="center")
    y = LEG_Y - 0.14 * 4
    ax.text(LEG_X, y, f"V = [[{V[0, 0]:+.1f},{V[0, 1]:+.1f}],[{V[1, 0]:+.1f},{V[1, 1]:+.1f}]]",
            fontsize=7, family="monospace", va="center")
    tail = f"  U = [{U[0]:+.0f},{U[1]:+.0f}]  c = 0" if show_readout else ""
    ax.text(LEG_X, y - 0.14, "b = [0,0]" + tail, fontsize=7, family="monospace", va="center")


def draw_word_slot(ax, cx, word, active, is_current):
    box_w = max(0.46, 0.085 * len(word) + 0.14)
    fc = (BLUE + "22") if active else "#F2F2F2"
    ax.add_patch(Rectangle((cx - box_w / 2, WORD_Y - 0.11), box_w, 0.22, ec=BLUE, fc=fc,
                            lw=1.1 if is_current else 0.8))
    ax.text(cx, WORD_Y, word, ha="center", va="center", fontsize=8.5, fontfamily=ARM_FONT,
            color="black" if active else GRAY)


def draw_state_circle(ax, cx, z, active, is_current, label):
    fc = GREEN + "33" if is_current else (GREEN + "18" if active else "#F2F2F2")
    ec = GREEN if is_current else (BLUE if active else "#CCCCCC")
    ax.add_patch(Circle((cx, STATE_Y), R_STATE, ec=ec, fc=fc, lw=1.6 if is_current else 0.9))
    if active:
        zr = r2(z)
        fw = "bold" if is_current else "normal"
        ax.text(cx, STATE_Y + 0.065, f"[{zr[0]:.2f},", ha="center", va="center",
                fontsize=7, fontweight=fw)
        ax.text(cx, STATE_Y - 0.075, f" {zr[1]:.2f}]", ha="center", va="center",
                fontsize=7, fontweight=fw)
    else:
        ax.text(cx, STATE_Y, "?", ha="center", va="center", fontsize=9, color=GRAY)
    ax.text(cx, STATE_Y + R_STATE + 0.04, label, ha="center", va="bottom", fontsize=9,
            color=GRAY)


def draw_chain(ax, steps, upto):
    """upto = index of last word whose slot is 'active' (-1 means none yet)."""
    for i in range(3):
        active = i <= upto
        is_current = (i == upto)
        x = SLOT_X[i]
        draw_word_slot(ax, x, steps[i]["word"] if active else "?", active, is_current)
        draw_state_circle(ax, x, steps[i]["z"] if active else None, active, is_current,
                          f"$z^{{[{i + 1}]}}$")
        ax.add_patch(FancyArrowPatch((x, WORD_Y + 0.12), (x, STATE_Y - R_STATE - 0.01),
                                      color=BLUE if active else "#DDDDDD",
                                      lw=1.0 if active else 0.7, **ARROW))
        if active:
            ax.text(x + 0.05, (WORD_Y + STATE_Y) / 2, "W", fontsize=7.5, color=BLUE,
                    va="center")
        if i > 0:
            a, b = SLOT_X[i - 1] + R_STATE + 0.01, x - R_STATE - 0.01
            ax.add_patch(FancyArrowPatch((a, STATE_Y), (b, STATE_Y),
                                          color=GREEN if active else "#DDDDDD",
                                          lw=1.2 if active else 0.7, **ARROW))
            if active:
                ax.text((a + b) / 2, STATE_Y + 0.04, "V", ha="center", va="bottom",
                        fontsize=7.5, color=GREEN)
    ax.text(SLOT_X[1] + 0.25, 0.04, "tanh applied at every state", ha="center",
            va="bottom", fontsize=7, color=GRAY, style="italic")


def draw_readout(ax, raw, score, stage):
    """stage: 'raw' shows only U -> raw box; 'score' also shows sigmoid -> final score."""
    x3 = SLOT_X[2]
    ux = x3 + 0.70                       # centre of the raw box
    a = x3 + R_STATE + 0.01
    ax.add_patch(FancyArrowPatch((a, STATE_Y), (ux - 0.19, STATE_Y), color=RED, lw=1.1,
                                  **ARROW))
    ax.text((a + ux - 0.19) / 2, STATE_Y + 0.04, "U", ha="center", va="bottom",
            fontsize=7.5, color=RED)
    ax.add_patch(FancyBboxPatch((ux - 0.18, STATE_Y - 0.10), 0.36, 0.20,
                                 boxstyle="round,pad=0.01", ec=RED, fc=RED + "18", lw=0.9))
    ax.text(ux, STATE_Y, f"{raw:.2f}", ha="center", va="center", fontsize=7.5)
    ax.text(ux, STATE_Y + 0.13, "raw", ha="center", va="bottom", fontsize=7, color=GRAY)
    if stage == "score":
        sx = ux + 0.86                   # centre of the score circle
        ax.add_patch(FancyArrowPatch((ux + 0.19, STATE_Y), (sx - 0.22, STATE_Y), color=RED,
                                      lw=1.1, **ARROW))
        ax.text((ux + sx) / 2, STATE_Y + 0.04, "sigmoid", ha="center", va="bottom",
                fontsize=7, color=RED)
        ax.add_patch(Circle((sx, STATE_Y), 0.21, ec=RED, fc=RED + "22", lw=1.4))
        ax.text(sx, STATE_Y, f"{score:.2f}", ha="center", va="center", fontsize=8,
                fontweight="bold")
        ax.text(sx, STATE_Y + 0.25, "score in (0, 1)", ha="center", va="bottom", fontsize=7,
                color=GRAY)


def header(ax, lines, boxed_last=False, size=7.5, **kw):
    """Left-aligned lines from the top of the figure; optionally box the last one."""
    for k, line in enumerate(lines):
        box = dict(boxstyle="round,pad=0.15", fc=GREEN + "12", ec=GREEN, lw=0.8)
        ax.text(0.05, FIG_H - 0.13 - 0.18 * k, line, ha="left", va="center", fontsize=size,
                fontfamily=ARM_FONT, bbox=box if (boxed_last and k == len(lines) - 1) else None,
                **kw)


def draw_frame(step, steps, raw, score, log):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    if step == 0:
        draw_chain(ax, steps, upto=-1)
        draw_legend(ax, active_idx=None, show_readout=False)
        ax.text(0.05, FIG_H - 0.13, "Toy net ready: 3-word vocabulary, 2-d state, "
                "$z^{[0]}=[0, 0]$", ha="left", va="center", fontsize=8, color=BLUE,
                fontweight="bold")
    elif step in (1, 2, 3):
        i = step - 1
        s = steps[i]
        draw_chain(ax, steps, upto=i)
        draw_legend(ax, active_idx=i, show_readout=False)
        wr = r2(s["w_term"]); vr = r2(s["v_term"]); pr = r2(s["pre"]); zr = r2(s["z"])
        # Plain text only (no mathtext "$...$"): mixing mathtext with an embedded Armenian
        # word silently drops the Armenian part instead of erroring.
        lines = [f"x[{i + 1}] = one-hot({s['word']})  ->  W^T x = row of W = "
                 f"[{wr[0]:.2f}, {wr[1]:.2f}]"]
        if i > 0:
            lines.append(f"+ V^T z[{i}] = [{vr[0]:.2f}, {vr[1]:.2f}]   + b = [0, 0]")
        else:
            lines.append("+ V^T z[0] = [0.00, 0.00] (state starts at zero)   + b = [0, 0]")
        lines.append(f"pre = [{pr[0]:.2f}, {pr[1]:.2f}]  ->  tanh  ->  "
                     f"z[{i + 1}] = [{zr[0]:.2f}, {zr[1]:.2f}]")
        header(ax, lines, boxed_last=True)
    elif step == 4:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=None, show_readout=True)
        draw_readout(ax, raw, score, stage="raw")
        ax.text(0.05, FIG_H - 0.13, "After the last word: readout $U^\\top z^{[3]} + c$",
                ha="left", va="center", fontsize=8, color=RED, fontweight="bold")
        ax.text(0.05, FIG_H - 0.31, "(it happens once, not at every step)", ha="left",
                va="center", fontsize=7.5, color=RED)
    elif step == 5:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=None, show_readout=True)
        draw_readout(ax, raw, score, stage="score")
        ax.text(0.05, FIG_H - 0.13, f"sigmoid({raw:.2f}) = {score:.2f}: one pass, one score",
                ha="left", va="center", fontsize=8, fontweight="bold")
    elif step == 6:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=0, show_readout=True)
        draw_readout(ax, raw, score, stage="score")
        ax.add_patch(Rectangle((SLOT_X[0] - 0.37, WORD_Y - 0.16), 0.74,
                                STATE_Y + R_STATE + 0.24 - (WORD_Y - 0.16), fill=False,
                                ec=ORANGE, lw=1.4, linestyle="--"))
        header(ax, [f"one-hot x W selects ONE row of W: that row IS "
                    f"{steps[0]['word']}'s own vector"], size=8, color=ORANGE,
               fontweight="bold")
        ax.text(0.05, FIG_H - 0.31, "The attention chapter gives this row a name: an "
                "embedding.", ha="left", va="center", fontsize=7.5, color=GRAY,
                style="italic")

    ax.text(FIG_W - 0.04, FIG_H - 0.04, f"step {step + 1} of 7", ha="right", va="top",
            fontsize=7, color=GRAY)
    out = FIG_DIR / f"forward_pass_{step}.pdf"
    # Writing seven PDFs back to back hits a transient Windows lock (OSError errno 22) on
    # one of them in most runs; see _learnings/2026-09-28-2313_errno22-on-write-is-a-
    # transient-lock.md. Bounded retry, then fail loudly.
    for attempt in range(1, 6):
        try:
            fig.savefig(out)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out} after {attempt} attempts: {e}")
                raise
            log.warning(f"write of {out} failed (errno 22), retry {attempt}/4 in 1 s")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {out}")


VOCAB = None  # set in main() after reading the data file


def main():
    global VOCAB
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB = first_three_words(log)
    steps, raw, score = run_forward(VOCAB, log)

    # LOCKED assertions -- if these fail, the matrices changed; update the docstring,
    # the static "Forward pass" summary frame, and unroll_anim.py's shared net together.
    assert list(r2(steps[0]["z"])) == [0.46, -0.46], steps[0]["z"]
    assert list(r2(steps[1]["z"])) == [0.65, -0.23], steps[1]["z"]
    assert list(r2(steps[2]["z"])) == [0.1, 0.59], steps[2]["z"]
    assert round(raw, 2) == -0.49, raw
    assert round(score, 2) == 0.38, score

    for step in range(7):
        draw_frame(step, steps, raw, score, log)
    log.info("done")


if __name__ == "__main__":
    main()
