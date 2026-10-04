"""Mandatory ANIM flip-book for the L20 RNN Foundations deck ("Unrolling").

Generates into ml/13_rnns/fig/:
  unroll_0.pdf .. unroll_5.pdf -- six flip-book frames shown via \\only<n> on one
                     beamer frame. Its job is narrowed to ONE punchline: the SAME 3
                     Armenian words (py_src/data/armenian_line.txt, first 3 words),
                     fed through the SAME toy 2-d RNN as forward_pass_anim.py, forward
                     (steps 0-2) then reversed (steps 3-5) -- same bag of words,
                     different final state AND a different final score.

Toy net (LOCKED, shared with forward_pass_anim.py -- if you change these matrices
here, change them there too, or the two ANIMs will disagree):
  W = [[ 0.5, -0.5],       V = [[ 0.5, -0.5],      b = [0, 0]
       [ 1.0,  0.0],            [ 1.0,  0.0]]      U = [1, -1],  c = 0
       [ 0.0,  1.0]]

LOCKED computed values (rounded 2dp, asserted below):
  forward:  z[1]=[0.46,-0.46]  z[2]=[0.65,-0.23]  z[3]=[0.10,0.59]   score=0.38
  reversed: z[1]=[0.00,0.76]   z[2]=[0.94,0.00]    z[3]=[0.75,-0.75]  score=0.82

Font: Armenian words are set in Segoe UI (ships with Windows 11, has Armenian
coverage; Noto Sans Armenian was checked and is not installed on this machine). Any
matplotlib "Glyph ... missing from current font" warning is promoted to a hard error
below, so a missing-glyph run crashes instead of silently shipping a tofu box.

Run with the project venv:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/unroll_anim.py
"""

import logging
import sys
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
BLUE, GREEN, GRAY, RED = "#0033A0", "#008C46", "#999999", "#D90012"

# --- toy 2-d net (LOCKED, see module docstring -- shared with forward_pass_anim.py) -
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
    logger = logging.getLogger("l20_unroll_anim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    # Log lines carry Armenian text; Windows console cp1252 cannot encode it, so
    # reconfigure stdout to utf-8 and force utf-8 on the file too -- otherwise those
    # lines silently vanish from both handlers instead of failing loudly.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "unroll_anim.log", encoding="utf-8")
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


def run_rnn(vocab, order):
    """Feed vocab words in the given index order through the toy net, h0 = [0,0].

    Returns (states, raw, score); states[i] = {"word", "z"} for step i.
    """
    z = np.zeros(2)
    states = []
    for idx in order:
        x = onehot(idx)
        pre = V.T @ z + W.T @ x + B
        z = np.tanh(pre)
        states.append({"word": vocab[idx], "z": z.copy()})
    raw = float(U @ z + C)
    score = float(sigmoid(raw))
    return states, raw, score


def r2(v):
    return np.round(v, 2)


# Drawn at the size the figure gets on the slide (L20 embeds it at 0.86\linewidth, about
# 4.7 in), axes spanning the figure, data units = inches, so font sizes are true slide
# sizes (style guide: 7-10 pt). Redrawn 2026-10-03: the old 11 in canvas left ~3 pt text.
FIG_W, FIG_H = 4.7, 2.3
SLOT_X = [0.45, 1.35, 2.25]
STATE_Y, WORD_Y, R_STATE = 1.02, 0.28, 0.24
ARROW = dict(arrowstyle="-|>", mutation_scale=7)


def draw_chain(ax, states, active_idx, raw, score):
    """Draw a 3-step unrolled chain; steps > active_idx are shown empty/gray. Once
    active_idx reaches the last word (2), the U -> sigmoid -> score readout appears."""
    for i, s in enumerate(states):
        word, z = s["word"], s["z"]
        cx = SLOT_X[i]
        active = i <= active_idx
        is_cur = i == active_idx

        box_w = max(0.46, 0.085 * len(word) + 0.14)
        ax.add_patch(Rectangle((cx - box_w / 2, WORD_Y - 0.11), box_w, 0.22, ec=BLUE,
                                fc=(BLUE + "22") if active else "#F2F2F2", lw=0.9))
        ax.text(cx, WORD_Y, word, ha="center", va="center", fontsize=8.5,
                fontfamily=ARM_FONT, color="black" if active else GRAY)

        fc = GREEN + "33" if is_cur else (GREEN + "18" if active else "#F2F2F2")
        ec = GREEN if is_cur else (BLUE if active else "#CCCCCC")
        ax.add_patch(Circle((cx, STATE_Y), R_STATE, ec=ec, fc=fc, lw=1.6 if is_cur else 0.9))
        if active:
            zr = r2(z)
            fw = "bold" if is_cur else "normal"
            ax.text(cx, STATE_Y + 0.065, f"[{zr[0]:.2f},", ha="center", va="center",
                    fontsize=7, fontweight=fw)
            ax.text(cx, STATE_Y - 0.075, f" {zr[1]:.2f}]", ha="center", va="center",
                    fontsize=7, fontweight=fw)
            ax.add_patch(FancyArrowPatch((cx, WORD_Y + 0.12), (cx, STATE_Y - R_STATE - 0.01),
                                          color=BLUE, lw=1.0, **ARROW))
            ax.text(cx + 0.05, (WORD_Y + STATE_Y) / 2, "W", fontsize=7.5, color=BLUE,
                    va="center")
        else:
            ax.text(cx, STATE_Y, "?", ha="center", va="center", fontsize=9, color=GRAY)

        if i > 0:
            a, b = SLOT_X[i - 1] + R_STATE + 0.01, cx - R_STATE - 0.01
            ax.add_patch(FancyArrowPatch((a, STATE_Y), (b, STATE_Y),
                                          color=GREEN if active else "#DDDDDD",
                                          lw=1.2 if active else 0.7, **ARROW))
            if active:
                ax.text((a + b) / 2, STATE_Y + 0.04, "V", ha="center", va="bottom",
                        fontsize=7.5, color=GREEN)

    if active_idx == 2:
        x3 = SLOT_X[2]
        ux = x3 + 0.72
        a = x3 + R_STATE + 0.01
        ax.add_patch(FancyArrowPatch((a, STATE_Y), (ux - 0.19, STATE_Y), color=RED, lw=1.1,
                                      **ARROW))
        ax.text((a + ux - 0.19) / 2, STATE_Y + 0.04, "U", ha="center", va="bottom",
                fontsize=7.5, color=RED)
        ax.add_patch(FancyBboxPatch((ux - 0.18, STATE_Y - 0.10), 0.36, 0.20,
                                     boxstyle="round,pad=0.01", ec=RED, fc=RED + "18", lw=0.9))
        ax.text(ux, STATE_Y, f"{raw:.2f}", ha="center", va="center", fontsize=7.5)
        sx = ux + 0.88
        ax.add_patch(FancyArrowPatch((ux + 0.19, STATE_Y), (sx - 0.22, STATE_Y), color=RED,
                                      lw=1.1, **ARROW))
        ax.text((ux + sx) / 2, STATE_Y + 0.04, "sigmoid", ha="center", va="bottom",
                fontsize=7, color=RED)
        ax.add_patch(Circle((sx, STATE_Y), 0.21, ec=RED, fc=RED + "22", lw=1.4))
        ax.text(sx, STATE_Y, f"{score:.2f}", ha="center", va="center", fontsize=8,
                fontweight="bold")
        ax.text(sx, STATE_Y + 0.25, "score", ha="center", va="bottom", fontsize=7,
                color=GRAY)


def draw_frame(step, fwd_states, rev_states, fwd_raw, fwd_score, rev_raw, rev_score, log):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    forward = step <= 2
    states = fwd_states if forward else rev_states
    r = step if forward else step - 3
    raw, score = (fwd_raw, fwd_score) if forward else (rev_raw, rev_score)
    color = BLUE if forward else RED
    name = "Forward" if forward else "Reversed"
    mark = "" if forward else "'"
    draw_chain(ax, states, active_idx=r, raw=raw, score=score)

    z = r2(states[r]["z"])
    # (text, size, color, weight) from the top down; Armenian lines use ARM_FONT
    lines = [(f"{name}: {states[0]['word']} -> {states[1]['word']} -> {states[2]['word']}",
              8.5, color, "bold")]
    if r == 0:
        lines.append((f"z[1]{mark} = tanh(W^T x) = [{z[0]:.2f}, {z[1]:.2f}]", 7.5, "black",
                      "normal"))
    else:
        lines.append((f"z[{r + 1}]{mark} = tanh(V^T z[{r}]{mark} + W^T x + b) = "
                      f"[{z[0]:.2f}, {z[1]:.2f}]", 7.5, "black", "normal"))
    if r == 2:
        lines.append((f"{name.lower()} score = {score:.2f}", 8, color, "bold"))
    if step == 5:
        lines.append((f"Same 3 words, different order: {fwd_score:.2f} vs {rev_score:.2f}",
                      8.5, "black", "bold"))
    for k, (txt, size, col, weight) in enumerate(lines):
        ax.text(0.05, FIG_H - 0.13 - 0.18 * k, txt, ha="left", va="center", fontsize=size,
                color=col, fontweight=weight, fontfamily=ARM_FONT)

    ax.text(FIG_W - 0.04, 0.04, f"step {step + 1} of 6", ha="right", va="bottom",
            fontsize=7, color=GRAY)
    out = FIG_DIR / f"unroll_{step}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vocab = first_three_words(log)
    fwd_states, fwd_raw, fwd_score = run_rnn(vocab, order=[0, 1, 2])
    rev_states, rev_raw, rev_score = run_rnn(vocab, order=[2, 1, 0])
    log.info(f"forward: {[(s['word'], r2(s['z']).tolist()) for s in fwd_states]} "
             f"raw={fwd_raw:.4f} score={fwd_score:.4f}")
    log.info(f"reversed: {[(s['word'], r2(s['z']).tolist()) for s in rev_states]} "
             f"raw={rev_raw:.4f} score={rev_score:.4f}")

    # LOCKED assertions -- must agree with forward_pass_anim.py's forward-pass numbers.
    assert list(r2(fwd_states[0]["z"])) == [0.46, -0.46], fwd_states[0]["z"]
    assert list(r2(fwd_states[1]["z"])) == [0.65, -0.23], fwd_states[1]["z"]
    assert list(r2(fwd_states[2]["z"])) == [0.1, 0.59], fwd_states[2]["z"]
    assert round(fwd_score, 2) == 0.38, fwd_score
    assert list(r2(rev_states[0]["z"])) == [0.0, 0.76], rev_states[0]["z"]
    assert list(r2(rev_states[1]["z"])) == [0.94, 0.0], rev_states[1]["z"]
    assert list(r2(rev_states[2]["z"])) == [0.75, -0.75], rev_states[2]["z"]
    assert round(rev_score, 2) == 0.82, rev_score

    for step in range(6):
        draw_frame(step, fwd_states, rev_states, fwd_raw, fwd_score, rev_raw, rev_score, log)
    log.info("done")


if __name__ == "__main__":
    main()
