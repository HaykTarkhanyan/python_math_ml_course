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

SLOT_X = [1.6, 5.4, 9.2]   # x-position of each word slot
STATE_Y = 1.6
WORD_Y = -0.9


def draw_legend(ax, active_idx, show_readout):
    """Persistent small numeric legend: W (rows = vocab words), V, b, and, once the
    readout has appeared, U and c. The active word's row of W is highlighted -- this
    is the "one-hot selects a row of W" visual."""
    x0, y0 = 10.9, 4.9
    ax.text(x0, y0, "W (rows = words)", fontsize=7.5, color="black", fontweight="bold")
    for i, word in enumerate(VOCAB):
        y = y0 - 0.36 * (i + 1)
        row = W[i]
        active = (i == active_idx)
        if active:
            ax.add_patch(Rectangle((x0 - 0.08, y - 0.16), 3.35, 0.32,
                                    fc=GREEN + "30", ec=GREEN, lw=1.1))
        ax.text(x0, y, word, fontsize=7.2, fontfamily=ARM_FONT,
                color="black" if active else GRAY)
        ax.text(x0 + 1.15, y, f"[{row[0]:+.2f}, {row[1]:+.2f}]", fontsize=7.2,
                family="monospace", color="black" if active else GRAY)
    y = y0 - 0.36 * 4 - 0.12
    ax.text(x0, y, f"V = [[{V[0,0]:+.1f},{V[0,1]:+.1f}], [{V[1,0]:+.1f},{V[1,1]:+.1f}]]"
            f"   b = [0, 0]", fontsize=6.8, family="monospace", color="black")
    if show_readout:
        y2 = y - 0.32
        ax.text(x0, y2, f"U = [{U[0]:+.1f}, {U[1]:+.1f}]   c = 0", fontsize=6.8,
                family="monospace", color="black")


def draw_word_slot(ax, cx, word, active, is_current):
    box_w = max(1.3, 0.24 * len(word) + 0.35)
    fc = (BLUE + "22") if active else "#F2F2F2"
    ax.add_patch(Rectangle((cx - box_w / 2, WORD_Y - 0.3), box_w, 0.6, ec=BLUE, fc=fc,
                            lw=1.3 if is_current else 1.0))
    ax.text(cx, WORD_Y, word, ha="center", va="center", fontsize=10, fontfamily=ARM_FONT,
            color="black" if active else GRAY)


def draw_state_circle(ax, cx, z, active, is_current, label):
    fc = GREEN + "33" if is_current else (GREEN + "18" if active else "#F2F2F2")
    ec = GREEN if is_current else (BLUE if active else "#CCCCCC")
    ax.add_patch(Circle((cx, STATE_Y), 0.68, ec=ec, fc=fc, lw=2.2 if is_current else 1.2))
    if active:
        zr = r2(z)
        ax.text(cx, STATE_Y + 0.08, f"[{zr[0]:.2f},", ha="center", va="center",
                fontsize=8.5, fontweight="bold" if is_current else "normal")
        ax.text(cx, STATE_Y - 0.18, f" {zr[1]:.2f}]", ha="center", va="center",
                fontsize=8.5, fontweight="bold" if is_current else "normal")
    else:
        ax.text(cx, STATE_Y, "?", ha="center", va="center", fontsize=13, color=GRAY)
    ax.text(cx, STATE_Y + 1.0, label, ha="center", fontsize=8, color=GRAY)


def draw_chain(ax, steps, upto):
    """upto = index of last word whose slot is 'active' (-1 means none yet)."""
    for i in range(3):
        active = i <= upto
        is_current = (i == upto)
        word = steps[i]["word"] if active else "?"
        draw_word_slot(ax, SLOT_X[i], word, active, is_current)
        z = steps[i]["z"] if active else None
        draw_state_circle(ax, SLOT_X[i], z, active, is_current, f"$z^{{[{i+1}]}}$")
        # W arrow (word -> state)
        arrow_color = BLUE if active else "#DDDDDD"
        ax.add_patch(FancyArrowPatch((SLOT_X[i], WORD_Y + 0.32), (SLOT_X[i], STATE_Y - 0.72),
                                      arrowstyle="-|>", mutation_scale=12,
                                      color=arrow_color, lw=1.4 if active else 1.0))
        if active:
            ax.text(SLOT_X[i] + 0.32, (WORD_Y + STATE_Y) / 2, "W", fontsize=9, color=BLUE)
        # V arrow (previous state -> this state)
        if i > 0:
            recurrent_active = i <= upto
            ax.add_patch(FancyArrowPatch((SLOT_X[i - 1] + 0.72, STATE_Y),
                                          (SLOT_X[i] - 0.72, STATE_Y),
                                          arrowstyle="-|>", mutation_scale=12,
                                          color=GREEN if recurrent_active else "#DDDDDD",
                                          lw=1.7 if recurrent_active else 1.0))
            if recurrent_active:
                ax.text((SLOT_X[i - 1] + SLOT_X[i]) / 2, STATE_Y + 0.35, "V", fontsize=9,
                        color=GREEN)
    ax.text(SLOT_X[2] - 1.0, STATE_Y - 1.6, "tanh applied at every state", fontsize=7.5,
            color=GRAY, style="italic")


def draw_readout(ax, steps, raw, score, stage):
    """stage: 'raw' shows only U -> raw box; 'score' also shows sigmoid -> final score."""
    z3 = steps[2]["z"]
    ux, uy = SLOT_X[2] + 1.55, STATE_Y
    ax.add_patch(FancyArrowPatch((SLOT_X[2] + 0.72, STATE_Y), (ux - 0.55, STATE_Y),
                                  arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.6))
    ax.text((SLOT_X[2] + ux) / 2, STATE_Y + 0.32, "U", fontsize=9, color=RED)
    ax.add_patch(FancyBboxPatch((ux - 0.55, STATE_Y - 0.3), 1.1, 0.6,
                                 boxstyle="round,pad=0.03", ec=RED, fc=RED + "18", lw=1.3))
    ax.text(ux, STATE_Y, f"{raw:.2f}", ha="center", va="center", fontsize=9)
    ax.text(ux, STATE_Y + 0.55, "raw", ha="center", fontsize=7.5, color=GRAY)

    if stage == "score":
        sx = ux + 1.75
        ax.add_patch(FancyArrowPatch((ux + 0.55, STATE_Y), (sx - 0.6, STATE_Y),
                                      arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.6))
        ax.text((ux + sx) / 2, STATE_Y + 0.32, "sigmoid", fontsize=8, color=RED)
        ax.add_patch(Circle((sx, STATE_Y), 0.62, ec=RED, fc=RED + "22", lw=2.0))
        ax.text(sx, STATE_Y, f"{score:.2f}", ha="center", va="center", fontsize=10,
                fontweight="bold")
        ax.text(sx, STATE_Y + 0.85, "score $\\in(0,1)$", ha="center", fontsize=7.5,
                color=GRAY)


def draw_frame(step, steps, raw, score, log):
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-1.0, 15.0)
    ax.set_ylim(-2.6, 5.4)
    ax.axis("off")

    if step == 0:
        draw_chain(ax, steps, upto=-1)
        draw_legend(ax, active_idx=None, show_readout=False)
        ax.text(6.0, 4.4, "Toy net ready: 3-word vocab, 2-d state, $z^{[0]}=[0,0]$",
                ha="center", fontsize=12, color=BLUE, fontweight="bold")
    elif step in (1, 2, 3):
        i = step - 1
        s = steps[i]
        draw_chain(ax, steps, upto=i)
        draw_legend(ax, active_idx=i, show_readout=False)
        wr = r2(s["w_term"]); vr = r2(s["v_term"]); pr = r2(s["pre"]); zr = r2(s["z"])
        # Plain text only (no mathtext "$...$") -- mixing mathtext with an embedded
        # Armenian word silently drops the Armenian portion instead of erroring, so
        # these annotations use the same plain arithmetic notation as unroll_anim.py.
        lines = [
            f"x[{i+1}] = one-hot({s['word']}) -> W^T x = row of W = "
            f"[{wr[0]:.2f}, {wr[1]:.2f}]",
        ]
        if i > 0:
            lines.append(f"+ V^T z[{i}] = [{vr[0]:.2f}, {vr[1]:.2f}]   + b = [0, 0]")
        else:
            lines.append("+ V^T z[0] = [0.00, 0.00]  (state starts at zero)  + b = [0, 0]")
        lines.append(f"pre = [{pr[0]:.2f}, {pr[1]:.2f}]  ->  tanh  ->  "
                      f"z[{i+1}] = [{zr[0]:.2f}, {zr[1]:.2f}]")
        for k, line in enumerate(lines):
            ax.text(6.0, 4.55 - 0.42 * k, line, ha="center", fontsize=11.5,
                    fontfamily=ARM_FONT,
                    bbox=dict(boxstyle="round", fc=GREEN + "12", ec=GREEN) if k == 2 else None)
    elif step == 4:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=None, show_readout=True)
        draw_readout(ax, steps, raw, score, stage="raw")
        ax.text(6.0, 4.4, "After the last word: readout $U^\\top z^{[3]} + c$"
                " (happens once, not every step)", ha="center", fontsize=11.5, color=RED)
    elif step == 5:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=None, show_readout=True)
        draw_readout(ax, steps, raw, score, stage="score")
        ax.text(6.0, 4.4, f"sigmoid({raw:.2f}) = {score:.2f} - one pass, one score",
                ha="center", fontsize=12, fontweight="bold", color="black")
    elif step == 6:
        draw_chain(ax, steps, upto=2)
        draw_legend(ax, active_idx=0, show_readout=True)
        draw_readout(ax, steps, raw, score, stage="score")
        ax.add_patch(Rectangle((SLOT_X[0] - 0.9, WORD_Y - 0.5), 1.8, 3.0, fill=False,
                                ec=ORANGE, lw=2.0, linestyle="--"))
        ax.text(6.0, 4.4, "one-hot x W selects ONE row of W - that row IS "
                f"{steps[0]['word']}'s own vector", ha="center", fontsize=11.5,
                color=ORANGE, fontweight="bold", fontfamily=ARM_FONT)
        ax.text(6.0, -2.3, "Next lecture: this row/column gets a name - an embedding.",
                ha="center", fontsize=10, color=GRAY, style="italic")

    ax.text(6.0, 5.1, f"step {step + 1} of 7", ha="center", fontsize=10, color=GRAY)
    out = FIG_DIR / f"forward_pass_{step}.pdf"
    fig.savefig(out)
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
