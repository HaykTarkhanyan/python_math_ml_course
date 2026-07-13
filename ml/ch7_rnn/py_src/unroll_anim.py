"""Mandatory ANIM flip-book for the L20 RNN Foundations deck ("Unrolling").

Generates into ml/ch7_rnn/fig/:
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
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/unroll_anim.py
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


def draw_chain(ax, states, active_idx, x0, title, title_color, raw, score):
    """Draw a 3-step unrolled chain; steps > active_idx are shown empty/gray. Once
    active_idx reaches the last word (2), the U -> sigmoid -> score readout appears."""
    dx = 2.8
    for i, s in enumerate(states):
        word, z = s["word"], s["z"]
        cx = x0 + i * dx
        active = i <= active_idx
        is_cur = i == active_idx

        box_w = max(1.1, 0.26 * len(word) + 0.4)
        ax.add_patch(Rectangle((cx - box_w / 2, -0.4), box_w, 0.6, ec=BLUE,
                                fc=(BLUE + "22") if active else "#F2F2F2", lw=1.3))
        ax.text(cx, -0.1, word, ha="center", va="center", fontsize=9,
                fontfamily=ARM_FONT, color="black" if active else GRAY)

        fc = GREEN + "33" if is_cur else (GREEN + "18" if active else "#F2F2F2")
        ec = GREEN if is_cur else (BLUE if active else "#CCCCCC")
        ax.add_patch(Circle((cx, 1.6), 0.62, ec=ec, fc=fc, lw=2.2 if is_cur else 1.2))
        if active:
            zr = r2(z)
            ax.text(cx, 1.72, f"[{zr[0]:.2f},", ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_cur else "normal")
            ax.text(cx, 1.46, f" {zr[1]:.2f}]", ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_cur else "normal")
        else:
            ax.text(cx, 1.6, "?", ha="center", va="center", fontsize=11, color=GRAY)

        if active:
            ax.add_patch(FancyArrowPatch((cx, 0.25), (cx, 0.94), arrowstyle="-|>",
                                          mutation_scale=12, color=BLUE, lw=1.3))
            ax.text(cx + 0.3, 0.55, "W", fontsize=8, color=BLUE)

        if i > 0:
            prev_cx = x0 + (i - 1) * dx
            recurrent_active = i <= active_idx
            ax.add_patch(FancyArrowPatch((prev_cx + 0.62, 1.6), (cx - 0.62, 1.6),
                                          arrowstyle="-|>", mutation_scale=12,
                                          color=GREEN if recurrent_active else "#DDDDDD",
                                          lw=1.6 if recurrent_active else 1.0))
            if recurrent_active:
                ax.text((prev_cx + cx) / 2, 1.88, "V", fontsize=8, color=GREEN)

    ax.text(x0 + dx, 2.7, title, ha="center", fontsize=12, color=title_color,
            fontweight="bold")

    if active_idx == 2:
        last_cx = x0 + 2 * dx
        ux = last_cx + 1.5
        ax.add_patch(FancyArrowPatch((last_cx + 0.62, 1.6), (ux - 0.5, 1.6),
                                      arrowstyle="-|>", mutation_scale=12, color=RED,
                                      lw=1.5))
        ax.text((last_cx + ux) / 2, 1.9, "U", fontsize=8, color=RED)
        ax.add_patch(FancyBboxPatch((ux - 0.5, 1.35), 1.0, 0.5, boxstyle="round,pad=0.03",
                                     ec=RED, fc=RED + "18", lw=1.2))
        ax.text(ux, 1.6, f"{raw:.2f}", ha="center", va="center", fontsize=8.5)
        sx = ux + 2.2
        ax.add_patch(FancyArrowPatch((ux + 0.5, 1.6), (sx - 0.55, 1.6), arrowstyle="-|>",
                                      mutation_scale=12, color=RED, lw=1.5))
        ax.text(ux + 0.85, 1.9, "sigmoid", fontsize=7, color=RED)
        ax.add_patch(Circle((sx, 1.6), 0.55, ec=RED, fc=RED + "22", lw=2.0))
        ax.text(sx, 1.6, f"{score:.2f}", ha="center", va="center", fontsize=9,
                fontweight="bold")
        ax.text(sx, 2.35, "score", ha="center", fontsize=7.5, color=GRAY)


def draw_frame(step, fwd_states, rev_states, fwd_raw, fwd_score, rev_raw, rev_score, log):
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_position([0, 0, 1, 1])

    if step <= 2:
        draw_chain(ax, fwd_states, active_idx=step, x0=1.0,
                   title=f"Forward: {fwd_states[0]['word']} -> {fwd_states[1]['word']} "
                         f"-> {fwd_states[2]['word']}",
                   title_color=BLUE, raw=fwd_raw, score=fwd_score)
        z = r2(fwd_states[step]["z"])
        if step == 0:
            ax.text(6.0, 4.2, f"z[1] = tanh(W^T x) = [{z[0]:.2f}, {z[1]:.2f}]",
                    ha="center", fontsize=12)
        else:
            ax.text(6.0, 4.2,
                    f"z[{step+1}] = tanh(V^T z[{step}] + W^T x + b) = [{z[0]:.2f}, {z[1]:.2f}]",
                    ha="center", fontsize=12)
        if step == 2:
            ax.text(6.0, 4.7, f"forward score = {fwd_score:.2f}", ha="center",
                    fontsize=12, fontweight="bold", color=BLUE)
    else:
        r = step - 3
        draw_chain(ax, rev_states, active_idx=r, x0=1.0,
                   title=f"Reversed: {rev_states[0]['word']} -> {rev_states[1]['word']} "
                         f"-> {rev_states[2]['word']}",
                   title_color=RED, raw=rev_raw, score=rev_score)
        z = r2(rev_states[r]["z"])
        if r == 0:
            ax.text(6.0, 4.2, f"z[1]' = tanh(W^T x) = [{z[0]:.2f}, {z[1]:.2f}]",
                    ha="center", fontsize=12)
        else:
            ax.text(6.0, 4.2,
                    f"z[{r+1}]' = tanh(V^T z[{r}]' + W^T x + b) = [{z[0]:.2f}, {z[1]:.2f}]",
                    ha="center", fontsize=12)
        if r == 2:
            ax.text(6.0, 4.7, f"reversed score = {rev_score:.2f}", ha="center",
                    fontsize=12, fontweight="bold", color=RED)
            ax.text(6.0, 5.15,
                    f"Same 3 words, different order: {fwd_score:.2f} vs {rev_score:.2f}",
                    ha="center", fontsize=12, fontweight="bold", color="black")

    ax.text(6.0, -1.1, f"step {step + 1} of 6", ha="center", fontsize=10, color=GRAY)
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1.4, 5.5)
    ax.axis("off")
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
