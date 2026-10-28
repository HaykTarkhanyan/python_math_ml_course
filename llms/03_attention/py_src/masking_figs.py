"""Figures for LLM-3's masking section (llms/03_attention/LLM3_attention.tex, added 2026-10-28):
the causal mask, causal language modeling (CLM), masked language modeling (MLM) and the padding mask.

Toy, demonstrative numbers (ml/SLIDE_STYLE.md). One word = one token here, on the deck's running
sentence "a fluffy blue creature roamed the verdant forest". The creature grid reuses the toy from
attention_figs.py, so the masked weights are the same ones LLM-4 used to show. Every number the slides
quote is asserted (QUOTED); save() (from attention_figs) refuses text outside a figure.

Generates into llms/03_attention/fig/:
  mask_two_games.pdf     -- predict: the blank at the end vs the blank in the middle
  mask_clm_shift.pdf     -- CLM: inputs and targets shifted by one; position 4 sees only 4 words
  mask_cheat.pdf         -- no mask: a head that copies the next word sees its own answer
  mask_steps.pdf         -- scores -> + minus infinity above the diagonal -> softmax: exact zeros
  mask_creature_grid.pdf -- the learned creature grid, without and with the causal mask
  mask_renorm.pdf        -- zero after softmax (sum < 1) vs minus infinity before (sum 1, same ratios)
  mask_one_pass.pdf      -- the staircase: one pass, 7 prefixes, 7 predictions
  mask_mlm.pdf           -- MLM: [MASK] in the input, both sides visible, a loss only at [MASK]
  mask_clm_vs_mlm.pdf    -- the two grids side by side: who may look, where the loss is
  mask_padding.pdf       -- a batch padded with [PAD]; pad keys masked, pad rows ignored
  mask_two_kinds.pdf     -- the trap: the [MASK] token (input) vs the attention mask (who looks at whom)

Run (~25 s):
  ./ma/Scripts/python.exe llms/03_attention/py_src/masking_figs.py
"""
import json
import logging
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from attention_figs import (ARMBLUE, ARMORANGE, ARMRED, FIG, FS, FS_SMALL, FS_TINY, GREEN_D, GREY, INK,
                            LIGHT, LOGS, RESULTS, SEED, TINT_RED, TOKENS, W, W_K1, W_Q1, X, arrow, canvas,
                            check, num, r2, save, softmax_rows)

WORDS = ["a", "fluffy", "blue", "creature", "roamed", "the", "verdant", "forest"]
ALLOWED_FC = "#C9D6EE"
BLOCKED_FC = "#EEEEEE"
TINT_GREEN = "#D5EBDD"

QUOTED = {
    "creature_masked": {"fluffy": [1, 0, 0], "blue": [0.5, 0.5, 0], "creature": [0.47, 0.47, 0.06]},
    # mask_steps / mask_renorm: row "fluffy" of the 5-word grid (82% of its weight sits on the future)
    "renorm_fluffy": {"zero_after": [0.06, 0.12, 0, 0, 0], "zero_after_sum": 0.18,
                      "masked": [0.33, 0.67, 0, 0, 0], "future_share": 0.82, "roamed": 0.54},
    "one_pass": 7,
}


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("masking_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "masking_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


# ----------------------------------------------------------------------------- helpers
def wbox(ax, x, y, text, w=0.52, h=0.24, fc="#F2F2F2", ec="#999999", color=INK, bold=False, hatch=None,
         fs=FS_TINY):
    """A word as a box; (x, y) is the bottom-left corner. Returns the box centre."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.7, hatch=hatch))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal")
    return x + w / 2, y + h / 2


def mgrid(ax, x0, y_top, cell, allowed, values=None, fmt=num, blocked_text="", row_labels=None,
          col_labels=None, row_colors=None, hl_rows=(), dim_rows=()):
    """An attention grid: allowed cells tinted (by value when given), blocked cells grey and hatched.
    Rows are queries (who looks), columns keys (who is looked at)."""
    cm = plt.get_cmap("Blues")
    n_r, n_c = allowed.shape
    for i in range(n_r):
        for j in range(n_c):
            x, y = x0 + j * cell, y_top - (i + 1) * cell
            if i in dim_rows:
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="#F7F7F7", edgecolor="white", lw=0.8))
            elif allowed[i, j]:
                fc = cm(0.08 + 0.5 * values[i, j]) if values is not None else ALLOWED_FC
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=fc, edgecolor="white", lw=0.8))
                if values is not None and fmt is not None:
                    ax.text(x + cell / 2, y + cell / 2, fmt(values[i, j]), ha="center", va="center",
                            fontsize=FS_TINY, color=INK)
            else:
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=BLOCKED_FC, edgecolor="white", lw=0.8,
                                       hatch="////"))
                if blocked_text:
                    ax.text(x + cell / 2, y + cell / 2, blocked_text, ha="center", va="center",
                            fontsize=FS_TINY, color=GREY)
    for i in hl_rows:
        ax.add_patch(Rectangle((x0, y_top - (i + 1) * cell), n_c * cell, cell, fill=False, edgecolor=ARMRED,
                               lw=1.2, zorder=5))
    if row_labels:
        for i, lab in enumerate(row_labels):
            c = (row_colors or {}).get(i, ARMRED if i in hl_rows else INK)
            ax.text(x0 - 0.04, y_top - (i + 0.5) * cell, lab, ha="right", va="center", fontsize=FS_TINY, color=c)
    if col_labels:
        for j, lab in enumerate(col_labels):
            ax.text(x0 + (j + 0.5) * cell - 0.03, y_top + 0.03, lab, ha="left", va="bottom", fontsize=FS_TINY,
                    rotation=55, rotation_mode="anchor")
    return x0 + n_c * cell, y_top - n_r * cell


def causal(n):
    return np.tril(np.ones((n, n), dtype=bool))


def five_scores():
    """The 5-word grid's scores (demonstrative, seed 509, one decimal so the slide can print them)."""
    rng = np.random.default_rng(SEED)
    return np.round(rng.normal(size=(5, 5)), 1) + 0.0      # + 0.0 turns -0.0 into 0.0


# ----------------------------------------------------------------------------- figures
def fig_two_games(log):
    fig, ax = canvas(1.0)
    games = [("game 1", 7), ("game 2", 2)]                     # which position is blank
    for r, (name, blank) in enumerate(games):
        y = 0.6 - r * 0.42
        ax.text(0.62, y + 0.12, name, ha="right", va="center", fontsize=FS_SMALL, fontweight="bold")
        for j, wd in enumerate(WORDS):
            x = 0.75 + j * 0.58
            if j == blank:
                wbox(ax, x, y, "?", fc=TINT_RED, ec=ARMRED, color=ARMRED, bold=True, fs=FS_SMALL)
            else:
                wbox(ax, x, y, wd)
    save(fig, "mask_two_games.pdf", log)


def fig_clm_shift(log):
    fig, ax = canvas(1.55)
    x0, step = 1.05, 0.6
    ax.text(x0 - 0.1, 0.47, "input", ha="right", va="center", fontsize=FS_SMALL)
    ax.text(x0 - 0.1, 1.12, "target:\nthe next word", ha="right", va="center", fontsize=FS_SMALL, color=GREEN_D)
    for j in range(7):
        x = x0 + j * step
        seen = j <= 3
        wbox(ax, x, 0.35, WORDS[j], fc="#DCE5F5" if seen else "#F2F2F2", ec=ARMBLUE if seen else "#999999")
        wbox(ax, x, 1.0, WORDS[j + 1], fc=TINT_GREEN, ec=GREEN_D, color=GREEN_D,
             bold=(j == 3))
        arrow(ax, (x + 0.26, 0.61), (x + 0.26, 0.98), color=ARMRED if j == 3 else LIGHT, lw=1.2 if j == 3 else 0.7)
    xb0, xb1 = x0, x0 + 3 * step + 0.52
    ax.plot([xb0, xb0, xb1, xb1], [0.3, 0.24, 0.24, 0.3], color=ARMBLUE, lw=0.8)
    ax.text((xb0 + xb1) / 2, 0.19, 'to predict "roamed", position 4 may use only these four words',
            ha="center", va="top", fontsize=FS_TINY, color=ARMBLUE)
    ax.text(x0 + 3.5 * step, 1.42, "8 words, 7 predictions: every position predicts the word after it",
            ha="center", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, "mask_clm_shift.pdf", log)


def fig_cheat(R, log):
    n = 7
    A = np.full((n, n), 0.1 / (n - 1))
    for i in range(n - 1):
        A[i, i + 1] = 0.9
    A[n - 1] = 1 / n
    if not np.allclose(A.sum(1), 1):
        raise AssertionError("cheat grid rows must sum to 1")
    fig, ax = canvas(2.15, 2.75)
    cell, x0, top = 0.18, 0.95, 1.68
    words = WORDS[:n]
    mgrid(ax, x0, top, cell, np.ones((n, n), bool), A, fmt=lambda v: ".9" if v > 0.5 else "",
          row_labels=words, col_labels=words, hl_rows=(2,))
    ax.add_patch(Rectangle((x0 + 3 * cell, top - 3 * cell), cell, cell, fill=False, edgecolor=ARMRED, lw=1.8,
                           zorder=6))
    ax.text(1.375, top - n * cell - 0.06, "no mask: a head that copies the next word.\n"
            '"blue" must predict "creature": it is right there', ha="center", va="top", fontsize=FS_TINY,
            color=ARMRED, linespacing=1.3)
    save(fig, "mask_cheat.pdf", log)


def fig_steps(R, log):
    S = five_scores()
    allowed = causal(5)
    masked = np.where(allowed, S, -np.inf)
    A = softmax_rows(masked)
    if float(np.triu(A, k=1).max()) != 0.0 or not np.allclose(A.sum(1), 1):
        raise AssertionError("mask_steps: future weights must be exactly 0 and rows sum to 1")
    R["steps"] = {"scores": S.tolist(), "weights": A.tolist()}
    words = WORDS[:5]
    fig, ax = canvas(2.1, 4.6)                               # drawn at its slot: 0.8 of the text width
    cell, top = 0.24, 1.68
    xs = [0.55, 1.95, 3.3]
    titles = ["1. scores  $QK^\\top/\\sqrt{d_k}$", "2. add the mask", "3. softmax by rows"]
    sfmt = lambda v: f"{v:.1f}".replace("-", "−")
    mgrid(ax, xs[0], top, cell, np.ones((5, 5), bool), np.clip((S + 2.5) / 5, 0, 1), fmt=None,
          row_labels=words, col_labels=words)
    for i in range(5):                                       # the scores printed, not the colour scale
        for j in range(5):
            ax.text(xs[0] + (j + 0.5) * cell, top - (i + 0.5) * cell, sfmt(S[i, j]), ha="center",
                    va="center", fontsize=FS_TINY)
    mgrid(ax, xs[1], top, cell, allowed, np.clip((S + 2.5) / 5, 0, 1), fmt=None, blocked_text="−∞",
          col_labels=words)
    for i in range(5):
        for j in range(i + 1):
            ax.text(xs[1] + (j + 0.5) * cell, top - (i + 0.5) * cell, sfmt(S[i, j]), ha="center",
                    va="center", fontsize=FS_TINY)
    mgrid(ax, xs[2], top, cell, allowed, A, blocked_text="0", col_labels=words)
    for x, t in zip(xs, titles):
        ax.text(x + 2.5 * cell, top - 5 * cell - 0.07, t, ha="center", va="top", fontsize=FS_SMALL)
    for x in xs[:2]:
        arrow(ax, (x + 5 * cell + 0.03, top - 2.5 * cell), (x + 5 * cell + 0.17, top - 2.5 * cell))
    save(fig, "mask_steps.pdf", log)
    return S


def fig_creature_grid(R, log):
    q, k = X @ W_Q1, X @ W_K1
    S = np.outer(q, k)
    A = softmax_rows(S)
    Am = softmax_rows(np.where(causal(3), S, -np.inf))
    for i, t in enumerate(TOKENS):
        check(f"creature grid, masked, {t}", Am[i], QUOTED["creature_masked"][t])
    check("creature row unchanged", A[2], Am[2])
    R["creature_masked"] = Am.tolist()
    fig, ax = canvas(1.75, 2.75)
    cell, top = 0.27, 1.1
    mgrid(ax, 0.62, top, cell, np.ones((3, 3), bool), A, col_labels=TOKENS, row_labels=TOKENS, hl_rows=(0,))
    mgrid(ax, 1.62, top, cell, causal(3), Am, blocked_text="0", col_labels=TOKENS, hl_rows=(0,))
    ax.text(0.62 + 1.5 * cell, top - 3 * cell - 0.06, "no mask", ha="center", va="top", fontsize=FS_SMALL)
    ax.text(1.62 + 1.5 * cell, top - 3 * cell - 0.06, "causal mask", ha="center", va="top", fontsize=FS_SMALL)
    save(fig, "mask_creature_grid.pdf", log)


def fig_renorm(S, R, log):
    s = S[1]                                                # row "fluffy": sees a, fluffy
    full = softmax_rows(s[None])[0]
    zero_after = np.where(np.arange(5) <= 1, full, 0.0)
    masked = softmax_rows(np.where(np.arange(5) <= 1, s, -np.inf)[None])[0]
    Q = QUOTED["renorm_fluffy"]
    check("zero after", zero_after, Q["zero_after"])
    check("zero after: sum", [zero_after.sum()], [Q["zero_after_sum"]])
    check("masked", masked, Q["masked"])
    check("future share, unmasked", [full[2:].sum(), full[4]], [Q["future_share"], Q["roamed"]])
    if not np.isclose(zero_after[0] / zero_after[1], masked[0] / masked[1]):
        raise AssertionError("masking first should keep the visible weights' ratio")
    R["renorm_fluffy"] = {"scores": s.tolist(), "zero_after": zero_after.tolist(), "masked": masked.tolist()}
    log.info(f"renorm: scores {s.tolist()}, zero after {r2(zero_after)} (sum {zero_after.sum():.2f}), "
             f"masked {r2(masked)}")
    fig, axes = plt.subplots(1, 2, figsize=(2.75, 1.6), sharey=True)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.8, bottom=0.27, wspace=0.12)
    panels = ((zero_after, "softmax, then\nzero the future", f"sum {zero_after.sum():.2f}", ARMRED),
              (masked, "−∞ first,\nthen softmax", "sum 1", GREEN_D))
    for ax, (v, title, note, c) in zip(axes, panels):
        cols = [ARMBLUE if j <= 1 else LIGHT for j in range(5)]
        bars = ax.bar(range(5), v, color=cols, width=0.7)
        ax.bar_label(bars, labels=[num(x) if j <= 1 else "0" for j, x in enumerate(v)], fontsize=FS_TINY, padding=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["a", "fl.", "bl.", "cr.", "ro."], fontsize=FS_TINY)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=FS_TINY, pad=2)
        ax.set_xlabel(note, fontsize=FS_SMALL, color=c, labelpad=2)
        ax.set_ylim(0, 0.95)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    save(fig, "mask_renorm.pdf", log)


def fig_one_pass(R, log):
    n = QUOTED["one_pass"]
    fig, ax = canvas(1.9)
    bw, step, h, top = 0.46, 0.5, 0.17, 1.72
    for t in range(1, n + 1):
        y = top - t * 0.22
        for j in range(t):
            wbox(ax, 0.3 + j * step, y, WORDS[j], w=bw, h=h, fc="#DCE5F5", ec=ARMBLUE)
        xt = 0.3 + t * step
        arrow(ax, (xt - 0.04, y + h / 2), (xt + 0.06, y + h / 2), color=INK, lw=0.7)
        wbox(ax, xt + 0.06, y, WORDS[t], w=bw, h=h, fc=TINT_GREEN, ec=GREEN_D, color=GREEN_D, bold=True)
    ax.text(0.3, top - 0.03, "row t: the first t words (what the mask lets it see)  →  the word to predict",
            ha="left", va="bottom", fontsize=FS_TINY, color=GREY)
    xb = 0.3 + n * step + 0.06 + bw + 0.12
    yb0, yb1 = top - n * 0.22, top - 0.22 + h
    ax.plot([xb, xb + 0.06, xb + 0.06, xb], [yb1, yb1, yb0, yb0], color=INK, lw=0.8)
    ax.text(xb + 0.12, (yb0 + yb1) / 2, f"one pass,\n{n} predictions,\n{n} losses", ha="left", va="center",
            fontsize=FS_SMALL, color=GREEN_D)
    R["one_pass"] = n
    save(fig, "mask_one_pass.pdf", log)


def fig_mlm(log):
    hidden = {2: "blue", 6: "verdant"}
    fig, ax = canvas(1.85)
    x0, step = 0.72, 0.5
    ax.text(x0 - 0.1, 1.65, "original", ha="right", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(x0, 1.65, " ".join(WORDS), ha="left", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(x0 - 0.1, 0.92, "input", ha="right", va="center", fontsize=FS_SMALL)
    ax.text(x0 - 0.1, 0.27, "predict", ha="right", va="center", fontsize=FS_SMALL, color=GREEN_D)
    centres = []
    for j, wd in enumerate(WORDS):
        x = x0 + j * step
        if j in hidden:
            centres.append(wbox(ax, x, 0.8, "[MASK]", w=0.46, fc=TINT_RED, ec=ARMRED, color=ARMRED, bold=True))
            wbox(ax, x, 0.15, hidden[j], w=0.46, fc=TINT_GREEN, ec=GREEN_D, color=GREEN_D, bold=True)
            arrow(ax, (x + 0.23, 0.78), (x + 0.23, 0.41), color=GREEN_D, lw=1.0)
        else:
            centres.append(wbox(ax, x, 0.8, wd, w=0.46))
            ax.text(x + 0.23, 0.27, "–", ha="center", va="center", fontsize=FS_SMALL, color=LIGHT)
    cx, cy = centres[2]
    for j in (0, 1, 3, 4, 5, 7):                         # [MASK] 1 reads both sides
        sx, _ = centres[j]
        ax.add_patch(FancyArrowPatch((sx, 1.06), (cx + (0.05 if sx > cx else -0.05), 1.06),
                                     connectionstyle=f"arc3,rad={-0.35 if sx < cx else 0.35}",
                                     arrowstyle="-|>", mutation_scale=6, color=ARMBLUE, lw=0.6, alpha=0.8))
    ax.text(x0 + 8 * step + 0.02, 0.92, "no mask:\nboth sides\nvisible",
            ha="left", va="center", fontsize=FS_TINY, color=ARMBLUE)
    ax.text(x0 + 8 * step + 0.02, 0.27, "a loss only\nat [MASK]", ha="left", va="center", fontsize=FS_TINY,
            color=GREEN_D)
    save(fig, "mask_mlm.pdf", log)


def fig_clm_vs_mlm(log):
    n = 6
    clm_words = WORDS[:n]
    mlm_words = ["a", "fluffy", "[MASK]", "creature", "roamed", "the"]
    fig, ax = canvas(1.95)
    cell, top = 0.19, 1.5
    for x0, words, allowed, title, targets in (
            (0.62, clm_words, causal(n), "CLM (GPT): the causal mask", WORDS[1:n + 1]),
            (3.42, mlm_words, np.ones((n, n), bool), "MLM (BERT): no attention mask",
             ["–", "–", "blue", "–", "–", "–"])):
        hl = (2,) if "[MASK]" in words else ()
        mgrid(ax, x0, top, cell, allowed, row_labels=words, col_labels=words, hl_rows=hl)
        xr = x0 + n * cell
        ax.text(xr + 0.08, top + 0.05, "predicts", ha="left", va="bottom", fontsize=FS_TINY, color=GREEN_D)
        for i, t in enumerate(targets):
            ax.text(xr + 0.08, top - (i + 0.5) * cell, t, ha="left", va="center", fontsize=FS_TINY,
                    color=GREEN_D if t != "–" else LIGHT, fontweight="bold" if t != "–" else "normal")
        ax.text(x0 + n * cell / 2 + 0.2, top - n * cell - 0.07, title, ha="center", va="top", fontsize=FS_SMALL,
                fontweight="bold")
    save(fig, "mask_clm_vs_mlm.pdf", log)


def fig_padding(log):
    fig, ax = canvas(1.75)
    s1 = ["the", "cat", "sat", "[PAD]", "[PAD]"]
    s2 = ["a", "fluffy", "blue", "creature", "roamed"]
    ax.text(1.25, 1.6, "a batch: two sentences, one length", ha="center", va="center", fontsize=FS_SMALL,
            fontweight="bold")
    for r, sent in enumerate((s1, s2)):
        y = 1.05 - r * 0.38
        for j, wd in enumerate(sent):
            pad = wd == "[PAD]"
            wbox(ax, 0.1 + j * 0.47, y, wd, w=0.44, fc=BLOCKED_FC if pad else "#F2F2F2",
                 ec=GREY if pad else "#999999", color=GREY if pad else INK, hatch="////" if pad else None)
    ax.text(1.25, 0.32, "the short one is filled up with [PAD]", ha="center", va="center", fontsize=FS_TINY,
            color=GREY)
    allowed = np.ones((5, 5), bool)
    allowed[:, 3:] = False
    cell, x0, top = 0.21, 3.3, 1.38
    mgrid(ax, x0, top, cell, allowed, blocked_text="−∞", row_labels=s1, col_labels=s1,
          dim_rows=(3, 4), row_colors={3: GREY, 4: GREY})
    xr = x0 + 5 * cell
    ax.text(xr + 0.08, top - 1.5 * cell, "real words never\nlook at [PAD]", ha="left", va="center",
            fontsize=FS_TINY, color=ARMBLUE)
    ax.text(xr + 0.08, top - 4 * cell, "[PAD] rows:\nthrown away", ha="left", va="center", fontsize=FS_TINY,
            color=GREY)
    ax.text(x0 + 2.5 * cell, top - 5 * cell - 0.07, "sentence 1, seeing both ways (BERT)", ha="center", va="top",
            fontsize=FS_TINY)
    save(fig, "mask_padding.pdf", log)


def fig_two_kinds(log):
    """The trap: the [MASK] token changes the input; the attention mask changes who looks at whom."""
    fig, ax = canvas(1.55)
    ax.text(1.3, 1.5, "the [MASK] token", ha="center", va="top", fontsize=FS, fontweight="bold",
            color=ARMRED)
    ax.text(1.3, 1.32, "hides a word in the input", ha="center", va="top", fontsize=FS_TINY, color=GREY)
    for j, wd in enumerate(["a", "fluffy", "[MASK]", "creature"]):
        m = wd == "[MASK]"
        wbox(ax, 0.2 + j * 0.56, 0.72, wd, w=0.52, fc=TINT_RED if m else "#F2F2F2",
             ec=ARMRED if m else "#999999", color=ARMRED if m else INK, bold=m)
    ax.text(1.3, 0.42, "MLM (BERT): the word is gone,\neverything else is visible",
            ha="center", va="center", fontsize=FS_TINY, linespacing=1.3)
    ax.plot([2.75, 2.75], [0.1, 1.45], color=LIGHT, lw=0.6)
    ax.text(4.15, 1.5, "the attention mask", ha="center", va="top", fontsize=FS, fontweight="bold",
            color=ARMBLUE)
    ax.text(4.15, 1.32, "hides positions from each other", ha="center", va="top", fontsize=FS_TINY,
            color=GREY)
    words = ["a", "fluffy", "blue", "creature"]
    mgrid(ax, 3.55, 1.12, 0.17, causal(4), row_labels=words)
    ax.text(4.32, 0.84, "no word is\nremoved", ha="left", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(4.15, 0.27, "CLM (GPT): the input is complete,\neach row sees only the past",
            ha="center", va="center", fontsize=FS_TINY, linespacing=1.3)
    save(fig, "mask_two_kinds.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"date": time.strftime("%Y-%m-%d"), "quoted": QUOTED}
    S = None

    def steps():
        nonlocal S
        S = fig_steps(R, log)

    jobs = [lambda: fig_two_games(log), lambda: fig_clm_shift(log), lambda: fig_cheat(R, log), steps,
            lambda: fig_creature_grid(R, log), lambda: fig_renorm(S, R, log), lambda: fig_one_pass(R, log),
            lambda: fig_mlm(log), lambda: fig_clm_vs_mlm(log), lambda: fig_padding(log),
            lambda: fig_two_kinds(log)]
    for i, job in enumerate(jobs, 1):
        job()
        el = time.time() - t0
        log.info(f"{i}/{len(jobs)}, {el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "masking_figs.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
