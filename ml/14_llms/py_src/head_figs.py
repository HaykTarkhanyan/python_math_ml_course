"""Figures for LLM-5 "Transformers in the world" (ml/14_llms/LLM5_transformers_in_the_world.tex).

Restructure of 2026-10-06 (INTERVIEW_QUEUE.md, LLM-5): the LM head by hand on GPT-2, a three-mask
cold open, and the three July figures (l25_l26_figs.py) redrawn at slot size.

Real numbers only for claims about GPT-2, from what is local:
  --gpt2   one run of the cached GPT-2 small (HF_HUB_OFFLINE=1, no download) on the map's prompt
           "The cat sat on the": block 12's output for the last token, the final LayerNorm, the
           tied token table, all 50,257 logits, the softmax -> results/head_gpt2.json (~30 s)
  (default) the figures, reading that JSON (~5 s)

Toy numbers (seeded): the cross-attention example keeps the July values (SEED 509, the same RNG
calls as l25_l26_figs.py), so its weights 0.399 / 0.216 / 0.385 do not move.

Drawn at the size each figure gets on the slide (5.5 in = full text width), 6.5-8 pt text, and
save() refuses any text outside its figure.

Generates into ml/14_llms/fig/:
  head_masks.pdf          -- cold open: three attention masks, unnamed ("which can write text?")
  head_masks_named.pdf    -- the same three, named: encoder-only, decoder-only, encoder-decoder
  head_cross_attention.pdf-- cross-attention with numbers (port of l26_cross_attention)
  head_three_arch.pdf     -- the three architectures (port of l26_three_arch)
  head_pipeline.pdf       -- the LM head on GPT-2: last vector -> final norm -> tied table -> logits
  head_top_tokens.pdf     -- GPT-2's top-8 next tokens after "The cat sat on the", and the tail
  head_quadratic.pdf      -- attention vs feed-forward FLOPs, crossover n = d f (port of l26_quadratic)

Run:
  ./ma/Scripts/python.exe ml/14_llms/py_src/head_figs.py --gpt2   # once (~30 s)
  ./ma/Scripts/python.exe ml/14_llms/py_src/head_figs.py          # figures (~5 s)
"""
import json
import math
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("error", message=".*missing from.*font.*")

SEED = 509
HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "head_gpt2.json"
PROMPT = "The cat sat on the"

# Real numbers the slides quote; --gpt2 must reproduce them and the figures check them.
QUOTED = {
    "top5": [[" floor", 7.6], [" bed", 6.5], [" couch", 5.4], [" ground", 5.2], [" edge", 4.8]],
    "vocab": 50257, "d_model": 768,
    # added 2026-10-27 (student review): the 3-logit shift toy and the log-sum-exp gap on GPT-2
    "shift_toy": {"logits": [-80, -81, -82], "p": [0.67, 0.24, 0.09]},
    "lse_gap": 2.57, "tail_times_floor": 13.1,
}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"
TINT_RED, TINT_BLUE, TINT_ORANGE, TINT_GREEN = "#F9D2D5", "#CFD8EE", "#FCE9B8", "#D5EFDB"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("head_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "head_figs.log", encoding="utf-8")):
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


def rbox(ax, x, y, w, h, fc, ec, lw=0.6, r=0.05, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color=GREY, lw=0.8, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    """Every drawn text must sit inside the figure (tick labels outside the axis range are skipped)."""
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
            fig.savefig(out, metadata={"CreationDate": None})   # same figure -> same bytes
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def show_tok(t):
    return t.replace(" ", "␣")


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    wte = model.transformer.wte.weight
    tied = wte.data_ptr() == model.lm_head.weight.data_ptr()
    captured = {}
    hook = model.transformer.h[-1].register_forward_hook(        # a tuple in older transformers
        lambda m, i, o: captured.update(h=o[0] if isinstance(o, tuple) else o))
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
        hook.remove()
        h = captured["h"][0, -1]                          # block 12's output, last token
        x = model.transformer.ln_f(h)                     # the final LayerNorm
        logits = x @ wte.T                                # the LM head: one dot product per row
    model_logits = out.logits[0, -1]
    gap = float((logits - model_logits).abs().max())
    if not tied or gap > 1e-3:
        raise ValueError(f"LM head check failed: tied={tied}, max logit gap {gap}")
    z = logits.numpy().astype(np.float64)
    p = softmax(z)
    order = np.argsort(-z)
    floor = tok.encode(" floor")[0]
    e = wte[floor].detach().numpy().astype(np.float64)
    xv = x.numpy().astype(np.float64)
    res = {
        "model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"), "prompt": PROMPT,
        "tokens": [tok.decode([i]) for i in ids[0].tolist()], "tied": tied, "max_logit_gap": gap,
        "vocab": int(z.size), "d_model": int(xv.size),
        "n_params": int(sum(q.numel() for q in model.parameters())),   # the tied table counted once
        "n_table": int(wte.numel()),
        "h_first4": [float(v) for v in h[:4]], "h_mean": float(h.mean()), "h_std": float(h.std(unbiased=False)),
        "h_norm": float(h.norm()), "x_first4": [float(v) for v in xv[:4]], "x_norm": float(np.linalg.norm(xv)),
        "e_floor_first4": [float(v) for v in e[:4]], "z_floor": float(z[floor]),
        "partial4": float(np.dot(xv[:4], e[:4])),
        "top": [[tok.decode([int(i)]), float(z[i]), float(p[i])] for i in order[:10]],
        "bottom": [tok.decode([int(order[-1])]), float(z[order[-1]]), float(p[order[-1]])],
        "logsumexp": float(np.log(np.exp(z - z.max()).sum()) + z.max()),
        "top5_share": float(p[order[:5]].sum()), "n_over_1pct": int((p > 0.01).sum()),
        "n_over_01pct": int((p > 0.001).sum()),
    }
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"tied={tied}, max |manual - model| logit = {gap:.2e}")
    log.info(f"top-10: {[(t, round(l, 2), round(100 * q, 2)) for t, l, q in res['top']]}")
    log.info(f"top-5 share {res['top5_share']:.3f}; tokens over 1% {res['n_over_1pct']}; "
             f"over 0.1% {res['n_over_01pct']}; wrote {GPT2_JSON.name}")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    G = json.loads(GPT2_JSON.read_text(encoding="utf-8"))
    got = [[t, round(100 * q, 1)] for t, _l, q in G["top"][:5]]
    if got != QUOTED["top5"] or G["vocab"] != QUOTED["vocab"] or G["d_model"] != QUOTED["d_model"]:
        raise ValueError(f"GPT-2 numbers changed: top-5 {got}; the slides quote {QUOTED['top5']}")
    if not G["tied"] or G["max_logit_gap"] > 1e-3:
        raise ValueError("the LM head is no longer the tied token table")
    # the by-hand frame's printed arithmetic, rounded as printed
    (_, z1, p1), (_, z2, p2) = G["top"][:2]
    x4, e4 = G["x_first4"], G["e_floor_first4"]
    by_hand = {
        "z_floor": (round(z1, 2), -80.65), "z_bed": (round(z2, 2), -80.81),
        "gap": (round(z1 - z2, 3), 0.157), "exp_gap": (round(float(np.exp(z1 - z2)), 2), 1.17),
        "p_ratio": (round(p1 / p2, 2), 1.17), "lse": (round(G["logsumexp"], 2), -78.08),
        "z_minus_lse": (round(z1 - G["logsumexp"], 3), -2.572),
        "exp_z_minus_lse": (round(100 * float(np.exp(-2.572)), 2), 7.64),
        "p_floor": (round(100 * float(np.exp(z1 - G["logsumexp"])), 2), 7.64),
        "x4": ([round(v, 3) for v in x4], [-0.284, -0.033, -0.962, 0.225]),
        "e4": ([round(v, 3) for v in e4], [-0.122, 0.161, -0.025, -0.024]),
        "partial4": (round(G["partial4"], 3), 0.048),
        # the four printed products, from the printed (rounded) numbers, must give the same 0.048
        "partial4_printed": (round(sum(a * b for a, b in zip([-0.284, -0.033, -0.962, 0.225],
                                                             [-0.122, 0.161, -0.025, -0.024])), 3), 0.048),
        "n_params": (G["n_params"], 124439808), "n_table": (G["n_table"], 38597376),
        "table_share": (round(100 * G["n_table"] / G["n_params"]), 31),
        "top5_share": (round(100 * G["top5_share"]), 30), "n_over_1pct": (G["n_over_1pct"], 14),
    }
    for k, (got_v, want) in by_hand.items():
        if got_v != want:
            raise ValueError(f"by-hand number {k} changed: {got_v}, the slide prints {want}")
    return G


# ----------------------------------------------------------------------------- figures
def mask_panel(ax, allowed, rows, cols, title, color, split=None):
    n_r, n_c = allowed.shape
    for i in range(n_r):
        for j in range(n_c):
            ax.add_patch(Rectangle((j, n_r - 1 - i), 1, 1, facecolor=color if allowed[i, j] else "white",
                                   edgecolor=LIGHT, lw=0.5))
    if split is not None:
        ax.plot([split, split], [0, n_r], color=INK, lw=1.0)
    ax.set_xlim(0, n_c); ax.set_ylim(0, n_r)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_c) + 0.5)
    ax.set_xticklabels(cols, rotation=90, fontsize=FS_TINY)
    ax.set_yticks(np.arange(n_r) + 0.5)
    ax.set_yticklabels(rows[::-1], fontsize=FS_TINY)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)


def fig_masks(named, log):
    words = ["the", "cat", "sat", "on", "the"]
    src, tgt = ["le", "chat", "noir"], ["the", "black", "cat"]
    full = np.ones((5, 5), bool)
    causal = np.tril(np.ones((5, 5), bool))
    cross = np.hstack([np.ones((3, 3), bool), np.tril(np.ones((3, 3), bool))])
    titles = (["Encoder-only (BERT)", "Decoder-only (GPT)", "Encoder-decoder (T5)"] if named
              else ["A", "B", "C"])
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.65), gridspec_kw={"width_ratios": [5, 5, 6]})
    mask_panel(axes[0], full, words, words, titles[0], TINT_BLUE)
    mask_panel(axes[1], causal, words, words, titles[1], TINT_RED)
    mask_panel(axes[2], cross, tgt, src + tgt, titles[2], TINT_GREEN, split=3)
    axes[2].text(1.5, -0.25, "encoder", ha="center", va="top", fontsize=FS_TINY, color=GREY)
    axes[2].text(4.5, -0.25, "decoder", ha="center", va="top", fontsize=FS_TINY, color=GREY)
    axes[0].set_ylabel("this token...", fontsize=FS_TINY)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.76, bottom=0.09, wspace=0.35)
    fig.canvas.draw()
    for ax, title in zip(axes, titles):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.95, title, va="top", ha="center", fontsize=FS, fontweight="bold", color=INK)
    save(fig, "head_masks_named.pdf" if named else "head_masks.pdf", log)


def fig_cross_attention(log):
    """Port of l25_l26_figs.fig_cross_attention: same SEED, same RNG calls, same numbers."""
    rng = np.random.default_rng(SEED)
    d_model, d_k = 8, 4
    source = ["le", "chat", "noir"]
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) * 0.5 for _ in range(3))
    enc_out = rng.normal(size=(len(source), d_model))
    dec_tok = rng.normal(size=(1, d_model))
    Q = dec_tok @ Wq
    K, V = enc_out @ Wk, enc_out @ Wv
    scores = (Q @ K.T / np.sqrt(d_k))[0]
    weights = softmax(scores)
    quoted_scores, quoted_weights = [0.339, -0.273, 0.304], [0.399, 0.216, 0.385]
    raw_le = float((Q @ K.T)[0, 0])                      # the slide: q . k_le = 0.6788, / sqrt(4) = 0.3394
    if round(raw_le, 4) != 0.6788 or round(raw_le / 2, 4) != 0.3394:
        raise ValueError(f"q . k_le changed: {raw_le}")
    if [round(float(s), 3) for s in scores] != quoted_scores or \
            [round(float(w), 3) for w in weights] != quoted_weights:
        raise ValueError(f"cross-attention numbers changed: {scores}, {weights}")
    log.info(f"cross-attention: scores {np.round(scores, 3)}, weights {np.round(weights, 3)}")

    fig = plt.figure(figsize=(4.62, 1.85))
    ax = fig.add_axes([0, 0, 0.6, 1])
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.add_patch(FancyBboxPatch((0.2, 3.5), 4.5, 2.2, boxstyle="round,pad=0.02",
                                facecolor="#F2F6FB", edgecolor="#C8D6E8", lw=0.8))
    ax.text(2.45, 5.15, "ENCODER: K and V", ha="center", fontsize=FS_TINY, color=ARMBLUE,
            fontweight="bold")
    for i, wtok in enumerate(source):
        ax.add_patch(FancyBboxPatch((0.45 + i * 1.42, 3.85), 1.25, 0.85, boxstyle="round,pad=0.02",
                                    facecolor=ARMBLUE, edgecolor="none"))
        ax.text(1.07 + i * 1.42, 4.27, wtok, ha="center", va="center", fontsize=FS_SMALL,
                color="white", fontweight="bold")
    ax.add_patch(FancyBboxPatch((6.2, 3.5), 3.6, 2.2, boxstyle="round,pad=0.02",
                                facecolor="#FDF3F3", edgecolor="#EFCFCF", lw=0.8))
    ax.text(8.0, 5.15, "DECODER: Q", ha="center", fontsize=FS_TINY, color=ARMRED,
            fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.2, 3.85), 1.6, 0.85, boxstyle="round,pad=0.02",
                                facecolor=ARMRED, edgecolor="none"))
    ax.text(8.0, 4.27, "the", ha="center", va="center", fontsize=FS_SMALL, color="white", fontweight="bold")
    for i, a in enumerate(weights):
        ax.add_patch(FancyArrowPatch((8.0, 3.85), (1.07 + i * 1.42, 3.85), arrowstyle="-|>",
                                     mutation_scale=7, color=GREEN, lw=0.5 + 3.5 * a,
                                     connectionstyle="arc3,rad=-0.3", zorder=4))
    ax.text(5.0, 1.3, "arrow thickness = attention weight", ha="center", fontsize=FS_TINY,
            color=GREEN, style="italic")
    bx = fig.add_axes([0.70, 0.17, 0.28, 0.62])
    bars = bx.bar(source, weights, color=[GREEN if a == weights.max() else ARMBLUE for a in weights])
    bx.bar_label(bars, fmt="%.3f", fontsize=FS_TINY, fontweight="bold", padding=1.5)
    bx.set_ylim(0, weights.max() * 1.35)
    bx.set_yticks([])
    bx.set_title("softmax over the\nencoder tokens", fontsize=FS_TINY)
    bx.tick_params(axis="x", length=0)
    clean(bx)
    bx.spines["left"].set_visible(False)
    save(fig, "head_cross_attention.pdf", log)


def fig_three_arch(log):
    fig, ax = canvas(1.7, 4.96)
    specs = [(0.12, "Encoder-only", "BERT", "every token sees\nevery other token", ARMBLUE,
              "understanding:\nclassify, tag, embed"),
             (1.72, "Decoder-only", "GPT", "each token sees\nonly the past", ARMRED,
              "generation:\nthe dominant design"),
             (3.32, "Encoder-decoder", "T5", "encoder reads all,\ndecoder attends to it", GREEN,
              "sequence to sequence:\ntranslation")]
    for x, name, model, how, color, use in specs:
        rbox(ax, x, 0.78, 1.52, 0.34, color, color)
        ax.text(x + 0.76, 0.95, f"{name} ({model})", ha="center", va="center", fontsize=FS_TINY,
                color="white", fontweight="bold")
        ax.text(x + 0.76, 0.5, how, ha="center", va="center", fontsize=FS_TINY, color=INK)
        ax.text(x + 0.76, 1.42, use, ha="center", va="center", fontsize=FS_TINY, color=color,
                fontweight="bold")
    ax.text(2.48, 0.08, "same block, three ways of deciding who may look at whom", ha="center",
            fontsize=FS_TINY, color=GREY, style="italic")
    save(fig, "head_three_arch.pdf", log)


def fig_pipeline(G, log):
    """The LM head on GPT-2: shapes, and the first entries of the real vectors."""
    fig, ax = canvas(1.7)
    f4 = lambda v: "[" + ", ".join(f"{a:+.2f}" for a in v) + ", ...]"
    stages = [
        (0.05, "last block's output\n(12th of 12), last token", f"768 numbers\n{f4(G['h_first4'][:3])}", TINT_BLUE, ARMBLUE),
        (1.42, "final LayerNorm", f"768 numbers\n{f4(G['x_first4'][:3])}", TINT_BLUE, ARMBLUE),
        (2.79, "dot product with each\nrow of the token table", "50,257 logits\n(one per token)",
         TINT_ORANGE, ARMORANGE),
        (4.16, "softmax", "50,257 probabilities\nsumming to 1", TINT_GREEN, GREEN),
    ]
    for x, title, body, fc, ec in stages:
        rbox(ax, x, 0.62, 1.29, 0.82, fc, ec)
        ax.text(x + 0.645, 1.24, title, ha="center", va="center", fontsize=FS_TINY, color=INK,
                fontweight="bold")
        ax.text(x + 0.645, 0.86, body, ha="center", va="center", fontsize=FS_TINY - 0.3, color=INK)
    for x in (1.34, 2.71, 4.08):
        arrow(ax, (x, 1.03), (x + 0.08, 1.03), color=GREY)
    ax.text(3.435, 0.4, "the same 50,257 x 768 table that turned IDs into vectors in LLM-2 (tied)",
            ha="center", fontsize=FS_TINY, color=ARMORANGE, style="italic")
    ax.text(2.75, 1.56, f"GPT-2 on \"{PROMPT}\" - what happens after the last block",
            ha="center", fontsize=FS_SMALL, color=INK, fontweight="bold")
    ax.text(2.75, 0.1, "the first entries of the real vectors are shown", ha="center",
            fontsize=FS_TINY, color=GREY)
    save(fig, "head_pipeline.pdf", log)


def fig_top_tokens(G, log):
    top = G["top"][:8]
    fig, ax = plt.subplots(figsize=(3.6, 1.75))
    names = [show_tok(t) for t, _, _ in top][::-1]
    probs = [100 * q for _, _, q in top][::-1]
    bars = ax.barh(names, probs, color=[ARMBLUE] * 3 + [ARMBLUE] * 5, height=0.62)
    ax.bar_label(bars, labels=[f"{q:.1f}%  (logit {l:.1f})" for _, l, q in
                               [(t, l, 100 * q) for t, l, q in top][::-1]],
                 fontsize=FS_TINY, padding=2)
    ax.set_xlim(0, max(probs) * 2.0)
    ax.set_xlabel("probability of the next token (%)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.set_title(f"after \"{PROMPT}\": the top 8 of 50,257", fontsize=FS_SMALL)
    fig.tight_layout(pad=0.3)
    save(fig, "head_top_tokens.pdf", log)


def fig_quadratic(log, d_model=4096, d_ff_mult=4):
    n = np.logspace(1, 5.7, 200)
    attn = 2 * n ** 2 * d_model
    ffn = 2 * n * d_model ** 2 * d_ff_mult
    crossover = d_model * d_ff_mult
    if crossover != 16384:
        raise ValueError("the slide quotes a crossover of 16,384 tokens")
    if round(128000 ** 2 / 1e9, 1) != 16.4:      # "one head's n x n scores at 128,000 tokens"
        raise ValueError(f"the slide quotes 16.4 billion scores, got {128000 ** 2 / 1e9:.2f}")
    fig, ax = plt.subplots(figsize=(4.13, 1.95))
    ax.loglog(n, attn, color=ARMRED, lw=1.4, label=r"attention $\propto n^2 d$")
    ax.loglog(n, ffn, color=ARMBLUE, lw=1.4, label=r"feed-forward $\propto n d^2 f$")
    # the three markers are labelled above the axes, where no line runs
    for ctx, lab, color, ls in [(2048, "2,048\ncontext", GREEN, ":"),
                                (crossover, f"crossover\n{crossover:,}", GREY, "--"),
                                (128000, "128k\ncontext", GREEN, ":")]:
        ax.axvline(ctx, color=color, ls=ls, lw=0.9)
        ax.text(ctx, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=FS_TINY, color=color, fontweight="bold")
    ax.set_xlabel("sequence length $n$ (tokens)")
    ax.set_ylabel("multiply-adds per layer")
    ax.tick_params(labelsize=8)          # 10^11's exponent prints at 0.7x: 7 pt ticks gave 4.9 pt
    ax.legend(frameon=False, fontsize=FS_TINY, loc="upper left")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "head_quadratic.pdf", log)
    return crossover


def check_shift_toy(G, log):
    """'By hand: three logits near -80' (LLM-5): a shift changes nothing, and GPT-2's log-sum-exp
    sits 2.57 above floor's logit, so the vocabulary carries 13.1x floor's weight (1/13.1 = 7.64%)."""
    z = np.array(QUOTED["shift_toy"]["logits"], dtype=float)
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    p0 = np.exp(z + 80) / np.exp(z + 80).sum()
    if not np.allclose(p, p0) or [round(float(v), 2) for v in p] != QUOTED["shift_toy"]["p"]:
        raise AssertionError(f"shift toy: {p}, {p0}")
    gap = G["logsumexp"] - G["z_floor"]
    if round(gap, 2) != QUOTED["lse_gap"] or round(math.exp(gap), 1) != QUOTED["tail_times_floor"]:
        raise AssertionError(f"lse gap {gap:.4f}, exp {math.exp(gap):.3f}")
    log.info(f"shift toy {np.round(p, 3)}; lse gap {gap:.3f}, x{math.exp(gap):.2f} floor's weight")


def main():
    log = setup_logging()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    G = load_gpt2()
    check_shift_toy(G, log)
    fig_masks(False, log)
    fig_masks(True, log)
    fig_cross_attention(log)
    fig_three_arch(log)
    fig_pipeline(G, log)
    fig_top_tokens(G, log)
    fig_quadratic(log)
    log.info("done: 7 figures")


if __name__ == "__main__":
    main()
