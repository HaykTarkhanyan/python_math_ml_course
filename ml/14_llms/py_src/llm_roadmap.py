"""The chapter's "you are here" map, as two illustrated frames per session.

Frame "pass" - one forward pass of a real model (GPT-2 small) on L24's sentence "The cat sat on the",
drawn the way Transformer Explainer (poloclub.github.io/transformer-explainer) does it: one row per
token, flowing left to right. Text -> token chips with their real IDs -> embedding vectors (real
values) -> a stack of N layers (attention arcs from the last token, real weights of one head; an
expand-and-contract MLP per token) -> the last token's vector -> LM head -> the real top-5
next-token probabilities -> the chosen token, looping back to the input.

Frame "life" - the life of a model, after Karpathy's "State of GPT" pipeline: pretrain -> scale ->
post-train -> prompt -> evaluate -> quantize -> LoRA, each a small drawing with one fact, grouped
as "make it" / "use it" / "run and adapt it".

Each session lights its stages: this session = full colour on a blue panel, earlier sessions =
full colour, later ones = faded. SESSIONS is the teaching order and must match
ml/14_llms/LLM_CHAPTER_PLAN.md, section 5.0 (DECISIONS #67).

Drawn at the size it gets on the slide (5.5 x 2.5 in, include at width=\\linewidth), so 7-8 pt
text stays 7-8 pt.

Two steps, so the model is not rerun for every redraw:
  ./ma/Scripts/python.exe ml/14_llms/py_src/llm_roadmap.py --gpt2   # GPT-2 numbers -> results/roadmap_gpt2.json (~2 min, mostly importing torch)
  ./ma/Scripts/python.exe ml/14_llms/py_src/llm_roadmap.py          # the maps, from that JSON (19-78 s measured)

Generates into ml/14_llms/fig/: roadmap_<key>_pass.pdf and roadmap_<key>_life.pdf for every
session key, plus roadmap_overview_pass.pdf / roadmap_overview_life.pdf (nothing lit).
"""
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
JSON_PATH = RESULTS / "roadmap_gpt2.json"

PROMPT = "The cat sat on the"
W, H = 5.5, 2.5                      # inches; the axes span the figure in inch coordinates
FS, FS_SMALL = 8, 7

ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
POPBLUE = "#3465A4"                  # ml/preamble.tex popblue (52, 101, 164)
HILITE = "#E3ECF7"                   # panel behind this session's stage
INK, GREY, FAINT = "#222222", "#777777", 0.32
CHIP_TINTS = ["#F9D2D5", "#CFD8EE", "#FCE9B8"]   # Armenian flag colours, light
CMAP = colormaps["RdBu_r"]

PASS_STAGES = [("tok", "Tokenizer"), ("emb", "Embedding"), ("attn", "Attention"), ("mlp", "MLP"),
               ("head", "LM head"), ("dec", "Decoding")]
LIFE_STAGES = [("pre", "Pretrain"), ("scale", "Scale"), ("post", "Post-train"), ("prompt", "Prompt"),
               ("eval", "Evaluate"), ("quant", "Quantize"), ("lora", "LoRA")]

# (file key, deck, stages lit). Teaching order. RAG and agents (own chapters) sit between
# "evaluation" and "quantization".
SESSIONS = [
    ("tokenization", "LLM-1", {"tok"}),
    ("attention", "L24", {"emb", "attn"}),
    ("block", "L25", {"attn", "mlp"}),
    ("transformers", "L26", {"head"}),
    ("decoding", "LLM-2", {"dec"}),
    ("pretraining", "LLM-3", {"pre"}),
    ("scaling", "LLM-4", {"scale"}),
    ("post_training", "LLM-5", {"post"}),
    ("prompting", "LLM-6", {"prompt"}),
    ("evaluation", "LLM-7", {"eval"}),
    ("quantization", "LLM-8", {"quant"}),
    ("lora", "LLM-9", {"lora"}),
    ("anatomy_2026", "LLM-10", {"attn", "mlp"}),
]


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("llm_roadmap")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS / "llm_roadmap.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


# ----------------------------------------------------------------------------- GPT-2 numbers
def compute_gpt2(log):
    """One forward pass of GPT-2 small on PROMPT; everything the "pass" frame draws."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the model is in the local cache
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    n = ids.shape[1]
    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top_p, top_i = probs.topk(5)

    # The head whose attention from the last token puts the most weight on "cat" (position 1),
    # i.e. one real head that looks back at the subject. Position 0 is skipped as a candidate
    # because most heads park their weight on the first token.
    best = max(((l, h) for l in range(len(out.attentions)) for h in range(out.attentions[l].shape[1])),
               key=lambda lh: out.attentions[lh[0]][0, lh[1], -1, 1].item())
    attn = out.attentions[best[0]][0, best[1], -1, :].tolist()

    res = {
        "model": "gpt2 (124M), local HF cache",
        "date": date.today().isoformat(),
        "prompt": PROMPT,
        "token_ids": ids[0].tolist(),
        "tokens": [tok.decode([i]) for i in ids[0].tolist()],
        "embedding_dims_0_7": model.transformer.wte.weight[ids[0], :8].tolist(),
        "attention_layer_head": list(best),
        "attention_head_note": "max over layers/heads of the last token's weight on position 1",
        "attention_from_last": attn,
        "last_hidden_dims_0_7": out.hidden_states[-1][0, -1, :8].tolist(),
        "top5": [[tok.decode([i]), p] for i, p in zip(top_i.tolist(), top_p.tolist())],
    }
    assert n == len(res["tokens"]) == 5, res["tokens"]
    RESULTS.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"tokens {res['tokens']} ids {res['token_ids']}")
    log.info(f"attention: layer {best[0]} head {best[1]}: {[round(a, 3) for a in attn]}")
    log.info(f"top5: {[(t, round(p, 3)) for t, p in res['top5']]}")
    log.info(f"wrote {JSON_PATH}")


# ----------------------------------------------------------------------------- drawing helpers
def new_canvas():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=0.8, r=0.04, alpha=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
                                linestyle=ls, zorder=z))


def arrow(ax, p0, p1, alpha=1.0, color=GREY, lw=0.8, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw,
                                 alpha=alpha, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=0, shrinkB=0))


def strip(ax, x, y, vals, cell=0.05, h=0.13, alpha=1.0, vmax=None):
    """A vector as a row of coloured cells (blue negative, red positive)."""
    vmax = vmax or max(abs(v) for v in vals)
    for k, v in enumerate(vals):
        ax.add_patch(Rectangle((x + k * cell, y - h / 2), cell, h, facecolor=CMAP(0.5 + 0.5 * v / vmax),
                               edgecolor="white", linewidth=0.4, alpha=alpha, zorder=3))
    return x + len(vals) * cell


def label(ax, x, y, text, state, size=FS):
    """Stage name. This session's stages: blue with an underline, not bold - bold makes
    "Embedding" and "Attention" collide when both are lit (L24)."""
    color = {"now": POPBLUE, "done": INK, "later": "#AAAAAA"}[state]
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=6)
    if state == "now":
        ax.plot([x - 0.22, x + 0.22], [y - 0.11, y - 0.11], color=POPBLUE, lw=1.6,
                solid_capstyle="round", zorder=6)


def state_of(lit_now, lit_before):
    def state(key):
        if lit_now is None:
            return "done"
        if key in lit_now:
            return "now"
        return "done" if key in lit_before else "later"
    return state


# ----------------------------------------------------------------------------- frame 1: one forward pass
ROWS_Y = [1.86, 1.58, 1.30, 1.02, 0.74]             # one row per token, top to bottom
# Stage column extents. Label centres must sit >= ~0.62 in apart: "Embedding" and "Attention"
# are each ~0.58 in wide at 8 pt.
X = {"text": (0.03, 0.47), "tok": (0.55, 1.22), "emb": (1.30, 1.78), "attn": (1.88, 2.48),
     "mlp": (2.50, 3.02), "head": (3.20, 4.80), "dec": (4.88, 5.46)}


def draw_pass(g, state):
    fig, ax = new_canvas()
    a = {k: (1.0 if state(k) != "later" else FAINT) for k, _ in PASS_STAGES}
    for key, name in PASS_STAGES:                         # panels and stage names
        x0, x1 = X[key]
        if state(key) == "now":
            rbox(ax, x0 - 0.03, 0.50, x1 - x0 + 0.06, 1.72, HILITE, "none", r=0.06, z=0)
        label(ax, (x0 + x1) / 2, 2.36, name, state(key))
    toks = [t.strip() for t in g["tokens"]]

    # Text, then token chips with their IDs.
    x0, x1 = X["text"]
    rbox(ax, x0, 0.98, x1 - x0, 0.64, "white", GREY, alpha=a["tok"])
    for k, line in enumerate(["The cat", "sat on", "the"]):
        ax.text((x0 + x1) / 2, 1.46 - k * 0.16, line, ha="center", va="center", fontsize=FS,
                style="italic", color=INK, alpha=a["tok"], zorder=4)
    ax.text((x0 + x1) / 2, 2.36, "text", ha="center", va="center", fontsize=FS, style="italic",
            color=INK if a["tok"] == 1 else "#AAAAAA")
    for k, (t, i, y) in enumerate(zip(toks, g["token_ids"], ROWS_Y)):
        arrow(ax, (x1 + 0.01, 1.30), (0.55, y), alpha=a["tok"], lw=0.5, color="#AAAAAA")
        rbox(ax, 0.56, y - 0.09, 0.34, 0.18, CHIP_TINTS[k % 3], "#999999", lw=0.5, alpha=a["tok"], z=3)
        ax.text(0.73, y, t, ha="center", va="center", fontsize=FS, color=INK, alpha=a["tok"], zorder=4)
        ax.text(0.93, y, str(i), ha="left", va="center", fontsize=FS_SMALL, family="monospace",
                color=GREY, alpha=a["tok"], zorder=4)

    # Embedding: each ID becomes a vector (first 8 of GPT-2's 768 dimensions, real values).
    emb = np.array(g["embedding_dims_0_7"])
    vmax = np.abs(emb).max()
    for vals, y in zip(emb, ROWS_Y):
        arrow(ax, (1.21, y), (X["emb"][0] - 0.01, y), alpha=a["emb"], lw=0.5)
        strip(ax, X["emb"][0], y, vals, cell=0.06, alpha=a["emb"], vmax=vmax)

    # The stack: N layers, each attention (arcs from the last token) then an MLP per token.
    sx0, sx1 = X["attn"][0], X["mlp"][1]
    layer_alpha = max(a["attn"], a["mlp"])
    for d in (0.08, 0.04):                                 # the layers behind
        rbox(ax, sx0 + d, 0.56 + d, sx1 - sx0, 1.50, "#F4F4F4", "#BBBBBB", lw=0.6, r=0.05,
             alpha=layer_alpha, z=1)
    rbox(ax, sx0, 0.56, sx1 - sx0, 1.50, "white", "#888888", lw=0.8, r=0.05, alpha=layer_alpha, z=2)
    ax.text(sx1 + 0.07, 2.12, r"$\times N$", ha="left", va="center", fontsize=FS, color=GREY,
            alpha=layer_alpha, zorder=6)
    dot_x = X["attn"][0] + 0.10
    w_att = np.array(g["attention_from_last"])
    for k, y in enumerate(ROWS_Y):
        arrow(ax, (X["emb"][1] + 0.01, y), (dot_x - 0.05, y), alpha=a["attn"], lw=0.5)
        ax.plot(dot_x, y, "o", ms=3.2, color=ARMBLUE, alpha=a["attn"], zorder=5)
    for k, (y, wk) in enumerate(zip(ROWS_Y[:-1], w_att[:-1])):   # last token looks back
        ax.add_patch(FancyArrowPatch((dot_x + 0.02, ROWS_Y[-1]), (dot_x + 0.02, y),
                                     connectionstyle="arc3,rad=0.55", arrowstyle="-",
                                     lw=0.4 + 5.0 * wk, color=ARMRED, alpha=a["attn"] * 0.85,
                                     zorder=4))
    for y in ROWS_Y:                                       # MLP: widen 4x, then narrow back
        m0, m1 = X["mlp"][0] + 0.06, X["mlp"][1] - 0.06
        arrow(ax, (X["attn"][1] - 0.06, y), (m0, y), alpha=a["mlp"], lw=0.5)
        mid = (m0 + m1) / 2
        ax.add_patch(Polygon([(m0, y - 0.035), (mid, y - 0.10), (m1, y - 0.035), (m1, y + 0.035),
                              (mid, y + 0.10), (m0, y + 0.035)], closed=True, facecolor="#FCE9B8",
                             edgecolor=ARMORANGE, lw=0.6, alpha=a["mlp"], zorder=3))

    # LM head: only the last token's vector is read; it becomes a distribution over 50,257 tokens.
    yl = ROWS_Y[-1]
    arrow(ax, (sx1 + 0.01, yl), (X["head"][0] + 0.02, yl), alpha=a["head"], lw=0.6)
    end = strip(ax, X["head"][0] + 0.04, yl, g["last_hidden_dims_0_7"][:6], alpha=a["head"])
    hx0 = end + 0.04
    ax.add_patch(Polygon([(hx0, yl - 0.07), (hx0 + 0.16, 0.66), (hx0 + 0.16, 1.96), (hx0, yl + 0.07)],
                         closed=True, facecolor="#CFD8EE", edgecolor=ARMBLUE, lw=0.6,
                         alpha=a["head"], zorder=3))
    top = g["top5"]
    bx0, bmax = hx0 + 0.62, 0.42
    for k, (t, p) in enumerate(top):
        y = 1.86 - k * 0.27
        ax.text(bx0 - 0.04, y, t.strip(), ha="right", va="center", fontsize=FS_SMALL, color=INK,
                alpha=a["head"], zorder=4)
        ax.add_patch(Rectangle((bx0, y - 0.07), bmax * p / top[0][1], 0.14,
                               facecolor=ARMRED if k == 0 else "#E8A0A6", alpha=a["head"], zorder=3))
        ax.text(bx0 + bmax * p / top[0][1] + 0.03, y, f"{100 * p:.0f}%", ha="left", va="center",
                fontsize=FS_SMALL, color=GREY, alpha=a["head"], zorder=4)

    # Decoding: pick one token, append it, run again.
    dx = (X["dec"][0] + X["dec"][1]) / 2
    rbox(ax, dx - 0.26, 1.77, 0.52, 0.18, "white", ARMRED, lw=1.0, alpha=a["dec"], z=3)
    ax.text(dx, 1.86, top[0][0].strip(), ha="center", va="center", fontsize=FS, color=ARMRED,
            fontweight="bold", alpha=a["dec"], zorder=4)
    ax.text(dx, 1.58, "pick one", ha="center", va="center", fontsize=FS_SMALL, color=GREY,
            alpha=a["dec"], zorder=4)
    loop = [(dx, 1.76), (dx, 0.30), (0.27, 0.30), (0.27, 0.97)]
    ax.plot([p[0] for p in loop[:-1]], [p[1] for p in loop[:-1]], color=ARMRED, lw=0.8,
            alpha=a["dec"] * 0.8, zorder=2)
    arrow(ax, loop[-2], loop[-1], alpha=a["dec"] * 0.8, color=ARMRED, lw=0.8)
    ax.text(2.6, 0.20, "append it to the text and run again", ha="center", va="center",
            fontsize=FS_SMALL, color=ARMRED, alpha=a["dec"], zorder=4)
    return fig


# ----------------------------------------------------------------------------- frame 2: the life of a model
CARD_W, CARD_GAP, CARD_Y, CARD_H = 0.68, 0.11, 0.30, 1.62
GROUPS = [("make it", 0, 2), ("use it", 3, 4), ("run and adapt it", 5, 6)]
FACTS = {"pre": ["next token on", "~15T tokens"], "scale": ["loss falls as", "a power law"],
         "post": ["base model", "to assistant"], "prompt": ["steer it,", "no training"],
         "eval": ["benchmarks", "and judges"], "quant": ["8B model:", "16 → 5 GB"],
         "lora": ["train < 1%", "of weights"]}       # <= 13 narrow characters per line


def icon_pre(ax, cx, cy, al):
    for k in range(4):
        x, y = cx - 0.20 + 0.05 * k, cy - 0.20 + 0.035 * k
        rbox(ax, x, y, 0.30, 0.36, "white", "#999999", lw=0.5, r=0.02, alpha=al, z=3 + k)
        for j in range(4):
            ax.plot([x + 0.04, x + 0.26 - 0.06 * (j == 3)], [y + 0.28 - 0.07 * j] * 2,
                    color="#BBBBBB", lw=0.6, alpha=al, zorder=3 + k)


def icon_scale(ax, cx, cy, al):
    x0, y0, s = cx - 0.24, cy - 0.20, 0.44
    ax.plot([x0, x0, x0 + s], [y0 + s * 0.9, y0, y0], color="#888888", lw=0.6, alpha=al, zorder=3)
    xs = np.linspace(0.08, 0.92, 5)
    ax.plot(x0 + xs * s, y0 + (0.82 - 0.72 * xs) * s, color=ARMBLUE, lw=1.0, alpha=al, zorder=4)
    ax.plot(x0 + xs * s, y0 + (0.82 - 0.72 * xs + np.array([.04, -.03, .02, -.02, .03])) * s, "o",
            ms=2.2, color=ARMRED, alpha=al, zorder=5)


def icon_post(ax, cx, cy, al):
    rbox(ax, cx - 0.27, cy + 0.02, 0.36, 0.17, "#EEEEEE", "#AAAAAA", lw=0.5, r=0.05, alpha=al, z=3)
    rbox(ax, cx - 0.09, cy - 0.21, 0.36, 0.17, "#CFD8EE", ARMBLUE, lw=0.5, r=0.05, alpha=al, z=3)
    for x, y in ((cx - 0.22, cy + 0.105), (cx - 0.04, cy - 0.125)):
        ax.plot([x, x + 0.24], [y, y], color="#999999", lw=0.6, alpha=al, zorder=4)
    ax.text(cx - 0.20, cy - 0.13, "✓", ha="center", va="center", fontsize=FS, color="#1A8A3A",
            alpha=al, zorder=4)
    ax.text(cx + 0.22, cy + 0.11, "✗", ha="center", va="center", fontsize=FS, color=ARMRED,
            alpha=al, zorder=4)


def icon_prompt(ax, cx, cy, al):
    rbox(ax, cx - 0.26, cy - 0.22, 0.52, 0.44, "white", "#999999", lw=0.5, r=0.03, alpha=al, z=3)
    for k, (q, w) in enumerate([("Q", 0.24), ("A", 0.16), ("Q", 0.20), ("A", None)]):
        y = cy + 0.15 - 0.10 * k
        ax.text(cx - 0.21, y, q, ha="left", va="center", fontsize=FS_SMALL, family="monospace",
                color=ARMBLUE if q == "A" else GREY, alpha=al, zorder=4)
        if w:
            ax.plot([cx - 0.11, cx - 0.11 + w], [y, y], color="#BBBBBB", lw=0.6, alpha=al, zorder=4)
        else:
            ax.plot([cx - 0.10, cx - 0.10], [y - 0.04, y + 0.04], color=INK, lw=0.8, alpha=al, zorder=4)


def icon_eval(ax, cx, cy, al):
    x0, y0 = cx - 0.24, cy - 0.20
    for k, (hgt, c) in enumerate([(0.30, ARMBLUE), (0.20, ARMBLUE), (0.36, ARMBLUE), (0.12, ARMRED)]):
        ax.add_patch(Rectangle((x0 + 0.03 + 0.12 * k, y0), 0.08, hgt, facecolor=c, alpha=al * 0.85,
                               zorder=3))
    ax.plot([x0, x0 + 0.50], [y0 + 0.27] * 2, color=GREY, lw=0.6, ls=(0, (2, 1.5)), alpha=al, zorder=4)
    ax.plot([x0, x0 + 0.50], [y0, y0], color="#888888", lw=0.6, alpha=al, zorder=4)


def icon_quant(ax, cx, cy, al):
    rng = np.random.default_rng(509)
    w = rng.standard_normal((4, 4))
    q = np.round(w / np.abs(w).max() * 1.5) / 1.5          # 2 bit: four levels, labelled so
    for grid, x0, c in ((w / np.abs(w).max(), cx - 0.28, 0.06), (q, cx + 0.06, 0.05)):
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((x0 + j * c, cy + 0.10 - i * c), c, c,
                                       facecolor=CMAP(0.5 + 0.5 * grid[i, j]), edgecolor="white",
                                       linewidth=0.3, alpha=al, zorder=3))
    arrow(ax, (cx - 0.03, cy + 0.03), (cx + 0.05, cy + 0.03), alpha=al, lw=0.6)
    ax.text(cx - 0.16, cy - 0.17, "16 bit", ha="center", va="center", fontsize=FS_SMALL, color=GREY,
            alpha=al, zorder=4)
    ax.text(cx + 0.16, cy - 0.17, "2 bit", ha="center", va="center", fontsize=FS_SMALL, color=GREY,
            alpha=al, zorder=4)


def icon_lora(ax, cx, cy, al):
    rbox(ax, cx - 0.29, cy - 0.17, 0.32, 0.32, "#E6E6E6", "#999999", lw=0.5, r=0.01, alpha=al, z=3)
    ax.text(cx - 0.13, cy - 0.01, "W", ha="center", va="center", fontsize=FS, color=GREY, alpha=al,
            zorder=4)
    ax.text(cx + 0.06, cy - 0.01, "+", ha="center", va="center", fontsize=FS, color=INK, alpha=al,
            zorder=4)
    ax.add_patch(Rectangle((cx + 0.11, cy - 0.17), 0.05, 0.32, facecolor=ARMORANGE, alpha=al, zorder=3))
    ax.add_patch(Rectangle((cx + 0.18, cy + 0.10), 0.12, 0.05, facecolor=ARMBLUE, alpha=al, zorder=3))
    ax.text(cx + 0.135, cy - 0.24, "B", ha="center", va="center", fontsize=FS_SMALL, color=INK,
            alpha=al, zorder=4)
    ax.text(cx + 0.24, cy + 0.02, "A", ha="center", va="center", fontsize=FS_SMALL, color=INK,
            alpha=al, zorder=4)


ICONS = {"pre": icon_pre, "scale": icon_scale, "post": icon_post, "prompt": icon_prompt,
         "eval": icon_eval, "quant": icon_quant, "lora": icon_lora}


def draw_life(state):
    fig, ax = new_canvas()
    x_start = (W - 7 * CARD_W - 6 * CARD_GAP) / 2
    xs = [x_start + k * (CARD_W + CARD_GAP) for k in range(7)]
    for name, i0, i1 in GROUPS:                            # brackets over the groups
        x0, x1 = xs[i0] + 0.03, xs[i1] + CARD_W - 0.03
        ax.plot([x0, x0, x1, x1], [2.10, 2.16, 2.16, 2.10], color=GREY, lw=0.7, zorder=2)
        ax.text((x0 + x1) / 2, 2.30, name, ha="center", va="center", fontsize=FS, style="italic",
                color="#555555")
    for k, ((key, name), x) in enumerate(zip(LIFE_STAGES, xs)):
        st = state(key)
        al = 1.0 if st != "later" else FAINT
        if st == "now":
            rbox(ax, x, CARD_Y, CARD_W, CARD_H, HILITE, POPBLUE, lw=1.4, r=0.06, z=1)
        else:
            rbox(ax, x, CARD_Y, CARD_W, CARD_H, "white", "#BBBBBB", lw=0.7, r=0.06, alpha=al, z=1)
        ICONS[key](ax, x + CARD_W / 2, 1.42, al)
        label(ax, x + CARD_W / 2, 0.92, name, st)
        for j, line in enumerate(FACTS[key]):
            ax.text(x + CARD_W / 2, 0.70 - 0.15 * j, line, ha="center", va="center",
                    fontsize=FS_SMALL, color=GREY if st != "later" else "#BBBBBB", zorder=4)
        if k < 6:
            arrow(ax, (x + CARD_W + 0.01, 1.11), (x + CARD_W + CARD_GAP - 0.01, 1.11), lw=0.7)
    return fig


# ----------------------------------------------------------------------------- output
def save(fig, out, log):
    # Back-to-back PDF writes hit a transient Windows lock (OSError errno 22) now and then;
    # see _learnings/2026-09-28-2313_errno22-on-write-is-a-transient-lock.md. Bounded retry,
    # then fail loudly.
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


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        log.info(f"done in {time.time() - t0:.0f} s")
        return
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"{JSON_PATH} missing - run with --gpt2 first")
    g = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    keys = {k for k, _ in PASS_STAGES + LIFE_STAGES}
    for name, _, lit in SESSIONS:
        assert lit <= keys, f"{name}: unknown stage in {lit}"
    FIG.mkdir(parents=True, exist_ok=True)

    jobs = [("overview", state_of(None, set()))]
    before = set()
    for name, _, lit in SESSIONS:
        jobs.append((name, state_of(lit, set(before))))
        before |= lit
    for k, (name, state) in enumerate(jobs, 1):
        save(draw_pass(g, state), FIG / f"roadmap_{name}_pass.pdf", log)
        save(draw_life(state), FIG / f"roadmap_{name}_life.pdf", log)
        elapsed = time.time() - t0
        log.info(f"{k}/{len(jobs)} {name}, {elapsed:.0f}s elapsed, ~{elapsed / k * (len(jobs) - k):.0f}s left")
    log.info(f"{2 * len(jobs)} maps in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
