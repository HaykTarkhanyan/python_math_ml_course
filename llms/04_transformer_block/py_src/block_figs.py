"""Figures for LLM-4 "The Transformer block" (llms/04_transformer_block/LLM4_transformer_block.tex).

Toy, demonstrative numbers for every by-hand frame (ml/SLIDE_STYLE.md, 2026-10-05): LLM-3's toy
vectors read as [adjective, fluffy, blue, noun], plus a fifth slot "cute" for the MLP neuron.
Every number the slides quote is asserted (QUOTED, TOY), so a changed toy fails loudly.

Real numbers only for claims about GPT-2, from what is local:
  --gpt2   one run of the cached GPT-2 small (HF_HUB_OFFLINE=1, no download): the four fact
           prompts of the cold open (top-5 next tokens), the GPT-2 tokens of the names, the
           attention weights of every head on the creature sentence, and the parameter counts
           from the safetensors header -> results/block_gpt2.json (~1-2 min, mostly importing torch)
  (default) the figures, reading that JSON (~5 s)

Drawn at the size each figure gets on the slide (5.5 in = full text width), 6.5-8 pt text, and
save() refuses any text outside its figure.

Generates into llms/04_transformer_block/fig/:
  blk_fact_probes.pdf    -- cold open: GPT-2's top-5 next words for the four prompts
  blk_two_heads.pdf      -- two toy heads ("adjectives?", "what colour?") and W_O placing their outputs
  blk_gpt2_entropy.pdf   -- attention entropy of the 12 heads in GPT-2's layer 5 (creature sentence)
  blk_gpt2_heads.pdf     -- three real heads: previous token, creature -> fluffy, the first token
  blk_gpt2_sink.pdf      -- share of attention on the first token, per layer (attention sinks)
  blk_permutation.pdf    -- one head is blind to word order; with positions added it is not
  blk_neuron.pdf         -- the toy MLP neuron as an AND gate: w.x + b for four inputs, ReLU at 0
  blk_params_block.pdf   -- one GPT-2 block: attention vs MLP vs norms
  blk_params_124m.pdf    -- where GPT-2's 124,439,808 parameters live
  blk_block.pdf          -- the block in GPT-2's order: norm, attention, add; norm, MLP, add
  (blk_causal_mask and blk_mask_grid moved to LLM-3 on 2026-10-28, as py_src/masking_figs.py there.)

Run:
  ./ma/Scripts/python.exe llms/04_transformer_block/py_src/block_figs.py --gpt2   # once (25 s measured, 1-2 min cold)
  ./ma/Scripts/python.exe llms/04_transformer_block/py_src/block_figs.py          # figures (~7 s measured)
"""
import json
import logging
import math
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
GPT2_JSON = RESULTS / "block_gpt2.json"

# ----------------------------------------------------------------------------- inputs
PROBES = [("Lionel Messi plays the sport of", " football"),
          ("Henrikh Mkhitaryan plays the sport of", " football"),
          ("Ella Fitzgerald was a famous American", " singer"),
          ("Elliott Smith was a famous American", " singer")]
NAMES = ["Lionel Messi", "Henrikh Mkhitaryan", "Ella Fitzgerald", "Elliott Smith"]
SENTENCE = "a fluffy blue creature roamed the verdant forest"

# Real numbers the slides quote (from the probe runs of 2026-10-05; --gpt2 must reproduce them).
QUOTED = {
    "probe_top": {"Lionel Messi plays the sport of": [" football", 35.1],
                  "Henrikh Mkhitaryan plays the sport of": [" hockey", 9.8],
                  "Ella Fitzgerald was a famous American": [" actress", 15.4],
                  "Elliott Smith was a famous American": [" writer", 4.6]},
    "probe_right": {"Lionel Messi plays the sport of": 35.1, "Henrikh Mkhitaryan plays the sport of": 4.9,
                    "Ella Fitzgerald was a famous American": 6.6, "Elliott Smith was a famous American": 2.9},
    "params": {"attention": 2_362_368, "mlp": 4_722_432, "norms": 3_072, "block": 7_087_872,
               "blocks12": 85_054_464, "wte": 38_597_376, "wpe": 786_432, "ln_f": 1_536,
               "total": 124_439_808, "mlp_share": 66.6, "wte_share": 31.0, "attn_share_model": 22.8},
}

# The toy (demonstrative). LLM-3's vectors; slots [adjective, fluffy, blue, noun].
SLOTS = ["adjective", "fluffy", "blue", "noun"]
TOKENS = ["fluffy", "blue", "creature"]
X = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 2]], dtype=float)
TOY = {
    # two heads, one number each: head 1 = LLM-3's "adjectives?", head 2 = "what colour?"
    "h1_weights": [0.47, 0.47, 0.06], "h2_weights": [0.11, 0.79, 0.11],
    "h1_out": 0.47, "h2_out": 0.79, "two_heads_update": [0, 0.47, 0.79, 0],
    # the MLP neuron on creature after LLM-3 (printed values), slots + "cute"
    "creature_after_llm3": [0, 0.47, 0.47, 2.13, 0], "neuron_w": [0, 1, 0, 0.5, 0], "neuron_b": -1.2,
    "neuron_pre": {"fluffy blue creature": 0.335, "plain creature": -0.2, "fluffy alone": -0.2,
                   "blue alone": -1.2},
    "neuron_down": 3, "cute": 1.0,
    # LayerNorm on creature's four numbers (LLM-3's printed result)
    "ln_in": [0, 0.47, 0.47, 2.13], "ln_mean": 0.77, "ln_spread": 0.81,
    "ln_out": [-0.95, -0.37, -0.37, 1.68],
    # induction head (added 2026-10-27): score 3 where a key's "preceded by" matches the query's token
    "induction": {"tokens": ["Messi", "plays", "football", ".", "Messi", "plays"], "match_score": 3,
                  "w_match": 0.80, "w_other": 0.04, "predicts": "football"},
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
    lg = logging.getLogger("block_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "block_figs.log", encoding="utf-8")):
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


def r2(v):
    return [round(float(x), 2) for x in np.ravel(v)]


def check(name, got, want):
    if r2(got) != r2(want):
        raise AssertionError(f"{name}: computed {r2(got)}, slides quote {r2(want)}")


def softmax_rows(S):
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def num(v):
    """0.50 -> '.5', 2.0 -> '2', 0.11 -> '.11'."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    s = s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)
    return s or "0"


def show_tok(t):
    return t.replace(" ", "␣")


# ----------------------------------------------------------------------------- GPT-2 (only --gpt2)
def compute_gpt2(log):
    import os
    import struct
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"), "probes": {},
           "name_tokens": {}}
    with torch.no_grad():
        for prompt, right in PROBES:
            ids = tok(prompt, return_tensors="pt").input_ids
            probs = model(ids).logits[0, -1].softmax(-1)
            top = probs.topk(5)
            res["probes"][prompt] = {
                "top5": [[tok.decode([int(i)]), round(100 * float(v), 1)] for v, i in zip(top.values, top.indices)],
                "right": right, "right_pct": round(100 * float(probs[tok.encode(right)[0]]), 1)}
            log.info(f"{prompt!r}: {res['probes'][prompt]['top5']}")
        for name in NAMES:
            res["name_tokens"][name] = [tok.decode([i]) for i in tok.encode(" " + name)]
        ids = tok(SENTENCE, return_tensors="pt").input_ids
        out = model(ids, output_attentions=True)
    res["sentence_tokens"] = [tok.decode([i]) for i in ids[0].tolist()]
    res["attentions"] = [[[[round(float(x), 4) for x in row] for row in head] for head in layer[0]]
                         for layer in out.attentions]            # [layer][head][query][key]
    st = next(Path(model.config._name_or_path).glob("model.safetensors"), None)
    if st is None:
        from huggingface_hub import hf_hub_download
        st = Path(hf_hub_download("gpt2", "model.safetensors"))     # offline: resolves to the cache
    with open(st, "rb") as f:
        hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    shapes = {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}
    count = lambda keys: sum(math.prod(shapes[k]) for k in keys)
    blk = [k for k in shapes if k.startswith("h.0.") and not k.endswith((".attn.bias", ".attn.masked_bias"))]
    res["params"] = {
        "attention": count([k for k in blk if k.startswith("h.0.attn.")]),
        "mlp": count([k for k in blk if k.startswith("h.0.mlp.")]),
        "norms": count([k for k in blk if k.startswith("h.0.ln_")]),
        "wte": count(["wte.weight"]), "wpe": count(["wpe.weight"]), "ln_f": count(["ln_f.weight", "ln_f.bias"]),
        "n_layer": model.config.n_layer,
        "total": count([k for k in shapes if not k.endswith((".attn.bias", ".attn.masked_bias"))])}
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    log.info(f"name tokens: {res['name_tokens']}; sentence tokens: {res['sentence_tokens']}")
    log.info(f"params: {res['params']}; wrote {GPT2_JSON.name}")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    G = json.loads(GPT2_JSON.read_text(encoding="utf-8"))
    for prompt, (word, pct) in QUOTED["probe_top"].items():
        got = G["probes"][prompt]["top5"][0]
        if got[0] != word or got[1] != pct:
            raise AssertionError(f"{prompt!r}: GPT-2's top word is {got}, slides quote {[word, pct]}")
        if G["probes"][prompt]["right_pct"] != QUOTED["probe_right"][prompt]:
            raise AssertionError(f"{prompt!r}: right answer at {G['probes'][prompt]['right_pct']}%, "
                                 f"slides quote {QUOTED['probe_right'][prompt]}%")
    P, Q = G["params"], QUOTED["params"]
    block = P["attention"] + P["mlp"] + P["norms"]
    got = {"attention": P["attention"], "mlp": P["mlp"], "norms": P["norms"], "block": block,
           "blocks12": block * P["n_layer"], "wte": P["wte"], "wpe": P["wpe"], "ln_f": P["ln_f"],
           "total": P["total"], "mlp_share": round(100 * P["mlp"] / block, 1),
           "wte_share": round(100 * P["wte"] / P["total"], 1),
           "attn_share_model": round(100 * P["attention"] * P["n_layer"] / P["total"], 1)}
    if got != Q:
        raise AssertionError(f"GPT-2 parameter counts {got} != quoted {Q}")
    if block * P["n_layer"] + P["wte"] + P["wpe"] + P["ln_f"] != P["total"]:
        raise AssertionError("the parts do not add up to the total")
    # Slide "Run it h times": 12 heads x 3 x (768 x 64) = 3 x 768^2 (one wide head's W_Q, W_K, W_V),
    # plus W_O 768 x 768, plus biases 2,304 + 768 = the attention bar.
    d, h, dk = 768, 12, 64
    if (h * 3 * d * dk, d * d, h * 3 * d * dk + d * d + 3 * d + d) != (1_769_472, 589_824, P["attention"]):
        raise AssertionError("multi-head parameter count does not match the attention bar")
    if (h * 3 * d * dk + d * d, P["attention"] - 4 * d * d) != (2_359_296, 3_072):   # the slide's total
        raise AssertionError("the slide quotes 4 x 768^2 = 2,359,296 weights plus 3,072 biases")
    return G


# ----------------------------------------------------------------------------- the toy, computed
def toy(R, log):
    T = TOY
    # two heads, one number each. Queries read "noun"; head 1's key reads "adjective", head 2's "blue".
    q = X @ np.array([0, 0, 0, 1.0])
    k1, k2 = X @ np.array([1.0, 0, 0, 0]), X @ np.array([0, 0, 1.0, 0])
    A1, A2 = softmax_rows(np.outer(q, k1)), softmax_rows(np.outer(q, k2))
    v1, v2 = X @ np.array([0, 1.0, 0, 0]), X @ np.array([0, 0, 1.0, 0])    # head 1 hands over "fluffy", head 2 "blue"
    h1, h2 = float(A1[2] @ v1), float(A2[2] @ v2)
    W_O = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=float)           # places h1 in "fluffy", h2 in "blue"
    check("h1 weights", A1[2], T["h1_weights"])
    check("h2 weights", A2[2], T["h2_weights"])
    check("h1 out", [h1], [T["h1_out"]])
    check("h2 out", [h2], [T["h2_out"]])
    check("two-head update", W_O @ np.array([h1, h2]), T["two_heads_update"])
    # the neuron (printed creature after LLM-3, plus "cute" = 0)
    w, b = np.array(T["neuron_w"]), T["neuron_b"]
    inputs = {"fluffy blue creature": T["creature_after_llm3"], "plain creature": [0, 0, 0, 2, 0],
              "fluffy alone": [1, 1, 0, 0, 0], "blue alone": [1, 0, 1, 0, 0]}
    pre = {k: float(w @ np.array(v) + b) for k, v in inputs.items()}
    for k, v in pre.items():                 # printed with 3 decimals (0.335), so compare at 3
        if round(v, 3) != T["neuron_pre"][k]:
            raise AssertionError(f"neuron {k}: computed {v:.4f}, slides quote {T['neuron_pre'][k]}")
    fires = {k: max(v, 0.0) for k, v in pre.items()}
    if [k for k, v in fires.items() if v > 0] != ["fluffy blue creature"]:
        raise AssertionError(f"the neuron should fire only on the fluffy blue creature: {fires}")
    if round(T["neuron_down"] * fires["fluffy blue creature"], 1) != T["cute"]:
        raise AssertionError("down-projection: 3 x 0.335 should print as 1.0")
    # LayerNorm (population variance, as LayerNorm does)
    x = np.array(T["ln_in"])
    mu, sd = x.mean(), x.std()
    check("ln mean", [mu], [T["ln_mean"]])
    check("ln spread", [sd], [T["ln_spread"]])
    check("ln out", (x - mu) / sd, T["ln_out"])
    R["toy"] = {"A1": A1.tolist(), "A2": A2.tolist(), "h": [h1, h2], "neuron_pre": pre,
                "ln": ((x - mu) / sd).tolist()}
    log.info(f"toy: heads {round(h1, 3)}, {round(h2, 3)}; neuron {pre}; LN {r2((x - mu) / sd)}")
    return A1, A2, pre


# ----------------------------------------------------------------------------- figures
def fig_fact_probes(G, log):
    fig, axes = plt.subplots(2, 2, figsize=(W, 2.35))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.08, wspace=0.55, hspace=0.75)
    for ax, (prompt, right) in zip(axes.ravel(), PROBES):
        top = G["probes"][prompt]["top5"]
        words = [t for t, _ in top][::-1]
        pcts = [p for _, p in top][::-1]
        cols = [GREEN if w == right else LIGHT for w in words]
        bars = ax.barh([show_tok(w) for w in words], pcts, color=cols, height=0.7)
        ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pcts], fontsize=FS_TINY, padding=2)
        ax.set_xlim(0, max(pcts) * 1.35)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=FS_TINY, length=0)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(f'"{prompt} ..."', fontsize=FS_TINY, loc="left", pad=3)
    save(fig, "blk_fact_probes.pdf", log)


def fig_two_heads(A1, A2, log):
    fig, ax = canvas(1.45, 3.1)
    cw, ch = 0.32, 0.26

    def row(x0, y, weights, label, key_note):
        ax.text(x0 - 0.06, y + ch / 2, label, ha="right", va="center", fontsize=FS_TINY, color=INK)
        cm = plt.get_cmap("Blues")
        for j, (t, w) in enumerate(zip(TOKENS, weights)):
            ax.add_patch(Rectangle((x0 + j * cw, y), cw, ch, facecolor=cm(0.1 + 0.55 * w),
                                   edgecolor="white", linewidth=0.8))
            ax.text(x0 + (j + 0.5) * cw, y + ch / 2, num(w), ha="center", va="center", fontsize=FS_TINY)
        ax.text(x0 + 3 * cw + 0.08, y + ch / 2, key_note, ha="left", va="center", fontsize=FS_TINY, color=GREY)

    x0 = 1.08
    for j, t in enumerate(TOKENS):
        ax.text(x0 + (j + 0.5) * cw, 1.3, t, ha="center", va="bottom", fontsize=FS_TINY)
    row(x0, 0.96, A1[2], 'head 1, "adjectives?"', "fluffy .47")
    row(x0, 0.6, A2[2], 'head 2, "what colour?"', "blue .79")
    ax.text(0.05, 0.04, "W_O puts both numbers into the stream:\n"
            "(adjective, fluffy, blue, noun)  +=  (0, .47, .79, 0)", fontsize=FS_TINY, color=INK)
    save(fig, "blk_two_heads.pdf", log)


def head_entropy(A):
    A = np.array(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(A > 0, A * np.log(A), 0.0), axis=1)
    return float(H[1:].mean())                         # skip row 0: it has one token to look at


# Real GPT-2 heads on SENTENCE (1-indexed layers, as the slides say them). Picked after measuring
# (2026-10-05): a previous-token head, a head where creature looks at fluffy, an attention sink.
PREV_HEAD, ADJ_HEAD = (5, 11), (6, 2)
GPT2_QUOTED = {"prev_weight": 1.0, "creature_to_fluffy": 0.62, "sink_heads_over_90": 39, "heads": 144,
               "first_share_layer1": 0.22, "first_share_layer10": 0.88, "entropy_layer": 5}


def fig_gpt2_heads(G, R, log):
    att = np.array(G["attentions"])                      # [layer][head][query][key]
    L, H, n, _ = att.shape
    toks = G["sentence_tokens"]
    cr, fl = toks.index(" creature"), toks.index(" fluffy")
    ent = np.array([[head_entropy(att[l, h]) for h in range(H)] for l in range(L)])
    first = att[:, :, 1:, 0].mean(axis=2)                 # share on token 0, rows 1..n-1
    prev = np.array([[np.mean([att[l, h, i, i - 1] for i in range(1, n)]) for h in range(H)] for l in range(L)])
    (pl, ph), (al, ah) = PREV_HEAD, ADJ_HEAD
    sl, sh = np.unravel_index(int(np.argmax(first)), first.shape)
    got = {"prev_weight": round(float(prev[pl - 1, ph]), 2),
           "creature_to_fluffy": round(float(att[al - 1, ah, cr, fl]), 2),
           "sink_heads_over_90": int((first > 0.9).sum()), "heads": L * H,
           "first_share_layer1": round(float(first[0].mean()), 2),
           "first_share_layer10": round(float(first[9].mean()), 2), "entropy_layer": pl}
    if got != GPT2_QUOTED:
        raise AssertionError(f"GPT-2 heads: measured {got}, slides quote {GPT2_QUOTED}")
    R["gpt2_heads"] = {**got, "sink_head": [int(sl) + 1, int(sh)], "sink_share": round(float(first[sl, sh]), 2),
                       "entropy": ent[pl - 1].round(2).tolist(), "tokens": toks,
                       "first_share_by_layer": first.mean(axis=1).round(2).tolist()}
    log.info(f"GPT-2 heads: {R['gpt2_heads']}")
    # entropy chart for the previous-token head's layer
    e = ent[pl - 1]
    fig, ax = plt.subplots(figsize=(2.9, 1.7))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.22)
    bars = ax.bar(range(H), e, color=[ARMRED if i == ph else ARMBLUE for i in range(H)], width=0.7)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in e], fontsize=FS_TINY - 0.5, padding=1)
    ax.set_xticks(range(H))
    ax.tick_params(axis="x", labelsize=FS_TINY, length=0)
    ax.tick_params(axis="y", labelsize=FS_TINY)
    ax.set_xlabel(f"head (GPT-2, layer {pl} of {L})", fontsize=FS_TINY)
    ax.set_ylabel("entropy", fontsize=FS_TINY)
    ax.set_ylim(0, e.max() * 1.3)
    clean(ax)
    save(fig, "blk_gpt2_entropy.pdf", log)
    # three heads
    shown = [show_tok(t) for t in toks]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.2))
    fig.subplots_adjust(left=0.09, right=0.995, top=0.82, bottom=0.25, wspace=0.12)
    panels = [((pl, ph), f"layer {pl}, head {ph}\nthe previous token"),
              ((al, ah), f"layer {al}, head {ah}\ncreature -> fluffy .62"),
              ((int(sl) + 1, int(sh)), f"layer {int(sl) + 1}, head {int(sh)}\nthe first token")]
    for k, (ax, ((l, h), title)) in enumerate(zip(axes, panels)):
        ax.imshow(att[l - 1, h], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(shown, rotation=70, ha="right", fontsize=FS_TINY - 1)
        ax.set_yticklabels(shown if k == 0 else [], fontsize=FS_TINY - 1)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=FS_TINY, pad=3)
    save(fig, "blk_gpt2_heads.pdf", log)


def fig_sink(G, log):
    """Share of attention on the very first token, per layer (rows 1..n-1, mean over heads)."""
    att = np.array(G["attentions"])
    share = att[:, :, 1:, 0].mean(axis=(1, 2))
    check("first-token share, layers 1 and 10", [share[0], share[9]],
          [GPT2_QUOTED["first_share_layer1"], GPT2_QUOTED["first_share_layer10"]])
    fig, ax = plt.subplots(figsize=(2.75, 1.6))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.24)
    bars = ax.bar(range(1, len(share) + 1), 100 * share, color=ARMBLUE, width=0.7)
    ax.bar_label(bars, labels=[f"{100 * v:.0f}" for v in share], fontsize=FS_TINY - 0.5, padding=1)
    ax.set_xticks(range(1, len(share) + 1))
    ax.tick_params(axis="x", labelsize=FS_TINY, length=0)
    ax.tick_params(axis="y", labelsize=FS_TINY)
    ax.set_ylim(0, 105)
    ax.set_xlabel("GPT-2 layer", fontsize=FS_TINY)
    ax.set_ylabel("% on the first token", fontsize=FS_TINY)
    clean(ax)
    save(fig, "blk_gpt2_sink.pdf", log)


def positional_encoding(n_pos, d_model):
    """The sinusoidal encoding from Vaswani et al. (2017), as in l25_l26_figs.py."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def fig_permutation(log):
    """Ported from l25_l26_figs.py (same seed, same numbers) and redrawn at slot size: one head
    gives "dog" the same output in both word orders; with positions added, it does not."""
    rng = np.random.default_rng(SEED)
    d_model, d_k = 16, 8
    words = ["dog", "bites", "man"]
    E = {w: rng.normal(size=d_model) for w in words}
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) * 0.4 for _ in range(3))
    Ea = np.stack([E[w] for w in ["dog", "bites", "man"]])
    Eb = np.stack([E[w] for w in ["man", "bites", "dog"]])

    def head(Em):
        S = (Em @ Wq) @ (Em @ Wk).T / np.sqrt(d_k)
        return softmax_rows(S) @ (Em @ Wv)

    gap = float(np.abs(head(Ea)[0] - head(Eb)[2]).max())
    pe = positional_encoding(3, d_model)
    a_pe, b_pe = head(Ea + pe)[0], head(Eb + pe)[2]
    gap_pe = float(np.abs(a_pe - b_pe).max())
    if f"{gap:.1e}" != "1.1e-16" or gap_pe < 1e-3:
        raise AssertionError(f"permutation: gap {gap:.1e} (slides: 1.1e-16), with positions {gap_pe:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.2, wspace=0.08)
    x = np.arange(d_k)
    for ax, (a, b, title) in zip(axes, [(head(Ea)[0], head(Eb)[2], f"no positions: max difference {gap:.1e}"),
                                        (a_pe, b_pe, f"positions added (LLM-2): max difference {gap_pe:.2f}")]):
        ax.bar(x - 0.19, a, 0.38, color=ARMBLUE, label='"dog bites man"')
        ax.bar(x + 0.19, b, 0.38, color=ARMORANGE, label='"man bites dog"')
        ax.set_title(title, fontsize=FS_TINY, pad=3)
        ax.set_xticks(x)
        ax.tick_params(labelsize=FS_TINY, length=0)
        ax.set_xlabel('the 8 output numbers for "dog"', fontsize=FS_TINY)
        clean(ax)
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.55 * (hi - lo))             # headroom for the legend
    axes[0].legend(frameon=False, fontsize=FS_TINY, loc="upper left", ncol=2)
    save(fig, "blk_permutation.pdf", log)
    return gap, gap_pe


def fig_induction(R, log):
    """Two heads in a circuit (Olsson et al. 2022), by hand. Head A (previous token) writes into each
    token "who came before me". Head B (induction) at the last token asks "who came after my token
    last time?": its query is the current token, each key is that position's "preceded by"."""
    T = TOY["induction"]
    toks = T["tokens"]
    before = [None] + toks[:-1]                         # what head A writes into each position
    q = toks[-1]
    scores = np.array([T["match_score"] if b == q else 0.0 for b in before])
    w = np.exp(scores) / np.exp(scores).sum()           # causal: the last token sees all six
    hit = int(np.argmax(w))
    if toks[hit] != T["predicts"] or round(float(w[hit]), 2) != T["w_match"]:
        raise AssertionError(f"induction: attends to {toks[hit]} with {w[hit]:.3f}")
    others = sorted({round(float(x), 2) for i, x in enumerate(w) if i != hit})
    if others != [T["w_other"]] or abs(w.sum() - 1) > 1e-9:
        raise AssertionError(f"induction: other weights {others}")
    R["induction"] = {"preceded_by": before, "weights": w.tolist(), "predicts": toks[hit]}

    fig, ax = plt.subplots(figsize=(W, 1.6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    xs = [0.35 + 0.72 * i for i in range(len(toks))]
    y = 0.95
    for i, (x, t) in enumerate(zip(xs, toks)):
        fc = TINT_RED if i == hit else (TINT_BLUE if i == len(toks) - 1 else "#F2F2F2")
        ax.add_patch(FancyBboxPatch((x - 0.30, y - 0.13), 0.60, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=fc, ec="#999999", lw=0.6))
        ax.text(x, y, t, ha="center", va="center", fontsize=FS)
        if before[i] is not None:
            ax.add_patch(FancyArrowPatch((x - 0.22, y + 0.14), (xs[i - 1] + 0.22, y + 0.14),
                                         connectionstyle="arc3,rad=0.5", arrowstyle="-|>", mutation_scale=6,
                                         color=GREY, lw=0.6))
            ax.text(x, y - 0.24, f"after {before[i]}", ha="center", va="center", fontsize=FS_TINY,
                    color=GREY)
    ax.text(xs[0] - 0.30, 1.47, "head A (previous token): each token notes who came before it",
            ha="left", va="center", fontsize=FS_TINY, color=GREY)
    ax.add_patch(FancyArrowPatch((xs[-1], y - 0.36), (xs[hit], y - 0.36), connectionstyle="arc3,rad=-0.16",
                                 arrowstyle="-|>", mutation_scale=8, color=ARMRED, lw=1.3))
    ax.text((xs[-1] + xs[hit]) / 2, 0.15, f"head B (induction), at the last \"{q}\": who came after \"{q}\" "
            f"last time?\nweight {w[hit]:.2f} on \"{toks[hit]}\", {T['w_other']:.2f} on each other token",
            ha="center", va="center", fontsize=FS_TINY, color=ARMRED, linespacing=1.3)
    xn = xs[-1] + 0.85
    ax.annotate("", xy=(xn - 0.30, y), xytext=(xs[-1] + 0.32, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.add_patch(FancyBboxPatch((xn - 0.28, y - 0.13), 0.72, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=TINT_GREEN, ec=GREEN, lw=1.0))
    ax.text(xn + 0.08, y, toks[hit], ha="center", va="center", fontsize=FS, fontweight="bold", color=GREEN)
    ax.text(xn + 0.08, y + 0.24, "next token", ha="center", va="center", fontsize=FS_TINY, color=GREEN)
    save(fig, "blk_induction.pdf", log)
    log.info(f"induction: weights {r2(w)} -> {toks[hit]}")


def fig_neuron(pre, log):
    order = ["blue alone", "fluffy alone", "plain creature", "fluffy blue creature"]
    shown = ["blue alone", "fluffy alone", "plain creature", "this creature"]
    vals = [pre[k] for k in order]
    fig, ax = plt.subplots(figsize=(2.75, 1.6))
    fig.subplots_adjust(left=0.36, right=0.95, top=0.84, bottom=0.3)
    fig.text(0.02, 0.97, "ReLU sets everything left of the red line to 0", fontsize=FS_TINY,
             color=ARMRED, va="top")
    cols = [GREEN if v > 0 else LIGHT for v in vals]
    bars = ax.barh(shown, vals, color=cols, height=0.6)
    ax.bar_label(bars, labels=[f"{v:+.3g}" if v != round(v, 1) else f"{v:+.1f}" for v in vals],
                 fontsize=FS_TINY, padding=2)
    ax.axvline(0, color=ARMRED, lw=0.9)
    ax.set_xlim(-1.6, 0.8)
    ax.set_xlabel("fluffy + 0.5 noun - 1.2", fontsize=FS_TINY)
    ax.tick_params(axis="y", labelsize=FS_TINY, length=0)
    ax.tick_params(axis="x", labelsize=FS_TINY)
    clean(ax)
    save(fig, "blk_neuron.pdf", log)


def hbar(name, labels, values, colors, log, w=W, h=1.2, left=0.22, xfmt=None):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=left, right=0.98, top=0.95, bottom=0.08)
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    total = sum(values)
    ax.bar_label(bars, labels=[xfmt(v, total) for v in values[::-1]], fontsize=FS_TINY, padding=3)
    ax.set_xlim(0, max(values) * 1.4)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=FS_SMALL, length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    save(fig, name, log)


def fig_params(log):
    P = QUOTED["params"]
    fmt = lambda v, tot: f"{v:,}  ({100 * v / tot:.1f}%)" if 100 * v / tot >= 0.1 else f"{v:,}  (<0.1%)"
    hbar("blk_params_block.pdf", ["attention", "MLP", "two norms"],
         [P["attention"], P["mlp"], P["norms"]], [ARMBLUE, ARMORANGE, GREY], log, w=W, h=0.95, xfmt=fmt)
    hbar("blk_params_124m.pdf", ["12 blocks", "token table (LLM-2)", "position table", "final norm"],
         [P["blocks12"], P["wte"], P["wpe"], P["ln_f"]], [ARMORANGE, ARMBLUE, GREEN, GREY], log,
         w=W, h=1.2, left=0.26, xfmt=fmt)


def fig_block(log):
    fig, ax = canvas(2.45, 2.6)
    xs = 0.55                                                   # the stream
    ax.plot([xs, xs], [0.18, 2.3], color=INK, lw=1.6, zorder=1)
    arrow(ax, (xs, 2.18), (xs, 2.36), color=INK, lw=1.4)
    ax.text(xs, 0.05, "x in  (n x 768)", ha="center", fontsize=FS_TINY)
    ax.text(xs + 0.08, 2.38, "x out  (n x 768)", ha="left", fontsize=FS_TINY)
    ax.text(0.08, 1.25, "residual stream", rotation=90, ha="center", va="center", fontsize=FS_TINY, color=GREY)

    def sub(y, label, fc, ec):
        arrow(ax, (xs, y), (1.05, y), color=GREY)
        rbox(ax, 1.05, y - 0.12, 0.42, 0.24, TINT_GREEN, GREEN)
        ax.text(1.26, y, "norm", ha="center", va="center", fontsize=FS_TINY)
        arrow(ax, (1.47, y), (1.6, y), color=GREY)
        rbox(ax, 1.6, y - 0.14, 0.9, 0.28 + 0.24, fc, ec)
        ax.text(2.05, y + 0.12, label, ha="center", va="center", fontsize=FS_SMALL)
        ax.plot([2.05, 2.05, xs + 0.09], [y + 0.38, y + 0.5, y + 0.5], color=GREY, lw=0.8)
        arrow(ax, (xs + 0.3, y + 0.5), (xs + 0.1, y + 0.5), color=GREY)
        ax.add_patch(plt.Circle((xs, y + 0.5), 0.08, facecolor="white", edgecolor=INK, lw=0.9, zorder=4))
        ax.text(xs, y + 0.5, "+", ha="center", va="center", fontsize=FS, zorder=5)

    sub(0.45, "attention", TINT_BLUE, ARMBLUE)
    sub(1.35, "MLP", TINT_ORANGE, ARMORANGE)
    save(fig, "blk_block.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        log.info(f"done in {time.time() - t0:.0f} s")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    G = load_gpt2()
    R = {"date": time.strftime("%Y-%m-%d"), "quoted": QUOTED, "toy": None}
    A1, A2, pre = toy(R, log)
    jobs = [lambda: fig_fact_probes(G, log), lambda: fig_two_heads(A1, A2, log),
            lambda: fig_gpt2_heads(G, R, log), lambda: fig_sink(G, log), lambda: fig_permutation(log),
            lambda: fig_neuron(pre, log), lambda: fig_params(log), lambda: fig_block(log),
            lambda: fig_induction(R, log)]
    for i, job in enumerate(jobs, 1):
        job()
        el = time.time() - t0
        log.info(f"{i}/{len(jobs)}, {el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "block_figs.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
