"""Figures for LLM-2 "Embeddings and position: from IDs to vectors" (ml/14_llms/LLM2_embeddings_position.tex).

Everything about GPT-2 is measured on its real weights: the token table (wte, 50,257 x 768) and the
position table (wpe, 1,024 x 768) are read straight from the local Hugging Face cache with
safetensors (no torch, so it runs in seconds), and token IDs come from tiktoken's "gpt2" encoding.
Numbers the slides quote in prose are asserted (QUOTED below), so a changed input fails loudly
instead of silently contradicting the deck. Model sizes and position schemes of other models are
constants copied from their config.json / model cards (fetched 2026-10-04, cited on the slides).

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text.

Generates into ml/14_llms/fig/: emb_*.pdf (one per figure function below).
Writes ml/14_llms/results/embeddings_position_figs.json (every measured number - the artifact of record).

Run: ./ma/Scripts/python.exe ml/14_llms/py_src/embeddings_position_figs.py   (~50 s measured 2026-10-04)
Needs the GPT-2 weights in the HF cache (the chapter map's --gpt2 step downloads them once).
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle
from matplotlib import colormaps
import tiktoken
from safetensors.numpy import load_file

# A missing glyph must crash the script, never render as an empty box.
warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"

SEED = 509
KARGIN = "ուրա պարավը, բերեք ատամը քաշեմ"        # instructor's Kargin Haghordum reference (cold open)
SENT_A, SENT_B = "The dog bites the man", "The man bites the dog"
BARE_A, BARE_B = "dog bites man", "man bites dog"

# The neighbour grid (frame "What the table learned"): spelling variants of the query are skipped
# there and shown on the next frame instead.
GRID_WORDS = [" Monday", " 1999", " Harry", " Batman", " Messi", " lol", " Armenia", " vodka"]
ODD_ROWS = [(" cat", 8), (" unhappy", 4), (" Einstein", 8)]
ANALOGIES = [(" man", " king", " woman"), (" man", " actor", " woman"), (" cat", " cats", " mouse"),
             (" Germany", " German", " Armenia"), (" good", " better", " bad"),
             (" France", " Paris", " Italy"), (" walk", " walked", " swim")]

# Numbers the deck states in prose. If one changes, fix the slide too.
QUOTED = {
    "ids_a": [464, 3290, 26081, 262, 582], "ids_b": [464, 582, 26081, 262, 3290],
    "bare_a": [9703, 26081, 582], "bare_b": [805, 26081, 3290], "id_cat": 3797, "id_dog": 3290,
    "vocab": 50257, "d_model": 768, "wte_numbers": 38597376, "params": 124439808,
    "wte_share_pct": 31.0, "n_positions": 1024, "wpe_numbers": 786432,
    "pos_gap_cos": {1: 0.997, 20: 0.960, 100: 0.471, 500: 0.246}, "pos0_norm": 9.88,
    "pos_median_norm": 3.37, "pos_pca3": 0.90, "pos_pc_sign_changes": [7, 7],
    "abs_cos_random": 0.029, "abs_cos_tok_pos": 0.025, "recovered": [1998, 2000],
    "wte_median_norm": 3.95, "dog_slots_cos": 0.88, "byte_tokens_near_mean": 17,
    "neigh": {(" Monday", " Tuesday"): (1, 0.89), (" unhappy", " happy"): (2, 0.66),
              (" Einstein", " Eisenhower"): (8, 0.52),
              (" SolidGoldMagikarp", " RandomRedditorWithNo"): (1, 0.69)},
    "analogy": {" king": (" queen", 0.69), " actor": (" actress", 0.82), " cats": (" mice", 0.70),
                " German": (" Armenian", 0.79), " better": (" worse", 0.63),
                " Paris": ("Paris", 0.62), " walked": (" swimming", 0.61)},
    "rome_rank": 5, "raw_input_wins": 3,       # of the 5 working analogies, without excluding inputs
    "rope_q": (2.0, 1.0), "rope_k": (1.0, 1.0), "rope_deg": 30, "rope_plain": 3.00,
    "rope_gap3": 1.00,
    "rope_slow_frac": {4096: 0.075, 32768: 0.60}, "rope_fast_turns_4096": 652,
    "rope_slow_rad": 0.000115, "rope_slow_rad_4096": 0.47,
    # the by-hand shuffle frames (toy_shuffle): weights and dog's output, 2 decimals
    "toy": {"A_nopos_w": [0.42, 0.42, 0.16], "nopos_out": [0.84, 0.58],
            "A_pos_scores": [2.25, 1.5, 0.0], "A_pos_w": [0.63, 0.3, 0.07], "A_pos_out": [1.25, 0.4],
            "B_pos_scores": [1.0, 1.5, 1.25], "B_pos_w": [0.25, 0.42, 0.33], "B_pos_out": [0.87, 0.84],
            "sum": [2.5, 2.5]},
}

# Other models' tables, from their config.json / model cards (fetched 2026-10-04):
# vocab x width = the input table; total parameters from the model card (rounded there).
TABLES = [("GPT-2 small (2019)", 50257, 768, 124439808, True),
          ("Qwen3-0.6B (2025)", 151936, 1024, 0.6e9, True),
          ("Qwen3-8B (2025)", 151936, 4096, 8.2e9, False),
          ("gpt-oss-20b (2025)", 201088, 2880, 21e9, False)]

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
POPBLUE = "#3465A4"
TINTS = {"red": "#F9D2D5", "blue": "#CFD8EE", "orange": "#FCE9B8", "grey": "#EEEEEE"}
INK, GREY = "#222222", "#777777"
CMAP = colormaps["RdBu_r"]

plt.rcParams.update({"font.family": SANS, "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("embeddings_position_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "embeddings_position_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


# ----------------------------------------------------------------------------- GPT-2
class GPT2:
    def __init__(self):
        snaps = sorted(Path.home().glob(".cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors"))
        if not snaps:
            raise FileNotFoundError("GPT-2 weights not in the HF cache - run llm_roadmap.py --gpt2 once")
        self.tensors = load_file(str(snaps[-1]))
        self.wte = self.tensors["wte.weight"].astype(np.float64)
        self.wpe = self.tensors["wpe.weight"].astype(np.float64)
        self.unit = self.wte / np.linalg.norm(self.wte, axis=1, keepdims=True)
        self.enc = tiktoken.get_encoding("gpt2")

    def n_params(self):
        # h.N.attn.bias / masked_bias are causal-mask buffers, not parameters (c_attn.bias is one)
        return sum(v.size for k, v in self.tensors.items()
                   if not k.endswith((".attn.bias", ".attn.masked_bias")))

    def tid(self, w):
        ids = self.enc.encode(w)
        if len(ids) != 1:
            raise ValueError(f"{w!r} is {len(ids)} tokens: {ids}")
        return ids[0]

    def text(self, i):
        return self.enc.decode([i])

    def neighbours(self, w, k, skip_variants=False):
        i = self.tid(w)
        sims = self.unit @ self.unit[i]
        out = []
        q = w.strip().lower()
        for j in np.argsort(-sims):
            if j == i:
                continue
            t = self.text(j)
            if skip_variants and (t.strip().lower() == q or (t.strip() and t.strip().lower() in q)):
                continue
            out.append((t, float(sims[j])))
            if len(out) == k:
                return out

    def analogy(self, a, b, c, k=5):
        ia, ib, ic = self.tid(a), self.tid(b), self.tid(c)
        v = self.unit[ib] - self.unit[ia] + self.unit[ic]
        sims = self.unit @ (v / np.linalg.norm(v))
        order = [j for j in np.argsort(-sims) if j not in (ia, ib, ic)][:k]
        return [(self.text(j), float(sims[j])) for j in order]


# ----------------------------------------------------------------------------- drawing helpers
def canvas(h, w=W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def show(s):
    return s.replace(" ", "␣")


def text_w(fig, s, fs, family):
    t = fig.text(0, 0, s, fontsize=fs, family=family)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def rbox(ax, x, y, w, h, fc, ec, lw=0.5, r=0.03, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def chip(fig, ax, x, y, s, fs=FS, h=0.21, pad=0.07, fc=TINTS["grey"], ec="#999999", lw=0.5,
         minw=0.0, xmax=W - 0.02, below=None, below_color=GREY, bold=False):
    """One token chip at (x, y=centre); returns (x_right, chip_width). Raises on overflow."""
    s = show(s)
    w = max(text_w(fig, s, fs, MONO) + pad, minw)
    if x + w > xmax:
        raise ValueError(f"chip {s!r} overflows ({x + w:.2f} > {xmax:.2f})")
    rbox(ax, x, y - h / 2, w, h, fc, ec, lw)
    ax.text(x + w / 2, y, s, ha="center", va="center", fontsize=fs, family=MONO, color=INK, zorder=3,
            fontweight="bold" if bold else "normal")
    if below is not None:
        ax.text(x + w / 2, y - h / 2 - 0.09, below, ha="center", va="center", fontsize=FS_TINY,
                family=MONO, color=below_color, zorder=3)
    return x + w, w


def strip(ax, x, y, vals, vmax, cell=0.045, h=0.11, z=3):
    """A vector as a row of coloured cells (blue negative, red positive)."""
    for k, v in enumerate(vals):
        ax.add_patch(Rectangle((x + k * cell, y - h / 2), cell, h, facecolor=CMAP(0.5 + 0.5 * v / vmax),
                               edgecolor="white", linewidth=0.4, zorder=z))
    return x + len(vals) * cell


def arrow(ax, p0, p1, color=GREY, lw=0.7, rad=0.0, z=3, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    """Every visible text must sit inside the figure: clipping fails here, not on the slide."""
    fig.canvas.draw()                    # axis labels and ticks are only placed at draw time
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = set()                       # tick objects outside the axis range exist but are not drawn
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
            raise ValueError(f"{name}: text {t.get_text()!r} leaves the figure "
                             f"(x {b.x0 / fig.dpi:.2f}-{b.x1 / fig.dpi:.2f} of {fb.x1 / fig.dpi:.2f} in, "
                             f"y {b.y0 / fig.dpi:.2f}-{b.y1 / fig.dpi:.2f} of {fb.y1 / fig.dpi:.2f} in)")


def save(fig, name, log):
    assert_texts_inside(fig, name)
    out = FIG / name
    for attempt in range(1, 6):        # transient Windows lock, see _learnings errno22 file
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


def r2(x):
    return round(float(x), 2)


# ----------------------------------------------------------------------------- figures
def fig_kargin(g, R, log):
    fig, ax = canvas(0.28, w=3.0)
    ax.text(1.5, 0.14, KARGIN, ha="center", va="center", fontsize=9.5, color=ARMRED, style="italic")
    if text_w(fig, KARGIN, 9.5, SANS) > 2.9:
        raise ValueError("Kargin line too wide for its slot")
    save(fig, "emb_kargin.pdf", log)


def fig_coldopen(g, R, log):
    """Both sentences: chips, IDs, each token's vector, and the two sums."""
    ids_a, ids_b = g.enc.encode(SENT_A), g.enc.encode(SENT_B)
    bare_a, bare_b = g.enc.encode(BARE_A), g.enc.encode(BARE_B)
    assert ids_a == QUOTED["ids_a"] and ids_b == QUOTED["ids_b"], (ids_a, ids_b)
    assert bare_a == QUOTED["bare_a"] and bare_b == QUOTED["bare_b"], (bare_a, bare_b)
    assert sorted(ids_a) == sorted(ids_b) and sorted(bare_a) != sorted(bare_b)
    sum_a, sum_b = g.wte[ids_a].sum(0), g.wte[ids_b].sum(0)
    pos_a = (g.wte[ids_a] + g.wpe[:5]).sum(0)
    pos_b = (g.wte[ids_b] + g.wpe[:5]).sum(0)
    d_tok, d_pos = float(np.abs(sum_a - sum_b).max()), float(np.abs(pos_a - pos_b).max())
    assert d_tok == 0.0 and d_pos == 0.0, (d_tok, d_pos)
    R["coldopen"] = {"ids_a": ids_a, "ids_b": ids_b, "bare_a": bare_a, "bare_b": bare_b,
                     "max_abs_sum_diff_tokens": d_tok, "max_abs_sum_diff_with_positions": d_pos}

    fill = {QUOTED["id_dog"]: TINTS["red"], 582: TINTS["blue"]}
    vmax = float(np.abs(g.wte[ids_a][:, :8]).max())
    fig, ax = canvas(1.72)
    for row, (sent, ids, y) in enumerate([(SENT_A, ids_a, 1.46), (SENT_B, ids_b, 0.66)]):
        ax.text(0.02, y + 0.20, f'"{sent}"', ha="left", va="center", fontsize=FS, style="italic",
                color=INK)
        x = 0.05
        for i in ids:
            xr, w = chip(fig, ax, x, y - 0.04, g.text(i), fc=fill.get(i, TINTS["grey"]), minw=0.42,
                         below=str(i))
            strip(ax, x + w / 2 - 0.18, y - 0.40, g.wte[i, :8], vmax, cell=0.045)
            x = xr + 0.06
        arrow(ax, (x + 0.02, y - 0.40), (x + 0.42, y - 0.40))
        ax.text(x + 0.22, y - 0.30, "sum", ha="center", va="center", fontsize=FS_SMALL, color=GREY)
        strip(ax, x + 0.48, y - 0.40, (sum_a if row == 0 else sum_b)[:8], float(np.abs(sum_a[:8]).max()),
              cell=0.045)
        sum_x = x + 0.48
    ax.text(sum_x + 0.72, 0.66, "identical:\nmax difference 0", ha="center", va="center",
            fontsize=FS_SMALL, color=ARMRED, linespacing=1.2)
    ax.text(0.02, 0.05, "first 8 of each token's 768 numbers shown; blue negative, red positive",
            ha="left", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, "emb_coldopen_ids.pdf", log)


def fig_table_sizes(g, R, log):
    n = g.n_params()
    assert n == QUOTED["params"], n
    assert g.wte.shape == (QUOTED["vocab"], QUOTED["d_model"]) and g.wte.size == QUOTED["wte_numbers"]
    assert g.wpe.shape == (QUOTED["n_positions"], QUOTED["d_model"]) and g.wpe.size == QUOTED["wpe_numbers"]
    assert round(100 * g.wte.size / n, 1) == QUOTED["wte_share_pct"], g.wte.size / n
    rows = []
    for name, v, d, total, tied in TABLES:
        rows.append({"model": name, "vocab": v, "width": d, "table": v * d, "total": total,
                     "share_pct": 100 * v * d / total, "tied": tied})
    R["table_sizes"] = rows
    fig = plt.figure(figsize=(3.2, 1.45))          # slot: a 0.58 column, ~3.2 in
    ax = fig.add_axes([0.40, 0.04, 0.45, 0.78])
    ys = np.arange(len(rows))[::-1]
    bars = ax.barh(ys, [r["share_pct"] for r in rows], color=[ARMRED, ARMBLUE, ARMBLUE, ARMBLUE],
                   height=0.6)
    ax.bar_label(bars, labels=[f"{r['share_pct']:.1f}%" if k == 0 else f"~{r['share_pct']:.0f}%"
                               for k, r in enumerate(rows)], fontsize=FS_SMALL, padding=2)
    ax.set_yticks(ys, [f"{r['model']}\n{r['vocab']:,} x {r['width']:,}" for r in rows], fontsize=FS_SMALL)
    ax.set_xlim(0, 38)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    fig.text(0.02, 0.97, "share of all parameters in the token table", ha="left", va="top",
             fontsize=FS_SMALL, color=GREY)
    save(fig, "emb_table_sizes.pdf", log)


def neighbour_row(fig, ax, g, w, ranks, x0, y, hi=None, qw=0.0, minw=0.58):
    """Query chip, then its neighbours at the given ranks (1 = nearest), rank and cosine below."""
    nb = g.neighbours(w, max(ranks))
    xr, _ = chip(fig, ax, x0, y, w, fc=TINTS["orange"], ec=ARMORANGE, lw=0.8, bold=True, minw=qw)
    arrow(ax, (xr + 0.03, y), (xr + 0.15, y), lw=0.6, ms=5)
    x, prev = xr + 0.19, 0
    for rank in ranks:
        if rank > prev + 1 and prev:
            ax.text(x + 0.06, y, "...", ha="center", va="center", fontsize=FS_SMALL, color=GREY)
            x += 0.16
        t, s = nb[rank - 1]
        is_hi = hi is not None and hi(w, t)
        x, _ = chip(fig, ax, x, y, t, fs=FS_SMALL, fc=TINTS["red"] if is_hi else "white",
                    ec=ARMRED if is_hi else "#AAAAAA", lw=1.0 if is_hi else 0.5,
                    below=f"#{rank} {s:.2f}", below_color=ARMRED if is_hi else GREY, minw=minw)
        x += 0.04
        prev = rank
    return [(t, r2(s)) for t, s in nb]


def fig_neighbours(g, R, log):
    """Eight cards: a word and its three nearest rows (spelling variants of the word skipped)."""
    fig, ax = canvas(1.88)
    out = {}
    cw, ch = 1.31, 0.86
    for k, w in enumerate(GRID_WORDS):
        col, row = k % 4, k // 4
        x, y = 0.04 + col * (cw + 0.05), 0.98 - row * (ch + 0.06)
        rbox(ax, x, y, cw, ch, "white", "#BBBBBB", lw=0.6, r=0.05, z=1)
        chip(fig, ax, x + 0.08, y + ch - 0.17, w, fc=TINTS["orange"], ec=ARMORANGE, lw=0.8, bold=True,
             xmax=x + cw - 0.05)
        nb = g.neighbours(w, 3, skip_variants=True)
        out[w] = [(t, r2(s)) for t, s in nb]
        for j, (t, s) in enumerate(nb):
            yy = y + ch - 0.40 - j * 0.15
            ax.text(x + 0.12, yy, t.strip(), ha="left", va="center", fontsize=FS_SMALL, family=MONO,
                    color=INK)
            ax.text(x + cw - 0.08, yy, f"{s:.2f}", ha="right", va="center", fontsize=FS_TINY,
                    family=MONO, color=GREY)
            if text_w(fig, t.strip(), FS_SMALL, MONO) + 0.12 + 0.35 > cw:
                raise ValueError(f"neighbour {t!r} too wide for its card")
    R["neighbours_grid"] = out
    assert out[" Monday"][0] == (" Tuesday", QUOTED["neigh"][(" Monday", " Tuesday")][1]), out[" Monday"]
    for w, v in out.items():
        log.info(f"neighbours {w!r}: {v}")
    save(fig, "emb_neighbours.pdf", log)


def fig_neighbours_odd(g, R, log):
    def hi(w, t):
        if w == " cat":
            return t.strip().lower() in ("cat", "cats")
        return t in (" happy", " Eisenhower")
    fig, ax = canvas(1.36)
    out = {}
    rows = [(" cat", [1, 2, 3, 4, 5, 7]), (" unhappy", [1, 2, 3, 4]), (" Einstein", [1, 3, 4, 7, 8])]
    assert [w for w, _ in rows] == [w for w, _ in ODD_ROWS]
    for r, (w, ranks) in enumerate(rows):
        out[w] = neighbour_row(fig, ax, g, w, ranks, 0.04, 1.18 - r * 0.46, hi=hi, qw=0.80)
    R["neighbours_odd"] = out
    for (w, t), (rank, s) in QUOTED["neigh"].items():
        if w in out:
            got = [x[0] for x in out[w]]
            assert t in got and got.index(t) + 1 == rank and out[w][rank - 1][1] == s, (w, t, out[w])
    save(fig, "emb_neighbours_odd.pdf", log)


def fig_glitch(g, R, log):
    w = " SolidGoldMagikarp"
    nb = g.neighbours(w, 4)
    rank, s = QUOTED["neigh"][(w, " RandomRedditorWithNo")]
    assert nb[rank - 1][0] == " RandomRedditorWithNo" and r2(nb[rank - 1][1]) == s, nb
    cen = g.wte.mean(0)
    dist = np.linalg.norm(g.wte - cen, axis=1)
    order = np.argsort(dist)
    named = [g.text(j) for j in order[:40] if g.text(j).strip().isprintable() and len(g.text(j).strip()) > 3]
    single = sum(1 for j in order[:20] if len(g.enc.decode_single_token_bytes(int(j))) == 1)
    named20 = [g.text(j) for j in order[:20] if len(g.enc.decode_single_token_bytes(int(j))) > 1]
    assert single == QUOTED["byte_tokens_near_mean"] and len(named20) == 20 - single, (single, named20)
    sgm_rank = int((dist < dist[g.tid(w)]).sum()) + 1
    R["glitch"] = {"neighbours": [(t, r2(x)) for t, x in nb], "closest_to_mean_named": named[:6],
                   "single_byte_in_closest_20": single, "multi_byte_in_closest_20": named20,
                   "sgm_rank_from_mean": sgm_rank}
    log.info(f"closest 20 to the mean row: {named20} + {single} single-byte tokens; "
             f"SolidGoldMagikarp ranks {sgm_rank}")
    fig, ax = canvas(0.95)
    neighbour_row(fig, ax, g, w, [1, 2, 3, 6], 0.04, 0.72, minw=0.62)
    ax.text(0.04, 0.24, "nearest the table's average:", ha="left", va="center", fontsize=FS_SMALL,
            color=INK)
    x = 1.72
    for t in named20:
        x, _ = chip(fig, ax, x, 0.24, t, fs=FS_SMALL, fc="white", ec="#AAAAAA")
        x += 0.04
    ax.text(x + 0.06, 0.24, f"+ {single} single-byte tokens", ha="left", va="center", fontsize=FS_SMALL,
            color=GREY)
    R["glitch"]["shown_named"] = named20
    save(fig, "emb_glitch.pdf", log)


def compute_analogies(g, R, log):
    out = {}
    for a, b, c in ANALOGIES:
        res = g.analogy(a, b, c)
        out[b] = {"a": a, "c": c, "top5": [(t, r2(s)) for t, s in res]}
        top, s = QUOTED["analogy"][b]
        assert res[0][0] == top and r2(res[0][1]) == s, (b, res[:2])
    raw_wins = 0
    for a, b, c in ANALOGIES[:5]:
        ia, ib, ic = g.tid(a), g.tid(b), g.tid(c)
        v = g.unit[ib] - g.unit[ia] + g.unit[ic]
        top = int(np.argmax(g.unit @ (v / np.linalg.norm(v))))
        out[b]["nearest_without_exclusion"] = g.text(top)
        raw_wins += int(top in (ia, ib, ic))
    assert raw_wins == QUOTED["raw_input_wins"], raw_wins
    rome = [t for t, _ in out[" Paris"]["top5"]]
    assert rome.index(" Rome") + 1 == QUOTED["rome_rank"], rome
    R["analogies"] = out
    for b, v in out.items():
        log.info(f"analogy {b!r} - {v['a']!r} + {v['c']!r} -> {v['top5'][:3]}")


def fig_add_position(g, R, log):
    """"dog" in slot 1 and slot 4: same token vector, different position vectors, different inputs."""
    d = QUOTED["id_dog"]
    x1, x4 = g.wte[d] + g.wpe[1], g.wte[d] + g.wpe[4]
    cos = float(x1 @ x4 / (np.linalg.norm(x1) * np.linalg.norm(x4)))
    R["dog_slots"] = {"cos_input_slot1_slot4": cos, "max_abs_diff": float(np.abs(x1 - x4).max())}
    log.info(f"dog@1 vs dog@4 input cosine {cos:.3f}")
    assert r2(cos) == QUOTED["dog_slots_cos"], cos
    vt = float(np.abs(g.wte[d, :8]).max())
    vp = float(np.abs(g.wpe[[1, 4], :8]).max())
    vx = float(np.abs(np.vstack([x1[:8], x4[:8]])).max())
    fig, ax = canvas(1.0)
    cols = [1.30, 2.30, 3.30]                     # left edges of the three strips (0.48 in each)
    for r, (slot, x) in enumerate([(1, x1), (4, x4)]):
        y = 0.62 - r * 0.38
        ax.text(0.04, y, f'"dog" in slot {slot}', ha="left", va="center", fontsize=FS, color=INK)
        strip(ax, cols[0], y, g.wte[d, :8], vt, cell=0.06)
        ax.text(cols[0] + 0.74, y, "+", ha="center", va="center", fontsize=FS + 1)
        strip(ax, cols[1], y, g.wpe[slot, :8], vp, cell=0.06)
        ax.text(cols[1] + 0.74, y, "=", ha="center", va="center", fontsize=FS + 1)
        strip(ax, cols[2], y, x[:8], vx, cell=0.06)
    for xc, lab in zip(cols, ["token: same", "position: differs", "input: differs"]):
        ax.text(xc + 0.24, 0.88, lab, ha="center", va="center", fontsize=FS_TINY, color=GREY)
    ax.text(4.75, 0.43, f"cosine of the two inputs:\n{cos:.2f}: close, not equal", ha="center",
            va="center", fontsize=FS_SMALL, color=ARMRED, linespacing=1.2)
    save(fig, "emb_add_position.pdf", log)


def fig_gpt2_positions(g, R, log):
    P = g.wpe
    unit = P / np.linalg.norm(P, axis=1, keepdims=True)
    C = unit @ unit.T
    gaps = np.arange(1, 1001)
    mean_cos = np.array([np.mean(np.diag(C, k)) for k in gaps])
    for k, v in QUOTED["pos_gap_cos"].items():
        assert round(float(mean_cos[k - 1]), 3) == v, (k, mean_cos[k - 1])
    X = P - P.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    pcs = X @ Vt[:3].T
    signs = [int(np.sum(np.diff(np.sign(pcs[:, k])) != 0)) for k in range(3)]
    norms = np.linalg.norm(P, axis=1)
    assert round(float(ev[:3].sum()), 2) == QUOTED["pos_pca3"], ev[:3]
    assert signs[:2] == QUOTED["pos_pc_sign_changes"], signs
    assert r2(norms[0]) == QUOTED["pos0_norm"] and r2(np.median(norms)) == QUOTED["pos_median_norm"]
    R["gpt2_positions"] = {"mean_cos_by_gap": {int(k): float(mean_cos[k - 1]) for k in (1, 5, 20, 100, 500)},
                           "pca_explained_top6": ev[:6].tolist(), "pc_sign_changes": signs,
                           "norm_pos0": float(norms[0]), "norm_median": float(np.median(norms))}

    fig = plt.figure(figsize=(W, 1.72))
    a1 = fig.add_axes([0.07, 0.26, 0.27, 0.58])
    Z = P[:, :48] - P[:, :48].mean(0)
    Z = Z / np.abs(Z[1:]).max(0)                     # position 0 is an outlier; scale on the rest
    a1.imshow(Z.T, aspect="auto", cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
    a1.set_xlabel("position (0 to 1,023)", labelpad=1)
    a1.set_ylabel("48 of 768 numbers", labelpad=1)
    a1.set_xticks([0, 512, 1023])
    a1.set_yticks([])
    a1.set_title("the table (each row rescaled)", fontsize=FS_SMALL, loc="left")
    a2 = fig.add_axes([0.41, 0.26, 0.26, 0.58])
    for k, c in ((0, ARMRED), (1, ARMBLUE)):
        a2.plot(np.arange(1024), pcs[:, k], color=c, lw=1.0)
    a2.set_xticks([0, 512, 1023])
    a2.set_yticks([])
    a2.set_xlabel("position", labelpad=1)
    a2.set_title("its two main directions", fontsize=FS_SMALL, loc="left")
    clean_axes(a2)
    a3 = fig.add_axes([0.74, 0.26, 0.24, 0.58])
    a3.plot(gaps, mean_cos, color=ARMBLUE, lw=1.0)
    a3.axhline(0, color="#BBBBBB", lw=0.5)
    for k, lab in ((1, f"{mean_cos[0]:.3f}"), (100, f"{mean_cos[99]:.2f}"), (500, f"{mean_cos[499]:.2f}")):
        a3.plot(k, mean_cos[k - 1], "o", ms=2.5, color=ARMRED)
        a3.annotate(lab, (k, mean_cos[k - 1]), xytext=(4, 2), textcoords="offset points",
                    fontsize=FS_TINY, color=ARMRED)
    a3.set_xlabel("distance between positions", labelpad=1)
    a3.set_ylabel("similarity (cosine)", labelpad=1)
    a3.set_ylim(float(mean_cos.min()) - 0.08, 1.12)
    a3.set_title("near = similar", fontsize=FS_SMALL, loc="left")
    clean_axes(a3)
    save(fig, "emb_gpt2_positions.pdf", log)


def sinusoid_pe(n, d):
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe


def fig_sinusoids(g, R, log):
    pe = sinusoid_pe(100, 64)
    fig = plt.figure(figsize=(W, 1.62))
    a1 = fig.add_axes([0.07, 0.23, 0.38, 0.62])
    a1.imshow(pe.T, aspect="auto", cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
    a1.set_xlabel("position", labelpad=1)
    a1.set_ylabel("dimension (of 64)", labelpad=1)
    a1.set_title("every position gets its own column", fontsize=FS_SMALL, loc="left")
    a2 = fig.add_axes([0.55, 0.23, 0.42, 0.62])
    for dim, c, lab in ((0, ARMRED, "dim 0 (fast)"), (16, ARMORANGE, "dim 16"),
                        (40, ARMBLUE, "dim 40 (slow)")):
        a2.plot(np.arange(100), pe[:, dim], color=c, lw=1.0, label=lab)
    a2.set_xlabel("position", labelpad=1)
    a2.set_yticks([-1, 0, 1])
    a2.legend(fontsize=FS_TINY, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.2, columnspacing=0.8)
    clean_axes(a2)
    save(fig, "emb_sinusoid_heatmap.pdf", log)

    pe = sinusoid_pe(200, 128)
    ks = np.arange(-30, 31)
    curves = {p: np.array([pe[p] @ pe[p + k] for k in ks]) for p in (40, 70, 100)}
    spread = max(float(np.abs(curves[40] - curves[p]).max()) for p in (70, 100))
    assert spread < 1e-9, spread
    R["sinusoid_gap"] = {"max_spread_between_positions": spread, "d_model": 128}
    fig = plt.figure(figsize=(3.4, 1.5))
    ax = fig.add_axes([0.16, 0.24, 0.80, 0.64])
    for (p, c), lw, ls in zip(((40, ARMRED), (70, ARMORANGE), (100, ARMBLUE)), (2.4, 1.4, 0.8),
                              ("-", "-", "--")):
        ax.plot(ks, curves[p], color=c, lw=lw, ls=ls, label=f"from position {p}")
    ax.set_xlabel("distance to the other position", labelpad=1)
    ax.set_ylabel("dot product", labelpad=1)
    ax.legend(fontsize=FS_TINY, frameon=False, loc="upper right", handlelength=1.6)
    ax.set_title("three curves, drawn on top of each other", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_sinusoid_gap.pdf", log)


def fig_add_vs_concat(g, R, log):
    rng = np.random.default_rng(SEED)
    Rv = rng.standard_normal((4000, QUOTED["d_model"]))
    Rv /= np.linalg.norm(Rv, axis=1, keepdims=True)
    rand = np.abs((Rv[:2000] * Rv[2000:]).sum(1))
    toks = rng.choice(len(g.wte), 2000, replace=False)
    poss = rng.integers(0, QUOTED["n_positions"], 2000)
    pu = g.wpe / np.linalg.norm(g.wpe, axis=1, keepdims=True)
    tokpos = np.abs((g.unit[toks] * pu[poss]).sum(1))
    hits, misses = 0, []
    for t, p in zip(toks, poss):
        v = g.wte[t] + g.wpe[p]
        best = int(np.argmax(g.unit @ (v / np.linalg.norm(v))))
        hits += int(best == t)
        if best != t:
            misses.append({"token": g.text(int(t)), "position": int(p), "nearest_instead": g.text(best)})
    log.info(f"token not recovered after adding a position: {misses}")
    res = {"abs_cos_random_mean": float(rand.mean()), "abs_cos_tok_pos_mean": float(tokpos.mean()),
           "recovered": [hits, len(toks)], "misses": misses,
           "wte_median_norm": float(np.median(np.linalg.norm(g.wte, axis=1)))}
    R["add_vs_concat"] = res
    assert round(res["abs_cos_random_mean"], 3) == QUOTED["abs_cos_random"], res
    assert round(res["abs_cos_tok_pos_mean"], 3) == QUOTED["abs_cos_tok_pos"], res
    assert res["recovered"] == QUOTED["recovered"], res
    assert r2(res["wte_median_norm"]) == QUOTED["wte_median_norm"], res
    fig = plt.figure(figsize=(3.4, 1.5))           # slot: a 0.62 column, ~3.4 in
    ax = fig.add_axes([0.04, 0.25, 0.92, 0.62])
    bins = np.linspace(0, 0.16, 33)
    ax.hist(rand, bins=bins, color="#BBBBBB", label=f"random pairs, mean {rand.mean():.3f}")
    ax.hist(tokpos, bins=bins, histtype="step", color=ARMRED, lw=1.3,
            label=f"GPT-2 token vs position, mean {tokpos.mean():.3f}")
    ax.set_xlabel("|cosine| between the two vectors (0 = perpendicular)", labelpad=1)
    ax.set_yticks([])
    ax.legend(fontsize=FS_TINY, frameon=False, loc="upper right")
    ax.set_title("2,000 pairs each, 768 dimensions", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    save(fig, "emb_add_vs_concat.pdf", log)


def rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def fig_rope_clock(g, R, log):
    fig, ax = canvas(1.55)
    for c, (cx, speed, name) in enumerate([(1.35, 40, "a fast pair: 40 degrees per slot"),
                                           (4.15, 8, "a slow pair: 8 degrees per slot")]):
        r = 0.55
        ax.add_patch(Circle((cx, 0.78), r, facecolor="white", edgecolor="#BBBBBB", lw=0.6, zorder=1))
        for slot in range(6):
            v = rot((1, 0), speed * slot) * r * 0.92
            col = CMAP(0.25 + 0.12 * slot)
            arrow(ax, (cx, 0.78), (cx + v[0], 0.78 + v[1]), color=col, lw=1.0, ms=5)
            lab = rot((1, 0), speed * slot) * (r + 0.12)
            ax.text(cx + lab[0], 0.78 + lab[1], str(slot), ha="center", va="center", fontsize=FS_TINY,
                    color=INK)
        ax.text(cx, 0.08, name, ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax.text(2.75, 0.78, "same token,\nslots 0 to 5", ha="center", va="center", fontsize=FS_SMALL,
            color=GREY, linespacing=1.2)
    save(fig, "emb_rope_clock.pdf", log)


def fig_rope_2d(g, R, log):
    q, k, th = np.array(QUOTED["rope_q"]), np.array(QUOTED["rope_k"]), QUOTED["rope_deg"]
    plain = float(q @ k)
    cases = [(5, 2), (10, 7), (10, 2)]
    scores = {}
    for m, n in cases:
        scores[(m, n)] = float(rot(q, th * m) @ rot(k, th * n))
    assert round(plain, 2) == QUOTED["rope_plain"], plain
    assert round(scores[(5, 2)], 2) == round(scores[(10, 7)], 2) == QUOTED["rope_gap3"], scores
    R["rope_by_hand"] = {"q": q.tolist(), "k": k.tolist(), "deg_per_slot": th, "plain": plain,
                         "scores": {f"q@{m},k@{n}": round(s, 4) for (m, n), s in scores.items()},
                         "rotated": {f"q@{m}": rot(q, th * m).round(4).tolist() for m, _ in cases} |
                                    {f"k@{n}": rot(k, th * n).round(4).tolist() for _, n in cases}}
    log.info(f"RoPE by hand: {R['rope_by_hand']}")
    fig, ax = canvas(1.62)
    for c, (m, n) in enumerate(cases):
        cx, cy, s = (1.15, 2.80, 4.40)[c], 0.88, 0.20
        ax.add_patch(Circle((cx, cy), 0.62, facecolor="none", edgecolor="#DDDDDD", lw=0.5, zorder=1))
        qv, kv = rot(q, th * m) * s, rot(k, th * n) * s
        arrow(ax, (cx, cy), (cx + qv[0], cy + qv[1]), color=ARMRED, lw=1.3, ms=6)
        arrow(ax, (cx, cy), (cx + kv[0], cy + kv[1]), color=ARMBLUE, lw=1.3, ms=6)
        a_q, a_k = math.degrees(math.atan2(qv[1], qv[0])), math.degrees(math.atan2(kv[1], kv[0]))
        lo, hi = sorted((a_q, a_k))
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        ax.add_patch(Wedge((cx, cy), 0.20, lo, hi, facecolor="#FCE9B8", edgecolor=ARMORANGE, lw=0.5,
                           zorder=1.5))
        for v, lab, col in ((qv, f"query @ {m}", ARMRED), (kv, f"key @ {n}", ARMBLUE)):
            u = v / np.linalg.norm(v)
            ax.text(cx + v[0] + 0.06 * u[0], cy + v[1] + 0.06 * u[1], lab, fontsize=FS_TINY, color=col,
                    ha="left" if u[0] >= 0 else "right", va="bottom" if u[1] >= 0 else "top", zorder=4)
        col = ARMRED if (m, n) != (10, 2) else INK
        ax.text(cx, 0.12, f"gap {m - n}: score {scores[(m, n)]:.2f}", ha="center", va="center",
                fontsize=FS_SMALL, color=col, fontweight="bold" if (m, n) != (10, 2) else "normal")
    save(fig, "emb_rope_2d.pdf", log)


def fig_rope_unseen(g, R, log):
    d, base, train, longer = 128, 10000, 4096, 32768
    slow = base ** (-(d - 2) / d)            # radians per token, the slowest of the 64 pairs
    fast = 1.0
    frac = {L: slow * L / (2 * math.pi) for L in (train, longer)}
    turns = fast * train / (2 * math.pi)
    assert round(frac[train], 3) == QUOTED["rope_slow_frac"][train], frac
    assert round(frac[longer], 2) == QUOTED["rope_slow_frac"][longer], frac
    assert round(turns) == QUOTED["rope_fast_turns_4096"], turns
    assert round(slow, 6) == QUOTED["rope_slow_rad"] and round(slow * train, 2) == QUOTED["rope_slow_rad_4096"], slow
    R["rope_unseen"] = {"d_head": d, "base": base, "slow_rad_per_token": slow,
                        "slow_fraction_of_turn": {str(k): v for k, v in frac.items()},
                        "fast_turns_in_train": turns}
    fig, ax = canvas(1.62)
    panels = [("fastest pair, 4,096 tokens", 1.0, None, f"{turns:.0f} full turns:\nevery angle seen"),
              ("slowest pair, 4,096 tokens", frac[train], None, f"{100 * frac[train]:.1f}% of one turn:\nall it ever saw"),
              ("slowest pair, 32,768 tokens", frac[longer], frac[train],
               f"{100 * frac[longer]:.0f}% of a turn:\nmost of it never seen")]
    for c, (title, f, seen, note) in enumerate(panels):
        cx, cy, r = 0.92 + c * 1.84, 0.88, 0.42
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#BBBBBB", lw=0.6, zorder=3))
        if f >= 1:
            ax.add_patch(Circle((cx, cy), r, facecolor="#CFD8EE", edgecolor="none", zorder=0.8))
        else:
            known = min(f, seen) if seen else f
            ax.add_patch(Wedge((cx, cy), r, 90 - 360 * known, 90, facecolor="#CFD8EE", edgecolor="none",
                               zorder=0.8))
            if seen:
                ax.add_patch(Wedge((cx, cy), r, 90 - 360 * f, 90 - 360 * seen, facecolor="#F9D2D5",
                                   edgecolor="none", zorder=0.8))
        ax.text(cx, cy + r + 0.13, title, ha="center", va="center", fontsize=FS_SMALL, color=INK)
        ax.text(cx, 0.20, note, ha="center", va="center", fontsize=FS_TINY,
                color=ARMRED if seen else GREY, linespacing=1.15)
    save(fig, "emb_rope_unseen.pdf", log)


def toy_shuffle(g, R, log):
    """The two by-hand frames: a simplified next layer (dot-product scores -> softmax -> weighted
    sum, no learned matrices) on 2-number toy vectors, for the token "dog", without and with
    position vectors. The summed inputs stay equal; dog's output does not."""
    vec = {"dog": np.array([1.0, 0.0]), "bites": np.array([1.0, 1.0]), "man": np.array([0.0, 1.0])}
    slot = [np.array([0.5, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.5])]
    T = QUOTED["toy"]
    out = {}
    for name, sent in (("A", ["dog", "bites", "man"]), ("B", ["man", "bites", "dog"])):
        for pos in (False, True):
            X = [vec[w] + (slot[i] if pos else 0) for i, w in enumerate(sent)]
            q = X[sent.index("dog")]
            s = np.array([q @ x for x in X])
            wts = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
            y = sum(wi * x for wi, x in zip(wts, X))
            out[f"{name}_{'pos' if pos else 'nopos'}"] = {
                "inputs": [x.tolist() for x in X], "scores": s.round(4).tolist(),
                "weights": wts.round(4).tolist(), "dog_output": y.round(4).tolist(),
                "sum_of_inputs": sum(X).tolist()}
    R["toy_shuffle"] = out
    rnd = lambda v: [round(float(x), 2) for x in v]
    assert rnd(out["A_nopos"]["weights"]) == T["A_nopos_w"], out["A_nopos"]
    assert rnd(out["A_nopos"]["dog_output"]) == rnd(out["B_nopos"]["dog_output"]) == T["nopos_out"]
    for k in ("A", "B"):
        assert rnd(out[f"{k}_pos"]["scores"]) == T[f"{k}_pos_scores"], out[f"{k}_pos"]
        assert rnd(out[f"{k}_pos"]["weights"]) == T[f"{k}_pos_w"], out[f"{k}_pos"]
        assert rnd(out[f"{k}_pos"]["dog_output"]) == T[f"{k}_pos_out"], out[f"{k}_pos"]
        assert out[f"{k}_pos"]["sum_of_inputs"] == T["sum"], out[f"{k}_pos"]
    log.info(f"toy shuffle: {out}")


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    g = GPT2()
    R = {"date": time.strftime("%Y-%m-%d"), "tiktoken": tiktoken.__version__, "model": "gpt2 (124M), HF cache"}
    jobs = [fig_kargin, fig_coldopen, fig_table_sizes, fig_neighbours, fig_neighbours_odd, fig_glitch,
            compute_analogies, fig_add_position, fig_gpt2_positions, fig_sinusoids, fig_add_vs_concat,
            fig_rope_clock, fig_rope_2d, fig_rope_unseen, toy_shuffle]
    for k, job in enumerate(jobs, 1):
        job(g, R, log)
        el = time.time() - t0
        log.info(f"{k}/{len(jobs)} {job.__name__}, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left")
    (RESULTS / "embeddings_position_figs.json").write_text(
        json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
