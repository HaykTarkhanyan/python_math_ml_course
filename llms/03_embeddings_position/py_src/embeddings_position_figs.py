"""Figures for LLM-3 "Embeddings and position: from IDs to vectors" (llms/03_embeddings_position/LLM3_embeddings_position.tex).

Everything about GPT-2 is measured on its real weights: the token table (wte, 50,257 x 768) and the
position table (wpe, 1,024 x 768) are read straight from the local Hugging Face cache with
safetensors (no torch, so it runs in seconds), and token IDs come from tiktoken's "gpt2" encoding.
Numbers the slides quote in prose are asserted (QUOTED below), so a changed input fails loudly
instead of silently contradicting the deck. Model sizes and position schemes of other models are
constants copied from their config.json / model cards (fetched 2026-10-04, cited on the slides).

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text.

Generates into llms/03_embeddings_position/fig/: emb_*.pdf (one per figure function below).
Writes llms/03_embeddings_position/results/embeddings_position_figs.json (every measured number - the artifact of record).

Run: ./ma/Scripts/python.exe llms/03_embeddings_position/py_src/embeddings_position_figs.py   (~50 s measured 2026-10-04)
Needs the GPT-2 weights in the HF cache (the chapter map's --gpt2 step downloads them once).

Rerun some figures only: ... embeddings_position_figs.py --only fig_unembed fig_one_pair   (a few s)
Jobs in NO_WEIGHTS run without GPT-2's weights; their numbers are merged into the results JSON.
fig_bank_layers also needs torch + transformers (GPT-2's forward pass, local files only; ~8 s measured
2026-11-09). If the weights are missing: HF_HUB_DISABLE_XET=1 and snapshot_download("gpt2") (~550 MB;
the xet path returned 404 for the "gpt2" alias on 2026-11-06).
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
from matplotlib.ticker import FuncFormatter
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
    # "By hand: four numbers, two speeds": q = k = (1, 0 | 1, 0), 90 and 30 degrees per slot
    "rope_pairs_deg": (90, 30), "rope_pairs_scores": {0: 2.0, 1: 0.87, 2: -0.5, 4: 0.5},
    "rope_pairs_check": {(5, 1): [(0.0, 1.0), (0.0, 1.0), (-0.866, 0.5), (0.866, 0.5)],
                         (9, 5): 0.5},
    # the by-hand shuffle frames (toy_shuffle): weights and dog's output, 2 decimals
    "toy": {"A_nopos_w": [0.42, 0.42, 0.16], "nopos_out": [0.84, 0.58],
            "A_pos_scores": [2.25, 1.5, 0.0], "A_pos_w": [0.63, 0.3, 0.07], "A_pos_out": [1.25, 0.4],
            "B_pos_scores": [1.0, 1.5, 1.25], "B_pos_w": [0.25, 0.42, 0.33], "B_pos_out": [0.87, 0.84],
            "sum": [2.5, 2.5]},
    # 2026-10-28 additions (instructor review of LLM-3; none of these needs GPT-2's weights)
    "unembed_scores": [-0.09, -0.26, 1.04, -0.31], "unembed_pct": [17, 15, 54, 14],
    "gap_pairs": {1: 1023, 100: 924, 1000: 24},
    "one_pair_deg": 30, "one_pair": {(1, 3): 0.5, (6, 8): 0.5},
    "fast_alias": (44, 1.00), "slow_gap1": 0.99995, "sum_peak": 64, "sum_gap1": 62, "sum_far_max": 39,
    "layer_params": {768: 85054464, 1536: 339978240}, "ln_f": 1536,
    "rand_abs_cos": {2: 0.64, 20: 0.18, 768: 0.029},
    "rope_turn_pairs_4096": (46, 18), "rope_slots_per_turn": {0: 6.3, 16: 63, 32: 628, 48: 6283, 63: 54410},
    # 2026-11-06 additions (DECISIONS #112)
    "wte_norm_range": [2.45, 6.32], "recovered_slot0": [1997, 2000],
    "rvt": {" coffee": {"trained": [(" espresso", 0.67), (" tea", 0.62), (" caffeine", 0.62)], "random_max": 0.17},
            " seven": {"trained": [(" six", 0.87), (" eight", 0.86), (" five", 0.85)], "random_max": 0.14}},
    "grad_toy": {"cos_mon_tue": [-0.85, 0.99], "cos_mon_banana": [0.08, -1.0], "first_push_cos_mon_tue": 1.0,
                 "first_push_cos_mon_banana": -1.0},
    "bank": {"last": (0.8, 0.54), "raw_last_different": 0.99},
    "rope_stretch": {"factor": 8, "fast_step_deg": {"as is": 57.3, "pi": 7.2, "yarn": 57.3},
                     "slow_fraction": {"as is": 0.602, "pi": 0.075, "yarn": 0.075}, "seen_fraction": 0.075},
}

# The 4-token toy table of the by-hand lookup frame, reused by the tying frame (fig_unembed)
TOY_E = {"the": (0.2, -0.1, 0.5), "cat": (0.9, 0.3, -0.4), "sat": (-0.6, 0.8, 0.1), "on": (0.0, -0.5, 0.7)}
TOY_H = (-0.5, 0.9, 0.2)        # "the last vector": made up so that "sat" wins after "the cat"

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
            fig.savefig(out, metadata={"CreationDate": None})   # an unchanged redraw stays byte-identical
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
    norms = np.linalg.norm(g.wte, axis=1)          # why the angle: rows differ in length (2026-11-09)
    R["wte_norm_range"] = [r2(norms.min()), r2(norms.max())]
    assert R["wte_norm_range"] == QUOTED["wte_norm_range"], R["wte_norm_range"]
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
    """The overview (three panels) plus each panel alone at its View 1/2/3 column size
    (2026-11-06: the panels used to be cropped out of the overview, whose right panel's y label
    overlapped the middle panel)."""
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

    Z = P[:, :48] - P[:, :48].mean(0)
    Z = Z / np.abs(Z[1:]).max(0)                     # position 0 is an outlier; scale on the rest

    def table(a):
        a.imshow(Z.T, aspect="auto", cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
        a.set_xlabel("position (0 to 1,023)", labelpad=1)
        a.set_ylabel("48 of 768 numbers", labelpad=1)
        a.set_xticks([0, 512, 1023])
        a.set_yticks([])
        a.set_title("the table (each row rescaled)", fontsize=FS_SMALL, loc="left")

    def waves(a):
        for k, c in ((0, ARMRED), (1, ARMBLUE)):
            a.plot(np.arange(1024), pcs[:, k], color=c, lw=1.0)
        a.set_xticks([0, 512, 1023])
        a.set_yticks([])
        a.set_xlabel("position", labelpad=1)
        a.set_title("its two main directions", fontsize=FS_SMALL, loc="left")
        clean_axes(a)

    def gap(a):
        a.plot(gaps, mean_cos, color=ARMBLUE, lw=1.0)
        a.axhline(0, color="#BBBBBB", lw=0.5)
        for k, lab in ((1, f"{mean_cos[0]:.3f}"), (100, f"{mean_cos[99]:.2f}"), (500, f"{mean_cos[499]:.2f}")):
            a.plot(k, mean_cos[k - 1], "o", ms=2.5, color=ARMRED)
            a.annotate(lab, (k, mean_cos[k - 1]), xytext=(4, 2), textcoords="offset points",
                       fontsize=FS_TINY, color=ARMRED)
        a.set_xlabel("distance between positions", labelpad=1)
        a.set_ylabel("similarity (cosine)", labelpad=1)
        a.set_ylim(float(mean_cos.min()) - 0.08, 1.12)
        a.set_title("near = similar", fontsize=FS_SMALL, loc="left")
        clean_axes(a)

    fig = plt.figure(figsize=(W, 1.72))
    table(fig.add_axes([0.06, 0.26, 0.25, 0.58]))
    waves(fig.add_axes([0.36, 0.26, 0.24, 0.58]))
    gap(fig.add_axes([0.74, 0.26, 0.24, 0.58]))      # room on its left for the y label
    save(fig, "emb_gpt2_positions.pdf", log)
    # one panel per View frame, drawn at its column width (0.47, 0.34, 0.40 of the text)
    for name, draw, w, h, rect in (("emb_gpt2_pos_table.pdf", table, 2.55, 2.0, [0.10, 0.20, 0.82, 0.70]),
                                   ("emb_gpt2_pos_waves.pdf", waves, 1.85, 1.9, [0.06, 0.20, 0.84, 0.70]),
                                   ("emb_gpt2_pos_gap.pdf", gap, 2.15, 1.9, [0.24, 0.20, 0.72, 0.70])):
        fig = plt.figure(figsize=(w, h))
        draw(fig.add_axes(rect))
        save(fig, name, log)


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
    a1.set_xticks([0, 20, 40, 60, 80, 99])          # instructor 2026-10-28: show where the axis ends
    a1.set_yticks([0, 20, 40, 63])
    a1.set_xlabel("position (0 to 99)", labelpad=1)
    a1.set_ylabel("dimension (0 to 63)", labelpad=1)
    a1.set_title("every position gets its own column", fontsize=FS_SMALL, loc="left")
    a2 = fig.add_axes([0.55, 0.23, 0.42, 0.62])
    for dim, c, lab in ((0, ARMRED, "dim 0 (fast)"), (16, ARMORANGE, "dim 16"),
                        (40, ARMBLUE, "dim 40 (slow)")):
        a2.plot(np.arange(100), pe[:, dim], color=c, lw=1.0, label=lab)
    a2.set_xlim(0, 99)
    a2.set_xticks([0, 20, 40, 60, 80, 99])
    a2.set_xlabel("position (0 to 99)", labelpad=1)
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
    # slot 0's row is ~3x longer than the rest: does it swamp the token? (student review, 2026-11-09)
    V0 = g.wte[toks] + g.wpe[0]
    hits0 = int((np.argmax((V0 / np.linalg.norm(V0, axis=1, keepdims=True)) @ g.unit.T, axis=1) == toks).sum())
    res = {"abs_cos_random_mean": float(rand.mean()), "abs_cos_tok_pos_mean": float(tokpos.mean()),
           "recovered": [hits, len(toks)], "misses": misses, "recovered_slot0": [hits0, len(toks)],
           "wte_median_norm": float(np.median(np.linalg.norm(g.wte, axis=1)))}
    assert res["recovered_slot0"] == QUOTED["recovered_slot0"], res["recovered_slot0"]
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
    fig, ax = canvas(1.34)                  # 2026-10-28: shorter, so the frame keeps it at full width
    panels = [("fastest pair, 4,096 tokens", 1.0, None, f"{turns:.0f} full turns:\nevery angle seen"),
              ("slowest pair, 4,096 tokens", frac[train], None, f"{100 * frac[train]:.1f}% of one turn:\nall it ever saw"),
              ("slowest pair, 32,768 tokens", frac[longer], frac[train],
               f"{100 * frac[longer]:.0f}% of a turn:\nmost of it never seen")]
    for c, (title, f, seen, note) in enumerate(panels):
        cx, cy, r = 0.92 + c * 1.84, 0.74, 0.36
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
        ax.text(cx, 0.17, note, ha="center", va="center", fontsize=FS_TINY,
                color=ARMRED if seen else GREY, linespacing=1.15)
    save(fig, "emb_rope_unseen.pdf", log)


def toy_rope_pairs(g, R, log):
    """The "four numbers, two speeds" frame: RoPE on a 4-number q and k (two pairs). The score is the
    sum of the pair scores; equal gaps give equal totals; the fast pair alone confuses gap 4 with 0."""
    q = k = np.array([1.0, 0.0, 1.0, 0.0])
    speeds = QUOTED["rope_pairs_deg"]

    def turn(v, slot):
        return np.concatenate([rot(v[2 * p:2 * p + 2], speeds[p] * slot) for p in range(2)])

    def score(m, n):
        return float(turn(q, m) @ turn(k, n))

    by_gap = {gap: round(score(gap + 3, 3), 2) for gap in QUOTED["rope_pairs_scores"]}
    assert by_gap == QUOTED["rope_pairs_scores"], by_gap
    for gap in by_gap:                                   # same gap, any slots: same total
        assert all(abs(score(gap + s, s) - score(gap, 0)) < 1e-9 for s in range(12)), gap
    qt, kt = turn(q, 5), turn(k, 1)
    shown = [tuple(round(float(x), 3) + 0.0 for x in v) for v in (qt[:2], kt[:2], qt[2:], kt[2:])]
    assert shown == QUOTED["rope_pairs_check"][(5, 1)], shown
    assert round(score(9, 5), 2) == QUOTED["rope_pairs_check"][(9, 5)] == by_gap[4]
    fast_only = [round(math.cos(math.radians(speeds[0] * gap)), 2) for gap in (0, 4)]
    assert fast_only[0] == fast_only[1] and by_gap[0] != by_gap[4], (fast_only, by_gap)
    R["rope_pairs"] = {"q": q.tolist(), "k": k.tolist(), "deg_per_slot": list(speeds),
                       "score_by_gap": by_gap, "turned_q5_k1": shown}
    log.info(f"RoPE, two pairs: {R['rope_pairs']}")


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


# ----------------------------------------------------------------------------- 2026-10-28 additions
# Instructor review of LLM-3 ("explain slower, show don't tell"). Toy or computed numbers only:
# none of these reads GPT-2's weights, so they run with --only (see main).
def fig_unembed(g, R, log):
    """Tying: the toy table of the by-hand lookup frame, read in (one-hot picks the "cat" row) and
    written out (the last vector h scored against every row, then softmax)."""
    names = list(TOY_E)
    E, h = np.array([TOY_E[n] for n in names]), np.array(TOY_H)
    s = E @ h
    p = np.exp(s - s.max())
    p /= p.sum()
    assert [r2(x) for x in s] == QUOTED["unembed_scores"], s
    assert [round(100 * float(x)) for x in p] == QUOTED["unembed_pct"], p
    R["unembed"] = {"E": E.tolist(), "h": list(TOY_H), "scores": s.round(4).tolist(),
                    "probs": p.round(4).tolist()}
    fig, ax = canvas(1.62)
    ys, x0, cw, ch = [1.04, 0.80, 0.56, 0.32], 1.30, 0.36, 0.19
    ax.text(0.05, 1.47, "in: fetch a row", ha="left", va="center", fontsize=FS, color=ARMRED,
            fontweight="bold")
    ax.text(2.62, 1.47, "out: score every row", ha="left", va="center", fontsize=FS, color=ARMBLUE,
            fontweight="bold")
    ax.text(x0 + 1.5 * cw, 1.26, "the table E", ha="center", va="center", fontsize=FS_SMALL, color=GREY)
    ax.text(2.66, 1.26, "h · row", ha="left", va="center", fontsize=FS_SMALL, color=GREY)
    ax.text(3.45, 1.26, "softmax: the next token", ha="left", va="center", fontsize=FS_SMALL, color=GREY)
    for r, (n, y) in enumerate(zip(names, ys)):
        ax.text(x0 - 0.06, y, n, ha="right", va="center", fontsize=FS, family=MONO, color=INK)
        for c in range(3):
            rbox(ax, x0 + c * cw, y - ch / 2, cw - 0.03, ch, TINTS["red"] if n == "cat" else "white",
                 ARMRED if n == "cat" else "#999999")
            ax.text(x0 + c * cw + (cw - 0.03) / 2, y, f"{E[r, c]:.1f}", ha="center", va="center",
                    fontsize=FS_SMALL, family=MONO, color=INK, zorder=3)
        arrow(ax, (x0 + 3 * cw, y), (2.62, y), color=ARMBLUE, lw=0.5, ms=4)
        ax.text(2.66, y, f"{s[r]:+.2f}", ha="left", va="center", fontsize=FS_SMALL, family=MONO,
                color=INK)
        win = n == "sat"
        ax.add_patch(Rectangle((3.45, y - 0.07), 2.0 * p[r], 0.14, facecolor=ARMBLUE if win else TINTS["blue"],
                               edgecolor="none", zorder=2))
        ax.text(3.45 + 2.0 * p[r] + 0.05, y, f"{n} {100 * p[r]:.0f}%", ha="left", va="center",
                fontsize=FS_SMALL, color=ARMBLUE if win else GREY, fontweight="bold" if win else "normal")
    xr, _ = chip(fig, ax, 0.12, ys[1], "cat", fc=TINTS["red"], ec=ARMRED, below="ID 1")
    arrow(ax, (xr + 0.06, ys[1]), (x0 - 0.36, ys[1]), color=ARMRED, lw=0.9)
    ax.text(0.05, 0.10, "row 1 goes into the layers", ha="left", va="center", fontsize=FS_TINY, color=ARMRED)
    ax.text(2.62, 0.10, "h: the last layer's vector, here (-0.5, 0.9, 0.2)", ha="left", va="center",
            fontsize=FS_TINY, color=ARMBLUE)
    save(fig, "emb_unembed.pdf", log)


def fig_cos_sign(g, R, log):
    """Cosine as an angle: GPT-2's gap-1 and gap-100 averages from view 3, and a pair pointing apart."""
    fig, ax = canvas(1.0, 4.6)
    cases = [(QUOTED["pos_gap_cos"][1], "gap 1: 0.997\nalmost the same direction"),
             (QUOTED["pos_gap_cos"][100], "gap 100: 0.47\nabout 60 degrees apart"),
             (-0.5, "a few hundred apart: below 0\npointing more away than toward")]
    for c, (cosv, lab) in enumerate(cases):
        cx, cy, r = 0.75 + c * 1.55, 0.50, 0.36
        for a, col in ((15, ARMBLUE), (15 + math.degrees(math.acos(cosv)), ARMRED)):
            v = rot((1, 0), a) * r
            arrow(ax, (cx, cy), (cx + v[0], cy + v[1]), color=col, lw=1.3, ms=6)
        ax.text(cx, 0.14, lab, ha="center", va="center", fontsize=FS_TINY, color=INK, linespacing=1.15)
    save(fig, "emb_cos_sign.pdf", log)


def fig_pack_pad(g, R, log):
    """Padding vs packing: three 1,024-token windows each. Toy document lengths."""
    L, xa, xb = 1024, 1.30, 5.05
    sx = lambda t: xa + (xb - xa) * t / L
    pad_rows = [[("A", 300)], [("B", 650)], [("C", 1024)]]
    stream = [("A", 300), ("B", 650), ("C", 1400), ("D", 500), ("E", 400)]
    pack_rows, row, used = [], [], 0
    for name, n in stream:
        while n > 0 and len(pack_rows) < 3:
            take = min(n, L - used)
            row.append((name, take))
            used, n = used + take, n - take
            if used == L:
                pack_rows.append(row)
                row, used = [], 0
    tint = {"A": TINTS["blue"], "B": TINTS["orange"], "C": TINTS["red"], "D": "#D9EAD3", "E": TINTS["blue"]}
    fig, ax = canvas(1.85)
    hit = {}
    for title, sub, rows, y0 in (("padding", "one document per\nwindow, blanks after", pad_rows, 1.42),
                                 ("packing", "documents glued\nend to end", pack_rows, 0.68)):
        ax.text(0.05, y0 + 0.10, title, ha="left", va="center", fontsize=FS, color=INK, fontweight="bold")
        ax.text(0.05, y0 - 0.17, sub, ha="left", va="center", fontsize=FS_TINY, color=GREY, linespacing=1.15)
        filled = 0
        for k, segs in enumerate(rows):
            y, t = y0 + 0.12 - k * 0.20, 0
            for name, n in segs:
                ax.add_patch(Rectangle((sx(t), y - 0.07), sx(t + n) - sx(t), 0.14, facecolor=tint[name],
                                       edgecolor="white", lw=0.6, zorder=2))
                ax.text((sx(t) + sx(t + n)) / 2, y, name, ha="center", va="center", fontsize=FS_TINY,
                        color=INK, zorder=3)
                t += n
            if t < L:
                ax.add_patch(Rectangle((sx(t), y - 0.07), sx(L) - sx(t), 0.14, facecolor="white",
                                       edgecolor="#BBBBBB", lw=0.4, hatch="////", zorder=2))
            filled += int(t > 1000)
        hit[title] = filled
        ax.text(5.12, y0 - 0.08, f"{filled} of 3\nfilled", ha="left", va="center", fontsize=FS_TINY,
                color=ARMRED if filled < 3 else ARMBLUE)
    assert hit == {"padding": 1, "packing": 3}, hit
    ax.plot([sx(1000)] * 2, [0.18, 1.66], color=INK, lw=0.6, ls="--", zorder=4)
    ax.text(sx(1000), 1.75, "slot 1,000", ha="center", va="center", fontsize=FS_TINY, color=INK)
    ax.text(xa, 0.07, "each bar: one training window, slots 0 to 1,023", ha="left", va="center",
            fontsize=FS_TINY, color=GREY)
    R["pack_pad"] = {"windows_with_slot_1000_filled": hit}
    save(fig, "emb_pack_pad.pdf", log)


def fig_gap_counts(g, R, log):
    """How many (query, key) pairs a gap of d has inside one 1,024-token window: 1,024 - d."""
    L = QUOTED["n_positions"]
    gaps = np.arange(1, L)
    pairs = L - gaps
    for d, n in QUOTED["gap_pairs"].items():
        assert pairs[d - 1] == n, (d, pairs[d - 1])
    R["gap_counts"] = {"window": L, "pairs_at_gap": {str(d): int(pairs[d - 1]) for d in (1, 100, 500, 1000)}}
    fig = plt.figure(figsize=(3.1, 1.5))
    ax = fig.add_axes([0.17, 0.25, 0.79, 0.62])
    ax.fill_between(gaps, pairs, color=TINTS["blue"], lw=0)
    ax.plot(gaps, pairs, color=ARMBLUE, lw=1.0)
    for d in (1, 500, 1000):
        ax.plot(d, pairs[d - 1], "o", ms=2.8, color=ARMRED, zorder=3)
        if d < 900:
            ax.annotate(f"gap {d:,}: {pairs[d - 1]:,}", (d, pairs[d - 1]), xytext=(4, 4),
                        textcoords="offset points", fontsize=FS_TINY, color=ARMRED)
        else:                      # the line runs into the corner: label above it, with a leader line
            ax.annotate(f"gap {d:,}: {pairs[d - 1]:,}", (d, pairs[d - 1]), xytext=(700, 360),
                        textcoords="data", fontsize=FS_TINY, color=ARMRED,
                        arrowprops={"arrowstyle": "-", "color": ARMRED, "lw": 0.5})
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1150)
    ax.set_xticks([1, 256, 512, 768, 1023])
    ax.set_xlabel("gap between the two tokens", labelpad=1)
    ax.set_ylabel("pairs per window", labelpad=1)
    ax.set_title("one 1,024-token window", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_gap_counts.pdf", log)


def fig_one_pair(g, R, log):
    """Why many clock hands: one fast pair repeats (gap 44 looks like gap 0), one slow pair barely
    moves (gap 1 looks like gap 0); the 64 pairs of d = 128 added give one peak, at gap 0."""
    deg = QUOTED["one_pair_deg"]
    hand = lambda slot: np.array([math.sin(math.radians(deg * slot)), math.cos(math.radians(deg * slot))])
    for (a, b), v in QUOTED["one_pair"].items():
        assert round(float(hand(a) @ hand(b)), 2) == v, (a, b)
    gaps = np.arange(0, 101)
    d = 128
    om = 10000.0 ** (-2 * np.arange(d // 2) / d)
    fast, slow, total = np.cos(gaps * om[0]), np.cos(gaps * om[32]), np.cos(np.outer(gaps, om)).sum(1)
    ga, va = QUOTED["fast_alias"]
    assert om[0] == 1.0 and abs(om[32] - 0.01) < 1e-12
    assert r2(fast[ga]) == va and round(float(slow[1]), 5) == QUOTED["slow_gap1"], (fast[ga], slow[1])
    assert round(float(total[0])) == QUOTED["sum_peak"]
    far = int(np.argmax(total[20:])) + 20              # the best match 20 or more slots away
    assert round(float(total[1])) == QUOTED["sum_gap1"] and round(float(total[far])) == QUOTED["sum_far_max"]
    R["one_pair"] = {"deg_per_slot": deg, "by_hand": {f"{a},{b}": float(hand(a) @ hand(b))
                                                      for a, b in QUOTED["one_pair"]},
                     "fast_cos_gap44": float(fast[ga]), "slow_cos_gap1": float(slow[1]),
                     "sum_gap0": float(total[0]), "sum_gap1": float(total[1]),
                     "sum_best_beyond_20": [far, float(total[far])]}
    log.info(f"one pair: {R['one_pair']}")
    fig = plt.figure(figsize=(W, 1.55))
    specs = [("one fast pair (1 radian per slot)", fast, [(0, ""), (ga, f"gap {ga}: {fast[ga]:.2f}")]),
             ("one slow pair (0.01 radian per slot)", slow, [(1, f"gap 1: {slow[1]:.5f}")]),
             ("all 64 pairs added (d = 128)", total, [(0, f"gap 0: {total[0]:.0f}"),
                                                     (far, f"past gap 20: at most {total[far]:.0f}")])]
    for c, (title, y, marks) in enumerate(specs):
        ax = fig.add_axes([0.085 + c * 0.32, 0.24, 0.25, 0.58])
        ax.plot(gaps, y, color=ARMBLUE, lw=0.8, marker="o", ms=1.3)
        for x, lab in marks:
            ax.plot(x, y[x], "o", ms=3.0, color=ARMRED, zorder=3)
            ax.annotate(lab, (x, y[x]), xytext=(4, 3),
                        textcoords="offset points", fontsize=FS_TINY, color=ARMRED)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        if c < 2:
            ax.set_ylim(-1.2, 1.35)
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_ylim(float(total.min()) - 3, 72)
            ax.set_yticks([0, 32, 64])
        ax.set_xlabel("gap", labelpad=1)
        if c == 0:
            ax.set_ylabel("dot product", labelpad=1)
        ax.set_title(title, fontsize=FS_SMALL, loc="left")
        clean_axes(ax)
    save(fig, "emb_one_pair.pdf", log)


def expected_abs_cos(d):
    """Average |cos| between two random directions in d dimensions (exact)."""
    return math.exp(math.lgamma(d / 2) - math.lgamma((d + 1) / 2)) / math.sqrt(math.pi)


def fig_random_cos(g, R, log):
    """Random directions get closer to perpendicular as the dimension grows."""
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(3.2, 1.5))
    ax = fig.add_axes([0.05, 0.25, 0.91, 0.62])
    bins = np.linspace(0, 1, 41)
    out = {}
    for d, c in ((2, ARMORANGE), (20, ARMBLUE), (768, ARMRED)):
        X = rng.standard_normal((4000, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        a = np.abs((X[:2000] * X[2000:]).sum(1))
        ex = expected_abs_cos(d)
        assert round(ex, 3 if d == 768 else 2) == QUOTED["rand_abs_cos"][d], (d, ex)
        out[str(d)] = {"expected": ex, "sample_mean": float(a.mean())}
        ax.hist(a, bins=bins, histtype="step", color=c, lw=1.2, density=True,
                label=f"{d} dimensions: average {ex:.3f}" if d == 768 else f"{d} dimensions: average {ex:.2f}")
    R["random_cos"] = out
    ax.set_xlabel("|cosine| of two random directions (0 = perpendicular)", labelpad=1)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.legend(fontsize=FS_TINY, frameon=False, loc="upper right")
    ax.set_title("2,000 random pairs each", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    save(fig, "emb_random_cos.pdf", log)


def toy_concat_cost(g, R, log):
    """GPT-2's 12 layers at width 768, and at width 1,536 (positions side by side, 768 each).
    Per layer: attention 4 d^2 + 4 d, MLP 8 d^2 + 5 d, two LayerNorms 4 d."""
    per_layer = lambda d: 12 * d * d + 13 * d
    tot = {d: 12 * per_layer(d) for d in (768, 1536)}
    assert tot == QUOTED["layer_params"], tot
    assert QUOTED["wte_numbers"] + QUOTED["wpe_numbers"] + tot[768] + QUOTED["ln_f"] == QUOTED["params"]
    R["concat_cost"] = {"layers_params": {str(k): v for k, v in tot.items()}, "ratio": tot[1536] / tot[768]}
    log.info(f"concat cost: {R['concat_cost']}")


def leak_scores(rope):
    """Score of "sat" (slot m + 3) looking at "cat" (slot m), for m = 0..100: toy 16-number vectors,
    random query/key matrices. rope=False adds sinusoid positions to the inputs; rope=True turns
    the query and key instead."""
    rng = np.random.default_rng(SEED)
    d = 16
    e_cat, e_sat = rng.standard_normal(d), rng.standard_normal(d)
    Wq, Wk = rng.standard_normal((d, d)) / math.sqrt(d), rng.standard_normal((d, d)) / math.sqrt(d)
    P = sinusoid_pe(110, d)
    om = 10000.0 ** (-2 * np.arange(d // 2) / d)

    def turn(v, slot):
        x, y, a = v[0::2], v[1::2], slot * om
        out = np.empty(d)
        out[0::2], out[1::2] = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
        return out

    ms = np.arange(0, 101)
    if rope:
        return ms, np.array([turn(Wq @ e_sat, m + 3) @ turn(Wk @ e_cat, m) for m in ms])
    return ms, np.array([(Wq @ (e_sat + P[m + 3])) @ (Wk @ (e_cat + P[m])) for m in ms])


def fig_address_leak(g, R, log):
    """The same two tokens, always 3 apart, slid through the text: with added positions the score
    depends on where they sit; with RoPE it does not (second figure, the payoff)."""
    ms, add = leak_scores(False)
    _, rope = leak_scores(True)
    assert float(np.ptp(rope)) < 1e-9 and float(np.ptp(add)) > 1.0, (np.ptp(rope), np.ptp(add))
    R["address_leak"] = {"added_min": float(add.min()), "added_max": float(add.max()),
                         "added_at": {str(m): float(add[m]) for m in (1, 41)}, "rope_score": float(rope[0])}
    log.info(f"address leak: {R['address_leak']}")
    for name, with_rope in (("emb_address_leak.pdf", False), ("emb_address_leak_rope.pdf", True)):
        fig = plt.figure(figsize=(3.3, 1.55))
        ax = fig.add_axes([0.14, 0.25, 0.83, 0.62])
        ax.plot(ms, add, color="#BBBBBB" if with_rope else ARMBLUE, lw=0.9, marker="o", ms=1.3,
                label="positions added")
        if with_rope:
            ax.plot(ms, rope, color=ARMRED, lw=1.4, label="RoPE: turned")
            ax.legend(fontsize=FS_TINY, frameon=False, loc="lower right", ncol=2)
        else:
            for m in (int(np.argmax(add)), int(np.argmin(add))):
                ax.plot(m, add[m], "o", ms=3.0, color=ARMRED, zorder=3)
                right = m < 60      # 2026-11-06: the low label sat on the x axis; now beside its point
                ax.annotate(f"slots {m}, {m + 3}: {add[m]:+.1f}", (m, add[m]), xytext=(5, -2) if right else (-5, -2),
                            textcoords="offset points", ha="left" if right else "right", fontsize=FS_TINY,
                            color=ARMRED)
        ax.axhline(0, color="#DDDDDD", lw=0.5, zorder=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(min(add.min(), rope.min()) - (4.5 if with_rope else 2.2), max(add.max(), rope.max()) + 0.9)
        ax.set_xlabel("slot of \"cat\" (\"sat\" is 3 slots later)", labelpad=1)
        ax.set_ylabel("score", labelpad=1)
        ax.set_title("same two tokens, same gap, moved along", fontsize=FS_SMALL, loc="left")
        clean_axes(ax)
        save(fig, name, log)


def fig_rope_split(g, R, log):
    """Step 1 of RoPE: an 8-number query read as 4 arrows."""
    q = [0.8, 0.6, -0.5, 0.9, 1.0, 0.1, -0.3, -0.7]
    cols = [ARMRED, ARMBLUE, ARMORANGE, "#7832A0"]
    fig, ax = canvas(1.5)
    ax.text(0.05, 1.22, "a query\n(8 numbers)", ha="left", va="center", fontsize=FS_SMALL, color=INK,
            linespacing=1.2)
    ax.text(0.05, 0.50, "the same query\nas 4 arrows", ha="left", va="center", fontsize=FS_SMALL, color=INK,
            linespacing=1.2)
    x0, cw = 1.25, 0.50
    for k, v in enumerate(q):
        rbox(ax, x0 + k * cw, 1.12, cw - 0.04, 0.20, TINTS["grey"], cols[k // 2], lw=0.8)
        ax.text(x0 + k * cw + (cw - 0.04) / 2, 1.22, f"{v:+.1f}", ha="center", va="center", fontsize=FS_SMALL,
                family=MONO, color=INK, zorder=3)
    for p in range(4):
        cx, cy, r = x0 + (2 * p + 1) * cw - 0.02, 0.50, 0.30
        ax.plot([x0 + 2 * p * cw + 0.02, x0 + (2 * p + 2) * cw - 0.06], [1.05, 1.05], color=cols[p], lw=1.2)
        ax.text(cx, 0.96, f"pair {p}", ha="center", va="center", fontsize=FS_TINY, color=cols[p])
        ax.plot([cx - r, cx + r], [cy, cy], color="#CCCCCC", lw=0.5)
        ax.plot([cx, cx], [cy - r, cy + r], color="#CCCCCC", lw=0.5)
        x, y = q[2 * p], q[2 * p + 1]
        arrow(ax, (cx, cy), (cx + 0.26 * x, cy + 0.26 * y), color=cols[p], lw=1.2, ms=6)
        ax.text(cx, 0.08, f"({x:.1f}, {y:.1f})", ha="center", va="center", fontsize=FS_TINY, family=MONO,
                color=cols[p])
    save(fig, "emb_rope_split.pdf", log)


PLAIN_LOG = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")   # 1,000 not 10^3


def rope_speeds(d=128, base=10000.0):
    om = base ** (-2 * np.arange(d // 2) / d)
    return om, 2 * np.pi / om


def fig_rope_speeds(g, R, log):
    """The 64 speeds of a 128-number query (base 10,000): slots per full turn, and how many turns
    each pair makes in a 4,096-token training text."""
    om, per_turn = rope_speeds()
    for i, v in QUOTED["rope_slots_per_turn"].items():
        assert round(float(per_turn[i]), {0: 1, 63: -1}.get(i, 0)) == v, (i, per_turn[i])
    turns = 4096 / per_turn
    full = int((turns >= 1).sum())
    assert (full, 64 - full) == QUOTED["rope_turn_pairs_4096"], full
    R["rope_speeds"] = {"slots_per_turn": {str(i): float(per_turn[i]) for i in (0, 16, 32, 45, 46, 48, 63)},
                        "turns_in_4096": {str(i): float(turns[i]) for i in (0, 45, 46, 63)},
                        "pairs_full_turn_4096": full}
    i = np.arange(64)
    fig = plt.figure(figsize=(3.2, 1.6))
    ax = fig.add_axes([0.17, 0.24, 0.80, 0.63])
    ax.semilogy(i, per_turn, "o", ms=2.0, color=ARMBLUE)
    for k, lab in ((0, "6.3 slots"), (32, "628"), (63, "54,410")):
        ax.plot(k, per_turn[k], "o", ms=3.2, color=ARMRED, zorder=3)
        ax.annotate(lab, (k, per_turn[k]), xytext=(8, -5) if k == 0 else ((-4, 4) if k == 63 else (4, -9)),
                    textcoords="offset points", ha="right" if k == 63 else "left", fontsize=FS_TINY, color=ARMRED)
    ax.set_xticks([0, 16, 32, 48, 63])
    ax.set_ylim(2.5, 9e4)
    ax.set_xlabel("pair (0 = fastest, 63 = slowest)", labelpad=1)
    ax.set_ylabel("slots per full turn", labelpad=1)
    ax.yaxis.set_major_formatter(PLAIN_LOG)
    ax.set_title("64 pairs, 64 speeds", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_rope_speeds.pdf", log)

    fig = plt.figure(figsize=(4.4, 1.6))
    ax = fig.add_axes([0.12, 0.24, 0.86, 0.63])
    ok = turns >= 1
    ax.semilogy(i[ok], turns[ok], "o", ms=2.2, color=ARMBLUE, label=f"{full} pairs: at least one full turn")
    ax.semilogy(i[~ok], turns[~ok], "o", ms=2.2, color=ARMRED, label=f"{64 - full} pairs: never a full turn")
    ax.axhline(1, color=INK, lw=0.6, ls="--")
    ax.text(1, 1.25, "one full turn", fontsize=FS_TINY, color=INK)
    for k in (0, 63):
        ax.annotate(f"{turns[k]:.0f} turns" if k == 0 else f"{turns[k]:.3f} of a turn", (k, turns[k]),
                    xytext=(6, 3) if k == 0 else (-6, -11), textcoords="offset points",
                    ha="left" if k == 0 else "right", fontsize=FS_TINY, color=ARMBLUE if k == 0 else ARMRED)
    ax.set_xticks([0, 16, 32, 46, 63])
    ax.set_xlabel("pair (0 = fastest, 63 = slowest)", labelpad=1)
    ax.set_ylabel("turns in 4,096 slots", labelpad=1)
    ax.set_ylim(0.04, 5000)
    ax.yaxis.set_major_formatter(PLAIN_LOG)
    ax.legend(fontsize=FS_TINY, frameon=False, loc="upper right")
    ax.set_title("one 4,096-token training text", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_rope_speeds_train.pdf", log)


def plain_chip(ax, x, y, w, s, fc, ec, h=0.21):
    """A labelled box in the sans font (chip() is for tokens: mono, spaces shown)."""
    rbox(ax, x, y - h / 2, w, h, fc, ec, lw=0.8)
    ax.text(x + w / 2, y, s, ha="center", va="center", fontsize=FS_SMALL, color=INK, zorder=3)
    return x + w


def fig_rope_where(g, R, log):
    """Where position enters: once at the bottom (GPT-2) vs into q and k inside every layer (RoPE)."""
    fig, ax = canvas(1.62)                    # drawn at full text width, kept short for the frame
    for c, (title, rope) in enumerate((("GPT-2: added once, at the bottom", False),
                                       ("RoPE: q and k turned in every layer", True))):
        x0 = 0.10 + c * 2.78
        ax.text(x0 + 1.25, 1.53, title, ha="center", va="center", fontsize=FS, color=INK, fontweight="bold")
        if rope:
            plain_chip(ax, x0 + 0.85, 0.22, 0.80, "token row", TINTS["blue"], ARMBLUE)
            ax.text(x0 + 1.25, 0.04, "no position here", ha="center", va="center", fontsize=FS_TINY, color=GREY)
        else:
            xr = plain_chip(ax, x0 + 0.18, 0.22, 0.80, "token row", TINTS["blue"], ARMBLUE)
            ax.text(xr + 0.12, 0.22, "+", ha="center", va="center", fontsize=FS, color=INK)
            plain_chip(ax, xr + 0.24, 0.22, 0.95, "position row", TINTS["orange"], ARMORANGE)
        arrow(ax, (x0 + 1.25, 0.33), (x0 + 1.25, 0.42), color=GREY)
        for y, lab in ((0.42, "layer 1"), (0.98, "last layer")):
            rbox(ax, x0, y, 2.5, 0.38, "white", "#999999", lw=0.6)
            ax.text(x0 + 0.06, y + 0.30, lab, ha="left", va="center", fontsize=FS_TINY, color=GREY)
            for j, v in enumerate("qkv"):
                turned = rope and v != "v"
                cx = x0 + 0.62 + j * 0.42
                rbox(ax, cx - 0.16, y + 0.04, 0.32, 0.20, TINTS["orange"] if turned else TINTS["grey"],
                     ARMORANGE if turned else "#999999", lw=0.8 if turned else 0.5)
                ax.text(cx, y + 0.14, v + (" ↻" if turned else ""), ha="center", va="center", fontsize=FS_SMALL,
                        color=INK, zorder=3)
            ax.text(x0 + 1.98, y + 0.17, "attention,\nthen MLP", ha="center", va="center", fontsize=FS_TINY,
                    color=INK, linespacing=1.1)
        ax.text(x0 + 1.25, 0.88, "...", ha="center", va="center", fontsize=FS, color=GREY)
        arrow(ax, (x0 + 1.25, 0.80), (x0 + 1.25, 0.84), color=GREY, ms=4)
        arrow(ax, (x0 + 1.25, 0.93), (x0 + 1.25, 0.98), color=GREY, ms=4)
    ax.plot([2.79, 2.79], [0.05, 1.58], color="#DDDDDD", lw=0.6)
    save(fig, "emb_rope_where.pdf", log)


# ----------------------------------------------------------------------------- 2026-11-06 additions
# The instructor picked eight new visuals after a review of the deck (DECISIONS #112).

RVT_WORDS = [" coffee", " seven"]       # not " Monday": the next frames ask students to predict it


def fig_random_vs_trained(g, R, log):
    """'Who fills the table?': nearest rows in a random table (GPT-2's start: normal, std 0.02) vs GPT-2's."""
    rng = np.random.default_rng(SEED)
    E = rng.normal(0.0, 0.02, g.wte.shape)
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    out = {}
    fig, ax = canvas(1.18)                 # four cards in the style of emb_neighbours.pdf
    cw, ch = 1.28, 0.86
    for r, w in enumerate(RVT_WORDS):
        i = g.tid(w)
        sims = U @ U[i]
        rnd = [(g.text(j), float(sims[j])) for j in np.argsort(-sims) if j != i][:3]
        trn = g.neighbours(w, 3, skip_variants=True)
        out[w] = {"random": [(t, r2(s)) for t, s in rnd], "trained": [(t, r2(s)) for t, s in trn]}
        for c, (lst, trained) in enumerate([(rnd, False), (trn, True)]):
            k = 2 * r + c
            x, y = 0.04 + k * (cw + 0.06) + (0.08 if r else 0.0), 0.04
            rbox(ax, x, y, cw, ch, TINTS["blue"] if trained else "white", ARMBLUE if trained else "#BBBBBB",
                 lw=0.6, r=0.05, z=1)
            ax.text(x + cw / 2, y + ch + 0.13, "after training" if trained else "random start", ha="center",
                    va="center", fontsize=FS_SMALL, color=ARMBLUE if trained else GREY, fontweight="bold")
            chip(fig, ax, x + 0.08, y + ch - 0.17, w, fc=TINTS["orange"], ec=ARMORANGE, lw=0.8, bold=True,
                 xmax=x + cw - 0.05)
            for j, (t, s) in enumerate(lst):
                yy = y + ch - 0.40 - j * 0.15
                ax.text(x + 0.12, yy, show(t), ha="left", va="center", fontsize=FS_SMALL, family=MONO, color=INK)
                ax.text(x + cw - 0.08, yy, f"{s:.2f}", ha="right", va="center", fontsize=FS_TINY, family=MONO,
                        color=ARMBLUE if trained else GREY)
                if text_w(fig, show(t), FS_SMALL, MONO) + 0.12 + 0.35 > cw:
                    raise ValueError(f"neighbour {t!r} too wide for its card")
    R["random_vs_trained"] = out
    for w, v in QUOTED["rvt"].items():
        assert out[w]["trained"] == v["trained"], (w, out[w]["trained"])
        assert max(s for _, s in out[w]["random"]) == v["random_max"], (w, out[w]["random"])
    log.info(f"random vs trained: {out}")
    save(fig, "emb_random_vs_trained.pdf", log)


# A toy bigram model in 2-D: three input rows, two output rows held still, plain gradient descent on
# the softmax loss of the next word. Starting rows are small, as at a real start.
GRAD_OUT = {"morning": (1.5, 0.0), "split": (0.0, -1.5)}   # at right angles, so the paths stay clear of them
GRAD_START = {"Monday": (-0.15, 0.05), "Tuesday": (0.10, -0.12), "banana": (0.04, 0.16)}
GRAD_NEXT = {"Monday": "morning", "Tuesday": "morning", "banana": "split"}
GRAD_LR, GRAD_STEPS = 0.5, 200


def grad_toy_run():
    outs = list(GRAD_OUT)
    U = np.array([GRAD_OUT[o] for o in outs])
    paths, first = {}, {}
    for w, e0 in GRAD_START.items():
        e, y = np.array(e0, dtype=float), outs.index(GRAD_NEXT[w])
        path = [e.copy()]
        for step in range(GRAD_STEPS):
            z = U @ e
            p = np.exp(z - z.max())
            p /= p.sum()
            grad = U.T @ p - U[y]                # gradient of -log p(next word) w.r.t. the row
            if step == 0:
                first[w] = -GRAD_LR * grad
            e = e - GRAD_LR * grad
            path.append(e.copy())
        paths[w] = np.array(path)
    return paths, first


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fig_grad_step(g, R, log):
    paths, first = grad_toy_run()
    m0, t0, b0 = (paths[w][0] for w in ("Monday", "Tuesday", "banana"))
    m1, t1, b1 = (paths[w][-1] for w in ("Monday", "Tuesday", "banana"))
    res = {"cos_mon_tue": [r2(cosv(m0, t0)), r2(cosv(m1, t1))], "cos_mon_banana": [r2(cosv(m0, b0)), r2(cosv(m1, b1))],
           "first_push_cos_mon_tue": r2(cosv(first["Monday"], first["Tuesday"])),
           "first_push_cos_mon_banana": r2(cosv(first["Monday"], first["banana"]))}
    R["grad_toy"] = res
    log.info(f"gradient toy: {res}")
    assert res == QUOTED["grad_toy"], res
    col = {"Monday": ARMRED, "Tuesday": ARMORANGE, "banana": ARMBLUE}
    fig = plt.figure(figsize=(4.6, 1.62))   # 2026-11-09: drawn at its slot size
    a1 = fig.add_axes([0.02, 0.03, 0.40, 0.74])
    a2 = fig.add_axes([0.50, 0.03, 0.48, 0.82])
    for a in (a1, a2):
        a.set_aspect("equal")
        a.axis("off")
        a.set_xticks([])            # axis("off") hides tick labels but keeps them as texts
        a.set_yticks([])
    # left: the starting rows and the direction of their first update
    a1.set_xlim(-0.32, 0.32)
    a1.set_ylim(-0.25, 0.27)
    a1.plot([-0.32, 0.32], [0, 0], color="#DDDDDD", lw=0.5, zorder=0)
    a1.plot([0, 0], [-0.25, 0.27], color="#DDDDDD", lw=0.5, zorder=0)
    for w, e in GRAD_START.items():
        d = first[w] / np.linalg.norm(first[w]) * 0.10
        a1.plot(*e, "o", ms=3.5, color=col[w], zorder=3)
        arrow(a1, tuple(e), tuple(e + d), color=col[w], lw=1.0, ms=7)
        dx = -0.012 if w == "Monday" else 0.012
        a1.text(e[0] + dx, e[1] - 0.035, w, ha="right" if w == "Monday" else "left", va="center",
                fontsize=FS_SMALL, color=col[w])
    a1.set_title("step 1 (zoomed in): Monday and" + chr(10) + "Tuesday get the same push", fontsize=FS_SMALL, loc="left", pad=2)
    # right: 200 updates, with the two output rows
    for o, u in GRAD_OUT.items():
        u = np.array(u)
        arrow(a2, (0, 0), tuple(u), color=GREY, lw=0.8, ms=6)
        a2.text(u[0] + 0.12, u[1] - (0.0 if u[0] else 0.05), f"output row:\n{o}", ha="left", va="center",
                fontsize=FS_TINY, color=GREY, linespacing=1.1)
    for w, p in paths.items():
        a2.plot(p[:, 0], p[:, 1], color=col[w], lw=1.0)
        a2.plot(*p[0], "o", ms=2.5, color=col[w])
        a2.plot(*p[-1], "o", ms=3.5, color=col[w])
        left = w == "Monday"                 # Monday and Tuesday end side by side
        a2.text(p[-1][0] + (-0.12 if left else 0.12), p[-1][1], w, ha="right" if left else "left", va="center",
                fontsize=FS_SMALL, color=col[w])
    allp = np.vstack(list(paths.values()) + [np.array(list(GRAD_OUT.values()))])
    a2.set_xlim(allp[:, 0].min() - 0.9, allp[:, 0].max() + 1.0)
    a2.set_ylim(allp[:, 1].min() - 0.15, allp[:, 1].max() + 0.15)
    a2.text(allp[:, 0].min() - 0.8, allp[:, 1].max() - 0.45,
            f"Monday vs Tuesday, cosine:\n{res['cos_mon_tue'][0]:.2f} at the start,\n{res['cos_mon_tue'][1]:.2f} now",
            ha="left", va="top", fontsize=FS_TINY, color=INK, linespacing=1.15)
    a2.set_title(f"{GRAD_STEPS} steps later", fontsize=FS_SMALL, loc="left", pad=2)
    save(fig, "emb_grad_step.pdf", log)


# " bank" as the 6th token of every sentence, so all 12 inputs to layer 1 are the same vector
BANK_RIVER = ["He sat on the river bank", "We walked along the river bank", "Fish hid under the river bank",
              "Kids played on the river bank", "He fell off the river bank", "Weeds covered the river bank"]
BANK_MONEY = ["He paid cash at the bank", "We opened accounts at the bank", "I keep savings at the bank",
              "Thieves broke into the bank", "She borrowed money from the bank", "We paid fees to the bank"]
BANK_REF = ("The committee met on Tuesday to discuss the new budget. Several members raised concerns about "
            "the cost of the project, while others argued that the investment would pay off within a few "
            "years. After a long debate, the proposal was sent back for revision. Meanwhile, the weather "
            "turned cold and many people stayed at home, reading books or watching films with their families.")


def fig_bank_layers(g, R, log):
    """GPT-2's vector for " bank" after each block, river sentences vs money sentences. Each layer's
    vectors share a big common part (all cosines near 1 in the last layer), so the layer's mean over
    an ordinary paragraph (BANK_REF) is subtracted before comparing."""
    import itertools
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True).eval()

    def states(s):
        with torch.no_grad():
            o = model(**tok(s, return_tensors="pt"), output_hidden_states=True)
        return np.stack([h[0].numpy().astype(np.float64) for h in o.hidden_states])   # (13, T, 768)

    for s in BANK_RIVER + BANK_MONEY:
        ids = g.enc.encode(s)
        assert len(ids) == 6 and ids[-1] == g.tid(" bank"), (s, ids)
    mean = states(BANK_REF).mean(1)
    last = {s: states(s)[:, -1, :] for s in BANK_RIVER + BANK_MONEY}
    same, diff, raw_last = [], [], []
    for L in range(13):
        c = {s: v[L] - mean[L] for s, v in last.items()}
        same.append(float(np.mean([cosv(c[a], c[b]) for grp in (BANK_RIVER, BANK_MONEY)
                                   for a, b in itertools.combinations(grp, 2)])))
        diff.append(float(np.mean([cosv(c[a], c[b]) for a in BANK_RIVER for b in BANK_MONEY])))
        raw_last.append(float(np.mean([cosv(last[a][L], last[b][L]) for a in BANK_RIVER for b in BANK_MONEY])))
    res = {"same": [round(v, 3) for v in same], "different": [round(v, 3) for v in diff],
           "raw_different": [round(v, 3) for v in raw_last]}
    R["bank_layers"] = res
    log.info(f"bank by layer: {res}")
    q = QUOTED["bank"]
    assert abs(same[0] - 1) < 1e-6 and abs(diff[0] - 1) < 1e-6, (same[0], diff[0])
    assert (r2(same[-1]), r2(diff[-1])) == q["last"] and r2(raw_last[-1]) == q["raw_last_different"], res
    fig = plt.figure(figsize=(3.05, 1.80))
    ax = fig.add_axes([0.15, 0.22, 0.60, 0.66])
    xs = np.arange(13)
    ax.plot(xs, same, color=ARMBLUE, lw=1.3, marker="o", ms=2.5)
    ax.plot(xs, diff, color=ARMRED, lw=1.3, marker="o", ms=2.5)
    ax.text(12.4, same[-1], f"same meaning\n{same[-1]:.2f}", ha="left", va="center", fontsize=FS_TINY,
            color=ARMBLUE, linespacing=1.1)
    ax.text(12.4, diff[-1], f"different\nmeaning\n{diff[-1]:.2f}", ha="left", va="center", fontsize=FS_TINY,
            color=ARMRED, linespacing=1.1)
    ax.annotate("same row in: 1.00", (0, 1.0), xytext=(6, 4), textcoords="offset points", fontsize=FS_TINY,
                color=INK)
    ax.set_xlim(-0.4, 12.3)
    ax.set_ylim(0.45, 1.08)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("after block (0 = the input)", labelpad=1)
    ax.set_ylabel("cosine", labelpad=1)
    ax.set_title('GPT-2\'s vector for " bank"', fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_bank_layers.pdf", log)


def fig_two_embeddings(g, R, log):
    """The deck's token table vs an embedding model (one vector per whole text)."""
    fig, ax = canvas(1.42)
    vals = np.sin(np.arange(8) * 1.7)
    # left: the token table
    ax.text(0.04, 1.32, "this deck: the token table", ha="left", va="center", fontsize=FS_SMALL,
            color=ARMBLUE, fontweight="bold")
    for k, (ctx, y) in enumerate([('in "river bank"', 1.00), ('in "bank account"', 0.52)]):
        xr, _ = chip(fig, ax, 0.06, y, " bank", fs=FS_TINY, fc=TINTS["orange"], ec=ARMORANGE, h=0.19)
        ax.text(0.06, y - 0.18, ctx, ha="left", va="center", fontsize=FS_TINY, color=GREY)
        arrow(ax, (xr + 0.05, y), (xr + 0.30, y), lw=0.6, ms=5)
        ax.text(xr + 0.175, y + 0.08, "row 3331", ha="center", va="bottom", fontsize=FS_TINY, color=GREY)
        strip(ax, xr + 0.35, y, vals, 1.0, cell=0.06)
    ax.text(0.06, 0.10, "one token in, one row out: the same\nrow for both banks", ha="left", va="center",
            fontsize=FS_TINY, color=INK, linespacing=1.15)
    ax.plot([2.38, 2.38], [0.05, 1.38], color="#DDDDDD", lw=0.6)
    # right: an embedding model
    ax.text(2.52, 1.32, "an embedding model (search, RAG)", ha="left", va="center", fontsize=FS_SMALL,
            color=ARMRED, fontweight="bold")
    for k, (s, y, sh) in enumerate([("We sat on the river bank", 1.00, 0.0), ("I opened a bank account", 0.52, 1.3)]):
        xr, _ = chip(fig, ax, 2.54, y, s, fs=FS_TINY, fc="white", ec="#AAAAAA", h=0.19)
        arrow(ax, (xr + 0.04, y), (xr + 0.18, y), lw=0.6, ms=5)
        rbox(ax, xr + 0.20, y - 0.11, 0.62, 0.22, TINTS["grey"], "#999999")
        ax.text(xr + 0.51, y, "all layers", ha="center", va="center", fontsize=FS_TINY, color=INK)
        arrow(ax, (xr + 0.84, y), (xr + 0.98, y), lw=0.6, ms=5)
        strip(ax, xr + 1.00, y, np.sin(np.arange(8) * 1.7 + sh), 1.0, cell=0.045)
    ax.text(2.54, 0.10, "a whole text in, one vector out, made from the last layer:\nsimilar texts get similar vectors",
            ha="left", va="center", fontsize=FS_TINY, color=INK, linespacing=1.15)
    save(fig, "emb_two_embeddings.pdf", log)


def fig_rope_stretch(g, R, log):
    """Position interpolation vs YaRN on the clock discs of emb_rope_unseen (4,096 -> 32,768, factor 8)."""
    d, base, train, longer = 128, 10000, 4096, 32768
    s = longer // train
    fast, slow = 1.0, base ** (-(d - 2) / d)
    fast_deg = {"as is": math.degrees(fast), "pi": math.degrees(fast / s), "yarn": math.degrees(fast)}
    slow_frac = {"as is": slow * longer / (2 * math.pi), "pi": slow / s * longer / (2 * math.pi),
                 "yarn": slow / s * longer / (2 * math.pi)}
    seen = slow * train / (2 * math.pi)
    res = {"factor": s, "fast_step_deg": {k: round(v, 1) for k, v in fast_deg.items()},
           "slow_fraction": {k: round(v, 3) for k, v in slow_frac.items()}, "seen_fraction": round(seen, 3)}
    R["rope_stretch"] = res
    log.info(f"rope stretch: {res}")
    assert res == QUOTED["rope_stretch"], res
    fig, ax = canvas(2.0, w=4.75)          # 2026-11-09: drawn at its slot size (was 5.5 in, shown at 80%)
    cols = [("as trained", "as is"), ("position interpolation:\nevery hand 8x slower", "pi"),
            ("YaRN: only the slow\nhands 8x slower", "yarn")]
    xs, r = [1.70, 2.95, 4.20], 0.27
    for c, (title, key) in enumerate(cols):
        ax.text(xs[c], 1.88, title, ha="center", va="center", fontsize=FS_SMALL, color=INK, linespacing=1.1)
        # top row: the fastest hand, slots 0 and 1
        cx, cy = xs[c], 1.36
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#BBBBBB", lw=0.6, zorder=3))
        ang = fast_deg[key]
        ax.add_patch(Wedge((cx, cy), r * 0.45, 0, ang, facecolor=TINTS["orange"], edgecolor="none", zorder=1))
        for a_deg, colr in ((0, ARMBLUE), (ang, ARMRED)):
            arrow(ax, (cx, cy), (cx + r * math.cos(math.radians(a_deg)), cy + r * math.sin(math.radians(a_deg))),
                  color=colr, lw=1.0, ms=6, z=4)
        note = f"{ang:.0f} degrees apart" + (": harder\nto tell apart" if key == "pi" else "\n(unchanged)" if key == "yarn" else "\n")
        ax.text(cx, cy - r - 0.15, note, ha="center", va="center", fontsize=FS_TINY,
                color=ARMRED if key == "pi" else GREY, linespacing=1.1)
        # bottom row: the slowest hand over 32,768 slots
        cy = 0.55
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="#BBBBBB", lw=0.6, zorder=3))
        f = slow_frac[key]
        ax.add_patch(Wedge((cx, cy), r, 90 - 360 * min(f, seen), 90, facecolor="#CFD8EE", edgecolor="none", zorder=0.8))
        if f > seen + 1e-9:
            ax.add_patch(Wedge((cx, cy), r, 90 - 360 * f, 90 - 360 * seen, facecolor="#F9D2D5", edgecolor="none",
                               zorder=0.8))
        note = (f"{100 * f:.0f}% of a turn,\nmostly never seen" if f > seen + 1e-9
                else f"{100 * f:.1f}% of a turn:\nall seen in training")
        ax.text(cx, cy - r - 0.13, note, ha="center", va="center", fontsize=FS_TINY,
                color=ARMRED if f > seen + 1e-9 else GREY, linespacing=1.1)
    ax.text(0.02, 1.36, "fastest hand:\nslot 0 (blue) and\nslot 1 (red)", ha="left", va="center", fontsize=FS_SMALL,
            color=INK, linespacing=1.15)
    ax.text(0.02, 0.55, "slowest hand:\nall 32,768 slots", ha="left", va="center", fontsize=FS_SMALL, color=INK,
            linespacing=1.15)
    save(fig, "emb_rope_stretch.pdf", log)


ALIBI_SLOPES = [2.0 ** -k for k in range(1, 9)]      # Press et al. (2022), section 3: 8 heads


def fig_alibi(g, R, log):
    dist = np.arange(0, 21)
    R["alibi"] = {"slopes": ALIBI_SLOPES, "penalty_at_20": [round(-m * 20, 3) for m in ALIBI_SLOPES]}
    assert R["alibi"]["penalty_at_20"][0] == -10.0 and R["alibi"]["penalty_at_20"][-1] == round(-20 / 256, 3)
    fig = plt.figure(figsize=(2.65, 1.75))
    ax = fig.add_axes([0.21, 0.22, 0.57, 0.66])
    shades = plt.get_cmap("Blues")(np.linspace(0.95, 0.35, len(ALIBI_SLOPES)))
    for m, c in zip(ALIBI_SLOPES, shades):
        ax.plot(dist, -m * dist, color=c, lw=1.1)
    for m, lab in ((ALIBI_SLOPES[0], "slope 1/2"), (ALIBI_SLOPES[1], "1/4"), (ALIBI_SLOPES[2], "1/8"),
                   (ALIBI_SLOPES[-1], "1/256")):
        ax.text(20.6, -m * 20, lab, ha="left", va="center", fontsize=FS_TINY, color=ARMBLUE)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("how many tokens back", labelpad=1)
    ax.set_yticks([0, -5, -10])
    ax.set_ylabel("added to the score", labelpad=1)
    ax.set_title("one slope per head, fixed", fontsize=FS_SMALL, loc="left")
    clean_axes(ax)
    save(fig, "emb_alibi.pdf", log)


def fig_nope(g, R, log):
    words = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(words)
    fig, ax = canvas(1.75, w=2.45)
    cell, x0, y0 = 0.20, 0.55, 0.20
    for i in range(n):                      # row i: the token doing the looking
        y = y0 + (n - 1 - i) * cell
        ax.text(x0 - 0.06, y + cell / 2, words[i], ha="right", va="center", fontsize=FS_TINY, color=INK)
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * cell, y), cell, cell, facecolor=TINTS["blue"] if j <= i else "white",
                                   edgecolor="#BBBBBB", lw=0.5))
        ax.text(x0 + n * cell + 0.06, y + cell / 2, f"sees {i + 1}", ha="left", va="center", fontsize=FS_TINY,
                color=ARMBLUE)
    for j in range(n):
        ax.text(x0 + (j + 0.5) * cell, y0 - 0.08, words[j], ha="center", va="center", fontsize=FS_TINY - 0.5,
                color=GREY, rotation=0)
    ax.text(0.04, y0 + n * cell + 0.14, "each token looks back only", ha="left", va="center", fontsize=FS_SMALL,
            color=INK)
    save(fig, "emb_nope.pdf", log)


# RULER (Hsieh et al., 2024), the leaderboard in github.com/NVIDIA/RULER README, read 2026-11-06.
# Effective length = the longest tested length (4K to 128K) where the model still beats Llama-2-7B's
# score at 4K (85.6%). Values in K tokens. Rows the RULER authors measured themselves, plus two 2025 rows
# the model makers reported (asterisk in the README; marked * and hatched here), so the chart does not
# read as if 2024 were today (2026-11-09, after a "you should know" note).
# Rows: Gemini-1.5-pro, glm-4-9b-chat-1m, Llama-3-70B-Instruct-Gradient-1048k, internlm2_5-7b-chat-1m,
# LWM-Text-Chat-1M, Yi-34B-200K, GPT-4-1106-preview (GPT-4 Turbo), Meta-Llama-3.1-8B-Instruct.
RULER_ROWS = [("Gemini 1.5 Pro (2024)", 1024, ">128K"), ("GLM-4 9B (2024)", 1024, 64),
              ("Llama 3 70B, 1M (2024)", 1024, 16), ("InternLM2.5 7B (2024)", 1024, 4),
              ("LWM 7B (2024)", 1024, "<4K"), ("Yi 34B (2023)", 200, 32),
              ("GPT-4 Turbo (2023)", 128, 64), ("Llama 3.1 8B (2024)", 128, 32),
              ("Qwen2.5 14B, 1M (2025)*", 1024, ">128K"), ("Qwen3-8B (2025)*", 128, 64)]


def fig_ruler(g, R, log):
    def kfmt(v):
        return "1M" if v == 1024 else f"{v}K"
    R["ruler"] = [{"model": m, "claimed_k": c, "effective": e, "maker_reported": m.endswith("*")}
                  for m, c, e in RULER_ROWS]
    fig = plt.figure(figsize=(3.25, 2.05))      # 2026-11-09: shorter, so the frame fits at full size
    ax = fig.add_axes([0.42, 0.11, 0.44, 0.78])
    ys = np.arange(len(RULER_ROWS))[::-1]
    for y, (m, c, e) in zip(ys, RULER_ROWS):
        ax.barh(y + 0.19, c, height=0.36, color="#DDDDDD", edgecolor="#AAAAAA", lw=0.4)
        ax.text(c * 1.12, y + 0.19, kfmt(c), ha="left", va="center", fontsize=FS_TINY, color=GREY)
        ev = 128 if e == ">128K" else 2 if e == "<4K" else e
        maker = m.endswith("*")
        ax.barh(y - 0.19, ev, height=0.36, color="white" if maker else (ARMBLUE if e != "<4K" else ARMRED),
                edgecolor=ARMBLUE, hatch="////" if maker else None, lw=0.6)
        ax.text(ev * 1.12, y - 0.19, e if isinstance(e, str) else f"{e}K", ha="left", va="center",
                fontsize=FS_TINY, color=ARMBLUE if e != "<4K" else ARMRED, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_xticks([])
    ax.minorticks_off()
    ax.set_yticks(ys, [m for m, _, _ in RULER_ROWS], fontsize=FS_TINY)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.02, 0.98, "grey: claimed length    blue: length that holds up    (log scale)", ha="left", va="top",
             fontsize=FS_TINY, color=INK)
    fig.text(0.02, 0.02, "* hatched: run by the model's makers, not the RULER team", ha="left", va="bottom",
             fontsize=FS_TINY, color=GREY)
    save(fig, "emb_ruler.pdf", log)


# StreamingLLM (Xiao et al., 2024), Table 1: Llama-2-13B, the first book of PG19 (65K tokens)
SINKS_PPL = [("first 4 tokens + last 1,020", 5.40), ("4 line breaks + last 1,020", 5.60),
             ("last 1,024 tokens only", 5158.07)]


def fig_sinks_ppl(g, R, log):
    R["sinks_ppl"] = SINKS_PPL
    fig = plt.figure(figsize=(2.6, 1.25))
    ax = fig.add_axes([0.50, 0.08, 0.30, 0.70])
    ys = np.arange(len(SINKS_PPL))[::-1]
    vals = [v for _, v in SINKS_PPL]
    bars = ax.barh(ys, vals, color=[ARMBLUE, ARMBLUE, ARMRED], height=0.55)
    ax.bar_label(bars, labels=[f"{v:,.2f}" if v < 100 else f"{v:,.0f}" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.set_xticks([])
    ax.minorticks_off()
    ax.set_yticks(ys, [k for k, _ in SINKS_PPL], fontsize=FS_TINY)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.02, 0.97, "perplexity (lower is better), log scale", ha="left", va="top", fontsize=FS_TINY,
             color=GREY)
    save(fig, "emb_sinks_ppl.pdf", log)


NO_WEIGHTS = {"fig_sinusoids", "fig_rope_clock", "fig_rope_2d", "toy_rope_pairs", "fig_rope_unseen",
              "fig_grad_step", "fig_two_embeddings", "fig_rope_stretch", "fig_alibi", "fig_nope", "fig_ruler",
              "fig_sinks_ppl",
              "toy_shuffle", "fig_unembed", "fig_cos_sign", "fig_pack_pad", "fig_gap_counts", "fig_one_pair", "fig_random_cos",
              "toy_concat_cost", "fig_address_leak", "fig_rope_split", "fig_rope_speeds", "fig_rope_where"}


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    jobs = [fig_kargin, fig_coldopen, fig_table_sizes, fig_neighbours, fig_neighbours_odd, fig_glitch,
            compute_analogies, fig_add_position, fig_gpt2_positions, fig_sinusoids, fig_add_vs_concat,
            fig_rope_clock, fig_rope_2d, toy_rope_pairs, fig_rope_unseen, toy_shuffle,
            fig_unembed, fig_cos_sign, fig_pack_pad, fig_gap_counts, fig_one_pair, fig_random_cos, toy_concat_cost,
            fig_address_leak, fig_rope_split, fig_rope_speeds, fig_rope_where,
            fig_random_vs_trained, fig_grad_step, fig_bank_layers, fig_two_embeddings, fig_rope_stretch,
            fig_alibi, fig_nope, fig_ruler, fig_sinks_ppl]
    out = RESULTS / "embeddings_position_figs.json"
    if "--only" in sys.argv:              # rerun some jobs; their numbers are merged into the JSON
        names = sys.argv[sys.argv.index("--only") + 1:]
        by_name = {j.__name__: j for j in jobs}
        unknown = [n for n in names if n not in by_name]
        if not names or unknown:
            raise SystemExit(f"--only needs job names from {sorted(by_name)}; unknown: {unknown}")
        jobs = [by_name[n] for n in names]
        R = json.loads(out.read_text(encoding="utf-8"))
        R.setdefault("partial_reruns", []).append({"date": time.strftime("%Y-%m-%d"), "jobs": names})
    else:
        R = {"date": time.strftime("%Y-%m-%d"), "tiktoken": tiktoken.__version__, "model": "gpt2 (124M), HF cache"}
    g = None if all(j.__name__ in NO_WEIGHTS for j in jobs) else GPT2()
    for k, job in enumerate(jobs, 1):
        job(g, R, log)
        el = time.time() - t0
        log.info(f"{k}/{len(jobs)} {job.__name__}, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left")
    out.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
