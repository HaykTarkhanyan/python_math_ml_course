"""Concept figures for L37 (Tabular Foundation Models).

NOTHING IS TRAINED, DOWNLOADED OR INSTALLED anywhere in this chapter (instructor decision,
TABULAR_FM_CHAPTER_PLAN.md 2026-08-07). No TabPFN weights are fetched, which also keeps the
chapter clear of the TabPFN-2.5 licence. Every figure is either exact arithmetic or a
clearly-labelled schematic drawn from a specified generative process.

Generates into ml/ch14_tabular_fm/fig/:
  pfn_inversion.pdf   -- fit-to-your-data vs fit-to-a-prior-over-datasets
  prior_samples.pdf   -- real draws from a small structural-causal-model prior
  two_d_attention.pdf -- the two-way attention: across a row, then down a column
  context_cost.pdf    -- total attention-pair count: flat vs row/column attention
  bnn_prior.pdf       -- random network weights define different functions
  attention_walkthrough.pdf -- a test row's cell attends across features, then examples
  scaling_timeline.pdf-- the six papers that exist only to make it take more rows

Run with the project venv:
    ./ma/.venv/Scripts/python.exe ml/ch14_tabular_fm/py_src/l37_tabular_figs.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "l37_tabular_figs.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


log = build_logger()


# ---------------------------------------------------------------------------------------
def fig_pfn_inversion():
    """The whole chapter in one picture: what gets fitted to what."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    def box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.6,
                               zorder=2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, fontweight=weight)

    # Left: ordinary supervised learning.
    ax = axes[0]
    box(ax, 0.5, 3.0, 3.0, 0.9, "your dataset", "#eeeeee", "0.5")
    box(ax, 0.5, 1.4, 3.0, 0.9, "fit a model\n(gradient steps)", BLUE + "22", BLUE)
    box(ax, 0.5, -0.2, 3.0, 0.9, "predictions", ORANGE + "33", ORANGE)
    for y0, y1 in [(3.0, 2.3), (1.4, 0.7)]:
        ax.annotate("", xy=(2.0, y1), xytext=(2.0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="0.35"))
    ax.text(2.0, 4.35, "Ordinary supervised learning", ha="center", fontsize=11,
            fontweight="bold")
    ax.text(2.0, -0.75, "training happens HERE,\nevery time, on your data",
            ha="center", fontsize=8.5, color=BLUE, style="italic")
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-1.4, 4.8); ax.axis("off")

    # Right: the PFN inversion.
    ax = axes[1]
    box(ax, 0.2, 3.0, 3.6, 0.9, "a PRIOR over datasets", RED + "22", RED, weight="bold")
    box(ax, 0.2, 1.4, 3.6, 0.9, "pre-train one transformer\non millions of synthetic tables",
        BLUE + "22", BLUE, fs=8.5)
    box(ax, 0.2, -0.2, 3.6, 0.9, "your dataset -> predictions\n(one forward pass, no fitting)",
        ORANGE + "33", ORANGE, fs=8.5)
    for y0, y1 in [(3.0, 2.3), (1.4, 0.7)]:
        ax.annotate("", xy=(2.0, y1), xytext=(2.0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="0.35"))
    ax.text(2.0, 4.35, "A prior-data fitted network", ha="center", fontsize=11,
            fontweight="bold")
    ax.text(2.0, -0.75, "training happened ONCE, before you,\nand never on real data",
            ha="center", fontsize=8.5, color=RED, style="italic")
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-1.4, 4.8); ax.axis("off")

    out = FIG / "pfn_inversion.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_prior_samples(n_show=4, n=120):
    """Make "sampled from a prior over structural causal models" concrete.

    Each panel is a dataset drawn from a small random SCM: a random DAG over 3 latent causes,
    random nonlinearities, then two observed features and a label. This is a SCHEMATIC of the
    idea, not TabPFN's actual prior, and the slide says so.
    """
    rng = np.random.default_rng(SEED)

    def draw():
        """One dataset from a tiny random structural causal model."""
        z = rng.normal(size=(n, 3))
        w = rng.normal(size=(3, 2)) * rng.choice([0.6, 1.8], size=(3, 2))
        act = rng.choice(["tanh", "square", "linear"])
        h = z @ w
        x = {"tanh": np.tanh(h), "square": h ** 2 * 0.4, "linear": h}[act]
        x += rng.normal(scale=0.25, size=x.shape)
        beta = rng.normal(size=2)
        y = (x @ beta + rng.normal(scale=0.4, size=n) > 0).astype(int)
        return x, y, act

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 3.5), constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        # The prior really does emit degenerate draws, but a panel that is 97% one class shows
        # the viewer nothing about mechanism diversity, which is what this figure is for.
        for attempt in range(200):
            x, y, act = draw()
            if 0.25 <= y.mean() <= 0.75:
                break
        else:
            raise RuntimeError("no balanced draw in 200 attempts - prior or filter is wrong")
        ax.scatter(x[y == 0, 0], x[y == 0, 1], s=12, c=BLUE, alpha=0.75, linewidths=0)
        ax.scatter(x[y == 1, 0], x[y == 1, 1], s=12, c=RED, alpha=0.75, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Task {k + 1}: {act} mechanism", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        log.info(f"prior sample {k + 1}: mechanism={act}, class balance={y.mean():.2f}")
    out = FIG / "prior_samples.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_two_d_attention(n=6, m=5):
    """Two-way attention: across a row (features), then down a column (samples)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    focus = (2, 2)

    for ax, mode in zip(axes, ["features", "samples"]):
        for i in range(n):
            for j in range(m):
                if (i, j) == focus:
                    fc, ec = RED, RED
                elif mode == "features" and i == focus[0]:
                    fc, ec = ORANGE + "66", ORANGE
                elif mode == "samples" and j == focus[1]:
                    fc, ec = ORANGE + "66", ORANGE
                else:
                    fc, ec = "white", "0.75"
                ax.add_patch(Rectangle((j, -i), 1, 1, facecolor=fc, edgecolor=ec, lw=1.2))
        ax.add_patch(Rectangle((0, -n + 1), m, n, fill=False, edgecolor="0.4", lw=1.6))
        title = ("1. Attention over FEATURES\na cell attends across its own row"
                 if mode == "features" else
                 "2. Attention over SAMPLES\nthe same cell attends down its own column")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("features (m)", fontsize=10)
        if mode == "features":
            ax.set_ylabel("samples (n)", fontsize=10)
        ax.set_xlim(-0.4, m + 0.4); ax.set_ylim(-n + 0.6, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Each cell has its own vector; each layer attends along two axes",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    out = FIG / "two_d_attention.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_context_cost():
    """All attention pairs: flat (nm)^2 vs n*m^2 + m*n^2 for axial attention.

    The Nature paper abbreviates scaling along each axis as O(n^2 + m^2), but that is not
    the total pair count across every row and column. Never divide totals by per-axis counts.
    """
    m = 100
    ns = np.array([100, 500, 1000, 5000, 10000])
    naive = (ns * m) ** 2.0
    fact = ns * m ** 2.0 + m * ns ** 2.0
    for n, a, b in zip(ns, naive, fact):
        log.info(f"n={n:6d} m={m}: naive={a:.3e}  factorised={b:.3e}  ratio={a / b:,.0f}x")

    fig, ax = plt.subplots(figsize=(6.2, 3.1), constrained_layout=True)
    ax.plot(ns, naive, "o-", color=RED, lw=2.2, ms=6,
            label=r"Flat: $(nm)^2$")
    ax.plot(ns, fact, "s-", color=BLUE, lw=2.2, ms=6,
            label=r"Row + column: $nm(n+m)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rows $n$  (100 features throughout)", fontsize=10)
    ax.set_ylabel("Total attention pairs", fontsize=10)
    ratio = naive[-1] / fact[-1]
    ax.annotate(f"{ratio:,.0f}x fewer pairs", xy=(ns[-1], fact[-1]),
                xytext=(1200, fact[-1] * 0.24), fontsize=10, color=BLUE,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.18, which="major")
    ax.spines[["top", "right"]].set_visible(False)
    out = FIG / "context_cost.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_scaling_timeline():
    """Six papers whose entire purpose is getting more rows into the context."""
    papers = [
        ("2023-11", "Sketching +\nfeature selection"),
        ("2024-02", "In-context\ndata distillation"),
        ("2024-02", "TuneTables\n(context optimization)"),
        ("2025-02", "TabPFN\nUnleashed"),
        ("2025-08", "Chunked\nTabPFN"),
        ("2025-10", "TabPFN-Wide\n(many features)"),
    ]
    xs = np.arange(len(papers))
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.plot(xs, np.zeros_like(xs), "-", color="0.75", lw=2, zorder=1)
    for i, (date, name) in enumerate(papers):
        ax.scatter([i], [0], s=150, color=BLUE, zorder=3)
        up = 1 if i % 2 == 0 else -1
        ax.annotate(name, xy=(i, 0), xytext=(i, 0.42 * up),
                    ha="center", va="bottom" if up > 0 else "top", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=1.0))
        ax.text(i, -0.18 * up, date, ha="center",
                va="top" if up > 0 else "bottom", fontsize=9, color="0.4")
    ax.set_ylim(-1.15, 1.15); ax.set_xlim(-0.6, len(papers) - 0.4)
    ax.axis("off")
    # NB: the title deliberately does NOT say "the shape of the method, not a frontier" - the
    # ceiling moved ~100x between the Nature paper and TabPFN-3, so that framing is wrong and
    # would contradict the slide's own callout.
    fig.tight_layout()
    out = FIG / "scaling_timeline.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
# Slow-down figures. Added 2026-08-08 (instructor: explain it slower and better). The first
# half of this deck asserts its hardest ideas - "a prior over datasets", "the training table
# IS the prompt" - in prose. These make each one concrete before it is used.

def _box(ax, x, y, w, h, text, color, fs=8, alpha=0.16):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, color="0.4", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig_scm_walkthrough(n=5):
    """ONE synthetic dataset being born, step by step.

    The deck previously said "draw a random causal graph, draw random mechanisms, push noise
    through, read off features and a label" in a single sentence and moved on. This walks
    through exactly that sentence once, with real numbers, so "sampled from a prior over
    datasets" stops being an incantation.
    """
    rng = np.random.default_rng(SEED + 3)

    # 1. a random causal graph over two roots and one derived feature
    # 2. random mechanisms
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x3 = np.tanh(1.7 * z1 - 0.9 * z2) + rng.normal(scale=0.15, size=n)
    y = ((0.8 * x3 + 0.5 * z2 + rng.normal(scale=0.3, size=n)) > 0).astype(int)

    fig = plt.figure(figsize=(8.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25], wspace=0.25)

    # panel 1: the graph
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.axis("off")
    pos = {"z1": (0.5, 2.3), "z2": (2.3, 2.3), "x3": (1.4, 1.3), "y": (1.4, 0.35)}
    for name, (px, py) in pos.items():
        c = {"y": RED}.get(name, BLUE)
        ax.add_patch(plt.Circle((px, py), 0.28, fc=c, ec=c, alpha=0.2, lw=1.4))
        ax.add_patch(plt.Circle((px, py), 0.28, fc="none", ec=c, lw=1.4))
        ax.text(px, py, f"${name[0]}_{name[1]}$" if name != "y" else "$y$",
                ha="center", va="center", fontsize=10)
    for a, b in [("z1", "x3"), ("z2", "x3"), ("x3", "y"), ("z2", "y")]:
        _arrow(ax, *pos[a], *pos[b], color="0.45")
    ax.set_title("1. draw a random causal graph", fontsize=9.5)

    # panel 2: the mechanisms
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_title("2. draw random mechanisms", fontsize=9.5)     # aligned with the other two
    ax.text(0.5, 0.72, r"$x_3 = \tanh(1.7 z_1 - 0.9 z_2) + \epsilon$",
            ha="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.50, r"$y = \mathbb{1}[\,0.8 x_3 + 0.5 z_2 + \epsilon > 0\,]$",
            ha="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.22, "the shapes, the weights and the\nnoise are all sampled too",
            ha="center", fontsize=8, color="0.4", transform=ax.transAxes)

    # panel 3: the resulting table
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.set_title("3. push noise through, read off a table", fontsize=9.5)
    cols = [r"$z_1$", r"$z_2$", r"$x_3$", "$y$"]
    tbl = ax.table(cellText=[[f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", str(d)]
                             for a, b, c, d in zip(z1, z2, x3, y)],
                   colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.12)
    for k in range(len(cols)):
        tbl[0, k].set_facecolor("#dfe6f2")
    for r in range(1, n + 1):
        tbl[r, 3].set_facecolor("#f7dcdc")
    fig.suptitle("This is ONE dataset. TabPFN's pre-training does this about 100 million "
                 "times, with a different graph every time.", fontsize=9.5, y=0.02)
    out = FIG / "scm_walkthrough.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_incontext_layout():
    """What actually goes into the context window: the training table IS the prompt."""
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 3.4); ax.axis("off")

    # the training block
    ax.add_patch(Rectangle((0.2, 1.15), 3.3, 1.75, fc=BLUE, ec=BLUE, alpha=0.10, lw=1.4))
    ax.text(1.85, 3.0, "training rows", ha="center", fontsize=9, color=BLUE)
    for r in range(5):
        for c in range(4):
            ax.add_patch(Rectangle((0.45 + c * 0.62, 2.5 - r * 0.3), 0.55, 0.24,
                                   fc=BLUE, ec="white", alpha=0.35, lw=0.6))
        ax.add_patch(Rectangle((0.45 + 4 * 0.62, 2.5 - r * 0.3), 0.55, 0.24,
                               fc=RED, ec="white", alpha=0.5, lw=0.6))
    ax.text(3.25, 2.62, "y", fontsize=8, color=RED, ha="center")

    # the test block
    ax.add_patch(Rectangle((3.75, 1.15), 2.2, 1.75, fc=ORANGE, ec=ORANGE, alpha=0.10, lw=1.4))
    ax.text(4.85, 3.0, "test rows", ha="center", fontsize=9, color="#B07800")
    for r in range(3):
        for c in range(4):
            ax.add_patch(Rectangle((3.95 + c * 0.42, 2.4 - r * 0.3), 0.36, 0.24,
                                   fc=ORANGE, ec="white", alpha=0.45, lw=0.6))
        ax.text(5.72, 2.52 - r * 0.3, "?", fontsize=9, color=RED, ha="center", va="center")

    _arrow(ax, 6.05, 2.0, 6.55, 2.0)
    _box(ax, 6.6, 1.35, 1.9, 1.3, "one frozen\ntransformer\n\nONE forward pass", "#7832A0",
         fs=8.5)
    _arrow(ax, 8.55, 2.0, 9.05, 2.0)
    _box(ax, 9.1, 1.5, 1.5, 1.0, "predictive\ndistribution\nfor each ?", "#008C46", fs=8)

    ax.text(5.4, 0.62, "Everything to the left of the arrow is the PROMPT. "
                       "No weights change at any point.",
            ha="center", fontsize=9.5, color="0.2")
    ax.text(5.4, 0.2, "This is in-context learning from ch9, with a spreadsheet in the "
                      "context instead of sentences.",
            ha="center", fontsize=8.5, style="italic", color="0.4")
    out = FIG / "incontext_layout.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_one_model_many_datasets(n=90):
    """One frozen network, three different datasets, three different answers, no training."""
    rng = np.random.default_rng(SEED)

    t = rng.uniform(0, np.pi, n)
    moons = np.c_[np.r_[np.cos(t), 1 - np.cos(t)],
                  np.r_[np.sin(t), 0.5 - np.sin(t)]] + rng.normal(0, 0.09, (2 * n, 2))
    moons_y = np.r_[np.zeros(n), np.ones(n)]

    blobs = np.r_[rng.normal([-1, -1], 0.45, (n, 2)), rng.normal([1.1, 1.0], 0.45, (n, 2))]
    blobs_y = np.r_[np.zeros(n), np.ones(n)]

    xor = rng.uniform(-1.5, 1.5, (2 * n, 2))
    xor_y = ((xor[:, 0] > 0) ^ (xor[:, 1] > 0)).astype(float)

    fig = plt.figure(figsize=(7.9, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.3)
    for k, (X, y, name) in enumerate([(moons, moons_y, "dataset A"),
                                      (blobs, blobs_y, "dataset B"),
                                      (xor, xor_y, "dataset C")]):
        ax = fig.add_subplot(gs[k])
        ax.scatter(*X[y == 0].T, s=9, color=BLUE, alpha=0.75)
        ax.scatter(*X[y == 1].T, s=9, color=RED, alpha=0.75)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9.5)

    ax = fig.add_subplot(gs[3])
    ax.axis("off")
    _box(ax, 0.05, 0.42, 0.9, 0.3, "Frozen network\nsame weights",
         "#7832A0", fs=10)
    for k in range(3):
        ax.annotate("", xy=(0.05, 0.57), xytext=(-0.25 - k * 0.02, 0.85 - k * 0.22),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.1))
    ax.text(0.5, 0.24, "three datasets,\nthree rules",
            ha="center", fontsize=10, transform=ax.transAxes, color="0.25")
    fig.suptitle("What is learned is not a decision boundary. It is the RULE that turns a "
                 "table into one.", fontsize=9.5, y=0.02)
    out = FIG / "one_model_many_datasets.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_column_meaning():
    """Why attending DOWN a column is not decoration: a raw number means nothing alone."""
    rng = np.random.default_rng(SEED)
    age = rng.normal(37, 9, 400)
    salary = rng.lognormal(mean=np.log(2200), sigma=0.5, size=400)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), constrained_layout=True)
    for ax, data, name, unit in [(axes[0], age, "column: age", "years"),
                                 (axes[1], salary, "column: monthly pay", "currency")]:
        ax.hist(data, bins=24, color=BLUE, alpha=0.65)
        ax.axvline(37, color=RED, lw=2.2)
        pct = (data < 37).mean() * 100
        ax.text(0.98, 0.96, f"37: {pct:.0f}th percentile", color=RED, fontsize=9,
                va="top", ha="right", transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel(unit, fontsize=10)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        log.info(f"column-meaning figure: 37 is at the {pct:.0f}th percentile of {name}")
    out = FIG / "column_meaning.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_bnn_prior():
    """Two weight draws from one small neural-network family give two task functions."""
    rng = np.random.default_rng(SEED + 37)
    x = np.linspace(-2.6, 2.6, 160)
    fig, ax = plt.subplots(figsize=(6.2, 2.5), constrained_layout=True)
    colors = [BLUE, RED]
    for k, color in enumerate(colors):
        w_in = rng.normal(size=(1, 8))
        bias = rng.normal(size=8)
        w_out = rng.normal(size=8)
        y = np.tanh(x[:, None] * w_in + bias) @ w_out
        y = y / np.std(y)
        ax.plot(x, y, color=color, lw=2.5, label=f"Draw weights {chr(65+k)}: one function")
        samples = np.array([-2.1, -1.2, -0.3, 0.6, 1.6, 2.3])
        values = np.interp(samples, x, y) + rng.normal(0, 0.12, len(samples))
        ax.scatter(samples, values, s=25, color=color, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xlabel("input x", fontsize=10)
    ax.set_ylabel("target y", fontsize=10)
    ax.set_title("Two sampled functions from the same network family", fontsize=11)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    out = FIG / "bnn_prior.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


def fig_attention_walkthrough():
    """A worked table: row comparison, then column comparison with masked test label."""
    rows = [
        ("train 1", "24", "112", "negative", "no"),
        ("train 2", "61", "155", "positive", "yes"),
        ("train 3", "43", "128", "negative", "no"),
        ("test", "58", "150", "positive", "?"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    headers = ["age", "pressure", "test", "label"]
    for stage, ax in enumerate(axes):
        ax.set_xlim(-0.9, 4.1); ax.set_ylim(-4.4, 0.9); ax.axis("off")
        ax.set_title("1  Compare features in the test row" if stage == 0 else
                     "2  Compare this feature across rows", fontsize=10, loc="left")
        for c, header in enumerate(headers):
            ax.text(c + 0.5, 0.36, header, ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="0.2")
        for r, vals in enumerate(rows):
            ax.text(-0.08, -r - 0.5, vals[0], ha="right", va="center", fontsize=8.5)
            for c, value in enumerate(vals[1:]):
                focus = r == 3 and c == 2
                context = (stage == 0 and r == 3 and c < 3) or (stage == 1 and c == 2 and r < 3)
                fc = RED if focus else ("#FCE8B2" if context else "#F2F4F7")
                ax.add_patch(Rectangle((c, -r - 1), 0.98, 0.94,
                                       facecolor=fc, edgecolor="white", lw=1.4))
                ax.text(c + 0.49, -r - 0.53, value, ha="center", va="center",
                        fontsize=9, color="white" if focus else "0.18",
                        fontweight="bold" if focus else "normal")
        ax.text(1.95, -4.18,
                "Combine 58, 150 and positive" if stage == 0 else
                "Compare positive test results and known labels",
                ha="center", va="center", fontsize=8.5, color=BLUE)
    out = FIG / "attention_walkthrough.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_pfn_inversion()
    fig_prior_samples()
    fig_two_d_attention()
    fig_context_cost()
    fig_scaling_timeline()
    fig_scm_walkthrough()
    fig_incontext_layout()
    fig_one_model_many_datasets()
    fig_column_meaning()
    fig_bnn_prior()
    fig_attention_walkthrough()
    log.info("done - 11 figures")
