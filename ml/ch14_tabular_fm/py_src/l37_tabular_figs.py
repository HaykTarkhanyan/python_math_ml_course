"""Concept figures for L37 (Tabular Foundation Models).

NOTHING IS TRAINED, DOWNLOADED OR INSTALLED anywhere in this chapter (instructor decision,
TABULAR_FM_CHAPTER_PLAN.md 2026-08-07). No TabPFN weights are fetched, which also keeps the
chapter clear of the TabPFN-2.5 licence. Every figure is either exact arithmetic or a
clearly-labelled schematic drawn from a specified generative process.

Generates into ml/ch14_tabular_fm/fig/:
  pfn_inversion.pdf   -- fit-to-your-data vs fit-to-a-prior-over-datasets
  prior_samples.pdf   -- real draws from a small structural-causal-model prior
  two_d_attention.pdf -- the two-way attention: across a row, then down a column
  context_cost.pdf    -- O((n.m)^2) naive vs O(n^2 + m^2) factorised, exact
  scaling_timeline.pdf-- the six papers that exist only to make it take more rows

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch14_tabular_fm/py_src/l37_tabular_figs.py
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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

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

    fig, axes = plt.subplots(1, n_show, figsize=(11.0, 2.9))
    for k, ax in enumerate(axes):
        # The prior really does emit degenerate draws, but a panel that is 97% one class shows
        # the viewer nothing about mechanism diversity, which is what this figure is for.
        for attempt in range(200):
            x, y, act = draw()
            if 0.25 <= y.mean() <= 0.75:
                break
        else:
            raise RuntimeError("no balanced draw in 200 attempts - prior or filter is wrong")
        ax.scatter(x[y == 0, 0], x[y == 0, 1], s=11, c=BLUE, alpha=0.75, linewidths=0)
        ax.scatter(x[y == 1, 0], x[y == 1, 1], s=11, c=RED, alpha=0.75, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"synthetic dataset {k + 1}\n({act} mechanism)", fontsize=8.5)
        log.info(f"prior sample {k + 1}: mechanism={act}, class balance={y.mean():.2f}")
    fig.suptitle("Draws from a prior over structural causal models - "
                 "TabPFN pre-trains on millions of these, and on no real tables",
                 fontsize=11, y=1.06)
    out = FIG / "prior_samples.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_two_d_attention(n=6, m=5):
    """Two-way attention: across a row (features), then down a column (samples)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
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
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("features (m)", fontsize=9)
        if mode == "features":
            ax.set_ylabel("samples (n)", fontsize=9)
        ax.set_xlim(-0.4, m + 0.4); ax.set_ylim(-n + 0.6, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Every cell gets its own representation, and attends twice per layer",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    out = FIG / "two_d_attention.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------------------------------
def fig_context_cost():
    """Exact arithmetic: flattening all cells vs the two-way factorisation.

    Nature (Hollmann et al. 2025): compute is O(n^2 + m^2), memory O(n.m). A naive transformer
    over all n*m cells as one sequence would be O((n*m)^2).
    """
    m = 100
    ns = np.array([100, 500, 1000, 5000, 10000])
    naive = (ns * m) ** 2.0
    fact = ns ** 2.0 + m ** 2.0
    for n, a, b in zip(ns, naive, fact):
        log.info(f"n={n:6d} m={m}: naive={a:.3e}  factorised={b:.3e}  ratio={a / b:,.0f}x")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(ns, naive, "o-", color=RED, lw=2.2, ms=7,
            label=r"naive: all $n\cdot m$ cells in one sequence, $(n\cdot m)^2$")
    ax.plot(ns, fact, "s-", color=BLUE, lw=2.2, ms=7,
            label=r"TabPFN two-way attention, $O(n^2+m^2)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rows $n$  (features $m=100$ throughout)", fontsize=10)
    ax.set_ylabel("attention pairs (log scale)", fontsize=10)
    ratio = naive[-1] / fact[-1]
    ax.annotate(f"at $n$=10,000:\n{ratio:,.0f}x cheaper",
                xy=(ns[-1], fact[-1]), xytext=(1200, fact[-1] * 0.02),
                fontsize=10, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.set_title("Why the factorisation is not just tidy\n"
                 "quadratic in each dimension separately, not in their product", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.plot(xs, np.zeros_like(xs), "-", color="0.75", lw=2, zorder=1)
    for i, (date, name) in enumerate(papers):
        ax.scatter([i], [0], s=150, color=BLUE, zorder=3)
        up = 1 if i % 2 == 0 else -1
        ax.annotate(name, xy=(i, 0), xytext=(i, 0.42 * up),
                    ha="center", va="bottom" if up > 0 else "top", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=1.0))
        ax.text(i, -0.18 * up, date, ha="center",
                va="top" if up > 0 else "bottom", fontsize=8, color="0.4")
    ax.set_ylim(-1.15, 1.15); ax.set_xlim(-0.6, len(papers) - 0.4)
    ax.axis("off")
    ax.set_title("Six papers whose whole purpose is fitting more rows in the context\n"
                 "that is the shape of the method, not a frontier", fontsize=11)
    fig.tight_layout()
    out = FIG / "scaling_timeline.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_pfn_inversion()
    fig_prior_samples()
    fig_two_d_attention()
    fig_context_cost()
    fig_scaling_timeline()
    log.info("done - 5 figures")
