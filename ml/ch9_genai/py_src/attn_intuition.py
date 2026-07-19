"""Intuition figures for the attention deck (L24). No model needed (numpy + matplotlib).

Generates into ml/ch9_genai/fig/:
  softmax_scaling.pdf   -- why divide by sqrt(d_k): raw QK^T saturates softmax; scaling fixes it.
  attn_illustrative.pdf -- a CLEAN, hand-set attention pattern on the chapter sentence, for
                           teaching the mechanics (labelled illustrative).

Run: ./ma/Scripts/python.exe ml/ch9_genai/py_src/attn_intuition.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED, ARMORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
LOGS = ROOT / "logs"


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("attn_intuition")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS / "attn_intuition.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def fig_softmax_scaling(log):
    rng = np.random.default_rng(SEED)
    dk = 64
    q = rng.standard_normal(dk)
    K = rng.standard_normal((6, dk))
    raw = K @ q                     # dot products; variance ~ dk
    scaled = raw / np.sqrt(dk)
    w_raw = softmax(raw)
    w_scaled = softmax(scaled)
    labels = [f"key {i+1}" for i in range(6)]
    log.info(f"raw score std={raw.std():.1f}, scaled std={scaled.std():.2f}; "
             f"max w_raw={w_raw.max():.2f}, max w_scaled={w_scaled.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].bar(labels, w_raw, color=ARMRED)
    axes[0].set_title(f"softmax of raw $QK^\\top$   (scores spread $\\approx\\pm{raw.std():.0f}$)",
                      fontsize=11)
    axes[1].bar(labels, w_scaled, color=ARMBLUE)
    axes[1].set_title("softmax of $QK^\\top / \\sqrt{d_k}$   ($\\approx\\pm 1$)", fontsize=11)
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("attention weight")
        ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=8, padding=1)
        ax.tick_params(axis="x", labelrotation=20)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(r"Why $\sqrt{d_k}$:  big dimensions make dot products large $\to$ softmax "
                 r"collapses to a near-hard argmax ($d_k=64$)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIG / "softmax_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def fig_illustrative(log):
    toks = ["Donkey", "don't", "die", "spring", "is", "coming"]
    n = len(toks)
    A = np.array([
        [0.55, 0.05, 0.30, 0.03, 0.02, 0.05],  # Donkey -> itself + die
        [0.10, 0.42, 0.38, 0.03, 0.05, 0.02],  # don't  -> die (negation)
        [0.46, 0.20, 0.28, 0.03, 0.01, 0.02],  # die    -> Donkey (who dies)
        [0.03, 0.02, 0.03, 0.50, 0.05, 0.37],  # spring -> coming
        [0.03, 0.03, 0.03, 0.26, 0.35, 0.30],  # is     -> spring/coming
        [0.02, 0.02, 0.02, 0.56, 0.08, 0.30],  # coming -> spring
    ])
    A = A / A.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=float(A.max()))
    ax.set_xticks(range(n))
    ax.set_xticklabels(toks, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(toks, fontsize=10)
    ax.set_xlabel("attended-to word", fontsize=10)
    ax.set_ylabel("query word", fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if A[i, j] > 0.33 else "#555")
    ax.set_title("An attention pattern that makes sense", fontsize=12)
    fig.suptitle('"Donkey don\'t die, spring is coming"  (illustrative weights)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "attn_illustrative.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def fig_dot_product(log):
    q = np.array([2.4, 0.0])
    keys = [
        ("aligned", np.array([2.1, 0.35]), ARMBLUE),
        ("$45^\\circ$", np.array([1.5, 1.5]), ARMORANGE),
        ("perpendicular", np.array([0.0, 2.1]), "#7a7a7a"),
        ("opposite", np.array([-1.9, 0.5]), ARMRED),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.axhline(0, color="#e5e5e5", lw=0.8)
    ax.axvline(0, color="#e5e5e5", lw=0.8)
    ax.annotate("", xy=q, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3.2, color="black"))
    ax.text(q[0] + 0.05, q[1] - 0.28, "query $\\mathbf{q}$", fontsize=12, fontweight="bold")
    for name, k, col in keys:
        ax.annotate("", xy=k, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col))
        d = float(q @ k)
        ax.text(k[0] * 1.06, k[1] * 1.06 + (0.12 if k[1] >= 0 else -0.2),
                f"{name}\n$\\mathbf{{q}}\\cdot\\mathbf{{k}}={d:.1f}$", fontsize=9.5, color=col)
    ax.set_xlim(-2.8, 3.4)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Dot product $=$ alignment  (big when they point the same way)", fontsize=12)
    fig.tight_layout()
    out = FIG / "dot_product_geometry.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def fig_worked_scores(log):
    toks = ["fluffy", "blue", "creature"]
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    S = X @ X.T
    scaled = S / np.sqrt(2.0)
    W = np.vstack([softmax(r) for r in scaled])
    log.info(f"worked example: creature weights = {W[2].round(2)}; output = {(W[2] @ X).round(2)}")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, M, title, fmt in [
        (axes[0], S, "scores  $S = QK^\\top$  (dot products)", "%.0f"),
        (axes[1], W, "weights  $=\\ \\mathrm{softmax}(S/\\sqrt{d})$", "%.2f"),
    ]:
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=float(M.max()))
        ax.set_xticks(range(3))
        ax.set_xticklabels(toks, fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(toks, fontsize=10)
        ax.set_xlabel("key (attended-to)", fontsize=9)
        ax.set_ylabel("query", fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > M.max() * 0.6 else "#333")
        ax.set_title(title, fontsize=11)
    fig.suptitle("A tiny worked example  ($Q=K=V=X$, 2-D embeddings)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG / "worked_scores.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_softmax_scaling(log)
    fig_illustrative(log)
    fig_dot_product(log)
    fig_worked_scores(log)
    log.info("done")


if __name__ == "__main__":
    main()
