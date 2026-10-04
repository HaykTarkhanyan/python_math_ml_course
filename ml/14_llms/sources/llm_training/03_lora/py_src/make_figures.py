"""Figures for the LoRA deck. Run: python3 make_figures.py -> ../fig/*.pdf"""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

ARM_RED = "#C81E28"; ARM_BLUE = "#1E46A0"; ARM_ORANGE = "#E6A01E"
GREEN = "#008C46"; VIOLET = "#7832A0"; GREY = "#8a8a8a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.titlesize": 13, "figure.dpi": 140})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig"); os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", os.path.normpath(p))


def fig_param_count():
    # GPT-3 175B trainable params (millions), log scale
    methods = ["Full FT", "Adapter$_H$", "Prefix-layer", "LoRA (37.7M)", "LoRA (4.7M)"]
    params = [175255.8, 40.1, 20.2, 37.7, 4.7]
    colors = [ARM_RED, GREY, GREY, ARM_BLUE, ARM_BLUE]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.barh(y, params, color=colors, zorder=3, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(methods)
    ax.invert_yaxis()
    for yi, p in zip(y, params):
        ax.text(p * 1.3, yi, f"{p:,.1f}M", va="center", fontsize=10)
    ax.set_xlabel("trainable parameters (millions, log scale)")
    ax.set_title("GPT-3 175B: LoRA trains ~10,000x fewer parameters")
    ax.set_xlim(1, 6e5)
    ax.annotate("", xy=(4.7, 4), xytext=(175255.8, 0),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
    ax.text(300, 2, r"$\approx$10,000$\times$", color=GREEN, fontsize=13, weight="bold")
    save(fig, "param_count.pdf")


def fig_rank():
    # Table 6, WikiSQL accuracy vs rank r
    r = [1, 2, 4, 8, 64]
    x = np.arange(len(r))
    wq = [68.8, 69.6, 70.5, 70.4, 70.0]
    wqv = [73.4, 73.3, 73.7, 73.8, 73.5]
    all4 = [74.1, 73.7, 74.0, 74.0, 73.9]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(x, wq, "-o", color=GREY, lw=2, label=r"$W_q$ only")
    ax.plot(x, wqv, "-o", color=ARM_BLUE, lw=2.4, label=r"$\{W_q, W_v\}$")
    ax.plot(x, all4, "-o", color=GREEN, lw=2, label=r"$\{W_q,W_k,W_v,W_o\}$")
    ax.set_xticks(x); ax.set_xticklabels(r)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("WikiSQL accuracy (%)")
    ax.set_title("Rank as low as $r=1$ already suffices for $\\{W_q,W_v\\}$")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(67, 75)
    ax.annotate("flat from r=1:\nlow intrinsic rank", (0, wqv[0]), (1.3, 71.3),
                fontsize=9.5, color=ARM_BLUE,
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE))
    save(fig, "rank.pdf")


def fig_which_matrices():
    # Table 5, fixed 18M budget
    cfg = [r"$W_q$", r"$W_k$", r"$W_v$", r"$W_o$", r"$W_q,W_k$", r"$W_q,W_v$", "all 4"]
    wikisql = [70.4, 70.0, 73.0, 73.2, 71.4, 73.7, 73.7]
    mnli = [91.0, 90.8, 91.0, 91.3, 91.3, 91.3, 91.7]
    x = np.arange(len(cfg)); w = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, data, title, lo, hi in [
        (axes[0], wikisql, "WikiSQL", 68, 75),
        (axes[1], mnli, "MultiNLI", 90, 92)]:
        cols = [GREEN if c in (r"$W_q,W_v$", "all 4") else ARM_BLUE for c in cfg]
        ax.bar(x, data, color=cols, zorder=3, width=0.62)
        ax.set_xticks(x); ax.set_xticklabels(cfg, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(lo, hi); ax.set_title(title)
        ax.set_ylabel("accuracy (%)")
        for xi, d in zip(x, data):
            ax.text(xi, d + (hi - lo) * 0.01, f"{d:.1f}", ha="center", fontsize=8)
    fig.suptitle("Same 18M budget: spread it over $\\{W_q,W_v\\}$ (green), not one matrix",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "which_matrices.pdf")


def fig_latency():
    # Table 1, GPT-2 medium, ms
    labels = ["FT / LoRA", "Adapter$_L$", "Adapter$_H$"]
    ms = [19.8, 23.9, 25.8]
    colors = [ARM_BLUE, ARM_ORANGE, ARM_RED]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.bar(x, ms, color=colors, zorder=3, width=0.55)
    for xi, m in zip(x, ms):
        ax.text(xi, m + 0.2, f"{m} ms", ha="center", fontsize=10)
    ax.text(1, 24.6, "+20.7%", ha="center", color=ARM_ORANGE, fontsize=10, weight="bold")
    ax.text(2, 26.5, "+30.3%", ha="center", color=ARM_RED, fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("inference latency (ms)")
    ax.set_title("Online inference (batch 1): adapters add latency, LoRA adds none")
    ax.set_ylim(0, 29)
    save(fig, "latency.pdf")


def fig_amplification():
    # Table 7, layer 48, Frobenius norms of delta-W projected onto top-r subspace of W
    cats = [r"$\|U^\top W_q V\|$" + "\n(self)", r"$\|U^\top \Delta W_q V\|$", "random"]
    r4 = [0.32, 21.67, 0.02]
    r64 = [1.90, 37.71, 0.33]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w / 2, r4, w, label="r=4", color=ARM_BLUE, zorder=3)
    ax.bar(x + w / 2, r64, w, label="r=64", color=ARM_ORANGE, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Frobenius norm (log)")
    ax.set_title(r"$\Delta W$ amplifies task directions already in $W$ (~21$\times$ at r=4)")
    ax.legend(frameon=False)
    ax.text(1, 28, "amplify 21.5x", ha="center", color=GREEN, fontsize=10, weight="bold")
    save(fig, "amplification.pdf")


def fig_low_rank_intuition():
    # "what low rank means": a rank-3 matrix = sum of 3 rank-1 outer products
    rng = np.random.default_rng(0)
    d = 12
    scales = [1.0, 0.7, 0.5]
    terms = [scales[i] * np.outer(rng.standard_normal(d), rng.standard_normal(d))
             for i in range(3)]
    total = sum(terms)
    mats = terms + [total]
    titles = [r"$b_1 a_1^{\top}$", r"$b_2 a_2^{\top}$", r"$b_3 a_3^{\top}$",
              r"$\Delta W=\sum_i b_i a_i^{\top}$"]
    vmax = np.abs(total).max()
    fig, axes = plt.subplots(1, 7, figsize=(11, 3.0),
                             gridspec_kw={"width_ratios": [1, 0.28, 1, 0.28, 1, 0.42, 1]})
    mat_axes = [axes[0], axes[2], axes[4], axes[6]]
    for ax, op in zip((axes[1], axes[3], axes[5]), ("+", "+", "=")):
        ax.axis("off"); ax.text(0.5, 0.5, op, ha="center", va="center", fontsize=22)
    for i, (ax, M, t) in enumerate(zip(mat_axes, mats, titles)):
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(t, fontsize=12.5, color=(ARM_BLUE if i == 3 else "#333"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
        if i < 3:
            ax.set_xlabel("rank 1", fontsize=9.5, color=GREY)
        else:
            ax.set_xlabel("rank 3", fontsize=9.5, color=ARM_BLUE)
    fig.suptitle(r"A rank-$r$ update = a sum of $r$ rank-1 blocks $\Rightarrow$ store the thin "
                 r"factors $B\,(d\times r)$, $A\,(r\times k)$, not the full grid",
                 fontsize=12, y=1.04)
    save(fig, "low_rank_intuition.pdf")


def fig_singular_decay():
    # "how we get the low rank": the update's singular values collapse, so the
    # best rank-r approx (truncated SVD, Eckart-Young) is nearly lossless.
    rng = np.random.default_rng(0)
    d, r_true = 64, 6
    W0 = rng.standard_normal((d, d)) / np.sqrt(d)          # full-rank pretrained weight
    U = rng.standard_normal((d, r_true))
    V = rng.standard_normal((r_true, d))
    signal = U @ np.diag(np.array([1.0, 0.72, 0.52, 0.38, 0.26, 0.17])) @ V
    dW = signal / np.linalg.norm(signal) + 0.0022 * rng.standard_normal((d, d))  # ~low-rank update
    s0 = np.linalg.svd(W0, compute_uv=False)
    sd = np.linalg.svd(dW, compute_uv=False)
    s0n, sdn = s0 / s0[0], sd / sd[0]
    energy = np.cumsum(sd ** 2) / np.sum(sd ** 2)
    r95 = int(np.searchsorted(energy, 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    k = np.arange(1, 25)
    ax.plot(k, s0n[:24], "-o", color=GREY, ms=4, lw=1.8, label=r"$W_0$ (pretrained)")
    ax.plot(k, sdn[:24], "-o", color=ARM_BLUE, ms=4, lw=2.3, label=r"$\Delta W$ (the update)")
    ax.set_yscale("log")
    ax.axvspan(0.5, r_true + 0.5, color=ARM_ORANGE, alpha=0.13)
    ax.text(r_true + 1.0, sdn[0] * 0.35, "top few\ncarry it all", color="#B77A00",
            fontsize=9.5, weight="bold")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("magnitude (norm., log)")
    ax.set_title(r"$\Delta W$'s singular values fall off a cliff")
    ax.legend(frameon=False)

    ax = axes[1]
    rr = np.arange(1, 25)
    ax.plot(rr, energy[:24], "-o", color=ARM_BLUE, ms=4, lw=2.3)
    ax.axhline(0.95, color=GREY, ls=":", lw=1.2)
    ax.axvline(r95, color=ARM_RED, ls="--", lw=1.6)
    ax.text(r95 + 0.5, 0.55, f"rank {r95} rebuilds\n95% of $\\Delta W$",
            color=ARM_RED, fontsize=10, weight="bold")
    ax.set_xlabel(r"rank $r$ kept (top-$r$ SVD)")
    ax.set_ylabel(r"fraction of $\Delta W$ rebuilt")
    ax.set_title("Best rank-$r$ approx (Eckart--Young) is near-lossless")
    ax.set_ylim(0, 1.03)
    fig.tight_layout()
    save(fig, "singular_decay.pdf")


if __name__ == "__main__":
    fig_param_count(); fig_rank(); fig_which_matrices(); fig_latency(); fig_amplification()
    fig_low_rank_intuition(); fig_singular_decay()
    print("all LoRA figures done")
