"""Residual-stream figure for L45 - the chapter's load-bearing concept.

Replaces a hand-drawn TikZ sketch. Two panels:

  left   the stream as a channel: every block READS the running vector and ADDS
         its output back, so nothing is ever overwritten;
  right  the same claim MEASURED on GPT-2 small - the norm of the residual
         stream at the final token position as it passes through 12 blocks,
         with the size of each block's write shown underneath.

The right panel is the point. "Everything is a sum" is easy to assert and easy
to nod along to; watching the norm accumulate makes it concrete, and the
reconstruction check at the end proves the sum is exact rather than approximate.

Usage (about 20 s on CPU with GPT-2 small cached; --plot-only redraws from
results/residual_stream.json in seconds):
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_residual_stream.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_residual_stream.py --plot-only
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from mi_common import FIG_DIR, load_results, save_results, setup_logging

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def measure(model, log):
    """Norm of the residual stream after each block, and each block's contribution."""
    from ioi_core import load_prompts
    data = load_prompts(model, log, n=64)
    _, cache = model.run_with_cache(data["clean"])

    n_layers = model.cfg.n_layers
    pos = -1  # final token: the position the answer is read off

    resid_norms = [cache["resid_pre", 0][:, pos, :].norm(dim=-1).mean().item()]
    attn_writes, mlp_writes = [], []
    for layer in range(n_layers):
        attn_writes.append(cache["attn_out", layer][:, pos, :].norm(dim=-1).mean().item())
        mlp_writes.append(cache["mlp_out", layer][:, pos, :].norm(dim=-1).mean().item())
        resid_norms.append(cache["resid_post", layer][:, pos, :].norm(dim=-1).mean().item())

    # The lecture claims the final residual IS the sum of the embedding and every write.
    # Verify it rather than assert it: rebuild the final vector from the parts.
    rebuilt = cache["resid_pre", 0][:, pos, :].clone()
    for layer in range(n_layers):
        rebuilt = rebuilt + cache["attn_out", layer][:, pos, :] + cache["mlp_out", layer][:, pos, :]
    actual = cache["resid_post", n_layers - 1][:, pos, :]
    gap = (rebuilt - actual).abs().max().item()
    if gap > 2e-3:
        raise RuntimeError(
            f"residual stream is not the sum of its parts (max gap {gap:.5f}). The slide claims "
            "this decomposition is exact, so it must not ship as an approximation."
        )
    log.info("reconstruction check: max |rebuilt - actual| = %.2e (exact, as claimed)", gap)
    log.info("resid norm %.1f -> %.1f over %d blocks", resid_norms[0], resid_norms[-1], n_layers)
    return resid_norms, attn_writes, mlp_writes, gap


def draw(resid_norms, attn_writes, mlp_writes, gap, log):
    """Drawn at its slot size (the full 5.5 in text width): fonts here are fonts on the slide."""
    fig = plt.figure(figsize=(5.45, 1.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.0], wspace=0.28)

    # ---------------------------------------------------------------- left: schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.add_patch(Rectangle((1.0, 3.6), 8.0, 0.9, facecolor="#dce6f5", edgecolor=BLUE, lw=1.2))
    ax.text(5.0, 4.05, "the residual stream", ha="center", va="center", fontsize=8, color=BLUE)
    ax.annotate("", xy=(0.98, 4.05), xytext=(0.0, 4.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY))
    ax.text(0.0, 4.75, "embedding", fontsize=7.5, color=GREY, ha="left")
    ax.annotate("", xy=(10.0, 4.05), xytext=(9.02, 4.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY))
    ax.text(10.0, 4.75, "unembedding", fontsize=7.5, color=GREY, ha="right")

    for x, name in [(2.2, "attn"), (4.1, "MLP"), (6.0, "attn"), (7.9, "MLP")]:
        ax.add_patch(Rectangle((x - 0.75, 1.55), 1.5, 0.95, facecolor="white", edgecolor="black",
                               lw=0.9))
        ax.text(x, 2.02, name, ha="center", va="center", fontsize=7.5)
        # read down, write back up: two arrows, never an overwrite
        ax.add_patch(FancyArrowPatch((x - 0.38, 3.58), (x - 0.38, 2.52), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.3, color=BLUE))
        ax.add_patch(FancyArrowPatch((x + 0.38, 2.52), (x + 0.38, 3.58), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.3, color=RED))
    ax.text(4.85, 0.95, "blue: reads a copy", fontsize=7, color=BLUE, ha="right")
    ax.text(5.15, 0.95, "red: ADDS its output", fontsize=7, color=RED, ha="left")
    ax.text(5.0, 0.0, "nothing is ever overwritten", fontsize=7.5, style="italic", ha="center")

    # ---------------------------------------------------------------- right: measured
    ax2 = fig.add_subplot(gs[0, 1])
    n = len(attn_writes)
    xs = np.arange(n + 1)
    ax2.plot(xs, resid_norms, color=BLUE, lw=1.8, marker="o", ms=2.8, label="size of the stream")
    w = 0.38
    ax2.bar(np.arange(n) + 1 - w / 2, attn_writes, width=w, color=RED, alpha=0.85,
            label="attention wrote")
    ax2.bar(np.arange(n) + 1 + w / 2, mlp_writes, width=w, color=ORANGE, alpha=0.9,
            label="MLP wrote")
    # x index k is the stream after layer k-1; label the embedding and every odd layer so 13
    # ticks fit in 2.5 in without touching.
    tick_pos = [0] + list(range(2, n + 1, 2))
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(["emb"] + [str(k - 1) for k in tick_pos[1:]], fontsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_xlabel("after layer", fontsize=8)
    ax2.set_ylabel("vector norm", fontsize=8)
    ax2.legend(fontsize=7, loc="upper left", frameon=False, handlelength=1.4)

    out = FIG_DIR / "residual_stream.pdf"
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    log.info("saved %s (reconstruction error %.1e)", out.name, gap)


def load_model(log):
    from ioi_core import load_model as _load
    return _load(log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("l45_residual_stream")
    if not args.plot_only:
        model = load_model(log)
        resid_norms, attn_writes, mlp_writes, gap = measure(model, log)
        save_results("residual_stream", {
            "resid_norms": resid_norms,
            "attn_writes": attn_writes,
            "mlp_writes": mlp_writes,
            "reconstruction_max_abs_error": gap,
        }, log)
    r = load_results("residual_stream")
    draw(r["resid_norms"], r["attn_writes"], r["mlp_writes"], r["reconstruction_max_abs_error"], log)


if __name__ == "__main__":
    main()
