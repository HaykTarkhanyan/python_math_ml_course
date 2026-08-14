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
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from ioi_core import FIG_DIR, load_model, load_prompts, setup_logging, save_results

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def measure(model, log):
    """Norm of the residual stream after each block, and each block's contribution."""
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
    fig = plt.figure(figsize=(12.4, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)

    # ---------------------------------------------------------------- left: schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Box stops short of both edges so the embedding / unembedding arrows and their
    # labels live outside it and cannot overlap the text inside.
    ax.add_patch(Rectangle((1.25, 3.05), 7.5, 0.62, facecolor="#dce6f5",
                           edgecolor=BLUE, lw=1.6))
    ax.text(5.0, 3.36, "the residual stream:  one vector per token, width 768",
            ha="center", va="center", fontsize=10.5, color=BLUE)

    ax.annotate("", xy=(1.22, 3.36), xytext=(0.30, 3.36),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=GREY))
    ax.text(0.30, 3.80, "embedding", fontsize=9.5, color=GREY, ha="left")
    ax.annotate("", xy=(9.72, 3.36), xytext=(8.80, 3.36),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=GREY))
    ax.text(9.72, 3.80, "unembedding", fontsize=9.5, color=GREY, ha="right")

    for x, name in [(2.1, "attn"), (3.55, "MLP"), (5.0, "attn"), (6.45, "MLP"), (7.9, "attn")]:
        ax.add_patch(Rectangle((x - 0.52, 1.28), 1.04, 0.66, facecolor="white",
                               edgecolor="black", lw=1.2))
        ax.text(x, 1.61, name, ha="center", va="center", fontsize=10)
        # read down, write back up: two arrows, never an overwrite
        ax.add_patch(FancyArrowPatch((x - 0.28, 3.02), (x - 0.28, 1.96),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.9, color=BLUE))
        ax.add_patch(FancyArrowPatch((x + 0.28, 1.96), (x + 0.28, 3.02),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.9, color=RED))

    ax.text(0.30, 0.72, "blue: reads a copy", fontsize=10.5, color=BLUE)
    ax.text(4.55, 0.72, "red: ADDS its output back", fontsize=10.5, color=RED)
    ax.text(5.0, 0.06, "nothing is ever overwritten", fontsize=11, color="black",
            style="italic", ha="center")
    ax.text(5.0, 5.42, "No component talks to any other directly.",
            ha="center", fontsize=12.5)
    ax.text(5.0, 4.82, "The stream is the only channel in the building.",
            ha="center", fontsize=11, color=GREY, style="italic")

    # ---------------------------------------------------------------- right: measured
    ax2 = fig.add_subplot(gs[0, 1])
    n = len(attn_writes)
    xs = np.arange(n + 1)
    ax2.plot(xs, resid_norms, color=BLUE, lw=2.6, marker="o", ms=4.5,
             label="norm of the running vector")
    ax2.set_xlabel("after block")
    ax2.set_ylabel("residual stream norm")
    ax2.set_title("Measured on GPT-2 small, final token", fontsize=12)

    w = 0.38
    ax2.bar(np.arange(n) + 1 - w / 2, attn_writes, width=w, color=RED, alpha=0.85,
            label="what attention wrote")
    ax2.bar(np.arange(n) + 1 + w / 2, mlp_writes, width=w, color=ORANGE, alpha=0.9,
            label="what the MLP wrote")
    ax2.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax2.set_xticks(range(0, n + 1, 2))
    ax2.margins(y=0.14)
    ax2.text(0.5, -0.235,
             f"The final vector is exactly the embedding plus all {2 * n} writes "
             f"(max error {gap:.0e}).",
             transform=ax2.transAxes, ha="center", fontsize=9.5, color=GREY)

    out = FIG_DIR / "residual_stream.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", out.name)


def main():
    log = setup_logging("l45_residual_stream")
    model = load_model(log)
    resid_norms, attn_writes, mlp_writes, gap = measure(model, log)
    save_results("residual_stream", {
        "resid_norms": resid_norms,
        "attn_writes": attn_writes,
        "mlp_writes": mlp_writes,
        "reconstruction_max_abs_error": gap,
    }, log)
    draw(resid_norms, attn_writes, mlp_writes, gap, log)


if __name__ == "__main__":
    main()
