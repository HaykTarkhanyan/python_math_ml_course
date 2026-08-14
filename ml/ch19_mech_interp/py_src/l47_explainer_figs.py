"""Explainer figures for L47 - the deck that had the most abstract content and the fewest pictures.

Three figures:

  sae_mechanism.pdf    what a sparse autoencoder actually does, plus a REAL tiny SAE
                       trained here that recovers features hidden in superposition.
  steering_geometry.pdf  what "add a vector to the residual stream" means, plus a
                       MEASURED sweep on GPT-2 small showing the output move.
  attribution_graph.pdf  a readable redraw of the Dallas -> Texas -> Austin graph
                       (schematic: it depicts a published qualitative finding).

The first two are measured, not asserted. That matters here because the SAE and
steering sections were previously carried entirely by prose.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

from ioi_core import FIG_DIR, SEED, load_model, setup_logging, save_results

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY, GREEN = "#666666", "#008C46"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- SAE
def train_toy_sae(log, n_true=8, d_model=4, d_sae=32, steps=8000):
    """Hide n_true sparse features in d_model dimensions, then recover them with an SAE.

    This is the superposition story from the previous section, run forwards: more
    features than dimensions, each rarely active. If the SAE works, its decoder
    directions should line up with the true feature directions one-for-one.
    """
    # ioi_core.load_model() disables grad globally for inference. This function trains,
    # so it must not depend on being called before that happens.
    torch.set_grad_enabled(True)
    torch.manual_seed(SEED)
    true_dirs = F.normalize(torch.randn(n_true, d_model), dim=1)

    def batch(bs=1024, p=0.06):
        active = (torch.rand(bs, n_true) < p).float()
        mag = active * torch.rand(bs, n_true) * 2.0
        return mag @ true_dirs, mag

    enc = nn.Linear(d_model, d_sae)
    dec = nn.Linear(d_sae, d_model, bias=False)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=3e-3)

    for step in range(steps):
        x, _ = batch()
        z = F.relu(enc(x))
        recon = dec(z)
        loss = F.mse_loss(recon, x) + 1.5e-3 * z.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        learned = F.normalize(dec.weight.T, dim=1)          # d_sae x d_model
        sim = (learned @ true_dirs.T).abs()                  # d_sae x n_true
        best_per_true = sim.max(dim=0).values
        x, _ = batch(4096)
        z = F.relu(enc(x))
        active_per_input = (z > 1e-3).float().sum(dim=1).mean().item()

    log.info("toy SAE: %d features in %d dims -> %d latents", n_true, d_model, d_sae)
    log.info("  best |cos| to each true feature: min %.3f, mean %.3f",
             best_per_true.min().item(), best_per_true.mean().item())
    log.info("  mean active latents per input: %.2f of %d", active_per_input, d_sae)
    if best_per_true.min().item() < 0.8:
        raise RuntimeError(
            f"toy SAE failed to recover every planted feature "
            f"(worst |cos| = {best_per_true.min().item():.3f}). The figure claims recovery, "
            "so it must not ship on a run where recovery did not happen."
        )
    return best_per_true.numpy(), active_per_input, n_true, d_model, d_sae


def fig_sae(log):
    best, active, n_true, d_model, d_sae = train_toy_sae(log)

    fig = plt.figure(figsize=(12.2, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.16)

    # -------- left: the mechanism
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    rng = np.random.default_rng(SEED)

    def strip(x0, y0, w, h, vals, color, edge):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=edge, lw=1.4))
        n = len(vals)
        bw = w / n
        for i, v in enumerate(vals):
            if v <= 0:
                continue
            ax.add_patch(Rectangle((x0 + i * bw, y0), bw, h * min(v, 1.0),
                                   facecolor=color, edgecolor="none"))

    dense_in = rng.uniform(0.25, 1.0, 40)
    sparse = np.zeros(90)
    for i in rng.choice(90, 5, replace=False):
        sparse[i] = rng.uniform(0.55, 1.0)
    dense_out = dense_in + rng.normal(0, 0.05, 40)

    # Labels are kept SHORT and stacked: at slide scale a long one-line label under the
    # middle strip runs into its neighbours and the three read as a single jumbled line.
    strip(0.25, 3.6, 2.1, 1.15, dense_in, BLUE, BLUE)
    ax.text(1.3, 3.25, "residual stream\n768 numbers,\nall busy",
            ha="center", va="top", fontsize=10, color=BLUE)

    strip(3.55, 3.6, 3.6, 1.15, sparse, ORANGE, ORANGE)
    ax.text(5.35, 3.25, "24,576 latents\nalmost all exactly zero",
            ha="center", va="top", fontsize=10, color="#B37A00")

    strip(8.35, 3.6, 1.4, 1.15, dense_out, GREEN, GREEN)
    ax.text(9.05, 3.25, "rebuilt\nvector", ha="center", va="top", fontsize=10, color=GREEN)

    for x0, x1, lab, col in [(2.42, 3.5, "encoder\n+ ReLU", "black"),
                             (7.22, 8.3, "decoder", "black")]:
        ax.add_patch(FancyArrowPatch((x0, 4.17), (x1, 4.17), arrowstyle="-|>",
                                     mutation_scale=16, lw=2.0, color=col))
        ax.text((x0 + x1) / 2, 4.45, lab, ha="center", va="bottom", fontsize=9.5)

    ax.text(5.0, 5.55, "Trade one busy vector for a very wide, very empty one",
            ha="center", fontsize=12.5)
    ax.text(5.0, 1.72, "trained on two things at once:", ha="center", fontsize=10.5, color=GREY)
    ax.text(5.0, 1.12, "rebuild the input   +   keep almost every latent at zero",
            ha="center", fontsize=11.5)
    ax.text(5.0, 0.32,
            "Nothing supervises what the latents mean. Sparsity alone does the work.",
            ha="center", fontsize=10, color=GREY, style="italic")

    # -------- right: a real one, on data where we know the answer
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(np.arange(1, len(best) + 1), best, color=BLUE, alpha=0.9)
    ax2.axhline(1.0, color=GREY, ls=":", lw=1.4)
    ax2.set_ylim(0, 1.14)
    ax2.set_xlabel("planted feature")
    ax2.set_ylabel("|cos| with its best learned latent")
    ax2.set_title(f"A real SAE, trained here: {n_true} features hidden in {d_model} dimensions",
                  fontsize=11.5)
    ax2.text(0.5, -0.30,
             f"Every planted feature recovered (worst {best.min():.2f}). "
             f"Mean {active:.1f} of {d_sae} latents active per input.",
             transform=ax2.transAxes, ha="center", fontsize=9.5, color=GREY)

    out = FIG_DIR / "sae_mechanism.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("saved %s", out.name)
    return {"best_cos_per_feature": best.tolist(), "mean_active_latents": active}


# ----------------------------------------------------------------------- steering
def fig_steering(model, log):
    """Measured steering: a difference-of-means direction, added with increasing strength."""
    torch.manual_seed(SEED)
    layer = 8

    pos_prompts = ["The wedding was", "At the wedding they", "Her wedding day was",
                   "The bride and groom", "The wedding ceremony was"]
    neg_prompts = ["The meeting was", "At the meeting they", "Her working day was",
                   "The manager and staff", "The meeting agenda was"]

    name = f"blocks.{layer}.hook_resid_post"

    def mean_resid(prompts):
        acts = []
        for p in prompts:
            # Cache only the one activation we need; caching every hook point makes
            # this ~10x slower for no reason.
            _, cache = model.run_with_cache(model.to_tokens(p), names_filter=name)
            acts.append(cache[name][0, -1, :])
        return torch.stack(acts).mean(0)

    direction = mean_resid(pos_prompts) - mean_resid(neg_prompts)
    direction = direction / direction.norm()

    # A prompt that actually invites a noun. GPT-2 small will never say " wedding" after
    # "I went to the store and saw", steered or not, so that prompt measures nothing.
    test = "The best part of the day was the"
    tokens = model.to_tokens(test)
    target = model.to_single_token(" wedding")

    _, base_cache = model.run_with_cache(tokens, names_filter=name)
    base_norm = base_cache[name][0, -1, :].norm().item()

    # Steering strength only means anything RELATIVE to the norm of what we add it to.
    # The stream at this layer has norm ~base_norm, so a unit vector is invisible.
    # Sweep alpha as a fraction of that norm instead.
    fracs = np.linspace(0, 1.6, 17)
    alphas = fracs * base_norm
    probs, top_tokens = [], []
    for a in alphas:
        # TransformerLens calls hooks as hook(activation, hook=hook_point), so the second
        # parameter must literally be named `hook`.
        def steer(resid, hook, a=a):
            resid[:, -1, :] = resid[:, -1, :] + a * direction
            return resid
        logits = model.run_with_hooks(tokens, fwd_hooks=[(name, steer)])
        dist = torch.softmax(logits[0, -1, :], dim=-1)
        probs.append(dist[target].item())
        top_tokens.append(model.to_string(dist.argmax().item()))

    peak = max(probs)
    log.info("top token at alpha=0: %r ; at strongest: %r", top_tokens[0], top_tokens[-1])
    log.info("steering: P(' wedding') %.5f -> peak %.5f (alpha up to %.1f x stream norm %.0f)",
             probs[0], peak, fracs[-1], base_norm)
    # A guard on direction alone is worthless here: a rise from 1e-7 to 2e-7 would pass it
    # while the plotted curve showed nothing. Require the effect to be visible.
    # Probabilities here span orders of magnitude, so the honest test is the RATIO plus a
    # floor low enough to be plotted on a log axis but high enough to be a real effect.
    if peak < 0.002 or peak < 50 * probs[0]:
        raise RuntimeError(
            f"steering effect too small to plot honestly: {probs[0]:.6f} -> {peak:.6f} "
            f"({peak / max(probs[0], 1e-12):.0f}x). The figure claims the output moves."
        )

    fig = plt.figure(figsize=(12.2, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.24)

    # -------- left: the geometry
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.4, 2.5); ax.set_ylim(-0.5, 2.0); ax.axis("off")
    r = np.array([1.5, 0.35])
    v = np.array([0.30, 1.15])
    for vec, base, color, lab, off in [
        (r, np.zeros(2), BLUE, r"residual stream $r$", (0.05, -0.22)),
        (v, r, ORANGE, r"$\alpha \cdot$ feature direction", (0.06, 0.02)),
    ]:
        ax.add_patch(FancyArrowPatch(tuple(base), tuple(base + vec), color=color, lw=2.6,
                                     arrowstyle="-|>", mutation_scale=18))
        ax.text(base[0] + vec[0] / 2 + off[0], base[1] + vec[1] / 2 + off[1], lab,
                color=color, fontsize=11)
    ax.add_patch(FancyArrowPatch((0, 0), tuple(r + v), color=GREEN, lw=2.6,
                                 arrowstyle="-|>", mutation_scale=18, linestyle="--"))
    ax.text(r[0] + v[0] + 0.04, r[1] + v[1] + 0.04, r"$r + \alpha v$", color=GREEN, fontsize=11.5)
    ax.set_title("Steering is one addition, at one layer", fontsize=12)
    ax.text(0.5, -0.12, "no weights change - the edit lasts for one forward pass",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=GREY)

    # -------- right: measured
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(fracs, np.array(probs) * 100, color=BLUE, lw=2.6, marker="o", ms=4)
    ax2.set_xlabel("steering strength, as a multiple of the stream's own norm")
    ax2.set_ylabel('P(" wedding")  %   (log scale)')
    ax2.set_title(f'GPT-2 small, layer {layer}:  "{test} ..."', fontsize=11)
    # Lower-right is the only clear region: the curve climbs across the upper-left.
    ax2.text(0.97, 0.20,
             f"{probs[0]*100:.4f}%  $\\rightarrow$  {peak*100:.2f}%"
             f"   ({peak / max(probs[0], 1e-12):.0f}$\\times$)",
             transform=ax2.transAxes, fontsize=11.5, color=BLUE, va="top", ha="right")
    ax2.text(0.97, 0.09, f'top token:  "{top_tokens[0].strip()}"  '
             f'$\\rightarrow$  "{top_tokens[-1].strip()}"',
             transform=ax2.transAxes, fontsize=10.5, color=GREY, va="top", ha="right")

    out = FIG_DIR / "steering_geometry.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("saved %s", out.name)
    return {"layer": layer, "alpha_as_fraction_of_stream_norm": fracs.tolist(),
            "stream_norm": base_norm, "probs": probs, "test_prompt": test,
            "top_tokens": top_tokens}


# --------------------------------------------------------------- attribution graph
def fig_attribution_graph(log):
    """Schematic redraw of a published qualitative finding (Anthropic, 2025)."""
    fig, ax = plt.subplots(figsize=(10.0, 3.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0.1, 5.4)

    nodes = {
        "dallas":  (0.95, 3.85, '"Dallas"', BLUE, "input"),
        "capital": (0.95, 1.35, '"capital"', BLUE, "input"),
        "texas":   (4.05, 3.85, "state: Texas", ORANGE, "feature"),
        "askcap":  (4.05, 1.35, "asking for\na capital", ORANGE, "feature"),
        "capoftx": (6.95, 2.60, "capital of\nTexas", ORANGE, "feature"),
        "austin":  (9.15, 2.60, '"Austin"', GREEN, "output"),
    }
    for x, y, label, color, kind in nodes.values():
        w, h = (1.35, 0.78) if kind != "feature" else (1.65, 0.92)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.06",
                                    facecolor="white", edgecolor=color, lw=2.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, color=color)

    edges = [("dallas", "texas"), ("capital", "askcap"),
             ("texas", "capoftx"), ("askcap", "capoftx"), ("capoftx", "austin")]
    for a, b in edges:
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        ax.add_patch(FancyArrowPatch((xa + 0.80, ya), (xb - 0.85, yb),
                                     arrowstyle="-|>", mutation_scale=16,
                                     lw=2.0, color=GREY,
                                     connectionstyle="arc3,rad=0.08"))

    ax.text(0.95, 5.00, "tokens in", ha="center", fontsize=10, color=GREY, style="italic")
    ax.text(5.10, 5.00, "features the model actually used", ha="center", fontsize=10,
            color=GREY, style="italic")
    ax.text(9.15, 5.00, "token out", ha="center", fontsize=10, color=GREY, style="italic")
    ax.text(5.0, 0.42,
            'Prompt: "Fact: the capital of the state containing Dallas is ..."',
            ha="center", fontsize=10.5)

    out = FIG_DIR / "attribution_graph.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info("saved %s", out.name)


def main():
    log = setup_logging("l47_explainer_figs")
    sae_stats = fig_sae(log)
    fig_attribution_graph(log)
    model = load_model(log)
    steer_stats = fig_steering(model, log)
    save_results("l47_explainers", {"sae": sae_stats, "steering": steer_stats}, log)


if __name__ == "__main__":
    main()
