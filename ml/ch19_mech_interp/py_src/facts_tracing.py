"""Gate G5 + figures for the optional deck "Where facts live" (xx_where_facts_live.tex), part 1.

GPT-2 small on a GPU when present (committed results: Colab T4, DECISIONS #41), else CPU with 2
threads. Three measurements:

  1. Which athlete -> sport facts does GPT-2 small actually know? ("{name} plays the sport of")
  2. MLP neurons as key -> value memories: which single neurons' output vectors push " basketball"
     up the most (read from W_out and W_U, no text), and do they fire on the Jordan prompt?
  3. Causal tracing (Meng et al., 2022): add noise to the name's embeddings so the model loses the
     fact, then restore the clean value at ONE (layer, position) and measure how much of
     P(correct sport) comes back. Done separately for the residual stream, the MLP outputs and the
     attention outputs, averaged over the known facts.

Results -> results/facts_tracing.json; figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/facts_tracing.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/facts_tracing.py --plot-only
"""

from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import BLUE, FIG_DIR, GREY, ORANGE, RED, SEED, load_results, pick_device, save_results, setup_logging

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

TEMPLATE = "{} plays the sport of"
ATHLETES = [
    ("Michael Jordan", " basketball"), ("LeBron James", " basketball"), ("Kobe Bryant", " basketball"),
    ("Shaquille O'Neal", " basketball"), ("Tiger Woods", " golf"), ("Jack Nicklaus", " golf"),
    ("Roger Federer", " tennis"), ("Rafael Nadal", " tennis"), ("Tom Brady", " football"),
    ("Peyton Manning", " football"), ("Derek Jeter", " baseball"), ("Wayne Gretzky", " hockey"),
    ("Sidney Crosby", " hockey"), ("Serena Williams", " tennis"), ("Babe Ruth", " baseball"),
    ("Lionel Messi", " soccer"), ("Andy Murray", " tennis"), ("Phil Mickelson", " golf"),
]
MIN_PROB = 0.4                  # a fact counts as "known" if the right sport is top-1 with >= 40%
N_NOISE = 3                     # noise samples per fact
# x the std of GPT-2's token embeddings. Meng et al. use 3x on GPT-2 XL; on GPT-2 small 3x also
# wrecks every later word (restoring the name's stream brings back only ~15%), while 2x still
# removes ~95% of P(correct). The 1x/2x/3x comparison is measured below and shown on a slide.
NOISE_SCALE = 2.0
WINDOW = 3                      # MLP / attention restored over 3 adjacent layers, as Meng et al. do
                                # with a window of 10 of GPT-2 XL's 48 layers
CATEGORIES = ["first name token", "other name tokens", "last name token", "plays", "the", "sport", "of"]
KINDS = {"resid": "blocks.{}.hook_resid_pre", "mlp": "blocks.{}.hook_mlp_out", "attn": "blocks.{}.hook_attn_out"}


def run(log) -> dict:
    import torch
    from transformer_lens import HookedTransformer

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    dev = pick_device()
    log.info(f"device: {dev}")
    model = HookedTransformer.from_pretrained("gpt2-small", device=dev)
    model.eval()
    gen = torch.Generator().manual_seed(SEED)
    out: dict = {}

    # ------------------------------------------------------------------ 1. which facts are known
    facts = []
    for name, sport in ATHLETES:
        toks = model.to_tokens(TEMPLATE.format(name))
        probs = model(toks)[0, -1].softmax(-1)
        ans = model.to_single_token(sport)
        top = int(probs.argmax())
        n_name = len(model.to_tokens(name, prepend_bos=False)[0])
        facts.append({"name": name, "sport": sport, "p": float(probs[ans]),
                      "top": model.tokenizer.decode(top), "known": top == ans and float(probs[ans]) >= MIN_PROB,
                      "n_name_tokens": n_name})
        log.info(f"{name:18s} {sport:12s} P={float(probs[ans]):.3f} top={model.tokenizer.decode(top)!r} "
                 f"known={facts[-1]['known']}")
    out["facts"] = facts
    known = [f for f in facts if f["known"]]
    log.info(f"{len(known)}/{len(facts)} facts known")

    # ------------------------------------------------------------------ 2. key -> value neurons
    bball = model.to_single_token(" basketball")
    value_scores = []                                       # per layer: W_out[layer] @ W_U[:, basketball]
    for layer in range(model.cfg.n_layers):
        value_scores.append((model.W_out[layer] @ model.W_U[:, bball]))
    vs = torch.stack(value_scores)                          # (12, 3072)
    flat = vs.flatten().topk(5)
    _, cache = model.run_with_cache(TEMPLATE.format("Michael Jordan"))
    _, cache_other = model.run_with_cache(TEMPLATE.format("Tiger Woods"))
    neurons = []
    for score, idx in zip(flat.values, flat.indices):
        layer, n = divmod(int(idx), model.cfg.d_mlp)
        promoted = (model.W_out[layer, n] @ model.W_U).topk(6).indices
        act_jordan = cache["post", layer][0, -1, n]
        act_woods = cache_other["post", layer][0, -1, n]
        neurons.append({"layer": layer, "neuron": n, "value_dot_basketball": float(score),
                        "promotes": [model.tokenizer.decode(int(i)) for i in promoted],
                        "act_jordan_last": float(act_jordan), "act_woods_last": float(act_woods)})
        log.info(f"L{layer}N{n}: pushes {neurons[-1]['promotes']} | fires {float(act_jordan):+.2f} on Jordan, "
                 f"{float(act_woods):+.2f} on Woods")
    out["basketball_neurons"] = neurons

    # ------------------------------------------------------------------ 3. causal tracing
    emb_std = float(model.W_E.std())
    noise_sd = NOISE_SCALE * emb_std
    log.info(f"noise: {NOISE_SCALE} x embedding std {emb_std:.4f} = {noise_sd:.4f}")
    per_fact = []
    for f in known:
        toks = model.to_tokens(TEMPLATE.format(f["name"]))
        n_pos = toks.shape[1]
        name_pos = list(range(1, 1 + f["n_name_tokens"]))       # position 0 is <|endoftext|>
        ans = model.to_single_token(f["sport"])
        batch = toks.repeat(N_NOISE, 1)
        noise = torch.randn(N_NOISE, len(name_pos), model.cfg.d_model, generator=gen).to(dev) * noise_sd

        def corrupt(act, hook):
            act[:, name_pos, :] = act[:, name_pos, :] + noise
            return act

        _, clean_cache = model.run_with_cache(toks)
        p_clean = float(model(toks)[0, -1].softmax(-1)[ans])
        p_corrupt = float(model.run_with_hooks(batch, fwd_hooks=[("hook_embed", corrupt)])[:, -1]
                          .softmax(-1)[:, ans].mean())
        # One forward pass per (kind, layer): the batch holds every position x every noise sample,
        # and copy number i restores position i // N_NOISE. (One pass per position took ~1 s each,
        # ~4.5 min per fact - stopped and rewritten 2026-09-23.)
        big = toks.repeat(n_pos * N_NOISE, 1)
        big_noise = noise.repeat(n_pos, 1, 1)
        rows = torch.arange(n_pos * N_NOISE, device=dev)
        pos_of_row = rows // N_NOISE

        def corrupt_big(act, hook):
            act[:, name_pos, :] = act[:, name_pos, :] + big_noise
            return act

        def restore(act, hook):
            act[rows, pos_of_row, :] = clean_cache[hook.name][0, pos_of_row, :]
            return act

        grids = {}
        for kind, pattern in KINDS.items():
            grid = np.zeros((model.cfg.n_layers, n_pos))
            for layer in range(model.cfg.n_layers):
                if kind == "resid":
                    window = [layer]
                else:                               # centred window, clipped at the ends
                    window = range(max(0, layer - WINDOW // 2), min(model.cfg.n_layers, layer + WINDOW // 2 + 1))
                hooks = [("hook_embed", corrupt_big)] + [(pattern.format(w), restore) for w in window]
                logits = model.run_with_hooks(big, fwd_hooks=hooks)
                p = logits[:, -1].softmax(-1)[:, ans].view(n_pos, N_NOISE).mean(1)
                grid[layer] = p.cpu().numpy()
            grids[kind] = grid
        # map positions to the categories shared by every prompt
        cat_of = {1: 0, name_pos[-1]: 2, n_pos - 4: 3, n_pos - 3: 4, n_pos - 2: 5, n_pos - 1: 6}
        cat = {}
        for kind, grid in grids.items():
            cols = np.full((model.cfg.n_layers, len(CATEGORIES)), np.nan)
            for c in range(len(CATEGORIES)):
                ps = [p for p in range(1, n_pos) if cat_of.get(p, 1) == c]
                if ps:
                    cols[:, c] = grid[:, ps].mean(1)
            # indirect effect: how much of P(correct) the restore brings back
            cat[kind] = ((cols - p_corrupt) / (p_clean - p_corrupt)).tolist()
        per_fact.append({"name": f["name"], "p_clean": p_clean, "p_corrupt": p_corrupt, "categories": cat})
        log.info(f"traced {f['name']}: P clean {p_clean:.3f}, corrupted {p_corrupt:.3f}")
    # ---- how much the picture depends on the noise size (Jordan only, 10 noise samples per setting)
    toks = model.to_tokens(TEMPLATE.format("Michael Jordan"))
    ans = model.to_single_token(" basketball")
    _, clean_cache = model.run_with_cache(toks)
    p_clean = float(model(toks)[0, -1].softmax(-1)[ans])
    last = toks.shape[1] - 1

    def traced(scale: float, restores: list) -> float:
        noise = torch.randn(10, 2, model.cfg.d_model, generator=gen).to(dev) * scale * emb_std

        def corrupt(act, hook):
            act[:, [1, 2], :] = act[:, [1, 2], :] + noise
            return act

        def make_restore(pos):
            def r(act, hook):
                act[:, pos, :] = clean_cache[hook.name][0, pos, :]
                return act
            return r
        hooks = [("hook_embed", corrupt)] + [(nm, make_restore(pos)) for nm, pos in restores]
        return float(model.run_with_hooks(toks.repeat(10, 1), fwd_hooks=hooks)[:, -1].softmax(-1)[:, ans].mean())

    noise_check = []
    for scale in (1.0, 2.0, 3.0):
        pc = traced(scale, [])
        frac = lambda p: (p - pc) / (p_clean - pc)
        row = {"scale": scale, "p_corrupt": pc,
               "resid_at_name": {L: frac(traced(scale, [(f"blocks.{L}.hook_resid_pre", 2)])) for L in (1, 4, 8)},
               "mlp_0_2_at_name": frac(traced(scale, [(f"blocks.{l}.hook_mlp_out", 2) for l in (0, 1, 2)])),
               "attn_9_11_at_last": frac(traced(scale, [(f"blocks.{l}.hook_attn_out", last) for l in (9, 10, 11)]))}
        noise_check.append(row)
        log.info(f"noise {scale}x: P corrupt {pc:.3f}, stream at name {row['resid_at_name']}, "
                 f"MLP 0-2 at name {row['mlp_0_2_at_name']:.2f}, attn 9-11 at 'of' {row['attn_9_11_at_last']:.2f}")

    avg = {}
    for kind in KINDS:
        stack = np.array([pf["categories"][kind] for pf in per_fact], dtype=float)
        avg[kind] = np.nanmean(stack, axis=0).tolist()
    out["device"] = dev
    out["tracing"] = {"noise_sd": noise_sd, "noise_scale": NOISE_SCALE, "window": WINDOW,
                      "noise_check_jordan": noise_check,
                      "n_noise": N_NOISE, "categories": CATEGORIES,
                      "per_fact": per_fact, "average_recovered_fraction": avg,
                      "mean_p_clean": float(np.mean([pf["p_clean"] for pf in per_fact])),
                      "mean_p_corrupt": float(np.mean([pf["p_corrupt"] for pf in per_fact]))}
    mlp = np.array(avg["mlp"])
    best_layer = int(np.nanargmax(mlp[:, 2]))
    out["tracing"]["mlp_peak_layer_last_name_token"] = best_layer
    log.info(f"average recovered fraction, MLP at last name token by layer: {np.round(mlp[:, 2], 2).tolist()} "
             f"-> peak layer {best_layer}")
    log.info(f"attn at last token by layer: {np.round(np.array(avg['attn'])[:, 6], 2).tolist()}")
    log.info(f"resid at last name token: {np.round(np.array(avg['resid'])[:, 2], 2).tolist()}")
    return out


def fig_tracing(res: dict) -> None:
    t = res["tracing"]
    cats = t["categories"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.4), sharey=True)
    titles = {"resid": "whole stream", "mlp": "MLP output only", "attn": "attention output only"}
    for ax, kind in zip(axes, ("resid", "mlp", "attn")):
        g = np.array(t["average_recovered_fraction"][kind]).T        # categories x layers
        im = ax.imshow(g, cmap="Purples" if kind == "resid" else ("Greens" if kind == "mlp" else "Reds"),
                       vmin=0, vmax=max(0.05, np.nanmax(g)), aspect="auto")
        ax.set_title(titles[kind], fontsize=10.5)
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, g.shape[1], 2))
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facts_tracing.pdf", bbox_inches="tight"); plt.close(fig)


def fig_known(res: dict) -> None:
    f = sorted(res["facts"], key=lambda r: -r["p"])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(f))[::-1]
    bars = ax.barh(y, [r["p"] for r in f], color=[BLUE if r["known"] else GREY for r in f])
    ax.bar_label(bars, labels=[f"{r['p']:.0%}" + ("" if r["known"] else f"  (says {r['top'].strip()})")
                               for r in f], fontsize=8, padding=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['name']} -> {r['sport'].strip()}" for r in f], fontsize=8)
    ax.axvline(MIN_PROB, color=RED, ls="--", lw=1)
    ax.set_xlim(0, 1.05); ax.set_xlabel("P(correct sport)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facts_known.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("facts_tracing")
    if not args.plot_only:
        save_results("facts_tracing", run(log), log)
    res = load_results("facts_tracing")
    fig_tracing(res); fig_known(res)
    log.info("wrote facts_tracing / facts_known")


if __name__ == "__main__":
    main()
