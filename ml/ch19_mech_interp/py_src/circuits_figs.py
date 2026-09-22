"""Gate G4 + figures for the optional deck "Circuits read off the weights" (xx_transformer_circuits.tex).

Everything here reads WEIGHTS, except the behavioural checks that say which heads are which.
Models: TransformerLens's attention-only toys attn-only-1l / attn-only-2l (8 heads per layer,
width 512, no MLPs) and GPT-2 small. CPU, light: two ~200 MB downloads and matrix algebra.

Measured:
  A. repeated random tokens: the 1-layer model cannot continue them, the 2-layer model can;
  B. per-head previous-token and induction scores (which heads are which, from behaviour);
  C. Q/K/V composition scores between every layer-0 and layer-1 head, vs a random-matrix baseline;
  D. copying, read off the OV circuit: eigenvalue score and "does token t map to itself";
  E. the zero-layer path W_E -> W_U as a bigram table;
  F. the induction head's preference read from weights through the previous-token head
     ("virtual" QK circuit): does it prefer the key whose PREVIOUS token equals the query token?
  G. GPT-2 small: which earlier heads K-compose most with its induction heads?

Results -> results/circuits_figs.json; figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/circuits_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/circuits_figs.py --plot-only
"""

from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import BLUE, FIG_DIR, GREY, ORANGE, RED, SEED, load_results, save_results, setup_logging

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

N_SEQ, REP_LEN = 20, 50
BIGRAM_WORDS = [" Barack", " United", " Hong", " prime", " according", " New", " import", " San"]
N_TOKENS_OV = 2000
N_TOKENS_QK = 300
GPT2_INDUCTION = [(5, 5), (6, 9), (5, 1)]     # found in "Opening the box" (results/l45_figs.json)


def repeated_tokens(model, gen):
    import torch
    rand = torch.randint(1000, 20000, (N_SEQ, REP_LEN), generator=gen)
    bos = torch.full((N_SEQ, 1), model.tokenizer.bos_token_id)
    return torch.cat([bos, rand, rand], dim=1)


def behaviour(model, seq, log) -> dict:
    """Per-position top-1 on repeated tokens, rank of the right token, and per-head scores."""
    import torch
    logits, cache = model.run_with_cache(seq)
    pred = logits[:, :-1].argmax(-1)
    target = seq[:, 1:]
    correct = (pred == target).float()
    rank = (logits[:, :-1] > logits[:, :-1].gather(-1, target[..., None])).sum(-1) + 1
    L = REP_LEN
    out = {"per_position_top1": correct.mean(0).tolist(),
           "top1_first": correct[:, :L].mean().item(), "top1_second": correct[:, L:].mean().item(),
           "median_rank_first": float(rank[:, :L].float().median()),
           "median_rank_second": float(rank[:, L:].float().median()),
           "prev_token_score": [], "induction_score": []}
    for layer in range(model.cfg.n_layers):
        pat = cache["pattern", layer]
        n = seq.shape[1]
        prev = torch.stack([pat[:, :, i, i - 1] for i in range(1, n)], -1).mean((0, 2))
        ind = torch.stack([pat[:, :, i, i - L + 1] for i in range(L + 1, n)], -1).mean((0, 2))
        out["prev_token_score"].append(prev.tolist())
        out["induction_score"].append(ind.tolist())
    log.info(f"  top-1 first copy {out['top1_first']:.3f}, second copy {out['top1_second']:.3f}; "
             f"median rank of the right token {out['median_rank_first']:.0f} -> {out['median_rank_second']:.0f}")
    return out


def copying(model, layer: int, head: int, ids, WUWE) -> dict:
    """OV circuit W_E W_OV W_U: eigenvalue copying score and how often token t maps to itself."""
    import torch
    W_OV = model.OV[layer, head].AB                                   # d_model x d_model
    lam = torch.linalg.eigvals(W_OV @ WUWE)                           # same nonzero eigenvalues as W_E W_OV W_U
    score = float(lam.real.sum() / lam.abs().sum())
    top1 = top10 = 0
    for chunk in torch.split(ids, 250):
        rows = model.W_E[chunk] @ W_OV @ model.W_U                    # (chunk, vocab)
        self_logit = rows.gather(1, chunk[:, None])
        r = (rows > self_logit).sum(1) + 1
        top1 += int((r == 1).sum()); top10 += int((r <= 10).sum())
    return {"eig_score": score, "self_top1": top1 / len(ids), "self_top10": top10 / len(ids)}


def random_baseline(model, n: int = 200) -> float:
    """Composition score between random matrices with the same low-rank shapes (d_model x d_head)."""
    import torch
    g = torch.Generator().manual_seed(SEED)
    d, dh = model.cfg.d_model, model.cfg.d_head
    vals = []
    for _ in range(n):
        A = torch.randn(d, dh, generator=g) @ torch.randn(dh, d, generator=g)      # like W_QK
        B = torch.randn(d, dh, generator=g) @ torch.randn(dh, d, generator=g)      # like W_OV
        vals.append(float((A @ B.T).norm() / (A.norm() * B.norm())))
    return float(np.mean(vals))


def k_composition(model, l0: int, h0: int, l1: int, h1: int) -> float:
    """K-composition of layer-l0 head h0 into layer-l1 head h1, on the factored rank-64 matrices.

    all_composition_scores() builds every pair at once and asks for 4 GB on GPT-2 small, so the
    GPT-2 section computes only the pairs it needs with this. Checked against the library's value
    on attn-only-2l in run().
    """
    qk, ov = model.QK[l1, h1], model.OV[l0, h0]
    return float((qk @ ov.T).norm() / (qk.norm() * ov.norm()))


def run(log) -> dict:
    import torch
    from transformer_lens import HookedTransformer

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(SEED)
    out: dict = {}

    # ------------------------------------------------------------------ 1-layer model
    m1 = HookedTransformer.from_pretrained("attn-only-1l", device="cpu")
    seq1 = repeated_tokens(m1, gen)
    log.info("attn-only-1l:")
    out["behaviour_1l"] = behaviour(m1, seq1, log)
    ids = torch.randint(1000, 30000, (N_TOKENS_OV,), generator=gen)
    WUWE = m1.W_U @ m1.W_E
    out["copying_1l"] = [copying(m1, 0, h, ids, WUWE) for h in range(m1.cfg.n_heads)]
    for h, c in enumerate(out["copying_1l"]):
        log.info(f"  1l head {h}: eig copying score {c['eig_score']:+.2f}, maps t->t top-1 "
                 f"{c['self_top1']:.2f}, top-10 {c['self_top10']:.2f}")
    # zero-layer path: embedding straight to unembedding (through the final LayerNorm)
    bigrams = []
    for w in BIGRAM_WORDS:
        t = m1.to_tokens(w, prepend_bos=False)[0]
        if len(t) != 1:
            log.info(f"  skip bigram word {w!r}: {len(t)} tokens")
            continue
        x = m1.ln_final(m1.W_E[t[0]][None, None, :])[0, 0]      # the model's own final LayerNorm
        probs = (x @ m1.W_U + m1.b_U).softmax(-1)
        top = probs.topk(3)
        bigrams.append({"word": w, "next": [m1.tokenizer.decode(int(i)) for i in top.indices],
                        "prob": top.values.tolist()})
        log.info(f"  direct path {w!r} -> {bigrams[-1]['next']} {[round(p, 3) for p in bigrams[-1]['prob']]}")
    out["bigrams_1l"] = bigrams
    del m1, WUWE

    # ------------------------------------------------------------------ 2-layer model
    m2 = HookedTransformer.from_pretrained("attn-only-2l", device="cpu")
    seq2 = repeated_tokens(m2, gen)
    log.info("attn-only-2l:")
    b2 = behaviour(m2, seq2, log)
    out["behaviour_2l"] = b2
    prev_head = int(np.argmax(b2["prev_token_score"][0]))
    ind_head = int(np.argmax(b2["induction_score"][1]))
    log.info(f"  previous-token head L0H{prev_head} ({b2['prev_token_score'][0][prev_head]:.2f}), "
             f"induction head L1H{ind_head} ({b2['induction_score'][1][ind_head]:.2f})")
    comp = {}
    for mode in ("Q", "K", "V"):
        comp[mode] = m2.all_composition_scores(mode)[0, :, 1, :].tolist()   # layer-0 head x layer-1 head
    base = random_baseline(m2)
    kc = np.array(comp["K"])
    manual = k_composition(m2, 0, prev_head, 1, ind_head)
    if abs(manual - kc[prev_head, ind_head]) > 1e-4:
        raise RuntimeError(f"hand-rolled K-composition {manual:.5f} != library {kc[prev_head, ind_head]:.5f}")
    rank_of_pair = int((kc > kc[prev_head, ind_head]).sum()) + 1
    log.info(f"  K-composition L0H{prev_head}->L1H{ind_head} = {kc[prev_head, ind_head]:.3f}, rank "
             f"{rank_of_pair} of {kc.size}; random baseline {base:.3f}; median {np.median(kc):.3f}")
    WUWE2 = m2.W_U @ m2.W_E
    ids2 = torch.randint(1000, 30000, (N_TOKENS_OV,), generator=gen)
    out["copying_2l_induction"] = copying(m2, 1, ind_head, ids2, WUWE2)
    out["copying_2l_prev"] = copying(m2, 0, prev_head, ids2, WUWE2)
    log.info(f"  OV copying: induction head {out['copying_2l_induction']}, prev-token head {out['copying_2l_prev']}")
    # virtual QK: query = token embedding; key = what the previous-token head wrote there, i.e. the
    # PREVIOUS token's embedding pushed through its OV circuit.
    q_ids = torch.randint(1000, 30000, (N_TOKENS_QK,), generator=gen)
    E = m2.W_E[q_ids]
    keys = E @ m2.OV[0, prev_head].AB
    scores = E @ m2.QK[1, ind_head].AB @ keys.T                   # [query token, previous token at key]
    diag_is_max = float((scores.argmax(1) == torch.arange(N_TOKENS_QK)).float().mean())
    diag_rank = ((scores > scores.diag()[:, None]).sum(1) + 1).float()
    # control: the same with a random layer-0 head instead of the previous-token head
    other = [h for h in range(m2.cfg.n_heads) if h != prev_head]
    ctrl = {}
    for h in other:
        k2 = E @ m2.OV[0, h].AB
        s2 = E @ m2.QK[1, ind_head].AB @ k2.T
        ctrl[h] = float((s2.argmax(1) == torch.arange(N_TOKENS_QK)).float().mean())
    out["virtual_qk"] = {"prev_head": prev_head, "ind_head": ind_head, "n_tokens": N_TOKENS_QK,
                         "diag_is_row_max": diag_is_max, "median_diag_rank": float(diag_rank.median()),
                         "chance": 1 / N_TOKENS_QK, "control_other_layer0_heads": ctrl,
                         "block": scores[:24, :24].tolist(),
                         "block_tokens": [m2.tokenizer.decode(int(i)) for i in q_ids[:24]]}
    log.info(f"  virtual QK through L0H{prev_head}: diagonal is the row max for {diag_is_max:.2%} of "
             f"{N_TOKENS_QK} tokens (chance {1 / N_TOKENS_QK:.2%}); median diagonal rank "
             f"{float(diag_rank.median()):.0f}; other layer-0 heads: { {h: round(v, 3) for h, v in ctrl.items()} }")
    out["composition_2l"] = {"scores": comp, "random_baseline": base, "prev_head": prev_head,
                             "ind_head": ind_head, "pair_rank": rank_of_pair}
    del m2, WUWE2

    # ------------------------------------------------------------------ GPT-2 small
    g2 = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    seqg = repeated_tokens(g2, gen)
    log.info("gpt2-small:")
    bg = behaviour(g2, seqg, log)
    def k_comp(l0, h0, l1, h1) -> float:
        return k_composition(g2, l0, h0, l1, h1)

    gpt2 = []
    for (l1, h1) in GPT2_INDUCTION:
        rows = []
        for l0 in range(l1):
            for h0 in range(g2.cfg.n_heads):
                rows.append({"head": f"L{l0}H{h0}", "k_comp": k_comp(l0, h0, l1, h1),
                             "prev_token_score": bg["prev_token_score"][l0][h0]})
        rows.sort(key=lambda r: -r["k_comp"])
        gpt2.append({"induction_head": f"L{l1}H{h1}",
                     "induction_score": bg["induction_score"][l1][h1], "top_sources": rows[:8],
                     "median_k_comp": float(np.median([r["k_comp"] for r in rows]))})
        log.info(f"  into L{l1}H{h1}: " + ", ".join(f"{r['head']} {r['k_comp']:.3f} (prev {r['prev_token_score']:.2f})"
                                                   for r in rows[:5]))
    prev_heads = sorted(((bg["prev_token_score"][l][h], f"L{l}H{h}") for l in range(12) for h in range(12)),
                        reverse=True)[:5]
    log.info(f"  strongest previous-token heads in GPT-2 small: {prev_heads}")
    out["gpt2"] = {"induction_heads": gpt2, "top_prev_token_heads": prev_heads,
                   "behaviour": {k: bg[k] for k in ("top1_first", "top1_second")},
                   "random_baseline": random_baseline(g2)}
    return out


# --------------------------------------------------------------------------- figures
def fig_repeat(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    L = REP_LEN
    for key, label, color in (("behaviour_1l", "one layer", GREY), ("behaviour_2l", "two layers", BLUE)):
        pp = res[key]["per_position_top1"]
        ax.plot(range(1, len(pp) + 1), pp, color=color, lw=2,
                label=f"{label}: {res[key]['top1_second']:.0%} of the repeat")
    ax.axvspan(0.5, L + 0.5, color=GREY, alpha=0.08)
    ax.text(L / 2, 0.9, "first copy", ha="center", fontsize=10, color=GREY)
    ax.text(1.5 * L, 0.9, "same tokens again", ha="center", fontsize=10, color=BLUE)
    ax.set_ylim(0, 1.05); ax.set_xlim(0.5, 2 * L + 0.5)
    ax.set_xlabel("position"); ax.set_ylabel("next token right")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55), fontsize=9.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_repeat.pdf", bbox_inches="tight"); plt.close(fig)


def fig_head_scores(res: dict) -> None:
    b = res["behaviour_2l"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, vals, title, color in ((axes[0], b["prev_token_score"][0], "layer 0: attends to the previous token", ORANGE),
                                   (axes[1], b["induction_score"][1], "layer 1: induction pattern", BLUE)):
        bars = ax.bar(range(len(vals)), vals, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=8.5, padding=2)
        ax.set_xticks(range(len(vals))); ax.set_xlabel("head")
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_head_scores.pdf", bbox_inches="tight"); plt.close(fig)


def fig_kcomp(res: dict) -> None:
    c = res["composition_2l"]
    k = np.array(c["scores"]["K"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(k, cmap="Blues", vmin=0, vmax=k.max())
    for (i, j), v in np.ndenumerate(k):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.6 * k.max() else "black")
    ax.add_patch(plt.Rectangle((c["ind_head"] - 0.5, c["prev_head"] - 0.5), 1, 1, fill=False,
                               edgecolor=RED, lw=2.2))
    ax.set_xticks(range(k.shape[1])); ax.set_yticks(range(k.shape[0]))
    ax.set_xlabel("layer-1 head (reads)"); ax.set_ylabel("layer-0 head (writes)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_kcomp.pdf", bbox_inches="tight"); plt.close(fig)


def fig_copying(res: dict) -> None:
    cp = res["copying_1l"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(cp))
    vals = [c["eig_score"] for c in cp]
    bars = ax.bar(x, vals, color=[BLUE if v > 0.5 else GREY for v in vals])
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in vals], fontsize=9, padding=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"head {h}" for h in x], fontsize=9.5)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel("copying score\n(eigenvalues)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_copying.pdf", bbox_inches="tight"); plt.close(fig)


def fig_virtual_qk(res: dict) -> None:
    v = res["virtual_qk"]
    blk = np.array(v["block"])
    blk = (blk - blk.mean(1, keepdims=True)) / blk.std(1, keepdims=True)   # row-standardise for display
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.imshow(blk, cmap="Blues")
    labels = [t.strip()[:9] for t in v["block_tokens"]]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("token BEFORE the key position"); ax.set_ylabel("query token")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_virtual_qk.pdf", bbox_inches="tight"); plt.close(fig)


def fig_gpt2(res: dict) -> None:
    g = res["gpt2"]
    fig, axes = plt.subplots(1, len(g["induction_heads"]), figsize=(7.6, 2.9), sharey=True)
    for ax, ih in zip(axes, g["induction_heads"]):
        top = ih["top_sources"][:5]
        vals = [t["k_comp"] for t in top]
        colors = [ORANGE if t["prev_token_score"] > 0.3 else GREY for t in top]
        bars = ax.bar(range(len(top)), vals, color=colors)
        ax.bar_label(bars, labels=[f"{x:.2f}" for x in vals], fontsize=8, padding=2)
        ax.set_xticks(range(len(top))); ax.set_xticklabels([t["head"] for t in top], rotation=45,
                                                          ha="right", fontsize=8.5)
        ax.axhline(ih["median_k_comp"], color=BLUE, ls="--", lw=1)
        ax.set_title(f"into {ih['induction_head']}", fontsize=10.5)
    axes[0].set_ylabel("K-composition")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "circuits_gpt2.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("circuits_figs")
    if not args.plot_only:
        save_results("circuits_figs", run(log), log)
    res = load_results("circuits_figs")
    fig_repeat(res); fig_head_scores(res); fig_kcomp(res); fig_copying(res)
    fig_virtual_qk(res); fig_gpt2(res)
    log.info("wrote circuits_repeat / head_scores / kcomp / copying / virtual_qk / gpt2")


if __name__ == "__main__":
    main()
