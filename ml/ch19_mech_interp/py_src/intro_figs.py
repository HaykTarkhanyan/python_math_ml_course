"""Figures for the intro deck (xx_mech_interp_intro.tex), everything except the toy MLP.

Measured on GPT-2 small, CPU:

  intro_three_hooks.pdf      three things the model does that the chapter will explain from the
                             inside: a fact, copying a pattern it has never seen, and who-gave-
                             what-to-whom.
  intro_feature_direction.pdf  "a feature is a direction": male -> female word pairs in GPT-2's
                             token embeddings all move along one shared direction.

Drawn (no measurement, but Python per the house rule because each carries its frame):

  intro_method_map.pdf       every interpretability method in the course placed by what it
                             needs from the model and whether it explains one prediction or the model.

Raw numbers go to results/intro_figs.json first; the measured figures are drawn from it.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/intro_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/intro_figs.py --plot-only
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from mi_common import (BLUE, FIG_DIR, GREEN, GREY, RED, SEED, load_results, save_results,
                       setup_logging)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

FACT_PROMPT = "Steve Jobs was the founder of"
IOI_PROMPT = "Then, Alex and William went to the restaurant. Alex gave a book to"
N_REPEAT_SEQS, REPEAT_LEN = 40, 25

GENDER_PAIRS = [
    (" king", " queen"), (" man", " woman"), (" boy", " girl"), (" he", " she"),
    (" his", " her"), (" father", " mother"), (" brother", " sister"), (" son", " daughter"),
    (" husband", " wife"), (" prince", " princess"), (" actor", " actress"), (" uncle", " aunt"),
    (" Mr", " Mrs"), (" men", " women"), (" boys", " girls"), (" kings", " queens"),
]
ANALOGIES = [(" king", " man", " woman"), (" walked", " walk", " run"), (" cats", " cat", " dog"),
             (" bigger", " big", " small"), (" Paris", " France", " Italy")]


# --------------------------------------------------------------------------- measurement
def single_token(model, s: str) -> int | None:
    t = model.to_tokens(s, prepend_bos=False)[0]
    return int(t.item()) if len(t) == 1 else None


def top_k(model, prompt: str, k: int = 5) -> list[dict]:
    probs = model(prompt)[0, -1].softmax(-1)
    vals, idx = probs.topk(k)
    return [{"token": model.tokenizer.decode(int(i)), "prob": float(v)} for v, i in zip(vals, idx)]


def run(log) -> dict:
    from transformer_lens import HookedTransformer   # slow import; only needed to measure

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    out: dict = {}

    out["fact"] = {"prompt": FACT_PROMPT, "top": top_k(model, FACT_PROMPT)}
    out["ioi"] = {"prompt": IOI_PROMPT, "top": top_k(model, IOI_PROMPT)}
    log.info(f"fact: {out['fact']['top'][:2]}")
    log.info(f"ioi: {out['ioi']['top'][:2]}")

    # Induction: random tokens, repeated once. Nothing in language predicts the first copy; the
    # only way to predict the second copy is to look back at the first.
    gen = torch.Generator().manual_seed(SEED)
    rand = torch.randint(1000, 20000, (N_REPEAT_SEQS, REPEAT_LEN), generator=gen)
    bos = torch.full((N_REPEAT_SEQS, 1), model.tokenizer.bos_token_id)
    seq = torch.cat([bos, rand, rand], dim=1)
    logits = model(seq)
    pred = logits[:, :-1].argmax(-1)
    correct = (pred == seq[:, 1:]).float()          # position t predicts token t+1
    per_pos = correct.mean(0)                        # length 2*REPEAT_LEN
    first = per_pos[:REPEAT_LEN].mean().item()
    second = per_pos[REPEAT_LEN:].mean().item()
    out["induction"] = {"n_seqs": N_REPEAT_SEQS, "repeat_len": REPEAT_LEN,
                        "per_position_top1": per_pos.tolist(),
                        "first_copy_top1": first, "second_copy_top1": second,
                        "example_tokens": [model.tokenizer.decode(int(t)) for t in rand[0, :8]]}
    log.info(f"induction: top-1 accuracy first copy {first:.3f}, second copy {second:.3f}")

    # "A feature is a direction": token-embedding arithmetic and the gender direction.
    we = model.W_E.detach()
    we_n = we / we.norm(dim=-1, keepdim=True)
    analogies = []
    for a, b, c in ANALOGIES:
        ids = [single_token(model, s) for s in (a, b, c)]
        if None in ids:
            raise ValueError(f"analogy word not a single token: {(a, b, c)}")
        v = we[ids[0]] - we[ids[1]] + we[ids[2]]
        sims = we_n @ (v / v.norm())
        ranked = [i for i in sims.topk(12).indices.tolist() if i not in ids][:5]
        analogies.append({"a": a, "b": b, "c": c,
                          "nearest": [{"token": model.tokenizer.decode(i), "cos": float(sims[i])}
                                      for i in ranked]})
        log.info(f"{a} -{b} +{c} -> {[x['token'] for x in analogies[-1]['nearest']]}")
    out["analogies"] = analogies

    pairs = []
    for m_w, f_w in GENDER_PAIRS:
        mi, fi = single_token(model, m_w), single_token(model, f_w)
        if mi is None or fi is None:
            log.info(f"skip pair {m_w}/{f_w}: not single tokens")
            continue
        pairs.append((m_w, f_w, mi, fi))
    diffs = torch.stack([we[fi] - we[mi] for *_, mi, fi in pairs])
    d = diffs.mean(0)
    d = d / d.norm()
    # Leave-one-out: build the direction without a pair, then check that pair still moves the
    # right way along it. Otherwise "all pairs move along d" is circular.
    loo_ok = 0
    for k in range(len(pairs)):
        dk = torch.cat([diffs[:k], diffs[k + 1:]]).mean(0)
        loo_ok += int(((we[pairs[k][3]] - we[pairs[k][2]]) @ dk).item() > 0)
    mids = torch.stack([(we[mi] + we[fi]) / 2 for *_, mi, fi in pairs])
    mids_c = mids - mids.mean(0)
    mids_perp = mids_c - (mids_c @ d)[:, None] * d[None, :]
    _, _, vt = torch.linalg.svd(mids_perp, full_matrices=False)
    e2 = vt[0]
    pts = []
    for m_w, f_w, mi, fi in pairs:
        pts.append({"male": m_w, "female": f_w,
                    "male_xy": [float(we[mi] @ d), float(we[mi] @ e2)],
                    "female_xy": [float(we[fi] @ d), float(we[fi] @ e2)]})
    # Cosine of each pair's difference with the shared direction, and with random directions.
    cos_pairs = (torch.nn.functional.normalize(diffs, dim=-1) @ d).tolist()
    rand_dirs = torch.nn.functional.normalize(torch.randn(1000, we.shape[1], generator=gen), dim=-1)
    cos_rand = (torch.nn.functional.normalize(diffs, dim=-1) @ rand_dirs.T).abs().mean().item()
    out["gender"] = {"pairs": pts, "n_pairs": len(pts), "loo_correct": loo_ok,
                     "cos_with_direction": cos_pairs, "mean_abs_cos_random": cos_rand}
    log.info(f"gender direction: {len(pts)} pairs, leave-one-out {loo_ok}/{len(pts)} move the right "
             f"way; mean cos with direction {np.mean(cos_pairs):.3f} vs random {cos_rand:.3f}")
    return out


# --------------------------------------------------------------------------- figures
def fig_three_hooks(res: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.2, 1]})

    def bars(ax, entry, answer, title):
        toks = [repr(t["token"].strip()) for t in entry["top"]][::-1]
        probs = [t["prob"] for t in entry["top"]][::-1]
        colors = [GREEN if t["token"] == answer else GREY for t in entry["top"]][::-1]
        b = ax.barh(range(len(toks)), probs, color=colors)
        ax.bar_label(b, labels=[f"{p:.0%}" for p in probs], padding=3, fontsize=10)
        ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=10)
        ax.set_xlim(0, 1.05); ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=11)

    bars(axes[0], res["fact"], " Apple", "a fact\n\"...was the founder of\"")
    ind = res["induction"]
    pp = np.array(ind["per_position_top1"])
    L = ind["repeat_len"]
    ax = axes[1]
    ax.plot(range(1, 2 * L + 1), pp, color=BLUE, lw=1.8)
    ax.axvspan(0.5, L + 0.5, color=GREY, alpha=0.08)
    ax.text(L / 2, 0.5, "first\ncopy", ha="center", va="center", fontsize=10, color=GREY)
    ax.text(1.5 * L, 0.5, "same\ntokens\nagain", ha="center", va="center", fontsize=10, color=BLUE)
    ax.set_ylim(0, 1.1); ax.set_xlim(0.5, 2 * L + 0.5)
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("next token guessed right")
    ax.set_title("copying a pattern\nit has never seen", fontsize=11)
    bars(axes[2], res["ioi"], " William", "who gave to whom\n\"...Alex gave a book to\"")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_three_hooks.pdf", bbox_inches="tight"); plt.close(fig)


def fig_feature_direction(res: dict) -> None:
    g = res["gender"]
    pairs = sorted(g["pairs"], key=lambda p: p["male_xy"][0])
    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    for k, p in enumerate(pairs):
        mx, fx = p["male_xy"][0], p["female_xy"][0]
        ax.add_patch(FancyArrowPatch((mx, k), (fx, k), arrowstyle="-|>", mutation_scale=10,
                                     color=GREY, lw=1.0, shrinkA=4, shrinkB=4))
        ax.scatter([mx], [k], color=BLUE, s=24, zorder=3)
        ax.scatter([fx], [k], color=RED, s=24, zorder=3)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{p['male'].strip()} \u2192 {p['female'].strip()}" for p in pairs],
                       fontsize=9.5)
    ax.set_xlabel("position along ONE direction\nin GPT-2's embedding space")
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_feature_direction.pdf", bbox_inches="tight"); plt.close(fig)


def fig_method_map() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cols = ["asks the model\nquestions", "reads its\ngradients", "reads its weights\nand activations"]
    rows = ["one prediction", "the whole model"]
    shade = ["#f3f3f3", "#eaf0fa", "#fbeaea"]
    for c in range(3):
        ax.add_patch(plt.Rectangle((c, 0), 1, 2, color=shade[c], zorder=0))
    ax.axhline(1, color="white", lw=3)
    methods = {
        (0, 0): ["LIME", "SHAP (one row)", "counterfactuals"],
        (0, 1): ["permutation importance", "PDP / ALE", "SHAP summary"],
        (1, 0): ["saliency maps", "integrated gradients", "Grad-CAM"],
        (1, 1): ["(rare)"],
        (2, 0): ["logit lens", "activation patching", "attribution graphs"],
        (2, 1): ["circuits", "probes", "sparse autoencoder features",
                 "linear-model coefficients*"],
    }
    for (c, r), names in methods.items():
        y0 = r + 0.5 + 0.13 * (len(names) - 1) / 2
        for k, n in enumerate(names):
            color = RED if c == 2 else (BLUE if c == 1 else "black")
            ax.text(c + 0.5, y0 - 0.13 * k, n, ha="center", va="center", fontsize=11.5, color=color,
                    style="italic" if n == "(rare)" else "normal")
    for c, lab in enumerate(cols):
        ax.text(c + 0.5, 2.06, lab, ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    for r, lab in enumerate(rows):
        ax.text(-0.04, r + 0.5, lab, ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.5, -0.1, "chapter 5", ha="center", va="top", fontsize=10.5, color=GREY)
    ax.text(1.5, -0.1, "chapter 6 + the vision lecture", ha="center", va="top", fontsize=10.5,
            color=BLUE)
    ax.text(2.5, -0.1, "this chapter", ha="center", va="top", fontsize=10.5, color=RED,
            fontweight="bold")
    ax.set_xlim(-0.75, 3.02); ax.set_ylim(-0.3, 2.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_method_map.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("intro_figs")
    FIG_DIR.mkdir(exist_ok=True)
    if not args.plot_only:
        save_results("intro_figs", run(log), log)
    res = load_results("intro_figs")
    fig_three_hooks(res)
    fig_feature_direction(res)
    fig_method_map()
    log.info("wrote intro_three_hooks / feature_direction / method_map")


if __name__ == "__main__":
    main()
