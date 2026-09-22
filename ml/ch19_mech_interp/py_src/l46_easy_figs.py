"""Easier-example measurements for the revised "Does it actually do that?" deck (was L46).

Added 2026-09-22 in the ch19 extension (EXTENSION_PLAN.md, task 3). GPT-2 small, CPU, ~100 short
forward passes.

  l46_easy_patch.pdf   the simplest possible activation patch, before IOI: run "Michael Jordan
                       plays the sport of" and paste in ONE residual-stream vector from "Tom Brady
                       plays the sport of". Grid over layer x position: where does the swap flip
                       basketball into football?

Also measured, for the "why zero is not neutral" frame: how far zero is from every output head
L9H9 actually produces on the IOI prompts.

Results -> results/l46_easy_figs.json; the figure is drawn from that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l46_easy_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l46_easy_figs.py --plot-only
"""

from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import FIG_DIR, SEED, load_results, save_results, setup_logging

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

TEMPLATE = "{} plays the sport of"
CLEAN = ("Michael Jordan", " basketball")     # the run we patch INTO
SOURCE = ("Tom Brady", " football")           # where the pasted vector comes FROM (same length)
HEAD = (9, 9)                                 # a name mover, for the zero-ablation distances


def run(log) -> dict:
    import torch
    from transformer_lens import HookedTransformer
    from ioi_core import load_prompts

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    out: dict = {}

    # --- the easy patch
    clean_p, source_p = TEMPLATE.format(CLEAN[0]), TEMPLATE.format(SOURCE[0])
    clean_t, source_t = model.to_tokens(clean_p), model.to_tokens(source_p)
    if clean_t.shape != source_t.shape:
        raise RuntimeError(f"prompts tokenize to different lengths: {clean_t.shape} vs {source_t.shape}")
    a_clean, a_source = model.to_single_token(CLEAN[1]), model.to_single_token(SOURCE[1])

    def metric(logits) -> float:
        return float(logits[0, -1, a_source] - logits[0, -1, a_clean])

    clean_logits = model(clean_t)
    source_logits, source_cache = model.run_with_cache(source_t)
    m_clean, m_source = metric(clean_logits), metric(source_logits)
    p_clean = clean_logits[0, -1].softmax(-1)
    p_source = source_logits[0, -1].softmax(-1)
    log.info(f"clean  {clean_p!r}: P(basketball)={float(p_clean[a_clean]):.3f} "
             f"P(football)={float(p_clean[a_source]):.3f} metric {m_clean:+.2f}")
    log.info(f"source {source_p!r}: P(basketball)={float(p_source[a_clean]):.3f} "
             f"P(football)={float(p_source[a_source]):.3f} metric {m_source:+.2f}")
    if not (m_clean < 0 < m_source):
        raise RuntimeError("the two prompts do not disagree the way the slide needs")

    n_layers, n_pos = model.cfg.n_layers, clean_t.shape[1]
    frac = np.zeros((n_layers, n_pos))
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(n_pos):
            def hook(act, hook, pos=pos):
                act[:, pos, :] = source_cache[hook.name][:, pos, :]
                return act
            patched = model.run_with_hooks(clean_t, fwd_hooks=[(name, hook)])
            frac[layer, pos] = (metric(patched) - m_clean) / (m_source - m_clean)
    tokens = [model.tokenizer.decode(int(t)) for t in clean_t[0]]
    src_tokens = [model.tokenizer.decode(int(t)) for t in source_t[0]]
    subj_last = max(i for i, t in enumerate(tokens) if t.strip() and t.strip() in CLEAN[0])
    log.info(f"tokens: {tokens}")
    log.info(f"last subject position {subj_last} ({tokens[subj_last]!r}): "
             f"{[round(x, 2) for x in frac[:, subj_last]]}")
    log.info(f"final position: {[round(x, 2) for x in frac[:, -1]]}")
    out["easy_patch"] = {
        "clean_prompt": clean_p, "source_prompt": source_p, "clean_answer": CLEAN[1],
        "source_answer": SOURCE[1], "tokens": tokens, "source_tokens": src_tokens,
        "metric_clean": m_clean, "metric_source": m_source,
        "p_clean": {"basketball": float(p_clean[a_clean]), "football": float(p_clean[a_source])},
        "p_source": {"basketball": float(p_source[a_clean]), "football": float(p_source[a_source])},
        "frac_flipped": frac.tolist(), "subject_last_pos": subj_last,
    }

    # --- zero-ablation distances for one head
    data = load_prompts(model, log, n=64)
    _, cache = model.run_with_cache(data["clean"])
    layer, head = HEAD
    z = cache["z", layer][:, -1, head, :]                        # (n, d_head)
    outs = (z @ model.W_O[layer, head]).numpy()                  # (n, d_model) what it writes
    norms = np.linalg.norm(outs, axis=1)
    mean = outs.mean(0)
    d_mean = np.linalg.norm(outs - mean, axis=1)
    out["zero_distance"] = {
        "head": f"L{layer}H{head}", "n_prompts": int(len(outs)),
        "norm_min": float(norms.min()), "norm_mean": float(norms.mean()), "norm_max": float(norms.max()),
        "dist_to_mean_mean": float(d_mean.mean()), "dist_to_mean_max": float(d_mean.max()),
    }
    log.info(f"{out['zero_distance']}")
    return out


def fig_easy_patch(res: dict) -> None:
    e = res["easy_patch"]
    frac = np.array(e["frac_flipped"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(frac, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(e["tokens"])))
    labels = [t.strip() or "(start)" for t in e["tokens"]]
    labels[0] = "(start)"
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(frac.shape[0]))
    ax.set_ylabel("layer (stream entering it)")
    for (i, j), v in np.ndenumerate(frac):
        if abs(v) >= 0.3:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("share of the switch to football")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l46_easy_patch.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("l46_easy_figs")
    if not args.plot_only:
        save_results("l46_easy_figs", run(log), log)
    fig_easy_patch(load_results("l46_easy_figs"))
    log.info("wrote l46_easy_patch")


if __name__ == "__main__":
    main()
