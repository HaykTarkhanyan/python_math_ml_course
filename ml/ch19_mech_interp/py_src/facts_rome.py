"""Gate G6 + figures for the optional deck "Where facts live" (xx_where_facts_live.tex), part 2.

A rank-one fact edit in the style of ROME (Meng et al., 2022), written from scratch for GPT-2 small,
CPU, 2 threads. The MLP output matrix W_out of one layer is treated as a key -> value memory:

    key   k*  = the MLP's hidden activation at the last token of the subject's name
    value     = what W_out writes for that key
    edit      = change W_out so that k* now writes (old value + delta), where delta is found by
                gradient descent to make the model say the new sport, and the change is spread
                along C^-1 k* (C = second moment of keys on ordinary text) so that other keys move
                as little as possible:

        W_out  <-  W_out + (C^-1 k*) delta^T / (k*^T C^-1 k*)

Then the report card: the edited prompt, paraphrases, other athletes, and a related fact.
And the question from Hase et al. (2023): does the layer causal tracing points at (part 1) make the
best place to edit? -> the same edit at every layer, for two facts.

Needs results/facts_tracing.json (run facts_tracing.py first) for the traced layer and the list of
facts GPT-2 small knows. Text for C: the first ~8k tokens of WikiText-2 (train split).

Results -> results/facts_rome.json; figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/facts_rome.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/facts_rome.py --plot-only
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
MAIN_EDIT = ("Michael Jordan", " basketball", " tennis")
SWEEP_EDITS = [MAIN_EDIT, ("Tom Brady", " football", " golf")]
SWEEP_N_OTHERS = 5          # specificity during the sweep is checked on 5 other athletes (CPU budget)
# Context prefixes for the edit. Only prefixes that give prompts of ONE token length are kept, so
# the whole set runs as a single batch (one forward+backward per optimisation step).
PREFIX_CANDIDATES = ["Today, ", "Also, ", "Well, ", "Yes, ", "Again, ", "Now, ", "So, ", "Still, "]
PARAPHRASES = ["{} is a professional", "{} is best known for playing", "The sport {} is famous for is"]
RIPPLE = {"Michael Jordan": [("Michael Jordan played for the Chicago", " Bulls")]}
N_COV_SEQS, COV_SEQ_LEN = 64, 128
STEPS, LR, WEIGHT_DECAY = 25, 0.1, 1e-3
SWEEP_LAYERS = list(range(12))


# --------------------------------------------------------------------------- helpers
def last_subject_pos(model, prompt: str, subject: str) -> int:
    """Index of the subject's last token, found by searching for its token ids inside the prompt.

    Counting the prefix's tokens separately is wrong: "Yesterday, " alone ends in a lone space token
    that merges into " Michael" inside the full prompt.
    """
    toks = model.to_tokens(prompt)[0].tolist()
    for variant in (" " + subject, subject):
        sub = model.to_tokens(variant, prepend_bos=False)[0].tolist()
        for start in range(len(toks) - len(sub) + 1):
            if toks[start:start + len(sub)] == sub:
                return start + len(sub) - 1
    raise RuntimeError(f"subject {subject!r} not found in the tokens of {prompt!r}")


def second_moments(model, log) -> list:
    """C_l = E[k k^T] of the MLP hidden activation (the keys), per layer, on ordinary text."""
    import torch
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(t for t in ds["text"][:4000] if t.strip())
    # truncate=False: to_tokens silently cuts to 1024 tokens otherwise (caught by the check below)
    ids = model.to_tokens(text, prepend_bos=False, truncate=False)[0][: N_COV_SEQS * COV_SEQ_LEN]
    if len(ids) < N_COV_SEQS * COV_SEQ_LEN:
        raise RuntimeError(f"only {len(ids)} tokens of WikiText, need {N_COV_SEQS * COV_SEQ_LEN}")
    seqs = ids.view(N_COV_SEQS, COV_SEQ_LEN)
    d = model.cfg.d_mlp
    C = [torch.zeros(d, d, device=model.cfg.device) for _ in range(model.cfg.n_layers)]
    n = 0
    for batch in torch.split(seqs, 8):
        _, cache = model.run_with_cache(batch, names_filter=lambda nm: nm.endswith("mlp.hook_post"))
        for layer in range(model.cfg.n_layers):
            h = cache["post", layer].reshape(-1, d)
            C[layer] += h.T @ h
        n += batch.numel()
    log.info(f"key second moments from {n} WikiText tokens")
    return [c / n for c in C]


def p_of(model, prompt: str, token: int) -> float:
    return float(model(prompt)[0, -1].softmax(-1)[token])


def compute_edit(model, subject: str, target: str, layer: int, C, log) -> tuple:
    """Return (delta_W, info) for the rank-one edit of `subject` -> `target` at `layer`."""
    import torch
    tgt = model.to_single_token(target)
    prompts = [TEMPLATE.format(subject)] + [p + TEMPLATE.format(subject) for p in PREFIX_CANDIDATES]
    by_len: dict = {}
    for pr in prompts:
        by_len.setdefault(model.to_tokens(pr).shape[1], []).append(pr)
    prompts = max(by_len.values(), key=len)
    if len(prompts) < 3:
        raise RuntimeError(f"fewer than 3 equal-length prompts for {subject!r}: {by_len}")
    positions = {last_subject_pos(model, pr, subject) for pr in prompts}
    if len(positions) != 1:
        raise RuntimeError(f"subject position differs across equal-length prompts: {positions}")
    pos = positions.pop()
    toks = model.to_tokens(prompts)                     # one batch, all the same length

    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=f"blocks.{layer}.mlp.hook_post")
        k_star = cache["post", layer][:, pos].mean(0)

    delta = torch.zeros(model.cfg.d_model, device=model.cfg.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    hook_name = f"blocks.{layer}.hook_mlp_out"

    def add(act, hook):
        act = act.clone()
        act[:, pos, :] = act[:, pos, :] + delta
        return act

    with torch.enable_grad():
        for step in range(STEPS):
            opt.zero_grad()
            logp = model.run_with_hooks(toks, fwd_hooks=[(hook_name, add)])[:, -1].log_softmax(-1)[:, tgt]
            loss = -logp.mean() + WEIGHT_DECAY * delta.pow(2).sum()
            loss.backward()
            opt.step()
            probs = logp.exp().tolist()
            if min(probs) > 0.9:
                break
    with torch.no_grad():
        d = delta.detach()
        Ck = torch.linalg.solve(C[layer] + 1e-4 * torch.eye(C[layer].shape[0], device=C[layer].device) * C[layer].diag().mean(), k_star)
        dW = torch.outer(Ck, d) / (Ck @ k_star)
    return dW, {"steps": step + 1, "delta_norm": float(d.norm()), "opt_min_p": min(probs),
                "n_prompts": len(prompts)}


def report(model, subject: str, orig: str, target: str, others: list) -> dict:
    import torch
    with torch.no_grad():
        o, t = model.to_single_token(orig), model.to_single_token(target)
        main = model(TEMPLATE.format(subject))[0, -1].softmax(-1)
        para = []
        for tpl in PARAPHRASES:
            pr = model(tpl.format(subject))[0, -1].softmax(-1)
            para.append({"prompt": tpl.format(subject), "p_orig": float(pr[o]), "p_target": float(pr[t])})
        spec = []
        for f in others:
            a = model.to_single_token(f["sport"])
            pr = model(TEMPLATE.format(f["name"]))[0, -1].softmax(-1)
            spec.append({"name": f["name"], "p_correct": float(pr[a]), "p_target": float(pr[t]),
                         "still_top1": int(pr.argmax()) == a})
        ripple = [{"prompt": pr_, "answer": ans, "p": p_of(model, pr_, model.to_single_token(ans))}
                  for pr_, ans in RIPPLE.get(subject, [])]
    return {"p_orig": float(main[o]), "p_target": float(main[t]), "paraphrases": para,
            "paraphrase_target_wins": float(np.mean([p["p_target"] > p["p_orig"] for p in para])),
            "others": spec, "others_still_correct": float(np.mean([s["still_top1"] for s in spec])),
            "others_mean_p_target": float(np.mean([s["p_target"] for s in spec])), "ripple": ripple}


def greedy(model, prompt: str, n: int = 14) -> str:
    import torch
    with torch.no_grad():
        return model.generate(prompt, max_new_tokens=n, do_sample=False, verbose=False)


# --------------------------------------------------------------------------- experiment
def run(log) -> dict:
    import torch
    from transformer_lens import HookedTransformer

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tr = load_results("facts_tracing")
    known = [f for f in tr["facts"] if f["known"]]
    edit_layer = tr["tracing"]["mlp_peak_layer_last_name_token"]
    mlp_curve = [row[2] for row in tr["tracing"]["average_recovered_fraction"]["mlp"]]
    log.info(f"edit layer from causal tracing: {edit_layer}")

    dev = pick_device()
    log.info(f"device: {dev}")
    model = HookedTransformer.from_pretrained("gpt2-small", device=dev)
    model.eval()
    for prm in model.parameters():          # only delta is optimised; never track grads for weights
        prm.requires_grad_(False)
    out: dict = {"edit_layer": edit_layer, "tracing_mlp_curve": mlp_curve, "device": dev}

    # ------------------------------------------------------------------ what the MLPs write AT the name
    subject, orig, _ = MAIN_EDIT
    prompt = TEMPLATE.format(subject)
    pos = last_subject_pos(model, prompt, subject)
    bb = model.to_single_token(orig)
    _, cache = model.run_with_cache(prompt)
    readout = []
    for layer in range(model.cfg.n_layers):
        v = cache["mlp_out", layer][0, pos]
        logits = model.ln_final(v[None, None, :])[0, 0] @ model.W_U
        top = logits.topk(8).indices
        readout.append({"layer": layer, "top": [model.tokenizer.decode(int(i)) for i in top],
                        "rank_answer": int((logits > logits[bb]).sum()) + 1})
    best = min(readout, key=lambda r: r["rank_answer"])
    for r in readout:
        log.info(f"  MLP L{r['layer']} writes at {subject.split()[-1]!r}: rank of {orig!r} {r['rank_answer']:>5}, "
                 f"top {r['top'][:6]}")
    # which single neurons, at the name position of the best layer, contributed most to the answer
    layer = best["layer"]
    acts = cache["post", layer][0, pos]
    contrib = acts * (model.W_out[layer] @ model.W_U[:, bb])
    top_n = contrib.topk(5)
    neurons = []
    for c, n in zip(top_n.values, top_n.indices):
        promoted = (model.W_out[layer, int(n)] @ model.W_U).topk(6).indices
        neurons.append({"neuron": int(n), "contribution": float(c), "activation": float(acts[n]),
                        "promotes": [model.tokenizer.decode(int(i)) for i in promoted]})
        log.info(f"  L{layer}N{int(n)} act {float(acts[n]):+.2f} contrib {float(c):+.3f} pushes {neurons[-1]['promotes']}")
    out["value_readout"] = {"subject": subject, "answer": orig, "position": pos, "per_layer": readout,
                            "best_layer": layer, "top_neurons_at_best_layer": neurons,
                            "n_neurons_needed_for_half": int((contrib.sort(descending=True).values.cumsum(0)
                                                              < 0.5 * contrib.clamp(min=0).sum()).sum()) + 1}
    log.info(f"  neurons needed for half of the positive push at L{layer}: "
             f"{out['value_readout']['n_neurons_needed_for_half']}")

    C = second_moments(model, log)

    # ------------------------------------------------------------------ the main edit
    subject, orig, target = MAIN_EDIT
    others = [f for f in known if f["name"] != subject]
    before = report(model, subject, orig, target, others)
    gen_before = greedy(model, f"{subject} is")
    W = model.blocks[edit_layer].mlp.W_out
    dW, info = compute_edit(model, subject, target, edit_layer, C, log)
    with torch.no_grad():
        W += dW
    after = report(model, subject, orig, target, others)
    gen_after = greedy(model, f"{subject} is")
    with torch.no_grad():
        W -= dW
    restored = p_of(model, TEMPLATE.format(subject), model.to_single_token(orig))
    if abs(restored - before["p_orig"]) > 1e-4:
        raise RuntimeError("weights did not restore after the edit")
    log.info(f"MAIN EDIT {subject}: {orig!r} -> {target!r} at layer {edit_layer} ({info})")
    log.info(f"  edit prompt: P(orig) {before['p_orig']:.3f} -> {after['p_orig']:.3f}, "
             f"P(target) {before['p_target']:.3f} -> {after['p_target']:.3f}")
    log.info(f"  paraphrases where target beats original: {before['paraphrase_target_wins']:.2f} -> "
             f"{after['paraphrase_target_wins']:.2f}")
    log.info(f"  other athletes still correct: {before['others_still_correct']:.2f} -> "
             f"{after['others_still_correct']:.2f}; mean P(target) {before['others_mean_p_target']:.4f} -> "
             f"{after['others_mean_p_target']:.4f}")
    log.info(f"  ripple: {before['ripple']} -> {after['ripple']}")
    log.info(f"  generate before: {gen_before!r}")
    log.info(f"  generate after:  {gen_after!r}")
    out["main"] = {"subject": subject, "orig": orig, "target": target, "info": info,
                   "before": before, "after": after, "generate_before": gen_before,
                   "generate_after": gen_after}

    # ------------------------------------------------------------------ the same edit at every layer
    sweep = []
    for subject, orig, target in SWEEP_EDITS:
        others = [f for f in known if f["name"] != subject][:SWEEP_N_OTHERS]
        rows = []
        for layer in SWEEP_LAYERS:
            W = model.blocks[layer].mlp.W_out
            dW, info = compute_edit(model, subject, target, layer, C, log)
            with torch.no_grad():
                W += dW
            r = report(model, subject, orig, target, others)
            with torch.no_grad():
                W -= dW
            rows.append({"layer": layer, "p_target": r["p_target"], "p_orig": r["p_orig"],
                         "paraphrase_target_wins": r["paraphrase_target_wins"],
                         "paraphrase_mean_p_target": float(np.mean([p["p_target"] for p in r["paraphrases"]])),
                         "others_still_correct": r["others_still_correct"], **info})
            log.info(f"  sweep {subject} L{layer}: P(target) {r['p_target']:.2f}, paraphrase wins "
                     f"{r['paraphrase_target_wins']:.2f}, others correct {r['others_still_correct']:.2f}")
        sweep.append({"subject": subject, "orig": orig, "target": target, "rows": rows})
    out["sweep"] = sweep
    gen = np.mean([[r["paraphrase_mean_p_target"] for r in s["rows"]] for s in sweep], axis=0)
    out["sweep_best_generalization_layer"] = int(np.argmax(gen))
    out["corr_tracing_vs_generalization"] = float(np.corrcoef(mlp_curve, gen)[0, 1])
    log.info(f"best layer for paraphrase generalization: {out['sweep_best_generalization_layer']}; traced "
             f"layer {edit_layer}; correlation over layers {out['corr_tracing_vs_generalization']:+.2f}")
    return out


# --------------------------------------------------------------------------- figures
def fig_report(res: dict) -> None:
    m = res["main"]
    b, a = m["before"], m["after"]
    labels = ["the edited\nprompt", "paraphrases\n(mean)", "other athletes\n(mean)"]
    before = [b["p_target"], np.mean([p["p_target"] for p in b["paraphrases"]]), b["others_mean_p_target"]]
    after = [a["p_target"], np.mean([p["p_target"] for p in a["paraphrases"]]), a["others_mean_p_target"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(labels))
    b1 = ax.bar(x - 0.2, before, 0.4, color=GREY, label="before the edit")
    b2 = ax.bar(x + 0.2, after, 0.4, color=RED, label="after the edit")
    ax.bar_label(b1, labels=[f"{v:.1%}" for v in before], fontsize=8.5, padding=2)
    ax.bar_label(b2, labels=[f"{v:.1%}" for v in after], fontsize=8.5, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f'P("{m["target"].strip()}")')
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facts_edit_report.pdf", bbox_inches="tight"); plt.close(fig)


def fig_sweep(res: dict) -> None:
    layers = np.arange(len(res["tracing_mlp_curve"]))
    gen = np.mean([[r["paraphrase_mean_p_target"] for r in s["rows"]] for s in res["sweep"]], axis=0)
    spec = np.mean([[r["others_still_correct"] for r in s["rows"]] for s in res["sweep"]], axis=0)
    # Drawn at the size of its half-column slot on the slide, so the fonts are not scaled down.
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(layers, res["tracing_mlp_curve"], color=ORANGE, alpha=0.6, label="causal tracing: MLP effect")
    ax.bar_label(bars, labels=[f"{v:.2f}" if v >= 0.01 else "" for v in res["tracing_mlp_curve"]],
                 fontsize=8, padding=1)
    ax.plot(layers, gen, "-o", color=RED, lw=2, label="edit reaches paraphrases")
    ax.plot(layers, spec, "--s", color=BLUE, lw=1.5, label="other athletes untouched")
    ax.axvline(res["edit_layer"], color=GREY, ls=":", lw=1)
    ax.set_xticks(layers); ax.set_xlabel("layer edited / traced"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facts_layer_sweep.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("facts_rome")
    if not args.plot_only:
        save_results("facts_rome", run(log), log)
    res = load_results("facts_rome")
    fig_report(res); fig_sweep(res)
    log.info("wrote facts_edit_report / facts_layer_sweep")


if __name__ == "__main__":
    main()
