"""Easier-example figures for the revised "Opening the box" deck (xx_opening_the_box.tex).

Added 2026-09-22 in the ch19 extension (EXTENSION_PLAN.md, task 2). GPT-2 small, CPU, a handful of
forward passes.

  l45_lens_fact.pdf     the logit lens on an easy prompt first ("Steve Jobs was the founder of"):
                        P(" Apple") and the top token after every layer.
  l45_probe_picture.pdf what "linearly readable" looks like: held-out prompts projected onto the
                        probe direction at layer 0 (not readable) and layer 3 (readable).

Also measured, for the "how a probe can fool you" frame: with 768 dimensions and 128 prompts, a
probe trained on SHUFFLED labels scores well above chance on its own training prompts (79.7%,
held back from 100% only by sklearn's default L2 penalty) and below chance on held-out ones
(43.8%, 5-fold CV).

Results -> results/l45_easy_figs.json; the figures are drawn from that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_easy_figs.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/l45_easy_figs.py --plot-only
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import (BLUE, CHAPTER_DIR, FIG_DIR, GREY, RED, SEED, load_results, save_results,
                       setup_logging)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

FACT_PROMPT = "Steve Jobs was the founder of"
FACT_ANSWER = " Apple"
N_PROBE = 128          # same prompts as the probe_by_layer figure (l45_figs.py)
PROBE_LAYERS = [0, 3]


def select_probe_prompts(model) -> tuple[list[str], np.ndarray]:
    """Same selection as l45_figs.fig_probe_by_layer: the largest equal-length group, first 128."""
    samples = json.loads((CHAPTER_DIR / "data" / "ioi_dataset.json").read_text(encoding="utf-8"))["probe"]
    by_length: dict[int, list[dict]] = {}
    for s in samples:
        by_length.setdefault(len(model.to_tokens(s["text"])[0]), []).append(s)
    group = by_length[max(by_length, key=lambda k: len(by_length[k]))][:N_PROBE]
    labels = np.array([s["has_duplicate"] for s in group], dtype=int)
    if not 0.4 < labels.mean() < 0.6:
        raise RuntimeError(f"probe labels unbalanced ({labels.mean():.2f}); fix the dataset")
    return [s["text"] for s in group], labels


def run(log) -> dict:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from transformer_lens import HookedTransformer

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    out: dict = {}

    # --- logit lens on the fact prompt
    _, cache = model.run_with_cache(FACT_PROMPT)
    resid, labels = cache.accumulated_resid(apply_ln=True, pos_slice=-1, return_labels=True)
    logits = resid[:, 0, :] @ model.W_U + model.b_U          # (13, vocab)
    probs = logits.softmax(-1)
    ans = model.to_single_token(FACT_ANSWER)
    rows = []
    for k, lab in enumerate(labels):
        top = int(probs[k].argmax())
        rows.append({"label": lab, "p_answer": float(probs[k, ans]),
                     "rank_answer": int((probs[k] > probs[k, ans]).sum()) + 1,
                     "top_token": model.tokenizer.decode(top), "top_prob": float(probs[k, top])})
        log.info(f"{lab:>12}: P(Apple)={rows[-1]['p_answer']:.4f} rank {rows[-1]['rank_answer']:>5}  "
                 f"top={rows[-1]['top_token']!r} ({rows[-1]['top_prob']:.3f})")
    final = model(FACT_PROMPT)[0, -1].softmax(-1)
    if abs(float(final[ans]) - rows[-1]["p_answer"]) > 1e-4:
        raise RuntimeError("logit lens at the last layer does not match the real output")
    out["lens_fact"] = {"prompt": FACT_PROMPT, "answer": FACT_ANSWER, "rows": rows}

    # --- probe picture + the shuffled-label control
    texts, y = select_probe_prompts(model)
    _, cache = model.run_with_cache(model.to_tokens(texts))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    pic = {}
    for layer in PROBE_LAYERS:
        x = cache["resid_post", layer][:, -1, :].numpy()
        clf = LogisticRegression(max_iter=4000, random_state=SEED).fit(x[tr], y[tr])
        w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        proj = x[te] @ w
        rest = x[te] - np.outer(proj, w)
        rest = rest - rest.mean(0)
        _, _, vt = np.linalg.svd(rest, full_matrices=False)
        pic[str(layer)] = {"x": proj.tolist(), "y": (rest @ vt[0]).tolist(), "label": y[te].tolist(),
                           "heldout_acc": float(clf.score(x[te], y[te])),
                           "train_acc": float(clf.score(x[tr], y[tr]))}
        log.info(f"layer {layer}: probe train acc {pic[str(layer)]['train_acc']:.3f}, "
                 f"held-out acc {pic[str(layer)]['heldout_acc']:.3f}")
    x3 = cache["resid_post", 3][:, -1, :].numpy()
    y_shuf = rng.permutation(y)
    fit_in = LogisticRegression(max_iter=4000, random_state=SEED).fit(x3, y_shuf).score(x3, y_shuf)
    cv = cross_val_score(LogisticRegression(max_iter=4000, random_state=SEED), x3, y_shuf, cv=5).mean()
    fit_true = LogisticRegression(max_iter=4000, random_state=SEED).fit(x3, y).score(x3, y)
    cv_true = cross_val_score(LogisticRegression(max_iter=4000, random_state=SEED), x3, y, cv=5).mean()
    out["probe_picture"] = pic
    out["shuffled_control_layer3"] = {"n_prompts": int(len(y)), "d_model": int(x3.shape[1]),
                                      "true_in_sample": float(fit_true), "true_cv": float(cv_true),
                                      "shuffled_in_sample": float(fit_in), "shuffled_cv": float(cv)}
    log.info(f"layer 3, TRUE labels: in-sample {fit_true:.3f}, 5-fold CV {cv_true:.3f}")
    log.info(f"layer 3, SHUFFLED labels: in-sample {fit_in:.3f}, 5-fold CV {cv:.3f}")
    return out


def fig_lens_fact(res: dict) -> None:
    rows = res["lens_fact"]["rows"]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(len(rows))
    p = [r["p_answer"] for r in rows]
    bars = ax.bar(x, p, color=[BLUE if r["top_token"] == FACT_ANSWER else GREY for r in rows])
    ax.bar_label(bars, labels=[f"{v:.0%}" if v >= 0.005 else "" for v in p], padding=2, fontsize=9)
    for k, r in enumerate(rows):
        ax.text(k, -0.14, r["top_token"].strip(), ha="center", va="top", fontsize=8.5,
                transform=ax.get_xaxis_transform(),
                color=BLUE if r["top_token"] == FACT_ANSWER else "black")
    ax.text(-0.9, -0.14, "top\nguess:", ha="right", va="top", fontsize=8.5,
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(["0"] + [str(k) for k in range(1, len(rows))], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('P(" Apple")')
    ax.set_xlabel("layers the stream has passed through", labelpad=36)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l45_lens_fact.pdf", bbox_inches="tight"); plt.close(fig)


def fig_probe_picture(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, layer in zip(axes, PROBE_LAYERS):
        d = res["probe_picture"][str(layer)]
        x, y, lab = np.array(d["x"]), np.array(d["y"]), np.array(d["label"])
        ax.scatter(x[lab == 0], y[lab == 0], s=16, color=GREY, label="three different names")
        ax.scatter(x[lab == 1], y[lab == 1], s=16, color=RED, label="a name repeats")
        ax.set_title(f"layer {layer}: {d['heldout_acc']:.0%} of unseen prompts right"
                     + ("\n(chance is 50%)" if d["heldout_acc"] < 0.6 else "\n(trained on 64 prompts)"),
                     fontsize=10.5)
        ax.set_xlabel("position along the probe's direction")
        ax.set_yticks([]); ax.set_xticks([])
    axes[0].set_ylabel("another direction")
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG_DIR / "l45_probe_picture.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("l45_easy_figs")
    if not args.plot_only:
        save_results("l45_easy_figs", run(log), log)
    res = load_results("l45_easy_figs")
    fig_lens_fact(res)
    fig_probe_picture(res)
    log.info("wrote l45_lens_fact / l45_probe_picture")


if __name__ == "__main__":
    main()
