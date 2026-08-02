"""Deck figures for the L22 sparse-autoencoder section - all REAL outputs of the SAE-on-RNN lab.

Trains the winning configuration from sae_rnn_lab.py's sweep (512 features, l1=0.2) once, then
emits:

  sae_pipeline.pdf        schematic: frozen RNN -> hidden vectors -> SAE (the conceptual reframe)
  sae_neuron_vs_sae.pdf   best neuron vs best SAE feature vs linear probe, per word
  sae_sparsity.pdf        the sparsity/fidelity frontier + where interpretability peaks
  sae_ablation.pdf        causal test: delete the feature, watch the prediction break

Run with the project venv (after sae_rnn_lab.py, which is where the sweep lives):
    ./ma/Scripts/python.exe ml/ch8_autoencoders/py_src/sae_rnn_figs.py
"""
import logging

import numpy as np
import torch

from sae_rnn_lab import (FIG, HIDDEN, N_WORDS, SEED, SEQ, f1_all_concepts, log, make_corpus,
                         make_lexicon, normalize_acts, probe_f1, train_rnn, train_sae)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
WIDTH, L1 = 512, 0.2


def fig_pipeline():
    """The conceptual reframe: an SAE eats another model's hidden vectors, not data."""
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    ax.axis("off")

    def box(x, y, w, h, label, color, sub=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                                   linewidth=1.1, alpha=0.20, zorder=1))
        ax.text(x + w / 2, y + h / 2 + (0.07 if sub else 0), label, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=2)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.13, sub, ha="center", va="center", fontsize=8,
                    style="italic", zorder=2)

    box(0.02, 0.30, 0.20, 0.42, "text in", BLUE, "jvn iixxz oof")
    box(0.28, 0.22, 0.22, 0.58, "trained RNN", ORANGE, "FROZEN - never updated")
    box(0.56, 0.30, 0.16, 0.42, "hidden $h$", RED, "32 numbers")
    box(0.78, 0.22, 0.20, 0.58, "the SAE", BLUE, "512 features")
    for x0, x1 in ((0.22, 0.28), (0.50, 0.56), (0.72, 0.78)):
        ax.annotate("", xy=(x1, 0.51), xytext=(x0, 0.51),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
    ax.text(0.64, 0.11, "every step gives one vector -\nthat pile IS the SAE's training set",
            ha="center", va="center", fontsize=8.5, color=RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.92)
    fig.tight_layout()
    fig.savefig(FIG / "sae_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved sae_pipeline.pdf")


def fig_scores(neuron, sae, probe):
    """Per-word F1: no neuron reads the concept, the dictionary gets much closer, probe is the ceiling."""
    order = np.argsort(-np.array(sae))
    n, s, p = np.array(neuron)[order], np.array(sae)[order], np.array(probe)[order]
    xs = np.arange(len(n))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.5),
                                  gridspec_kw={"width_ratios": [2.4, 1]})
    ax.plot(xs, p, "o-", color=ORANGE, ms=3, lw=1.2, label=f"linear probe (supervised)  mean {p.mean():.2f}")
    ax.plot(xs, s, "o-", color=BLUE, ms=3, lw=1.2, label=f"best SAE feature  mean {s.mean():.2f}")
    ax.plot(xs, n, "o-", color=RED, ms=3, lw=1.2, label=f"best single neuron  mean {n.mean():.2f}")
    ax.set_xlabel("word in the lexicon (sorted by SAE score)")
    ax.set_ylabel("F1 for \"am I inside this word?\"")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    means = [n.mean(), s.mean(), p.mean()]
    bars = ax2.bar(["single\nneuron", "SAE\nfeature", "linear\nprobe"], means,
                   color=[RED, BLUE, ORANGE], alpha=0.85)
    ax2.bar_label(bars, fmt="%.3f", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("mean F1")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("the probe is supervised:\nit is the ceiling, not a rival", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "sae_neuron_vs_sae.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved sae_neuron_vs_sae.pdf")


def fig_sparsity(rows):
    """L0 vs fidelity vs interpretability - the knob does not maximize both at once."""
    l0 = [r["l0"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(l0, [r["ve"] for r in rows], "o-", color=BLUE, label="reconstruction (variance explained)")
    ax.plot(l0, [r["f1"] for r in rows], "s-", color=RED, label="interpretability (mean F1)")
    for r in rows:
        ax.annotate(f"$\\lambda$={r['l1']}", (r["l0"], r["f1"]), textcoords="offset points",
                    xytext=(0, -13), fontsize=7.5, ha="center", color=RED)
    ax.set_xscale("log")
    ax.set_xlabel("$L_0$  (features active at once, log scale)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "sae_sparsity.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved sae_sparsity.pdf")


def fig_ablation(drop_feat, drop_rand, label):
    """The causal test: correlation says the feature marks the word, ablation says it CAUSES it."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(["ablate the\nword feature", "ablate a random\nfeature (control)"],
                  [drop_feat, drop_rand], color=[RED, "#999999"], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=10)
    ax.set_ylabel("drop in P(correct next character)")
    ax.set_title(f"deleting one feature while the RNN spells {label}", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(drop_feat, drop_rand) * 1.35 + 1e-3)
    fig.tight_layout()
    fig.savefig(FIG / "sae_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved sae_ablation.pdf")


def main():
    log.info("=" * 78)
    log.info("building L22 SAE figures (real lab outputs)")
    fig_pipeline()

    lex = make_lexicon(N_WORDS, SEED)
    text, wid, off = make_corpus(lex, 120000, SEED)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_seq = (len(ids) - 1) // SEQ
    x = ids[: n_seq * SEQ].view(n_seq, SEQ)
    y = ids[1 : n_seq * SEQ + 1].view(n_seq, SEQ)
    model, loss = train_rnn(x, y, len(vocab), HIDDEN)
    log.info("RNN loss %.4f (uniform %.4f)", loss, np.log(len(vocab)))

    with torch.no_grad():
        _, h = model(x, return_hidden=True)
    H = h.reshape(-1, HIDDEN).numpy()
    pos = np.arange(n_seq * SEQ)
    w, o = wid[pos], off[pos]
    inword = o >= 0
    H_in, w_in, o_in = H[inword], w[inword], o[inword]
    Xn = torch.tensor(normalize_acts(H_in), dtype=torch.float32)

    rng = np.random.default_rng(SEED)
    score_idx = np.where(o_in >= 2)[0]
    score_idx = rng.choice(score_idx, 15000, replace=False)
    ws = w_in[score_idx]
    concepts = {f"w{k}": (ws == k).astype(np.int8) for k in range(N_WORDS)
                if (ws == k).sum() >= 20}

    neuron = f1_all_concepts(H_in[score_idx], concepts)
    neuron_best = [float(v.max()) for v in neuron.values()]
    probe = [probe_f1(H_in[score_idx], lab) for lab in concepts.values()]

    sae, l0, ve = train_sae(Xn, WIDTH, L1)
    with torch.no_grad():
        Z = sae.encode(Xn).numpy()
    alive = (Z > 0).sum(0) > 0
    f1 = f1_all_concepts(Z[score_idx], concepts)
    sae_best = [float(np.where(alive, v, -1).max()) for v in f1.values()]
    log.info("neuron %.3f | SAE %.3f | probe %.3f  (L0 %.2f, varexp %.3f)",
             np.mean(neuron_best), np.mean(sae_best), np.mean(probe), l0, ve)
    fig_scores(neuron_best, sae_best, probe)

    # ---- sparsity frontier (re-uses the sweep, one width)
    rows = []
    # The low end matters: as l1 -> 0 the code stops being sparse and features go polysemantic
    # again, so interpretability should turn over. Sweep far enough to SEE that, or say it doesn't.
    for l1 in (0.005, 0.02, 0.05, 0.2, 0.5, 1.0, 2.0):
        s, l0_, ve_ = train_sae(Xn, WIDTH, l1)
        with torch.no_grad():
            Z_ = s.encode(Xn).numpy()
        al = (Z_ > 0).sum(0) > 0
        f = f1_all_concepts(Z_[score_idx], concepts)
        rows.append({"l1": l1, "l0": l0_, "ve": ve_,
                     "f1": float(np.mean([np.where(al, v, -1).max() for v in f.values()])),
                     "alive": int(al.sum())})
        log.info("  lambda %.2f -> L0 %6.2f  varexp %.3f  F1 %.3f  alive %d",
                 l1, l0_, ve_, rows[-1]["f1"], rows[-1]["alive"])
    fig_sparsity(rows)

    # ---- causal ablation on the best-identified word
    # The causal test needs a POSITION-SPECIFIC concept. "Somewhere inside word k" is not what
    # decides the next character - "at letter j of word k" is. Selecting on the word-level concept
    # can pick a feature whose ablation barely moves the prediction, which reads as a clean null
    # result rather than a bad choice of feature. (Measured: 0.002 drop that way, 0.09 this way.)
    ws_, os_ = w_in[score_idx], o_in[score_idx]
    pos_con = {}
    for k in range(N_WORDS):
        for j in range(2, len(lex[k]) - 1):        # need another in-word character after it
            m = ((ws_ == k) & (os_ == j)).astype(np.int8)
            if m.sum() >= 25:
                pos_con[f"w{k}@{j}"] = m
    if not pos_con:
        raise SystemExit("no (word, offset) concept has enough support for a causal test")
    pf1 = f1_all_concepts(Z[score_idx], pos_con)
    best_key = max(pos_con, key=lambda kk: np.where(alive, pf1[kk], -1).max())
    feat = int(np.where(alive, pf1[best_key], -1).argmax())
    word_idx, off_j = int(best_key[1:].split("@")[0]), int(best_key.split("@")[1])
    log.info("causal test: letter %d of '%s' via feature #%d (F1 %.3f)", off_j + 1,
             lex[word_idx], feat, float(np.where(alive, pf1[best_key], -1).max()))

    scale = np.sqrt(HIDDEN) / np.linalg.norm(H_in - H_in.mean(0), axis=1).mean()
    mu = torch.tensor(H_in.mean(0), dtype=torch.float32)

    cand = np.where((w == word_idx) & (o == off_j))[0]
    cand = cand[cand + 1 < len(w)][:400]
    if len(cand) == 0:
        raise SystemExit(f"no positions found for {best_key} - cannot ablate")
    nxt = torch.tensor([stoi[text[i + 1]] for i in cand], dtype=torch.long)
    Hc = torch.tensor(H[cand], dtype=torch.float32)

    def p_correct(hmat):
        with torch.no_grad():
            return torch.softmax(model.head(hmat), -1)[torch.arange(len(nxt)), nxt].mean().item()

    with torch.no_grad():
        zc = sae.encode((Hc - mu) * scale)
        base = sae.decode(zc) / scale + mu
        z_abl = zc.clone()
        z_abl[:, feat] = 0.0
        abl = sae.decode(z_abl) / scale + mu
        rand_feat = int(rng.choice(np.where(alive)[0]))
        z_rnd = zc.clone()
        z_rnd[:, rand_feat] = 0.0
        rnd = sae.decode(z_rnd) / scale + mu

    p_base, p_abl, p_rnd = p_correct(base), p_correct(abl), p_correct(rnd)
    log.info("P(correct next char): reconstruction %.3f | ablate feature #%d %.3f | "
             "ablate random #%d %.3f", p_base, feat, p_abl, rand_feat, p_rnd)
    fig_ablation(p_base - p_abl, p_base - p_rnd, f"letter {off_j + 1} of \"{lex[word_idx]}\"")

    # ---- max-activating contexts, for the notebook's feature dashboard
    acts = Z[:, feat]
    top = np.argsort(-acts)[:12]
    log.info("top activating contexts for feature #%d:", feat)
    idx_in = np.where(inword)[0]
    for t in top:
        p = int(idx_in[t])
        log.info("   %8.3f   ...%s[%s]%s...", acts[t], text[max(0, p - 10):p], text[p],
                 text[p + 1:p + 6])


if __name__ == "__main__":
    main()
