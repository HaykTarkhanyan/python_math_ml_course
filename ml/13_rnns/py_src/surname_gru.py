"""Measured figures for L21's language-modeling section: a character GRU trained on the ch11
name inventor's 689 Armenian surnames. Replaces the July illustrative char-LSTM panels
(charlm_anim.py / charlm_generate.py, now in py_src/archive/).

Same data, split, model and training loop as Part 4 of the chapter practical
(xx_rnn_memory_solution.ipynb), so the slides and the practical show the same numbers. The
script asserts that it reproduces the practical's results (GRU best val loss 1.605 at epoch 30,
window MLP 1.673) and stops if it does not.

What it measures, saved to ml/13_rnns/results/surname_gru.json (the artifact of record; the
figures are drawn from it and can be redrawn with --plot-only):
  1. validation loss per epoch of the GRU, plus two references on the same split: the ch11
     window MLP (K = 3) and uniform guessing over the 39 symbols;
  2. the trained GRU's next-character distribution at every step of one validation surname
     (a name it never trained on): the first validation name of length 6 ending in -յան;
  3. ten sampled surnames at plain temperature 1 after epochs 0, 1, 3, 10 and the best epoch.

Figures (fig/): surname_next_<k>.pdf (one per step, a click-through), surname_samples_<k>.pdf
(one per checkpoint), surname_loss.pdf.

CPU only, 2 threads. Runtime: ~55 s measured on this laptop (experiment ~31 s, figures the rest). Run with:
    ./ma/Scripts/python.exe ml/13_rnns/py_src/surname_gru.py
    ./ma/Scripts/python.exe ml/13_rnns/py_src/surname_gru.py --plot-only
"""

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# FAIL LOUDLY: a missing Armenian glyph must crash the script, never render as tofu.
warnings.filterwarnings("error", message=".*missing from current font.*")

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
RES_DIR = CH_DIR / "results"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS = RES_DIR / "surname_gru.json"
DATA = REPO_ROOT / "ml" / "11_neural_networks" / "data" / "surnames_hy.txt"

ARM_FONT = "Segoe UI"
RED, BLUE, ORANGE, GREY = "#D90012", "#0033A0", "#F2A800", "#777777"

# The practical's configuration (Part 4) - change it there first, never only here.
EMB, HIDDEN, LR, BATCH, EPOCHS, CLIP = 16, 64, 3e-3, 64, 45, 1.0
K, MLP_EMB, MLP_H, MLP_EPOCHS, MLP_BATCH = 3, 8, 128, 80, 1024
EXPECT_GRU, EXPECT_GRU_EP, EXPECT_MLP, TOL = 1.605, 30, 1.673, 0.002

SAMPLE_EPOCHS = [0, 1, 3, 10]       # plus the best epoch, added after training
N_SAMPLES, TOP_K = 10, 6


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("surname_gru")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "surname_gru.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------- data (as the practical)
def load_data():
    names = DATA.read_text(encoding="utf-8").split()
    chars = sorted(set("".join(names)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(names))
    n_tr = int(0.8 * len(names))
    tr = [names[i] for i in perm[:n_tr]]
    va = [names[i] for i in perm[n_tr:]]
    return names, stoi, itos, tr, va


def encode_names(batch, stoi):
    L = max(len(n) for n in batch) + 1
    x = torch.zeros(len(batch), L, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, n in enumerate(batch):
        ids = [stoi[c] for c in n]
        x[i, 1:len(ids) + 1] = torch.tensor(ids)
        y[i, :len(ids)] = torch.tensor(ids)
        y[i, len(ids)] = stoi["."]
    return x, y


class CharRNN(nn.Module):
    def __init__(self, nv, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(nv, EMB)
        self.rnn = nn.GRU(EMB, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, nv)

    def forward(self, x, state=None):
        o, state = self.rnn(self.emb(x), state)
        return self.out(o), state


class WindowMLP(nn.Module):
    def __init__(self, nv):
        super().__init__()
        self.emb = nn.Embedding(nv, MLP_EMB)
        self.net = nn.Sequential(nn.Linear(K * MLP_EMB, MLP_H), nn.ReLU(), nn.Linear(MLP_H, nv))

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))


def val_loss(model, names, stoi, nv):
    x, y = encode_names(names, stoi)
    with torch.no_grad():
        logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, nv), y.reshape(-1), ignore_index=-100).item()


def sample_names(model, itos, stoi, n, seed, max_len=20):
    g = torch.Generator().manual_seed(seed)
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            x = torch.zeros(1, 1, dtype=torch.long)
            state, letters = None, []
            for _ in range(max_len):
                logits, state = model(x, state)
                probs = F.softmax(logits[0, -1], dim=-1)          # temperature 1: plain sampling
                nxt = torch.multinomial(probs, 1, generator=g).item()
                if nxt == stoi["."]:
                    break
                letters.append(itos[nxt])
                x = torch.tensor([[nxt]])
            out.append("".join(letters))
    return out


# ---------------------------------------------------------------- experiment
def run(log) -> dict:
    t0 = time.perf_counter()
    torch.set_num_threads(2)
    names, stoi, itos, tr, va = load_data()
    nv = len(stoi)
    log.info(f"{len(names)} surnames, vocabulary {nv}, train {len(tr)}, val {len(va)}")
    tr_set = set(tr)

    model = CharRNN(nv)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    curve = [val_loss(model, va, stoi, nv)]                  # epoch 0 = untrained
    samples = {}

    def record_samples(ep, mdl):
        s = sample_names(mdl, itos, stoi, N_SAMPLES, seed=SEED + ep)
        samples[ep] = [{"name": n, "in_training_set": n in tr_set} for n in s]

    record_samples(0, model)
    best, best_ep, best_state = float("inf"), 0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(tr), generator=g)
        for s in range(0, len(tr), BATCH):
            x, y = encode_names([tr[i] for i in idx[s:s + BATCH]], stoi)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, nv), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        model.eval()
        vl = val_loss(model, va, stoi, nv)
        curve.append(vl)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep in SAMPLE_EPOCHS:
            record_samples(ep, model)
        if ep % 5 == 0:
            el = time.perf_counter() - t0
            log.info(f"epoch {ep}/{EPOCHS}  val {vl:.3f}  {el:.0f}s elapsed, "
                     f"~{el / ep * (EPOCHS - ep):.0f}s left")
    model.load_state_dict(best_state)
    log.info(f"GRU best val loss {best:.4f} at epoch {best_ep}")
    if abs(best - EXPECT_GRU) > TOL or best_ep != EXPECT_GRU_EP:
        raise RuntimeError(f"GRU did not reproduce the practical: {best:.4f} at epoch {best_ep}, "
                           f"expected {EXPECT_GRU} at epoch {EXPECT_GRU_EP}")
    record_samples(best_ep, model)

    # the ch11 window MLP, retrained exactly as in the practical
    torch.manual_seed(SEED)

    def windows(name_list):
        X, Y = [], []
        for n in name_list:
            ctx = [0] * K
            for ch in n + ".":
                X.append(ctx)
                Y.append(stoi[ch])
                ctx = ctx[1:] + [stoi[ch]]
        return torch.tensor(X), torch.tensor(Y)

    Xtr, Ytr = windows(tr)
    Xva, Yva = windows(va)
    mlp = WindowMLP(nv)
    mopt = torch.optim.Adam(mlp.parameters(), lr=LR)
    mg = torch.Generator().manual_seed(SEED)
    mlp_best = float("inf")
    for _ in range(MLP_EPOCHS):
        idx = torch.randperm(len(Xtr), generator=mg)
        for s in range(0, len(Xtr), MLP_BATCH):
            b = idx[s:s + MLP_BATCH]
            loss = F.cross_entropy(mlp(Xtr[b]), Ytr[b])
            mopt.zero_grad()
            loss.backward()
            mopt.step()
        with torch.no_grad():
            mlp_best = min(mlp_best, F.cross_entropy(mlp(Xva), Yva).item())
    log.info(f"window MLP (K={K}) best val loss {mlp_best:.4f}")
    if abs(mlp_best - EXPECT_MLP) > TOL:
        raise RuntimeError(f"MLP did not reproduce the practical: {mlp_best:.4f} vs {EXPECT_MLP}")

    # next-character distributions along one validation surname
    cands = sorted(n for n in va if len(n) == 6 and n.endswith("յան"))
    if not cands:
        raise RuntimeError("no validation surname of length 6 ending in -յան")
    name = cands[0]
    ids = [stoi[c] for c in name]
    x = torch.tensor([[0] + ids])
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits[0], dim=-1)
    steps = []
    targets = ids + [stoi["."]]
    for t in range(len(targets)):
        p = probs[t]
        top = torch.topk(p, TOP_K)
        steps.append({
            "read": "." + name[:t],
            "true_next": itos[targets[t]],
            "p_true": round(float(p[targets[t]]), 4),
            "top": [{"char": itos[int(i)], "p": round(float(v), 4)}
                    for v, i in zip(top.values, top.indices)],
        })
        log.info(f"read '.{name[:t]}' -> true '{itos[targets[t]]}' p={float(p[targets[t]]):.3f}")

    res = {
        "config": {"emb": EMB, "hidden": HIDDEN, "lr": LR, "batch": BATCH, "epochs": EPOCHS,
                   "clip": CLIP, "seed": SEED, "threads": 2, "mlp_window": K},
        "n_names": len(names), "n_train": len(tr), "n_val": len(va), "vocab": nv,
        "val_curve": [round(v, 4) for v in curve],
        "gru_best": round(best, 4), "gru_best_epoch": best_ep,
        "mlp_window_best": round(mlp_best, 4),
        "uniform": round(math.log(nv), 4),
        "example_name": name, "steps": steps,
        "samples": {str(k): v for k, v in samples.items()},
        "torch": torch.__version__,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    RES_DIR.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {RESULTS} ({res['runtime_s']} s)")
    return res


# ---------------------------------------------------------------- figures (from the JSON)
def char_label(c: str) -> str:
    return "end" if c == "." else c


def fig_next(res, log):
    name = res["example_name"]
    for k, st in enumerate(res["steps"]):
        fig, ax = plt.subplots(figsize=(3.6, 1.95))
        top = st["top"]
        labels = [char_label(d["char"]) for d in top]
        vals = [d["p"] for d in top]
        cols = [BLUE if d["char"] == st["true_next"] else GREY for d in top]
        bars = ax.bar(range(len(top)), vals, color=cols)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=1)
        ax.set_xticks(range(len(top)), labels, fontsize=9, fontfamily=ARM_FONT)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("P(next)", fontsize=7.5)
        read = st["read"][1:] if len(st["read"]) > 1 else ""
        shown = f"«{read}»" if read else "nothing yet (start symbol)"
        ax.set_title(f"read so far: {shown}", fontsize=8.5, fontfamily=ARM_FONT)
        if st["true_next"] not in [d["char"] for d in top]:
            ax.text(0.98, 0.95, f"true next «{char_label(st['true_next'])}»: "
                    f"p = {st['p_true']:.2f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7, color=BLUE, fontfamily=ARM_FONT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        out = FIG_DIR / f"surname_next_{k}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info(f"wrote {out}")
    log.info(f"example name: {name}, {len(res['steps'])} steps")


def fig_samples(res, log):
    curve = res["val_curve"]
    eps = sorted(int(e) for e in res["samples"])
    for k, ep in enumerate(eps):
        fig, ax = plt.subplots(figsize=(4.5, 1.55))
        ax.axis("off")
        title = ("untrained (epoch 0)" if ep == 0 else
                 f"after epoch {ep}" + (" - the best epoch" if ep == res["gru_best_epoch"] else ""))
        ax.text(0.0, 1.0, f"{title}   ·   val loss {curve[ep]:.2f} nats/char",
                fontsize=8.5, fontweight="bold", va="top", transform=ax.transAxes)
        items = res["samples"][str(ep)]
        shown = []
        for it in items:
            nm = it["name"] if it["name"] else "(empty)"
            if len(nm) > 10:                      # display only; the JSON keeps the full name
                nm = nm[:9] + "…"
            shown.append(nm + ("*" if it["in_training_set"] else ""))
        lines = ["    ".join(shown[i:i + 4]) for i in range(0, len(shown), 4)]
        ax.text(0.01, 0.66, "\n".join(lines), fontsize=9, va="top", linespacing=1.5,
                transform=ax.transAxes, fontfamily=ARM_FONT, color=BLUE)
        n_copy = sum(it["in_training_set"] for it in items)
        ax.text(0.0, 0.0, f"* = a real surname from the training set ({n_copy} of {len(items)})",
                fontsize=6.5, color=GREY, va="bottom", transform=ax.transAxes)
        out = FIG_DIR / f"surname_samples_{k}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info(f"wrote {out}")


def fig_loss(res, log):
    curve = res["val_curve"]
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    ax.plot(range(len(curve)), curve, color=BLUE, lw=1.4, label="GRU, reads the whole name")
    ax.axhline(res["mlp_window_best"], color=ORANGE, ls="--", lw=1.2,
               label=f"ch11 MLP, last {res['config']['mlp_window']} letters")
    ax.axhline(res["uniform"], color=GREY, ls=":", lw=1.0, label="uniform guessing")
    be, bv = res["gru_best_epoch"], res["gru_best"]
    ax.plot([be], [bv], "o", color=RED, ms=4)
    ax.annotate(f"best {bv:.3f}\n(epoch {be})", (be, bv), xytext=(be - 14, bv + 0.55),
                fontsize=7, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.text(len(curve) - 1, res["mlp_window_best"] + 0.04, f"{res['mlp_window_best']:.3f}",
            fontsize=7, color=ORANGE, ha="right", va="bottom")
    ax.set_xlabel("epoch", fontsize=7.5)
    ax.set_ylabel("val loss (nats / char)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_ylim(1.4, res["uniform"] + 0.15)
    ax.legend(fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.6), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "surname_loss.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    if args.plot_only:
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
    else:
        res = run(log)
    FIG_DIR.mkdir(exist_ok=True)
    fig_next(res, log)
    fig_samples(res, log)
    fig_loss(res, log)


if __name__ == "__main__":
    main()
