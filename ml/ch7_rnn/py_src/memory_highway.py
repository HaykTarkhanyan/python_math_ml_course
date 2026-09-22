"""Measured figure for L21 "Does it work?": does the LSTM's additive highway actually keep
information alive longer than a vanilla RNN - and what decides it?

Two experiments, results saved to ml/ch7_rnn/results/memory_highway.json (the artifact of
record; the figure is derived from it and can be redrawn with --plot-only):

1. SENSITIVITY AT INITIALISATION (deterministic, no training). For a length-100 sequence of
   random one-hot tokens, how much does the input t steps back still move the final hidden
   state? We measure ||d(u . h_T) / d x_t|| averaged over a batch, for a vanilla tanh RNN and
   for LSTMs whose forget-gate bias starts at 0 (PyTorch's default: gate ~ sigmoid(0) = 0.5),
   1 and 3. Along the LSTM's cell highway d c_t / d c_{t-1} = diag(f_t), so the forget gate
   is a per-step decay factor the network controls - this experiment shows the dial.

2. MEMORY SPAN AFTER TRAINING. First-token recall: token 1 is one of 4 keys, tokens 2..T are
   random distractors from 4 other symbols, and the model must output the key after reading
   all T tokens (chance = 25%). Same budget for every model (up to 1500 Adam steps, batch 64,
   gradient clipping at 1, early stop once a validation batch is >= 99% correct), 2 seeds.

Measured on the first exploratory run (scratchpad, 2026-09-23, seed 509): vanilla RNN solves
T=40 (100%) but is at chance at T=80; LSTM with PyTorch's default init is at chance at T=40;
LSTM with forget bias 3 solves T=40 and reaches 76% at T=80. The deck frame states only what
the JSON produced by THIS script says.

CPU only, torch limited to 2 threads (a 16 GB laptop), results checkpointed after every run
so an interrupted sweep resumes where it stopped.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/memory_highway.py
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/memory_highway.py --plot-only
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
RES_DIR = CH_DIR / "results"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS = RES_DIR / "memory_highway.json"
RED, BLUE, ORANGE, GREY = "#D90012", "#0033A0", "#F2A800", "#777777"

# --- experiment 1 -------------------------------------------------------------------
SENS_T, SENS_H, SENS_D, SENS_B = 100, 32, 8, 64
SENS_MODELS = [("rnn", None), ("lstm", 0.0), ("lstm", 1.0), ("lstm", 3.0)]

# --- experiment 2 -------------------------------------------------------------------
N_KEYS, N_DIST, HIDDEN, BATCH = 4, 4, 32, 64
VOCAB = N_KEYS + N_DIST
MAX_STEPS, EVAL_EVERY, SOLVED = 1500, 100, 0.99
LR, CLIP = 3e-3, 1.0
SPAN_TS = [10, 20, 40, 80]
SPAN_MODELS = [("rnn", None), ("lstm", 0.0), ("lstm", 3.0)]
SPAN_SEEDS = [509, 510]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("memory_highway")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "memory_highway.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def model_name(kind, fb):
    return "rnn" if kind == "rnn" else f"lstm_fb{fb:g}"


def make_recurrent(kind, fb, d_in, hidden, seed):
    """nn.RNN (tanh) or nn.LSTM. For the LSTM, set the TOTAL forget-gate bias to fb
    (PyTorch keeps two bias vectors, gate order i, f, g, o - each gets fb/2)."""
    torch.manual_seed(seed)
    if kind == "rnn":
        return nn.RNN(d_in, hidden, batch_first=True)
    if kind != "lstm":
        raise ValueError(f"unknown kind {kind!r}")
    m = nn.LSTM(d_in, hidden, batch_first=True)
    with torch.no_grad():
        for b in (m.bias_ih_l0, m.bias_hh_l0):
            b[hidden:2 * hidden].fill_(fb / 2)
    return m


def sensitivity(kind, fb):
    """||d(u . h_T)/d x_t|| averaged over the batch, indexed by distance back from the end."""
    rec = make_recurrent(kind, fb, SENS_D, SENS_H, SEED)
    g = torch.Generator().manual_seed(SEED)
    tok = torch.randint(0, SENS_D, (SENS_B, SENS_T), generator=g)
    x = nn.functional.one_hot(tok, SENS_D).float().requires_grad_(True)
    out, _ = rec(x)
    u = torch.randn(SENS_H, generator=g)
    (out[:, -1] @ u).sum().backward()
    per_t = x.grad.norm(dim=2).mean(dim=0)          # (T,), index = position in sequence
    return per_t.flip(0).tolist()                    # index = distance back from the end


class Recall(nn.Module):
    def __init__(self, kind, fb, seed):
        super().__init__()
        self.rec = make_recurrent(kind, fb, VOCAB, HIDDEN, seed)
        self.head = nn.Linear(HIDDEN, N_KEYS)

    def forward(self, x):
        out, _ = self.rec(x)
        return self.head(out[:, -1])


def recall_batch(T, n, g):
    key = torch.randint(0, N_KEYS, (n,), generator=g)
    distract = torch.randint(N_KEYS, VOCAB, (n, T - 1), generator=g)
    seq = torch.cat([key[:, None], distract], dim=1)
    return nn.functional.one_hot(seq, VOCAB).float(), key


def train_span(kind, fb, T, seed):
    torch.manual_seed(seed)
    net = Recall(kind, fb, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    g_val = torch.Generator().manual_seed(seed + 10_000)
    x_val, y_val = recall_batch(T, 1000, g_val)
    steps_used, acc = MAX_STEPS, 0.0
    for step in range(1, MAX_STEPS + 1):
        x, y = recall_batch(T, BATCH, g)
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                acc = (net(x_val).argmax(1) == y_val).float().mean().item()
            if acc >= SOLVED:
                steps_used = step
                break
    g_test = torch.Generator().manual_seed(seed + 20_000)
    x_te, y_te = recall_batch(T, 2000, g_test)
    with torch.no_grad():
        test_acc = (net(x_te).argmax(1) == y_te).float().mean().item()
    return {"model": model_name(kind, fb), "T": T, "seed": seed, "test_acc": test_acc,
            "steps": steps_used, "solved": steps_used < MAX_STEPS or test_acc >= SOLVED}


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return {"config": {}, "sensitivity": {}, "span": []}


def save_results(res):
    RES_DIR.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1), encoding="utf-8")


def run(log):
    torch.set_num_threads(2)
    res = load_results()
    res["config"] = {
        "sensitivity": {"T": SENS_T, "hidden": SENS_H, "vocab": SENS_D, "batch": SENS_B,
                        "seed": SEED},
        "span": {"keys": N_KEYS, "distractors": N_DIST, "hidden": HIDDEN, "batch": BATCH,
                 "max_steps": MAX_STEPS, "eval_every": EVAL_EVERY, "solved": SOLVED,
                 "lr": LR, "clip": CLIP, "Ts": SPAN_TS, "seeds": SPAN_SEEDS},
    }
    for kind, fb in SENS_MODELS:
        name = model_name(kind, fb)
        res["sensitivity"][name] = sensitivity(kind, fb)
        s = res["sensitivity"][name]
        log.info(f"sensitivity {name:<10} d=0 {s[0]:.2e}  d=20 {s[20]:.2e}  "
                 f"d=40 {s[40]:.2e}  d=99 {s[99]:.2e}")
    save_results(res)

    done = {(r["model"], r["T"], r["seed"]) for r in res["span"]}
    for kind, fb in SPAN_MODELS:
        for T in SPAN_TS:
            for seed in SPAN_SEEDS:
                key = (model_name(kind, fb), T, seed)
                if key in done:
                    log.info(f"skip (cached) {key}")
                    continue
                t0 = time.perf_counter()
                r = train_span(kind, fb, T, seed)
                r["seconds"] = round(time.perf_counter() - t0, 1)
                res["span"].append(r)
                save_results(res)                      # checkpoint after every run
                log.info(f"span {r['model']:<10} T={T:<3} seed={seed} "
                         f"acc={r['test_acc']:.3f} steps={r['steps']} ({r['seconds']}s)")
    return res


def plot(res, log):
    FIG_DIR.mkdir(exist_ok=True)
    # sized for a 16:9 slide at ~full width, so fonts stay >= ~6 pt after scaling
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    style = {"rnn": (RED, "vanilla RNN"),
             "lstm_fb0": (GREY, "LSTM, gate half-shut (bias 0)"),
             "lstm_fb1": (ORANGE, "LSTM, bias 1 (Keras default)"),
             "lstm_fb3": (BLUE, "LSTM, gate open (bias 3)")}
    for name, s in res["sensitivity"].items():
        c, lab = style[name]
        s = np.maximum(np.array(s), 1e-20)
        a1.semilogy(np.arange(len(s)), s, color=c, lw=1.8, label=lab)
    a1.set_ylim(1e-20, 10)
    a1.set_xlabel("how many steps back the input was", fontsize=8)
    a1.set_ylabel("effect on the final state (log)", fontsize=8)
    a1.set_title("Before training: how far back\ndoes the net 'hear'?", fontsize=9)
    a1.legend(fontsize=6.5, loc="lower left")
    a1.tick_params(labelsize=7)
    a1.grid(alpha=0.3)

    span = res["span"]
    width = 0.26
    names = [n for n in ("rnn", "lstm_fb0", "lstm_fb3") if any(r["model"] == n for r in span)]
    Ts = sorted({r["T"] for r in span})
    x = np.arange(len(Ts))
    for i, name in enumerate(names):
        c, lab = style[name]
        per_seed = [[r["test_acc"] for r in span if r["model"] == name and r["T"] == T]
                    for T in Ts]
        means = [np.mean(v) for v in per_seed]
        xs = x + (i - 1) * width
        bars = a2.bar(xs, np.array(means) * 100, width, color=c, alpha=0.85,
                      label=lab.split(" (")[0])
        a2.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)
        for xi, v in zip(xs, per_seed):          # every seed as a dot: means hide coin flips
            a2.scatter([xi] * len(v), np.array(v) * 100, s=9, color="k", zorder=3)
    a2.axhline(100 / N_KEYS, color="k", ls="--", lw=1)
    a2.text(x[0] - 0.5, 100 / N_KEYS + 2, "chance", ha="left", fontsize=7)
    a2.set_xticks(x, [f"T={T}" for T in Ts])
    a2.set_ylim(0, 112)
    a2.set_ylabel("test accuracy, %\n(bar = mean, dots = 2 seeds)", fontsize=8)
    a2.set_title("After training: recall the first\ntoken after T steps", fontsize=9)
    a2.tick_params(labelsize=7)
    a2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "memory_highway.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    if args.plot_only:
        if not RESULTS.exists():
            raise FileNotFoundError(f"{RESULTS} missing - run without --plot-only first")
        res = load_results()
    else:
        res = run(log)
    plot(res, log)


if __name__ == "__main__":
    main()
