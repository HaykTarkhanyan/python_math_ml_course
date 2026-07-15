"""Real figure for the L20 RNN Foundations deck ("Empirical proof") and the L21 "Road
to Attention" deck ("Does it work? Proof + price").

Generates into ml/ch7_rnn/fig/:
  gradient_flow_vanilla.pdf      -- gradient norm per time-step, vanilla RNN only (L20).
  gradient_flow_vanilla_lstm.pdf -- same, with the LSTM curve added (generated for L21,
                                     NOT overwritten by the code below -- L20's compiled
                                     PDF depends on gradient_flow_vanilla.pdf existing).
  gradient_flow_comparison.pdf   -- L21's actual payoff figure: a QUICK re-run (more
                                     steps than the original 300, but still a
                                     CPU-seconds run, NOT extended retraining -- capped
                                     by instructor scope change, see L21_DECISIONS.md)
                                     to check whether the LSTM's gradient advantage
                                     becomes honestly visible. The frame reports
                                     whatever this run actually shows.

Toy recall task: a length-30 sequence of iid scalars; the target is the value seen at
t=0 (a "recall the first token" task). Loss is MSE between a linear readout of the
FINAL hidden state and that target.

The recurrence is unrolled MANUALLY with nn.RNNCell / nn.LSTMCell (one Python-level
call per time-step) so every intermediate hidden state h_t is its own node in the
autograd graph; retain_grad() on each h_t then gives the true d(loss)/d(h_t) after one
backward pass. (A fused nn.RNN/nn.LSTM call hides the per-step recurrence inside one
kernel -- retaining grad on its stacked output only shows the direct loss-touch point,
not the decay through the recurrent chain, which is the whole point of this figure.)

Tiny net (hidden=16), tiny data, ~300 training steps for the original two figures (CPU
seconds), ~1200 steps for the quick comparison re-run (still CPU seconds -- roughly
4x, not an "extended" run of the kind the instructor ruled out).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/gradient_flow.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 509
HIDDEN = 16
SEQ_LEN = 30
BATCH = 64
TRAIN_STEPS = 300
TRAIN_STEPS_QUICK = 1200  # L21 comparison re-run -- quick, not extended (see docstring)
LR = 1e-2
BLUE, RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l20_gradient_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "gradient_flow.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def make_batch(rng):
    x = torch.randn(BATCH, SEQ_LEN, 1, generator=rng)
    y = x[:, 0, 0].clone()  # recall task: target = the very first token
    return x, y


def run_unrolled(cell, x, cell_type: str, retain=False):
    """Manually unroll `cell` over the sequence. Returns the list of per-step hidden
    states (Tensors), so callers can retain_grad() on each one."""
    batch = x.size(0)
    h = torch.zeros(batch, HIDDEN)
    c = torch.zeros(batch, HIDDEN) if cell_type == "lstm" else None
    hs = []
    for t in range(SEQ_LEN):
        xt = x[:, t, :]
        if cell_type == "lstm":
            h, c = cell(xt, (h, c))
        else:
            h = cell(xt, h)
        if retain:
            h.retain_grad()
        hs.append(h)
    return hs


def train_and_measure(cell_type: str, rng: torch.Generator, log: logging.Logger,
                       train_steps: int = TRAIN_STEPS):
    """Train a tiny recurrent cell on the recall task, then measure the gradient norm
    of the final loss w.r.t. the hidden state at every time-step."""
    if cell_type == "rnn":
        cell = nn.RNNCell(input_size=1, hidden_size=HIDDEN, nonlinearity="tanh")
    elif cell_type == "lstm":
        cell = nn.LSTMCell(input_size=1, hidden_size=HIDDEN)
    else:
        raise ValueError(f"unknown cell_type: {cell_type}")

    readout = nn.Linear(HIDDEN, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(readout.parameters()), lr=LR)

    for step in range(train_steps):
        x, y = make_batch(rng)
        hs = run_unrolled(cell, x, cell_type, retain=False)
        pred = readout(hs[-1]).squeeze(-1)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == train_steps - 1:
            log.info(f"[{cell_type}] step {step}: mse={loss.item():.4f}")

    # one more pass, retaining grad on every per-step hidden state
    x, y = make_batch(rng)
    hs = run_unrolled(cell, x, cell_type, retain=True)
    pred = readout(hs[-1]).squeeze(-1)
    loss = F.mse_loss(pred, y)
    cell.zero_grad(); readout.zero_grad()
    loss.backward()

    norms = []
    for t, h in enumerate(hs):
        if h.grad is None:
            raise RuntimeError(f"[{cell_type}] h.grad is None at step {t} -- "
                                "retain_grad() failed")
        norms.append(h.grad.norm().item())
    return np.array(norms)


def fig_gradient_flow(norms_rnn, norms_lstm, log):
    t = np.arange(SEQ_LEN)

    # variant 1: vanilla-only (L20)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(t, norms_rnn, color=BLUE, lw=2.2, marker="o", markersize=3,
            label="vanilla RNN")
    ax.set_yscale("log")
    ax.set_xlabel("time step (0 = earliest token, 29 = last, where the loss sits)",
                  fontsize=10)
    ax.set_ylabel(r"$\|\partial \mathcal{L} / \partial h_t\|$ (log scale)", fontsize=11)
    ax.set_title("Gradient norm per time-step: dead within ~15 steps", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out1 = FIG_DIR / "gradient_flow_vanilla.pdf"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out1}")

    # variant 2: vanilla + LSTM (L21 payoff)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(t, norms_rnn, color=BLUE, lw=2.2, marker="o", markersize=3,
            label="vanilla RNN")
    ax.plot(t, norms_lstm, color=RED, lw=2.2, marker="s", markersize=3, label="LSTM")
    ax.set_yscale("log")
    ax.set_xlabel("time step (0 = earliest token, 29 = last, where the loss sits)",
                  fontsize=10)
    ax.set_ylabel(r"$\|\partial \mathcal{L} / \partial h_t\|$ (log scale)", fontsize=11)
    ax.set_title("LSTM keeps the gradient alive far longer", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out2 = FIG_DIR / "gradient_flow_vanilla_lstm.pdf"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out2}")


def fig_gradient_comparison(norms_rnn, norms_lstm, log, honest_gap: bool):
    """L21 payoff figure -- gradient_flow_comparison.pdf, from a QUICK re-run at
    TRAIN_STEPS_QUICK steps (not the extended retraining the outline originally asked
    for; the instructor capped this at a CPU-seconds quick run, see L21_DECISIONS.md).
    Reports whatever the run actually shows -- the title/annotation change depending on
    whether the gap turned out honestly dramatic, per the house "report what the data
    shows" rule."""
    t = np.arange(SEQ_LEN)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(t, norms_rnn, color=BLUE, lw=2.2, marker="o", markersize=3,
            label="vanilla RNN")
    ax.plot(t, norms_lstm, color=RED, lw=2.2, marker="s", markersize=3, label="LSTM")
    ax.set_yscale("log")
    ax.set_xlabel("time step (0 = earliest token, 29 = last, where the loss sits)",
                  fontsize=10)
    ax.set_ylabel(r"$\|\partial \mathcal{L} / \partial h_t\|$ (log scale)", fontsize=11)
    title = ("LSTM keeps the gradient alive longer (measured)" if honest_gap else
              "Measured at this training budget: no dramatic LSTM advantage yet")
    ax.set_title(title, fontsize=11.5)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fig.text(0.5, -0.02,
             f"quick re-run, {TRAIN_STEPS_QUICK} steps -- measured, not tuned to flatter",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout()
    out = FIG_DIR / "gradient_flow_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out} (honest_gap={honest_gap})")


def main():
    """NOTE (L21 build): this run intentionally does NOT call fig_gradient_flow() --
    that would regenerate gradient_flow_vanilla.pdf and gradient_flow_vanilla_lstm.pdf,
    which is a hard "do not overwrite" per the L21 brief (L20's compiled PDF depends on
    the former existing unchanged). Both files already exist from the L20 build and are
    left untouched. This run only produces the NEW L21 payoff figure."""
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for must_exist in ["gradient_flow_vanilla.pdf", "gradient_flow_vanilla_lstm.pdf"]:
        if not (FIG_DIR / must_exist).exists():
            raise FileNotFoundError(
                f"{must_exist} is missing from {FIG_DIR} -- this script no longer "
                "regenerates it (hard rule: do not overwrite L20's dependency); run "
                "fig_gradient_flow() manually first if it is genuinely missing."
            )

    # L21 payoff: a QUICK re-run (capped by the instructor, no extended retraining) to
    # see whether the LSTM's gradient advantage becomes honestly visible.
    log.info(f"quick comparison re-run: {TRAIN_STEPS_QUICK} steps (L21, capped -- "
             "no extended retraining per instructor scope change)")
    torch.manual_seed(SEED)
    rng_quick = torch.Generator().manual_seed(SEED)
    log.info("[quick] training vanilla RNN")
    norms_rnn_quick = train_and_measure("rnn", rng_quick, log, train_steps=TRAIN_STEPS_QUICK)
    log.info(f"[quick] vanilla RNN gradient norms per step: "
             f"{np.round(norms_rnn_quick, 8)}")
    log.info("[quick] training LSTM")
    norms_lstm_quick = train_and_measure("lstm", rng_quick, log,
                                          train_steps=TRAIN_STEPS_QUICK)
    log.info(f"[quick] LSTM gradient norms per step: {np.round(norms_lstm_quick, 8)}")

    # "Honest gap": LSTM's early-time-step gradient norm is at least an order of
    # magnitude larger than vanilla's, averaged over the first third of the sequence
    # (where vanishing bites hardest).
    early = slice(0, SEQ_LEN // 3)
    ratio = float(np.mean(norms_lstm_quick[early]) / max(np.mean(norms_rnn_quick[early]),
                                                          1e-12))
    honest_gap = ratio >= 10.0
    log.info(f"[quick] early-step (t<{SEQ_LEN // 3}) mean gradient ratio "
             f"LSTM/vanilla = {ratio:.2f}x -> honest_gap={honest_gap}")
    fig_gradient_comparison(norms_rnn_quick, norms_lstm_quick, log, honest_gap)

    log.info("done")


if __name__ == "__main__":
    main()
