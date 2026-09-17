"""Figure for the AdamW frames of xx_optimization (why Adam + L2 is not weight decay).

Generates into ml/11_neural_networks/fig/:
  adam_l2_vs_adamw.pdf -- mean weight over training for two groups of 100 weights that start at
                          1.0 and receive gradients that are PURE NOISE (no signal to learn):
                          std 10 ("large gradients") vs std 0.01 ("small gradients").
                          Left: torch.optim.Adam(weight_decay=0.1)  (L2 added to the gradient).
                          Right: torch.optim.AdamW(weight_decay=0.1) (decoupled decay).
                          Dotted: plain decay (1 - lr * wd)^t for reference.

Run with the project venv:
    ./ma/Scripts/python.exe ml/11_neural_networks/py_src/adamw_decay_fig.py
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 509
RED, BLUE = "#D90012", "#0033A0"
LR, WD, STEPS, N_WEIGHTS = 1e-3, 0.1, 20_000, 100
GRAD_STD = {"large gradients (std 10)": 10.0, "small gradients (std 0.01)": 0.01}

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]


def setup_logging() -> logging.Logger:
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "adamw_decay_fig.log", encoding="utf-8")],
    )
    return logging.getLogger("adamw_decay_fig")


def simulate(opt_name: str, log: logging.Logger) -> dict:
    """Mean weight per step for each gradient-scale group, under one optimizer."""
    torch.manual_seed(SEED)
    params = {name: torch.nn.Parameter(torch.ones(N_WEIGHTS)) for name in GRAD_STD}
    opt_cls = torch.optim.Adam if opt_name == "Adam" else torch.optim.AdamW
    opt = opt_cls(params.values(), lr=LR, weight_decay=WD)
    history = {name: np.empty(STEPS + 1) for name in GRAD_STD}
    for name, p in params.items():
        history[name][0] = p.detach().mean().item()
    for t in range(1, STEPS + 1):
        for name, p in params.items():
            p.grad = torch.randn(N_WEIGHTS) * GRAD_STD[name]   # noise only, no signal
        opt.step()
        for name, p in params.items():
            history[name][t] = p.detach().mean().item()
    for name in GRAD_STD:
        log.info(f"{opt_name}: {name}: mean weight after {STEPS} steps = {history[name][-1]:.3f}")
    return history


def main() -> None:
    log = setup_logging()
    steps = np.arange(STEPS + 1)
    plain = (1 - LR * WD) ** steps
    log.info(f"plain decay reference after {STEPS} steps = {plain[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, opt_name, title in [(axes[0], "Adam", "Adam(weight_decay=0.1): L2 penalty"),
                                (axes[1], "AdamW", "AdamW(weight_decay=0.1)")]:
        hist = simulate(opt_name, log)
        ax.plot(steps, plain, color="gray", ls=":", lw=1.4, label="plain decay")
        for (name, curve), color in zip(hist.items(), [RED, BLUE]):
            ax.plot(steps, curve, color=color, lw=2.0, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean weight", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    out = CH_DIR / "fig" / "adam_l2_vs_adamw.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
