"""Double descent: test error vs model capacity, with label noise.

Two modes, because the first one tried did NOT reproduce the phenomenon:

  --mode mlp  A one-hidden-layer MLP trained with Adam, sweeping hidden width.
              MEASURED 2026-09-14: test error falls monotonically (72.8% -> 11.9%),
              NO peak at the interpolation threshold. Kept for the record; do not use
              it to claim double descent.

  --mode rf   Random ReLU features (frozen random hidden layer) with a minimum-norm
              least-squares readout - Belkin et al.'s original setting. The
              interpolation threshold sits exactly at n_features = n_train, where the
              linear system becomes square and ill-conditioned, so the spike is sharp
              and reproducible. Closed form: seconds, not minutes.

Both are still "a one-hidden-layer network"; rf just freezes the hidden layer and solves the
output layer exactly instead of descending to it.

    ./ma/Scripts/python.exe ml/11_neural_networks/py_src/double_descent.py --mode rf

Artifacts are TAGGED by mode (CONVENTIONS.md: never overwrite a run's outputs):
    data/double_descent_<mode>.json   <- artifact of record
    fig/double_descent_<mode>.pdf     <- derived from the JSON
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 509
N_TRAIN = 500          # the interpolation threshold lands at n_features = N_TRAIN in rf mode
LABEL_NOISE = 0.15     # without noise the peak is invisible
EPOCHS = 600           # mlp mode only
LR = 1e-2              # mlp mode only

WIDTHS_MLP = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 75, 110, 160, 240, 350, 500]
# Dense around N_TRAIN=500: that is where the spike lives, and a coarse grid steps over it.
WIDTHS_RF = [5, 10, 20, 40, 70, 120, 200, 300, 400, 460, 490, 500, 510, 540, 600,
             700, 900, 1200, 1800, 2600, 4000]

# Armenian flag palette (3+ colours rule, CLAUDE.md).
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

CHAPTER = Path(__file__).resolve().parent.parent
REPO = CHAPTER.parent.parent


def setup_logging() -> logging.Logger:
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "double_descent.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("double_descent")


def build_data(log: logging.Logger):
    """Digits, subsampled and label-noised. Offline: no download."""
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=N_TRAIN, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    rng = np.random.default_rng(SEED)
    n_flip = int(LABEL_NOISE * len(ytr))
    flip_idx = rng.choice(len(ytr), size=n_flip, replace=False)
    ytr = ytr.copy()
    ytr[flip_idx] = rng.integers(0, 10, size=n_flip)
    log.info(
        "train=%d test=%d features=%d classes=10 noised_labels=%d (%.0f%%)",
        len(ytr), len(yte), Xtr.shape[1], n_flip, 100 * LABEL_NOISE,
    )
    return Xtr, ytr, Xte, yte


def run_rf(m: int, Xtr, ytr, Xte, yte) -> tuple[float, float, int]:
    """Frozen random ReLU features + minimum-norm least-squares readout."""
    rng = np.random.default_rng(SEED)
    d = Xtr.shape[1]
    W1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, m))
    b1 = rng.normal(0.0, 0.1, size=m)

    Ptr = np.maximum(0.0, Xtr @ W1 + b1)
    Pte = np.maximum(0.0, Xte @ W1 + b1)

    Y = np.zeros((len(ytr), 10))
    Y[np.arange(len(ytr)), ytr] = 1.0

    # lstsq returns the minimum-norm solution when the system is underdetermined, which is
    # exactly the interpolating estimator whose norm blows up at n_features == n_train.
    W2, *_ = np.linalg.lstsq(Ptr, Y, rcond=None)

    tr_err = float((Ptr @ W2).argmax(1).__ne__(ytr).mean())
    te_err = float((Pte @ W2).argmax(1).__ne__(yte).mean())
    return tr_err, te_err, m * 10


def run_mlp(width: int, Xtr, ytr, Xte, yte) -> tuple[float, float, int]:
    """Trained one-hidden-layer MLP. Did not reproduce the peak; kept for the record."""
    torch.manual_seed(SEED)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    model = nn.Sequential(nn.Linear(Xtr.shape[1], width), nn.ReLU(), nn.Linear(width, 10))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_err = (model(Xtr_t).argmax(1) != ytr_t).float().mean().item()
        te_err = (model(Xte_t).argmax(1) != yte_t).float().mean().item()
    return tr_err, te_err, n_params


def plot(results: list[dict], mode: str, out: Path, log: logging.Logger) -> None:
    xs = [r["capacity"] for r in results]
    tr = [100 * r["train_error"] for r in results]
    te = [100 * r["test_error"] for r in results]
    xlabel = ("random features (log scale)" if mode == "rf"
              else "hidden layer width (log scale)")

    # Sized and typed for a slide: the figure sits in a ~0.56-width column of a 4:3 frame, so
    # a small canvas with large fonts stays legible from the back of a lecture room. No title -
    # the slide states the setup, and the JSON keeps the full config.
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(xs, te, "o-", color=RED, lw=2.4, ms=5, label="test error")
    ax.plot(xs, tr, "s--", color=BLUE, lw=1.8, ms=4, label="train error")
    if mode == "rf":
        ax.axvline(N_TRAIN, color=ORANGE, lw=2.2, ls=":",
                   label=f"threshold (= {N_TRAIN} samples)")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("error (%)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    log.info("wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rf", "mlp"], default="rf")
    args = parser.parse_args()
    log = setup_logging()
    Xtr, ytr, Xte, yte = build_data(log)

    grid = WIDTHS_RF if args.mode == "rf" else WIDTHS_MLP
    runner = run_rf if args.mode == "rf" else run_mlp

    results = []
    t0 = time.perf_counter()
    for cap in grid:
        tr_err, te_err, n_params = runner(cap, Xtr, ytr, Xte, yte)
        results.append({"capacity": cap, "n_params": n_params,
                        "train_error": tr_err, "test_error": te_err})
        log.info("[%s] capacity=%5d params=%7d train_err=%.3f test_err=%.3f",
                 args.mode, cap, n_params, tr_err, te_err)
    log.info("[%s] sweep done in %.1fs", args.mode, time.perf_counter() - t0)

    peak = max(results, key=lambda r: r["test_error"] if r["capacity"] >= 20 else -1)
    log.info("[%s] worst test error past capacity 20: %.3f at capacity %d",
             args.mode, peak["test_error"], peak["capacity"])

    data_dir = CHAPTER / "data"
    data_dir.mkdir(exist_ok=True)
    meta = {"mode": args.mode, "seed": SEED, "n_train": N_TRAIN,
            "label_noise": LABEL_NOISE, "epochs": EPOCHS, "lr": LR, "results": results}
    out_json = data_dir / f"double_descent_{args.mode}.json"
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s", out_json)

    fig_dir = CHAPTER / "fig"
    fig_dir.mkdir(exist_ok=True)
    plot(results, args.mode, fig_dir / f"double_descent_{args.mode}.pdf", log)


if __name__ == "__main__":
    main()
