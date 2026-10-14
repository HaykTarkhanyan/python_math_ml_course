"""Transposed convolution, stamp by stamp - a 5-click animation for the [51] Vision Tasks deck.

Frame "Transposed convolution: the output must be an image": a 2x2 input, a 3x3 kernel,
stride 2. Each input number scales a copy of the kernel ("a stamp"), the copies are placed
2 cells apart, and where they overlap they ADD. The 2x2 map becomes 5x5:
(2 - 1) * stride + kernel = 5. The middle row and column collect two stamps (the centre four)
- the uneven overlap behind the checkerboard watch-out.

  input  [[1, 2],      kernel [[1, 1, 1],
          [3, 1]]              [1, 2, 1],
                               [1, 1, 1]]

The final 5x5 is checked against torch.nn.functional.conv_transpose2d before drawing.

Generates into ml/12_cnn/fig/:
  transposed_anim_0.pdf .. transposed_anim_4.pdf  (empty, then one stamp per click)

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/transposed_anim.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
STAMP_COLORS = [ARM_BLUE, ARM_ORANGE, ARM_RED, "#008c46"]

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

X = np.array([[1, 2], [3, 1]])
K = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
STRIDE = 2
EXPECTED = np.array([[1, 1, 3, 2, 2],
                     [1, 2, 3, 4, 2],
                     [4, 4, 7, 3, 3],
                     [3, 6, 4, 2, 1],
                     [3, 3, 4, 1, 1]])


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_transposed_anim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "transposed_anim.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def stamps() -> list:
    """Partial sums after 0, 1, 2, 3, 4 stamps, row-major over the input."""
    n = (X.shape[0] - 1) * STRIDE + K.shape[0]
    out, history = np.zeros((n, n), dtype=int), []
    history.append(out.copy())
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i * STRIDE:i * STRIDE + 3, j * STRIDE:j * STRIDE + 3] += X[i, j] * K
            history.append(out.copy())
    return history


def check(final: np.ndarray, log: logging.Logger) -> None:
    t = torch.nn.functional.conv_transpose2d(
        torch.tensor(X, dtype=torch.float32)[None, None],
        torch.tensor(K, dtype=torch.float32)[None, None], stride=STRIDE)[0, 0].numpy()
    if not (np.array_equal(final, EXPECTED) and np.allclose(t, EXPECTED)):
        raise AssertionError(f"transposed conv drifted:\n{final}\ntorch:\n{t}")
    log.info(f"5x5 output matches torch conv_transpose2d:\n{final}")


def grid(ax, x0, y0, vals, cell, fc="white", highlight=None, hl_color=None, show=None):
    n = vals.shape[0]
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * cell, y0 - i * cell
            on = highlight is not None and highlight[i, j]
            ax.add_patch(Rectangle((x, y - cell), cell, cell, ec="black", lw=0.6,
                                   fc=hl_color if on else fc, alpha=1.0))
            if show is None or show[i, j]:
                ax.text(x + cell / 2, y - cell / 2, str(vals[i, j]), ha="center",
                        va="center", fontsize=8, color="white" if on else "black",
                        fontweight="bold" if on else "normal")


def draw(history: list, log: logging.Logger) -> None:
    cell = 0.25
    for step, partial in enumerate(history):
        fig = plt.figure(figsize=(5.0, 1.45))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 5.0); ax.set_ylim(0, 1.45); ax.axis("off")
        top = 1.2
        # input: the cell that is stamping now is coloured
        hl_in = np.zeros_like(X, dtype=bool)
        color = None
        if step > 0:
            i, j = divmod(step - 1, X.shape[1])
            hl_in[i, j] = True
            color = STAMP_COLORS[step - 1]
        grid(ax, 0.08, top - 0.27, X, cell, fc="#dce6f5", highlight=hl_in, hl_color=color)
        ax.text(0.08 + cell, top - 0.12, "input 2x2", ha="center", fontsize=8)
        ax.text(0.78, top - 0.27 - cell, "x", ha="center", va="center", fontsize=11)
        grid(ax, 0.95, top - 0.13, K, cell, fc="#f2f2f2")
        ax.text(0.95 + 1.5 * cell, top + 0.02, "kernel 3x3 (learned)", ha="center", fontsize=8)
        ax.annotate("", (2.2, top - 0.55), (1.82, top - 0.55),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
        ax.text(2.01, top - 0.42, "stride 2", ha="center", fontsize=7.5)
        # output: show every cell touched so far; the current stamp's 3x3 is coloured
        hl_out = np.zeros_like(partial, dtype=bool)
        if step > 0:
            i, j = divmod(step - 1, X.shape[1])
            hl_out[i * STRIDE:i * STRIDE + 3, j * STRIDE:j * STRIDE + 3] = True
        touched = np.zeros_like(partial, dtype=bool)
        for s in range(step):
            i, j = divmod(s, X.shape[1])
            touched[i * STRIDE:i * STRIDE + 3, j * STRIDE:j * STRIDE + 3] = True
        grid(ax, 2.35, top + 0.07, partial, cell, fc="#fbfbfb", highlight=hl_out,
             hl_color=color, show=touched)
        ax.text(2.35 + 2.5 * cell, top + 0.12, "output 5x5", ha="center", fontsize=8)
        if step == len(history) - 1:            # bold outline: cells that got 2+ stamps
            for i in range(partial.shape[0]):
                for j in range(partial.shape[1]):
                    if i == 2 or j == 2:
                        ax.add_patch(Rectangle((2.35 + j * cell, top + 0.07 - (i + 1) * cell),
                                               cell, cell, fill=False, ec="black", lw=2.0))
        if step == 0:
            msg = "each input number\nwill stamp a scaled\ncopy of the kernel"
        elif step < 4:
            i, j = divmod(step - 1, X.shape[1])
            msg = f"stamp {step}: {X[i, j]} x kernel,\nplaced at row {i * STRIDE}, col {j * STRIDE}"
        else:
            msg = ("stamp 4 - done.\nBold cells: 2+ stamps,\nADDED. Centre:\n"
                   "1 + 2 + 3 + 1 = 7")
        ax.text(3.75, top - 0.3, msg, ha="left", va="top", fontsize=7.5, linespacing=1.15)
        out = FIG_DIR / f"transposed_anim_{step}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    history = stamps()
    check(history[-1], log)
    draw(history, log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
