"""Real figure for the L16 CNN Foundations deck (Section 2, what a spectrum is).

Generates into ml/ch6_cnn/fig/:
  spectrum_1d.pdf  -- left: a 1D signal that is the sum of three sine waves (shown faint);
                      right: its Fourier amplitude spectrum - three clean spikes at exactly
                      those three frequencies. The transform "reads off how much of each
                      frequency" is present. The 2D astronaut spectrum on the next frame is
                      the same idea in two dimensions.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/spectrum_1d.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette for 3+ series (red/blue/orange); bars carry value labels.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# three pure tones: (frequency in cycles over the window, amplitude, colour)
TONES = [(2, 1.0, ARM_RED), (5, 0.5, ARM_ORANGE), (9, 0.3, ARM_BLUE)]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_spectrum_1d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "spectrum_1d.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    n = 512
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.zeros_like(t)
    components = []
    for freq, amp, col in TONES:
        comp = amp * np.sin(2 * np.pi * freq * t)
        components.append((comp, col))
        signal += comp
    log.info(f"built signal from {len(TONES)} tones")

    amp_spec = np.abs(np.fft.rfft(signal)) / (n / 2)      # single-sided amplitude
    freqs = np.fft.rfftfreq(n, d=1.0 / n)                  # -> cycles over the window

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))

    ax = axes[0]
    for comp, col in components:
        ax.plot(t, comp, color=col, lw=1.0, alpha=0.35)
    ax.plot(t, signal, color="black", lw=2.2, label="signal = sum of the three")
    ax.set_title("A signal in time", fontsize=12)
    ax.set_xlabel("time"); ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    keep = freqs <= 14
    colours = ["#cccccc"] * int(keep.sum())
    for freq, _amp, col in TONES:
        colours[int(round(freq))] = col
    bars = ax.bar(freqs[keep], amp_spec[keep], width=0.35, color=colours)
    ax.set_title(r"Its spectrum $\mathcal{F}$: how much of each frequency", fontsize=12)
    ax.set_xlabel("frequency"); ax.set_ylabel("amount")
    ax.set_xticks([f for f, _, _ in TONES])
    for freq, amp, _ in TONES:
        ax.text(freq, amp + 0.03, f"{amp:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.2); ax.grid(axis="y", alpha=0.2)

    fig.suptitle("The Fourier transform reads off the frequencies hiding in a signal",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "spectrum_1d.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
