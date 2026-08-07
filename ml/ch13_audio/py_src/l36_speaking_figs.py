"""Concept figures for L36 (Audio-Language Models II - how a model speaks).

NO MODEL IS TRAINED and NO AUDIO FILE IS READ (instructor decisions, AUDIO_CHAPTER_PLAN.md
2026-08-07). The one measurement here is k-means residual quantization, which is clustering,
not network training, and runs in seconds on CPU.

Generates into ml/ch13_audio/fig/:
  vq_residual_stages.pdf -- THE CHAPTER'S MEASUREMENT: what each residual level actually buys
  split_rvq.pdf          -- Mimi's failed design next to the one it shipped
  codebook_patterns.pdf  -- flatten vs parallel vs delay, as step-numbered grids
  duplex_streams.pdf     -- Moshi's 17 streams on a timeline

READ THIS BEFORE QUOTING vq_residual_stages: it is a STAND-IN, not a codec. It quantizes
log-mel frames, not a learned encoder's latents. Log-mel has already discarded phase (L35
frame 12), so these frames cannot be turned back into audio at all, and a real codec trains its
codebooks jointly with a decoder under a spectral and adversarial loss. The shape of the curve
transfers; the numbers do not. That caveat is printed on the slide, not just here.

Codebooks are fit on a 60 s synthesized corpus and applied to a HELD-OUT utterance, so the
reported error is not training-set reconstruction.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch13_audio/py_src/l36_speaking_figs.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import (N_MELS, SR, build_logger, log_mel, quantize, residual_vq,
                          synth_corpus, synth_panir)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
FIG = Path(__file__).resolve().parents[1] / "fig"

N_LEVELS = 8      # Moshi runs Q=8; the released Mimi checkpoint ships 32
K = 64            # codebook size per level; 2048 in Mimi, but 6,000 frames cannot support that

log = build_logger("l36_speaking_figs")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.relative_to(FIG.parents[2])}")


# ---------------------------------------------------------------- 1. the measurement

def fig_vq_residual_stages():
    """What does each extra residual level actually buy? Measured, not asserted."""
    train = log_mel(synth_corpus(60.0)).T
    held, marks = synth_panir(f0=132.0)          # a voice the codebooks never saw
    test = log_mel(held).T
    log.info(f"train {train.shape} frames, held-out {test.shape} frames, "
             f"{N_LEVELS} levels, K={K}")

    codebooks, _, _ = residual_vq(train, N_LEVELS, K)

    # Apply the trained codebooks to held-out data, level by level.
    residual, approx, recons = test.copy(), np.zeros_like(test), []
    for cb in codebooks:
        _, q = quantize(residual, cb)
        approx = approx + q
        residual = residual - q
        recons.append(approx.copy())

    # Baseline is a codebook of size 1: predict the training mean for every frame. Reporting
    # against raw signal energy instead would flatter level 1, which mostly learns the mean.
    base_mse = ((test - train.mean(axis=0)) ** 2).mean()
    mses = [((test - r) ** 2).mean() for r in recons]
    bits = [(lvl + 1) * np.log2(K) for lvl in range(N_LEVELS)]

    log.info(f"baseline (size-1 codebook = the mean) MSE {base_mse:.4f}")
    prev = base_mse
    for lvl, m in enumerate(mses, start=1):
        log.info(f"  level {lvl}: MSE {m:.4f}  cumulative {10 * np.log10(base_mse / m):5.2f} dB"
                 f"  this level {10 * np.log10(prev / m):5.2f} dB"
                 f"  {int(bits[lvl - 1])} bits/frame")
        prev = m

    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)

    dur = len(held) / SR
    show = [(0, "level 1 only"), (1, "levels 1-2"), (3, "levels 1-4"), (7, "levels 1-8")]
    ax = fig.add_subplot(gs[0, :])
    panels = [test.T] + [recons[i].T for i, _ in show]
    titles = ["original log-mel"] + [t for _, t in show]
    combined = np.concatenate(panels, axis=1)
    ax.imshow(combined, origin="lower", aspect="auto", cmap="magma")
    for i in range(1, len(panels)):
        ax.axvline(i * test.shape[0], color="w", lw=1.4)
    for i, t in enumerate(titles):
        ax.text((i + 0.5) * test.shape[0], N_MELS * 1.03, t, ha="center", fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel("mel bin")
    ax.set_title(f"Held-out utterance, reconstructed from {K}-entry codebooks "
                 f"(fit on a different 60 s corpus)", fontsize=10, pad=18)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(range(1, N_LEVELS + 1), mses, "o-", color=BLUE, ms=5)
    ax1.axhline(base_mse, color="0.5", ls="--", lw=1)
    # Log scale, or the size-1 baseline (8.57) flattens every value that matters (0.04-0.18).
    ax1.set_yscale("log")
    ax1.set_ylim(min(mses) * 0.55, base_mse * 2.2)
    ax1.text(1.0, base_mse * 1.25, "size-1 codebook (the mean)", fontsize=7.5, color="0.4")
    ax1.set_xlabel("residual levels used")
    ax1.set_ylabel("reconstruction MSE (log)")
    ax1.grid(alpha=0.25, which="both")

    ax2 = fig.add_subplot(gs[1, 1])
    drops = [10 * np.log10(base_mse / mses[0])] + [
        10 * np.log10(mses[i - 1] / mses[i]) for i in range(1, N_LEVELS)]
    # Level 1 dwarfs the rest, so on a linear axis levels 5-8 are sub-pixel and their labels
    # collide. A student reviewer could not read the 8th bar and so could not check the total.
    colors = [RED] + [ORANGE] * (N_LEVELS - 1)
    bars = ax2.bar(range(1, N_LEVELS + 1), drops, color=colors, alpha=0.9)
    ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, rotation=90)
    ax2.set_xlabel("residual level")
    ax2.set_ylabel("dB gained by this level")
    ax2.set_xticks(range(1, N_LEVELS + 1))
    ax2.set_yscale("log")
    ax2.set_ylim(0.2, max(drops) * 4.5)
    ax2.set_title(f"level 1: {drops[0]:.1f} dB     levels 2-8: {sum(drops[1:]):.1f} dB together",
                  fontsize=8.5)
    ax2.grid(alpha=0.25, axis="y", which="both")

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(bits, [10 * np.log10(base_mse / m) for m in mses], "o-", color=RED, ms=5)
    ax3.set_xlabel("bits per frame")
    ax3.set_ylabel("total dB improvement\nover the baseline")
    ax3.grid(alpha=0.25)
    fig.suptitle("Stand-in, not a codec: k-means on log-mel frames, which cannot be turned "
                 "back into sound. Read the shape, not the numbers.", fontsize=9.5, y=0.02)
    save(fig, "vq_residual_stages")
    return mses, drops


# ---------------------------------------------------------------- 2. Mimi's split

def _box(ax, x, y, w, h, text, color, fontsize=8.5, alpha=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=color, ec=color, alpha=alpha, lw=1.4))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="none", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, xy_from, xy_to, color="0.35"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.2, shrinkA=1, shrinkB=1))


def fig_split_rvq():
    """The failed design next to the shipped one.

    The draft of this chapter said Mimi distills WavLM into RVQ level 1. It does not - that was
    tried and it degraded audio quality, so Mimi runs a plain semantic VQ in PARALLEL with a
    7-level acoustic RVQ and sums the outputs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("The obvious design: distill semantics into level 1", fontsize=10, color=RED)
    _box(ax, 0.06, 0.62, 0.24, 0.16, "encoder\noutput", "0.35")
    for i, (yy, lab) in enumerate([(0.40, "RVQ level 1\n(+ WavLM loss)"),
                                   (0.20, "RVQ levels 2-8")]):
        _box(ax, 0.06, yy, 0.24, 0.14, lab, [RED, BLUE][i])
    _arrow(ax, (0.18, 0.62), (0.18, 0.55))
    _arrow(ax, (0.18, 0.40), (0.18, 0.35))
    ax.text(0.40, 0.50, "one chain: levels 2-8 must\nmodel whatever level 1\nleft behind",
            fontsize=8.5, va="center")
    ax.add_patch(FancyBboxPatch((0.36, 0.06), 0.60, 0.24, boxstyle="round,pad=0.015",
                                fc=RED, ec=RED, alpha=0.10, lw=1.3))
    ax.text(0.66, 0.18, "Measured: phonetic discriminability up,\n"
                        "audio quality DOWN. Not shipped.",
            ha="center", va="center", fontsize=9, color=RED)

    ax = axes[1]
    ax.set_title("What Mimi ships: a split", fontsize=10, color="#008C46")
    _box(ax, 0.06, 0.72, 0.26, 0.15, "encoder output", "0.35")
    _box(ax, 0.06, 0.40, 0.26, 0.18, "plain VQ\nsemantic\n(WavLM distilled)", ORANGE)
    _box(ax, 0.44, 0.40, 0.26, 0.18, "RVQ, 7 levels\nacoustic", BLUE)
    _box(ax, 0.25, 0.10, 0.26, 0.14, "sum the outputs", "#008C46")
    _arrow(ax, (0.19, 0.72), (0.19, 0.58))
    _arrow(ax, (0.28, 0.79), (0.57, 0.58))
    _arrow(ax, (0.19, 0.40), (0.33, 0.24))
    _arrow(ax, (0.57, 0.40), (0.44, 0.24))
    ax.text(0.79, 0.20, "in parallel, so acoustic\ndetail need not live in\nthe semantic "
                        "quantizer's\nresidual", fontsize=8.5, va="center", ha="center")
    fig.suptitle("They tried the obvious thing, measured it, and restructured "
                 "(Défossez et al., 2024)", fontsize=9.5, y=0.02)
    save(fig, "split_rvq")


# ---------------------------------------------------------------- 3. codebook patterns

def fig_codebook_patterns():
    """How do you lay K tokens per timestep on a 1D tape? Three answers, with their costs."""
    T, Kb = 6, 4
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))

    def draw(ax, steps, title, color, cost):
        for t in range(T):
            for k in range(Kb):
                s = steps[k][t]
                ax.add_patch(Rectangle((t, Kb - 1 - k), 0.94, 0.94,
                                       fc=color, ec="white", lw=1.2,
                                       alpha=0.28 + 0.6 * (s / max(1, np.max(steps)))))
                ax.text(t + 0.47, Kb - 1 - k + 0.47, "-" if s < 0 else str(s),
                        ha="center", va="center", fontsize=8)
        ax.set_xlim(-0.1, T + 0.1)
        ax.set_ylim(-0.1, Kb + 0.1)
        ax.set_xticks([t + 0.47 for t in range(T)])
        ax.set_xticklabels([f"t{t + 1}" for t in range(T)], fontsize=8)
        ax.set_yticks([Kb - 1 - k + 0.47 for k in range(Kb)])
        ax.set_yticklabels([f"cb{k + 1}" for k in range(Kb)], fontsize=8)
        ax.set_title(f"{title}\n{cost}", fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    flat = np.array([[t * Kb + k + 1 for t in range(T)] for k in range(Kb)])
    draw(axes[0], flat, "flatten", BLUE, f"K x T = {Kb * T} steps - best quality, slowest")

    par = np.array([[t + 1 for t in range(T)] for _ in range(Kb)])
    draw(axes[1], par, "parallel", RED,
         f"T = {T} steps - fastest, assumes the K are independent")

    dly = np.array([[t + k + 1 for t in range(T)] for k in range(Kb)])
    draw(axes[2], dly, "delay", ORANGE, f"T + K - 1 = {T + Kb - 1} steps - MusicGen's choice")
    fig.suptitle("The number in each cell is the generation step at which that token is "
                 "emitted (K=4 codebooks, T=6 frames)", fontsize=9, y=0.01)
    fig.tight_layout()
    save(fig, "codebook_patterns")


# ---------------------------------------------------------------- 4. Moshi's streams

def fig_duplex_streams():
    """17 streams at 12.5 Hz. The model is always listening and always speaking."""
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    n_steps = 25
    rng = np.random.default_rng(509)

    rows = ([("text (inner monologue)", "#008C46")]
            + [(f"Moshi audio {i + 1}", BLUE) for i in range(8)]
            + [(f"user audio {i + 1}", ORANGE) for i in range(8)])

    user_active = np.zeros(n_steps, bool)
    user_active[1:11] = True
    user_active[19:23] = True          # the user interrupts
    moshi_active = np.zeros(n_steps, bool)
    moshi_active[9:22] = True          # Moshi starts before the user has finished

    for r, (label, color) in enumerate(rows):
        y = len(rows) - 1 - r
        if label.startswith("user"):
            active = user_active
        elif label.startswith("Moshi"):
            active = moshi_active
        else:
            active = moshi_active
        for t in range(n_steps):
            on = active[t]
            if label.startswith("text"):
                on = active[t] and (t % 3 == 0)      # text is sparser than audio
            ax.add_patch(Rectangle((t, y), 0.9, 0.8,
                                   fc=color if on else "0.92",
                                   ec="white", lw=0.5, alpha=0.95 if on else 1.0))
        ax.text(-0.4, y + 0.4, label, ha="right", va="center", fontsize=7.5)

    ax.axvspan(9, 11, color="0.2", alpha=0.10)
    ax.text(10, len(rows) + 0.5, "both speaking", ha="center", fontsize=8.5)
    ax.axvspan(19, 22, color=RED, alpha=0.10)
    ax.text(20.5, len(rows) + 0.5, "user interrupts", ha="center", fontsize=8.5, color=RED)

    ax.set_xlim(-6, n_steps)
    ax.set_ylim(-0.3, len(rows) + 1.4)
    ax.set_yticks([])
    ax.set_xticks(range(0, n_steps, 5))
    ax.set_xticklabels([f"{t / 12.5:.1f}s" for t in range(0, n_steps, 5)], fontsize=8)
    ax.set_xlabel("time (one column = one 12.5 Hz frame = 80 ms)")
    ax.set_title("Moshi predicts K = 2Q+1 = 17 streams every frame. Nothing here is a turn.",
                 fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, "duplex_streams")


def main():
    log.info("=" * 70)
    log.info("L36 figures - k-means only, no trained network, no recording")
    mses, drops = fig_vq_residual_stages()
    front_loaded = drops[0] > 2 * np.mean(drops[1:])
    log.info(f"per-level dB gains {[round(d, 2) for d in drops]} -> "
             f"{'FRONT-LOADED' if front_loaded else 'roughly constant'}")
    fig_split_rvq()
    fig_codebook_patterns()
    fig_duplex_streams()
    log.info("L36 figures done")


if __name__ == "__main__":
    main()
