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


# ---------------------------------------------------------------- explanatory diagrams
# Added 2026-08-08 (instructor: expand, more illustrations). The residual-quantization
# mechanism - the central idea of the deck - previously had no picture at all.

def fig_rvq_mechanism():
    """Residual quantization, one level at a time, with the residual actually shrinking.

    The bars are REAL: a frame from the corpus, quantized level by level with the same
    codebooks the measurement uses. The point is visual - what is left over gets smaller.
    """
    train = log_mel(synth_corpus(60.0)).T
    codebooks, _, _ = residual_vq(train, 4, K)
    x = train[len(train) // 3]

    residual, approx = x.copy(), np.zeros_like(x)
    rows = [("the frame itself", x.copy(), None)]
    for lvl, cb in enumerate(codebooks, start=1):
        _, q = quantize(residual[None, :], cb)
        approx = approx + q[0]
        residual = residual - q[0]
        rows.append((f"after level {lvl}", approx.copy(), residual.copy()))
    log.info("rvq mechanism figure: residual norms "
             f"{[round(float(np.linalg.norm(r[2])), 2) for r in rows[1:]]}")

    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 5.4),
                             gridspec_kw={"width_ratios": [1, 1]})
    for r, (label, approxv, resid) in enumerate(rows):
        axes[r, 0].plot(approxv, lw=1.4, color=BLUE)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylim(x.min() - 0.5, x.max() + 0.5)
        if resid is None:
            axes[r, 1].axis("off")
            axes[r, 1].text(0.5, 0.5, "nothing quantized yet", ha="center", va="center",
                            fontsize=8, color="0.5", transform=axes[r, 1].transAxes)
        else:
            axes[r, 1].plot(resid, lw=1.2, color=RED)
            axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
            axes[r, 1].set_ylim(-3, 3)
            axes[r, 1].text(0.98, 0.85, f"$\\|r\\| = {np.linalg.norm(resid):.1f}$",
                            transform=axes[r, 1].transAxes, ha="right", fontsize=8,
                            color=RED)
    axes[0, 0].set_title("what the decoder would receive", fontsize=9.5, color=BLUE)
    axes[0, 1].set_title("what is still LEFT OVER (the residual)", fontsize=9.5, color=RED)
    fig.suptitle("Each level quantizes the previous level's mistake. The red curve is what "
                 "the next level has to explain.", fontsize=9.5, y=0.02)
    fig.tight_layout()
    save(fig, "rvq_mechanism")


def _box(ax, x, y, w, h, text, color, fs=8, alpha=0.16):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arr(ax, x0, y0, x1, y1, color="0.4", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig_codec_architecture():
    """Encoder, quantizer, decoder - and the fact that it lives on WAVEFORMS."""
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.0); ax.axis("off")

    _box(ax, 0.15, 1.25, 1.5, 0.8, "waveform\n24 kHz", ORANGE, fs=8)
    _arr(ax, 1.65, 1.65, 2.05, 1.65)
    _box(ax, 2.05, 1.25, 1.7, 0.8, "conv encoder\nstride 1920", BLUE, fs=8)
    _arr(ax, 3.75, 1.65, 4.15, 1.65)
    _box(ax, 4.15, 1.25, 1.8, 0.8, "residual\nquantizer (8)", "#7832A0", fs=8)
    _arr(ax, 5.95, 1.65, 6.35, 1.65)
    _box(ax, 6.35, 1.25, 1.7, 0.8, "conv decoder", BLUE, fs=8)
    _arr(ax, 8.05, 1.65, 8.45, 1.65)
    _box(ax, 8.45, 1.25, 1.5, 0.8, "waveform\nagain", ORANGE, fs=8)

    ax.text(2.9, 1.05, "12.5 frames/s", ha="center", fontsize=7.5, color="0.35")
    ax.text(5.05, 1.05, "8 x 11 bits\n= 1.1 kbps", ha="center", va="top", fontsize=7.5,
            color="0.35")
    ax.text(5.5, 2.6, "Waveform in, waveform out - NOT a spectrogram, because a spectrogram "
                      "cannot be played back (L35).", ha="center", fontsize=9.5, color=RED)
    ax.text(5.5, 0.35, "Trained end to end with a multi-scale spectral loss plus "
                       "discriminators - not squared error.",
            ha="center", fontsize=8.5, style="italic", color="0.3")
    save(fig, "codec_architecture")


def fig_audiolm_stages():
    """AudioLM's three stages: long-range structure and local fidelity are different jobs."""
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.2); ax.axis("off")
    stages = [
        ("Stage 1\nsemantic tokens\n(from w2v-BERT)", "what is being said,\nover seconds",
         "#008C46"),
        ("Stage 2\ncoarse acoustic\n(SoundStream)", "whose voice,\nwhat room", BLUE),
        ("Stage 3\nfine acoustic", "the last details\nof fidelity", ORANGE),
    ]
    x = 0.4
    for i, (label, sub, color) in enumerate(stages):
        _box(ax, x, 1.35, 2.6, 1.15, label, color, fs=8.5)
        ax.text(x + 1.3, 1.05, sub, ha="center", va="top", fontsize=8, color="0.3")
        if i < len(stages) - 1:
            _arr(ax, x + 2.6, 1.92, x + 3.15, 1.92)
            ax.text(x + 2.87, 2.05, "conditions", fontsize=6.5, ha="center", color="0.45")
        x += 3.15
    ax.text(5.2, 2.9, "Each stage is its own transformer, conditioned on everything before it",
            ha="center", fontsize=9.5)
    ax.text(5.2, 0.35, "One model doing all three spends its capacity in the wrong place: "
                       "gorgeous audio that says nothing.",
            ha="center", fontsize=8.5, style="italic", color="0.3")
    save(fig, "audiolm_stages")


def fig_inner_monologue():
    """Moshi writes what it is about to say, then says it - 12.5 times a second."""
    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.4); ax.axis("off")

    words = ["", "the", "", "cheese", "", "is", "ready", ""]
    n = len(words)
    w = 1.15
    for t in range(n):
        x = 0.55 + t * w
        _box(ax, x, 2.25, w - 0.12, 0.55, words[t] if words[t] else "-",
             "#008C46" if words[t] else "0.75", fs=8.5)
        for k in range(3):
            _box(ax, x, 1.55 - k * 0.42, w - 0.12, 0.34, "", BLUE, alpha=0.30 - k * 0.06)
        _arr(ax, x + (w - 0.12) / 2, 2.25, x + (w - 0.12) / 2, 1.92, color="#008C46", lw=1.1)
    ax.text(0.45, 2.52, "text", ha="right", va="center", fontsize=9, color="#008C46")
    ax.text(0.45, 1.1, "audio\n(8 codebooks)", ha="right", va="center", fontsize=9,
            color=BLUE)
    ax.text(5.3, 3.1, "Within each 80 ms frame, the TEXT token is predicted FIRST, and the "
                      "audio is conditioned on it", ha="center", fontsize=9.5)
    ax.text(5.3, 0.32, "So the language model's fluency reaches the speech - and shifting "
                       "this stream turns the same model into a recogniser or a synthesiser.",
            ha="center", fontsize=8.5, style="italic", color="0.3")
    save(fig, "inner_monologue")


def fig_rq_transformer():
    """Why it takes two transformers: one across time, a small one within a frame."""
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.6); ax.axis("off")

    for t in range(4):
        x = 0.6 + t * 1.5
        _box(ax, x, 2.5, 1.25, 0.7, f"frame {t+1}", "#7832A0", fs=8)
        if t < 3:
            _arr(ax, x + 1.25, 2.85, x + 1.5, 2.85, color="#7832A0", lw=1.6)
    ax.text(0.5, 2.85, "temporal\n32 layers, 4096 wide\nsteps at 12.5 Hz", ha="right",
            va="center", fontsize=8, color="#7832A0")

    _arr(ax, 1.2, 2.5, 1.2, 2.1, color="0.5")
    for k in range(4):
        _box(ax, 0.6, 1.6 - k * 0.42, 1.25, 0.34,
             ["text", "audio 1", "audio 2", "..."][k], BLUE, fs=7)
        if k < 3:
            _arr(ax, 1.9, 1.77 - k * 0.42, 1.9, 1.77 - (k + 1) * 0.42, color=BLUE, lw=1.1)
    ax.text(2.15, 1.1, "depth transformer\n6 layers, 1024 wide\nruns INSIDE one frame,\n"
                       "across all 17 streams", ha="left", va="center", fontsize=8,
            color=BLUE)

    ax.text(7.4, 1.9, "Why not one big vocabulary?\n"
                      "$2048^{17}$ entries.\n\n"
                      "Why not 17 separate steps?\n"
                      "the frame rate dies.",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RED, lw=1.2))
    save(fig, "rq_transformer")


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
    fig_rvq_mechanism()
    fig_codec_architecture()
    fig_audiolm_stages()
    fig_inner_monologue()
    fig_rq_transformer()
    log.info("L36 figures done")


if __name__ == "__main__":
    main()
