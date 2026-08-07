"""Concept figures for L35 (Audio-Language Models I - how a model hears).

NO MODEL IS TRAINED and NO AUDIO FILE IS READ (instructor decisions, AUDIO_CHAPTER_PLAN.md
2026-08-07). Every waveform is synthesized by py_src/audio_common.py and is labelled as
synthetic on the slide; every other number here is exact arithmetic from a cited source.

Generates into ml/ch13_audio/fig/:
  waveform_zoom.pdf       -- one signal at 2 s / 100 ms / 5 ms, down to individual samples
  framing_window.pdf      -- 25 ms windows at a 10 ms hop, overlapping
  phase_discard.pdf       -- same magnitude spectrogram, true vs random phase
  mel_filterbank.pdf      -- the 80 triangular mel filters, warped
  spectrogram_stages.pdf  -- waveform -> power -> mel -> log-mel, the Whisper front end
  token_rate_ladder.pdf   -- representations per second of audio, log axis
  latency_budget.pdf      -- naive cascade vs tuned cascade vs native
  audio_token_budget.pdf  -- a minute of audio against its own transcript, and an image

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch13_audio/py_src/l35_hearing_figs.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import (HOP, HOP_MS, N_FFT, N_MELS, SR, WIN_MS, build_logger, istft,
                          log_mel, mel_filterbank, stft_mag, synth_panir)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
FIG = Path(__file__).resolve().parents[1] / "fig"

log = build_logger("l35_hearing_figs")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.relative_to(FIG.parents[2])}")


# ---------------------------------------------------------------- 1. waveform at three zooms

def fig_waveform_zoom(wave):
    """The frame that makes '16,000 numbers per second' concrete rather than a claim."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    t = np.arange(len(wave)) / SR

    axes[0].plot(t, wave, lw=0.4, color=BLUE)
    axes[0].set_title(f"the whole word ({len(wave):,} samples)", fontsize=10)
    axes[0].set_xlabel("seconds")

    lo, hi = int(0.40 * SR), int(0.50 * SR)          # 100 ms inside the vowel Ի
    axes[1].plot(t[lo:hi], wave[lo:hi], lw=0.8, color=BLUE)
    axes[1].set_title(f"100 ms ({hi - lo:,} samples)", fontsize=10)
    axes[1].set_xlabel("seconds")

    lo2, hi2 = int(0.440 * SR), int(0.445 * SR)      # 5 ms - individual samples visible
    axes[2].plot(t[lo2:hi2], wave[lo2:hi2], "o-", ms=3.5, lw=0.9, color=RED)
    axes[2].set_title(f"5 ms ({hi2 - lo2} samples)", fontsize=10)
    axes[2].set_xlabel("seconds")

    for ax in axes:
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "waveform_zoom")


# ---------------------------------------------------------------- 2. framing

def fig_framing_window(wave):
    """25 ms window, 10 ms hop. The overlap is the part students always ask about."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 4.2),
                                   gridspec_kw={"height_ratios": [2, 1]})
    lo, hi = int(0.30 * SR), int(0.38 * SR)
    seg = wave[lo:hi]
    t = np.arange(len(seg)) / SR * 1000

    ax0.plot(t, seg, lw=0.6, color="0.35")
    for i, color in zip(range(5), [BLUE, RED, ORANGE, BLUE, RED]):
        start = i * HOP
        if start + N_FFT > len(seg):
            break
        w = np.hanning(N_FFT)
        ax0.plot(t[start:start + N_FFT], w * 0.9, color=color, lw=1.6, alpha=0.85)
    ax0.set_ylabel("amplitude")
    ax0.set_title(f"{WIN_MS:.0f} ms windows, {HOP_MS:.0f} ms hop - each window overlaps its "
                  f"neighbour by {WIN_MS - HOP_MS:.0f} ms", fontsize=10)
    ax0.grid(alpha=0.25)

    for i in range(5):
        start_ms = i * HOP_MS
        ax1.add_patch(Rectangle((start_ms, i * 0.16), WIN_MS, 0.13,
                                color=[BLUE, RED, ORANGE, BLUE, RED][i], alpha=0.75))
        ax1.text(start_ms + WIN_MS + 1, i * 0.16 + 0.04, f"frame {i + 1}", fontsize=8)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(-0.03, 0.85)
    ax1.set_yticks([])
    ax1.set_xlabel("milliseconds")
    fig.tight_layout()
    save(fig, "framing_window")


# ---------------------------------------------------------------- 3. phase

def fig_phase_discard(wave):
    """Why keeping only magnitude is a real loss - the frame the whole of L36 leans on.

    Same magnitude spectrogram, once with the true phase and once with random phase. The
    waveforms are unrecognisably different, which is exactly why a spectrogram cannot be
    played back and why codecs work on waveforms instead.
    """
    rng = np.random.default_rng(509)
    spec = stft_mag(wave, return_complex=True)
    mag = np.abs(spec)

    true_back = istft(mag * np.exp(1j * np.angle(spec)))
    rand_back = istft(mag * np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape)))
    # Framing drops the partial tail frame, so the reconstruction is shorter than the input.
    # Compare only where overlap-add is fully supported (one window in from each end).
    n = len(true_back)
    core = slice(N_FFT, n - N_FFT)
    err = np.abs(true_back[core] - wave[:n][core]).max()
    log.info(f"phase figure: reconstruction {n:,} samples vs {len(wave):,} in "
             f"(tail frame dropped by framing); max abs error over the interior {err:.2e}")
    if err > 1e-8:
        raise RuntimeError(f"istft(stft(x)) should be exact in the interior, got {err:.2e}")
    wave = wave[:n]

    fig, axes = plt.subplots(2, 2, figsize=(10, 4.6))
    t = np.arange(len(wave)) / SR
    for col, (sig, name, color) in enumerate([(true_back, "true phase kept", BLUE),
                                              (rand_back, "phase replaced by noise", RED)]):
        axes[0, col].plot(t, sig, lw=0.4, color=color)
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_ylabel("amplitude")
        axes[0, col].set_xlabel("seconds")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].imshow(np.log10(np.maximum(stft_mag(sig), 1e-8)), origin="lower",
                            aspect="auto", cmap="magma",
                            extent=[0, len(sig) / SR, 0, SR / 2000])
        axes[1, col].set_ylabel("kHz")
        axes[1, col].set_xlabel("seconds")
    fig.suptitle("Identical magnitude spectrogram fed in. Only the phase differs.", fontsize=10)
    fig.tight_layout()
    save(fig, "phase_discard")


# ---------------------------------------------------------------- 4. mel filterbank

def fig_mel_filterbank():
    """The warping, shown rather than asserted: narrow filters low, wide filters high."""
    fb, bins, edges = mel_filterbank()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.3))

    for m in range(0, N_MELS, 2):
        ax0.plot(bins, fb[m], lw=0.8,
                 color=[BLUE, RED, ORANGE][(m // 2) % 3], alpha=0.75)
    ax0.set_xlabel("frequency (Hz)")
    ax0.set_ylabel("filter gain")
    ax0.set_title(f"{N_MELS} triangular mel filters (every 2nd shown)", fontsize=10)
    ax0.grid(alpha=0.25)

    widths = edges[2:] - edges[:-2]
    ax1.plot(edges[1:-1], widths, "o-", ms=2.5, color=BLUE)
    ax1.set_xlabel("filter centre frequency (Hz)")
    ax1.set_ylabel("filter width (Hz)")
    ax1.set_title("a filter near 8 kHz is ~{:.0f}x wider than one near 200 Hz".format(
        widths[-1] / widths[np.argmin(np.abs(edges[1:-1] - 200))]), fontsize=10)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "mel_filterbank")


# ---------------------------------------------------------------- 5. the four stages

def fig_spectrogram_stages(wave, marks):
    """waveform -> power spectrogram -> mel -> log-mel. Exactly what Whisper is fed."""
    power = stft_mag(wave) ** 2
    fb, _, _ = mel_filterbank()
    mel = fb @ power
    lmel = np.log10(np.maximum(mel, 1e-10))
    dur = len(wave) / SR

    # Wide and short: four stacked panels at 7.2in tall overflow a 16:9 frame that also has to
    # carry a callout box, and Beamer clips the box silently rather than warning.
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 5.3))
    axes[0].plot(np.arange(len(wave)) / SR, wave, lw=0.4, color=BLUE)
    axes[0].set_xlim(0, dur)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"1. waveform - {len(wave):,} numbers", fontsize=10)

    for ax, data, title, ylab, ymax in [
        (axes[1], np.log10(np.maximum(power, 1e-10)),
         f"2. power spectrogram - {power.shape[0]} frequency bins x {power.shape[1]} frames "
         f"(shown on a log scale so anything is visible)", "kHz", SR / 2000),
        (axes[2], mel, f"3. mel - {mel.shape[0]} bins x {mel.shape[1]} frames, and on a linear "
         f"scale almost all of it is invisible", "mel bin", N_MELS),
        (axes[3], lmel, f"4. log-mel - the same {lmel.shape[0]} x {lmel.shape[1]}, log compressed",
         "mel bin", N_MELS)]:
        ax.imshow(data, origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, ymax])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)

    for lab, a, b in marks:
        if lab != "silence":
            axes[3].axvline(a, color="w", lw=0.7, alpha=0.6)
            axes[3].text((a + b) / 2, N_MELS * 0.88, lab.split()[0], color="w",
                         ha="center", fontsize=9)
    axes[3].set_xlabel("seconds")
    fig.tight_layout()
    save(fig, "spectrogram_stages")


# ---------------------------------------------------------------- 6. the token-rate ladder

def fig_token_rate_ladder():
    """Every rate in both decks, on one log axis. All exact, all cited on the slide."""
    rows = [
        ("raw samples\n(16 kHz)", 16000, RED),
        ("STFT frames\n(10 ms hop)", 100, BLUE),
        ("Whisper encoder\n(after stride-2 conv)", 50, BLUE),
        ("Qwen2-Audio\n(stride-2 pooled)", 25, BLUE),
        # Not "AuT" - the acronym is not defined until three frames after this figure appears.
        ("codec frames\n(lowest anyone ships, L36)", 12.5, ORANGE),
        ("English text\n(for comparison)", 2.5, "0.45"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bars = ax.barh(range(len(rows)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("units per second of audio (log scale)")
    ax.set_xlim(1, 60000)
    ax.bar_label(bars, labels=[f"{v:,g}" for v in vals], padding=4, fontsize=9)
    ax.set_title("The whole job of an audio front end: get from the top bar to the bottom one",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    save(fig, "token_rate_ladder")


# ---------------------------------------------------------------- 7. latency

def fig_latency_budget():
    """Three tiers, not two. Two would overstate the case and undercut the next frame."""
    stacks = [
        ("naive cascade", [("endpointing", 500), ("speech recognition", 500),
                           ("LLM first token", 700), ("speech synthesis", 600)]),
        ("tuned streaming\ncascade", [("endpointing", 250), ("speech recognition", 150),
                                      ("LLM first token", 250), ("speech synthesis", 150)]),
        ("native\n(Moshi, measured)", [("one model, end to end", 200)]),
    ]
    palette = [BLUE, RED, ORANGE, "0.55"]

    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    for row, (name, parts) in enumerate(stacks):
        left = 0
        for j, (part, ms) in enumerate(parts):
            ax.barh(row, ms, left=left, color=palette[j % len(palette)], alpha=0.9,
                    edgecolor="white", lw=0.8)
            # Only label a segment wide enough to hold the text; the tuned-cascade segments are
            # 150-250 ms and were previously drawn with unreadable compressed labels inside them.
            if ms >= 450:
                ax.text(left + ms / 2, row, f"{part}\n{ms} ms", ha="center", va="center",
                        fontsize=7.5, color="white")
            elif row == 2:
                ax.text(left + ms / 2, row, f"{ms} ms", ha="center", va="center",
                        fontsize=8, color="white")
            left += ms
        ax.text(left + 40, row, f"{left:,} ms", va="center", fontsize=9.5, fontweight="bold")

    # The tuned row's segments are too narrow to label in place, so name them in the clear
    # space to the right of that row's total rather than under the bar, where they collided.
    ax.text(1080, 1, "same four stages, each faster:\n"
                     "endpointing 250 / recognition 150 / LLM 250 / synthesis 150",
            fontsize=7.5, color="0.35", va="center")

    ax.set_yticks(range(len(stacks)))
    ax.set_yticklabels([s[0] for s in stacks], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds from the moment you stop speaking")
    ax.set_xlim(0, 2700)
    ax.set_title("The bottom bar is Moshi: 160 ms in theory, 200 ms measured. "
                 "Qwen3-Omni reports 234 ms.", fontsize=9.5)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    save(fig, "latency_budget")


# ---------------------------------------------------------------- 8. the token bill

def fig_audio_token_budget():
    """One minute of speech costs roughly ten times its own transcript."""
    rows = [
        ("1 min of audio\n(Gemini, 32 tok/s)", 1920, RED),
        ("its own transcript\n(~150 words)", 150, "0.45"),
        ("one 336px image\n(LLaVA-1.5, L33)", 576, BLUE),
        ("1 min of audio out\n(25 tok/s)", 1500, ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    bars = ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows],
                  alpha=0.9, width=0.62)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("tokens")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 2250)
    ax.set_title("Audio costs about 13x its own transcript. That ratio is why the cascade "
                 "refuses to die.", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "audio_token_budget")


def main():
    log.info("=" * 70)
    log.info("L35 figures - synthesized speech only, no recording, no trained model")
    wave, marks = synth_panir()
    log.info(f"synthesized PANIR: {len(wave):,} samples = {len(wave) / SR:.2f}s at {SR} Hz")
    log.info(f"segments: {[m[0] for m in marks]}")

    n_frames = 1 + (len(wave) - N_FFT) // HOP
    log.info(f"framing: n_fft={N_FFT} hop={HOP} -> {n_frames} frames "
             f"({n_frames / (len(wave) / SR):.0f} frames/sec)")
    log.info(f"30 s of audio -> {30 * SR:,} samples -> {30 * 100:,} frames "
             f"-> {30 * 50:,} Whisper vectors -> {30 * 25:,} pooled")

    fig_waveform_zoom(wave)
    fig_framing_window(wave)
    fig_phase_discard(wave)
    fig_mel_filterbank()
    fig_spectrogram_stages(wave, marks)
    fig_token_rate_ladder()
    fig_latency_budget()
    fig_audio_token_budget()
    log.info("L35 figures done")


if __name__ == "__main__":
    main()
