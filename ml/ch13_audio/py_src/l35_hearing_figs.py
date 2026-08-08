"""Concept figures for L35 (Audio-Language Models I - how a model hears).

NO MODEL IS TRAINED (instructor decision, AUDIO_CHAPTER_PLAN.md 2026-08-07). Every number
here is exact arithmetic from a cited source.

Audio sources, updated 2026-08-08: the chapter is still built on the synthesized waveform from
py_src/audio_common.py, labelled as synthetic on every slide that shows it. It now ALSO reads
ONE real recording - fig/img/panir_real.wav, a human saying the same word - used by exactly two
figures (real_vs_synth, single_spectrum). It exists so the "synthesized" label is checkable
rather than a disclaimer. The original "no audio file is read" note is therefore obsolete.

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
from audio_common import (HOP, HOP_MS, N_FFT, N_MELS, SR, WIN_MS, build_logger, frame_signal,
                          istft, load_real_panir,
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


# ---------------------------------------------------------------- explanatory diagrams
# Added 2026-08-08 (instructor: expand, more illustrations, make it more understandable).
# Several frames previously carried their whole argument in prose or a table.

def _panel(ax, x, y, w, h, text, color, fs=8, alpha=0.16):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=color, alpha=alpha, lw=1.3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, color="0.4", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig_real_vs_synth(synth):
    """The synthetic signal beside a real recording of the same word.

    Every frame in this chapter is labelled "synthesized". This figure is what makes that
    label honest rather than a disclaimer: the structure matches, and the differences are
    visible and namable.
    """
    real = load_real_panir()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.5]})
    for col, (sig, name) in enumerate([(real, "real recording (a human, 1.28 s)"),
                                       (synth, "our formant synthesis (0.80 s)")]):
        t = np.arange(len(sig)) / SR
        axes[0, col].plot(t, sig, lw=0.4, color=[RED, BLUE][col])
        axes[0, col].set_xlim(0, t[-1])
        axes[0, col].set_title(name, fontsize=10, color=[RED, BLUE][col])
        axes[0, col].set_ylabel("amplitude")
        axes[0, col].set_xticks([])
        axes[1, col].imshow(log_mel(sig), origin="lower", aspect="auto", cmap="magma",
                            extent=[0, len(sig) / SR, 0, N_MELS])
        axes[1, col].set_xlabel("seconds")
        axes[1, col].set_ylabel("mel bin")
    fig.suptitle("Same word. The real one has noise, breath and drifting pitch; "
                 "the formants and the burst sit in the same places.", fontsize=9.5, y=0.02)
    fig.tight_layout()
    save(fig, "real_vs_synth")


def fig_single_spectrum():
    """How to READ a spectrum: harmonics are the pitch, peaks in the envelope are formants."""
    real = load_real_panir()
    frames = frame_signal(real)
    energy = (frames ** 2).sum(axis=1)
    idx = int(np.argmax(energy))                 # the loudest frame: a vowel
    frame = frames[idx] * np.hanning(N_FFT)
    spec = np.abs(np.fft.rfft(frame, n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    db = 20 * np.log10(np.maximum(spec, 1e-8))

    # A wide moving average in the log domain leaves the envelope and averages the comb away.
    win = 15
    env = np.convolve(db, np.ones(win) / win, mode="same")
    log.info(f"single-spectrum figure: frame {idx} of {len(frames)} "
             f"({idx * HOP / SR:.2f}s), peak {freqs[np.argmax(spec)]:.0f} Hz")

    # Find the formants rather than hard-coding where they "should" be: the two lowest
    # prominent peaks of the envelope below 3.5 kHz.
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(env[freqs < 3500], prominence=2.0)
    formants = freqs[peaks][:2]
    log.info(f"single-spectrum figure: envelope peaks at {np.round(freqs[peaks][:4], 0)} Hz")

    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    ax.plot(freqs, db, lw=0.8, color="0.62", label="spectrum of one 25 ms frame")
    ax.plot(freqs, env, lw=2.4, color=RED, label="envelope (the shape of the vocal tract)")
    top = env.max()
    for name, f in zip(["F1", "F2"], formants):
        ax.annotate(name, xy=(f, env[np.argmin(np.abs(freqs - f))]),
                    xytext=(f, top + 13), ha="center", fontsize=11, color=RED,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlim(0, 5000)
    ax.set_ylim(db.min() - 3, top + 20)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("One frame of real speech. The fine ripple is PITCH (harmonics of the vocal "
                 "folds);\nthe broad bumps are FORMANTS, and they decide which vowel you hear.",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "single_spectrum")


def fig_whisper_pipeline():
    """30 seconds of audio through Whisper's front end, with the number at every stage."""
    fig, ax = plt.subplots(figsize=(11.0, 2.7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.7); ax.axis("off")
    stages = [
        ("30 s of audio", "480,000\nsamples", ORANGE),
        ("log-mel\n(25 ms / 10 ms)", "80 x 3,000", BLUE),
        ("conv 1\n(stride 1)", "3,000", BLUE),
        ("conv 2\n(stride 2)", "1,500", RED),
        ("transformer\nencoder", "1,500\n= 50 Hz", "#7832A0"),
        ("stride-2 pool\n(Qwen2-Audio)", "750\n= 25 Hz", "#008C46"),
    ]
    x = 0.15
    for i, (label, count, color) in enumerate(stages):
        _panel(ax, x, 1.15, 1.5, 0.85, label, color, fs=7.5)
        ax.text(x + 0.75, 0.82, count, ha="center", va="top", fontsize=8.5,
                fontweight="bold", color="0.2")
        if i < len(stages) - 1:
            _arrow(ax, x + 1.5, 1.57, x + 1.75, 1.57)
        x += 1.78
    ax.text(5.5, 2.45, "Only ONE of the two convolutions has stride 2. "
                       "Two would give 750, and everyone quotes 1,500.",
            ha="center", fontsize=9, color=RED)
    ax.text(5.5, 0.25, "The transcript of the same 30 seconds is about 75 text tokens.",
            ha="center", fontsize=9, style="italic", color="0.3")
    save(fig, "whisper_pipeline")


def fig_three_audio_designs():
    """The three ways to attach ears to a language model, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    specs = [
        ("A - projector", BLUE, "Qwen2-Audio",
         [("audio", ORANGE), ("Whisper-large-v3\n(50 Hz)", BLUE),
          ("stride-2 pool\n+ linear", "#008C46"), ("LLM", "#7832A0")],
         "750 tokens per 30 s\n(scales with length)"),
        ("B - resampler", RED, "SALMONN",
         [("audio", ORANGE), ("Whisper-v2 + BEATs\n(two encoders)", RED),
          ("window-level\nQ-Former", "#008C46"), ("LLM", "#7832A0")],
         "88 tokens per 30 s\n(fixed, whatever is in it)"),
        ("C - native", ORANGE, "Qwen3-Omni",
         [("audio", ORANGE), ("AuT, trained with\nthe model (12.5 Hz)", ORANGE),
          ("Thinker (text)\n+ Talker (speech)", "#008C46"), ("one model", "#7832A0")],
         "375 tokens per 30 s\nand it can answer aloud"),
    ]
    for ax, (title, color, who, boxes, cost) in zip(axes, specs):
        ax.set_xlim(0, 3); ax.set_ylim(0, 4.7); ax.axis("off")
        ax.set_title(f"{title}\n{who}", fontsize=10.5, color=color)
        for i, (label, c) in enumerate(boxes):
            y = 3.65 - i * 0.85
            _panel(ax, 0.3, y, 2.4, 0.62, label, c, fs=7.5)
            if i < len(boxes) - 1:
                _arrow(ax, 1.5, y, 1.5, y - 0.23)
        ax.text(1.5, 0.25, cost, fontsize=8.5, ha="center", color="0.25")
    fig.suptitle("The same question as the vision chapter: how many tokens does the language "
                 "model see, and who decides?", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "three_audio_designs")


def fig_speech_vs_audio():
    """Why SALMONN carries two encoders: a speech encoder is TRAINED to discard the rest."""
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    contents = ["the words", "who is speaking", "the emotion", "a dog barking",
                "music underneath", "a door slamming"]
    for k, c in enumerate(contents):
        _panel(ax, 0.15, 2.75 - k * 0.48, 2.1, 0.38, c, "0.55", fs=7.5)
    ax.text(1.2, 3.25, "what is in the sound", fontsize=8.5, ha="center")

    _panel(ax, 3.0, 1.9, 1.9, 0.7, "speech encoder\n(Whisper)", BLUE, fs=8)
    _panel(ax, 3.0, 0.7, 1.9, 0.7, "audio encoder\n(BEATs)", "#008C46", fs=8)
    for k in range(len(contents)):
        y = 2.94 - k * 0.48
        _arrow(ax, 2.25, y, 3.0, 2.25 if k < 3 else 1.05, color="0.75", lw=0.7)

    _panel(ax, 5.6, 1.9, 1.9, 0.7, "keeps the words,\nDISCARDS the rest", RED, fs=7.5)
    _panel(ax, 5.6, 0.7, 1.9, 0.7, "keeps all of it", "#008C46", fs=8)
    _arrow(ax, 4.9, 2.25, 5.6, 2.25)
    _arrow(ax, 4.9, 1.05, 5.6, 1.05)

    ax.text(8.7, 1.65, "If you want a model\nthat HEARS and not one\nthat TRANSCRIBES,\n"
                       "you need both.", ha="center", fontsize=9, color="0.2")
    ax.text(5.0, 0.15, "A transcription objective treats everything that is not a word as "
                       "noise to suppress.", ha="center", fontsize=8.5, style="italic",
            color="0.35")
    save(fig, "speech_vs_audio")


def main():
    log.info("=" * 70)
    log.info("L35 figures - synthesized speech, plus one real recording for comparison")
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
    fig_real_vs_synth(wave)
    fig_single_spectrum()
    fig_whisper_pipeline()
    fig_three_audio_designs()
    fig_speech_vs_audio()
    log.info("L35 figures done")


if __name__ == "__main__":
    main()
