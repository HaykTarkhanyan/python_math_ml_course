"""Shared audio synthesis and signal processing for ch13 (Audio-Language Models).

Every waveform used to TEACH is synthesized here with a source-filter formant model, so the
chapter needs no `librosa` and no `soundfile` - `scipy.signal` is enough.

Updated 2026-08-08: `load_real_panir()` reads one real recording, used only to let the deck put
real speech beside the synthetic signal. The chapter's original "no audio file is read" rule is
relaxed to exactly that one file; everything else is still generated.

The synthesized speech is LABELLED AS SYNTHETIC on every slide that shows it. Real speech is
messier (jitter, shimmer, coarticulation, room), but the structure the slides point at -
formants, a nasal dip, a burst, a trill - is genuinely there in this signal.

Everything downstream of `stft_mag` is written from scratch rather than called from a library,
because L35 spends six frames deriving exactly these steps and a black-box call would defeat it.

Run nothing here directly; imported by l35_hearing_figs.py and l36_speaking_figs.py.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

SEED = 509
SR = 16_000          # speech convention; Mimi runs at 24 kHz, EnCodec/MusicGen at 32 kHz
WIN_MS = 25.0        # 25 ms: speech is roughly stationary this long, and it resolves pitch
HOP_MS = 10.0
N_FFT = int(SR * WIN_MS / 1000)      # 400 samples -> 201 rfft bins
HOP = int(SR * HOP_MS / 1000)        # 160 samples -> 100 frames/sec
N_MELS = 80                          # Whisper tiny..large-v2; large-v3 uses 128

REPO_ROOT = Path(__file__).resolve().parents[3]

# Formant tables (F1, F2, F3) in Hz, adult-male-ish reference values.
VOWELS = {
    "a": (730, 1090, 2440),
    "i": (270, 2290, 3010),
    "e": (530, 1840, 2480),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
    "schwa": (490, 1350, 1690),
}


def build_logger(name):
    """Console + file logging, per the repo rule. Log dir is created; FileHandler will not."""
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------- source-filter synthesis

def _resonator(x, freq, bw, sr=SR):
    """One two-pole resonator - the filter half of the source-filter model.

    A formant is a resonance of the vocal tract, and a resonance is a pole pair. Placing the
    poles at radius r = exp(-pi*B/sr) and angle 2*pi*F/sr puts a peak at F with bandwidth B.
    """
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    gain = (1 - 2 * r * np.cos(theta) + r * r)   # unit gain at DC, so formants do not stack up
    return lfilter([gain], a, x)


def _glottal_source(n, f0, sr=SR, rng=None):
    """Voiced excitation: a pulse train at F0 with slight jitter, tilted -12 dB/octave.

    The tilt matters visually - without it the synthesized spectrogram has flat harmonics all
    the way up and looks nothing like speech in the figure L35 asks students to read.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    src = np.zeros(n)
    period = sr / f0
    pos = 0.0
    while pos < n:
        idx = int(pos)
        if idx < n:
            src[idx] = 1.0
        pos += period * (1.0 + 0.01 * rng.standard_normal())   # jitter
    # two one-pole low-passes ~= -12 dB/octave spectral tilt of the glottal pulse
    src = lfilter([1.0], [1.0, -0.97], src)
    return lfilter([1.0], [1.0, -0.97], src)


def _segment(kind, dur_s, f0, formants, sr=SR, rng=None, amp=1.0):
    """One phone. `kind` selects the excitation; `formants` shapes it."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = int(dur_s * sr)
    if n <= 0:
        raise ValueError(f"segment duration {dur_s}s gives {n} samples")

    if kind == "voiced":
        src = _glottal_source(n, f0, sr, rng)
    elif kind == "noise":
        src = rng.standard_normal(n) * 0.5
    elif kind == "silence":
        return np.zeros(n)
    else:
        raise ValueError(f"unknown segment kind {kind!r}")

    out = np.zeros(n)
    for f, bw in formants:
        out += _resonator(src, f, bw, sr)
    # Lip radiation. Glottal *flow* is one-sided (air only blows outward), but what a microphone
    # records is pressure, which is roughly its derivative - so real speech is zero-mean. Without
    # this the waveform figure sits entirely above zero and looks nothing like speech.
    out = lfilter([1.0, -0.98], [1.0], out)
    # 5 ms raised-cosine edges, or every phone boundary becomes a broadband click
    edge = max(1, int(0.005 * sr))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
    out[:edge] *= ramp
    out[-edge:] *= ramp[::-1]
    return amp * out


def synth_panir(f0=120.0, sr=SR, seed=SEED):
    """The word ՊԱՆԻՐ (/p a n i r/), synthesized. Returns (waveform, segment boundaries).

    Boundaries are returned so L35 frame 15 can annotate the spectrogram with what each region
    is - that frame is the one that turns the spectrogram from a coloured rectangle into
    something a student can read.
    """
    rng = np.random.default_rng(seed)
    parts, marks, t = [], [], 0.0

    def add(label, wave):
        nonlocal t
        parts.append(wave)
        marks.append((label, t, t + len(wave) / sr))
        t += len(wave) / sr

    a1, a2, a3 = VOWELS["a"]
    i1, i2, i3 = VOWELS["i"]
    r1, r2, r3 = VOWELS["schwa"]

    # Պ /p/ - a voiceless stop is a silence followed by a burst. The silence is the phone.
    add("Պ closure", _segment("silence", 0.06, f0, [], sr, rng))
    add("Պ burst", _segment("noise", 0.012, f0, [(900, 400), (2200, 500)], sr, rng, amp=0.35))
    # Ա /a/ - low F1, high-ish F2: the two formants sit far apart and are easy to point at
    add("Ա", _segment("voiced", 0.20, f0, [(a1, 70), (a2, 90), (a3, 140)], sr, rng))
    # Ն /n/ - nasal murmur: energy collapses to a low resonance, higher formants damped hard
    add("Ն", _segment("voiced", 0.10, f0, [(250, 60), (1000, 250), (2400, 350)], sr, rng, amp=0.35))
    # Ի /i/ - the F1/F2 split is the mirror image of Ա, which is the point of choosing it
    add("Ի", _segment("voiced", 0.18, f0, [(i1, 60), (i2, 100), (i3, 150)], sr, rng))
    # Ր /r/ - Armenian ր is a tap/trill: a vowel-like sound amplitude-modulated at ~25 Hz
    r = _segment("voiced", 0.20, f0, [(r1, 80), (r2, 110), (r3, 160)], sr, rng)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 25 * np.arange(len(r)) / sr)
    add("Ր", r * mod)
    add("silence", _segment("silence", 0.05, f0, [], sr, rng))

    wave = np.concatenate(parts)
    peak = np.abs(wave).max()
    if peak == 0:
        raise RuntimeError("synthesized ՊԱՆԻՐ is all zeros - synthesis is broken")
    return wave / peak, marks


def load_real_panir(sr=SR):
    """A real human saying PANIR, 1.28 s at 16 kHz.

    Downloaded 2026-08-08 from Wikimedia Commons (Hy-panir.ogg) and converted with ffmpeg to
    16 kHz mono. It exists so the deck can put the synthesized signal next to real speech and
    let students see exactly how the two differ - the synthetic one is labelled as synthetic
    throughout, and this is what makes that label checkable rather than a disclaimer.
    """
    from scipy.io import wavfile
    path = Path(__file__).resolve().parents[1] / "fig" / "img" / "panir_real.wav"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - re-download and convert with ffmpeg")
    got, x = wavfile.read(path)
    if got != sr:
        raise ValueError(f"{path} is {got} Hz, expected {sr} - reconvert with ffmpeg -ar {sr}")
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    if peak == 0:
        raise RuntimeError(f"{path} is silent")
    return x / peak


def synth_corpus(seconds=60.0, sr=SR, seed=SEED):
    """A varied consonant-vowel corpus, for fitting quantizer codebooks.

    L36's measurement needs far more than one word. A 2 s utterance is only ~200 frames, and
    k-means with K above ~32 on 200 points fits noise and reports training-set reconstruction.
    60 s gives ~6,000 frames, which is enough for K=64 to mean something.
    """
    rng = np.random.default_rng(seed)
    names = list(VOWELS)
    parts, total = [], 0.0
    while total < seconds:
        f0 = rng.uniform(90, 220)                      # varied speakers
        if rng.random() < 0.35:                        # a consonantal onset, sometimes
            parts.append(_segment("silence", rng.uniform(0.03, 0.07), f0, [], sr, rng))
            parts.append(_segment("noise", rng.uniform(0.01, 0.03), f0,
                                  [(rng.uniform(700, 3000), 400)], sr, rng, amp=0.3))
        v = VOWELS[names[rng.integers(len(names))]]
        jitter = rng.uniform(0.9, 1.1, size=3)         # vowels vary between speakers
        f1, f2, f3 = np.array(v) * jitter
        parts.append(_segment("voiced", rng.uniform(0.08, 0.22), f0,
                              [(f1, 70), (f2, 100), (f3, 150)], sr, rng))
        total = sum(len(p) for p in parts) / sr
    wave = np.concatenate(parts)[: int(seconds * sr)]
    return wave / np.abs(wave).max()


# ---------------------------------------------------------------- spectral front end

def frame_signal(x, n_fft=N_FFT, hop=HOP):
    """Slice into overlapping frames. This is the 160x sequence-length cut, done by hand."""
    if len(x) < n_fft:
        raise ValueError(f"signal of {len(x)} samples is shorter than one {n_fft}-sample frame")
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def stft_mag(x, n_fft=N_FFT, hop=HOP, return_complex=False):
    """Windowed short-time Fourier transform. Returns (freq_bins, frames)."""
    frames = frame_signal(x, n_fft, hop) * np.hanning(n_fft)[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1).T
    return spec if return_complex else np.abs(spec)


def istft(spec, n_fft=N_FFT, hop=HOP, length=None):
    """Overlap-add inverse. Only needed to prove a point in L35 frame 12 (phase)."""
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * np.hanning(n_fft)[None, :]
    n = (frames.shape[0] - 1) * hop + n_fft
    out, norm = np.zeros(n), np.zeros(n)
    win_sq = np.hanning(n_fft) ** 2
    for i in range(frames.shape[0]):
        out[i * hop: i * hop + n_fft] += frames[i]
        norm[i * hop: i * hop + n_fft] += win_sq
    out = out / np.maximum(norm, 1e-8)
    return out[:length] if length else out


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Triangular mel filters, built from scratch. Returns (n_mels, n_fft//2+1).

    Written out rather than imported because L35 frame 13 shows this exact matrix as a figure.
    """
    fmax = fmax if fmax is not None else sr / 2
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (bins - lo) / max(mid - lo, 1e-9)
        falling = (hi - bins) / max(hi - mid, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))
    if not np.any(fb.sum(axis=1) > 0):
        raise RuntimeError("mel filterbank is empty - check n_fft/n_mels/sr")
    return fb, bins, edges


def log_mel(x, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """The finished Whisper-style front end: |STFT|^2 -> mel -> log. Returns (n_mels, frames)."""
    power = stft_mag(x, n_fft, hop) ** 2
    fb, _, _ = mel_filterbank(sr, n_fft, n_mels)
    return np.log10(np.maximum(fb @ power, 1e-10))


# ---------------------------------------------------------------- residual quantization

def _nearest(x, cb):
    """Index of the nearest row of `cb` for each row of `x`.

    Expanded as |x|^2 - 2 x.cb^T + |cb|^2 rather than the literal (x[:,None,:] - cb)**2, which
    would allocate n*k*d floats - 245 MB at n=6000, k=64, d=80, per iteration.
    """
    d = (x * x).sum(1)[:, None] - 2.0 * x @ cb.T + (cb * cb).sum(1)[None, :]
    return d.argmin(axis=1)


def kmeans_codebook(x, k, seed=SEED, iters=40):
    """Plain Lloyd k-means. Kept here rather than imported so the level-by-level loop in L36
    reads as the same three lines the slide shows."""
    if len(x) < k:
        raise ValueError(f"{len(x)} frames cannot fit a codebook of {k} - fitting noise")
    rng = np.random.default_rng(seed)
    cb = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = _nearest(x, cb)
        for j in range(k):
            hit = assign == j
            if hit.any():
                cb[j] = x[hit].mean(axis=0)
    return cb, assign


def quantize(x, cb):
    """Nearest codebook entry for each row. Returns (indices, quantized vectors)."""
    idx = _nearest(x, cb)
    return idx, cb[idx]


def residual_vq(x, n_levels, k, seed=SEED):
    """Residual vector quantization: quantize, keep what is left over, quantize that too.

    Returns (codebooks, indices per level, reconstruction after each level). This is the whole
    idea of the codec in ten lines - the real thing differs by training the codebooks jointly
    with a decoder under a spectral and adversarial loss, not by being more complicated here.
    """
    residual = x.copy()
    approx = np.zeros_like(x)
    codebooks, all_idx, recons = [], [], []
    for level in range(n_levels):
        cb, _ = kmeans_codebook(residual, k, seed=seed + level)
        idx, q = quantize(residual, cb)
        approx = approx + q
        residual = residual - q
        codebooks.append(cb)
        all_idx.append(idx)
        recons.append(approx.copy())
    return codebooks, all_idx, recons
