# Chapter plan — Audio-Language Models (L35, L36)

**Status:** BUILT 2026-08-07. L35 = 32 frames (34 pages, two `\pause` reveals), L36 = 32 frames.
Both compiled twice, cleaned, and checked for silent clipping. Registered in `_quarto.yml`.

## Student review (2026-08-07, one Sonnet agent per deck, PNGs only)

Both reviewers could explain the material back correctly, and neither lost the thread
structurally. What they caught was **rhetoric that did not match the arithmetic**, which is
exactly what a layout pass cannot see:

| Finding | Verdict | Fix |
|---|---|---|
| L36: "levels 2-8 add 6.7 dB" but the chart only shows 6.4 | **Claim CORRECT, figure at fault.** Levels 2-8 is seven levels summing to 6.72; the reviewer could not read the 8th bar (0.45 dB) | Log axis, 2-decimal rotated labels, and the totals printed in the panel title so a student can check it |
| L36: caveat says "read the shape, not the numbers" directly above a box **bolding** those numbers, then the recap quotes them caveat-free | **REAL, and the sharpest finding.** It read as hedging that did not survive contact with the deck | Takeaway and recap now lead with the **ratio** (level 1 buys ~2.5x what levels 2-8 add), which is the claim that actually survives the stand-in caveat |
| L35: "roughly a fifth of the latency" | **WRONG.** 200/800 is a quarter | "about a quarter of a tuned cascade, a tenth of a naive one" |
| L35: recap "loses about a second" | **WRONG.** 600 ms vs tuned, 2,100 ms vs naive | "between 0.6 and 2 seconds" |
| L35: frame 20 says "hold on to that ratio (10x), we will meet it again", frame 30 then shows 13x | **REAL.** Different systems, and the deck promised the same number | Frame 30 now says explicitly that one is counted from encoder frames and the other read off a price list, and that the agreement in order of magnitude is the point |
| L35: "100 x 80 = 8,000" uses the 80 mel bins three frames before they exist | **REAL - the one place the reviewer lost the thread** | Arithmetic moved to the mel frame where 80 is defined; the framing frame now makes the point qualitatively |
| L35: latency chart bar says 200 ms, caption cites Qwen3-Omni's 234 ms | **REAL ambiguity** | Bar relabelled "native (Moshi, measured)"; title names which number belongs to which system |
| L36: Moshi's 160 ms appears with no derivation | **REAL.** Checked the paper: it reports 160 ms and does **not** decompose it | Slide gives the 80 ms frame for scale and says explicitly not to quote a decomposition |
| Mimi / EnCodec / MusicGen / vocoder / w2v-BERT / "full duplex" used without a gloss | **REAL** | One-clause glosses added. The WavLM gloss was a **self-inflicted regression** - it was trimmed while fixing a clipped frame earlier the same day |
| L35 latency chart: tuned-cascade segment labels unreadable | **REAL** | Labels moved to clear space beside the row |

Not changed: both reviewers rated the numeric through-line and the figures on frames 20, 27
(L36) and 6, 10, 13, 14, 21 (L35) as clear. No invented problems in either report.

**Build note - Beamer clipped three frames with zero LaTeX warnings.** Both decks logged
`Overfull \vbox` = 0 while a callout box on L35 frame 12, another on L35 frame 14, and one on
L36 frame 16 were losing their last lines outright. The log is not a clipping check. What
caught it was rendering every page and looking for ink in the bottom band of the slide, outside
the strip where the page number sits - worth keeping as a habit for any deck in this repo.

## Review corrections applied (2026-08-07)

A review pass fact-checked every number against primary sources. Five claims in the first draft
were **wrong**, three of them load-bearing. All verified below before the fix was accepted.

| First draft said | Verdict | Correct, and the source |
|---|---|---|
| Whisper: "two convolutions of stride 2 -> 1,500 vectors" | **WRONG, self-contradictory** | Only `conv2` has stride 2 (`conv1` is stride 1). Two stride-2 convs would give 750, contradicting the draft's own 3000 -> 1500. Verified in [`openai/whisper/model.py`](https://github.com/openai/whisper/blob/main/whisper/model.py). |
| Whisper "80 mel bins" while also citing large-v3 | **INCONSISTENT** | 80 for tiny..large-v2; **128 for large-v3**. Say which, and note the change. |
| Mimi: "128x downsampling ... 1,280x cut" | **WRONG** | The 128x is Kyutai's **toy** codec in the explainer, not Mimi. Mimi is 24 kHz -> 12.5 Hz = **1920x** (strides 4,5,6,8 plus a final stride-2 conv). Verified in the [Moshi paper](https://arxiv.org/html/2410.00037v2). |
| Mimi "up to 32 levels" | **CONTRADICTED ITSELF** | The checkpoint ships 32; **Moshi runs Q=8**, codebook 2048, **1.1 kbps** at 12.5 Hz (8 x 11 bits x 12.5 = 1100 bps, arithmetic checks). |
| "Mimi distills WavLM into **level 1**" | **WRONG - architecture, not just loss** | They tried exactly that and it failed: *"While distillation significantly improves the phonetic discriminability of the first quantizer (as measured by ABX), it also affects audio quality negatively."* The fix is a **split**: *"Rather than a single RVQ with 8 levels, we distill semantic information into a plain VQ and apply an RVQ with 7 levels in parallel. We sum their outputs."* Verified in the Moshi paper. |
| "one codebook would need **billions** of entries" | **WRONG by ~17 orders of magnitude** | 2048^8 = 2^88 ~ **3x10^26**. "Billions" makes the impossible sound merely awkward. |
| Latency "300-500 ms floor for native" | **TOO CONSERVATIVE, and unfair to cascades** | Moshi: **160 ms theoretical, 200 ms in practice**; Qwen3-Omni: **234 ms**. And 1.5-3 s describes an *un-optimised* cascade. Present three tiers, or the honest-counter-argument frame undercuts the chart. |
| "~3 text tokens/sec" and "75 tokens per 30 s" | **INCONSISTENT** | 3 x 30 = 90. Use **2.5/sec** everywhere. |
| MusicGen delay = "T+8" | **IMPRECISE** | Flatten costs **K·T**, delay costs **T+K-1**, and MusicGen uses **K=4**, so T+3. |
| Qwen2-Audio "fed straight in" | **IMPRECISE** | There *is* a learned adapter: a **single linear** projector (LLaVA-1.0's design, not 1.5's MLP). |

**All 13 author+year citations checked out as written.** Two disambiguations needed: WavLM (Chen
et al., 2022) and BEATs (Chen et al., 2023) share a first author (Sanyuan Chen), and the
RQ-Transformer is **Lee et al., 2022**, adopted by Moshi rather than invented by it.

### The `semantic_vs_acoustic` measurement is CUT

The planned experiment - same word in two synthesized "voices", measuring per-level token
agreement - **cannot distinguish its hypothesis from its confound**. The two voices differ only
in fundamental frequency, and at 16 kHz the low mel filters are narrower than the harmonic
spacing, so changing F0 moves those channels directly. Level-1 agreement would move for a
low-level spectral reason unrelated to semantics, and the parameters could be tuned to produce
either result. "Expected negative result" is not a plan for a figure that has not been run.

Replaced with published evidence, which is far stronger and is what the slide should have said:

- **SpeechTokenizer (Zhang et al., 2024)**: resynthesis from EnCodec's **RVQ-1 alone** retains
  **0.92 speaker similarity** while content collapses. Exactly backwards from "semantic".
- **Mimi's own ablation** (above): distilling into RVQ-1 hurt audio quality, forcing the split.

New figure `split_rvq` draws the failed design next to the shipped one.

### Two pedagogical holes the review found, both accepted

1. **Phase is never mentioned.** The draft went FFT -> magnitude -> mel with no statement that
   the transform returns complex numbers and we **discard phase**. That omission breaks three
   later frames: why vocoders exist at all, why codecs work on waveforms rather than mel, and
   the caveat the `vq_residual_stages` figure needs. New frame 12, with a figure.
2. **Codecs eat waveforms, not mel - never stated.** L35 spends six frames building mel, then
   L36 says "encoder -> quantizer -> decoder" with no reset, so every student concludes codecs
   quantize mel frames. They do not. Stated explicitly on frame 43, and it doubles as the
   caveat for the chapter's own measurement.



**Source:** written from scratch, from a web sweep on 2026-08-07. No existing deck in this repo
covers it (`grep` confirms: spectrogram, mel, codec, RVQ, Whisper appear nowhere in `ml/`).

## Instructor decisions (2026-08-07)

1. **Two decks** - L35 "How a model hears", L36 "How a model speaks". Exact parallel to
   ch12's L33/L34.
2. **Full signal-processing primer (~6 frames).** Students have had no signal-processing
   course, so the spectrogram is built from scratch rather than assumed: sampling, framing,
   Fast Fourier Transform (FFT), the mel scale, log compression. Everything downstream
   (frame rates, token rates) is arithmetic on those numbers, so the primer pays for itself.
3. **Synthesized audio, not a recording.** All figures run on a numpy formant synthesizer -
   no `librosa`, no `soundfile`, no new dependency (`scipy` already ships `stft`). The
   synthetic waveform is **labelled as synthetic on the slide**; a real recording is messier
   but has the same structure.
4. **Intuition-first**, like `ch11_rl` and `ch12_vlm`. Equations shown and explained, not
   derived. *Deliberate deviation from `ml/SLIDE_STYLE.md`, which asks for full step-by-step
   derivations. Recorded here so the next person does not "fix" it.* It bites in exactly two
   places: the residual-quantization recursion and the straight-through estimator - and the
   straight-through estimator was already waved through in L34, so this stays consistent.

## Placement

- New chapter folder `ml/ch13_audio/`, decks **L35** and **L36** (the L-sequence ends at L34, VLM).
- Sidebar: after `ml/ch12_vlm/`. Registering it touches `_quarto.yml`, the one shared file -
  do that in a single small commit at the end.

---

## Why this chapter exists

ch12 answered "how does a chat model see the photo I pasted". The identical question about
voice is now more visible to a student than the image one: they talk to a phone every day, and
the thing that changed in 2024-2026 is that it answers **in under half a second, in a voice,
and notices they sounded annoyed**. Nothing in this course explains that.

It is also the cheapest chapter in the course to justify, because ch12 did most of the work:

| Prerequisite | What this chapter uses it for |
|---|---|
| ch8 autoencoders | a codec is an autoencoder with a quantized bottleneck |
| ch8b GANs | codecs are trained adversarially, not on mean squared error |
| ch9 attention | the encoder is a transformer; quadratic cost is the whole constraint |
| ch12 L33 | encoder -> projector -> LLM is literally the same recipe |
| ch12 L34 | codebooks, quantization cost, straight-through estimator |

## The through-line: ՊԱՆԻՐ, now out loud

ch10's diffusion model **wrote** ՊԱՆԻՐ. ch12 **tokenized** it. This chapter **says** it.

The word is synthesized with a formant model in numpy (plosive burst, `/a/`, nasal murmur,
`/i/`, trilled `/r/`), then pushed through the entire pipeline both directions: waveform ->
frames -> mel -> residual quantization -> tokens -> back. Every figure in both decks runs on
that same two-second signal.

It also gives the chapter **its own measurement**, in the house style, mirroring L34's
codebook study: *what does each extra residual-quantization level actually buy?* And a second,
honest one: *is the first level really "semantic"?* (Prediction: **no**, in a plain residual
quantizer - which is precisely why Mimi has to distill WavLM into it. A negative result that
explains why a real system is built the way it is.)

---

## L35 — How a model hears (~32 frames)

### Cold open
1. Title.
2. **A voice message, and an answer that noticed the tone.** You send audio. It replies, and it
   picked up that you were annoyed. Nothing in this course does that. Bridge from L33: same
   bridge problem, harder signal.
3. **Two questions that sound the same and are not** - how a model *hears* (settled, one recipe)
   vs how it *speaks* (not settled, L36). Mirrors L33 frame 2 deliberately.
4. Outline.

### Section 1 — A second of sound is 16,000 numbers
5. `[plain]` transition: *Sound is not a sequence either.*
6. **What a waveform is.** *Figure: `waveform_zoom`* - the same signal at 2 s, 100 ms, 5 ms.
   At the deepest zoom you see individual samples. 16 kHz, and why (speech energy sits below
   8 kHz; Nyquist). **One clause the review demanded:** 16 kHz is a *speech* convention - Mimi
   runs at 24 kHz, EnCodec and MusicGen at 32 kHz - otherwise frame 21's ladder silently mixes
   rates derived from different sample rates.
7. **The problem, as arithmetic.** 16,000 numbers per second against roughly **2.5** text tokens
   per second. A 30-second clip is 480,000 steps, and attention is quadratic. Dead on arrival.
8. **Predict-first:** could you just feed raw samples to a transformer? Reveal: no - and not only
   for cost. Sample-level models (WaveNet, van den Oord et al., 2016) babble; nothing holds
   structure across 16,000 steps of context.

### Section 2 — Turning sound into a picture
9. `[plain]` transition: *Make it a picture with time on one axis.*
10. **Framing, and why 25 ms** (merged - it was one sentence, not a frame). 25 ms window, 10 ms
    hop; short enough that speech is roughly stationary, long enough to resolve pitch.
    *Figure: `framing_window`.* 100 frames per second = a **160x shorter sequence**. Say
    "sequence", not "compression": the *data* only halves (16,000/s -> 100 x 80 = 8,000/s).
    Attention cares about length, which is the whole point.
11. **A spectrum per frame.** FFT, and what you can read off it: pitch harmonics, and formants
    (the resonances that distinguish `/a/` from `/i/`).
12. **NEW - Magnitude and phase, and why we throw one away.** The transform returns *complex*
    numbers; we keep magnitude and discard phase. *Figure: `phase_discard`* - same magnitude
    spectrogram, correct phase vs random phase, and the second one is garbage. **This frame pays
    for itself three times over:** it is why vocoders exist, why L36's codecs work on waveforms
    and not on mel, and the caveat the chapter's own measurement needs.
13. **The mel scale.** *Figure: `mel_filterbank`.* Hearing is roughly logarithmic in frequency, so
    the filters are narrow low and wide high. 201 FFT bins collapse to 80 mel bins.
14. **Log, and done.** *Figure: `spectrogram_stages`* - four panels, waveform -> power spectrogram
    -> mel -> log-mel. "This is exactly what Whisper is fed."
15. **Read the picture.** On our own ՊԱՆԻՐ spectrogram: the burst of Պ, the formants of Ա, the
    nasal dip of Ն, the high second formant of Ի, the trill of Ր. Makes the front end concrete
    instead of a black box.

### Section 3 — Bolting ears onto a language model
16. `[plain]` transition: *You have seen this recipe before.*
17. **The three-part recipe**, drawn once: encoder -> projector -> LLM. Point out explicitly that
    this is L33 frame 16 with a different encoder. The chapter's cheapest win.
18. **Whisper (Radford et al., 2022)** as the standard encoder. 30-second chunks, **80** mel bins
    (**128** in large-v3 - say which, the two are not interchangeable), 3,000 frames, then **two
    convolutions, the second with stride 2** -> **1,500 vectors = 50 Hz**. It is an
    encoder-decoder, and almost everyone uses only the encoder.
19. **NEW - Self-supervised speech models**, one frame, because three of them are load-bearing
    later and the draft named them as if known. Masked prediction on speech: WavLM
    (Chen et al., 2022), w2v-BERT (Chung et al., 2021), BEATs (Chen et al., 2023). Continuous
    frames -> k-means -> discrete "units". Without this, "semantic tokens" in L36 is magic.
20. **Worked numbers, by hand.** 30 s -> 480,000 samples -> 3,000 frames -> 1,500 encoder vectors
    -> 750 after a stride-2 pool. The transcript of the same 30 s is about **75** text tokens.
    A 10x gap, and that gap is the entire economics of audio in an LLM.
21. *Figure: `token_rate_ladder`* - representations per second on a log axis: 16,000 samples,
    100 frames, 50 Whisper, 25 pooled, 12.5 (labelled "the lowest rate anyone ships - L36"),
    2.5 text. The single most useful figure in L35.
22. **Design A - a projector (Qwen2-Audio, Chu et al., 2024).** Whisper-large-v3 encoder,
    stride-2 pooled to ~25 Hz, then a **single linear** projector - LLaVA-1.0's design, not
    1.5's two-layer MLP. The audio LLaVA.
23. **Design B - a resampler (SALMONN, Tang et al., 2024).** Whisper-large-v2 **plus BEATs**, a
    window-level Q-Former compressing 30 s to **88 tokens**, i.e. one token per 340 ms,
    regardless of what is in it.
24. **Design C - native from pretraining (Qwen3-Omni, Qwen Team, 2025).** Audio encoder trained
    with the model rather than bolted on: AuT, 20M hours, 12.5 Hz output, Thinker/Talker
    mixture-of-experts, 234 ms theoretical first packet.
25. **Why SALMONN carries two encoders.** A speech encoder is trained to discard everything that
    is not words - a dog bark, a door, music. If you want an *audio* model and not a *speech*
    model you need a general-audio encoder alongside it. Non-obvious, and it explains the whole
    "audio understanding" benchmark gap.

### Section 4 — What the cascade throws away
26. `[plain]` transition: *The obvious pipeline, and its bill.*
27. **The cascade, drawn once and annotated** (merged, per review): speech recognition -> LLM ->
    speech synthesis, with what falls out at each arrow - speaker identity, emotion, sarcasm,
    hesitation, pace, overlap, background sound, and whether the sound was speech at all.
    Concrete: *"say that again, but angrier"* is unanswerable from a transcript. Plus what the
    cascade does about turn-taking: a **voice-activity detector (VAD)** decides you stopped
    talking, which is exactly why interrupting it feels wrong.
28. **Predict-first:** how much latency does the cascade actually cost? Reveal:
    *Figure: `latency_budget`* - **three** tiers, not two: naive cascade 1.5-3 s, tuned streaming
    cascade ~0.5-1 s, native ~200 ms (Moshi 160 ms theoretical / 200 ms measured; Qwen3-Omni
    234 ms). Two tiers would have overstated the case and undercut frame 29.
29. **The honest counter-argument.** Cascades still dominate production because you can swap
    vendors, log the transcript, and audit it. Native is a handful of vendors and a black box.
    State the 2026 position, do not sell.
30. **The token bill.** *Figure: `audio_token_budget`.* Gemini charges 32 tokens per second of
    audio; one minute is 1,920 tokens against ~150 for its own transcript. Half a frame - the
    review correctly flagged it as a repeat of L33's token-budget slide.
31. **Where audio LLMs still fail:** long recordings, counting and separating speakers, exact
    timestamps, music, and low-resource languages - **Armenian among them**. And the documented
    one, the counterpart to ch12's hallucination frame: **Whisper hallucinates on silence**
    (Koenecke et al., 2024) - ~1% of transcriptions contain invented phrases, **38% of those
    carry explicit harms**, concentrated on long non-vocal stretches.
32. Recap + `Next:` box -> L36.

---

## L36 — How a model speaks (~34 frames)

### Cold open
33. Title.
34. **It answered out loud, in about 200 ms, in a voice.** And it did not call a speech-synthesis API.
35. **The contradiction** (mirrors L34 frame 27): a language model emits discrete tokens from a
    finite vocabulary; sound is continuous and arrives 16,000 numbers per second. Something has
    to give, and what gives is that we invent a vocabulary for sound.
36. Outline.

### Section 5 — A vocabulary for sound
37. `[plain]` transition: *What if a sound were made of words?*
38. **The naive answer, and why it fails.** Quantize each sample to 256 levels - this is literally
    what WaveNet did (mu-law). The vocabulary is fine; the sequence length is hopeless.
39. **Vector quantization**, one frame. Same move as L34: snap each encoder frame to the nearest
    entry of a learned codebook. The audio becomes a grid of integers. Signposted back to ch8.
40. **Why one codebook cannot work.** Matching Mimi's quality with a single codebook needs
    2048^8 ~ **3x10^26** entries. (Draft said "billions" - wrong by seventeen orders of
    magnitude, and it killed the punchline.)
41. **Residual vector quantization (RVQ)** - the idea that made audio tokenizable. Quantize; take
    what is left over; quantize that too; repeat. Level 1 gets the gist, level 8 the polish.
    The recursion is shown and explained, not derived (decision 4).
42. *Figure: `vq_residual_stages`* - **the chapter's own measurement**, with the stand-in caveat
    printed on the slide (see "Measurement caveats" below). Residual quantization on 60 s of
    synthesized frames: reconstruction after 1, 2, 4 and 8 levels, plus the per-level dB drop.
43. **The codec**: encoder -> residual quantizer -> decoder, trained end to end - and **on raw
    waveforms, not on mel** (the review's second pedagogical hole: six frames of mel in L35 make
    every student assume otherwise). SoundStream (Zeghidour et al., 2021), EnCodec (Défossez
    et al., 2022), Mimi (Défossez et al., 2024). Mimi: 24 kHz -> **12.5 Hz = 1920x**, **Q=8**
    codebooks of 2048, **1.1 kbps**. (The checkpoint ships 32 quantizers; Moshi runs 8.)
44. **Straight-through estimator**, one frame, flagged not derived: rounding has zero gradient, so
    pretend the backward pass went through the identity. Signpost to L34, do not re-explain.
45. **Why the loss is not mean squared error.** Multi-scale spectral loss plus adversarial
    discriminators, because ears do not measure squared error on waveforms. Connects to ch8b.
46. **One sentence on the alternatives**, so the chapter stays consistent with ch12: finite scalar
    and lookup-free quantization are the audio analogue of the MAGVIT-2 trick from L34, and
    single-codebook low-frame-rate tokenizers exist. A student who remembers ch12 will ask.

### Section 6 — Semantic tokens and acoustic tokens
47. `[plain]` transition: *What was said, and how it sounded.*
48. **The design goal** (stated as a goal, not a property): level 1 should carry *what was said*,
    the rest *who said it and how*. Kyutai's demonstration: swap the voice, the words survive.
49. **It does not happen for free** - and here is the measurement, published rather than ours.
    **SpeechTokenizer (Zhang et al., 2024)** probed EnCodec: resynthesis from **RVQ-1 alone**
    retains **0.92 speaker similarity** while the content collapses. The first level of a plain
    residual quantizer keeps the *voice* almost perfectly - exactly backwards from "semantic".
50. **Mimi's fix, and the failed attempt first.** Distilling WavLM into RVQ level 1 *did* improve
    phonetic discriminability - and **hurt audio quality**. So Mimi splits: a plain semantic VQ
    running **in parallel** with a 7-level acoustic RVQ, outputs summed, which removes the
    constraint that acoustic detail live in the semantic quantizer's residual.
    *Figure: `split_rvq`* - the failed design next to the shipped one. **This is a better frame
    than the draft's: they tried the obvious thing, measured it, and restructured.**
51. **AudioLM (Borsos et al., 2022)**: three stages - semantic tokens from w2v-BERT, then coarse
    acoustic, then fine acoustic, each conditioned on the last. Long-term structure and local
    fidelity are different jobs and get different models.

### Section 7 — Generating sound left to right
52. `[plain]` transition: *Next-token prediction, on a sound.*
53. **The layout problem.** Residual quantization gives 8 tokens per timestep; a language model
    emits one at a time. How do you lay a 2D grid on a 1D tape?
54. *Figure: `codebook_patterns`* - flatten (best quality, **K·T** steps), parallel (fastest,
    **T** steps, wrong independence assumption), delay (**T+K-1**; MusicGen uses **K=4**, so T+3).
55. **VALL-E (Wang et al., 2023)**: speech synthesis as next-token prediction on codec tokens. A
    **3-second** clip of a voice is just the *prompt*, so zero-shot voice cloning falls out with
    no fine-tuning at all. State the deepfake consequence plainly, once, without moralising -
    and state the technical mitigation in the same breath, since omitting it is the less honest
    half: audio watermarking (AudioSeal; Moshi ships one).
56. **NEW - the other lineage: non-autoregressive speech synthesis.** Flow matching and diffusion
    (F5-TTS, MaskGCT, CosyVoice 2, Voicebox) went the opposite way from VALL-E. **This is the
    chapter's best tie-back:** it is the *same* autoregressive-vs-diffusion fork the students
    just saw for images in L34 section 7, recurring verbatim for audio. The review called its
    absence the single biggest gap.
57. **Back to a waveform - two lineages, not one.** Mel + a separate vocoder (the Tacotron /
    HiFi-GAN era) versus codec tokens + the codec's own decoder (now). The draft's "a separate
    vocoder still shows up" was a misconception waiting to happen: in a codec system the decoder
    *is* the vocoder.
58. **Worked trace, by hand.** One second of audio through the entire pipeline in both directions,
    with every number named. The frame that ties L35 and L36 together.

### Section 8 — Talking at the same time
59. `[plain]` transition: *A conversation is not turn-taking.*
60. **Why full duplex is a genuinely new problem.** An image arrives all at once; a conversation
    does not. Backchannels, interruptions, overlap. A turn-based model cannot do this *by
    construction*, no matter how fast it gets - the VAD from frame 27 is a guess about when you
    finished. **This is the one idea audio has that ch12 has no counterpart for.**
61. **Moshi (Défossez et al., 2024)**: **K = 2Q+1 = 17** parallel streams at 12.5 Hz - 1 text,
    8 of its own audio, 8 of the user's. *Figure: `duplex_streams`.* The model is always
    listening and always speaking; most of the time what it speaks is silence.
62. **Inner monologue**: predict the text token for a frame *before* that frame's audio tokens.
    Keeps an LLM's linguistic quality while the output stays genuinely speech-to-speech - and
    streaming recognition and synthesis fall out of the same model for free.
63. **RQ-Transformer** (Lee et al., 2022 - **adopted** by Moshi, not invented by it): a temporal
    transformer (32 layers, 4096 wide) stepping at 12.5 Hz plus a small depth transformer
    (6 layers, 1024 wide) across the 17 streams inside one frame. Why widening the vocabulary
    to 2048^17 instead is not an option.
64. **The 2026 landscape**, as model cards (year · org · repo, per `SLIDE_STYLE.md`): Moshi,
    Qwen3-Omni, GPT-realtime, Gemini native audio. Open weights vs closed.
65. **What is still unsolved**, one honest frame: audio memory over long context, speaker
    separation inside the model, immature evaluation (**name** the benchmarks - S2SBench for
    intelligence degradation, MMAU / AIR-Bench for audio understanding - "immature" with no
    example is padding), voice deepfakes, and the **disputed** question of whether training a
    model to speak costs it text reasoning: S2SBench measures a real drop, while Qwen3-Omni
    claims no degradation at all. Present both; the disagreement is the honest 2026 state.
66. Recap + `Next:` box -> `ml/llm_training/`.

---

## Figure budget (all Python, `py_src/` -> `fig/`, run with the `ma` venv)

All twelve run on the same synthesized ՊԱՆԻՐ signal. No new dependency: `scipy.signal` provides
the short-time Fourier transform, `sklearn` the k-means used for quantization.

| Figure | Deck | What it shows | Cost |
|---|---|---|---|
| `waveform_zoom` | L35/6 | one signal at 2 s / 100 ms / 5 ms | free |
| `framing_window` | L35/10 | 25 ms windows at 10 ms hop, overlapping | free |
| `phase_discard` | L35/12 | **NEW** same magnitude, true vs random phase | free |
| `mel_filterbank` | L35/13 | triangular mel filters, warped | free |
| `spectrogram_stages` | L35/14 | waveform -> power -> mel -> log-mel | free |
| `token_rate_ladder` | L35/21 | representations per second, log axis | free |
| `latency_budget` | L35/28 | three tiers, cascade to native | free |
| `audio_token_budget` | L35/30 | audio vs its transcript vs an image | free |
| `vq_residual_stages` | L36/42 | **real** residual quantization, 1/2/4/8 levels | k-means, seconds |
| `split_rvq` | L36/50 | **replaces `semantic_vs_acoustic`** - failed vs shipped Mimi design | free |
| `codebook_patterns` | L36/54 | flatten vs parallel vs delay, as grids | free |
| `duplex_streams` | L36/61 | Moshi's 17 streams on a timeline | free |

Bar charts get `ax.bar_label`; 3+ colours use the Armenian flag palette, per `CLAUDE.md`.

**Compute:** k-means on ~6,000 80-dimensional frames, eight times. Seconds on CPU. Nothing here
approaches the ch10 diffusion runs or even ch12's. Both scripts follow the `TAG` convention in
`CONVENTIONS.md` and log to `logs/`.

### Measurement caveats (must appear on the slide, not just in this file)

`vq_residual_stages` is a **stand-in, not a codec**, and the review identified a build blocker
that is fixed here: a 2-second signal is only 200 frames, so k-means with K above ~32 would be
fitting noise and reporting *training-set* reconstruction. The script therefore synthesizes
**60 s of varied syllables** (~6,000 frames) from the same formant model. Slide text:

> Stand-in, not a codec. This is k-means residual quantization on **log-mel frames**, not on a
> learned encoder's latents. Two consequences: the error is measured on a **feature**, not on
> sound - log-mel has already discarded phase (frame 12), so these frames cannot be turned back
> into audio at all; and a real codec's codebooks are trained **jointly with the decoder** under
> a spectral and adversarial loss, so they are shaped to what the decoder needs, whereas k-means
> minimises plain Euclidean error on a representation nobody optimised. Take the **shape of the
> curve** from this figure, not the numbers.

The error decay is guaranteed by construction (each level quantizes a smaller-norm residual), so
the figure only earns its place by reporting the **per-level dB drop** and saying whether it is
roughly constant - the textbook prediction - or front-loaded.

---

## Citations to verify at build time

`SLIDE_STYLE.md` requires author + year, verified rather than remembered. To confirm before
baking in: WaveNet (van den Oord et al., 2016), Whisper (Radford et al., 2022), SoundStream
(Zeghidour et al., 2021), EnCodec (Défossez et al., 2022), AudioLM (Borsos et al., 2022),
VALL-E (Wang et al., 2023), MusicGen (Copet et al., 2023), SALMONN (Tang et al., 2024),
Qwen2-Audio (Chu et al., 2024), Moshi/Mimi (Défossez et al., 2024), WavLM (Chen et al., 2022),
BEATs (Chen et al., 2023), w2v-BERT (Chung et al., 2021).

Abbreviations that must be expanded on first use (not in the exempt set): RVQ, STFT, FFT, ASR,
TTS, VAD, AuT, BEATs, MoE. Run the mechanical check from `SLIDE_STYLE.md` before declaring done.

## Sources swept (2026-08-07)

- [Kyutai - Neural audio codecs: how to get audio into LLMs](https://kyutai.org/codec-explainer/) — the clearest explanation of residual quantization anywhere; 128x downsampling, 12.5 Hz, semantic distillation
- [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf) — 17 streams, inner monologue, RQ-Transformer
- [Qwen3-Omni Technical Report](https://arxiv.org/pdf/2509.17765) — AuT encoder, 20M hours, 12.5 Hz, Thinker/Talker
- [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/pdf/2209.03143) — semantic/coarse/fine hierarchy
- [VALL-E: Neural Codec Language Models are Zero-Shot TTS](https://arxiv.org/pdf/2301.02111) — codec tokens as a language
- [MusicGen: Simple and Controllable Music Generation](https://arxiv.org/pdf/2306.05284) — the delay pattern
- [Introducing Whisper](https://openai.com/index/whisper/) — 30 s chunks, 80-channel log-mel, 25 ms / 10 ms
- [Gemini API audio docs](https://ai.google.dev/gemini-api/docs/audio) — 32 tokens per second, 9.5 h limit
- [Cascaded vs speech-to-speech, 2026 state](https://futureagi.com/blog/cascaded-voice-ai-vs-speech-to-speech-2026/) — latency budgets and why production still cascades
