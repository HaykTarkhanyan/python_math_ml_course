# ch20 — Subliminal learning

**Status:** built 2026-08-14. `L48_subliminal_learning.tex` — 56 frames / 64 pages,
0 errors, 0 overfull vbox, both layout detectors clean, acronym check clean
(`AI`, `GPT`, `MNIST`, all exempt).

## Measured results (all from `results/subliminal_mnist.json`)

| Quantity | Value |
|---|---|
| Teacher test accuracy | **97.8%** |
| Auxiliary-head drift over 5 teacher epochs | **0.000e+00** (exactly zero, guarded) |
| Student, **shared** init | 10.0% → **20.4%** |
| Student, **different** init (control) | 11.6% → **13.7%** |
| Student on pure noise (paper's setup) | 10.0% → 14.4% |
| `cos(dtheta_T, dtheta_S)`, shared init | **0 / 200 negative**, min +0.063 |
| `cos(dtheta_T, dtheta_S)`, different init | **91 / 200 negative**, mean +0.002 |

**What did not reproduce.** The paper's headline (>50%) and its striking variant where
the student is distilled on *pure noise*. On noise our student reaches 14.4% against a
10% floor. The paper does not publish the learning rate or schedule, so this is a gap in
our setup rather than a contradiction of theirs. The deck says so on its own frame
("What reproduced here, and what did not") rather than quoting a number we did not get.

**A design error caught late, worth remembering.** The first version compared
*shared-init on MNIST inputs* against *different-init on noise*. That control is
worthless: nothing happens on noise either way, so the comparison silently varied two
things at once. Treatment and control now differ in exactly one thing, the
initialisation, and run on identical inputs and loss.

## What this chapter is

A deliberate **retelling of one video**: Welch Labs, *These Numbers Can Make AI Dangerous
[Subliminal Learning]* (33:04, 4 Sep 2025). The instructor asked for the video's content,
not a survey built around it. So the deck follows the video's arc beat for beat, and the
scope rule is: **if it is not in the video, it needs a reason to be here.**

Three things earn their place despite not being in the video, all of them corrections or
verifications rather than additions:

1. **We run the MNIST experiment ourselves** instead of quoting its result. Same claim,
   our numbers. This is the house norm (`ml/ch19_mech_interp` did the same with GPT-2)
   and it costs one CPU-minute.
2. **The different-init control**, which the video only implies. It is the falsifiable
   half of the argument and it is what makes the GPT-4.1/GPT-4o story land.
3. **The token-entanglement source is a blog post, not a paper**, and its mechanism is the
   **softmax bottleneck**. The video says neither. Citing it as a paper would be wrong.

## Source of record

`_reference_welchlabs_subliminal/` — transcript, 50 frames in timestamp order, beat map,
and the exact Fig. 8 values transcribed from frame f08. See its README.

Underlying work:

- Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks & Evans (2025),
  [arXiv:2507.14805](https://arxiv.org/abs/2507.14805).
- Zur, Loftus, Orgad, Ying, Sahin & Bau (2025), *It's Owl in the Numbers*,
  <https://owls.baulab.info/> (blog post).

## Deck

One deck, `L48_subliminal_learning.tex`. The video is ~29 minutes of content once the
sponsor read (01:45–03:30) and the poster outro (31:00+) are dropped, which is one
90-minute session with room to stop and work through the derivation.

| § | Title | Video beat | Frames |
|---|---|---|---|
| — | Cold open: the eagle experiment | 00:00–01:45 | ~5 |
| 1 | What was actually done | 03:30–04:30 | ~6 |
| 2 | What it is not — five clues | 04:30–09:10 | ~12 |
| 3 | Shrink it: MNIST | 09:10–12:40 | ~11 |
| 4 | Why it happens — the proof | 12:40–25:20 | ~14 |
| 5 | What it means | 25:20–31:00 | ~9 |

Section 4 is the one that needs the lecturer to slow down. It is eight algebraic steps and
every one of them is small; the risk is a student losing the thread at step 3 and nodding
through the rest. Each step gets its own frame with the *reason* for the step in words
above the algebra, not just the algebra.

## Figures

All Python-generated into `fig/` from `py_src/`, per `ml/SLIDE_STYLE.md`. Raw numbers land
in `results/subliminal_mnist.json` first; `make_figs.py` reads that file, so re-plotting
never re-runs the experiment.

| Figure | Source | What it shows |
|---|---|---|
| `transfer_matrix.pdf` | transcribed from paper Fig. 8 | Diagonal transfers, off-diagonal does not, GPT-4.1↔GPT-4o is the exception |
| `mnist_result.pdf` | **measured here** | Digit accuracy vs distillation step: shared-init student climbs, different-init student does not |
| `aux_head_frozen.pdf` | **measured here** | The auxiliary head receives zero gradient; the trunk beneath it moves |
| `cosine_hist.pdf` | **measured here** | cos(Δθ_T, Δθ_S) for 200 single steps: shared-init never negative, different-init centred on 0 |
| `projection_geometry.pdf` | schematic | The student update is the teacher update projected onto ∇g₀ |

Full-bleed video stills (attribution line on each): f03, f04, f08, f21, f40, f47.

## The one idea to protect

The auxiliary outputs are a **fixed random projection of a representation that is
changing**. The teacher's aux head never trains (zero gradient), so the only reason its
aux numbers move is that the trunk underneath learned something. A student with the *same*
projection can invert that; a student with a different projection cannot. Everything else
in the chapter — the cross-model matrix, the proof, the GPT-4.1/4o exception — is a
consequence of that one sentence, and it should be said early and repeated.

## Open questions

1. Homework: not yet specified. Natural task is "change the number of auxiliary outputs
   and find where the effect dies", which is cheap and uses the script already written.
2. Cross-link from `ml/ch19_mech_interp` (L47 argues for monitoring; this is a case where
   monitoring the text cannot work) and from wherever distillation is taught.
3. Placement in the course calendar — this is one more session on an already-slipping
   schedule. Flagged, not decided.
