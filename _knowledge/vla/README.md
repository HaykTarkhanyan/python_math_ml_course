# Vision-Language-Action models (VLAs)

Reference material for an `ml/` chapter. Gathered 2026-08-07, entirely from web search, because
this field turns over every few months and any model list older than about six months is wrong.

## In three sentences

A **vision-language-action model** is a pretrained vision-language model that has been fine-tuned
so that its output is not text but robot actions: given camera images plus a sentence like "put
the bowl in the sink", it emits joint or end-effector commands directly, with no hand-written
perception, planning, or control stack in between. The bet is that internet-scale visual and
linguistic knowledge is what a robot has always been missing, and that once you have it, a
*single* model can drive many different robots at many different tasks - a "generalist robot
policy" instead of one policy per task. The catch, and it is a big one, is that unlike text or
images there is no internet of robot data: the entire world's supply of robot manipulation
demonstrations is on the order of a hundred thousand hours against roughly 300 trillion tokens of
text, so almost every open problem in the field traces back to that gap.

## Why this is worth a chapter here

It is the cleanest live example of the course's central pattern - *pretrain on a huge generic
corpus, adapt to a narrow task* - applied in a domain where the pretraining corpus **does not
exist**. That makes it pedagogically much richer than another architecture chapter, because the
interesting content is not the transformer, it is what people do when scaling laws have no fuel:
cross-embodiment pooling, simulation, human video, data engines, and RL on real hardware.

It also lands squarely on prerequisites this course already teaches. VLAs are a VLM (ch9/genai),
a diffusion model (ch10) used as a policy head, imitation learning, and increasingly RL (ch11) -
all four in one system. And it is the best available case study of a discipline where **the
benchmark numbers are known to be misleading**: models scoring 95%+ on the standard benchmark
collapse to 0.0% when the same task is restated in different words
([LIBERO-PRO](https://arxiv.org/abs/2510.03827)). That is a lesson worth teaching on its own.

Robotics-specific vocabulary (end effector, teleoperation, embodiment, action chunking) is
explained where it appears; no robotics background is assumed.

## Files

| File | Contents |
|---|---|
| `01_what_is_a_vla.md` | The core idea, the observation-to-action loop, action tokenization vs diffusion/flow heads, control frequency and latency, what "generalist policy" means |
| `02_key_models.md` | RT-1 through Gemini Robotics 2, with dates, origins, sizes, and open-weight status. Table plus prose |
| `03_data_and_training.md` | Where robot data comes from, why scarcity is *the* constraint, sim-to-real, cross-embodiment transfer and when it backfires |
| `04_evaluation_and_failure_modes.md` | Benchmarks, why real-robot evaluation is close to non-reproducible, linguistic fragility, embodied red-teaming (RedVLA and relatives) |
| `05_open_problems.md` | What is genuinely unsolved, where the field disagrees with itself, and the safety dimension |
| `sources.md` | Every URL used, grouped by topic, with what it was good for and what could not be verified |

## Reading order

`01` then `02` is the "what is it" pair and covers roughly one lecture. `03` and `04` are the
substance of the second lecture and the part a student will not get from a blog post. `05` is
discussion material.

## One warning about this literature

Robotics publishes an unusual amount of marketing. Company blog posts show a robot folding
laundry for eighteen hours and the reader is invited to infer a success rate that is never
stated. Throughout these notes, claims are labelled: **measured** (a number from a paper or a
labelled chart, with n where available), **company-reported** (a number a lab published about its
own closed model, unreproducible by anyone else), and **demo** (a video with no number attached).
Do not let the three blur together in the lecture.
