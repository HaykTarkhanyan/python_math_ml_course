# RNN Lecture Block: Design

**Date:** 2026-06-16
**Status:** Reviewed and revised; awaiting final approval before implementation plan
**Location of work:** `ml_new/ch5_neural_networks/`

## Problem

`ml_new/ch5_neural_networks/L17_rnns_lstm.tex` is a skeleton — 7 frames, every content frame a `% TODO: port from lecture_i2dl/rnn` comment, with 5 sections (sequential data + vanilla RNN, BPTT, LSTM/GRU, attention, applications) crammed into one deck and no hands-on path or homework.

The instructor wants RNNs taught as a **deep, practical, 2-lecture block** with a coding component, following the course's per-topic delivery pattern: **slide deck + practical homework + solution codebase**. This mirrors the CNN block (`CNN_BLOCK_DESIGN.md`) that precedes it in the same chapter.

## Goal

Replace the single `L17_rnns_lstm.tex` skeleton with a planned **2-lecture RNN block**, each lecture a three-part unit:

1. a project-styled Beamer **slide deck**,
2. a **practical homework** (problems in the NN-chapter `.qmd`),
3. a **solution codebase** (`HW{n}_solution.ipynb`, PyTorch, Colab-runnable).

This spec defines structure, per-lecture content, homework arc, sourcing, and conventions. Authoring is the implementation phase (separate plan).

## Decisions (locked during brainstorming)

| Dimension | Decision |
|---|---|
| Lecture count / depth | **2 lectures**, deep + practical. "Deep" = breadth + hands-on mastery, not mathematical depth |
| Attention | **Teaser / bridge only** — hands off to the existing `misc/dl4nlp/02_transformers.tex`. No deep dive (avoids duplicating dl4nlp) |
| Framework | **PyTorch**, run on **Google Colab** free GPU (instructor machine has no CUDA). HW1 Part A is pure NumPy, local |
| Math rigor | **Mechanics + intuition, light proofs.** No BPTT derivation; vanishing gradient taught via numeric intuition (see L17a) |
| Delivery per unit | slide deck + practical homework (`.qmd`) + solution codebase (`.ipynb`) |
| Numbering | **L17a / L17b** — splits the existing `L17_rnns_lstm.tex`; leaves L18 (optimizers) unchanged. Consistent with existing letter-suffix companions (L01b, L05b, L13b, L16a/b/c) |

## Prerequisites & chapter dependencies

Assumes students arrive (from L14/L15) knowing the MLP, backprop, and a **basic PyTorch training loop** (optimizer / loss.backward / step, epochs, autograd). **L16 (CNN) precedes this block** — lean on two callbacks students just saw:

- **Weight sharing:** CNNs share kernels across *space*; RNNs share weights across *time*. Name the parallel explicitly.
- **Gradient highways:** ResNet's skip connection (L16b) and the LSTM cell state (L17b) both preserve gradient flow. Reference, don't re-derive.

## Scope

### In scope

Two lecture units in `ml_new/ch5_neural_networks/`:

- **L17a — Sequential Data & Vanilla RNNs**
- **L17b — Gated RNNs & Generating Sequences** (attention as a bridge)

Plus the practical homeworks (in the NN-chapter `.qmd`) and two `HW{n}_solution.ipynb` notebooks. Delete `L17_rnns_lstm.tex` once L17a/b compile and land.

**Homework `.qmd`:** CNN and RNN homeworks share the NN-chapter `.qmd` at `ml_new/ch5_neural_networks/05_neural_networks__concepts.qmd` (created by whichever block lands first), following course conventions: `{data-difficulty="1|2|3"}` 🧀 levels and `{.content-visible when-profile="solution"}`.

### Out of scope (YAGNI)

- Transformer / self-attention deep dive — owned by `misc/dl4nlp/02_transformers.tex`.
- seq2seq-with-attention *implementation* — conceptual only.
- Word-embedding deep dive (word2vec/CBOW/skip-gram) — owned by `dl4nlp/01_pre_transformer.tex`.
- Bidirectional RNNs beyond a one-frame mention.
- Renumbering L18.

## Per-lecture deck outlines

Target ~18-22 frames per deck (one ~90-min session each).

### L17a — Sequential Data & Vanilla RNNs

Port from `_reference/lecture_i2dl/slides/rnn/{introduction, backprop}` + the task taxonomy from `rnn/applications`.

1. **Why recurrent nets?** Sequential data (text, time series, speech). Why MLPs/CNNs fall short: variable length, order matters. Predict-first `\pause`: how do you feed a variable-length sentence to a fixed-input MLP?
2. **Sequence-task taxonomy** (moved here from L17b to balance load): one-to-many (captioning), many-to-one (sentiment), many-to-many aligned (POS tagging), many-to-many seq2seq (translation). Frame **language modeling as next-character prediction** — sets up HW2.
3. **The vanilla RNN.** Hidden state, recurrence `h_t = f(W_x x_t + W_h h_{t-1} + b)`, unrolling through time, **weight sharing across time** (callback to CNN weight sharing across space).
4. **Backprop Through Time.** Unroll then backprop — mechanics + intuition, not a full derivation.
5. **Vanishing / exploding gradients.** Taught via **concrete numeric intuition**: repeated multiplication, `0.9^50 ≈ 0.005` vs `1.1^50 ≈ 117`. Predict-first `\pause`: what happens to the gradient after 50 steps? → motivates gated architectures. The core payoff of the block.

### L17b — Gated RNNs & Generating Sequences

Port from `rnn/{modernrnn, attention, applications}`.

1. **The fix: gating.** LSTM as a **differentiable latch** — a memory cell, like a bit of learnable digital RAM the network reads / writes / erases. Three gates with crisp, purpose-labeled roles: **forget** = what to discard, **input** = what new to add, **output** = what to expose. The cell state as a **"gradient highway"** that flows through the sequence with minimal modification (callback to ResNet skip connections, L16b). Use descriptive, unambiguous notation — "gate-variable soup" is a documented source of student confusion.
2. **GRU** briefly; LSTM-vs-GRU tradeoff.
3. **Headline application: char-level language modeling → text generation.** How sampling works; **temperature** (simplified — cross-ref `dl4nlp/04_decoding_strategies` for the full treatment). Sets up HW2.
4. **Attention teaser.** The fixed-size-context bottleneck in seq2seq → attention as a soft lookup → "**attention replaces recurrence**" → bridge to `dl4nlp/02_transformers`. Conceptual only. (Framed distinctly from L16c's "transformers for vision" teaser; together they form a deliberate "transformers supersede both CNNs and RNNs" motif.)
5. **Wrap-up.**

## Practical homeworks + solution codebases

### HW1 (L17a) — Build an RNN by hand, watch the gradient vanish

Deliberate arc with HW2: vanilla RNN fails on long dependencies → LSTM succeeds.

- **Part A — Vanilla RNN from scratch** (NumPy, local). Implement the RNN cell *forward* pass. Run on a toy task where **short sequences work and long ones fail** — e.g. "recall the first token" / sequence-copy at varying lengths. Students see RNNs work short-range.
- **Part B — Watch the gradient vanish** (PyTorch, local or Colab). Build the unrolled RNN with torch tensors and use **autograd + gradient hooks** to measure per-timestep gradient norms; plot them shrinking over long sequences. **Students observe vanishing gradients WITHOUT implementing backprop-through-time by hand** — autograd does the backward pass; they only inspect it.

### HW2 (L17b) — Char-level text generator (LSTM, PyTorch, Colab)

Train a char-level LSTM and generate text; sample at different temperatures (simplified — cross-ref `dl4nlp/04`).

- **Default target: Armenian names** (city/village names, or first names) — constrained vocabulary, short sequences, fast to train within a Colab homework budget, and students can immediately tell if output "looks Armenian." This is the reliable, charming version of the classic char-RNN demo.
- **Stretch:** Tumanyan literary text (public domain) — needs more data/compute; offer as a bonus, not the default.
- **Fallback:** tiny-Shakespeare (canonical, guaranteed clean).
- **Feasibility:** cap epochs, ship a checkpoint so Colab disconnects don't block the writeup.

Payoff: from "the network can't remember" (HW1 vanishing gradient) to "the LSTM generates plausible Armenian names" (HW2) — the homeworks tell the same story as the lectures.

## Style & pedagogy conventions

- Project Beamer style: `\documentclass[aspectratio=169]{beamer}`, `\input{../preamble}`, dove theme, popblue / sampred / paramgreen / warnred palette.
- TikZ for the unrolled-RNN and LSTM-cell diagrams — no external images.
- Predict-first `\pause` on the counter-intuitive points (variable-length input, gradient after 50 steps).
- Armenian local examples alongside canonical Western ones.
- Target ~18-22 frames per deck.
- Run `beamer-overflow-check` after each deck compiles.

## Open decisions (resolve before/at implementation)

1. **Text-gen corpus** — Armenian names default vs Tumanyan stretch vs Shakespeare fallback. Default (names) needs a clean source list; confirm one exists.
2. **GRU depth** — full treatment or brief? (Leaning brief; LSTM is the star.)
3. **seq2seq depth** — how much encoder-decoder to show before the attention bridge? (Leaning high-level.)
4. **Bidirectional RNNs** — one-frame mention or skip?
5. **NN-chapter `.qmd`** — shared across L14-L18 vs per-lecture. (Spec assumes shared.)

## Sourcing summary

All upstream present locally — **no sourcing gap** (unlike CNN's missing modern-CNN source).

| Lecture | Local upstream |
|---|---|
| L17a Foundations | `rnn/{introduction, backprop}` + task taxonomy from `rnn/applications` |
| L17b Gated + generation | `rnn/{modernrnn, attention, applications}` |

## Design review fixes (2026-06-16)

Self-review pass before writing. Folded in:

1. **HW1 uses autograd, not hand-rolled BPTT.** Measuring per-timestep gradient norms via autograd + hooks is the tractable way to *observe* vanishing gradients; implementing backprop-through-time by hand is a feasibility trap and not the learning goal.
2. **Rebalanced the decks.** Moved the sequence-task taxonomy / applications survey into L17a — fills out an otherwise-thin L17a and lightens an overloaded L17b.
3. **HW2 default = Armenian names, not literary text.** Constrained vocabulary gives reliable, charming results inside a homework time budget; literary Tumanyan needs more data/compute and risks gibberish output. Names default, Tumanyan stretch, Shakespeare fallback.
4. **Vanishing gradient via numeric intuition** (`0.9^50` vs `1.1^50`) rather than a Jacobian-norm proof — the right altitude under the light-proofs stance.

Minor: simplified temperature/tokenization with cross-refs to `dl4nlp/03,04` (don't re-teach); distinct framing of the L16c vs L17b transformer teasers; `.qmd` + difficulty conventions named; HW1 toy task chosen so short-works/long-fails contrast is crisp.

## Pedagogy research findings (2026-06-16)

Web research into RNN/LSTM-teaching best practices. Folded into the outlines above:

1. **Teach the vanilla RNN before the LSTM** — the intuition transfers (an LSTM is a kind of RNN). Validates the L17a → L17b ordering.
2. **Vanishing vs exploding is a weights-vs-1 story** — gradients vanish when recurrent weights are slightly <1 and explode when slightly >1. Directly confirms the `0.9^50` vs `1.1^50` numeric framing in L17a.
3. **LSTM = a "differentiable latch."** Framing the cell as a learnable memory latch (the unit of digital RAM) is a documented intuition hook. Added to L17b.
4. **Crisp gate roles + clean notation.** forget = discard, input = add, output = expose, with descriptive variable names. "Gate-variable soup" is a documented source of confusion. Added as an L17b slide directive.
5. **Cell state as a minimal-modification path** is *why* gradients survive — confirms the "gradient highway" callback to ResNet (L16b).
6. **char-rnn is the canonical hands-on.** Karpathy's "Unreasonable Effectiveness of RNNs" + `min-char-rnn` (~100 lines of pure NumPy, BPTT included) is the standard teaching vehicle, and vetted course assignments exist (Gettysburg model-AI, UPenn CIS530). Validates HW2; offer `min-char-rnn` as an optional "see it whole in 100 lines" extension. NB: that gist hand-implements BPTT — HW1 deliberately uses autograd to *observe* gradients instead of deriving them.

Sources:

- [Karpathy — The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [karpathy/char-rnn (GitHub)](https://github.com/karpathy/char-rnn)
- [Fundamentals of RNN and LSTM (arXiv 1808.03314)](https://arxiv.org/pdf/1808.03314)
- [Why LSTMs Stop Your Gradients From Vanishing (Weberna)](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)
- [Prevent the Vanishing Gradient Problem with LSTM (Baeldung)](https://www.baeldung.com/cs/lstm-vanishing-gradient-prevention)
- [Understanding How RNNs Model Text (Gettysburg model-AI assignment)](http://modelai.gettysburg.edu/2018/rnntext/index.html)

## Status / next step

Design reviewed and revised, with pedagogy research folded in. When picked up, the next step is the `writing-plans` skill → a deck-by-deck + notebook-by-notebook implementation plan. No authoring has started. Companion spec: `CNN_BLOCK_DESIGN.md` (same chapter, same pattern).
