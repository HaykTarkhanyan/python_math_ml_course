# RNN chapter project - "Two networks, one vector" (OUTLINE, for approval)

Drafted 2026-10-24 from the instructor interview (DECISIONS #96); revised the same day after a
self-review (budget, Task B metrics, shared dataset, honesty about the tool). **Built 2026-10-24:**
`seq2seq_dates_and_spaces_solution.ipynb` (executed on a Colab T4, ~11 min, every cell with output) and
the task version `seq2seq_dates_and_spaces.ipynb`, from `py_src/build_seq2seq_project_nb.py` and
`py_src/build_seq2seq_project_tasks.py`; curves in `results/seq2seq_wall.json`. The bidirectional
seq2seq encoder became an unexecuted "going further" note; the one-directional tagger is a run bonus.

## The brief, as agreed

- **The only hands-on material of the chapter.** No class session, so the notebook has to teach on
  its own: much more explanation than the old practical, every step motivated before it is coded.
- **Delayed seq2seq** ([53]'s 4th taxonomy picture): an encoder reads the whole input into one
  vector C, a decoder writes the output one character at a time, feeding each one back in.
  Students met that feed-back loop in the ch11 name-inventor project, so it is reused, not taught
  from zero. Greedy decoding only (always take the top character), no sampling.
- **C is the decoder's starting state only** (Sutskever et al., 2014 style): the encoder's last
  (z, c) becomes the decoder's first. Feeding C again at every step (Cho et al., 2014, the dashed
  arrows in [54]) is mentioned, not built: one design, explained well, beats two.
- **Two Armenian tasks, one piece of code.** The same encoder-decoder class, retrained (the
  vocabularies differ). That is the point of seq2seq: any sequence in, any sequence out.
  - **Task A, dates:** a date in words or mixed forms in, the date in digits out.
    `հոկտեմբերի քսաներեք, երկու հազար քսանվեց թ.` -> `2026-10-23`. Every form reorders the input
    (day-month-year in, year-month-day out), like the rows that ran backwards on [54]'s alignment grid.
  - **Task B, spaces:** `եսայսամենիննայումեմլուռ` -> `ես այս ամենին նայում եմ լուռ`. The decoder must
    rebuild the whole sentence from C alone: [54]'s bottleneck, measured.
  - **Said plainly in the notebook:** Task B is the wrong tool on purpose, and Part 9 proves it.
    A per-letter tagger ([53]'s 5th picture, synced) decides "space or not" after each letter,
    reading the letters on both sides with a bidirectional LSTM. It has no bottleneck, so its
    error curve should stay flat where the seq2seq curves rise (instructor, 2026-10-24: "add the
    per letter tagger as well"). It also hints at why attention works: each output step gets to
    read its own part of the input.
- **Budget: Google Colab with a T4 GPU, about 8 minutes for the whole notebook** (revised
  2026-10-24 after measuring; see "Measured" below). On a laptop CPU the same training is ~30x
  slower (394 s vs 12 s for one model), so the notebook asks for a T4 runtime first. 5 trainings in
  the main path: Task A once, Task B seq2seq at 3 sizes of C, the tagger once. The solution
  notebook is executed on Colab (`colab exec -f`), so every cell has its output.
- **509, the instructor's favourite number, in a few places:** the running example date is
  5 September (05.09): `սեպտեմբերի հինգ, երկու հազար քսանվեց թ.` -> `2026-09-05`, used in the
  fixed examples and the traced batch; the unseen-dates test set has 5,090 dates; the seed is 509
  everywhere (the course default anyway).
- **Deliverable:** the executed task notebook with every `# YOUR CODE HERE` filled and every
  discussion question answered in a markdown cell. Reviewed with the `review-submission` skill.
- **Files:** `seq2seq_dates_and_spaces_solution.ipynb` (built and executed by
  `py_src/build_seq2seq_project_nb.py`) and the student version `seq2seq_dates_and_spaces.ipynb` derived
  from it, the same two-step pipeline as the old practical. Descriptive name, no index prefix (instructor, 2026-10-24).

## Data (instructor: "get some texts from the web and make data by merging")

- **Task A vocabulary from the web, pairs by merging.** Armenian number words (units, teens, tens,
  hundreds, thousands) and the 12 month names in the genitive (`հոկտեմբերի`) are taken from web
  references and cited in the notebook, not typed from my memory. A generator merges them into
  dates in four surface forms, mixed at random (instructor-checked 2026-10-24: these four are how
  people write dates; a fifth, year-first form `2026 թվականի սեպտեմբերի 5-ը` was dropped):
  1. all words: `սեպտեմբերի հինգ, երկու հազար քսանվեց թ.`
  2. digits + month word: `5 սեպտեմբերի 2026`
  3. short month: `5 սեպ 2026` (CLDR's three-letter forms)
  4. all digits: `05.09.2026`
  Years 1900-2099, every valid day (leap years included). Two test sets: unseen dates from seen
  years, and a **held-out decade (2050-2059)** never seen in training, so the model has to write
  years it never wrote. The instructor checks 30 generated samples before anything is trained.
- **The generator is given** (instructor, Q17): students read and run it, they do not write it.
  It is seeded (509), so every student gets the same dataset; asserts check its size and a few
  fixed pairs, including the running example `2026-09-05`.
- **Task B text from the web.** A few hundred Armenian Wikipedia articles (modern spelling), fetched
  once by `py_src/build_seq2seq_data.py`, cleaned (lowercased; chunks with Latin letters or digits
  dropped) and shipped as `data/hy_sentences.txt` (target: under 1 MB), so students never download
  anything. Inputs are built by merging consecutive words into chunks of 5 to 60 letters, and the
  encoder reads them **reversed** (Sutskever et al., 2014): without it the model barely learns
  (measured below), so reversing is the default and "what if you don't?" is a bonus.
  **Funny cases on show** (instructor, Q18): the bulk is ordinary Wikipedia text, but the sentences
  the notebook displays come from funny articles (food, Kargin Haghordum, Vardavar and the like),
  and the wrap-up collects the model's funniest wrong splits. Strings with two valid splits, if the
  instructor knows good Armenian ones, make the best predict-first questions.

## Metrics (defined in Part 0, used everywhere)

- **Exact match:** the whole output is right. The headline number for Task A; for Task B it drops
  to near zero on long inputs, which is part of the story, so it is never the only number.
- **Character error rate:** edit distance to the target divided by its length (Levenshtein, via a
  small package, pinned). The main Task B number.
- **Where the first mistake is,** as a fraction of the target length. Robust where per-position
  comparison is not (one dropped letter shifts everything after it). Answers Part 8's question.
- **Letters vs spaces (Task B):** strip the spaces from the output; if what is left equals the
  input, every error was a spacing error. Otherwise the decoder also forgot or changed letters,
  which is the bottleneck showing.

## Notebook structure

Every part follows the same rhythm: **why** (a short explanation, often with a figure) -> **predict**
(a question before the run) -> **code** (small cells, one idea each) -> **check** (asserts that print
what they test) -> **read the result** (what the number means, linked back to the slide it tests).

| Part | Level | What students do | Lecture link |
|---|---|---|---|
| 0. Orientation | - | The two tasks, the delayed seq2seq picture, the four metrics, how the asserts work, the time budget per part | [53] taxonomy |
| 1. Dates as data | 🧀 | Read and run the given generator (vocabulary tables merged into four forms), look at 20 samples; small task: write your own birthday in all four forms by hand, then check against the generator | - |
| 2. Characters to tensors | 🧀 | Vocabularies, `<sos>`/`<eos>`/`<pad>`, padding + mask, one batch traced by hand with its shapes printed | [53] padding |
| 3. The encoder | 🧀🧀 | An LSTM reads the date; its last (z, c) is C. Print C for two dates; why one vector? | [54] encoder-decoder |
| 4. The decoder and teacher forcing | 🧀🧀 | The decoder starts from C. Training: the true previous character goes in (teacher forcing), with the one-step shift drawn out and the loss ignoring pads. Inference: its own guess goes in (greedy), and why that mismatch matters | [54], ch11 loop |
| 5. Train on dates | 🧀🧀 | Train, then exact match on unseen dates and on the held-out 2050s, split by input format and by field (day, month, year). Predict-first: which format and which field fail most? | - |
| 6. Same code, new task: spaces | 🧀🧀 | Load the Armenian text, build (no-space, spaced) pairs at 10-80 letters, retrain the same class | [53] taxonomy |
| 7. The wall, measured | 🧀🧀🧀 | One model per size of C (3 sizes), each trained on mixed lengths and scored per length bucket. Character error rate vs input length, one curve per size: does a bigger C move the wall or remove it? | [54] bottleneck, alignment grid |
| 8. Where it breaks | 🧀🧀🧀 | Where the first mistake falls, and letters-vs-spaces errors, for short vs long inputs: does the decoder lose the start of long sentences ([54]: "early words get overwritten") or the end? | [54] bottleneck |
| 9. The right tool: a tagger | 🧀🧀 | Same data, a bidirectional LSTM that outputs "space or not" after every letter. Its error rate goes on Part 7's plot: a flat line next to the rising seq2seq curves. Why no wall: no single C, every output reads its own position (the idea attention generalises) | [53] synced + bidirectional |
| Bonus 🎁 | 🧀🧀🧀 | Reverse the input (Sutskever, Vinyals & Le, 2014) and a bidirectional encoder for the seq2seq model: do they move its curve? A one-directional tagger: how much do the letters on the right matter? Not in the 10-minute budget | [53] bidirectional |
| Wrap-up | - | One table: claim, what you measured. The curves are saved to `results/seq2seq_wall.json`; attention, which removes the wall without giving up delayed seq2seq, is LLM-3 | LLM-3 pointer |

"More detail than usual" concretely means: a figure for the encoder-decoder with the chapter's own
example, a drawn teacher-forcing shift, a "common bugs" box per part (off-by-one in the shift,
forgetting `ignore_index` for pads, comparing padded strings), every hyperparameter explained when
it appears, and a written answer key for every discussion question in the solution notebook.

## Build order and checks

1. Fetch the vocabulary, build the date generator, **instructor checks 30 samples**.
2. Fetch and clean the Armenian text, ship `data/hy_sentences.txt`, pick the funny display sentences.
3. **Measure before writing prose:** train both tasks at the planned sizes plus the tagger, time
   them on this laptop, check that Task B's error rate actually rises with length at every size of
   C, and that the tagger's stays flat. If
   it does not, change the lengths or the sizes before any explanation is written (DECISIONS #96).
   Whatever the sweep shows about bigger C is what the notebook says, even if it softens [54]'s
   "fixed capacity" box; the prose is written after the numbers, not before.
4. Write the solution notebook around the measured numbers; every number in the prose is printed by
   a cell. Derive the task version. Runtime comments on every cell slower than ~2 s.
5. Link both notebooks on `rnn.qmd` under the homework heading.

## Measured (2026-10-24, before any prose)

Probe runs (scratchpad `s2s_probe.py`), batch 64, character error rate (CER) by input length:

- **Laptop CPU is too slow.** Task B at 10-80 letters, C = 128, 600 steps: 1,119 s and the model
  learned nothing (it wrote «այն այն այն ...» for everything, CER ~0.88 at every length). Task A, 1,500
  steps: 412 s on 2 threads. The same Task B model on a Colab T4: ~12 s per 1,500 steps.
- **Reversing the input matters.** C = 128, 1,500 steps, 5-30 letters: CER 0.06 -> 0.55 reversed vs
  0.35 -> 0.71 not reversed.
- **The wall is real, not under-training.** C = 256, 5-30 letters: 1,500 / 3,000 / 12,000 steps give
  CER 0.46 / 0.38 / 0.32 on the longest bucket, while short chunks stay at 0.04-0.05. At 12,000
  steps (a plateau), longest bucket: C = 64 0.63, C = 128 0.47, C = 256 0.32. A bigger C pushes the
  wall back; none removes it.
- **The tagger has no wall.** CER 0.01-0.02 at every length from 5 to 65 letters, letters always kept.
- **5-60 letters, C = 256, 12,000 steps:** CER 0.06 at 5 letters -> 0.63 at 65; the decoder writes the
  start right, then invents Armenian-looking words.
- **Dates:** unseen dates 100% from 600 steps on; the held-out 2050s stay at 57-68% (dates in words
  33%) whatever the training length: "2050" comes out as "2005".
- **T4 times:** dates 3,000 steps 30 s; spaces 12,000 steps 97-161 s; tagger 3,000 steps 20-28 s.

## Answered (2026-10-24)

- **Q17:** the date generator is given in the notebook; students read and run it.
- **Q18:** Armenian Wikipedia, with funny articles for the sentences on show.
- **Q19:** English.
- **Q20:** four forms (all words, digits + month, short month, all digits); the year-first form is
  dropped.
- **Q24-Q27 (the 30-sample check):** 1900s as `հազար ինը հարյուր...` (no `մեկ`); the day has no
  `-ը` (`սեպտեմբերի հինգ`); short months are CLDR's `հնվ փտվ մրտ ապր մյս հնս հլս օգս սեպ հոկ նոյ դեկ`;
  the instructor skimmed the samples ("seems okayish"), not a line-by-line check. Sources: month
  names from Unicode CLDR (cldr-json, hy); number words from CLDR's spellout rules, corrected to
  Wiktionary where they disagree (90 = իննսուն, 10 = տասը, hundreds as two words).
- **Q21:** no LLM-3 promise; the wrap-up only points to attention there.
- **Q28/Q29 (budget, CPU use):** superseded by the instructor's "run it on Colab" (2026-10-24):
  probes and the solution notebook run on a Colab T4 through the `colab` CLI in WSL.
