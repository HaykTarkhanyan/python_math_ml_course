# WORKFLOWS.md - AI assistant task playbook

This repo is worked on with AI assistants of varying capability. `CLAUDE.md` holds
the rules; this file maps each recurring task to its workflow, skill, and
source-of-truth file, so nothing depends on the assistant inferring it. If a task
matches a row below, follow the referenced workflow - do not improvise a new one.

## Task map

| When the task is... | Use | Source of truth |
|---|---|---|
| Create a new slide deck | `slide-style` skill (interview -> outline -> approval -> build) | `ml/SLIDE_STYLE.md` |
| Edit an existing deck | make the edit, then `compile-deck` skill | `ml/SLIDE_STYLE.md` |
| Polish / review a deck before delivery (fact-check + overflow + opt-in student review) | deck-polish loop (section below) | `ml/SLIDE_STYLE.md` |
| Compile / verify any `.tex` | `compile-deck` skill | - |
| Generate figures for a deck | `make-figures` skill | `ml/SLIDE_STYLE.md` (Figures section) |
| Hunt clipped / overflowing slides | `beamer-overflow-check` skill | - |
| Create or edit a math homework `.qmd` (`math/XX_*.qmd`) | follow `homework_structure_guide.md` for format, but note its `math/Homeworks/` path examples are stale - files live directly under `math/`; register in `_quarto.yml` (exact case) | `homework_structure_guide.md` |
| Create or edit an ML chapter page (`ml/NN_topic/NN_topic.qmd`) | follow the QMD structure template in `CONVENTIONS.md`; register in `_quarto.yml` (exact case) | `CONVENTIONS.md` (QMD structure) |
| Solutions package (LaTeX PDF + ipynb) | `generate-solutions` skill | `math/SOLUTIONS_STATUS.md` |
| Collapsible solutions inside a `.qmd` | `add-inline-solutions` skill | `math/SOLUTIONS_STATUS.md` |
| Add YouTube / video links after a session | `update-youtube` skill | `_meta/youtube_channel.md` |
| Pull reference material from a video link (transcript, screenshots, borrow visuals into slides) | `youtube-reference` skill (yt-dlp + ffmpeg pipeline -> `_reference_<slug>/`) | `.claude/skills/youtube-reference/SKILL.md` |
| Build a quiz Google Form | `google-forms-builder` skill | - |
| Delete LaTeX build junk | `clean-tex` skill or `clean_latex.py` | - |
| End of session / commit the day's work | `wrap-session` skill | `_work_sessions/*.toml` |

## Deck polish / review loop

Used when polishing an existing deck before delivery (`misc/dl4nlp/*` or `ml/` decks). Run in order:

1. **Fact-check** - web-verify every cited number, date, attribution, and formula against the source paper; fix errors. (Facts before layout.)
2. **Self-driven overflow / polish pass** - compile, render each page to PNG (`pdftoppm -png -r 120`), read the images, fix silent Beamer overflow (boxes drawn over each other, clipped text, title clips), re-render the changed pages to confirm. This catches *layout* faults.

   **Run BOTH detectors - they catch different failures and neither catches the other's:**

   ```bash
   ./ma/Scripts/python.exe non_essential/detect_clipped_slides.py DECK.pdf      # clipped at the frame edge
   ./ma/Scripts/python.exe non_essential/detect_footer_collisions.py DECK.pdf   # grown into the page-number band
   ```

   `detect_clipped_slides.py` looks for content clipped at the frame boundary. It is **blind** to a `tcolorbox` or paragraph that simply grows downward until it sits on the page number - the text is still drawn, so nothing is "clipped", but the last line is unreadable. (Found 2026-08-13: it reported 0 flagged for all three `ml/ch19_mech_interp` decks while **13 frames** had content in the footer band, two of which lost a sentence outright. Both decks had also passed 2x `pdflatex` with 0 overfull-vbox warnings, because Beamer does not warn about this at all.) See `_learnings/2026-08-13-1930_a-passing-check-is-only-as-good-as-its-coverage.md`.

   **Full-bleed frames are detected and excluded automatically** (added 2026-08-14 while building `ml/ch20`, the first deck to use six of them). A full-bleed still fills the footer band by design, so it used to be flagged every time. The detector now measures whether the image reaches the page border and reports those separately without failing the build:

   ```
   L48_subliminal_learning.pdf: clean
       (6 full-bleed image frame(s) not counted: 6, 8, 23, 27, 47, 59)
   ```

   If you change that heuristic, re-run it on `ml/ch19_mech_interp` (which has no full-bleed frames and must stay at 0) to prove it did not go blind. See `_learnings/2026-08-14-0315_the-footer-detector-cannot-see-a-full-bleed-frame.md`.

   **`detect_clipped_slides.py` needs the SOURCE, not just the PDF** - it compares the two. It accepts a directory, a `.tex`, or a `.pdf` (whose sibling `.tex` it finds), and **raises** on anything else. Before 2026-08-14 it silently reported "0 flagged" for any argument it could not parse, including a `.pdf` path and a path that did not exist, so several decks were recorded "clean" by a run that checked nothing. If it ever prints 0 without naming the `.tex` files it checked, do not believe it.

   Baseline as of 2026-08-14: `ch13`-`ch20` run 0-2 flags; the older chapters (`01`-`08`, `ch11`) run 8-18 and have **not** been triaged. At least one is real: `ch11_rl/L32` page 12 loses "and the loss grows linearly forever." off the bottom of a `tcolorbox`.
2b. **Acronym check** (mechanical, 10 seconds - the style rule alone has not been enough):

   ```bash
   sed 's/%.*//' DECK.tex | grep -oE '\b[A-Z]{2,6}\b' | sort -u
   ```

   Every acronym listed must be **spelled out somewhere in the deck**, at or before first use. Frame titles and forward references count as uses and define nothing. See the Abbreviations bullet in `ml/SLIDE_STYLE.md` for the failure modes. (Shipped `CFI`, `LOCO`, `SAGE`, `ICE`, `SHAP`, `LIME`, `GAM` undefined across the interpretability chapter, 2026-07-28 - caught by the instructor, not by a read-through.)
3. **Student review - opt-in, ASK FIRST.** One **Sonnet** subagent per deck reads ONLY the rendered slide PNGs (render at `-r 150` so body text is legible). It must never open the `.tex` - seeing the source contaminates the review, and the whole point is to see what a student in the room sees. One agent, roughly 40-90k tokens and 5-8 min per deck. **Ask before launching**, every time.

   **Default to the PEDAGOGY review.** Ask what it was like to *learn* from the deck, and explicitly forbid proofreading ("a separate pass already found the typos, clipped text and undefined terms - ignore any you notice"). Tell it two constraints a reader does not otherwise feel: you cannot re-read a slide once it is gone, and you cannot pause the lecturer. Then ask for exactly these:

   1. **Pacing** by page range, and the single frame most needing the lecturer to stop for five minutes.
   2. **What it accepted without actually understanding** - claims it nodded along to but could not explain to a friend. Usually the most valuable section.
   3. **Where it wanted a worked example and did not get one.**
   4. **Self-efficacy checklist** - 4-6 deck-specific "could you now actually do X?" items, answered yes / partly / no with what is missing.
   5. **Three exam questions it could now answer, and three it could not** but feels it should.
   6. **What to cut**, unsentimentally.
   7. **Whether the running example helped or was overhead.**
   8. **Where a 90-minute session should end**, and why there.
   9. **The three questions it would bring to office hours.**

   Add 1-2 deck-specific probes where the deck does something unusual (e.g. "this lecture teaches a technique that then measurably failed - did that feel like a wasted lecture, or did the failure teach more than a success would?").

   **The QA-shaped variant** (report factual errors, clipped content, undefined notation) is the older form of this step. It is now largely redundant: layout is covered by the margin scan in step 2, acronyms by step 2b, and arithmetic by the author. Run it only for a deck dense in conceptual diagrams, where an inverted label is possible. Note that a pedagogy reviewer catches those anyway, because a slide teaching the opposite of the truth is confusing first.

   *Evidence (2026-08-10, ch17_rag).* The QA pass on three decks returned mostly layout findings plus one real catch: a 2x2 on grounding whose off-diagonal labels were swapped, which `pdflatex`, the margin scan and the acronym grep had all passed. The pedagogy pass on the same three decks found what none of those could: a forward dependency (semantic chunking used embeddings 20 slides before they were taught), a worked-example asymmetry repeated in all three decks (the first method in a section gets hand-computed numbers, its siblings get a formula and a result chart), and the chapter-level gap that a student finished able to *evaluate* a RAG system but not *build* one. All three reviewers independently chose the same kind of session break, arguing from cognitive load. See `_learnings/2026-08-10-2210_automated-deck-checks-cannot-see-meaning.md`.

4. **Verify + apply** - verify each finding against the source (Sonnet is strong but not infallible), apply the quick wins, re-render to confirm. Expect roughly half a dozen new overflows: almost every added frame or paragraph pushes something past the bottom edge, so re-run the margin scan after *each* batch, not once at the end.
5. **Clean + commit** - `clean_latex.py`, then commit the deck as its own unit.

## Definition of done

| Artifact | Done means |
|---|---|
| Slide deck | 2x pdflatex passes, 0 `!` lines in the `.log`, no `end{center>`-style typos, **`non_essential/detect_clipped_slides.py` AND `non_essential/detect_footer_collisions.py` both run, every flag checked against the rendered page**, overflow checked visually, **acronym check run** (see below), aux files cleaned, `% Provenance:` block present (`ml/` decks only - stat/optim decks don't use them) |
| Figure script | runs end-to-end under the `ma` venv, PDFs in sibling `fig/`, log in `logs/`, figures actually embedded in the deck and the deck recompiled |
| Homework `.qmd` | registered in `_quarto.yml` with exact-case path, blank line before every list / blockquote / fence, difficulty markers set |
| Commit | no `.aux`/`.log`/`.nav` staged, message explains the change, push only when asked |

## Hard rules (each of these has caused real damage)

1. Python runs through `./ma/Scripts/python.exe`. Never `uv run --with ...`, never a bare `python` for repo scripts.
2. No parallel heavy compute and no subagent fan-outs (multi-agent code review included) without explicit user approval. "Quick" always means zero subagents. The one recognized exception is the deck **student review** (step 3 of the deck-polish loop), one agent per deck - it is a standard step, but still ask before launching it, and say how many decks and roughly what it will cost. (Froze the machine 2026-05-21; drained the usage quota 2026-07-06.)
3. pdflatex twice, always. One pass leaves a blank Outline frame and stale page counters.
4. `_quarto.yml` paths are case-sensitive on CI (Linux) even though Windows hides the mismatch locally.

## Knowledge index

| File | Contains |
|---|---|
| `CLAUDE.md` | environment, build, rules - always loaded |
| `WORKFLOWS.md` | this file |
| `ml/SLIDE_STYLE.md` | full deck style guide (single source of truth) |
| `homework_structure_guide.md` | homework `.qmd` format: YAML, difficulty, solution blocks |
| `LEARNINGS.md` | dated gotchas and incidents that have bitten before |
| `CONVENTIONS.md` | codified recurring decisions (naming, style, structure) |
| `_work_sessions/*.toml` | per-session logs (one TOML each); `PROGRESS.md` is the old read-only archive |
| `DEFERRED_TODO.md` | deferred-topics parking lot (older scratch list: `debt.md`) |
| `notes.md` | misc commands (Quarto solutions render, armtex header, deployment) |
| `ml/00_plan.md`, `ml/plan_manual.md` | ML course plan |
| `math/Lectures/stat/00_plan.md` | stat teaching plan |
| `_meta/youtube_channel.md` | video registry (all published videos) |
| `math/SOLUTIONS_STATUS.md` | per-qmd solutions tracker |
| `ml/MISSING_TOPICS.md`, `ml/CURRICULUM_GAPS_PRE_NN.md` | known curriculum gaps |
