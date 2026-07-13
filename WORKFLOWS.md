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
| Compile / verify any `.tex` | `compile-deck` skill | - |
| Generate figures for a deck | `make-figures` skill | `ml/SLIDE_STYLE.md` (Figures section) |
| Hunt clipped / overflowing slides | `beamer-overflow-check` skill | - |
| Create or edit a math homework `.qmd` (`math/XX_*.qmd`) | follow `homework_structure_guide.md` for format, but note its `math/Homeworks/` path examples are stale - files live directly under `math/`; register in `_quarto.yml` (exact case) | `homework_structure_guide.md` |
| Create or edit an ML chapter page (`ml/NN_topic/NN_topic.qmd`) | follow the QMD structure template in `CONVENTIONS.md`; register in `_quarto.yml` (exact case) | `CONVENTIONS.md` (QMD structure) |
| Solutions package (LaTeX PDF + ipynb) | `generate-solutions` skill | `math/SOLUTIONS_STATUS.md` |
| Collapsible solutions inside a `.qmd` | `add-inline-solutions` skill | `math/SOLUTIONS_STATUS.md` |
| Add YouTube / video links after a session | `update-youtube` skill | `_meta/youtube_channel.md` |
| Build a quiz Google Form | `google-forms-builder` skill | - |
| Delete LaTeX build junk | `clean-tex` skill or `clean_latex.py` | - |
| End of session / commit the day's work | `wrap-session` skill | `PROGRESS.md` |

## Definition of done

| Artifact | Done means |
|---|---|
| Slide deck | 2x pdflatex passes, 0 `!` lines in the `.log`, no `end{center>`-style typos, overflow checked visually, aux files cleaned, `% Provenance:` block present (`ml/` decks only - stat/optim decks don't use them) |
| Figure script | runs end-to-end under the `ma` venv, PDFs in sibling `fig/`, log in `logs/`, figures actually embedded in the deck and the deck recompiled |
| Homework `.qmd` | registered in `_quarto.yml` with exact-case path, blank line before every list / blockquote / fence, difficulty markers set |
| Commit | no `.aux`/`.log`/`.nav` staged, message explains the change, push only when asked |

## Hard rules (each of these has caused real damage)

1. Python runs through `./ma/Scripts/python.exe`. Never `uv run --with ...`, never a bare `python` for repo scripts.
2. No parallel heavy compute and no subagent fan-outs (multi-agent code review included) without explicit user approval. "Quick" always means zero subagents. (Froze the machine 2026-05-21; drained the usage quota 2026-07-06.)
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
| `PROGRESS.md` | session log: done / pending / next |
| `DEFERRED_TODO.md` | deferred-topics parking lot (older scratch list: `debt.md`) |
| `notes.md` | misc commands (Quarto solutions render, armtex header, deployment) |
| `ml/00_plan.md`, `ml/plan_manual.md` | ML course plan |
| `math/Lectures/stat/00_plan.md` | stat teaching plan |
| `_meta/youtube_channel.md` | video registry (all published videos) |
| `math/SOLUTIONS_STATUS.md` | per-qmd solutions tracker |
| `ml/MISSING_TOPICS.md`, `ml/CURRICULUM_GAPS_PRE_NN.md` | known curriculum gaps |
