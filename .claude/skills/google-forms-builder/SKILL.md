---
name: google-forms-builder
description: >-
  Use when the user wants to create, build, generate, or populate a Google Form or
  quiz - especially from existing questions, a CSV, a spreadsheet, or a question bank -
  or says things like "turn these into a Google Form", "make a quiz form", "Google Forms
  from CSV", or wants an auto-graded quiz with answer keys, points, and per-answer
  feedback. Also use when adding many questions by clicking through the Google Forms UI
  would be slow or error-prone. Covers multiple-choice quizzes, sections, and bilingual forms.
---

# Google Forms Builder

Create a Google Form (especially an auto-graded quiz) from a structured question bank.

## The approach

Google Forms has **no native CSV/spreadsheet import** — the built-in "Import questions" only
pulls from another existing Form, and third-party add-ons want broad Drive access and are
flaky. Use **Google Apps Script** (`FormApp`) instead, as a **self-contained `.gs`**: a script
that embeds the questions as a JavaScript array and builds the form, which you paste into a
fresh Apps Script project and run. Self-contained (data embedded) beats reading a CSV from
Drive because it skips the upload/file-id step and keeps the OAuth scope to **Forms only** (no
Drive).

Don't drive the Forms editor UI question-by-question for a real set — a 40-question quiz with
answer keys and feedback is ~500+ fragile interactions against an obfuscated single-page app.
(For a throwaway 2-3 question form, doing it by hand is of course fine.)

## Workflow

1. **Get the questions into a CSV** (or a list you write to CSV) — one row per question:
   question text, the option texts, the correct-answer letter, points, correct-feedback,
   wrong-feedback, and optionally a `section`/`area` column. For a quiz, the wrong-feedback is
   most useful when each distractor is a *named misconception* the feedback explains (Forms
   gives one feedback block per question, not per option).

2. **Generate the `.gs`** with `scripts/generate_form_gs.py config.json`. The JSON config sets
   the title/description, quiz on/off, column names, optional sections, numbering, and optional
   bilingual joins. See `scripts/example_config.json` (mono) and `example_config_bilingual.json`
   (two columns per field). The script does all text assembly in Python, so the emitted `.gs` is
   the same simple template regardless of language.

3. **Build it in the browser** with Playwright. Follow `references/playwright-runbook.md` for the
   exact ordered steps and verification — open `script.new`, paste the `.gs` via the OS clipboard,
   save, run, the user authorizes, then read the execution log for the Edit + Published URLs.

4. **Verify before claiming success**: confirm the log says `Execution completed` with no error,
   and read the rendered question titles back from the form (count them, check `Total points`)
   rather than assuming.

## Gotchas (the make-or-break bits; full detail in the runbook)

- **Paste, never type, into the Apps Script editor.** It's Monaco — typing fires auto-indent and
  bracket-matching that corrupt the code; pasting from the OS clipboard (`Ctrl/Cmd+V`) inserts
  verbatim. Verify with `monaco.editor.getModels()[0].getValue()` before saving.
- **UTF-8 for non-ASCII** (Armenian, accents): copy the file as UTF-8 (Windows PowerShell
  `Get-Content -Raw -Encoding UTF8`) or it becomes mojibake on the clipboard; verify a known
  substring survived before pasting.
- **Reuse ONE Apps Script project to build several forms with no re-auth.** Authorization is per
  project per scope set — same Forms scope means you can swap in different `buildForm()` code,
  save, and run again with no second consent prompt.
- **Authorization is the user's call.** The first run prompts for the Forms scope (and an
  "unverified app" interstitial for a personal script); let the user approve it, don't automate
  the consent.
- **Quiz feedback is per question, not per option** (`setFeedbackForCorrect` /
  `setFeedbackForIncorrect`) — so write the wrong-answer feedback to name what each distractor is.
- **Sections** come from `addPageBreakItem()`, and the title/description is its own first page, so
  N areas produce N+1 pages. Keep rows grouped by section (sort by a fixed area order).
- **Timing**: 40-50 questions take ~60-90s; poll the execution log for the `Built N` / Edit /
  Published lines rather than assuming it finished.

## When you need to hand-write or extend the builder

`references/formapp-api.md` has the `FormApp` cheatsheet and a hand-editable `.gs` template — use
it to add a non-multiple-choice question type, an image, a different grading rule, etc.

## Files

- `scripts/generate_form_gs.py` — config-driven generator: CSV + JSON config → self-contained `build_form.gs`.
- `scripts/example_config.json`, `scripts/example_config_bilingual.json` — annotated example configs.
- `references/playwright-runbook.md` — exact, ordered browser-automation steps with verification.
- `references/formapp-api.md` — `FormApp` cheatsheet and a hand-editable `.gs` template.
