# Math Understanding Test

An intuition-focused self-assessment covering the whole math section (modules 00-30),
delivered as a bilingual (Armenian + English) auto-graded Google Form.

Design spec: `../../docs/superpowers/specs/2026-05-30-math-understanding-test-design.md`

## Files

| File | Role |
|---|---|
| `build_csv.py` | Authoring layer. Holds all questions as readable `q(...)` blocks and emits the CSV with loud validation. Edit questions here. |
| `math_understanding_test.csv` | Generated source of truth (21 columns). Consumed by the Apps Script. Do not hand-edit if you plan to re-run `build_csv.py` - it will be overwritten. |
| `build_form.gs` | Google Apps Script that reads the CSV from Drive and builds the quiz Form. |

## Editing questions

Edit the `q(...)` blocks in `build_csv.py`, then regenerate:

```bash
cd math/tests
python build_csv.py
```

Validation is strict (fail-loud): duplicate id, bad answer letter, unknown
area/template, or any empty field raises and no CSV is written. On success it
logs the per-area and per-template counts.

The `q(...)` signature:

```python
q(id, area, module, difficulty, template,
  question_hy, question_en,
  options,          # list of 4 (hy, en) tuples, in A,B,C,D order
  correct,          # "A" | "B" | "C" | "D"
  fb_ok,            # (hy, en) shown when correct
  fb_no,            # (hy, en) shown when wrong - addresses all 3 distractors
  points=1)
```

Allowed `area`: preliminaries, linear_algebra, calculus, optimization,
probability, statistics, info_theory, curse_dim.
Allowed `template`: misconception, predict, flaw, match, compare.

## Building the Google Form

1. Upload `math_understanding_test.csv` to your Google Drive.
2. Open it in Drive and copy the file id from the URL
   (`https://drive.google.com/file/d/<FILE_ID>/view`).
3. Go to https://script.google.com, create a new project, and paste the contents
   of `build_form.gs`.
4. Set `CSV_FILE_ID` at the top of the script to the id from step 2.
5. Run the `buildForm` function. Authorize the script when prompted (it needs
   access to Drive to read the CSV and to create a Form).
6. Open **View > Logs**. The script prints the Form's **Edit URL** and
   **Published URL**. Share the published URL with students.

The generated Form:

- is in quiz mode (auto-graded),
- has one section per math area in module order, with bilingual section titles,
- shows each question and its 4 options bilingually (`Armenian / English`),
- gives per-question feedback on submit (correct and incorrect), with the
  misconception explanations,
- does not shuffle options (the feedback refers to options A-D).

## Constraints worth knowing

- **Feedback is per question, not per option.** Google Forms allows one
  "correct" and one "incorrect" feedback block per question. The incorrect
  feedback therefore names what each distractor represents.
- **No option shuffling**, because the feedback text references A/B/C/D.
- **No native per-area subscore.** Forms reports one total score. A per-area
  breakdown can be computed later from the linked responses Sheet using the
  `area` column (not built here).
- Keep the CSV **UTF-8**; the script reads it as UTF-8 so the Armenian decodes.

## Re-running

Running `buildForm` again creates a **new** Form each time (it does not update an
existing one). Delete old drafts from Drive if they pile up.
