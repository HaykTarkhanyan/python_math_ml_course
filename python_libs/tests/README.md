# Python Libraries Understanding Test

An intuition-focused self-assessment of the **Python libraries / tooling** section
(`python_libs/` 01-18): the data stack (NumPy, Pandas, viz), engineering practices
(logging, testing, parallelism, scraping), and backend + modern coding (SQL,
Pydantic, FastAPI, pathlib). Pure-language content is a separate test under
`../../python/tests/`.

English-only sibling of `../../math/tests/` (bilingual). Same pipeline.

## Files

| File | Role |
|---|---|
| `build_csv.py` | Authoring layer. Holds the questions as readable `q(...)` blocks and emits the CSV with loud validation. Edit questions here. |
| `python_libs_understanding_test.csv` | Generated source of truth (14 columns). Consumed by the Apps Script. Do not hand-edit if you plan to re-run `build_csv.py` - it will be overwritten. |
| `build_form.gs` | Google Apps Script that reads the CSV from Drive and builds the quiz Form. |

## Coverage

| Area | Notebooks |
|---|---|
| Data stack (NumPy / Pandas / viz) | `python_libs/01-07` |
| Engineering (logging / testing / parallel / scraping) | `python_libs/08-11` |
| Backend + modern (SQL / Pydantic / FastAPI / clean code) | `python_libs/12-18` |

## Editing questions

Edit the `q(...)` blocks in `build_csv.py`, then regenerate:

```bash
cd python_libs/tests
python build_csv.py
```

Validation is strict (fail-loud): duplicate id, bad answer letter, unknown
area/template, or any empty field raises and no CSV is written. On success it
logs the per-area and per-template counts.

The `q(...)` signature:

```python
q(id, area, module, difficulty, template,
  question,
  options,          # list of 4 option strings, in A,B,C,D order
  correct,          # "A" | "B" | "C" | "D"
  fb_ok,            # feedback string shown when correct
  fb_no,            # feedback string shown when wrong - addresses all 3 distractors
  points=1)
```

Allowed `area`: data_stack, engineering, backend_modern.
Allowed `template`: misconception, predict, flaw, match, compare.

## Building the Google Form

1. Upload `python_libs_understanding_test.csv` to your Google Drive.
2. Open it in Drive and copy the file id from the URL
   (`https://drive.google.com/file/d/<FILE_ID>/view`).
3. Go to https://script.google.com, create a new project, and paste the contents
   of `build_form.gs`.
4. Set `CSV_FILE_ID` at the top of the script to the id from step 2.
5. Run the `buildForm` function. Authorize when prompted (it needs Drive access
   to read the CSV and to create a Form).
6. Open **View > Logs** for the Form's **Edit URL** and **Published URL**.

The generated Form is in quiz mode (auto-graded), one section per area, with
per-question feedback on submit, and does not shuffle options (feedback refers to
A-D).

## Constraints worth knowing

- **Feedback is per question, not per option** (Forms limitation), so the
  incorrect feedback names what each distractor represents.
- **No option shuffling**, because feedback references A/B/C/D.
- **No native per-area subscore** - Forms reports one total; a per-area breakdown
  can be computed later from the linked responses Sheet via the `area` column.
- Keep the CSV **UTF-8**.

## Re-running

`buildForm` creates a **new** Form each run (it does not update an existing one).
Delete old drafts from Drive if they pile up.
