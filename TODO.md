# TODO

- [ ] **Review all 3 understanding-test question banks before sharing widely with students** _(added 2026-06-04)_

  Check question wording, correct answers, and the per-answer feedback for accuracy in each.
  The CSV is the source of truth; the readable place to review/edit is each `build_csv.py`.

  | Test | Source to review | Live form (edit) |
  |---|---|---|
  | Math (50 Q, bilingual) | `math/tests/build_csv.py` → `math/tests/math_understanding_test.csv` | https://docs.google.com/forms/d/1o0bWJSw55KIL7T7ZCD9v26QOVdWGh9Dh1xx4Ubq7CoM/edit |
  | Pure Python (40 Q) | `python/tests/build_csv.py` → `python/tests/python_understanding_test.csv` | https://docs.google.com/forms/d/1b5vNV3h-swnWSEwcvVzoR9IC6WU5Z8I01QkYklo7b8A/edit |
  | Python libraries (40 Q) | `python_libs/tests/build_csv.py` → `python_libs/tests/python_libs_understanding_test.csv` | https://docs.google.com/forms/d/1URAujDiJ1lv-lNvsI_4Wk6S7eA0PK1hYdLt9DYIEFLk/edit |

  To apply edits: change the `q(...)` blocks in `build_csv.py`, re-run `python build_csv.py`,
  then re-run `python make_selfcontained_gs.py` and rebuild the form (paste + run in Apps Script).

  Also: delete the old un-numbered Pure Python form once the numbered one above is confirmed
  (https://docs.google.com/forms/d/1Ax2FCVM6gzPd7Q8ojH5T4rGxb2E-twrUM_hOSqk5EeQ/edit).
