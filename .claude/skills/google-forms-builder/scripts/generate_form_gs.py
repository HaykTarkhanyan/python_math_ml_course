"""Generate a self-contained Google Apps Script that builds a Google Form / quiz.

Reads a CSV question bank + a JSON config, and emits one `build_form.gs` with the
questions embedded as a JS array and a `buildForm()` that creates the form via
FormApp. Paste the output into a new Apps Script project (script.new) and run it.

Usage:
    python generate_form_gs.py config.json

All text assembly (bilingual joins, etc.) happens here in Python, so the emitted
.gs is always the same simple template. Validation fails loudly before writing.

Config schema (see example_config.json):
{
  "csv_path": "questions.csv",         # required
  "output_path": "build_form.gs",      # default: build_form.gs next to the config
  "form_title": "My Quiz",             # required
  "form_description": "",              # optional
  "is_quiz": true,                     # default true: auto-graded with points/feedback
  "number_questions": true,            # default false: prefix titles with "1. ", "2. "
  "joins": {"question": "\\n", "option": " / ", "feedback": "\\n\\n"},  # for multi-column fields
  "columns": {                         # each text field is a LIST of CSV columns
    "question": ["question"],          #   1 column = monolingual; 2+ = joined (e.g. bilingual)
    "options":  [["optA"],["optB"],["optC"],["optD"]],
    "correct":  "correct",             # column with A/B/C/D (or a 1-based number)
    "points":   "points",              # column name, or null for a flat 1 point
    "feedback_correct": ["feedback_correct"],   # list of columns, or null
    "feedback_wrong":   ["feedback_wrong"]      # list of columns, or null
  },
  "section_column": "area",            # optional: start a new Form section when this changes
  "section_titles": {"area1": "Title 1"}  # optional: code -> section title (falls back to code)
}
"""

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("generate_form_gs")

DEFAULT_JOINS = {"question": "\n", "option": " / ", "feedback": "\n\n"}


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def join_cols(row, cols, sep, where):
    parts = []
    for c in cols:
        if c not in row:
            fail(f"{where}: CSV has no column '{c}'")
        parts.append((row[c] or "").strip())
    return sep.join(p for p in parts if p != "")


def correct_to_index(value, n_options, where):
    v = (value or "").strip()
    if len(v) == 1 and v.upper().isalpha():
        idx = ord(v.upper()) - ord("A")
    elif v.isdigit():
        idx = int(v) - 1  # 1-based
    else:
        fail(f"{where}: 'correct' must be a letter (A..) or 1-based number, got '{value}'")
    if not (0 <= idx < n_options):
        fail(f"{where}: correct index {idx} out of range for {n_options} options")
    return idx


def build_rows(cfg):
    csv_path = Path(cfg["csv_path"])
    if not csv_path.exists():
        fail(f"csv_path not found: {csv_path}")
    joins = {**DEFAULT_JOINS, **cfg.get("joins", {})}
    cols = cfg["columns"]
    section_col = cfg.get("section_column")

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        fail("CSV has no data rows")

    out = []
    for i, r in enumerate(rows):
        where = f"row {i + 1}"
        options = [
            join_cols(r, opt_cols, joins["option"], f"{where} option")
            for opt_cols in cols["options"]
        ]
        if any(o == "" for o in options):
            fail(f"{where}: an option is empty")
        row = {
            "question": join_cols(r, cols["question"], joins["question"], f"{where} question"),
            "options": options,
            "correctIndex": correct_to_index(r.get(cols["correct"]), len(options), where),
        }
        if not row["question"]:
            fail(f"{where}: question is empty")
        if cfg.get("is_quiz", True):
            pts_col = cols.get("points")
            pts = r.get(pts_col) if pts_col else None
            row["points"] = int(pts) if (pts not in (None, "")) else 1
            if cols.get("feedback_correct"):
                row["fbc"] = join_cols(r, cols["feedback_correct"], joins["feedback"], f"{where} fbc")
            if cols.get("feedback_wrong"):
                row["fbw"] = join_cols(r, cols["feedback_wrong"], joins["feedback"], f"{where} fbw")
        if section_col:
            if section_col not in r:
                fail(f"{where}: section_column '{section_col}' not in CSV")
            row["area"] = r[section_col]
        out.append(row)
    return out


GS_TEMPLATE = """/** AUTO-GENERATED self-contained Google Form builder.
 * Paste into a new Apps Script project (script.new) and run buildForm().
 * No Drive upload needed - the questions are embedded below.
 */
var FORM_TITLE = {title};
var FORM_DESCRIPTION = {description};
var IS_QUIZ = {is_quiz};
var NUMBERED = {numbered};
var USE_SECTIONS = {use_sections};
var SECTION_TITLES = {section_titles};
var ROWS = {rows};

function buildForm() {{
  var form = FormApp.create(FORM_TITLE);
  if (FORM_DESCRIPTION) form.setDescription(FORM_DESCRIPTION);
  if (IS_QUIZ) form.setIsQuiz(true);
  var currentArea = null, n = 0;
  for (var i = 0; i < ROWS.length; i++) {{
    var r = ROWS[i];
    if (USE_SECTIONS && r.area !== currentArea) {{
      form.addPageBreakItem().setTitle(SECTION_TITLES[r.area] != null ? SECTION_TITLES[r.area] : r.area);
      currentArea = r.area;
    }}
    n++;
    var item = form.addMultipleChoiceItem();
    item.setTitle(NUMBERED ? (n + '. ' + r.question) : r.question);
    item.setRequired(true);
    var choices = [];
    for (var c = 0; c < r.options.length; c++) {{
      choices.push(item.createChoice(r.options[c], c === r.correctIndex));
    }}
    item.setChoices(choices);
    if (IS_QUIZ) {{
      if (r.points != null) item.setPoints(r.points);
      if (r.fbc) item.setFeedbackForCorrect(FormApp.createFeedback().setText(r.fbc).build());
      if (r.fbw) item.setFeedbackForIncorrect(FormApp.createFeedback().setText(r.fbw).build());
    }}
  }}
  Logger.log('Built ' + n + ' questions.');
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
}}
"""


def main():
    if len(sys.argv) != 2:
        fail("usage: python generate_form_gs.py config.json")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for req in ("csv_path", "form_title", "columns"):
        if req not in cfg:
            fail(f"config missing required key: {req}")

    rows = build_rows(cfg)
    out_path = Path(cfg.get("output_path", "build_form.gs"))

    gs = GS_TEMPLATE.format(
        title=json.dumps(cfg["form_title"]),
        description=json.dumps(cfg.get("form_description", "")),
        is_quiz="true" if cfg.get("is_quiz", True) else "false",
        numbered="true" if cfg.get("number_questions", False) else "false",
        use_sections="true" if cfg.get("section_column") else "false",
        section_titles=json.dumps(cfg.get("section_titles", {}), ensure_ascii=False),
        rows=json.dumps(rows, ensure_ascii=False),
    )
    out_path.write_text(gs, encoding="utf-8")
    log.info(f"Wrote {out_path} ({len(gs)} chars, {len(rows)} questions)")


if __name__ == "__main__":
    main()
