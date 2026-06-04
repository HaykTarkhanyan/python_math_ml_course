"""Generate a self-contained Apps Script (questions embedded) for the math test.

Bilingual (Armenian + English): each item shows both languages. Produces
build_form_selfcontained.gs that you paste into an Apps Script project and run
buildForm() - no Drive upload needed. Question titles are numbered 1., 2., ...

Run: python make_selfcontained_gs.py
"""

import csv
import json
import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "make_selfcontained_gs.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("make_selfcontained_gs")

HERE = Path(__file__).parent
CSV = HERE / "math_understanding_test.csv"
OUT = HERE / "build_form_selfcontained.gs"

FORM_TITLE = "Մաթեմատիկայի ըմբռնման թեստ / Math Understanding Test"
FORM_DESCRIPTION = (
    "Ինքնագնահատման թեստ՝ ստուգելու մաթեմատիկայի ինտուիցիան (ոչ հաշվարկը): "
    "Ընտրիր մեկ պատասխան: Ուղարկելուց հետո կտեսնես ճիշտ պատասխանը և բացատրությունը:\n\n"
    "A self-assessment of your math intuition (not calculation). Pick one answer. "
    "After submitting you will see the correct answer and an explanation."
)
AREA_TITLES = {
    "preliminaries": "Նախապատրաստություն / Preliminaries",
    "linear_algebra": "Գծային հանրահաշիվ / Linear Algebra",
    "calculus": "Մաթեմատիկական անալիզ / Calculus",
    "optimization": "Օպտիմիզացիա / Optimization",
    "probability": "Հավանականություն / Probability",
    "statistics": "Վիճակագրություն / Statistics",
    "info_theory": "Ինֆորմացիայի տեսություն / Information Theory",
    "curse_dim": "Չափողականության անեծք / Curse of Dimensionality",
}


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows_json = json.dumps(rows, ensure_ascii=False)
    area_json = json.dumps(AREA_TITLES, ensure_ascii=False)

    gs = f"""/** AUTO-GENERATED self-contained quiz builder (math test, bilingual).
 * Paste into an Apps Script project (script.google.com) and run buildForm().
 */
var FORM_TITLE = {json.dumps(FORM_TITLE)};
var FORM_DESCRIPTION = {json.dumps(FORM_DESCRIPTION)};
var AREA_TITLES = {area_json};
var LETTERS = ['A', 'B', 'C', 'D'];
var ROWS = {rows_json};

function buildForm() {{
  var form = FormApp.create(FORM_TITLE);
  form.setDescription(FORM_DESCRIPTION);
  form.setIsQuiz(true);
  var currentArea = null, count = 0;
  for (var i = 0; i < ROWS.length; i++) {{
    var r = ROWS[i];
    if (AREA_TITLES[r.area] === undefined) throw new Error(r.id + ': unknown area ' + r.area);
    if (r.area !== currentArea) {{
      form.addPageBreakItem().setTitle(AREA_TITLES[r.area]);
      currentArea = r.area;
    }}
    if (LETTERS.indexOf(r.correct) === -1) throw new Error(r.id + ': bad correct ' + r.correct);
    var item = form.addMultipleChoiceItem();
    item.setTitle((i + 1) + '. ' + r.question_hy + '\\n' + r.question_en);
    item.setRequired(true);
    var choices = [];
    for (var c = 0; c < LETTERS.length; c++) {{
      var L = LETTERS[c];
      choices.push(item.createChoice(r['opt' + L + '_hy'] + ' / ' + r['opt' + L + '_en'], L === r.correct));
    }}
    item.setChoices(choices);
    item.setPoints(parseInt(r.points, 10) || 1);
    item.setFeedbackForCorrect(FormApp.createFeedback().setText(r.feedback_correct_hy + '\\n\\n' + r.feedback_correct_en).build());
    item.setFeedbackForIncorrect(FormApp.createFeedback().setText(r.feedback_wrong_hy + '\\n\\n' + r.feedback_wrong_en).build());
    count++;
  }}
  Logger.log('Built ' + count + ' questions.');
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
}}
"""

    OUT.write_text(gs, encoding="utf-8")
    log.info(f"Wrote {OUT.name} ({len(gs)} chars, {len(rows)} questions)")


if __name__ == "__main__":
    sys.exit(main())
