# FormApp API cheatsheet + hand-editable template

Use this when the generator (`scripts/generate_form_gs.py`) doesn't cover what you need — a
non-multiple-choice question type, images, a different grading rule, shuffles, etc. — and you
want to hand-write or tweak the `.gs`.

Docs: https://developers.google.com/apps-script/reference/forms

## Key calls

| Goal | Call |
|---|---|
| Create a form | `var form = FormApp.create('Title')` |
| Description | `form.setDescription('...')` |
| Make it an auto-graded quiz | `form.setIsQuiz(true)` |
| Add a multiple-choice question | `var item = form.addMultipleChoiceItem()` |
| Title / required | `item.setTitle('...')` ; `item.setRequired(true)` |
| Build a choice (mark correct) | `item.createChoice('text', isCorrectBool)` |
| Set the choices | `item.setChoices([c1, c2, c3, c4])` |
| Points (quiz) | `item.setPoints(1)` |
| Feedback for correct / incorrect | `item.setFeedbackForCorrect(fb)` ; `item.setFeedbackForIncorrect(fb)` |
| Build a feedback object | `FormApp.createFeedback().setText('...').build()` |
| Start a new section | `form.addPageBreakItem().setTitle('Section title')` |
| Other item types | `addCheckboxItem`, `addTextItem`, `addParagraphTextItem`, `addScaleItem`, `addGridItem`, `addImageItem`, `addListItem`, ... |
| URLs (log these) | `form.getEditUrl()` ; `form.getPublishedUrl()` |
| Email collection | `form.setCollectEmail(true)` (default off = anonymous) |

### Critical constraints (Forms, not Apps Script)
- **Feedback is per question, not per option.** There is one correct-feedback and one
  incorrect-feedback block per item — not one per choice. Write the incorrect feedback to name
  what each distractor represents.
- **Choice order is fixed; `FormApp` cannot shuffle choices.** Per-choice shuffling exists
  only in the Forms REST API (`shuffle: true`), not in Apps Script — `setChoices` /
  `setChoiceValues` render in the order you pass. That's fine here: keep options in order so
  the per-question feedback can reference A/B/C/D.
- The form's **title/description is its own first page**; each `addPageBreakItem()` starts a
  new section after it. N sections of questions ⇒ N+1 pages.
- Keep the script **Forms-only** (no `DriveApp`/`SpreadsheetApp`) so the OAuth consent asks for
  the Forms scope only.

## Minimal hand-editable template

Embed the data as `ROWS` (each row: `question`, `options[]`, `correctIndex`, optional
`points`, `fbc`, `fbw`, and optional `area`). Build it from your data however you like, then
paste this and run `buildForm()`.

```js
var FORM_TITLE = 'My Quiz';
var FORM_DESCRIPTION = 'Pick one answer.';
var IS_QUIZ = true;
var NUMBERED = true;                 // prefix titles "1. ", "2. "
var USE_SECTIONS = true;
var SECTION_TITLES = { intro: 'Section One' };
var ROWS = [
  { area: 'intro', question: 'What is 2 + 2?', options: ['3', '4', '5', '22'],
    correctIndex: 1, points: 1,
    fbc: 'Correct.', fbw: 'It is 4; "22" is string concatenation, not addition.' }
];

function buildForm() {
  var form = FormApp.create(FORM_TITLE);
  if (FORM_DESCRIPTION) form.setDescription(FORM_DESCRIPTION);
  if (IS_QUIZ) form.setIsQuiz(true);
  var currentArea = null, n = 0;
  for (var i = 0; i < ROWS.length; i++) {
    var r = ROWS[i];
    if (USE_SECTIONS && r.area !== currentArea) {
      form.addPageBreakItem().setTitle(SECTION_TITLES[r.area] != null ? SECTION_TITLES[r.area] : r.area);
      currentArea = r.area;
    }
    n++;
    var item = form.addMultipleChoiceItem();
    item.setTitle(NUMBERED ? (n + '. ' + r.question) : r.question);
    item.setRequired(true);
    var choices = [];
    for (var c = 0; c < r.options.length; c++) {
      choices.push(item.createChoice(r.options[c], c === r.correctIndex));
    }
    item.setChoices(choices);
    if (IS_QUIZ) {
      if (r.points != null) item.setPoints(r.points);
      if (r.fbc) item.setFeedbackForCorrect(FormApp.createFeedback().setText(r.fbc).build());
      if (r.fbw) item.setFeedbackForIncorrect(FormApp.createFeedback().setText(r.fbw).build());
    }
  }
  Logger.log('Built ' + n + ' questions.');
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
}
```

## Bilingual / multi-line text
Put both languages in one string with a newline: `item.setTitle('Հայերեն\nEnglish')`. For
choices, a single line reads better: `'հայերեն / English'`. The generator does this for you
when a field maps to 2+ CSV columns (see `joins` in the config).
