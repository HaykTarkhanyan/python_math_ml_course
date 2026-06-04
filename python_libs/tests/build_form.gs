/**
 * build_form.gs - Google Apps Script
 *
 * Builds an English auto-graded Google Form quiz from
 * python_libs_understanding_test.csv (the Python libraries / tooling test:
 * NumPy, Pandas, viz, logging, testing, parallelism, SQL, Pydantic, FastAPI).
 * One section per area, multiple-choice items with points and per-question feedback.
 *
 * Setup:
 *   1. Upload python_libs_understanding_test.csv to your Google Drive.
 *   2. Open the file in Drive, copy its file id from the URL, paste into CSV_FILE_ID below.
 *   3. Create a new Apps Script project (script.google.com), paste this file in.
 *   4. Run buildForm(). Authorize when prompted.
 *   5. The published + edit URLs are printed to the Apps Script log (View > Logs).
 *
 * Fails loudly: a malformed row (bad correct letter, empty option, unknown area,
 * wrong column count) throws an error naming the row; nothing is silently skipped.
 */

// ---- CONFIG ----------------------------------------------------------------
var CSV_FILE_ID = 'PASTE_CSV_FILE_ID_HERE';

var FORM_TITLE = 'Python Libraries Understanding Test';
var FORM_DESCRIPTION =
  'A self-assessment of your understanding of the Python data + engineering stack ' +
  '(NumPy, Pandas, testing, SQL, FastAPI, and more) - not memorization. ' +
  'Pick one answer. After submitting you will see the correct answer and an explanation.';

// Section titles, keyed by the CSV "area" code (rows are already in course order).
var AREA_TITLES = {
  data_stack: 'Data Stack (NumPy / Pandas / Viz)',
  engineering: 'Engineering Practices',
  backend_modern: 'Backend and Modern Coding'
};

var REQUIRED_COLUMNS = [
  'id', 'area', 'module', 'difficulty', 'template',
  'question',
  'optA', 'optB', 'optC', 'optD',
  'correct', 'points',
  'feedback_correct', 'feedback_wrong'
];
var LETTERS = ['A', 'B', 'C', 'D'];

// ---- MAIN ------------------------------------------------------------------
function buildForm() {
  if (CSV_FILE_ID === 'PASTE_CSV_FILE_ID_HERE') {
    throw new Error('Set CSV_FILE_ID to your uploaded CSV file id first.');
  }

  var rows = readCsv_(CSV_FILE_ID);
  var header = rows[0];
  var idx = headerIndex_(header);

  var form = FormApp.create(FORM_TITLE);
  form.setDescription(FORM_DESCRIPTION);
  form.setIsQuiz(true);

  var currentArea = null;
  var count = 0;

  for (var r = 1; r < rows.length; r++) {
    var row = rows[r];
    if (row.length === 1 && row[0] === '') continue;  // trailing blank line
    if (row.length !== header.length) {
      throw new Error('Row ' + r + ': expected ' + header.length +
                      ' columns, got ' + row.length);
    }

    var get = function (name) { return row[idx[name]]; };
    var id = get('id');
    var area = get('area');

    if (AREA_TITLES[area] === undefined) {
      throw new Error(id + ': unknown area "' + area + '"');
    }

    // New section when the area changes.
    if (area !== currentArea) {
      form.addPageBreakItem().setTitle(AREA_TITLES[area]);
      currentArea = area;
    }

    var correct = get('correct');
    if (LETTERS.indexOf(correct) === -1) {
      throw new Error(id + ': correct must be A-D, got "' + correct + '"');
    }

    var item = form.addMultipleChoiceItem();
    item.setTitle(get('question'));
    item.setRequired(true);

    var choices = [];
    for (var c = 0; c < LETTERS.length; c++) {
      var L = LETTERS[c];
      var opt = get('opt' + L);
      if (opt === '') {
        throw new Error(id + ': option ' + L + ' is empty');
      }
      choices.push(item.createChoice(opt, L === correct));
    }
    item.setChoices(choices);

    var points = parseInt(get('points'), 10);
    if (isNaN(points) || points < 1) {
      throw new Error(id + ': points must be a positive integer');
    }
    item.setPoints(points);

    item.setFeedbackForCorrect(
      FormApp.createFeedback().setText(get('feedback_correct')).build());
    item.setFeedbackForIncorrect(
      FormApp.createFeedback().setText(get('feedback_wrong')).build());

    count++;
  }

  Logger.log('Built quiz with ' + count + ' questions.');
  Logger.log('Edit URL:      ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
}

// ---- HELPERS ---------------------------------------------------------------
function readCsv_(fileId) {
  var blob = DriveApp.getFileById(fileId).getBlob();
  var text = blob.getDataAsString('UTF-8');
  var rows = Utilities.parseCsv(text);
  if (!rows || rows.length < 2) {
    throw new Error('CSV is empty or has no data rows.');
  }
  return rows;
}

function headerIndex_(header) {
  var idx = {};
  for (var i = 0; i < header.length; i++) idx[header[i]] = i;
  var missing = [];
  for (var j = 0; j < REQUIRED_COLUMNS.length; j++) {
    if (idx[REQUIRED_COLUMNS[j]] === undefined) missing.push(REQUIRED_COLUMNS[j]);
  }
  if (missing.length) {
    throw new Error('CSV missing required column(s): ' + missing.join(', '));
  }
  return idx;
}
