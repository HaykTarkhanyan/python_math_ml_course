---
name: review-submission
description: >
  Use when the user shares a student's project submission for review - a GitHub repo or
  notebook link, a zip, or a file - or says "new submission", "here is the submission",
  "review X's project", "check this student's work". Sets up the private
  _submissions/NN_task/student/ folder, finds the brief and the reference solution,
  verifies every claim by re-running code, and writes FEEDBACK.md in the house format.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Glob
  - Grep
user-invocable: true
---

# Reviewing a student project submission

The layout and conventions are in `_submissions/README.md`. That folder is gitignored
because the course repo is public and the reviews name students and criticise their work.
**Nothing about a student goes into a tracked file**: not this skill, not `_learnings/`,
`_work_sessions/`, `DECISIONS.md` or a commit message. A session log may say "reviewed a
ch11 submission", without a name.

```
_submissions/NN_task/
  REVIEW_SUMMARY.md        <- cross-student walkthrough, written only once several are in
  <student>/               <- firstname_lastname, or the GitHub handle if that is all we have
    FEEDBACK.md            <- the review
    submission/            <- their work exactly as received (git clone / file / zip)
    evidence/              <- verify_claims.py, its .json and .log, any extracted old versions
```

## 1. Identify the project

`NN` is the chapter number from `ml/`. Existing task folders (`ls _submissions/`) are the
registry. Reuse a folder if one exists, otherwise add a row here.

| Task folder | Brief | Reference solution |
|---|---|---|
| `09_image_compression` | `ml/09_clustering/09_clustering.qmd` (image-compression project) | `ml/09_clustering/35_image_compression_solution.ipynb` |
| `10_clip_clustering` | `ml/10_dimensionality_reduction/build_your_own_google_photos_hy.md` | `ml/10_dimensionality_reduction/37_image_clusters_solution.ipynb` |
| `11_name_inventor` | `ml/11_neural_networks/name_inventor_hy.md` | `ml/11_neural_networks/47_name_inventor_solution.ipynb` |

For a new project: the brief is the chapter `.qmd`'s `Գործնական` section or a sibling
`*_hy.md`, and the solution is `ml/NN_*/*_solution.ipynb`. If two candidates fit, ask.

## 2. Get the submission and pin what you reviewed

```bash
mkdir -p _submissions/NN_task/<student>/evidence
git clone --depth 50 <url> _submissions/NN_task/<student>/submission
git -C <clone> log --oneline -1                     # the commit the review is about
git -C <clone> ls-remote --heads origin             # side branches
gh api "repos/OWNER/REPO/compare/main...BRANCH" --jq '.ahead_by, .files[].filename'
git -C <clone> log --stat --date=iso -- "<project dir>"   # how the work evolved
```

- Students keep one repo for every project, so look at the **whole project folder**, not
  only the linked file. Second notebooks, empty files, saved checkpoints and requirements
  files all count.
- If a "clean up" commit rewrote the notebook, extract the first version into `evidence/`
  (`git show SHA:path > evidence/original_SHA.ipynb`). The original shows which decisions
  were the student's own.

## 3. Read the context before the submission

1. **The brief, in full.** Its tasks or questions are the rubric. The review gets a table
   with one row per task/question.
2. **The reference solution's key numbers**: baselines, headline metric, noise floor. Pull
   markdown headings and stream outputs with a short script instead of reading 100+ cells.
3. **This student's earlier reviews**: `ls _submissions/*/<student>/FEEDBACK.md`. List
   every recurring note (Run All, `.DS_Store`, README, pinned requirements, conclusion
   sentences...) and check each one again. Say plainly what has improved since last time,
   and flag a note that is repeated for the third time as such.

## 4. Read the submission

Dump each notebook with its outputs (cells, `execution_count`, stream text, errors). Check:

- Execution counts: does it start at 1 and run in order? If not, it was never run top to
  bottom (Run All).
- Every task/question in the brief: answered in code, answered in prose, or missing.
- Numbers presented without a comparison point: a loss with no baseline, a score with no
  noise estimate, a "98% novel" with no quality measure.

## 5. Verify, don't eyeball

Every criticism that rests on a number gets measured before it goes in. A wrong criticism
is worse than a missing one. Put the probes in `evidence/verify_claims.py`, have it write
`verify_claims.json`, and quote numbers from that file.

- **Reproduce their headline number first**, from their saved model if they committed one,
  on their exact split. If it does not reproduce, that is the first finding.
- **Compute the brief's baselines on their split**, using the reference solution's method,
  so their number has something to be read against.
- **Architectural criticism needs a controlled run**: change one thing at a time against
  their own config and protocol, with 3 seeds each so the spread shows the noise floor.
- **Measure the output, not only the loss**: for generators, sample from their model and
  count what the brief cares about.
- Keep it light: one process, `torch.set_num_threads(2)`, runtime stated up front. Ask
  before anything heavier (global CLAUDE.md). Time one epoch before guessing: a 9-run x
  50-epoch probe on 5k examples took 12 min, against a guess of 3-6.
- Run probes with `python -u` and print progress with an ETA. Piped through `tee`, plain
  `python` block-buffers everything until exit, so "eta?" has no answer.
- Windows gotchas: build label arrays with `dtype=np.int64` (numpy defaults to int32 here),
  and use `set -o pipefail` when piping a probe through `tee`, or a crash exits 0.

## 6. Write `FEEDBACK.md`

The house format, as in the existing reviews. Address the student as "you", in English,
with plain hyphens and straight quotes. Use they/them for students who have not stated
their pronouns; a handle or a first name is not a statement of pronouns.

```markdown
# Feedback: `<notebook>` (<student>)

**Submission:** <repo url> - `<project dir>/`
**Assignment:** `<brief path>` - *<project title>*
**Reviewed:** YYYY-MM-DD - commit `<sha>`
**Reference solution:** `<path>`

## Summary            <- verdict in 2-4 paragraphs, then "**How this was verified.**"
## What is good       <- specific and earned; improvements since the last review go here
## The main problem   <- the one finding that changes the grade, with measured numbers
## Correctness and method   <- numbered findings, each with evidence and a concrete fix
## Housekeeping       <- small items; mark recurring ones "same note as last time"
## What to do, in order     <- 3-8 steps, the biggest payoff first, then "**Bonus:** ..." items
```

**Instructor preferences** (stated 2026-10-01, on the ch11 review):

- **No "Against the brief" status table.** Cover the brief's questions inside the findings and in
  "What to do". Use the brief internally as the checklist, but don't print it as a table.
- **Multi-seed reruns are a bonus, not a requirement.** Report the measured seed spread, and
  suggest "re-run the top configs with 3 seeds" under **Bonus**, not as a main finding.
- **Neural nets: ReLU in hidden layers, not tanh/sigmoid.** Lecture [45]: "ReLU by default",
  "tanh: avoid, they saturate and vanish the gradient".
- **Dropout only once the curves show overfitting.** Start at 0, compare train vs val, then add
  plenty of it if needed. Lecture [43]: "Diagnose from the curves, then turn one knob." Check
  *how* the student measures train loss: in `model.train()` mode, dropout inflates it and hides
  the gap.

Findings carry their evidence inline: the measured number, the reproduced output, the
two-line fix. **Label every number by whose run it came from.** "Your notebook printed X" must
quote their stored output. Numbers from your re-run are "in my re-run". When the two differ
(unseeded shuffles, another torch version), say so, or the student will distrust the whole table.
Finish with a separate read of the whole file against `verify_claims.json` and their stored
outputs. The first draft of the ch11 review had two re-run numbers presented as the student's,
and a "bigger than anything in your grid" claim that the grid itself contradicted. "I checked and it does not change your answer" is worth writing too; it
stops the student chasing a non-issue.

**Instructor-only observations** (for example, a polished rewrite that may be LLM-made,
which the brief allows only if the student can defend every line) go in the chat report,
never in `FEEDBACK.md`. Phrase them as something to ask in class, not as an accusation.

## 7. Report back

In chat: the verdict in 2-3 sentences, the top findings with their numbers, the
instructor-only notes, and the path to `FEEDBACK.md`. Do not contact the student, open an
issue or comment on their repo unless asked. That publishes to someone else.

Write `REVIEW_SUMMARY.md` only once a task has several reviewed submissions. Its value is
in the cross-student patterns.
