# "Slide N" means the printed footer frame number, not the PDF page

**Symptom.** The instructor asked to fix "slide 11's refresher table" and to look at "slide 25" in
`27_feature_selection.pdf`. Opening PDF pages 11 and 25 showed the wrong frames — page 11 was a
different topic entirely, and page 25 had nothing resembling what was described.

**Cause.** Beamer's footer shows a **frame** counter (`11 / 36`), while a PDF viewer shows a
**page** number. `\pause` and other overlay specifications emit one PDF page per overlay step but
increment the frame counter only once, so the two drift apart as soon as a deck uses `\pause`.

In deck 27 the drift was one page by frame 11 and one page by frame 25:

| Instructor says | Footer reads | Actual PDF page |
|---|---|---|
| slide 11 | `11 / 36` | 12 |
| slide 25 | `25 / 36` | 26 |

Deck 28 drifts further because it has more predict-first pauses: 51 frames across 59 PDF pages,
so by the end the offset is 8.

**Consequences.** Every instruction of the form "fix slide N" has to be resolved against the
**footer**, not the page number. Two ways to do that:

```bash
# find which PDF page carries a given frame number in its footer
pdftotext -layout DECK.pdf - | awk 'BEGIN{p=1} /\f/{p++} /^ *11 *\/ *36/{print "frame 11 is on page", p}'

# or go the other way: list every frame title with its PDF page
pdftotext -layout DECK.pdf - | awk 'BEGIN{p=1} /\f/{p++} /SOME TITLE/{print p": "$0}'
```

Do **not** assume they match, and do not quietly "correct" the instructor's number — confirm
against the rendered footer first, then act.

**Related trap already in the decks.** The same confusion is baked into deck source: nine
cross-references in decks 26 and 27 say things like "the multiple-testing correction from slide
20", and those numbers were inherited from the outline's *frame* numbering rather than measured
from the built PDF. They are still wrong as of 2026-08-12 and are logged in
`_work_sessions/2026-08-12-1530_*.toml` under `pending`. The durable fix is `\label`/`\ref` —
Beamer resolves `\ref` to the frame number, so it can never drift again.
