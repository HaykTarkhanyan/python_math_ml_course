# The clipped-slide detector works. The bug survived because nobody ran it

**Symptom.** `32_clustering.tex`'s Lloyd animation frame silently lost its closing line -
"Converges --- but only to a **local** optimum (the same alternating idea as EM, later)" - on all
**6** overlay pages. The frame had shipped that way since the deck was written in June 2026, and
had survived a full content review (2026-07-07) and a Sonnet student review of the rendered pages
(2026-08-09). I found it by accident, eyeballing the page while checking something unrelated.

**Cause.** The figure sat at `width=0.6\textwidth`. Bisected against the pristine original from
git, changing **only** one thing per variant:

```
A_pristine_original      footnote in text layer: NO
B_longer_title_only      footnote in text layer: NO
C_new_footnote_only      footnote in text layer: NO
D_width_049_only         footnote in text layer: YES
```

So it is purely a height budget: title + 3 lines of body + a figure that tall leaves no room, and
Beamer drops the remainder. `0.49` fits. Nothing about the wording mattered.

**The compile log is useless here, exactly as `ml/SLIDE_STYLE.md` says:**

```
$ grep -c "Overfull \\vbox" 32_clustering.log
0
```

Zero warnings, correct page count, exit 0.

## The actual lesson

**`non_essential/detect_clipped_slides.py` catches this, in about two seconds.** Run against the
pristine broken original it prints:

```
  frame  7 [Lloyd's algorithm: assign, update, repeat]  (page 7)
      NOT RENDERED: its cluster mean watch it converge converges but only to
                    local optimum the same alternating idea as later
```

That is the missing text, named exactly, with the frame and page. The tool was written for
precisely this failure mode and it did its job the moment it was finally pointed at this deck.

The bug did not survive 10 weeks because the tooling is weak, and **not because the workflow
forgot to ask for it either.** `WORKFLOWS.md`'s definition of done for a slide deck already
requires it, in bold:

> **`non_essential/detect_clipped_slides.py` AND `non_essential/detect_footer_collisions.py` both
> run, every flag checked against the rendered page**

So the process was right and was simply not executed - twice, on a deck that was actively edited
in between (the 2026-08-09 pass regenerated figures and added frames, which is exactly when the
gate applies).

**Why it gets skipped is worth naming, because it will happen again:** a visual pass *feels* like
it covers clipping. It does not, and cannot. Looking at a slide shows you what is on it; only a
comparison against the `.tex` shows you what should be there and is not. Absence has no visual
signature. The 2026-08-09 review was additionally a Sonnet subagent reading rendered PNGs, which
structurally cannot run a script - so the human-facing build step was the only place this could
have been caught, and that is the step that skipped it.

```bash
./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/09_clustering/32_clustering.tex
```

Its output needs eyes - on this deck it reports 3 false positives alongside the 1 real find: the
DBSCAN `eps` frame, the curse-of-dimensionality frame, and the `lstlisting` code frame. All three
texts were verified present on their rendered pages. One real find against three false ones is
still a two-minute check that caught a bug ten weeks of looking did not.

`detect_footer_collisions.py` was also run, and flags 9 pages on this deck - the 7 agglomerative
overlays plus 2 frames of mine. All 9 are content sitting *in* the footer band without touching
the page number; checked at 2.6x zoom on the tightest one. Band occupancy is not collision, which
is why the tool says to verify every flag rather than trusting the count.

## The theory I tested and disproved, and how I nearly published it

My first run of the detector was against a scratch copy I believed had the bug reintroduced. It
reported the frame clean at **100% coverage**, which led me to conclude the tool was blind to
content typeset past the page edge - that such text stays in the PDF text layer and defeats any
source-vs-render comparison. I had started writing that up.

It was wrong. The scratch copy never had the bug in it. I built it with a throwaway script whose
`str.replace` silently matched nothing while the script printed its success message anyway:

```python
t = p.read_text().replace(OLD, NEW)   # matched nothing
p.write_text(t)
print('reverted km_anim widths to 0.6')   # printed regardless
```

A `grep` afterwards showed all six lines still at `0.49`. The deck I "proved" the tool blind on
was the *fixed* deck.

The rule in `CLAUDE.md` - never silent fallbacks, fail loudly - is usually read as being about
shipped code. It applies with more force to throwaway analysis scripts, because those produce
**conclusions**, and a wrong conclusion gets written into `DECISIONS.md` and outlives the script by
years. I had already written the wrong cause into `DECISIONS.md` #23 and `ml/09_clustering/REVIEW.md`
before catching this, and had to correct both.

The bisect script that gave the right answer differed in one respect: it asserted its own
substitution took effect before trusting the result.

```python
assert src != ORIG or name.startswith("A"), f"{name}: substitution did not apply"
```

**Any script that sets up an experimental condition must verify the condition was actually set
up.** A no-op that reports success does not give you a null result - it gives you a confident
answer to a question you never asked.

## Consequences

- **Nothing to add to `WORKFLOWS.md` - the requirement is already there.** Adding a rule to fix a
  skipped rule just grows the document. What was missing was execution, so the honest takeaway is
  to run the two detectors as the *first* thing after a deck compiles, before opening the PDF at
  all. Once you have looked at the slides you feel done, and that feeling is what suppresses the
  check.
- **A "visual review" claim should state whether the mechanical checks ran.** Both prior reviews of
  this deck reported clean without saying they had skipped the gate, which is what let it look
  covered. See `_learnings/2026-08-13-1930_a-passing-check-is-only-as-good-as-its-coverage.md` for
  the sibling case: the same tool reporting 0 while 13 frames sat in the footer band.
- A tall figure plus a bottom caption is the risky shape. If a frame has a figure above
  `~0.55\textwidth` **and** a closing line, check that the line survived.
- The other Beamer overlay frames in this deck (DBSCAN at `0.56`, agglomerative at `0.82`) were
  checked and are fine - the agglomerative figure is much wider than tall, so it costs less height.
