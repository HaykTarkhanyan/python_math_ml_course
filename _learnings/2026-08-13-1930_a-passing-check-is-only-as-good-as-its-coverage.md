# A passing check is only as good as its coverage: 13 broken frames reported as clean

**Context.** Building `ml/ch19_mech_interp` (three Beamer decks, 141 frames). Each deck was
declared verified on the strength of three checks that all passed.

**Symptom.** Every automated gate was green:

```
pdflatex x2          0 errors, 0 Overfull \vbox
acronym grep         pass
detect_clipped_slides.py   TOTAL frames flagged: 0   (x3 decks)
```

A manual read of the rendered slides then found a frame whose final sentence ran underneath the
page number, and another whose citation was cut off mid-word. A purpose-built scan found
**13 such frames across the three decks** - roughly one frame in eleven.

**Cause.** Three independent blind spots lining up:

1. **Beamer does not warn.** It clips or overlaps content silently; `Overfull \vbox` fires for a
   box exceeding its own height, not for a frame's content colliding with the footer template.
2. **`detect_clipped_slides.py` tests something else.** It looks for content clipped at the
   *frame boundary*. A `tcolorbox` that grows downward until it sits on the page number is not
   clipped - every pixel is drawn - so the detector correctly reports nothing.
3. **I quoted the detector as evidence without asking what it tested.** "0 flagged" was reported
   three times as proof the decks were clean.

**Consequences.**

- `non_essential/detect_footer_collisions.py` now exists: render each page, inspect the bottom
  5.5% band, ignore the right-hand corner where the page number lives, flag any remaining ink.
  Twenty lines of real logic. It found all 13 immediately and drove them to 0.
- `WORKFLOWS.md` definition-of-done now requires **both** detectors, with a note that neither
  catches the other's failure mode.
- The two failures are genuinely disjoint: after fixing all 13 footer collisions,
  `detect_clipped_slides.py` still reports 0 - it was never going to find them.

**The transferable lesson.** A green check is evidence about *what the check tests*, not about
the artifact. Before citing a tool's pass as verification, read what it actually measures. The
failure here was not writing a weak detector - `detect_clipped_slides.py` does its job correctly.
It was treating its silence as coverage it never claimed.

**Cheap tell that generalises.** Both defects were visible in the first four rendered pages I
looked at, after three tools said the decks were clean. When a check passes on work that has
never been eyeballed, look at a handful of samples anyway - not to replace the tool, but to find
out what it is not looking at.
