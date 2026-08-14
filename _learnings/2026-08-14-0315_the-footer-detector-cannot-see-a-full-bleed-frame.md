# The footer detector flags every full-bleed frame, and the obvious fix is wrong

**Context.** `ml/ch20_subliminal_learning/L48` is the first deck in this repo to use
full-bleed video stills heavily - six of them, an approved default per `ml/SLIDE_STYLE.md`.

**Symptom.** `detect_footer_collisions.py` flagged 8 frames. Five were reported at ~99%
ink in the footer band:

```
page   6   ink 98.62%
page  23   ink 99.30%
page  27   ink 99.80%
page  15   ink  3.06%     <- a real collision
page  34   ink  7.10%     <- a real collision
```

The ~99% ones are photographs filling the slide. Nothing is wrong with them. The detector
counts dark pixels in the bottom strip, and a photo has dark pixels everywhere.

**First fix, which failed.** Distinguish a picture from a text overflow by how dark the
*rest* of the page is: a text frame is mostly white, a photo is mostly not. Flag only if
body ink < 50%.

This cut the false positives from 5 to 3, and the survivors showed why the idea was
wrong: a still of **code cards on a desk** and a still of a **white sheet of paper** are
both full-bleed images that are mostly *light*. Keying on darkness asks "is this page
dark", when the actual question is "is this page a picture".

**The fix that works.** Test what "full-bleed" literally means: the image reaches the page
edge. A Beamer text frame always leaves white at the extreme border; an image that bleeds
covers it. Sample the outer ~1.5% ring and measure how much is non-white.

```
FULL_BLEED_BORDER_COVERAGE = 0.90
```

All six stills classified correctly, both dark and light, and the two real collisions
still flagged. They are now reported separately and excluded from the exit code:

```
L48_subliminal_learning.pdf: clean
    (6 full-bleed image frame(s) not counted: 6, 8, 23, 27, 47, 59)
```

**Regression check.** Changing a tool that gates every deck needs proof it did not go
blind. `ml/ch19_mech_interp` reported 0 flagged before the change and 0 after - it has no
full-bleed frames, so the new branch must never fire there. Run this check whenever the
heuristic is touched.

**The transferable lesson.** When a detector produces false positives, fix it by encoding
the *definition* of the thing you want to exclude, not a property that happens to
correlate with it in the examples in front of you. "Full-bleed images are dark" was true
of the first sample and false of the second. "Full-bleed images touch the edge" is what
the term means, so it cannot drift.
