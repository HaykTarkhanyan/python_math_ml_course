# A figure drawn wider than its slot shrinks every label in it, and no check notices

**Symptom.** The instructor said the CNN chapter figures had "small text". Measured, it was worse
than small: on the committed `ml/12_cnn` decks, figure labels rendered at **2.3-5 pt** against
10.9 pt body text. All four decks had passed 2x `pdflatex`, `detect_clipped_slides.py` and
`detect_footer_collisions.py`, and looked fine on 45-60 dpi contact sheets, where 3 pt text and
6 pt text both read as "small print".

Spans under 5.5 pt (two or more characters), committed PDFs at `7bcf8d2`:

| deck | smallest span | spans < 5.5 pt | pages flagged |
|---|---|---|---|
| L16 | 2.4 pt | 373 of 4053 | 28 |
| L17 | 3.4 pt | 299 of 2928 | 15 |
| L18 | 3.9 pt | 7 of 1698 | 6 |
| L19 | 2.3 pt | 96 of 2916 | 12 |

After the redraw: 0 pages flagged in all five decks of the chapter.

**Cause.** The scripts were written figure-first: `figsize=(10, 4)` with matplotlib's default
10 pt fonts, then embedded with `width=\linewidth` into a 3.5 in column. LaTeX scales the whole
PDF, fonts included, so 10 pt became 3.5 pt. The scale factor is just `slot width / figsize
width`, and for most of the chapter it was 0.3-0.5. Nothing in the pipeline compares the two
numbers, and the vector PDF stays sharp at any zoom, so the problem is invisible until a
projector.

**Fix.** Draw each figure **at the size it occupies on the slide**: `figsize` = the slot width in
inches (16:9 Beamer text width is about 5.5 in, so a 0.6 column is ~3.3 in), fonts 7-10 pt, and
embed at that same width (`width=3.3in`, or `\linewidth` of a column that wide). Then 8 pt in
matplotlib is 8 pt on the slide.

How the slot scale was measured: PyMuPDF reports text spans with their size *as rendered on the
page*, including text inside embedded figure PDFs. Matching a figure's span text against the
same text in the standalone figure PDF gives `rendered / native` = the scale. That probe is now
`non_essential/detect_small_figure_text.py` (threshold 5.5 pt, exits 1 if any page has smaller
text).

**Tested and disproven.** A uniform transform (multiply `figsize` by the measured scale, map all
fonts into 7-10.5 pt) fixes the size but **not** the layout: **16 of the 19** transformed
scripts needed at least one hand patch afterwards - legends on top of curves, titles colliding
with suptitles, annotation arrows through text, bar labels past the axes. Each redrawn figure
still had to be looked at (at 200 dpi, not on a contact sheet) and patched by hand. Budget for
that; the transform is a starting point, not a fix.

**Consequences.**

- Run `detect_small_figure_text.py` with the other two detectors in the deck-polish loop. It is
  blind to raster figures (PNG/JPG carry no text); look at those.
- Contact sheets cannot judge text size. Use the detector, or render single pages at 120+ dpi.
- When a figure needs more room than its slot, change the layout (a 1 x 4 row became a 2 x 2
  grid in `fourier_view.py`), not the font size.
