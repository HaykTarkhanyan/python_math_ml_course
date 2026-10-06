# A matplotlib diagram drawn in data coordinates is about 23% narrower than its figsize

**Symptom.** Redrawing L44's agent-loop diagram at slot size (`figsize=(5.07, 2.0)`, `xlim(0, 9.4)`,
6.5 pt bold box labels), the arithmetic said a 2.2-unit box was 1.19 in wide and its label
"enough to answer?" (about 1.07 in) fitted. On the slide the label ran out of the box on both sides,
and so did "observe the result".

**Cause.** `plt.subplots()` keeps matplotlib's default subplot margins (`left=0.125, right=0.9,
bottom=0.11, top=0.88`) even when `ax.axis("off")` hides the axes. The axes, and with them every
data unit, span only 77.5% of the canvas width and 77% of its height. `savefig(bbox_inches="tight")`
then crops the empty margins away, so the PDF looks as if the drawing filled the figure, and the
figure's native width still matches the slot. The real box was 0.775 x 1.19 = 0.92 in, narrower
than its label.

**Fix.** For box-and-arrow diagrams in data coordinates, make the data span the canvas:

```python
fig, ax = plt.subplots(figsize=(5.07, 2.0))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)   # leave room for a title
ax.set_xlim(0, 9.4); ax.set_ylim(0, 3.9); ax.axis("off")
```

After that, 1 data unit = figsize width / xlim span, and text-in-box sizing can be done on paper.
Two L44 diagrams (01, 02) got the same line, and their canvases could then be 0.3 in shorter for
the same drawing.

**Consequences.** When text "should fit" by the numbers and does not, check the subplot margins
before shrinking the font. Plots with real axes (ticks, labels) do need those margins; this is for
`axis("off")` diagrams.
