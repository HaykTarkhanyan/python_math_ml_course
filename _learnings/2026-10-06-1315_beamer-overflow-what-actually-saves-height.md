# Fixing beamer overflow: what saved height and what did not (52 frames, four RAG/agents decks)

**Context.** The 2026-10-05/06 layout pass took L41-L44 from 52 overflowing frames to 0
(`Overfull \vbox ... detected at line N` in the log; the line is the frame's `\end{frame}`).

**What worked, roughly in order of payoff.**

- **Widen a figure, do not shrink it.** Frames had horizontal room and never vertical room. A
  figure redrawn at 0.90 width instead of 0.66 keeps its fonts and gets *shorter* for the same
  content, because side labels stop forcing extra height. Done for about ten figures.
- **Two columns.** A centred 0.4-width figure with a paragraph and a box stacked under it became a
  0.5 + 0.46 column pair three times; each frame dropped 10-38 pt.
- **One wrapped line is about 12 pt** (11 pt body) or 10 pt in `\small`. Most frames 2-15 pt over
  were fixed by tightening one sentence so it stopped wrapping, without losing content.
- **A scripted gap pass.** In only the frames the log flags, scale every positive
  `\vspace{X em}` with X >= 0.3 by 0.45. On L43 that took 21 overflowing frames to 9 in one run; the
  rest needed a line or a shorter figure. Every frame it touched was then looked at rendered; none
  looked cramped (the August decks used 0.7-1.0 em gaps everywhere).
- **Table rows:** `\\[0.3em]` -> `\\[0.1em]` over four rows saves about 8 pt.

**What did not work.**

- **Removing a `\vspace` at the very top of a frame.** It changed nothing (L42's recap stayed
  16 pt over). Glue at the top of the frame box is discarded anyway.
- **`{\small ...}` around a beamer `itemize`.** L42's recap went from 16 pt over to **63.6 pt**
  over: the list's item spacing grew under the size change. `\setlength{\itemsep}{0.1em}` inside
  the list also made it slightly worse (19.3 pt), so beamer's default is already tighter than that.
  Shortening two items to one line each fixed it.
- **Negative `\vspace` after a figure** helps a little (0.2 em is about 2 pt) but cannot carry a
  frame that is 10 pt over.

**Consequences.** For an overflowing frame: first ask whether its figure can be wider and shorter,
or the frame two columns; then save whole lines; gap-shrinking last. Recompile after each batch,
because a fix in one frame never affects another but a figure redraw can push a frame that was fine.
