# Armenian letters and guillemets in a deck's body text stop pdflatex

**Symptom.** After adding one remark to L21's click-through frame, the deck stopped compiling:

```
! LaTeX Error: Command \guillemetleft unavailable in encoding OT1.
l.362 \end{frame}
!  ==> Fatal error occurred, no output PDF file produced!
```

The remark was `near-certain from «յ» on: 95.8\% of the surnames end in «-յան»`.

**Cause.** The decks compile with pdflatex and the default OT1 font encoding (`ml/preamble.tex`).
OT1 has no guillemets (`«` `»`) and no Armenian glyphs, so the first one in typeset text is a fatal
error. The error points at `\end{frame}`, not at the offending line, because the frame body is
read as one unit.

**Consequences.**

- Armenian goes **inside the matplotlib figures** (Segoe UI, glyph warnings promoted to errors -
  see `ml/ch7_rnn/py_src/forward_pass_anim.py`), never in the slide text. Every RNN deck already
  followed this; the slip was mine.
- In slide text, transliterate (`"-yan"`) and use straight quotes.
- `%` comments and the `% Provenance:` block after `\end{document}` may contain Armenian: pdflatex
  discards comment text before typesetting it. That is why the decks' comments are full of it
  without any problem.
- Cheap check before compiling - Armenian block, guillemets, em/en dashes, curly quotes in
  non-comment lines of the body:

  ```python
  import re
  body = open(DECK, encoding="utf-8").read().split(r"\end{document}")[0]
  for i, l in enumerate(body.splitlines(), 1):
      code = "" if l.lstrip().startswith("%") else l.split("%")[0]
      if re.search("[԰-֏«»–—“”]", code):
          print(i, code.strip())
  ```

  (Run it from a `.py` file, not a Bash heredoc - see
  `2026-09-08-2105_bash-heredoc-mangles-backslashes.md`.)
