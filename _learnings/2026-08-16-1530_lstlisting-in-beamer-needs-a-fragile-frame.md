# "Illegal parameter number in definition of \iterate" means a missing [fragile]

**Symptom.** Compiling a new Beamer deck (`35_umap.tex`) died with:

```
! Illegal parameter number in definition of \iterate.
<to be read again>
l.496 \end{frame}
!  ==> Fatal error occurred, no output PDF file produced!
```

Line 496 is a plain `\end{frame}`. There is nothing wrong at line 496. `\iterate` is a **pgffor**
internal, and the deck's only `\foreach` loops are 200 lines earlier in a different frame - so the
error names the wrong macro, in the wrong package, at the wrong line.

**Cause.** The frame contained an `lstlisting` block whose Python comments used `#`:

```latex
\begin{frame}{In practice}          % <- missing [fragile]
\begin{lstlisting}
    n_neighbors=15,      # local <-> global
\end{lstlisting}
```

Beamer reads a frame's whole body before typesetting it, so verbatim-like content must be
protected with `[fragile]`. Without it, `#` is scanned as a macro-parameter token, and the
resulting mess surfaces wherever the next macro definition happens to be - here, pgffor's.

**Fix.** `\begin{frame}[fragile]{In practice}`.

**Consequences.**

- Any frame containing `lstlisting`, `verbatim`, `minted`, or `\verb` needs `[fragile]`. It is
  cheap to add pre-emptively, and there is no penalty for using it when it is not required.
- **Do not trust the reported line number or macro name for this class of error.** The signal is
  the *combination*: a fatal error naming an internal from an unrelated package, pointing at a
  structural token like `\end{frame}`. Go looking for verbatim content in that frame, not for a
  bug where the error points.

Not previously recorded in `_learnings/` or `LEARNINGS.md`, despite this repo having ~30 Beamer
decks - most of them simply never put code on a slide, because `ml/SLIDE_STYLE.md` limits decks to
at most one canonical snippet per topic.
