# `\vskip 4pt` followed by a sentence starting with "Plus" breaks the compile

**Symptom.** `L32f_llm_alignment.tex` failed with two errors that pointed at the wrong place:

```
! Missing number, treated as zero.
<to be read again>
                   g
l.398 \end{frame}
! Illegal unit of measure (pt inserted).
```

Line 398 is `\end{frame}`. The actual offending text was 10 lines earlier, and nothing on line
398 involves a number. Beamer defers frame typesetting to `\end{frame}`, so **every error inside
a frame is reported at the `\end{frame}` line** - the reported line number is useless for
locating the cause.

**Cause.** The source was:

```latex
\vskip 4pt
Plus generation: every update needs fresh samples from the current policy...
```

`\vskip` takes a *glue* specification, not a dimension: `<dimen> plus <dimen> minus <dimen>`.
TeX keywords are case-insensitive and whitespace/newlines are skipped, so after reading `4pt`
TeX kept scanning, found the word **"Plus"**, accepted it as the glue keyword `plus`, then
demanded a number and hit the `g` of "generation". Hence `<to be read again> g`.

The same trap fires for a sentence starting with **"Minus"**, and for `\hskip`, `\vspace`,
`\hspace`, and any other glue-taking primitive.

**Consequences.** Any prose line after a bare `\vskip`/`\hskip` that starts with "Plus" or
"Minus" is a landmine. It is invisible on inspection because the LaTeX looks completely normal,
and the error message points at the wrong line.

**Fixes**, in order of preference:

1. **Reword** so the sentence does not start with Plus/Minus ("And then generation: ..." is what
   shipped). Zero markup, no future reader has to know about this.
2. **`\vskip 4pt\relax`** - `\relax` stops the glue scan dead. Use this if the wording matters.
3. `\vspace{4pt}` takes a braced argument and is not vulnerable in the same way.

**How to find it next time.** The reported line is the frame's `\end{frame}`, so bisect the
frame body rather than reading around the reported line. Extract the frame into a standalone
file and cut it down - **but keep environments balanced**, because truncating mid-`tcolorbox`
produces its own "missing number" style errors and sends you chasing a phantom:

```bash
sed -n '361,398p' DECK.tex > _frame.tex     # the failing frame
# then test: head -N _frame.tex + matching \end{...} lines, not a raw truncation
```

**Grep for the pattern across the repo before it bites again:**

```bash
grep -n -A1 '\\vskip [0-9.]*pt *$' ml/**/*.tex | grep -iE '^\S+-\s*(Plus|Minus)\b'
```

Recorded 2026-08-12 while building the 7-deck RL chapter (`ml/ch11_rl`).
