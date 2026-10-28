# A `\newcommand` macro never `\ifx`-matches `\empty`

**Symptom.** LLM-3's corner equation (a `frametitle` hook that draws the current piece only when
`\curpiece` is set) fired on every titled frame, and the deck stopped at the first one:

```
! Argument of \attni has an extra }.
l.136 \end{frame}
```

**Cause.** The switch was defined as `\newcommand{\curpiece}{}` and tested with
`\ifx\curpiece\empty`. `\newcommand` (without `*`) makes a `\long` macro; LaTeX's `\empty` is not
`\long`. `\ifx` compares the macro's status as well as its text, so an empty `\long` macro is never
equal to `\empty`: the "off" branch never runs, and the hook expanded `\attni` with no arguments.

**Fix.** Define the switch with plain `\def` (and set it with `\xdef`, also not `\long`):

```latex
\def\curpiece{}   % \def, not \newcommand: a \long macro never \ifx-matches \empty
\newcommand{\setpiece}[1]{\xdef\curpiece{#1}}
... \ifx\curpiece\empty\else <draw it>\fi
```

After the change: 0 errors, the strip shows only on frames after `\setpiece{\pQK}` and disappears
after `\setpiece{}` (checked on the rendered pages, 2026-10-28).

**Consequences.** Any on/off flag tested against `\empty` must come from `\def`/`\edef`/`\xdef`, or be
compared with `\ifdefempty` (etoolbox), which ignores `\long`. `\newcommand*` also works (it is not
`\long`).
