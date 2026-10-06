# TeX silently drops a `\quad` at the start of a line, so hand-indented code loses its indentation

**Symptom.** L44's code boxes (`\codebox{...}`, a tcolorbox with `\ttfamily`) showed the agent loop
and the JSON tool schema with every line flush left:

```
while True:
reply = model(messages, tools=tools)
if not reply.tool_calls:
return reply.text
```

As Python that is wrong (the `return` and the `for` body sit at the wrong level). It had been like
this since August, in the committed PDF, through two compiles per edit, the margin scan, the
clipped-text detector and an August student review. A 2026-10-06 student review reading the
rendered page did not flag it either; it was noticed while looking at a contact sheet.

**Cause.** The source indented with spacing commands after each line break:

```latex
\codebox{while True:\\
\quad reply = model(messages, tools=tools)\\
\qquad return reply.text}
```

`\quad` and `\qquad` are horizontal glue. Glue right after a line break is *discardable*: TeX
removes it at the start of the new line, exactly as it removes an ordinary space there. Nothing
warns.

**Fix.** Use the starred form, which is not discarded: `\hspace*{1em}` for `\quad`,
`\hspace*{2em}` for `\qquad`, `\hspace*{3em}` for `\qquad\quad`. One regex over the deck fixed all
16 lines:

```python
re.compile(r"(\\\\[ \t]*\n[ \t]*)(\\qquad\\quad|\\qquad|\\quad)\s?")   # -> \hspace*{N em}
```

A grep over `ml/ch17_rag`, `ml/ch18_agents` and `ml/14_llms` found the pattern only in L44.

**Consequences.**

- Any hand-indented code in a non-verbatim box: indent with `\hspace*`, or use a verbatim
  environment (which needs a `[fragile]` frame, see
  `2026-08-16-1530_lstlisting-in-beamer-needs-a-fragile-frame.md`).
- Look at code slides rendered, not in the source: the source looks indented.
