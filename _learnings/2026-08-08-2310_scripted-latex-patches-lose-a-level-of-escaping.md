# Never batch-patch LaTeX with a script that interpolates backslashes

**Symptom.** Six small edits to a `.tex` file were applied in one Python script run through a
quoted heredoc, to save round trips. The result:

| Edits attempted | 6 |
|---|---|
| Applied correctly | 2 |
| Corrupted the file | 1 |
| Silently did nothing | 3 |

The corrupted line, seen in the file afterwards:

```
  	extbf{AMI Labs} - the company LeCun leaves Meta to found, ...
```

A literal **tab character** followed by `extbf`. And the three no-ops left a `\wlslide{}` call
pointing at a deleted image, so the next compile died:

```
! Package pdftex.def Error: File `borrowed/welchlabs/euclidean_to_goal.jpg' not found
```

**Cause.** One level of backslash escaping was lost between the shell and Python, so
`"\\textbf"` arrived as `"\textbf"` - which Python reads as `\t` + `extbf`, a tab. The same
mechanism silently broke every search string containing `\begin`: `\b` is a backspace character,
so `"\begin{frame}{The result...}"` never matched anything in the file.

**The dangerous part is not the corruption - it is the silence.** `str.replace()` returns the
string unchanged when it finds no match. It does not raise, warn, or report a count. Three edits
did nothing at all and the script printed `L40 patched` and exited 0.

**Consequences.**

- Only **one** of the four failures was self-announcing, and it was the least important one (a
  missing image the compiler noticed). The tab corruption and the three silent no-ops would all
  have shipped if that unrelated compile error had not forced a closer look.
- Redone with targeted `Edit` calls, which fail loudly when the anchor does not match - that is
  the whole point of them, and it is exactly what was traded away for speed.
- Cost accounting: the "efficient" batch took one call to write, one to discover the damage, one
  to assess the extent, and five to redo properly. Eight calls to save three.

**Disproven along the way.** The first theory was that the quoted heredoc (`<<'PY'`) had failed
to suppress shell expansion. It had not - a quoted heredoc does suppress it. The loss happened
elsewhere in the tool's shell handling, which means **quoting the heredoc is not a sufficient
defence** and the same trap is waiting on the next attempt.

**The rules that follow:**

1. Use `Edit` for LaTeX. It matches exactly or fails; there is no silent third outcome.
2. If a bulk change is genuinely unavoidable, **assert on every replacement** (`assert old in
   text`) so a miss is an exception rather than a no-op, and grep the result for control
   characters afterwards: `grep -nP '\t|\x08|\x0c' FILE`.
3. Never let a patch script report success on the basis of having run.
