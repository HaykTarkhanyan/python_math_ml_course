# `grep -c 'Overfull \\vbox'` inside `"...$(...)..."` counts nothing: use Python or a script file

**Symptom.** On 2026-11-06/09 every compile check of the embeddings deck printed `overfull-v: 0`, from

```bash
echo "errors: $(grep -cE '^!' X.log) overfull-v: $(grep -c 'Overfull \\vbox' X.log)"
```

The same deck, compiled by a script file that contained `grep -c 'Overfull \\vbox' "$dir/$stem.log"`, reported
7. The log had them all along:

```
$ grep "Overfull" LLM3_embeddings_position.log | head -3
Overfull \hbox (0.94458pt too wide) in paragraph at lines 65--65
Overfull \vbox (1.25504pt too high) detected at line 73
Overfull \vbox (1.34792pt too high) detected at line 281
```

**Cause.** Typed directly into the Bash tool, the `$(...)` sits inside a double-quoted string, so `\\` collapses to
`\` before grep runs: grep gets `Overfull \vbox`, and GNU grep reads `\v` as a plain `v`. The pattern becomes
"Overfull vbox", which never occurs. A later `grep -o "Overfull \\\\vbox ..."` attempt failed the same way. In a
script file, the single quotes keep both backslashes and the count is right.

**Consequences.**
- Count overflows with Python (the scratchpad helper `overfull_frames.py` lists each one with its frame title) or
  with a pattern that has no backslash: `grep -c "Overfull .vbox"`.
- A check that prints 0 needs one positive control. Here a plain `grep -c Overfull X.log` (9) would have exposed
  the problem at once.
- Several "0 overfull" claims in that session were wrong. The real 7 (+4 from new text) were then trimmed to 0.
