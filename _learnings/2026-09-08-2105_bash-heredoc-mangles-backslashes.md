# Bash-tool heredocs eat backslashes, so never patch LaTeX or Python with one

**Symptom.** Patching files by piping a Python script through a *quoted* heredoc
(`python - <<'PYEOF' ... PYEOF`) silently corrupts every backslash escape in the payload.
Two separate failures in one session:

1. A replacement string containing `"inference\\nolny"` was written to disk as a **real
   newline**, producing:

   ```
     File ".../make_figures.py", line 89
       ax.text(4, 330, "inference
                       ^
   SyntaxError: unterminated string literal
   ```

2. A LaTeX patch whose search key was `"\\includegraphics[width=%s\\textwidth]{%s}"`
   matched nothing and blew the assertion:

   ```
   AssertionError: fig/a100_tflops.pdf
   ```

   because `\\textwidth` arrived at Python as `\textwidth`, and `\t` is a valid Python
   escape, so the literal became `<TAB>extwidth` and could never match the file.

**Cause.** A single-quoted heredoc delimiter is supposed to disable all expansion, and in a
plain interactive shell it does. Through this Bash tool the payload still gets one round of
backslash processing before `python` sees it, so `\\` collapses to `\`. Whatever the layer
responsible is, the practical consequence is that **the code Python executes is not the code
you wrote**, and the damage lands exactly on the escapes that matter (`\n`, `\t`,
`\textwidth`, `\includegraphics`).

Failure mode 2 is the nastier one: it does not crash inside the string, it just fails to
match, so a naive patch script "succeeds" while changing nothing. Only the explicit `assert`
caught it. A patch script with no assertion would have reported success and silently done
nothing.

**Consequences / what to do instead.**

- **Use the Edit tool for file edits containing backslashes.** LaTeX (`\includegraphics`,
  `\textwidth`, `\textbf`), Python escapes (`\n`, `\t`), regex - all of it. Several Edit
  calls in one message are cheap and cannot be mangled.
- If a script really must be piped, **write it to a file first** (Write tool), then run it as
  `python scratch_patch.py`. The Write tool does not touch the content.
- **Always `assert` that a search key was found** in any patch script. Silent no-op is worse
  than a crash, and the crash is what saved this session.
- Tested and disproven: switching the delimiter (`<<'EOF'` vs `<<"EOF"` vs `<<EOF`) does not
  fix it. The mangling happens before the shell's heredoc quoting would apply.

Related: this is the same class of problem as the `sed`/`grep` escaping quirks already
noted for this machine, but heredocs hide it better because the script *looks* correct in
the transcript.
