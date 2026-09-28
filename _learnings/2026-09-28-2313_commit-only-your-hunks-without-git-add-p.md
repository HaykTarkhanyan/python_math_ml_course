# Committing only your own hunks of a file someone else also edited (no `git add -p`)

**Symptom.** `DECISIONS.md`, `CONVENTIONS.md` and `WORKFLOWS.md` held another session's
uncommitted work (DECISIONS #45-#47, youtube-reference conventions) plus this session's
additions (#48-#53, link rules). `git add -p` is interactive, so unavailable to the Bash tool,
and `git add <file>` would have committed the other session's text too.

**What worked.** Build the file you want to commit, write it straight into the index, leave the
working tree alone:

```python
head = git("show", "HEAD:DECISIONS.md")                  # blob as committed
new = head.replace(top, top + my_block)                  # HEAD + only my edit
sha = git("hash-object", "-w", "--stdin", "--no-filters", input=new)
git("update-index", "--cacheinfo", f"100644,{sha},DECISIONS.md")
```

Then verify both sides:

```
git diff --cached -- DECISIONS.md   -> only #48-#53        (+127)
git diff          -- DECISIONS.md   -> only #45-#47        (+79, same as before I touched it)
```

**Consequences.**

- Anchor each edit on text that exists in `HEAD` and assert it matches exactly once.
- Keep the blob's line endings (`git show HEAD:path` returns them as committed).
- Script: `stage_docs.py` in the 2026-09-28 session scratchpad; the recipe above is the whole idea.
