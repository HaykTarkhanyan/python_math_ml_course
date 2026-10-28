# Moving a folder while one of its PDFs is open in Edge orphans the tab

**Symptom.** On 2026-10-28 I opened `ml/14_llms/LLM2_embeddings_position_notes.pdf` in Edge for the
instructor to annotate, then, in the same session, moved all of `ml/14_llms/` to `llms/` and
removed the old folder. The move went through without an error, so nothing looked wrong. The
built-in "you should know" plugin flagged it afterwards: the instructor's tab still showed a file
whose path no longer existed.

**Cause.** Edge does not lock an open PDF (it reads the file in), so Windows lets the folder be
renamed or moved underneath it. The tab keeps its old path. What a later Save does there was not
tested: it either fails or recreates the old folder, and in both cases the ink is not where the
course looks for it.

**Evidence.**

```
$ grep -o '...LLM2_embeddings_position_notes.pdf' <session transcript> | sort | uniq -c
      2 C:\\Users\\...\\01_python_math_ml_course\\ml\\14_llms\\LLM2_embeddings_position_notes.pdf
$ ls ml/14_llms
ls: cannot access 'ml/14_llms': No such file or directory
$ cmp -s llms/02_embeddings_position/LLM2_embeddings_position.pdf \
         llms/02_embeddings_position/LLM2_embeddings_position_notes.pdf && echo same
same          # the copy at the new path had no ink yet, so nothing was lost this time
```

**Consequences.**

- Before moving or renaming a folder, check whether this session opened any file in it for the
  instructor (`Start-Process` calls in the transcript). If so, ask them to save and close it first,
  then reopen it from the new path after the move.
- The same goes for a recompile that changes the page count. Ink saved into a notes PDF that is
  still on the old layout no longer lines up page by page with the new deck.
