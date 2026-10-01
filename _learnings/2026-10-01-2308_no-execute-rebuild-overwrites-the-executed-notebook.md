# A `--no-execute` rebuild silently overwrites the executed notebook

**Symptom.** `build_gesture_snake_nb.py` had just executed the Gesture Snake notebook on the
instructor's recordings (157 s), writing outputs into `gesture_snake/gesture_snake_solution.ipynb`.
To add the result notes I edited the builder and ran it with `--no-execute` to look at the new
markdown. That wrote the same file - without outputs. The executed version was gone; one more run
was needed.

**Cause.** The builder writes `OUT` in both modes. `--no-execute` means "do not run", not "do not
write over a run". Nothing warns: the log line is the same `wrote ... (62 cells)` either way.

**Consequences.**

- Put the final text into the builder **before** the last executed run, then run it once. The
  outputs and the text it reads come out of the same run, and no merge step is needed. That is
  also what the "write prose after measuring" learning
  (`2026-08-13-2015_write-the-practical-after-measuring-not-before.md`) needs: measure in a scratch
  run, write the text, then do the final run.
- If you must preview markdown after an executed run, copy the executed file aside first.
- The same trap exists in every `build_*_nb.py` in the repo that has a `--no-execute` flag.
