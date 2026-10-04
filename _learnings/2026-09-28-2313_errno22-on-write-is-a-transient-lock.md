# `OSError: [Errno 22] Invalid argument` when writing a file is a transient lock here

**Symptom.** Batch-editing ~100 site files, two writes failed mid-run with the same error on
files that had just been read successfully:

```
OSError: [Errno 22] Invalid argument: 'math/26_stat_classical_tests.qmd'
OSError: [Errno 22] Invalid argument: 'python_libs/10_scraping__parallelization.ipynb'
```

**Cause.** Not established. The pattern (a path that is valid, readable, and writable a second
later) fits a short-lived lock by another process on a just-touched file, most likely Defender
scanning it. OneDrive sync is off on this machine, so it is not that.

**What was checked.** The failed `open(..., "w")` did not truncate the file (byte count and
`git diff --numstat` unchanged); a plain rerun succeeded; a bounded retry (5 attempts, 1 s apart)
succeeded on its first attempt.

**More evidence, 2026-10-03/04** (redrawing the RNN figures): six failures in one evening, on
matplotlib PDF writes (`forward_pass_1/4/5.pdf`, `memory_highway.pdf`) and once on a `.tex` file.
Every time the file had been read by pdflatex or pdftoppm within the previous few minutes, and
the script was writing several files back to back. A `wb` open of the same file seconds later
succeeded. A retry loop logged the lock lasting about 4 s (attempts 1-4 failed one second apart,
attempt 5 worked). Still consistent with an on-access scan; still not proven. The failed `open`
never truncated the file (`.tex` checked: same byte count and frame count before the retry).
`ml/ch7_rnn/py_src/forward_pass_anim.py` now retries a failed `savefig` up to 5 times, 1 s apart,
then raises.

**Consequences.**

- Batch writers: make the script idempotent, so a rerun only redoes what is left, and verify
  the failed file before retrying.
- A bounded retry on `OSError` for writes is fine; after the last attempt it must still fail
  loudly, never skip the file.
