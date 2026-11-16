# Piping a Python probe through `tee` hides both its progress and its crash

**Symptom.** A ~12 min verification probe ran as `python probe.py 2>&1 | tee probe.log` in the
background. Two things went wrong:

1. The first run crashed at second 0 with a traceback, and the task still reported
   `exit code 0`.
2. The second run printed per-run progress with an ETA, but the output file stayed **empty for the
   whole 12 minutes**. When the user asked "eta?", there was nothing to read. The only signal was
   process CPU time (`Get-Process python`: 901 CPU-s after 11.7 min).

**Cause.**

1. A pipeline's exit status is the **last** command's, and `tee` succeeds. The Python crash is
   discarded.
2. Python block-buffers stdout when it is not a terminal (~8 KB). Through a pipe, every `print`
   waits in the buffer until the buffer fills or the process exits. A probe that prints a few
   hundred bytes in total shows nothing until it finishes.

**Fix.** Both, every time a probe is teed:

```bash
set -o pipefail
./ma/Scripts/python.exe -u probe.py 2>&1 | tee probe.log    # -u = unbuffered stdout
```

(`PYTHONUNBUFFERED=1` or `print(..., flush=True)` also work. `-u` is the one you cannot forget
per line.)

**Consequences.**

- The 3-6 min runtime guess was off by 2-3x (actual 714 s), and with buffered output that was
  invisible until the end. A one-epoch timing run done first (1.3 s/epoch, single thread) would
  have given a real estimate. The `review-submission` skill now says both.
- Related, same session: the crash itself was the known int32-label gotcha
  (`2026-10-01-0044_windows-numpy-int32-labels-hidden-by-augmentation.md`). It was hit again in
  fresh code because `np.array(list_of_ints)` was written without `dtype=np.int64`.
