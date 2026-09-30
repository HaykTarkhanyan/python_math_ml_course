# A slow `ma` kernel start trips nbclient's 60 s startup timeout - with an error that hides it

**Symptom.** Executing the gesture notebook through `nbclient` failed after 113 s with no cell
output at all. The only traceback was a cleanup error at interpreter exit:

```
File ".../nbclient/client.py", line 504, in _async_cleanup_kernel
    assert self.km is not None
AssertionError:
RESULT: FAILED in 113 s
```

**Cause - measured, not guessed.** A one-cell notebook on the same kernel:

```
NotebookClient(nb, kernel_name="ma", startup_timeout=180).execute()
kernel OK in 102.2 s
```

`nbclient`'s default `startup_timeout` is 60 s. Earlier the same evening the whole notebook,
kernel start included, had run in 211 s, so the start time varies a lot. Why it was slow (system
load, a scanner on the venv) was not established.

**Fix.** `NotebookClient(..., startup_timeout=300)` in `build_gesture_snake_nb.py`, and wrappers
that catch the exception must **print** it - the bare "FAILED" plus the atexit noise is what made
this look like a notebook bug.

**Consequences.** When a notebook run fails with no cell outputs, time a one-cell notebook on the
same kernel before debugging the notebook itself.
