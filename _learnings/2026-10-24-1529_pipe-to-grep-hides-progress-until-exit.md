# Piping a background job through `grep` or `head` hides its progress until it exits

**Symptom.** A 10-minute training probe ran in the background as
`python s2s_probe.py ... | grep -v '^ *\[' | head -80`. The instructor asked "eta?" seven minutes in.
The output file was empty, and the only answer was a guess from the process's CPU time
(`Get-Process` showed 854 s of CPU).

**Cause.** When its output goes to a file rather than a terminal, `grep` buffers it in blocks and
writes only when the buffer fills or the input ends. `head` waits too. The script's own
`print(..., flush=True)` lines reach `grep` at once and then sit in its buffer.

**Fix, verified the same day.**
- `grep --line-buffered`: the restarted sweep's progress lines (`step 100/1500 ... ~1707s left`)
  appeared in the output file as they were printed.
- `python -u` for the script itself.
- No `| head` on a long job: let it write everything, and filter when reading.

**Consequences.** This is now a global rule (`~/.claude/CLAUDE.md`, "Tell me how long things take"):
design long jobs so an ETA can be read mid-run. I repeated the mistake once more in the same
session, minutes after writing the rule, and had to stop and restart the chain with
`--line-buffered`. Check the pipe every time.
