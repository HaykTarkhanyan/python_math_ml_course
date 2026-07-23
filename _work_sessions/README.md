# _work_sessions

One TOML file per work session. This replaces the single ever-growing
`PROGRESS.md` (now a read-only archive). Files sort chronologically by their
date-time filename prefix, so the newest session is last.

## Filename

`YYYY-MM-DD-HHMM_slug.toml` - e.g. `2026-07-23-1430_dl4nlp-decoding-deck.toml`.
`HHMM` is the 24-hour local start time (Armenia / Berlin), with no colon so it
is Windows-safe. Date + time in the prefix keep files sorting chronologically
and make multiple sessions on the same day distinct without any `_b` suffix.

## Schema

Only `title`, `date`, and `done` are required; the rest are optional.

- `title`        - one-line summary of the session
- `date`         - session date, TOML date (unquoted): `2026-07-22`
- `time`         - local start time (Armenia / Berlin), TOML time (unquoted): `14:30:00`
- `session_name` - Claude Code session name, if any
- `areas`        - top-level folders touched (array of strings)
- `tags`         - freeform tags for searching (array of strings)
- `done`         - what was accomplished (array, most important first)
- `pending`      - unfinished or flagged items (array)
- `next`         - what the next session should pick up (array)
- `files`        - key files created or heavily changed (array; not exhaustive)
- `commits`      - git commit hashes made this session (array)
- `notes`        - freeform context that does not fit above (multi-line string)

## How to use it

- **Session start:** skim the last 2-3 files here (newest filenames) for recent
  context before doing substantial work.
- **Session end:** add a new file (the `wrap-session` skill does this).

Entries are point-in-time snapshots; for anything old, verify against `git log`
before acting on it.
