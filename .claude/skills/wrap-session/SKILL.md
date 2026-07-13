---
name: wrap-session
description: >
  Use when a work session is ending - the user says "wrap up", "let's stop
  here", "commit this", or a lecture/practical was just delivered and its
  materials need to be linked and logged.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Glob
  - Grep
user-invocable: true
---

# Session wrap-up checklist

Work through these in order and report each as done or skipped (with reason).

1. **Clean LaTeX junk:** `./ma/Scripts/python.exe clean_latex.py` (repo-wide).

2. **Review `git status`:** nothing unintended staged - no `.aux`/`.log`/`.nav`
   files, no oversized files, no PII datasets (the voter registry is
   deliberately git-ignored).

3. **Update `PROGRESS.md`:** add a dated section at the top - what was done,
   what is pending, what comes next. Concrete file paths, not vague summaries.

4. **If a lecture or practical was delivered today:**
   - Ask the user for the YouTube link, then add it to the matching `.qmd`
     (`update-youtube` skill). Link text must match the YouTube title exactly.
   - Mark the session complete in the plan file (`ml/00_plan.md` or
     `math/Lectures/stat/00_plan.md`).

5. **Capture knowledge - ask the user explicitly:**
   - New gotcha discovered? -> `LEARNINGS.md` (dated entry)
   - Decision made for the second time? -> `CONVENTIONS.md`
   - Topic cut or postponed? -> `DEFERRED_TODO.md`
   - Was CLAUDE.md contradicted or corrected this session? -> propose the edit.

6. **Commit and push** (only if the user wants): short imperative subject with
   an area prefix (`ml: ...`, `Chapter 3: ...`). Quick path:
   `make push render=false msg="..."`. A push to `main` triggers the GitHub
   Actions site render (~5 min) - no local rendering.
