---
name: clean-tex
description: >
  Delete all LaTeX build artifacts (.aux, .log, .nav, .out, .snm, .toc, .vrb,
  .fls, .fdb_latexmk, .synctex.gz) from the entire repo. Use when build files
  accumulate and clutter the workspace.
allowed-tools:
  - Bash
user-invocable: true
---

# Clean LaTeX Build Artifacts

Delete all LaTeX intermediate/build files from the project directory.

Run this single command:

```bash
find "$CLAUDE_PROJECT_DIR" -type f \( \
  -name "*.aux" -o -name "*.log" -o -name "*.nav" -o -name "*.out" \
  -o -name "*.snm" -o -name "*.toc" -o -name "*.vrb" -o -name "*.fls" \
  -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o -name "*.synctex" \
  -o -name "*.bbl" -o -name "*.blg" -o -name "*.run.xml" -o -name "*.bcf" \
  -o -name "*.idx" -o -name "*.ilg" -o -name "*.ind" \
  -o -name "*.lof" -o -name "*.lot" -o -name "*.fmt" \
  -o -name "*.dvi" -o -name "*.xdv" \
\) -delete
```

Report how many files were deleted. Do NOT delete .tex or .pdf files.
