# Filename case: Windows ignores it, the Linux site build does not

**Symptom.** 8 of 24 dead internal links on the live site were case mismatches that work
locally, e.g. `index.qmd` linking `python/01_intro.ipynb` while git had `python/01_Intro.ipynb`.
Quarto printed them in the CI log and still passed:

```
WARN: Unable to resolve link target: python/01_intro.ipynb
WARN: Unable to resolve link target: python/02_conditions.ipynb
...
```

Two notebooks were even lowercase **on disk** (`09_files_packages_terminal.ipynb`,
`14_classes.ipynb`) while git stored them capitalized, so any tool that reads names from the
disk writes links that break on the build.

**Cause.** NTFS and git on Windows (`core.ignorecase=true`) treat `A` and `a` as one file; the
GitHub Actions runner and GitHub Pages are case-sensitive.

**Consequences / how to rename.**

- Page files are lowercase snake_case (CONVENTIONS, DECISIONS #48).
- A case-only rename needs a temporary name, or Windows treats it as a no-op:
  `git mv python/01_Intro.ipynb python/__tmp.ipynb && git mv python/__tmp.ipynb python/01_intro.ipynb`.
  This also works when the disk copy already has the new case.
- Keep old URLs alive with `aliases: [01_Intro.html]` in the page's front matter. Verified on the
  live site after the 2026-09-28 deploy: all 8 old URLs serve a 200 redirect page and a browser
  lands on the lowercase page, with `#anchors` kept (`01_Intro.html#1-print` ->
  `01_intro.html#1-print`); the CI render printed 0 "Unable to resolve link target" warnings
  (15 before). Untested caveat: on a Windows local render the alias file and the page would be
  the same file, so renders stay CI-only.
- Take paths for links from `git ls-files`, never from the directory listing.
