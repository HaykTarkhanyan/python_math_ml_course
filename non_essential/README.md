# non_essential/

Maintained, reusable tooling that is **not** needed to build or run the course. Nothing here is
imported by live code. Everything here is expected to still work.

(Load-bearing scripts stay at the repo root: `clean_latex.py`, `render_only_changed.py`.)

| Script | What it does | When you would run it |
|---|---|---|
| `detect_clipped_slides.py` | Compares the prose in each Beamer frame against the prose actually present in the rendered PDF, and reports what did not make it onto the slide. | On any deck before committing it, as part of the deck polish loop. Added 2026-08-08. |
| `check_links.py` | Finds broken links on the website. Default (source) mode checks every relative and GitHub/Colab link in the pages of `_quarto.yml` against git-tracked files, case-sensitively like the Linux build, plus placeholder links and unfilled template targets (~2 s, offline). `--live` crawls the deployed site and checks anchors, YouTube videos and external URLs (~3-5 min). Exit 1 if anything is broken; details in `logs/check_links_<mode>.json`. Needs `requests==2.32.5`, `beautifulsoup4==4.15.0`, `tqdm==4.67.1` (all in `ma`). | Source mode before pushing site changes; `--live` every few weeks for link rot. Deliberately not a CI gate (instructor's call, DECISIONS #53). Added 2026-09-28 after a crawl found 24 dead internal links the build had only warned about. |
| `deck_contact_sheet.py` | Renders a deck into 3x3 contact sheets with page numbers stamped on, so a whole deck can be looked at in a few images. | After the detectors, for what they cannot see: unreadably small figure text, overlay steps that shift, badly wrapped tables. Added 2026-09-23 while building the ch19 extension, where it caught figures drawn too wide (text at ~4pt on the slide) and a TikZ diagram jumping between overlay steps - both passed both detectors. |

## Why `detect_clipped_slides.py` exists

Beamer discards overflowing content **silently** - no Overfull vbox warning, correct page count,
clean log. `LEARNINGS.md` records this and concludes that only visual inspection catches it,
which is true but means eyeballing every page of every deck; a reviewer who spot-checks will
miss things.

It earned its place on 2026-08-07: `ml/ch12_vlm/L34_vlm_drawing.tex` had silently lost the tail
of the final list item on its **recap frame** - the last slide of the chapter - and the deck had
already passed a self-review that happened to check other pages. `pdflatex` reported zero
problems.

Usage:

```bash
./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch12_vlm
```

Exit code 1 if anything is flagged, 0 if clean.

**It produces false positives by design.** N-grams that span a figure, a table or a column
boundary get flagged, as does LaTeX residue and any accented character (the tokenizer is
ASCII-only, so "Muller" becomes "ller"). Treat a flag as a place to look, not a verdict - open
the rendered page and decide.

It also only knows about **text**. A clipped figure, a box drawn past the slide edge, or an
overlapping label are invisible to it, so visual checking is still required. What this changes is
that the checking becomes targeted rather than exhaustive.
