---
name: beamer-overflow-check
description: >
  Detect and fix content overflow, cutoff, and clipping issues in Beamer
  LaTeX slide decks. Renders each PDF page as an image, visually inspects
  for clipped TikZ diagrams, text running off edges, content overlapping
  page numbers, or frames with missing bottom content.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Glob
  - Grep
user-invocable: true
---

# Beamer Overflow / Cutoff Detection and Fix

Beamer silently clips content that exceeds the frame area — it does NOT
produce overfull vbox warnings like normal LaTeX. The only reliable
detection method is visual inspection of rendered pages.

## Step 1: Find and compile the slide deck(s)

```
# Find .tex files in the target directory
# Compile twice for correct page counters
pdflatex -interaction=nonstopmode FILENAME.tex
pdflatex -interaction=nonstopmode FILENAME.tex
```

Verify zero `!` errors in the .log file:
```
grep -c "^!" FILENAME.log
```

## Step 2: Render all pages to PNG images

Use `pdftoppm` to convert each page to an image:
```
mkdir -p /tmp/beamer_check
pdftoppm -png -r 150 FILENAME.pdf /tmp/beamer_check/deck
```

Note: filenames may be zero-padded (`deck-01.png`) or not (`deck-1.png`)
depending on page count. Check with `ls /tmp/beamer_check/` first, then
convert to Windows paths with `cygpath -w` if needed for the Read tool.

## Step 3: Visually inspect every page

Read each PNG with the Read tool. Look for these issues:

| Issue | What to look for |
|-------|-----------------|
| **TikZ clipped on right** | Rightmost node partially visible or cut off |
| **Bottom content cut off** | Text, listings, or blocks missing at bottom edge |
| **Page number overlap** | Content text running into the `N / M` footer |
| **Severe overflow** | Large portions of the slide missing entirely |
| **Wrong page count** | Footer shows `8/7` — needs second pdflatex pass |

## Step 4: Fix each overflow issue

Apply fixes based on the issue type:

### TikZ diagrams too wide
Reduce `text width` and horizontal `node distance`:
```latex
% Before (too wide)
node distance=0.6cm and 1.0cm,
box/.style={..., text width=3cm, ...}

% After (fits 16:9 frame)
node distance=0.5cm and 0.6cm,
box/.style={..., text width=2.2cm, ...}
```
Rule of thumb for 16:9 Beamer (14.4cm usable width):
- 3 nodes across: text width up to 3.5cm
- 4 nodes across: text width up to 2.8cm
- 5 nodes across: text width up to 2.2cm

Also consider `font=\footnotesize` on nodes.

### Bottom content overflow
In order of preference:
1. **Reduce spacing**: `\bigskip` → `\smallskip` or `\medskip`
2. **Compress lists**: `\setlength\itemsep{0em}` after `\begin{itemize}`
3. **Shrink text**: `\small` or `\footnotesize` for secondary content
4. **Compact code listings**: flatten JSON/code structure to fewer lines
5. **Shorten content**: remove or abbreviate less essential items
6. **Split into two frames**: last resort for severely overfull slides

### Page number overlap
Footer text overlapping `N / M` — the content is almost fitting but not
quite. Use `\footnotesize` for the bottom-most text, or remove a
`\bigskip` / blank line above it.

## Step 5: Recompile and re-verify

After fixing, recompile (2 passes) and re-render only the fixed pages:
```
pdflatex -interaction=nonstopmode FILENAME.tex
pdflatex -interaction=nonstopmode FILENAME.tex
pdftoppm -png -r 150 -f PAGE -l PAGE FILENAME.pdf /tmp/beamer_check/fix
```

Read the re-rendered PNG to confirm the fix. Repeat until all pages pass.

## Step 6: Clean up

```
rm -rf /tmp/beamer_check
```

## Common Beamer frame capacity guidelines (16:9, default theme)

- **Text-only frame**: ~18 lines of `\normalsize`, ~22 of `\small`
- **Frame with title**: subtract ~2 lines for the frametitle
- **Two-column frame**: each column fits ~14 lines of `\small`
- **Frame with code listing**: `\scriptsize` monospace fits ~20 lines
- **Frame with TikZ + text below**: TikZ typically uses 40-50% of height
- **Frame with block environments**: blocks add ~1.5 lines of padding each
