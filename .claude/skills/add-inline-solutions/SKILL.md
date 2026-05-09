---
name: add-inline-solutions
description: >
  Add or convert collapsible Quarto solution callouts in homework .qmd files.
  Different from `generate-solutions`, which writes a separate LaTeX/Jupyter
  solutions package.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Glob
  - Grep
user-invocable: true
---

# Add Inline Solutions to Homework .qmd

Add new collapsible solution callouts, or convert existing profile-gated
solutions to the modern collapsible style. Solutions render by default in
the live site, **collapsed**; students click to reveal.

## Required pattern

```
::: {.callout-tip collapse="true" title="Solution"}

[content]

:::
```

`title="Solution"` makes the word "Solution" visible in the *collapsed*
header — better UX than the default `Tip` placeholder.

If the callout nests another `:::` block (raw HTML SVG via `{=html}`,
sub-callout, etc.), bump the outer fence to 4 colons (`::::`) so the
parser knows where each ends.

## Display math: break long equalities

Quarto/MathJax does **not** auto-wrap display math, and the rendered
homework column is narrow (~600-700px after the right-side TOC eats
real estate). A chained equality of three or more steps written on one
line will overflow horizontally — the right end runs under the TOC or
gets clipped.

Break any chained equality with three or more `=` steps into an
`aligned` block:

```
$$\begin{aligned}
\text{LHS} &= \text{step 1} \\
&= \text{step 2} \\
&= \text{step 3}
\end{aligned}$$
```

The `&=` aligns each `=` sign vertically. `aligned` works inside
solution callouts without bumping fence count (it's MathJax syntax,
not pandoc fence syntax).

Single-line display equations (one `=`, like $f'(x) = 2x e^{x^2}$) are
fine — the issue is *only* chains. Rule of thumb: if your `$$...$$`
line is >120 characters in the source, it's probably too wide rendered.
Visual check: render and view in a browser at typical width before
claiming the work is done.

## Old → new style conversion

Replace this:

```
::: {.content-visible when-profile="solution"}

#### Solution {.solution-header}
[content]
:::
```

With the new callout above. Drop both the `content-visible` wrapper *and*
the inner `#### Solution` heading — `title="Solution"` already covers it.

## Workflow

1. **Locate work.** Resolve the target file (e.g. `math/NN_*.qmd`). For
   long files (HW 16 is >1400 lines) use `Grep` first to find problem
   headers (`^### \d`) and existing callouts
   (`callout-tip collapse|when-profile="solution"`). Don't `Read` the
   whole file when targeted reads suffice.

2. **Skip.** Don't add solutions for:
   - Pure coding tasks
   - Image-only problems with no text statement
   - Author-marked "in-progress" problems (often Armenian comments like
     *"Սա հետոյվա"* = "this is for later")
   - Open-ended bonus problems
   An external video link is *not* itself a reason to skip — students
   benefit from a written solution alongside.

3. **Solve.** Compute carefully; verify numerics with `python -c` for
   anything non-trivial. Pick depth from the problem's difficulty
   indicator (`{data-difficulty="1|2|3"}`, rendered as 🧀 / 🧀🧀 / 🧀🧀🧀).

4. **Insert via Edit.** Anchor on the *next* problem's full header *with
   attributes* — real headers look like
   `### 03: Title {data-difficulty="2"}`, not `### 03 Title`. The previous
   problem's last line plus the next header together form a unique anchor
   in nearly every file.

5. **Convert** any old-style blocks in the file at the same time, for
   consistency.

6. **Render to verify** (exit code must be 0):

   ```
   quarto render math/NN_FILE.qmd --to html --output-dir /tmp/q-hwNN 2>&1 | tail -10
   ```

   Render failures usually mean fence-nesting bugs (outer needs more
   colons than inner) or unbalanced `:::`.

7. **Report.** List problems solved, problems converted, problems skipped
   (and why), and any issues spotted (numbering duplicates/gaps,
   question/answer mismatches, typos). Ask before committing.

## Defer to other sources

The following live elsewhere — read them when relevant rather than
duplicating here:

- **Course position** (which concepts students have seen by HW N):
  `memory/feedback_concept_position.md`. Critical — never reference
  unseen concepts (no eigenvalues in HW 02, no random variables before
  HW 17, etc.).
- **No em-dashes, Armenian flag colors for plots (red `#D90012`, blue
  `#0033A0`, orange `#F2A800`), don't over-engineer**: project and
  global `CLAUDE.md`.

## Depth calibration

The difficulty attribute is the calibration:

- **🧀 (diff 1)**: a few lines of computation plus a sentence on what it
  means. No need for ML asides or sanity checks if the problem doesn't
  invite them.
- **🧀🧀 (diff 2)**: full computation with intermediate steps; a brief
  intuitive remark or sanity check; an ML/real-world tie-in *only when
  natural*.
- **🧀🧀🧀 (diff 3)**: sub-headers (`**a)**`, `**Part 1:**`), full
  derivation, forward/backward references to related problems, and a
  pedagogical "why this matters" if there's a deep idea involved.

Pure-math problems get pure-math solutions. Don't manufacture ML hooks.
For known-good examples in different topics, read one or two solutions
from a recently-solved problem in the same area (linear algebra,
calculus, probability) — but don't copy length blindly; calibrate to
the problem you're solving.

## Platform note

The `start "" "$(cygpath -w …)"` pattern to auto-open rendered HTML is
Windows-only. On macOS use `open`, on Linux `xdg-open`. Or just print
the path and let the user open it.

## Don't

- Commit automatically — wait for the user
- Touch files other than the target `.qmd`
- Add a `#### Solution` heading inside the callout
- Cite numbers without verifying — if you write "$x \approx 309$ years"
  or any specific quantitative claim, compute it first with `python -c`.
  Hallucinated numbers in solutions train students to distrust the
  source.
