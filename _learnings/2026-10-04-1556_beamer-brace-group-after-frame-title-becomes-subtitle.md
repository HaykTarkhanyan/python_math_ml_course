# A brace group right after a Beamer frame title becomes the frame's subtitle

**Symptom.** Building `ml/14_llms/LLM1_tokenization.tex` (2026-10-04), six frames rendered their
first paragraph in blue, small, directly under the title, styled like a subtitle instead of body
text. Everything else on those frames looked normal. pdflatex reported 0 errors and 0 overfull
boxes, and all three layout detectors (`detect_clipped_slides.py`, `detect_footer_collisions.py`,
`detect_small_figure_text.py`) reported 0. Only looking at the rendered pages caught it.

**Cause.** Beamer's frame syntax is `\begin{frame}<overlay>[options]{title}{subtitle}`. TeX skips
whitespace and a single newline while looking for the next argument, so a frame that opens with
a font-size group as its first paragraph:

```latex
\begin{frame}{UTF-8: one to four bytes per character}
  {\small UTF-8 (Unicode Transformation Format, 8-bit) writes each code point as 1 to 4 bytes.}
```

passes `{\small ...}` as the **subtitle** argument. The six frames that hit it: UTF-8, BPE: the
idea, Pre-tokenization, Decoding, Special tokens, Glitch tokens: what GPT-3 said.

**Fix.** Put anything that is not a brace group between the title and the paragraph. We used
`\vskip 2pt`:

```latex
\begin{frame}{UTF-8: one to four bytes per character}
  \vskip 2pt
  {\small UTF-8 (Unicode Transformation Format, 8-bit) writes each code point as 1 to 4 bytes.}
```

(`\small` without braces, or an explicit empty subtitle `{}`, would also work. A `{\small ...}`
group further down the frame, after a figure or a list, is fine.)

**Check for it** (tested: it flags the broken form above and passes the fixed one; on 2026-10-04
it found no remaining case in any deck under `ml/` or `misc/`):

```bash
./ma/Scripts/python.exe -c "
import re, glob
pat = re.compile(r'\\\\begin\{frame\}(?:<[^>]*>)?(?:\[[^\]]*\])?\{[^\n]*\}[ \t]*\n[ \t]*\{')
for f in glob.glob('ml/**/*.tex', recursive=True) + glob.glob('misc/**/*.tex', recursive=True):
    n = len(pat.findall(open(f, encoding='utf-8', errors='replace').read()))
    if n: print(f, n)
"
```

**Consequences.** This belongs with the other silent Beamer failures: it compiles cleanly, passes
every automated check, and still puts the wrong thing on the slide. Look at the rendered pages
(the deck-polish loop's step 2) and run the check above when a deck opens frames with a
`{\small ...}` paragraph.
