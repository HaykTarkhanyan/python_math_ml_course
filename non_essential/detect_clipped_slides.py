"""Detect silently CLIPPED Beamer content: text in the .tex that never reached the .pdf.

Beamer drops overflowing content WITHOUT an Overfull vbox warning, so the compile log is clean,
the page count is right, and the deck looks finished. This closes most of that gap mechanically:
it compares the prose of each frame in the source against the prose actually on the page it
rendered to, and reports what did not make it.

    ./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch19_mech_interp
    ./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch19_mech_interp/L45_opening_the_box.tex
    ./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch19_mech_interp/L45_opening_the_box.pdf

Exit code is 1 if anything is flagged, 0 if clean, so it can gate a build.

WHAT IT CAUGHT (2026-08-07): L34_vlm_drawing's recap frame silently lost the tail of its final
list item on the last slide of the chapter, with zero Overfull vbox warnings and a correct page
count. It had already survived a self-review that spot-checked other pages.

WHAT IT CAUGHT (2026-08-14, first run after the rewrite): ch11_rl/L32_rl_problem page 12 ends
"The mistake is permanent," - the closing "and the loss grows linearly forever." is in the .tex
and never reached the page. A tcolorbox overflowed and Beamer dropped the rest without a word.
That chapter was considered finished.

REWRITTEN 2026-08-14, for two reasons - both of which had made it useless in practice:

  1. It could not fail. It only ever globbed `*.tex` out of the argument, so passing it a .pdf
     path - or a path that did not exist at all - matched nothing and printed
     "TOTAL frames flagged: 0", exit 0. Several decks in this repo were reported "clean" by a
     run that had checked nothing. It now resolves .tex/.pdf/directory and RAISES on anything
     it cannot check.
  2. False positives swamped it: 60 of 137 frames across ch19, 39 of 51 in ch20. Almost all
     came from LaTeX that is never printed but survived stripping - `tcolorbox` option keys
     (colback, colframe, boxrule, arc, title), tikz overlay options (remember picture, anchor
     south west), and math. Those are now removed before comparison, and matching is scoped to
     the page a frame actually rendered to rather than the whole document.

READ THE OUTPUT, DO NOT TRUST IT BLINDLY. A flag is a place to look, not a verdict.

WHAT IT CANNOT DO: it only knows text. A clipped figure, a box drawn off-slide, or an
overlapping label are invisible to it - use non_essential/detect_footer_collisions.py for the
footer case, and your eyes for the rest.
"""

import re
import sys
from pathlib import Path

import fitz

TAIL = 7          # n-gram length; long enough to be specific, short enough to survive wrapping

# A frame must match its page at least this well before we trust the alignment. Below it, the
# frame is reported as unmatched rather than silently compared against the wrong page.
MIN_PAGE_COVERAGE = 0.45

# LaTeX command and environment names that survive stripping and would otherwise look like
# missing prose. Extend freely - a wrong entry only costs a missed flag in one n-gram.
JUNK = re.compile(r"^(pt|em|ex|cm|mm|in|armred|armblue|paramgreen|armorange|popblue|violet1|"
                  r"orange1|sampred|lightbg|linewidth|textwidth|parbox|fcolorbox|vskip|"
                  r"vfill|hfill|centering|small|footnotesize|scriptsize|large|textbf|emph|"
                  r"item|itemsep|column|columns|center|minipage|tabular|toprule|midrule|"
                  r"bottomrule|multicolumn|rowcolor|includegraphics|begin|end|frame|plain|"
                  r"document|tikzpicture|enumerate|itemize|pause|l|c|r|t|b)$")

# Environments whose body is markup, not prose the reader sees as sentences.
DROP_ENVS = ("tikzpicture", "verbatim", "lstlisting", "semiverbatim")


def prose(text):
    """Alphabetic words only, with LaTeX residue and bare numbers dropped.

    Hyphens - and any whitespace around them - are deleted on both sides of the comparison.
    Three separate artifacts need this:

      * a line break at a SOURCE hyphen ("byte-\\nfor-byte") extracts as "bytefor-byte",
        which tokenises differently from the source's "byte-for-byte";
      * LaTeX's own hyphenation ("interpretabil-\\nity") needs the opposite treatment;
      * brace-stripping inserts a space that the reader never sees - "\\textbf{Exactly}-orthogonal"
        becomes "Exactly -orthogonal" in the source but stays "Exactly-orthogonal" in the PDF.

    Collapsing hyphen-plus-surrounding-space to nothing makes all three agree. It also merges
    this repo's " - " dash into its neighbours, which looks wrong but is harmless: the same
    transformation is applied to source and PDF, and only agreement matters.
    """
    # ONE rule, and the order matters: \s already covers the newline, so a separate "-\n"
    # step run first would collapse a line-broken dash into "a b" while the source collapsed
    # it to "ab", and every dashed sentence would flag. The character class covers the en and
    # em dashes too, because LaTeX turns "--" in the source into an en dash in the PDF and the
    # two sides must normalise identically.
    # Join a hyphenated line break first (covers both TeX's own hyphenation, "interpretabil-\n
    # ity", and a source hyphen that happened to land at a break, "byte-\nfor-byte"), then
    # delete every remaining dash. Whitespace is deliberately NOT touched: collapsing it would
    # fuse the neighbours of a standalone dash, which is what a table cell meaning "nothing"
    # looks like. The class covers the en/em dashes because LaTeX turns "--" into one.
    text = text.replace("\u00ad", "").replace("\ufffd", "")
    text = re.sub(r"[-\u2010-\u2015]\n", "", text)
    text = re.sub(r"[-\u2010-\u2015]", "", text)
    return [w for w in re.findall(r"[A-Za-z]+", text.lower())
            if len(w) > 1 and not JUNK.match(w)]


def strip_tex(t):
    """Remove everything that will not be printed, so what is left is what we expect to see."""
    t = re.sub(r"(?<!\\)%.*", "", t)

    # Environments whose contents are coordinates and options rather than sentences.
    for env in DROP_ENVS:
        t = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", t, flags=re.S)

    # Math never survives PDF extraction reliably; drop it from both sides of the comparison.
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)

    # Colour commands take the colour as their FIRST braced argument and the visible text as
    # the second, so exactly one group may be eaten - \textcolor{gray}{...} must not swallow
    # the text. Left in, the colour name reads as prose and flags every frame that uses one.
    t = re.sub(r"\\(textcolor|color|cellcolor|rowcolor|columncolor|colorbox)\s*"
               r"(\[[^\]]*\])?\s*\{[^{}]*\}", " ", t)

    # Non-printing commands, with all of their braced arguments.
    t = re.sub(r"\\(label|ref|cite|hypertarget|usebackgroundtemplate|definecolor|"
               r"setbeamer[a-zA-Z]*|includegraphics)\s*(\[[^\]]*\])?\s*(\{[^{}]*\})*", " ", t)

    # Tabular-likes carry a COLUMN SPEC in a second brace group: \begin{tabular}{@{}lrp{3cm}@{}}.
    # Left alone it reads as prose ("lr", "ccc", "rp") and flags every table in the deck.
    # The inner alternation allows one level of nesting, which p{...} and @{} need.
    brace = r"\{(?:[^{}]|\{[^{}]*\})*\}"
    t = re.sub(rf"\\begin\s*\{{(?:tabular|tabularx|array|longtable)\*?\}}\s*"
               rf"(\[[^\]]*\])?\s*(\{{[^{{}}]*\}})?\s*{brace}", " ", t)

    # \begin{env}[options] / \end{env} - the option block was the single biggest source of
    # false positives (tcolorbox's colback/colframe/boxrule/arc/title are never printed).
    t = re.sub(r"\\begin\s*\{[^}]*\}\s*(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\end\s*\{[^}]*\}", " ", t)

    # Any remaining command plus its optional-argument block.
    t = re.sub(r"\\[a-zA-Z@]+\s*(\[[^\]]*\])?", " ", t)

    # Braces normally become a space, but NOT where they sit against a hyphen:
    # "\textbf{Exactly}-orthogonal" must strip to "Exactly-orthogonal" the way the reader sees
    # it, not to "Exactly -orthogonal", or it tokenises differently from the PDF.
    t = re.sub(r"\}(?=-)", "", t)
    t = re.sub(r"(?<=-)\{", "", t)
    return re.sub(r"[{}$&\\~^_]", " ", t)


def grams_of(words):
    return {tuple(words[i:i + TAIL]) for i in range(len(words) - TAIL + 1)}


def resolve(target: Path):
    """Return [(tex, pdf), ...] or raise. Never silently return nothing."""
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")

    if target.is_dir():
        texs = sorted(target.glob("*.tex"))
        if not texs:
            raise FileNotFoundError(f"no .tex files in {target}")
    elif target.suffix.lower() in (".tex", ".pdf"):
        texs = [target.with_suffix(".tex")]
        if not texs[0].exists():
            raise FileNotFoundError(
                f"{texs[0]} not found - this check needs the SOURCE, not just the PDF, "
                "because it compares the two."
            )
    else:
        raise ValueError(f"{target} is not a .tex, a .pdf, or a directory")

    pairs = []
    for tex in texs:
        pdf = tex.with_suffix(".pdf")
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} not found - compile {tex.name} first")
        pairs.append((tex, pdf))
    return pairs


def check(tex, pdf):
    body = tex.read_text(encoding="utf-8").split(r"\end{document}")[0]
    # Content strictly BETWEEN begin/end - otherwise each chunk carries \end{frame} plus the
    # section markup that follows it, and every frame looks broken.
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", body, re.S)

    doc = fitz.open(pdf)
    pages = [prose(pg.get_text()) for pg in doc]
    page_grams = [grams_of(p) for p in pages]
    page_sets = [set(p) for p in pages]

    bad = 0
    for n, fr in enumerate(frames, 1):
        w = prose(strip_tex(fr))
        if len(w) < TAIL:
            continue

        # Scope the comparison to the page this frame actually rendered to. Comparing against
        # the whole document lets any page mask a clip on any other; comparing in source order
        # breaks on \pause, which turns one frame into several pages.
        want = set(w)
        best, cover = max(
            ((i, len(want & s) / len(want)) for i, s in enumerate(page_sets)),
            key=lambda t: t[1],
        )
        title = re.match(r"\{([^}]*)\}", fr.strip())
        label = title.group(1)[:44] if title else "plain"

        if cover < MIN_PAGE_COVERAGE:
            bad += 1
            print(f"  frame {n:2d} [{label}]")
            print(f"      NO MATCHING PAGE (best {cover:.0%}) - frame may be missing entirely")
            continue

        grams, words_on_page = page_grams[best], page_sets[best]
        missing, last = [], -99
        for i in range(len(w) - TAIL + 1):
            window = tuple(w[i:i + TAIL])
            # Exact ordered match, or - for tables and multi-column frames, where PDF text
            # extraction order does not follow source order - every word present on the page.
            if window in grams or all(x in words_on_page for x in window):
                continue
            if i - last > TAIL:
                missing.append(list(window))
            else:
                missing[-1].append(w[i + TAIL - 1])
            last = i

        if missing:
            bad += 1
            print(f"  frame {n:2d} [{label}]  (page {best + 1})")
            for m in missing:
                print(f"      NOT RENDERED: {' '.join(m)}")

    print(f"  {tex.name}: {len(frames)} frames, {bad} flagged")
    return bad


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: detect_clipped_slides.py <dir | deck.tex | deck.pdf>")
    total = 0
    for tex, pdf in resolve(Path(sys.argv[1])):
        total += check(tex, pdf)
    print(f"\nTOTAL frames flagged: {total}  (verify each against the rendered page)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
