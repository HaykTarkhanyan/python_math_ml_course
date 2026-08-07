"""Detect silently CLIPPED Beamer content: text in the .tex that never reached the .pdf.

Beamer drops overflowing content WITHOUT an Overfull vbox warning, so the compile log is clean,
the page count is right, and the deck looks finished. `LEARNINGS.md` records this and concludes
that only visual inspection catches it. That is true but weak: it means every page of every deck
has to be eyeballed, and a reviewer who spot-checks will miss things.

This closes most of the gap mechanically. It compares the prose in each frame of the source
against the prose actually present in the rendered PDF, and reports what did not make it.

    ./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch12_vlm

Exit code is 1 if anything is flagged, 0 if clean, so it can gate a build.

WHAT IT CAUGHT (2026-08-07): L34_vlm_drawing's recap frame silently lost the tail of its final
list item - "decoupled encoders (Janus). GPT-4o most likely resembles the second." - on the last
slide of the chapter, with zero Overfull vbox warnings and a correct page count. It had already
survived a self-review that spot-checked other pages.

READ THE OUTPUT, DO NOT TRUST IT BLINDLY. False positives are common and benign:
  - n-grams that span a figure, a table, or a column boundary
  - LaTeX residue that survives stripping (column specs like `llp`, tikz node text)
  - accented characters, which the ASCII-only tokenizer splits ("Muller" -> "ller")
Every flag needs a look at the rendered page before it is believed. A flag is a place to look,
not a verdict.

WHAT IT CANNOT DO: it only knows text. A clipped figure, a box drawn off-slide, or an
overlapping label are invisible to it. Visual inspection is still required; this just makes it
targeted instead of exhaustive.
"""

import re
import sys
from pathlib import Path

import fitz

TAIL = 7          # n-gram length; long enough to be specific, short enough to survive wrapping

# LaTeX command names and environment names that survive brace-stripping and would otherwise
# look like missing prose. Extend freely - a wrong entry only costs a missed flag in one n-gram.
JUNK = re.compile(r"^(pt|em|ex|cm|mm|in|armred|armblue|paramgreen|armorange|popblue|violet1|"
                  r"orange1|sampred|lightbg|linewidth|textwidth|parbox|fcolorbox|vskip|"
                  r"vfill|hfill|centering|small|footnotesize|scriptsize|large|textbf|emph|"
                  r"item|itemsep|column|columns|center|minipage|tabular|toprule|midrule|"
                  r"bottomrule|multicolumn|rowcolor|includegraphics|begin|end|frame|plain|"
                  r"document|tikzpicture|enumerate|itemize|pause|l|c|r|t|b)$")


def prose(text):
    """Alphabetic words only, with LaTeX residue and bare numbers dropped."""
    text = text.replace("-\n", "").replace("\u00ad", "")
    return [w for w in re.findall(r"[A-Za-z]+", text.lower())
            if len(w) > 1 and not JUNK.match(w)]


def strip_tex(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\s*(\[[^\]]*\])?", " ", t)
    return re.sub(r"[{}$&\\~^_]", " ", t)


def check(tex, pdf):
    body = tex.read_text(encoding="utf-8").split(r"\end{document}")[0]
    # Content strictly BETWEEN begin/end - otherwise each chunk carries \end{frame} plus the
    # section markup that follows it, and every frame looks broken.
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", body, re.S)

    doc = fitz.open(pdf)
    rendered = prose(" ".join(pg.get_text() for pg in doc))
    grams = {tuple(rendered[i:i + TAIL]) for i in range(len(rendered) - TAIL + 1)}

    bad = 0
    for n, fr in enumerate(frames, 1):
        w = prose(strip_tex(fr))
        if len(w) < TAIL:
            continue
        # Check EVERY window, not just the frame's tail: in a two-column frame the left column
        # can clip while the right column renders, so the last words prove nothing.
        missing, last = [], -99
        for i in range(len(w) - TAIL + 1):
            if tuple(w[i:i + TAIL]) not in grams:
                if i - last > TAIL:
                    missing.append(w[i:i + TAIL])
                else:
                    missing[-1].append(w[i + TAIL - 1])
                last = i
        if missing:
            bad += 1
            title = re.match(r"\{([^}]*)\}", fr.strip())
            print(f"  frame {n:2d} [{title.group(1)[:44] if title else 'plain'}]")
            for m in missing:
                print(f"      NOT RENDERED: {' '.join(m)}")
    print(f"  {tex.name}: {len(frames)} frames, {bad} flagged")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: detect_clipped_slides.py <dir-with-tex-and-pdf>")
    target = Path(sys.argv[1])
    total = 0
    for tex in sorted(target.glob("*.tex")):
        pdf = tex.with_suffix(".pdf")
        if pdf.exists():
            total += check(tex, pdf)
        else:
            print(f"  {tex.name}: no matching PDF, skipped")
    print(f"\nTOTAL frames flagged: {total}  (verify each against the rendered page)")
    sys.exit(1 if total else 0)
