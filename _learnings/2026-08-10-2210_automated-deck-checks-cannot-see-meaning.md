# Every automated deck check passes a slide that teaches the opposite of the truth

**Symptom.** L43 shipped a 2x2 grid on grounding ("Licensing 'I do not know'") whose two
off-diagonal cells were labelled backwards:

| | the system answers | the system refuses |
|---|---|---|
| **context contains the answer** | Correct answer | **"Made up", the failure everyone fears** |
| **it does not** | **"Refused anyway", annoying but safe** | "I do not know" |

Fabrication is answering *without* support, so "Made up" belongs bottom-left. Refusing when
the answer was sitting in the context is merely a wasted opportunity, so "Refused anyway"
belongs top-right. As shipped, a core concept slide taught students to associate
hallucination with the wrong quadrant.

**Cause.** The cell list in `l43_diagram_figs.py` had the `(col, row)` coordinates right and
the titles swapped between two entries. Nothing downstream could notice.

**What did not catch it.** Everything that ran on that deck passed it:

- `margin_check.py` (pixel scan for clipped content) - **passed**, the layout was perfect.
- The acronym grep from `WORKFLOWS.md` - **passed**, no undefined acronyms.
- `pdflatex -halt-on-error`, twice - **passed**, it compiled cleanly.
- The building agent's own final audit, which *did* catch three arithmetic slips in the same
  deck ("42 words" -> 49, "four of seven" -> five of seven, "within one rank" -> two ranks).
- My own independent verification, which checked page count, figure count, artifact count,
  margins, and that the honest caveats appeared in the slide body rather than only in
  comments. All green.

**What did catch it.** A Sonnet student-review subagent reading only the rendered PNGs, on
its first pass, ranked as its number one finding. Its reasoning was semantic, not textual:
it read the row and column headers, worked out what each cell *meant*, and noticed the
meaning contradicted the label.

**Consequences.**

- The margin scanner and the acronym grep verify *form*. They cannot verify *meaning*, and
  no amount of hardening will change that. A diagram can be pixel-perfect, compile cleanly,
  define every term, and still be wrong.
- The student review is therefore not an optional polish step for a deck containing
  conceptual diagrams - it is the only check in the pipeline that reads for sense. Budget
  for it (about 40-90k tokens and 5-8 minutes per deck; the three-deck RAG chapter cost
  roughly 320k tokens total).
- The caption was also rewritten to name the boxes ("Tighten the prompt and *Made up*
  shrinks") rather than their positions ("the top-right box shrinks"). Position-based prose
  silently goes wrong the moment a layout changes; label-based prose cannot.
- Two of the same review round's findings were also invisible to form-checking: prose that
  contradicted the chart beside it ("RAKE and YAKE made it worse" where the plotted rank was
  unchanged at 2), and a table header that made a correct number look impossible
  ("Query words present" showing 4 for a three-word query).

**The transferable rule.** Automated checks are necessary and cheap, so keep running them -
but treat a green board as evidence that nothing is *broken*, never as evidence that
something is *right*. For any frame carrying a conceptual claim - a quadrant chart, a
decision table, a labelled diagram - the only real check is a reader who does not already
know what the slide was supposed to say. See also
[2026-08-10-2016_rag-figure-contradicted-its-own-claim.md](2026-08-10-2016_rag-figure-contradicted-its-own-claim.md),
which is the same failure one step earlier: writing the conclusion before the measurement.
