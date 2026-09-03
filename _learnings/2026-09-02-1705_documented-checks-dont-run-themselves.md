# A check that lives as prose in a style guide does not get run

**Symptom.** The ch10 dimensionality-reduction decks - reviewed once in July, revised heavily in
August - shipped with `t-SNE` and `UMAP` never expanded *in the deck that introduces them to the
course*, plus unexpanded `LSA`, `EVR`, `DR`. Separately, the 2026-08-16 renumber sweep fixed 32
files that referenced the old `L13b`/`L13c` names but left three student-visible `L13b`/`L13c`
references inside the two renamed decks themselves. Both classes of defect survived a build, a
delivery-oriented revision pass, and multiple close readings.

**Cause.** Both checks exist only as instructions a human (or model) must remember to run:

- `ml/SLIDE_STYLE.md` documents the acronym grep and even says "this rule keeps getting broken,
  so verify it mechanically" - but nothing runs it, so it was not run.
- The rename sweep grepped for files *referencing* the old names; the renamed files' own bodies
  were edited for content that day, so their self-references (`"Next (L13c):"`) read as familiar
  and were skimmed past. A reference sweep must include the renamed files themselves.

Found 2026-09-02 by finally running the documented check verbatim:

```bash
sed 's/%.*//' DECK.tex | grep -oE '\b[A-Z]{2,6}\b' | sort -u   # then check each expansion
grep -n "L13" ml/10_dimensionality_reduction/*.tex             # 7 hits, 3 student-visible
```

**Consequences.**

- The blind spot is worst for a deck's *own subject*: nobody expands "t-SNE" in the t-SNE
  lecture because everyone in the room already knows what the deck is about. The style guide's
  observed failure list (frame titles, forward references, mid-deck lists) should add this case:
  the headline method itself.
- Any check the style guide phrases as "run this before the deck is done" is a candidate for the
  deck-done tooling (`detect_clipped_slides.py` precedent - which itself taught the same lesson:
  see 2026-08-20-1745 "the clipped slide detector works, nobody ran it"). Until it is wired into
  a script, treat "documented" as "not happening".

Related: [[2026-08-20-1745_the-clipped-slide-detector-works-nobody-ran-it]],
[[2026-08-20-1900_git-grep-cannot-see-untracked-work]]
