# Look at every borrowed video still before writing its caption

**Symptom.** Two of the 3Blue1Brown stills extracted in July (`ml/14_llms/fig/borrowed/3b1b/`) did
not show what their names and the manifest said:

- `fs_creature.jpg` (grabbed at 04:12) is a second frame of the murder-mystery scene from 04:11,
  not the "fluffy blue creature" - L24 showed the mystery twice for months.
- `fs_qkspace.jpg` is the pi creature asking "What does W_Q do to the non-nouns?", not the
  query/key space; the restructured LLM-3 first captioned it "W_Q and W_K project every word into
  one smaller space", which the picture does not show.

The first was caught on a 40-dpi contact sheet of the deck (two identical black pages); the second
only when the rebuilt deck was rendered and read.

**Cause.** Single-frame grabs at a timestamp land on whatever the animation shows at that second;
the manifest was written from the transcript, not from the images.

**Consequences.**

- Before using a borrowed still, look at it (a contact sheet of the whole folder takes seconds:
  PIL thumbnails with the filename on each). All 13 chapter-7 stills were checked this way on
  2026-10-05 before LLM-4 used them.
- Write captions from the image, not from the manifest or the timestamp.
- Note a known-bad still in the folder's `README.md` rather than deleting it.
