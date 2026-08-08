# Borrowed video stills ship broken in ways no automated check catches

**Symptom.** Chapter 16 embeds 34 full-bleed stills pulled from a two-part video documentary.
Three of them were defective, and the defects survived every check the repo has:

- `pdflatex` compiled with **0 errors** and **0 `Overfull \vbox`**
- `non_essential/detect_clipped_slides.py` reported **0 frames flagged**
- a bespoke contrast check confirmed the attribution node was readable on all 34
- I tiled all 135 rendered pages into a contact sheet and looked at it

All four passed. A first-time-student review reading the rendered pages found all three in one
pass.

**The three defects:**

1. **Mid-animation grab.** `pusht_decoded_actions.jpg` was supposed to show a decoded prediction
   (press up, the effector moves up). Grabbed at 24:30, it caught the animation partway through:
   a garbled half-rendered scribble where the predicted frame should be, and a sliced label. The
   student's report: *"I could not tell if this was intentionally showing the model's prediction
   degenerating, or a broken render."*
2. **Clipped at every timestamp.** `euclidean_to_goal.jpg` lost its left edge. Re-grabbing at
   28:42 / 28:50 / 28:56 / 29:02 / 29:08 / 29:14 produced a clipped frame **every time** - the
   diagram pans horizontally, so no settled frame exists. It was dropped rather than shipped.
3. **Illustration that contradicts its own slide.** The prose described a robot moving an object
   to *a picture of Taylor Swift*; the still showed wooden shapes and a figurine, no picture. The
   correct frame was 20 seconds earlier at 18:38, which shows the framed photos **and** the
   on-screen caption.

**Cause.** Two different mistakes, and only one of them was ignorance.

The timestamps were chosen from the transcript, which tells you when a point is *narrated* - not
when its visual finishes rendering. Animated explainers hold each beat for a second or two after
the narration lands. **The repo's own `youtube-reference` skill already says this**: *"grab a
second or two after the narration lands, not mid-transition/scroll."* I did not follow it.

The Taylor Swift mismatch is a separate failure: I never checked that the image illustrates the
claim the slide makes. A still can be perfectly rendered and still be the wrong still.

**Consequences.**

- No automated check can catch any of this. A missing image fails the build loudly; a *wrong* or
  *half-rendered* image compiles perfectly. Clip detectors look for content at page edges, and a
  full-bleed photo has content at every edge by construction.
- The contact-sheet pass is **not** sufficient. At tile size a garbled scribble reads as "some
  diagram", and you cannot tell that a photo fails to contain what the neighbouring slide
  promises. It catches missing images; it does not catch wrong ones.
- Re-grabbing is cheap (seconds) and should be done in **batches of candidates**, tiled and
  compared, rather than one timestamp per beat. That is how the replacements were found: six
  candidates around each beat, tiled 3x2, pick the settled one.

**The rule that follows:** for every borrowed still, ask two questions that no tool will ask for
you - *has the animation finished?* and *does this image contain what the slide says it does?*
Then have someone who has not seen the source material look at the rendered pages.
