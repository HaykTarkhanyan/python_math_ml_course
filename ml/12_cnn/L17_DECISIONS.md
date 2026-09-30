# L17 CNN Architectures - build decisions + open questions

Built 2026-07-14 from the approved `L17_cnn_architectures_OUTLINE.md`. This file logs
only what the outline left open (or where reality diverged) + questions for the
reviewer. Provenance block at the end of the `.tex` has the full source list.

## Decisions taken (choose-sensibly-and-log)

1. **ILSVRC 2012 error = 16.4%.** Some sources say 15.3% for AlexNet; that was the
   SuperVision entry trained WITH extra (pre-ILSVRC-2012) data. The winning
   no-outside-data entry was 16.4% - matches the outline's number and the canonical
   CS231n chart. Kept 16.4, cited "Russakovsky et al. 2015" on the chart.
2. **GoogLeNet parameters: "under 7M".** Web sources disagree (5M in most blogs,
   6.8M in the Wolfram/torchvision model cards, "12x fewer than AlexNet" in the
   paper). Per the outline's own instruction, the slide says "under 7M parameters"
   vs AlexNet's 60M.
3. **Winner names moved into the chart's x-tick labels** (not floating above bars):
   avoids the label collisions the first draft had, and - important for the
   flip-book - a name only appears once its bar is revealed, so the anim does not
   spoil the 2012 reveal. Frame 0 (empty axes) is generated but unused in the deck;
   the cold open starts at frame 1 (2010 bar).
4. **AlexNet dense-head share verified by computation**: FC layers = 58.6M of
   ~61.0M total (96.2%), grouped convs accounted. Slide says "roughly 58 of its
   60M" - the paper's own "60 million" total kept.
5. **Receptive-field forward pointer corrected L18 -> L19.** The outline (written
   before the 4-deck split) says dilated convolutions come in L18; the chapter plan
   moved all conv variants to L19. The slide points to L19.
6. **Two link buttons on the LeNet frame** (Adam Harley 3D viz + CNN Explainer).
   The build notes say "one per frame max" but the same notes explicitly list both
   for this frame; the frame spec won.
7. **VLM chat link = claude.ai** (the sanctioned default; outline said pick one, do
   not list all).
8. **SAM demo link = https://sam2.metademolab.com/** - the outline's
   segment-anything.com/demo did NOT resolve (DNS failure) at build time; Meta's
   SAM 2 demo is live and confirmed. If segment-anything.com comes back, either
   works.
9. **GoogLeNet architecture figure = d2l's 90-degree (horizontal) variant** - the
   vertical original wastes a 16:9 slide.
10. **ImageNet sample photos = AlexNet paper Figure 4 (left panel) crop**: real
    ILSVRC test images WITH the model's five guesses, so one image doubles as the
    top-5-error illustration on the ILSVRC frame. Beats a hand-assembled montage.
11. **Residual-block ANIM candidate dropped** (outline listed it as optional "(a) or
    (b), pick at least one"): the imagenet_anim flip-book (a, 6 frames) is the
    deck's ANIM; the residual frame uses the d2l plain-vs-residual side-by-side
    diagram, which already carries the build-up.
12. **Ribbon section highlights**: 2009 ImageNet added as a ribbon event (the
    outline's event list starts at 1989 LeNet and jumps to 2012, but Section 1
    needs a "you are here"). Variants: S1=2009, S2={1989,2012,2014}, S3=2015,
    S4=2014, S5=2026 "today", S6={2017..2025 frontier events}.
13. **"Most-cited paper" superlative avoided** on the ResNet transition (not in the
    verified-facts block) - replaced with "the fix reshaped every deep network
    since".
14. **SSI status line**: "left in 2024 to found Safe Superintelligence (SSI) -
    still research-only, no product, by choice." Re-verified 2026-07: $32B
    valuation, ~$6B raised, no product/papers; Sutskever became CEO after Daniel
    Gross left (Jul 2025). Deliberately kept to one clause on the slide.
15. **AMI Labs figure re-verified 2026-07**: $1.03B seed at $3.5B pre-money,
    announced Mar 2026 (TechCrunch); slide says "about $1B within months" per the
    outline's phrasing, source line kept verbatim ("CNBC, Nov 2025; MIT Technology
    Review, Jan 2026").
16. **Krizhevsky beat verified + kept soft**: left Google 2017 ("lost interest"),
    advisor at Dessa, later venture capital (Two Bear Capital, as of 2025). Slide
    keeps the outline's exact soft phrasing ("quietly stepped away from the field")
    with no employer details.
17. **torchvision code frame numbers verified live**: resnet18 = 11,689,512 params
    (printed in the snippet comment); `weights="IMAGENET1K_V1"` API verified against
    installed torchvision 0.24.0.
18. **torchvision pinned to 0.24.0+cpu** (matches the ma venv's torch 2.9.0+cpu;
    an unpinned install would have pulled torch 2.10 and replaced the CPU wheel).
19. **2013 bar labeled "ZFNet"** (the outline offers "Clarifai/ZFNet"): ZFNet is
    the architecture name every textbook chart uses; Clarifai was the winning
    team's company. One name keeps the bar label clean.
20. **cifar_grid.py DROPPED (instructor scope change mid-build, 2026-07-14: no
    dataset downloads, no training).** The in-flight CIFAR-10 download was
    stopped and the partial tar deleted. The recap frame's HW2 bridge uses
    WEB-IMG instead: `fig/borrowed/cifar10_grid.png` (the canonical
    PyTorch-tutorial class grid - 10 labeled classes x 10 samples). Real dataset
    images, nothing invented, no download machinery. pretrained_filters.py kept
    per the same instruction (weights already cached; load-bearing payoff frame).

## Open questions for the reviewer

1. The cold open reveals bars 2010 -> 2015 in 6 clicks; the "Predict: what happened
   in 2012?" pause sits after the 2011 bar. Is a 6-click open too slow to start a
   lecture? (Collapsing to 4 clicks is a 2-line edit.)
2. The Fei-Fei Li frame is text-only (three bullets + takeaway box) - the only
   story frame without a visual. A WordNet-hierarchy graphic or an early-ImageNet
   screenshot could fill it, but nothing iconic exists that isn't a portrait
   (excluded by the no-portraits decision). OK as is?
3. "One paper, three fates": SSI may announce something between now and delivery -
   the line "still research-only, no product, by choice" is the most perishable
   claim in the deck. Re-check the week of the lecture.
4. The recap frame carries the CIFAR-10 grid + an HW2 one-liner. SLIDE_STYLE says
   "no HW frame in the deck"; this is a caption, not a frame, and the outline
   explicitly places the grid in the recap/Next area - but flag if it reads as
   homework creep.
5. Adam Harley's page hosts four demos (2D/3D x dense/conv); the button links the
   landing page rather than deep-linking one demo. Deep-link the 3D conv demo
   instead?
