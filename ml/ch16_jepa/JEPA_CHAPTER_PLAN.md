# Chapter plan - JEPA and world models (L39, L40)

**Status:** DRAFT OUTLINE, awaiting instructor approval. Nothing built yet.
**Date:** 2026-08-08.

**Research:** `_knowledge/jepa/` (8 files). The I-JEPA paper and the JEPA tutorial were downloaded
and read in full; every number below marked from them is `[read]`. See `_knowledge/jepa/sources.md`
for the status of everything else, including two claims that circulate widely and are **wrong**.

**Video reference:** `_reference_welchlabs_lecun_p1/` and `_reference_welchlabs_lecun_p2/` -
Welch Labs, *"Yann LeCun's $1B Bet Against LLMs"*, parts 1 (37:24) and 2 (40:57), fetched
2026-08-08 at 1080p with transcripts and 43 stills. **Both READMEs carry a full timestamped beat
map** - read those before building, they are the shot list. The series interviewed LeCun directly
and consulted Stephane Deny, David Fan, Nicolas Ballas, Randall Balestriero and the VL-JEPA
authors, so its LeCun quotes are primary.

---

## Why this chapter exists

Every chapter so far has taught a model by teaching its architecture. This one is about the
**objective** - specifically about one question the course has been answering implicitly for
fifteen chapters and never asked out loud:

> **What should a model be asked to predict?**

The course has now used four different answers. Chapter 8 predicts the input. Chapter 10 predicts
a distribution. The LLM material predicts the next token. Chapter 12's CLIP predicts nothing - it
predicts *whether two things match*. JEPA is the first one in the course that predicts **a
representation the model itself invented**, and the evidence that this matters is unusually clean:

| Target | Arch | Epochs | 1% ImageNet top-1 |
|---|---|---|---|
| Target-encoder output | ViT-L/16 | 500 | **66.9** |
| Pixels | ViT-L/16 | 800 | **40.7** |

Same everything else. The pixel run gets 60% more training and loses 26.2 points `[read, Table 7]`.

The second reason is that this chapter carries a **live, unsettled dispute** that students can
watch being adjudicated with real money: the most prominent AI researcher alive says the dominant
paradigm is a dead end, left Meta over it in November 2025, and raised $1.03B in March 2026 to
prove it. The evidence today cuts both ways. That is far better teaching material than a settled
result, provided the chapter refuses to pick a side.

**Write it as a question, not a survey, and not an advertisement.**

---

## Instructor decisions (2026-08-08)

**1. Two decks.** L39 (the objective) and L40 (the alternative stack). Confirmed - the one-deck
fallback at the bottom of this file is dead unless reopened.

**2. No project component, and the student project is deferred.** Same shape as ch14's "no
practical, no project, no training" decision. **Explanatory slides only.** The LeWorldModel term
project - the one genuinely reproducible artifact in the chapter - is parked in `DEFERRED_TODO.md`
rather than dropped, because it is the strongest project this chapter could ever have and the
reason it is strong will not expire. The project ideas stay in `_knowledge/jepa/06_teaching_notes.md`
as research, not course content.

**3. The collapse experiment is deferred, not cancelled.** "Not yet" - so **no Python experiment
runs for this chapter now.** Consequence handled below at L39 frame 15: the frame survives, its
figure does not, and the interim treatment is specified so the deck is buildable today and the
figure can drop in later **without renumbering anything**. Also parked in `DEFERRED_TODO.md`.

**4. Lean on the videos hard, especially part 2.** *"I like the video a lot, incorporate a lot of
stuff from there, especially the 2nd one."* So the Welch Labs series is not a garnish, it is a
primary spine for L40. Applied below: the **alternative-stack** restructure stays, part 2 gets
**five more frames** it had been compressing (the visual-bandwidth argument, the VLA critique split
into its two real prongs with its counter-evidence, the decoded-rollout demo, the "learned video
game" framing), and the still count goes up.

**5. Borrowed stills: use them freely. Guided screening is fine.** *"I'm fine with guided
screening, the important thing is student content, not author."* **My pacing objection is
overruled and that is the right call** - I was optimising for the deck feeling like ours, which is
a worse objective than the room understanding JEPA. So: **all 26 shortlisted stills are in, plus 8
more promoted from the reserve** (listed below). Attribution stays correct and visible; nothing
else about authorship gets optimised for.

**6. Energy-based models: stated, one picture.** One frame plus the sculpted-vs-flat landscape
figure, matching the ch11/ch12/ch14 precedent. L39 frame 16 as already written; figure 5 stays on
the build list.

**7. Borrowed video frames go in as full-bleed stills, always.** Never redrawn into house-style
figures, never boxed with a caption. The instructor likes the architecture diagrams specifically,
so those get priority. Consequences, all applied below:

- Use the 16:9 `\wlslide{}` macro from the `youtube-reference` skill (this deck is
  `aspectratio=169`, so the still fills the frame exactly - no letterboxing needed).
- Attribution is a small white-70% node in the bottom-left corner: `Welch Labs (2026)`. Nothing
  else on the slide - no title, no caption, no bullet.
- Frames land in `ml/ch16_jepa/fig/borrowed/welchlabs/` with descriptive names, matching how
  `ml/ch9_attention/fig/borrowed/3b1b/` is organised.
- **This does not displace the Python figures that carry data.** Split recorded in the figure
  table below: borrowed stills for *architecture and narrative*, Python for *every number and
  every measurement*. A Welch Labs still is someone else's explanation; a bar chart of the I-JEPA
  ablations is our evidence, and those stay ours.

## Decisions still needed from the instructor

**None. All decisions are settled** - the outline is approved to build prose against, pending the
review noted at the bottom of this file.

1. ~~**Chapter number and placement.**~~ **CONFIRMED 2026-08-08:** `ml/ch16_jepa/`, decks **L39**
   and **L40**, after ch15 VLA. It depends on ch12 (CLIP), ch11 (planning) and ch15 (compounding
   error, data scarcity), so it could not have moved earlier without losing its best hooks.

2. ~~**How much of the LeCun-versus-LLMs story?**~~ **ANSWERED via decision 4** - lean in, and lean
   on part 2 especially. The dispute becomes the deck's spine: open on "VLAs are doomed" over
   chapter 15's own robots, split the critique into its two real prongs, and keep the
   counter-evidence frame so it stays fair.

3. ~~**Energy-based models: stated or derived?**~~ **ANSWERED - stated, one picture.** See
   decision 6. L39 frame 16, figure 5 stays.

4. ~~**Retrieve the blocked tutorial first.**~~ **DONE, 2026-08-08.** Pulled through Playwright
   (Cloudflare Turnstile cleared in-browser, PDF fetched from inside the page session) and read in
   full. It paid off - it supplied the unified two-term loss, a cleaner anti-collapse taxonomy, the
   world-model lineage, and three real criticisms of I-JEPA that were not in the other sources.
   The frames it added are marked **[tutorial]** below. Caveat recorded in `sources.md`: the
   authors are a wireless-communications group and it is an unpublished preprint, so it is used for
   structure and framing, not for numbers about other people's models.

---

## The borrowed stills (full-bleed), in order

43 stills were pulled; these are the ones that earn a slide. Source file paths are relative to
`ml/ch16_jepa/_reference_welchlabs_lecun_p{1,2}/frames/`. **Architecture diagrams are marked ARCH**
- those are the priority set.

### L39

| # | Frame | What it shows | Slide |
|---|---|---|---|
| S1 | p1 `f01_00-01-05` | **ARCH** JEPA in its simplest form: x and y into two encoders, embeddings out | f3 or f10 |
| S2 | p1 `f02_00-04-25` | The "intelligence is a cake" graphic - RL cherry, supervised icing, self-supervised bulk | f2, optional |
| S3 | p1 `f06_00-12-05` | **The counting argument**: GPT-2's 50,257 discrete outputs beside video's 256^(1920x1080x3) ~ 10^(15,000,000), annotated against 10^80 atoms | **f8** |
| S4 | p1 `f08_00-13-12` | The bouncing ball: several real trajectories, then the averaged blurry frame | **f8-9** |
| S5 | p1 `f10_00-16-20` | **ARCH** Siamese network on signature pairs | f5-6 |
| S6 | p1 `f11_00-17-45` | **ARCH** Representation collapse - two encoders emitting the identical vector for different inputs | **f12** |
| S7 | p1 `f14_00-24-25` | **ARCH** Barlow Twins: the observed cross-correlation matrix beside the identity-matrix target | **f17** |
| S8 | p1 `f16_00-28-15` | Scoreboard: AlexNet 59.3 / Barlow Twins 73.2 / ViT 88.6 / **DINOv3 88.4**, with the paper's "first time a self-supervised model has reached comparable results" quote | **f36** |
| S9 | p1 `f17_00-28-40` | DINO patch-similarity segmenting a hand with no labels | f36 |
| S10 | p1 `f18_00-31-20` | **ARCH** The core JEPA: video and next frame into encoders, predictor between the embeddings | **f10** |
| S11 | p1 `f19_00-31-55` | The dashcam shot with its optical-flow field - the leaves argument, visualised | **f9** |

### L40

| # | Frame | What it shows | Slide |
|---|---|---|---|
| S12 | p2 `f01_00-01-25` | **ARCH** **The alternative stack** - V-JEPA 2 / VLM / VLA down one side, VL-JEPA / LeWorldModel down the other | **f4, the roadmap** |
| S13 | p2 `f04_00-05-20` | **ARCH** V-JEPA 2 masked-video training, embeddings and predictor over 3-D video blocks | f5 |
| S14 | p2 `f05_00-06-20` | **ARCH** **V-JEPA beside CLIP** - the same slide showing language-free vs caption-pinned | **f6** |
| S15 | p2 `f06_00-07-35` | TempCompass: the pineapple question, forwards and reversed | predict-first item |
| S16 | p2 `f07_00-09-25` | **ARCH** JEPA / JEPA-VLM / VLM mapped onto each other | VL-JEPA frames |
| S17 | p2 `f08_00-11-25` | **ARCH** The mushroom: a correct answer marked wrong for its phrasing | VL-JEPA frames |
| S18 | p2 `f09_00-12-05` | VL-JEPA vs VLM learning curves, matched encoder and data | VL-JEPA frames |
| S19 | p2 `f11_00-14-25` | **ARCH** VLA: "peel the zucchini" through vision encoder, LLM, action expert, control signals - with the VLM and VLA boundaries drawn | **ch15 callback** |
| S20 | p2 `f14_00-23-25` | **ARCH** LeWorldModel on PushT: current frame plus action into the predictor, next frame out | **f17-18** |
| S21 | p2 `f17_00-26-05` | **World model rollout beside the real environment**, four runs, with visible drift | **f24** |
| S22 | p2 `f18_00-27-35` | 500 random candidate trajectories | **f21** |
| S23 | p2 `f20_00-29-25` | The same trajectories colour-coded by embedding distance to goal, elite set emerging | **f21** |
| S24 | p2 `f19_00-28-45` | **ARCH** "Measure Euclidean distance" - predicted rollout embedding against goal embedding | **f22** |
| S25 | p2 `f21_00-32-35` | **ARCH** **Hierarchical planning**: high-level predictor (pink) proposing a subgoal to the low-level predictor (blue), start frame to goal frame. Credits Zhang et al., arXiv 2604.03208 | **hierarchy frames** |
| S26 | p2 `f22_00-33-35` | LeCun's abstraction ladder: NYU office -> go to the airport -> catch a plane -> Paris tomorrow | **hierarchy frames** |

### Promoted from reserve (decision 5)

All 26 above are in. These 8 were shortlisted but unassigned; with the pacing objection dropped
they earn slides too.

| # | Frame | What it shows | Slide |
|---|---|---|---|
| S27 | p1 `f03_00-08-10` | Video frames of the ball, before any model - sets up the prediction task | L39 f7 |
| S28 | p1 `f05_00-11-40` | GPT-2's discrete output list, 50,257 of them | L39 f8, before S3 |
| S29 | p1 `f09_00-15-45` | **ARCH** Siamese pair with embeddings, before the fraud framing | L39 f5 |
| S30 | p1 `f13_00-23-20` | Two neuron-activation traces with the Pearson coefficient | L39 f19, before S7 |
| S31 | p1 `f21_00-33-05` | **ARCH** Goal-state planning: predicted embedding against goal embedding | L39 f43 bridge, or L40 f28 |
| S32 | p2 `f12_00-18-25` | Teleoperation rig collecting demonstrations | L40, cloning-doesn't-scale frame |
| S33 | p2 `f13_00-19-05` | The RT-2 era robot - the counter-evidence frame | L40, counter-evidence frame |
| S34 | p2 `f15_00-24-30` | **ARCH** PushT decoded predictions: press up and the effector goes up | L40 f24 |

**34 stills total across ~92 frames** - roughly one in three. That is a guided screening by design,
per decision 5.

## Deck 1 - L39: "Predicting the wrong thing"

*The objective. Still images only: no time, no actions, no robots.*
**~41 frames.**

### Cold open

1. **Predict first.** Same model, same masking, same architecture, trained twice - once predicting
   pixels, once predicting representations. How big is the gap? *(`\pause`, then: 26.2 points, and
   the pixel run got 60% more epochs.)*
2. The question this chapter actually asks: not "which architecture" but **"what should the model
   be asked to predict?"**
3. The four answers this course has already used - input (ch8), distribution (ch10), next token
   (LLMs), does-this-match (ch12). The fifth is new.
4. Roadmap.

### Section 1 - Three ways to build a self-supervised task

*Transition slide: "You have already built two of them."*

5. **The joint-embedding architecture, via CLIP.** Straight callback to ch12: two towers, shared
   space, cosine similarity. No new material - just renaming what they know.
6. **What a JEA cannot do.** It scores whether `x` and `y` match. It has no way to say *what* `y`
   is. Left half of an image, right half: a JEA can rate them compatible and cannot describe the
   right half.
7. **The generative architecture.** Hide, reconstruct in input space. ch8's autoencoder, MAE, and
   next-token prediction are all this shape.
8. **Why that wastes capacity - the counting argument.** **[S3 full-bleed]** GPT-2 has **50,257**
   discrete next-token outputs. Full-HD video has 256^(1920x1080x3) possible next frames, about
   **10^(15 million)**, against roughly 10^80 atoms in the observable universe. *(Checked: 1920 x
   1080 x 3 x log10(256) = 1.5 x 10^7. The number is right.)* You cannot enumerate that, so the
   model must emit pixel values directly - and then uncertainty has nowhere to go.
9. **The bouncing ball.** **[S4 full-bleed]** An LLM completing "the ball bounced to the ___" has a
   separate output per token and can raise *both* left and right independently. A video model
   forced to emit one frame can only produce **the average of the outcomes**, which is a blur.
   Replaces the abstract 1-D figure I first planned - this is the same argument with a concrete
   object, and it is ch15's action-multimodality argument a third time. Say so.
10. **The leaves.** **[S11 full-bleed]** LeCun's own example, and the best line in either video: a
    generative model predicting a dashcam feed *"will spend most of its resources predicting the
    random motion of the leaves on the trees bordering the road - things that are essentially not
    predictable, but they have a lot of pixels."* The still pairs the road with its optical-flow
    field, which makes "a lot of pixels, no information" visible at a glance.
11. **JEPA.** **[S10 full-bleed, then S1]** Encode both sides, predict `s_y` from `s_x` through a
    predictor. Welch Labs' diagram is cleaner than anything I would draw and the instructor wants
    architecture stills, so borrow it - but **still build the house-style three-panel figure**
    (JEA / generative / JEPA side by side) as the thing the chapter refers back to. The still
    introduces; our figure is the reference.
12. **The sentence.** "JEPA gives the model permission to throw information away." Full stop, let
    it sit.

### Section 2 - The catch: collapse

*Transition: "and the rest of the design exists to stop it throwing everything away."*

13. **Write down the degenerate solution.** **[S6 full-bleed]** Both encoders output the constant
    `c`, predictor outputs `c`. Loss is exactly zero. Representation is worthless. The Welch Labs
    still shows two encoders emitting the identical vector for visibly different inputs, which is
    the whole failure in one picture.
14. **This is not mode collapse.** Explicit disambiguation against ch8b - same word, different
    failure. Mode collapse: a generator covering one mode. Representation collapse: an encoder
    mapping everything to one point.
15. **The loss does not tell you it broke.** ***Experiment deferred (decision 3) - build this frame
    without a figure for now.*** Make the point in words and arithmetic, which is enough: in the
    degenerate solution of frame 13 the loss is **exactly zero**, the best value it can take, and
    the representation is worthless. So a falling loss curve is **not evidence of learning here**,
    and there is no threshold that means "trained". State the practical consequence, which is the
    part students will actually meet: **you cannot early-stop on this loss, and model selection
    needs a downstream probe** - which is expensive, which is why fewer configurations get tried
    than in a supervised setting. Forward-reference LeJEPA at frame 42, whose most useful claim is
    that its loss finally *does* correlate with downstream accuracy.

    *When the experiment is un-deferred it slots in here as a second frame (15b) and nothing
    downstream renumbers. Spec is parked in `DEFERRED_TODO.md`.*
16. **The energy picture.** `F(x,y)` low on compatible pairs. The failure is not low energy on the
    data, it is low energy *everywhere*. *Figure: sculpted vs flat landscape.*
17. **One loss, three choices.** **[tutorial]** Put up the two-term objective -
    `L = E[d(pred, target)] + lambda * R(...)` - and make the point that **every** JEPA has this
    shape and the families differ *only in what `R` is*. Far better than three unrelated tricks.
18. **The three families**, as three answers to "what is `R`?": contrastive (a second data term
    with negatives) / teacher-student (absent from the loss; collapse blocked structurally) /
    moment-matching (an explicit penalty on batch statistics). Table with the cost of each.
19. **Where the third family came from.** **[S7 full-bleed]** Worth telling as history, because it
    is LeCun's own account of the turn: the joint-embedding tricks were *"kind of hacks"* until
    **Stephane Deny** brought in **Horace Barlow's 1961** hypothesis that visual neurons reduce
    redundancy between each other. Barlow Twins applies it directly - take the batch of activations
    from each encoder, compute the **cross-correlation matrix** between them, and drive it toward
    the **identity**: diagonal to 1 (the two views agree), off-diagonal to 0 (the dimensions stop
    duplicating each other). The still shows the observed matrix beside the identity target. VICReg
    is the simplified successor, SIGReg the 2025 one.
20. **Which one I-JEPA uses.** EMA teacher, momentum 0.996 rising linearly to 1.0 `[read]`.
21. **Say it plainly: this is a heuristic, not a theorem.** No general proof that EMA prevents
    collapse - and **C-JEPA's authors argue it does not fully work**, adding VICReg's three terms on
    the same backbone to fix it. **[tutorial]** Flag forward to the end of the deck.

### Section 3 - I-JEPA, in detail

*Transition: "The first one that worked."*

22. **The three networks.** Context encoder (trained), target encoder (EMA, not trained by
    gradients), predictor (a deliberately *narrow* ViT: width fixed at 384 regardless of backbone,
    depth 6/12/16) `[read]`.
23. **What you keep is the target encoder. The predictor is thrown away.** Say this clearly - L40
    reverses it, and the reversal is one of the best slides in the chapter.
24. **Multi-block masking.** 4 target blocks at scale (0.15, 0.2), aspect (0.75, 1.5); one context
    block at (0.85, 1.0), unit aspect, minus any overlap; ~25% of patches survive `[read]`.
    *Figure: drawn on `ml/ch12_vlm/fig/img/yerevan_market.jpg` - reusing the ch12 photo on purpose.*
25. **The subtle bit: targets are masked at the *output* of the target encoder.** The target
    encoder sees the whole image. That is what makes the targets semantic rather than local
    texture - and it is an asymmetry worth admitting.
26. **The full forward pass, worked once,** end to end with shapes.
27. **Ablation 1 - masking strategy.** *Bar chart with labels on bars:* multi-block **54.2**, block
    20.2, random 17.6, rasterized **15.5** `[read, Table 6]`.
28. **Read the rasterized row.** Same 25% context budget as the winner. **39 points worse.** It is
    not how *much* context, it is *where*.
29. **Ablation 2 - the target space.** *Bar chart:* 66.9 vs 40.7 `[read, Table 7]`. Payoff of frame 1.
30. **What the predictor actually represents.** Decoded predictions keep pose and object part
    consistent while background and fine detail vary - the model is representing what is
    predictable and staying silent about the rest. Flag honestly that the decoder is not part of
    I-JEPA and these are hand-picked figures.
31. **What I-JEPA still gets wrong: it assumes it knows *where*.** **[tutorial]** Fixed positional
    embeddings mean the predictor is told the exact location of the patch it must predict. StoP-JEPA's
    example: *given only part of a dog, you cannot locate its tail precisely.* Their fix models each
    masked position as a Gaussian random variable with learned covariance - a few lines of code, no
    extra compute, better probing. **This is frame 8's argument again, applied to position instead of
    content** - point that out explicitly.
32. Section recap.

### Section 4 - Does it work? Reading a scoreboard honestly

*Transition: "This is the part most summaries get wrong."*

33. **Table 1, on screen** `[read]`: I-JEPA 79.3 (ViT-H/14, 300 ep) against MAE 77.2 (1600 ep),
    data2vec 77.3 (1600 ep), DINO 80.1, iBOT 81.0.
34. **The honest reading.** I-JEPA beats everything in its own category by a clear margin at a
    quarter to a half the epochs. It **does not** beat DINO or iBOT at 224px. The paper's own words
    are "decreases the gap". *Do not let this slide overclaim.*
35. **What it does win: efficiency.** ViT-H/14 on 16 A100s in under 72 hours; cheaper than a
    ViT-S/16 trained with iBOT `[read]`.
36. **Why.** Augmentation-based methods process 10+ crops per image per step. I-JEPA processes one
    view, and the context encoder only sees 25% of it. *Figure: accuracy vs pretraining epochs.*
37. **A slide about sources.** A widely-circulated summary reports I-JEPA 72.4 vs MAE 59.8 on 1%
    ImageNet. The paper says 73.3 vs 71.5, and MAE needed 1600 epochs against I-JEPA's 300. The
    gap was inflated sixfold - **and the true story is the better one.** Generalise the lesson.
38. **Where the image line actually went.** The default general-purpose image encoder in 2026 is
    DINOv2/v3 - self-distillation with hand-crafted augmentations, the family I-JEPA was positioned
    against. Meta ships both. Being right about the objective did not automatically win the
    benchmark.
39. Takeaway box.

### Close

40. **The objective generalises.** One frame, the modality catalogue: audio, EEG, ECG, point
    clouds, molecules, tabular, remote sensing, genomics. Note the pattern - these are all domains
    with **few labels and lots of unlabelled data**, which is exactly where this pays.
41. **What it structurally cannot do.** **[tutorial]** JEPA is not designed for pixel-, waveform-
    or token-level reconstruction - it *deliberately discards* what a decoder would need. So it does
    not compete with ch10's diffusion models; they are not applying for the same job. **The reason
    it is good at understanding is the reason it cannot generate.** Pre-empts the obvious question
    and sharpens the whole chapter.
42. **What is unresolved.** SALT (Sep 2025) reports a *frozen* teacher beating EMA at matched
    FLOPs. LeJEPA (Nov 2025) removes the teacher, the stop-gradient and the predictor entirely and
    replaces them with one regulariser with a proof behind it - and, more useful in practice, a
    loss that finally correlates with downstream accuracy.
43. **Bridge.** Everything so far happened inside one still photograph. Next: add time, then add
    actions, and the same objective turns into something you can plan with.

---
## Deck 2 - L40: "Working up the alternative stack"

*Video, actions, planning, and an open bet.*
**~49 frames** (44 below plus the five part-2 additions in the next block).

### Part-2 additions (decision 4), folded into the skeleton

The outline below was written before the instruction to lean harder on part 2. These five frames
are additions to it; **the built skeleton `L40_jepa_world_models.tex` is now the authoritative
frame list** and already contains them in position.

| New frame | Where | Content |
|---|---|---|
| **Visual bandwidth** | cold open, after f3 | LeCun's back-of-the-envelope: an average **4-year-old has taken in more information through the visual cortex** than the largest LLM sees in *all* of its training text. This is the single best motivation for "intelligence does not start in language" and I had left it out. Pairs with the data-asymmetry figure |
| **A learned video game** | section 3, after f24 | Welch Labs' framing of what the world model *is*: press a key, the decoded world changes correctly - *"a learned simulated version of the world"*. **[S34 full-bleed]** Makes latent prediction concrete before the planning machinery arrives |
| **Cloning does not scale** | section 5, splitting old f41 | LeCun's first prong, with the teleoperation rig **[S32]**: you cannot collect demonstrations for every variation, and outside them the policy is *"completely helpless... brittle"* |
| **No explicit planning** | section 5 | The second prong: a VLA is trained and deployed end to end, so we have limited visibility into the planning process - a black box from instruction to action |
| **The counter-evidence** | section 5 | **[S33]** Kept deliberately, from the video's own honesty: RT-2 generalised to a concept absent from its demonstrations, and pi models do tasks outside their demo data. **Generalisation is a sliding scale, not a yes/no.** Without this frame the critique is unfair |

> **Restructured 2026-08-08 after the Welch Labs series.** The old shape was *image to video ->
> what is a world model -> actions -> limits -> the bet*. Part 2 of the series has a much better
> organising idea and I have adopted it: **every layer of the mainstream stack has a JEPA
> counterpart.** A VLA is built on a VLM, which is built on a vision encoder; JEPA offers a
> replacement at each level - V-JEPA 2 for the encoder, VL-JEPA for the VLM, LeWorldModel for the
> VLA. That turns the deck from a survey into a climb with a destination, it makes ch12 and ch15
> load-bearing prerequisites instead of passing callbacks, and it gives the closing argument
> somewhere to stand. The old section list is recoverable from git if this is the wrong call.

### Cold open

1. **The provocation.** **[S19 full-bleed]** Open on chapter 15's world - a Physical Intelligence
   robot doing real manipulation - then LeCun, flatly: **"VLAs are doomed."** Beside it, JEPA
   taking **60 seconds** to move a cup off a platform. Let the room sit with the mismatch between
   the confidence and the demo.
2. **Predict first.** A robot picks and places novel objects in a lab whose data the model never
   saw, no task-specific training, no reward. **How many hours of robot data?** *(`\pause`: under
   62. Alongside 1,000,000 hours of internet video.)*
3. *Figure (ours): the data asymmetry, log scale, labelled.*
4. **The map for today.** **[S12 full-bleed]** The alternative stack. A VLA sits on a VLM, which
   sits on a vision encoder. **JEPA has a candidate for all three.** We climb it: encoder, then
   VLM, then robot. Welch Labs' single diagram is the whole roadmap.

### Section 1 - Level 1: the vision encoder

*Transition: "Start at the bottom. Replace CLIP."*

5. **V-JEPA: mask in space and time.** **[S13 full-bleed]** Same recipe as L39, now over video
   blocks, so the model must represent motion rather than layout.
6. **Why Something-Something-v2 is the honest benchmark.** Its classes are defined by *motion*
   ("pushing something from left to right"), so appearance alone cannot solve it.
7. **Frozen evaluation.** No fine-tuning, just a probe on fixed features - much harder and more
   honest than the fine-tuned numbers video papers usually report.
8. **V-JEPA 2 (Jun 2025):** 1.2B parameters, >1M hours of video plus 1M images `[read]`.
9. **The contrast that matters.** **[S14 full-bleed]** V-JEPA beside CLIP on one slide. CLIP's
   representation is **pinned to caption embeddings**; V-JEPA is **blissfully unaware of language**
   and may represent "cat" however it likes, so long as that helps fill in missing video. AMI Labs
   states the philosophy outright: *real intelligence does not start in language, it starts in the
   world.*
10. **So does a language-free encoder even interface with a language model?** Predict-first. The
    answer is yes, and the V-JEPA 2 authors flag it as **"contrary to conventional wisdom"**:
    77.3 SSv2, state of the art on Epic-Kitchens-100 anticipation at 39.7 recall@5, 84.0
    PerceptionTest and 76.9 TempCompass once aligned to an 8B LLM `[read]`.
11. **What those benchmarks actually ask.** **[S15 full-bleed]** The TempCompass pineapple
    question - then the *same clip played backwards*, which changes the correct answer. Welch Labs
    report ChatGPT 5.5 getting both wrong. Good predict-first item; verify the model claim before
    quoting it, it is theirs not ours.

### Section 2 - Level 2: the vision-language model

*Transition: "Now replace the whole VLM."*

12. **The one-line change.** **[S16 full-bleed]** A standard VLM emits text one token at a time.
    **VL-JEPA predicts the embedding of the target text instead** - in one shot. Everything else
    (vision encoder in, prompt conditioning) maps across unchanged.
13. **Why that should help - the mushroom.** **[S17 full-bleed]** Asked whether a mushroom is safe,
    the training target might be *"do not eat this mushroom"*. A generative VLM answering *"this
    mushroom is not safe to eat"* is **penalised for a correct answer**. In embedding space the two
    land in nearly the same place. **This is the leaves argument, moved from pixels to language** -
    third appearance of the same idea; make the room notice.
14. **The controlled result.** **[S18 full-bleed]** Same vision encoder, same data, same config:
    VL-JEPA reaches **35%** video classification after 5M examples where the standard VLM is at
    **20%**. Efficiency, not just accuracy.
15. **And it punches above its size.** ~1.6B parameters beating 7B-13B models on GQA compositional
    reasoning; 65.7% on WorldPrediction-WM against GPT-4o's 53.3 and Gemini-2.0's 55.6 `[read]`.
16. **The wrinkle - do not skip it.** VL-JEPA **is not generative**, so those benchmark numbers
    come from a *multiple-choice protocol*: encode every candidate answer, pick the nearest to the
    predicted embedding. They also trained a separate text decoder to make it speak. **The caveat
    travels with the number, always.** This is L39's "it structurally cannot reconstruct", arriving
    as a practical bill.

### Section 3 - Level 3: the robot, and what a world model is

*Transition: "Top of the stack. This is where the disagreement is real."*

17. **Two definitions of "world model" in circulation.** **A - generative**: produce the
    *observation* you would see next (Sora, Genie, Cosmos); you can watch its predictions.
    **B - latent predictive**: produce the next state in the model's own representation; you cannot
    watch it, and it is judged by whether you can plan with it.
18. **LeCun's argument for B**, and the fair counter: a generative world model is **inspectable** -
    you can see the spoon pass through the bowl. A latent one that learned something wrong gives
    you no handle.
19. **This is not new, and LeCun says so.** ch11 has the ancestors (Ha and Schmidhuber, Dreamer,
    TD-MPC). Quote him directly: this is **classical optimal control** from the late 1950s - *what
    is not classical is that you learn the model, and less classical still that you learn a
    representation and put the model in that space.* Intellectually honest, and it lands better
    than a claim of novelty.
20. **The instructive difference.** In model-based RL the latent cannot collapse because it must
    also predict **reward** and **value** - grounded external signals. **JEPA has no reward.**
21. So all of L39's anti-collapse machinery is **the price of dropping reward**. One sentence that
    retroactively explains deck 1.
22. **Action conditioning.** **[S20 full-bleed]** Stage 1: 1M+ hours of video, no actions.
    Stage 2: freeze it, learn `s_{t+1} = P(s_t, a_t)` from under 62 hours of unlabelled DROID
    video `[read]`.
23. **The reversal.** In I-JEPA you keep the encoder and discard the predictor. Here **the predictor
    is the entire product**. Same architecture, opposite half is the deliverable. *Callback to L39
    frame 23.*
24. **It learns physics from pixels alone.** LeWorldModel on PushT: press up, the decoded prediction
    moves up; left, left. Nobody told it the T is rigid. *(A decoder is trained separately just to
    let us look - it is not part of the model.)*
25. **And it drifts.** **[S21 full-bleed]** Model rollout beside the real environment over ~18
    steps: good agreement, then visible divergence. Welch Labs' own words - it *"goes off the rails
    sometimes"*. **ch15's compounding error, in a third setting**, and the reason the horizon must
    stay short.

### Section 4 - Planning as search in embedding space

*Transition: "Now use it."*

26. **The loop, written out.** Sample K action sequences, roll each forward in latent space, score
    by distance to the goal embedding, keep the best, resample, execute **one** action, replan.
27. **CEM, in three stills.** **[S22, S23 full-bleed]** 500 random candidate trajectories; then the
    same trajectories colour-coded by embedding distance to the goal, with an elite set emerging;
    resample from the elite mean and repeat. The clearest visual explanation of the cross-entropy
    method I have seen anywhere.
28. **What the score actually is.** **[S24 full-bleed]** Euclidean distance between the predicted
    terminal embedding and the goal embedding. **No reward was specified and no reward model was
    trained. You say what you want by showing a picture.**
29. **Nothing is ever rendered.** The whole rollout happens in the representation - which is what
    makes it fast enough to consider.
30. **Execute one step, then replan.** MPC, and the *other* answer to compounding error: the horizon
    keeps resetting. Contrast explicitly with ch15's action chunking - same disease, different cure.
31. **The result.** Zero-shot on Franka arms in two labs Meta had no data from; 65-80% on
    pick-and-place with novel objects `[read]`.
32. **The honest numbers.** ~16 seconds of planning per action for V-JEPA 2-AC *(single-source -
    verify before building)*; LeWorldModel reliably plans about **5 steps** ahead on PushT. Not a
    controller. A demonstration that the representation contains enough physics to plan in.
33. **The limitation hiding inside "say what you want with a picture".** **[tutorial]** Planning is
    **goal-conditioned** - it assumes a target representation exists. "Put the cup on the shelf",
    fine, photograph it. **"Tidy the kitchen", "make this safe", "explore" - there is no image to
    aim at.** Those need reward design, goal proposal or subgoal generation: exactly what this was
    advertised as avoiding.
34. **The proposed fix: hierarchy.** **[S25 full-bleed]** A high-level predictor proposes a
    **subgoal**; a low-level predictor plans to it. Two levels take PushT from **5 steps to 15**
    (Zhang et al., arXiv 2604.03208). Note this is also the answer to frame 33.
35. **Why hierarchy is the right shape.** **[S26 full-bleed]** LeCun's ladder: sitting in his NYU
    office, he cannot plan a trip to Paris in millisecond muscle control - he plans *go to the
    airport, catch a plane*, and only expands the bottom rungs when he gets there. Detail buys
    accuracy at short range and destroys it at long range.
36. **The open part.** The hope is the hierarchy is **learned, not designed** - the way CNN feature
    hierarchies emerged. But LeCun concedes it needs semi-expert trajectories: you cannot learn
    high-level structure from random flailing. Largely still a proposal; shipped systems are
    single-level.

### Section 5 - The honest scoreboard, and the bet

*Transition: "Meta shipped a benchmark its own flagship model fails."*

37. **Three benchmarks released with V-JEPA 2.** IntPhys 2 (is this physically possible?),
    MVPBench, CausalVQA (what *could* happen). **Note: absent from both videos** - this section is
    our counterweight to a sympathetic source.
38. **How MVPBench is built.** Minimally-different video pairs with different correct answers, so a
    shortcut solution scores at chance *by construction*. A general benchmark-design technique.
39. *Figure (ours): human vs model.* Humans near ceiling, models at or near **chance** `[read]`.
    A model that is state of the art at anticipating actions cannot tell whether a scene is
    physically possible.
40. **A second admission.** V-JEPA 2's dense features were "noisy" and "fragmented" - ADE20K mIoU
    **24.4**, NYUv2 depth RMSE 0.642. V-JEPA 2.1 (Mar 2026): **47.8** and 0.350 `[read]`. Note SSv2
    went 77.3 to 76.9 while segmentation doubled - **"the representation is good" is not one
    number.**
41. **The fair verdict on VLA.** LeCun's critique is two-pronged - behavioural cloning does not
    scale, and there is no explicit planning. But the counterweight is real and the video supplies
    it itself: RT-2 generalised to a concept absent from its demonstrations, and pi models do tasks
    outside their demo data. Generalisation is a sliding scale, not a yes/no.
42. **Score the whole thesis.** Four rows: good efficient encoders (**supported**); a video model
    you can plan with, slowly (**supported, with caveats**); intuitive physics (**not supported** -
    IntPhys 2); replaces next-token prediction for language (**not demonstrated** - VL-JEPA is
    1.6B beating GPT-4o on one benchmark, multiple-choice).
43. **AMI Labs, and what LeCun actually plans to do.** $1.03B at $3.5B pre-money, Paris, no revenue
    plans. More concrete than the press coverage: the first targets are **complex systems you
    cannot reduce to equations** - a jet engine, a chemical plant, a diabetes patient's blood
    sugar - explicitly *not* robot arms, because *"if you can write down the equations, just write
    them down."* Meanwhile Meta shipped V-JEPA 2.1 without him.
44. **Close.** The objective is a genuine contribution and is already standard across a dozen
    modalities. The world-model thesis is an open bet with one measured failure and several
    promising results, being tested right now with real money. *Anyone who leaves this room certain
    either way was taught badly.*

## Figures to build (`ml/ch16_jepa/py_src/` to `ml/ch16_jepa/fig/`)

**The stills changed this list.** The rule from the instructor decision: **borrowed stills carry
architecture and narrative; Python carries every number.** Three figures I had planned are now
better served by a Welch Labs still and are cut; three are unaffected because they plot data
nobody else has drawn; and the ones that survive alongside a still do so for a stated reason.

### Still to build in Python - priority order

| # | Figure | Deck | Notes |
|---|---|---|---|
| 1 | **Three architectures side by side** (JEA / generative / JEPA) | L39 f11 | **Kept despite S10.** The still introduces JEPA; this is the reference diagram the deck points back to five times, and it must be in our visual language |
| 2 | **Two ablation bar charts** (54.2 / 20.2 / 17.6 / 15.5, and 66.9 / 40.7) | L39 f27, f29 | Our evidence, our chart. Labels on bars |
| 3 | I-JEPA masking drawn on a real photo | L39 f24 | Reuses `ml/ch12_vlm/fig/img/yerevan_market.jpg`. No still covers this - Welch Labs never shows I-JEPA's actual masking |
| 4 | **Human vs model on the three physics benchmarks** | L40 f39 | **Highest-priority L40 figure.** Keeps the chapter honest, and it is the one thing in the chapter that **neither video contains** |
| 5 | Energy landscape: sculpted vs flat | L39 f16 | Now carries more weight - it is the only visual in the collapse section besides S6 |
| 6 | Accuracy vs pretraining epochs, I-JEPA / MAE / data2vec / iBOT | L39 f34 | Honest axes - do not crop out iBOT ending higher. Different data from S8, which is the DINOv3 scoreboard |
| 7 | Data asymmetry, 1M hours vs 62 hours, log scale | L40 f3 | |
| 8 | Encoder-to-world-model lineage: I-JEPA to IWM to Seq-JEPA/PLDM to V-JEPA 2-AC/LeWM | L40 | **[tutorial]** Styled table rather than a plot. Optional now that S12 carries the stack framing |
| 9 | Timeline, 2022 position paper to V-JEPA 2.1 and AMI Labs | either | Lowest priority |

**All nine are plotting or drawing published numbers - no model is trained, nothing is measured.**
That satisfies decision 3 in full: the chapter as specified runs no experiments.

Armenian flag colours (`#D90012`, `#0033A0`, `#F2A800`) wherever 3+ series appear, per house style.

### Cut - a borrowed still does it better

| Was | Now | Why |
|---|---|---|
| Plausible continuations + their L2-optimal mean | **S4** (the bouncing ball) | A concrete bouncing ball beats an abstract 1-D signal, and it is LeCun's own framing |
| Planning in latent space: fan-out, score, winner | **S22 + S23 + S24** | Welch Labs animate the actual cross-entropy method on a real task across three frames. I would have drawn a worse static version of exactly this |
| *(new, never planned)* | **S12** | The alternative-stack diagram reorganised all of L40. Nothing I had planned did that job |

**Net effect: 12 planned figures becomes 10, of which 8 are genuinely load-bearing.** That is
several hours of figure work saved, and the chapter gets better diagrams for the parts that are
somebody else's explanation anyway.

---

## ~~Fallback: one deck, ~40 frames~~ - DEAD (two decks confirmed 2026-08-08)

Kept only so the reasoning survives if the decision is ever reopened. Frame numbers refer to the
outlines above as numbered now.

Keep **L39** 1-4, 7-12 (the argument for the objective, with S3/S4/S11), 13-16 (collapse, trimmed),
17-18 (the two-term loss and the families), 22-29 (mechanics and both ablations), 33-34 (the honest
scoreboard); then **L40** 1-4 (cold open and the stack), 9 (V-JEPA vs CLIP), 12-16 (VL-JEPA,
including the wrinkle), 17-21 (the two definitions and the reward point), 22-32 (actions and
planning), 37-39 (IntPhys 2), 42-44 (the bet).

**What gets cut:** the CLIP/JEA framing (assume ch12 covers it), the Barlow Twins origin story,
the efficiency argument, the secondary-source slide, the DINOv2/v3 reality check, StoP-JEPA, the
whole hierarchy section, V-JEPA 2.1's dense-feature admission, and the modality catalogue.

That is a worse loss than it was before the videos. The hierarchy section (L40 f34-36) is now the
answer to the goal-conditioning limitation, so cutting it leaves the deck's sharpest criticism
hanging unanswered. **Recommendation for two decks is stronger than it was.**

---

## Definition of done

Per `WORKFLOWS.md`: 2x pdflatex passes, zero `!` lines in the log,
`non_essential/detect_clipped_slides.py` run with every flag checked against the rendered page,
acronym check run (JEPA, JEA, EMA, MAE, ViT, SSL, MPC, CEM, SSv2, VQA, RMSE, mIoU, SIGReg all
need spelling out at or before first use), aux files cleaned, `% Provenance:` block present, and
the chapter registered in `_quarto.yml` after `ml/ch15_vla/vla.qmd`.

Additionally, for the borrowed stills:

- Every `\wlslide{}` frame carries the `Welch Labs (2026)` attribution node. **Check the rendered
  page, not the source** - the node sits over the image and can land on a dark region and vanish.
- Frames copied into `fig/borrowed/welchlabs/` with descriptive names, not `fNN_HH-MM-SS.jpg`.
- The `% Provenance:` block names the two source videos with their URLs and fetch date.
- `_reference_*/video.mp4` and `*.vtt` stay git-ignored (~355 MB across the two). The transcripts,
  READMEs and 43 stills (~6 MB) are committed - they are the re-derivable record.
- Compile twice: the full-bleed macro uses `remember picture`, which needs a second pass.
