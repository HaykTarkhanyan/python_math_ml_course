# Lecture 11: Frontier & Open Problems - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 11.05.2026
- Source: `slides.pdf` in this folder (47 frames, 1280x720, no text layer), re-extracted on 2026-10-04 from the 720p YouTube recording with the slide repo's own `lectures/extract_slides.py` (github.com/idanL1212000/RobotLerning-2026-ETH-Zurich). That repo's `week11_slides.pdf` is only 640x360, so its frame numbering differs from this one.
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/eL4lcy1KNzE (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_lucas_beyer.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 12: Frontier & Open Problems

Oier Mees - ETH zurich, Microsoft - 11.05.2026

[Note: the slide says "Lecture 12", but the course page and the YouTube title list this as Lecture 11 (week 11, May 11).]

## Slide 1: Every Week on Twitter: Robotics is Solved! (frame 2)

[Figure: screenshots of three social-media posts, summarised: a venture investor quoting a robotics startup's launch of a new "robotic brain" model (with a photo of a two-armed humanoid cooking at a stove) as "the endgame for robotics"; a humanoid-company founder announcing an imminent "ChatGPT moment" for robotics (Jan 7, 2024, 2.4M views); and an NVIDIA researcher promoting a talk on a "roadmap for solving Physical AGI" as a parallel to the LLM success story.]

## Slide 2: How Autonomous Robots (Usually) Generalize (frame 3)

[Figure: two video stills: a humanoid robot fallen on the floor next to a shopping cart in a cafeteria, captioned "Boston Dynamics"; a two-armed mobile robot in a kitchen, captioned "Fu et. al., Stanford 2024".]

[Note: slide 3 does not appear in the captured frames; the numbering jumps from 2 to 4.]

## Slide 4: [untitled: scales of generalization] (frame 4)

[Figure: nested ellipses, from smallest to largest: **1 object** inside **1 room** inside **1 building** inside **the world**.]

## Slide 5: The Quest for the Best of Both Worlds (frame 5)

| Non-Embodied Foundation Models | Specialist Embodied Models |
|---|---|
| [Figure: a **Transformer** over rows of Images and Language tokens] | [Figure: robot arms preparing food at a table (4x)] |
| ✅ Internet-scale Data | ✅ Physical Grounding |
| ✅ Broad Generalization | ✅ Embodied Sensing |
| ❌ Lacks Physical Grounding | ❌ Narrow, Task-specific Data |
| ❌ Non-Embodied Sensing | ❌ Poor Generalization |

## Slide 6: The Path to Embodied Intelligence (frames 6-8)

- Today's foundation models lack physical grounding
- Embodied data inherently **multimodal, spatial & temporal**

[Figure: built up over three frames. **Embodied Data** (Multimodal, Temporal, Spatial, Physical) and **Foundation Models** are connected by arrows in both directions (a loop); Foundation Models -> **AI Agents for the Digital & Physical World**.]

## Slide 7: What is the Best Backbone for Robotics? (frame 9)

[Three options in boxes:] Vision-Language Model | Generative Video Model | Doesn't matter if we have enough data to train from **scratch**?

## Slide 8: What is the Best Data Recipe? (frame 10)

[Figure: two data pyramids. Left: **Web Data** (Wikipedia, Common Crawl) at the base, **Simulation Data** (a simulated robot fleet) in the middle, **Real-World Data** (a person teleoperating a robot arm in VR) on top. Right: Web Data at the base and Real-World Data on top, with no simulation layer.]

## Slide 9: What is the Best Data Recipe? [video and robot-only pyramids] (frame 11)

[Figure: two more pyramids. Left: **Internet Videos** (YouTube) at the base, **Human Videos** (a collage of egocentric clips) in the middle, **Real-World Data** (VR teleoperation) on top. Right: every layer is labelled **Real-World Data**, each a collage of robot clips, the base the largest.]

## Slide 10: Data Collection Interfaces (frame 12)

[Figure: five photos:] UMI-Style (a hand-held gripper used at a sink) - Gloves for Dexterous Hands (a robot hand next to a person wearing a sensor glove) - Static Bimanual Puppeteering (a person between two leader arms on a frame) - VR (a person in a VR headset teleoperating a robot arm) - Mobile Bimanual Puppeteering (a person pushing a mobile two-arm rig to clean a restroom, 10x speed)

## Slide 11: Data Scalability vs Hardware Alignment (frame 13)

[Figure: plot of Data Scalability (log scale) against Hardware Alignment. Three points on a dashed line falling to the right: **Egocentric Videos** at $10^7$ hr, **Data Wearables** at $10^5$ hr, **Teleop** at $10^3$ hr. A vertical dashed line splits the x-axis into "Sensorized Human Data" (left) and "Robot in the loop" (right); Data Wearables sit almost on the boundary.]

## Slide 12: Towards Dexterous Generalist Policies (frame 14)

Dexterity and generalization require different ingredients

[Figure: Venn diagram. **Dexterity**: High-Frequency Control; Heterogeneous Multimodal Sensing. **Generalization**: Foundation Models; Mobility. Overlap: **Goal**.]

## Slide 13: Dexterity Requires More Than Vision (frame 15)

[Figure: video still of the PR2 robot at the Albert-Ludwigs-Universität Freiburg "TidyUpRobot" setup, grasping a blue cup near a cardboard box, with the overlay "Failure modes".]

Reference: *Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics*, **Mees,** Tatarchenko et al., IROS, 2019.

## Slide 14: Recap: Native Multimodal Models (frame 16)

- Instead of adding vision to an existing LLM, train all modalities jointly from scratch

[Figure: same diagram as week 7, slide 25: image, audio, video and text inputs (each optional) go through their encoders into an interleaved token sequence (any order, any mix), then a **unified multimodal transformer** "trained from scratch - every weight has always seen every modality", giving text (+ image) output.]

## Slide 15: The Long Tail of Robot Sensing (frame 17)

- **Scarcity**: many relevant sensing modalities (e.g. touch) not available at scale
- **Pairing**: cross-modal paired data (e.g. RGB + depth+ tactile + force simultaneously) is nearly nonexistent
- **Heterogeneity**: sensors vary across robot embodiments, making transfer hard even when data exists

## Slide 16: Reason Across New Modalities? (frame 18)

[Figure: overlapping circles: **Language** in the centre, overlapping **Vision** (top), **Audio** (bottom left) and **Touch** (bottom right).]

## Slide 17: Giving VLAs Senses They Were Never Trained On (frame 19)

[Figure: **Vision** (camera images), **Touch** (a tactile sensor image) and **Audio** (a spectrogram) feed into **FuSe Finetuning** ("Generative+contrastive losses ground multimodal inputs in language"), which starts from a **Pre-trained Generalist Policy** (Octo or a VLA). **Multi-Modal Language** instructions also feed in, e.g. "Push the button that plays piano", "Pick the object that feels squishy".]

Reference: *Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding*, Jones\*, **Mees\***, Sferrazza\*, Stachowicz, Abbeel, Levine. ICRA, 2025.

[Note: slide 18 does not appear in the captured frames; the numbering jumps from 17 to 19.]

## Slide 19: When & How to Reason Intelligently? (frame 20)

[Figure: a "policy confidence" bar running from certain through uncertain to unknown, with three columns:]

| ✓ no reasoning needed (certain) | adaptive TTC (uncertain) | escalate / stop (unknown) |
|---|---|---|
| **In-distribution task** - Known objects, scene, goal | **Near-distribution task** - Novelty or low confidence | **Out-of-distribution task** - Novel goal or failure mode |
| -> **Reactive policy** - No reasoning tokens (~50 Hz control loop) | -> **Uncertainty detection** -> **Scale test-time compute** - ↓ frequency, ↑ reasoning | -> **Escalate / ask for help** - Pause and defer |

Below a dashed line, "test-time compute can take many forms": **More thinking tokens** (CoT, GRPO, RL reasoning); **Larger / specialist model** (Foundation model fallback); **Human in the loop** (Teleop, correction, label).

## Slide 20: [untitled: introspection] (frame 21)

**How does a model know what it doesn't know?**

Generalization across many axes: objects, environments, embodiments, instructions, tasks...

**Open Problem: Introspection**

## Slide 21: Current Robot Models Rely on Imitation Learning (frame 22)

Policy bounded by the demonstrations in the dataset

[Figure: "Dataset": states 1, 2, 3; the demonstrated path goes 1 -> 2 -> 3.]

## Slide 22: Can we Scale RL for Robotics? (frame 23)

Imitation learning:

- Imitates seen behaviors in the data

Reinforcement learning (RL):

- Learns near optimal policies from suboptimal data
- Stitches suboptimal trajectories

[Figure: "Dataset" with the path 1 -> 2 -> 3; "Offline-RL policy" going straight 1 -> 3.]

## Slide 23: Robot Learning Data Flywheel (frame 24)

[Figure: a cycle of four circles:] **Increased Deployments** (improved reliability) -> **More Training Data** (deployment data, suboptimal demos, human corrections) -> **Better Learning** (BC, RL, fine-tuning) -> **More Capable Robots** (multiple embodiments) -> back to Increased Deployments

## Slide 24: Co-Training Expert & Autonomous Data (frame 25)

| Expert trajectories - Human demos, teleoperation | | Autonomous rollouts - Self-generated experience |
|---|---|---|
| ✓ High quality | | ✓ Cheap and scalable |
| ✓ On-manifold states | -> **Distribution mismatch** - state spaces don't overlap <- | ✓ Covers failure modes |
| ✓ Short, smooth, dense | -> **Temporal mismatch** - frequency, length, smoothness <- | ✗ Noisy, suboptimal |
| ✗ Expensive, limited scale | | ✗ Off-manifold states |
| ✗ Bounded by human skill | | ✗ Longer, jerky, variable |

Both mismatches lead to: **New algorithms required** - weighting, alignment, joint learning

## Slide 25: Lifelong Learning (frame 26)

[Figure: performance against time. During pre-training the curve is flat; at "deploy" it splits into **lifelong learning** (green, rising toward high) and **reality today** (dashed, slowly declining toward low); the vertical distance between them is labelled "gap".]

open problem: how do we get there?

## Slide 26: Rapid Adaptation: In-Context Learning (frame 27)

[Figure: **Policy failure** (novel task, environment...) branches into three remedies: **Corrective instructions** (language feedback), **Robot demo in context** (same embodiment), **Cross-embodiment demo in context** (human or novel robot); all three lead to **Adapted policy** (no retraining required).]

open problem: few-shot, real-time adaptation

## Slide 27: Towards a Unified Model for Mobile Manipulation (frame 28)

**today**: SLAM + metric map -> Localization stack -> Nav planner -> (hard handoff) -> VLA (manipulation only); Maps are static - world changes, can't precompute all

**vision** - which env. representation? Metric map, Gaussian splat, Topological map, Scene graph, BEV projection, Walkthrough video (?)

- Mapless adaptation - reason from current obs + memory
- ⚠ context length bottleneck - can't fit a city map
- **Unified model** - implicit state estimation, loop closures, whole body control -> **Navigation** (mobile base) and **Manipulation** (bimanual, dexterous)

Open points along the bottom: **Replace SLAM** (implicit localization) - **Env. representation** (map, video, implicit?) - **Whole body control** (nav + manipulation jointly)

## Slide 28: Multimodal Memories Across Time & Space (frame 29)

what gets stored? video, depth, language, touch, audio, proprio, bboxes, ...

| Short-horizon memory (seconds) | Long-horizon memory (minutes - hours) | Lifelong memory (months - years) |
|---|---|---|
| dense video, recent frames | language, semantic events | episodic + semantic |
| occlusions, grasp correction | task progress, steps done | smarter over deployment |
| high token cost | loses spatial detail | storage + retrieval at scale |
| doesn't scale to minutes | when to write, what to forget? | staleness, consolidation |

All three feed **cross-modal alignment** (joint reasoning + retrieval across all modalities and timescales) -> **Unified multi-scale memory** (video + language + touch + ...)

open problem: store, align, retrieve, and forget across modalities, tasks, and a lifetime of deployment

[Note: slide 29 does not appear in the captured frames; the numbering jumps from 28 to 30.]

## Slide 30: First Steps: Autonomous Improvement (frame 30)

- Leverage foundation models to enable autonomous improvement without human interventions

[Figure: an **Offline Dataset** and an **Online Dataset** train the **Robot Policy**, which is deployed on Robots 1-5 (kitchen scenes) for "Continuous Improvement". A **Task Proposals** foundation model suggests tasks to the robots, and a **Reward Detector** foundation model scores their attempts, feeding "Autonomous Data Collection" back into the online dataset.]

Reference: *Autonomous Improvement of Instruction Following Skills via Foundation Models*, Zhou\*, Atreya\*, Lee, Walke, **Mees**, Levine. CoRL, 2024.

## Slide 31: First Steps: Adaptation via Human Feedback (frame 31)

- Leverage **verbal** or **visual cues from humans** to **adapt** to new tasks

[Figure, left: PALO. For a new task ("put the turnip in the drawer"), "PALO: Policy Adaptation via Language Optimization" uses a VLM to search in language space for a task decomposition from **only 5 demos**, while "Policy Finetuning in parameter space" needs **> 100 Demostrations** (crossed out). A plot of success rate against number of demonstrations: PALO is near 0.8 from about 10 demos; finetuning climbs slowly to about 0.65 at 60.]

Reference: *Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation*, Myers\*, Zheng\*, **Mees** et al. CoRL 2024.

[Figure, right: the PR2 dialogue from week 3, slide 15 ("Fetch the round yellow thing" / "Do you mean the lemon in the middle?" / "Yes" / "Place it left of the object on the bottom").]

Reference: *Composing Pick-and-Place Tasks By Grounding Language*, **Mees,** Burgard ISER 2021.

[Note: week 3 cites the same paper as ISER 2020; this slide says ISER 2021. Both are defensible: it was presented at ISER 2020 and the Springer proceedings came out in 2021.]

## Slide 32: Ingredients for Embodied Intelligence (frame 32)

[Four boxes:] Intelligent Embodied Reasoning - Dexterous Mobile Manipulation - Lifelong Learning - Rapid Adaptation

## Slide 33: [untitled: transition] (frame 33)

**After this course, you have the tools to start advancing the frontier in robot learning!**

## Slide 34: [untitled: transition question] (frame 34)

**But, how to do research in robot learning?**

## Slide 35: The (Harsh) Reality of Research (frame 35)

1. Most research is **incremental** - today's self-driving cars rely on on 40 years of work: from Pomerleau's ALVINN in 1986 to the DARPA Challenge to modern end-to-end learning
2. Most research ideas never become papers.
3. Most papers don't stand the **test of time**.
4. The most impactful ideas are often the simplest, because **simple ideas can be scaled**

[Note: ALVINN appeared at NIPS 1988, with proceedings published in 1989; the slide's 1986 is earlier than either.]

## Slide 36: The Recipe for Good Research Problems (frames 36-38)

- Needed ingredients:
  1. an important problem
  2. a plan for how to tackle it
  3. **excitement!** - Research requires tons of time and effort. You will be more likely to succeed if you are excited!
  4. be your own reviewer #2 - What is the most likely reason your idea could fail? If you can answer that honestly and still believe in the idea, proceed
- Example ideas:
  - Cure cancer (important, but how?)
  - Algorithm that improves 1% on Libero benchmark (missing problem)
- If you are succesfull, **how does it help the community**?

[Ingredients 3 and 4 are added one per build (frames 37-38), each replacing the "Example ideas" block with its explanation; the slide numbers run 36-38.]

## Slide 39: Styles of Research (frame 39)

| Method-driven | Problem-driven |
|---|---|
| You start with an idea, but need to find the problem for it | You start with a problem, but need to find the method for it |
| "Video diffusion models just dropped, let me apply it to robot manipulation... somehow" | "How can I make my VLA work with a novel camera viewpoint? |

Both can lead to impactful research, but problem-driven is safer for a PhD student

## Slide 40: Debugging Your Research (frame 40)

- Start with something that **should work**, then make it incrementally harder
- Talk to colleagues, authors of papers you are building upon, advisors etc.
- Visualize your model's data & outputs to understand its behavior
- Revisit your initial assumptions after experiments

## Slide 41: Share Your Research (frame 41)

- Sharing your findings is how you **find your community**
- **Open-source** code & data: the most direct way for your research to be useful to others
- An image is worth a thousand words: polish your figures
- **Adapt** to your **audience**: even experts may know nothing about your specific topic

## Slide 42: Share Your Research [social media] (frame 42)

- If nobody knows about your research, it didn't happen!
- Disseminate your research on social media
  - Find the line between making people curious about your work
  - And overhyping it like this:

[Figure: a large red cross next to the same "ChatGPT moment for robotics is happening tomorrow" post from slide 1 (Jan 7, 2024, 2.4M views).]

## Slide 43: Personal Backstory (frame 43)

Early PhD: frustrated with thousand ROS nodes, disjoint models and errors accumulating through a complex pipeline

Fun fact: robot broke a week before the deadline

-> [arrow]

- Can I ditch ROS and learn everything (motor skills+ language) end2end for a cool end of PhD demo?
- What would it take?
- Wrote a proposal that funded me for the next PhD years

[Figure: the three-panel PR2 language-grounding figure from week 3, slide 15.]

## Slide 44: Personal Backstory [continued] (frame 44)

Convinced my advisor to buy me a Franka robot, spent 1 year setting it up with a custom made table, VR control etc.

[Figure: two photos of a Franka robot arm at a custom wooden desk setup (the one used for the play-data work in week 3), one with a person operating it.]

[Note: slide numbers 45-47 do not appear in the captured frames; the next frame shows no slide number, and the numbering resumes at 48.]

## [untitled, no slide number: closing message] (frame 45)

**This was my last lecture of the course!**

Thank you for your patience, this was a new course and your mid-term feedback was very helpful.

**Teaching this has been an absolute privilege**, even while juggling two full-time jobs.

I am looking forward to seeing your projects on demo day!

## Slide 48: Huge Thanks to the Teaching Assistants (frame 46)

[Figure: group photo of the teaching assistants on a rooftop terrace, with the ETH Zurich main building dome and the lake in the background.]

This course would not have been possible without them

## Slide 49: References (frame 47)

- Uni Freiburg, Deep Learning Lab
- UC Berkeley Deep RL
- Stanford University, Deep RL
- Cornell Robot Learning
