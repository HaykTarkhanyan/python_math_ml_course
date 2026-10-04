# Lecture 1: Introduction to Robot Learning - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 16.02.2026
- Source: `slides.pdf` in this folder (64 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week01_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://www.youtube.com/watch?v=X0k14u6pSxw (transcript: `transcript_lecture.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 1: Introduction to Robot Learning

Oier Mees - Microsoft - 16.02.2026

## Slide 1: About Me (frame 2)

- From Basque Country, Spain
- PhD in Freiburg, Germany
- PostDoc in Berkeley, USA
- Research at Microsoft

[Figure: four illustrated travel posters: a map of Spain with a pin on the Basque Country, a map of Germany with a pin on Freiburg, the Berkeley campanile, and Zurich.]

[Note: slide 2 does not appear in the captured frames; the numbering jumps from 1 to 3.]

## Slide 3: Course Staff: TAs (frame 3)

Alexey Gavryushin, Jonas Pai, Liam Achenbach, Nicola Irmiger, Tianxu An, Šimon Sukup, Nicole Damblon, Zador Pataki

[Figure: a portrait photo above each name; each name also has an ethz.ch e-mail address, omitted here.]

## Slide 4: Course Staff: TAs [continued] (frame 4)

Carl Brander

[Figure: five grey placeholder silhouettes.] Several open TA positions: This could be you!

## Slide 5: The Plan for Today (frame 5)

- Course Goals & Logistics
- Why study Robot Learning?

[Figure: screenshot of the course web page (Computer Vision and Geometry Lab header, "Robot Learning - Oier Mees" banner over a collage of robot photos, then course title, semester, lecturer, teaching assistants, catalogue link 263-5911-00L, lecture Mon 16:15-18:00 in room NO C 60, and the format: each session begins with a lecture on the core topic followed by a student-led paper discussion, with short guest lectures on selected weeks).]

## Slide 6: Course Goal (frame 6)

- **Mastery of Fundamentals:** Imitation, Reinforcement and Policy Learning
- **Practical Skills:** Hands-on simulation and real-robot policy deployment
- **Frontier Models:** Exploration of Foundation Models for robotics
- **Systems Design:** Scalable pipelines for perception, control and reasoning

## Slide 7: Course Goal: Inspire you! (frame 7)

[Top row, left of the "ChatGPT Moment" arrow:]

- Translation: Seq2Seq... [Figure: unrolled recurrent network]
- Object Detection: R-CNN... [Figure: R-CNN pipeline: 1. Input image, 2. Extract region proposals (~2k), 3. Compute CNN features, 4. Classify regions]
- Image Captioning: DenseCap.. [Figure: CNN -> region features -> LSTM captioning pipeline]

-> **ChatGPT Moment** -> **Transformer** [Figure: one block with a row of **Images** tokens and a row of **Language** tokens] Vision-Language Models, Generative Models...

[Bottom row:] [Figure: covers of the *Springer Handbook of Robotics* (Siciliano, Khatib, eds.) and *Probabilistic Robotics* (Thrun, Burgard, Fox)] -> **ChatGPT for Robotics?** / you [above and below the arrow] -> [Figure: collage of robot photos]

## Slide 8: Course Structure (frame 8)

- Website: https://cvg.ethz.ch/lectures/Robot-Learning
- Mondays 16:15-17:45 :
  - Lecture 45 min + 45 min Paper Discussion
  - Guest lectures throughout course
- Thursdays 10:15-12:00
  - TA Led Practice Sessions
- Individual Homework: 4x assignments
- Group Projects on real SO-101 robots

[Figure: photo of two small SO-101 robot arms on a table, one moved by a person's hand.]

## Slide 9: Paper Discussion (frame 9)

- 3x papers per lecture, available at project course
- 15 minutes per paper, groups of 4 students
- Group A presents & defends paper, Group B criticizes it

Enter paper groups here: [Figure: QR code and a docs.google.com spreadsheet link for signing up.]

[Figure: screenshot of the "Lecture Tentative Schedule" table from the course page, weeks 1-5 with each week's paper-discussion readings.]

## Slide 10: Individual Homework (frame 10)

- Assignments: https://github.com/mees-robot-learning-course/ethz-course-2026
- Submission: Gradescope
- HW1: Pytorch & Numpy Tutorial - 16.02-26.02
- HW2: Robot Control & MDPs - 23.02-05.03
- HW3: Imitation Learning - 02.03-16.03
- HW4: Reinforcement Learning - 16.03-30.03

[Note: the course page later lists different due dates (March 5, March 12, March 26, April 16).]

## Slide 11: Homework Advice (frame 11)

- Neural networks take some time to train!
- We try to make the homeworks fast to train

[Figure: three-panel meme of a man waiting alone, captioned "Me waiting for my neural network to finish training".]

Don't start the night before the deadline working on the assignment!

## Slide 12: Course Grading (frame 12)

- Paper Presentation & Discussion (Group): 20 %
- Practical Homework (Coding Assignments): 40 %
- Final Project (Group): 40 %

[Figure: cartoon "Course Grading Forecast": a morning sun over books (Paper Presentation, 20%), a midday sun over a student coding with a robot arm (Practical Homework, 40%), and a rainbow over a robot with a pot of gold (Final Project, 40%).]

## Slide 13: Feedback Welcome (frame 13)

- We are working hard to offer a great course
- We will probably make mistakes
- We would love to hear your feedback!

[Figure: meme of a speaker in sunglasses at a World Economic Forum podium, captioned "Are we improvising and going to make mistakes? For sure".]

## Slide 14: Robots in Science Fiction (frame 14)

[Figure: six film and TV stills, captioned:] Q the Automaton, 1918 - Metropolis, 1927 - Star Wars, 1977 - The Jetsons, 1962 - Transformers, 2007 - Wall-E, 2008

## Slide 15: Why Robots (frame 15)

[Figure: four photos, no text: a Mars rover; a da Vinci surgical robot with a surgeon at the console; miners drilling underground; an elderly person's hands held by younger hands. Read as: space exploration, surgery, dangerous work, care.]

## Slide 16: Shakey the Robot (frames 16-23)

[Figure: an embedded historical film playing across eight frames: Shakey, a wheeled robot with a camera and rangefinder mast (overlay text "Research performed by"), researchers working at a blackboard, Shakey navigating a room with large blocks and a ramp, close-ups of its camera head and of its bump-sensor bar.]

Shakey the Robot, Stanford 1966

## Slide 17: Robot Engineering (frame 24)

- Sense-Plan-Act
- Similar to Shakey (1966)
- Structured Environments

[Figure: a person programming an orange industrial robot arm with a handheld teach pendant; a car-body assembly line lined with orange robot arms; a warehouse where small mobile robots carry shelving pods.]

## Slide 18: When State Estimation Fails (frames 25-31)

[Figure: three embedded videos playing across seven frames, each a compilation of robots falling: "DARPA Robotics Challenge 2015" (humanoid robots toppling at a doorway and on rubble), "Boston Dynamics 2018" (an Atlas humanoid falling in a kitchen, labelled Boston Dynamics | TED), "Boston Dynamics 2022" (Atlas on a parkour course with orange barriers and boxes).]

## Slide 19: Moravec's Paradox (1988) (frames 32-33)

- **Abstract thinking**: hard for animals, easy for AI
- **Sensorimotor skills**: easy for animals, hard for AI

[Figure: left, a PR2 two-armed robot tidying objects on a table in front of an Albert-Ludwigs-Universitat Freiburg "TidyUpRobot" banner, captioned "Mees et al., 2019"; right, two orangutans handling branches.]

## Slide 20: Robot Learning [Venn diagram] (frame 34)

[Figure: Venn diagram. Left circle **Robotics**: Physical World, Engineer Solutions. Right circle **Machine Learning**: Digital World, Data-driven Solutions. Overlap: **Robot Learning**.]

## Slide 21: Robot Learning [definition] (frame 35)

Solving **Robotics** via **Machine Learning**

- Robotics: Perception, Control, ...
- Machine Learning: Imitation Learning, Reinforcement Learning, Dynamics Learning, Representation Learning, ...

## Slide 22: Are These Robots? (frame 36)

[Figure: three photos: a dishwasher in a kitchen; a da Vinci surgical robot with a surgeon at the console; a phone camera view with a ChatGPT logo, the assistant pointing at a screw with the hint "Use this one for the shelf".]

## Slide 23: Why Don't We Already Have Autonomous Robots? (frames 37-39)

[Figure: embedded video of a robot tidying toy blocks in a living room, captioned "Berger and Wyrobeck, Stanford 2007" (one frame is black, mid-video).]

[Final build adds a thought bubble:] Robot is being teleoperated!

## Slide 24: Recent Hardware Advances (frame 40)

- Humanoid Robots [Figure: lineup of humanoids labelled Boston Dynamics Atlas, Unitree G1, Figure AI Figure 02, Tesla Optimus, Sanctuary AI Phoenix, Agility Digit, Apptronik Apollo, Fourier GR-1, 1x Technologies NEO]
- NVIDIA GPUs [Figure: a row of GPU server racks]

## Slide 25: Recent AI Advances: Data Scaling (frame 41)

- Avg time for a human to read dataset

[Figure: three circles sized by reading time:] Imagenet - 2 years; GPT-2 - 60 years; Llama 3 - 90,000 years

## Slide 26: Recent AI Advances: Models (frame 42)

[Figure: three images: instance segmentation of a street scene with cyclists; the AlphaGo match at a Go board; the Gemini logo.]

[Note: slides 27 and 28 do not appear in the captured frames; the numbering jumps from 26 to 29.]

## Slide 29: Recent Advances in Robot Learning (frame 43)

**Large Datasets**

[Figure: grid of many robot-manipulation scenes from different labs.]

Reference: *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, Open X-Embodiment Collaboration, **Mees** et al., ICRA, 2024. **Best Conference Paper Award (out of 1765 papers)**

**Large Models**

- [octopus emoji] Octo
- Isaac GR00T
- $\pi_0$
- RT-2
- ...

## Slide 30: [untitled: policy inputs and outputs] (frame 44)

[Figure: diagram. A camera image of a tabletop (towel, spoon, toy watermelon slice) and the instruction "Pick up the spoon" both feed a **Policy** box, which outputs ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$ (picture of a robot arm).]

[Note: slide 31 does not appear in the captured frames; the numbering jumps from 30 to 32.]

## Slide 32: [untitled: vision-language model inputs and outputs] (frame 45)

[Figure: same diagram layout. A photo of the Statue of Liberty and the New York skyline plus the prompt "Caption the scene" feed a **Vision-Language Model** box, which outputs "The picture shows the Statue of Liberty in NY".]

## Slide 33: Key: Robotics as Multimodal Sequence Modeling (frame 46)

[Figure: three groups of tokens in a row, Language (green), Image (blue), Action (purple), each above its source: the instruction "Pick up the spoon", the tabletop camera image, and ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$]

## Slide 34: [untitled: transition question] (frame 47)

**So we train a large transformer on robot data and we are done?**

## Slide 35: Ingredients for Scaling Robot Learning (frames 48-51)

| Large Datasets | Large Models | Scalable Evaluation |
|---|---|---|
| Robot Data is **Scarce** | Robot Data is **Multimodal & Heterogeneous** | Robot Evals are **Tedious & Expensive** |
| Collecting Data requires **Human Supervision** | Robots need **High Frequency Control** | **Difficult to reproduce** |

[Figure: pipeline along the bottom: a dataset cylinder -> **Robot Foundation Model** -> collage of robots deployed in many environments. The three columns appear one per build.]

## Slide 36: Algorithms for Robotic Learning (frame 52)

**Imitation Learning**

- Given labeled data: $\mathcal{D} = \{(x_i, y_i)\}$, learn $f(x) \approx y$
- Assumes: inputs x are independently, identically distributed (i.i.d.)
- [Figure: Dataset -> Supervised Learning]

**Reinforcement Learning**

- Learn behavior: $\pi(a \mid s)$
- data is **not** i.i.d., actions affect future states
- [Figure: loop: Dataset -> RL -> "trial" in the world (globe icon) -> "reward" -> back into Dataset]

## Slide 37: Course Syllabus: 1st Half (frames 53-56)

- Robot Learning Fundamentals & Algorithms
  - Robot Control & MDPs
  - Imitation Learning
  - Reinforcement Learning (Online & Offline)

[Figure: embedded video, "Homework Assignments in Simulation": three camera views (left_wrist, angle, top) of a simulated robot arm picking up a red cube and placing it in a box, with a teleoperation recording overlay "REC | ep 0/2 | substeps 10 - SPACE rec | ENTER end ep | r reset | ESC quit".]

## Slide 38: Course Syllabus: 2nd Half (frame 57)

- Scaling Robot Learning
  - Generative Models
  - Sequence Modeling & Transformers
  - World Models
  - Robot Foundation Models & Embodied Reasoning

[Figure: a person at a desk working with two small robot arms, captioned "Group Projects with Real Robots".]

## Slide 39: After the Course (frames 58-59)

- You will understand:

[Figure: left, the teleoperated living-room tidying robot ("Berger and Wyrobeck, Stanford 2007"); arrow to right, an embedded video ("Physical Intelligence, 2025") of a mobile two-armed robot cleaning a kitchen and a bedroom, with overlays "autonomous, unseen, 6x speed" and "autonomous, unseen, 10x speed" and a spoken instruction "Please clean up the spill too".]

[Note: slide 40 does not appear in the captured frames; the numbering jumps from 39 to 41.]

## Slide 41: After the Course [continued] (frames 60-62)

- You will understand:

[Figure: left, embedded clips from the "DARPA Robotics Challenge 2015" (humanoids falling at a doorway, one climbing out of a utility vehicle); arrow to right, "Figure AI, 2026": a humanoid robot loading a dishwasher in a kitchen.]

## Slide 42: Why You Should Study Robot Learning (frame 63)

- Broad skills from foundation model training to PID control
- Advances often applicable to all robotic applications

[Figure: "I want you" recruiting-poster meme captioned "I want you to solve physical AGI - Enlist in robot learning".]

## End (frame 64)

Thank you for your attention and enjoy the course!
