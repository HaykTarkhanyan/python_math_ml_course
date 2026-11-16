# Week 01 study guide: Introduction to Robot Learning

- Written by Claude from this week's slides (`slides.md`), the transcript and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Nothing technical is assumed this week. It helps to know what a neural network classifier is, what "training data" and "test data" mean, and roughly what a transformer and a vision-language model (VLM) do, because the last third of the lecture argues that a robot policy is "a VLM whose answers are actions".

### Extra comments

**Shakey was SRI, not Stanford University.** The slide says "Stanford 1966"; the robot was built at SRI (then the Stanford Research Institute, which split from the university in 1970) and ran from 1966 to 1972. The lecture's fun fact holds: A* search (Hart, Nilsson and Raphael, 1968) was invented for Shakey's navigation, and so was the STRIPS planner (Fikes and Nilsson, 1971), the ancestor of most symbolic task planners. ([26:01](https://youtu.be/X0k14u6pSxw?t=1561), [slide 16](slides.md#slide-16))

**Sense-plan-act and the closed-world assumption.** Sense the world, build a model, plan in the model, execute the plan open-loop. It works beautifully when the model is right, which is why car factories still run this way, and fails when the world departs from the model (a part slightly out of place, an unexpected obstacle). The first big rebellion was Rodney Brooks' behaviour-based robotics in the mid-1980s: tight reactive loops instead of one big world model. Learning-based control is the second. ([29:01](https://youtu.be/X0k14u6pSxw?t=1741), [slide 17](slides.md#slide-17))

**What "state estimation" means.** Working out where the robot and the objects around it are, from noisy sensors. The falling-robot videos are what happens when that estimate is off by a few centimetres and the controller trusts it completely; the QR codes on the walls in the Atlas clip are there to make the estimate easier. The classic reference for estimation with Kalman and particle filters is *Probabilistic Robotics* (Thrun, Burgard, Fox), the book on slide 7. Dieter Fox is the week 12 guest. ([31:02](https://youtu.be/X0k14u6pSxw?t=1862), [slide 18](slides.md#slide-18))

**Why Moravec's paradox exists.** Two explanations, which agree. Moravec's own: sensorimotor skill was tuned by hundreds of millions of years of evolution and runs below conscious awareness, so we never notice how hard it is; abstract reasoning is evolutionarily new, so it feels hard to us but is computationally shallow. The lecture's data explanation: an LLM learns from trillions of tokens of text about reasoning, but there is no internet of joint torques for grasping a mug. That second reason is why "where does robot data come from" keeps coming back in weeks 9 and 11. ([33:00](https://youtu.be/X0k14u6pSxw?t=1980), [slide 19](slides.md#slide-19))

**The PR1 video is good news, not bad news.** The 2007 tidying robot was teleoperated. So the hardware could already do the task; what was missing was the controller that decides what to do. That is the gap this course is about. The PR1 team (Eric Berger, Keenan Wyrobek) went on to build the PR2 at Willow Garage, the robot the ROS software stack grew up on. ([40:00](https://youtu.be/X0k14u6pSxw?t=2400), [slide 23](slides.md#slide-23))

**Check the data-scale numbers yourself.** Llama 3 was pretrained on about 15 trillion tokens. A person reads about 250 words a minute, roughly 330 tokens. Nonstop, that is $15 \times 10^{12} / 330 \approx 4.5 \times 10^{10}$ minutes, and a year has about $5.3 \times 10^{5}$ minutes, so about 86,000 years: the slide's 90,000 is right. The point for robotics: robot datasets are counted in hours of robot time, and even the largest are several orders of magnitude smaller. ([44:01](https://youtu.be/X0k14u6pSxw?t=2641), [slide 25](slides.md#slide-25))

**AlphaGo was not pure reinforcement learning.** It started from a network trained to imitate human expert moves, then improved by self-play RL, and chose moves with Monte Carlo tree search. Move 37 (game 2 against Lee Sedol, March 2016) came out of that combination. The lecture's point still stands: imitation alone is capped by the demonstrator, while optimising a reward can find moves no demonstrator made. Keep this contrast in mind for weeks 3 to 5. ([46:00](https://youtu.be/X0k14u6pSxw?t=2760))

**Open X-Embodiment, corrected.** The captions say "over 200 institutions in 2004"; the paper (arXiv 2023, ICRA 2024 best paper) reports 22 robot types from 21 institutions, 527 skills, and over a million trajectories, all converted to one format. The models it names (RT-1-X, RT-2-X) showed positive transfer: training on other robots' data helped each robot. ([49:00](https://youtu.be/X0k14u6pSxw?t=2940), [slide 29](slides.md#slide-29))

**Reading $[\Delta x, \Delta\theta, \Delta\text{Grip}]$.** The policy does not output motor torques. It outputs a small change in the gripper's position ($\Delta x$, 3 numbers) and orientation ($\Delta\theta$, 3 numbers) plus a gripper open/close command: 7 numbers per step, usually 3 to 50 times a second. A lower-level controller turns that into joint motions (inverse kinematics and PID, which is week 2). RT-1 and RT-2 cut each of those numbers into 256 bins and treat each bin as a token, which is what makes "actions are just another kind of token" on slide 33 literal. ([50:01](https://youtu.be/X0k14u6pSxw?t=3001), [slide 30](slides.md#slide-30))

**Why i.i.d. is the dividing line.** In supervised learning the test inputs come from the same distribution as the training inputs, whatever the model predicts. A policy's predictions are actions, and actions decide the next input. A small mistake moves the robot to a state no demonstration ever visited, where the policy is worse, so it makes a bigger mistake. Week 3 turns this into a bound: error grows with the square of the task length for naive imitation. ([55:01](https://youtu.be/X0k14u6pSxw?t=3301), [slide 36](slides.md#slide-36))

**The SO-101 arm.** The group-project robot is the low-cost open-source arm used with Hugging Face's LeRobot library. "Puppeteering" in the lecture is leader-follower teleoperation: you move a second, passive copy of the arm by hand, and the robot copies the joint angles. That is how you will collect your own demonstrations. ([19:00](https://youtu.be/X0k14u6pSxw?t=1140), [slide 8](slides.md#slide-8))

### Going further

- Rich Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019, two pages): the argument behind "from hand-coded to learned" in one essay.
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864): read the abstract and the dataset figure; the rest returns in week 9.
- [Moravec's paradox](https://en.wikipedia.org/wiki/Moravec%27s_paradox) on Wikipedia, with the original quotes from Moravec, Minsky and Pinker.
- Companion courses for the first half: Russ Tedrake's [Robotic Manipulation](https://manipulation.csail.mit.edu/) notes (MIT) for week 2, and Sergey Levine's [CS 285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) (Berkeley) for weeks 3 to 5.

## Lecture quiz

### Q1

Which statement is Moravec's paradox?

- A) Skills that animals find effortless, like grasping, are hard for AI, while abstract reasoning like chess is comparatively easy.
- B) Robots will never match humans at abstract reasoning.
- C) More compute makes sensorimotor control easy but leaves reasoning hard.
- D) The more carefully a robot plans, the more often it falls.

<details><summary>Answer</summary>

**A.** Moravec (1988): "abstract thinking: hard for animals, easy for AI; sensorimotor skills: easy for animals, hard for AI". B gets the direction backwards: chess engines beat world champions decades ago, while grasping an unfamiliar mug is still not reliably solved. ([slide 19](slides.md#slide-19), [33:00](https://youtu.be/X0k14u6pSxw?t=1980))

</details>

### Q2

In the "are these robots?" discussion, why is the da Vinci surgical system not counted as an autonomous robot?

- A) It has no sensors.
- B) It cannot change the physical world.
- C) The surgeon drives every motion through teleoperation; the machine only scales and steadies the movements.
- D) It is too expensive to be a product.

<details><summary>Answer</summary>

**C.** The da Vinci clearly acts on the world (so B is wrong) and has cameras and joint encoders (so A is wrong). It decides nothing: a human closes the loop. The lecture's target combines three things: an appliance's reliability, a surgical arm's dexterity, and a foundation model's general reasoning. ([39:02](https://youtu.be/X0k14u6pSxw?t=2342), [slide 22](slides.md#slide-22))

</details>

### Q3

What does the 2007 Stanford PR1 tidying video show?

- A) In 2007 robot hardware was too weak to tidy a room.
- B) Tidying a room had already been solved by 2007.
- C) Teleoperation is too slow to be useful for data collection.
- D) The robot was teleoperated, so the hardware could do the job; the missing piece was the software that decides what to do.

<details><summary>Answer</summary>

**D.** The thought bubble on the last build says it: "Robot is being teleoperated!". The body was capable; the brain was a human. Teleoperation is in fact how most robot training data is collected today, so C is the opposite of the course's message. ([41:01](https://youtu.be/X0k14u6pSxw?t=2461), [slide 23](slides.md#slide-23))

</details>

### Q4

Llama 3 was pretrained on about 15 trillion tokens. If a person reads about 330 tokens a minute without ever stopping, how long would it take to read the training set?

- A) About 900 years
- B) About 9,000 years
- C) About 90,000 years
- D) About 9 million years

<details><summary>Answer</summary>

**C.** $15 \times 10^{12} / 330 \approx 4.5 \times 10^{10}$ minutes. A year has $60 \times 24 \times 365 \approx 5.3 \times 10^5$ minutes, so about $8.6 \times 10^4$ years, matching the slide's 90,000. The order-of-magnitude habit is worth building: it is the quickest way to see why robot data is "scarce". ([slide 25](slides.md#slide-25), [44:01](https://youtu.be/X0k14u6pSxw?t=2641))

</details>

### Q5

The policy on slide 30 outputs $[\Delta x, \Delta\theta, \Delta\text{Grip}]$. What is that?

- A) A small change in the gripper's position and orientation, plus a gripper open/close command.
- B) A torque for each motor of the arm.
- C) Pixel coordinates of the object to pick.
- D) A language description of the next step.

<details><summary>Answer</summary>

**A.** These are end-effector deltas: where the hand should move next, relative to where it is now. Turning that into joint motions and motor commands is the job of the low-level controller (inverse kinematics and PID, week 2). Torques (B) are what the lowest layer of the controller finally sends, not what this policy predicts. ([slide 30](slides.md#slide-30), [50:01](https://youtu.be/X0k14u6pSxw?t=3001))

</details>

### Q6

Slide 35 lists why "train a large transformer on robot data and we are done" does not quite work. Which of these is NOT on it?

- A) Robot data is scarce and collecting it needs human supervision.
- B) Robots need high-frequency control, so model size and speed matter.
- C) Robot evaluations are tedious, expensive and hard to reproduce.
- D) Transformers cannot take images as input.

<details><summary>Answer</summary>

**D.** Transformers take images fine (vision transformers, VLMs); the slide's point is the opposite, that the same architecture can take language, image and action tokens. The three real obstacles map to the three ingredients: large datasets (A), large models that still run in real time (B), and scalable evaluation (C). A 100-billion-parameter model that needs a second per answer cannot steer a robot at 50 Hz. ([slide 35](slides.md#slide-35), [52:00](https://youtu.be/X0k14u6pSxw?t=3120))

</details>

### Q7

Why does the i.i.d. assumption of supervised learning break when a learned policy controls a robot?

- A) Robot images are too high-dimensional.
- B) Demonstrations are noisy.
- C) The policy's own actions decide which states it sees next, so after a mistake it can end up in states unlike any in the training data.
- D) Rewards in robotics are sparse.

<details><summary>Answer</summary>

**C.** In image classification, a wrong label does not change the next image. In control it does: the policy shapes its own test distribution. Noise (B) and sparse rewards (D) are real problems, but they exist with or without i.i.d. data. Week 3 makes this precise as compounding error. ([slide 36](slides.md#slide-36), [55:01](https://youtu.be/X0k14u6pSxw?t=3301))

</details>

### Q8

What does the lecture say about factory robots that follow the sense-plan-act recipe?

- A) They are precise and very successful, but only in structured environments; when the world deviates from their programmed model, they do not adapt.
- B) They rarely work, even in factories.
- C) They learn their plans from data.
- D) They cannot use cameras.

<details><summary>Answer</summary>

**A.** "Only as good as the logic we program into them." Factories are engineered so the world matches the model: fixed fixtures, known part positions. The course is about leaving that closed world. ([30:02](https://youtu.be/X0k14u6pSxw?t=1802), [slide 17](slides.md#slide-17))

</details>

### Q9

Why does the lecture bring up AlphaGo's move 37?

- A) To show that games are harder than robotics.
- B) As an example of a system finding a strategy no human had shown it, the "discovery era" robotics would like to reach.
- C) To show that AlphaGo learned only by copying human games.
- D) To show that large language models can play Go.

<details><summary>Answer</summary>

**B.** Move 37 looked like a mistake to human experts and turned out to be strong. Copying humans (C) caps you at the demonstrator's level; AlphaGo started from imitation but improved past it with self-play reinforcement learning and search. That imitation-versus-reward contrast is the backbone of weeks 3 to 5. ([46:00](https://youtu.be/X0k14u6pSxw?t=2760))

</details>
