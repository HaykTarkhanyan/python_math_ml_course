# Week 02 study guide: Robot Control & MDPs

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Matrix multiplication and inverses, the chain rule and gradients, a 2D rotation matrix, and basic probability (conditional distributions, expectations). If $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ rotating a vector by $\theta$ is familiar, you are set. Homework 2 uses exactly this lecture, so it pays to work through the quintic and PID slides with pen and paper.

### Extra comments

**Why 3x3 and 4x4 matrices for 2D and 3D.** Adding a constant 1 to every point ("homogeneous coordinates") lets one matrix do rotation and translation together, so a whole chain of moves is a product of matrices. The inverse is cheap and worth memorising: for $T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}$, $T^{-1} = \begin{bmatrix} R^T & -R^T p \\ 0 & 1 \end{bmatrix}$, because $R^{-1} = R^T$ for a rotation. On slide 4 the definition of $SO(n)$ writes $m \times m$; it should be $n \times n$. ([05:00](https://youtu.be/5-Bb84eTTqQ?t=300), [slide 4](slides.md#slide-4))

**The index-cancelling rule for chaining.** Read $T^{A}_{B}$ as "B's pose expressed in A's frame". Neighbouring indices cancel: $T^{A}_{B} \, T^{B}_{C} = T^{A}_{C}$. If the indices do not touch, you have the order or a direction wrong, the most common bug in robotics code. $T^{\text{Base}}_{\text{Sensor}}$ (where the lidar is bolted on) comes from calibration and is called the extrinsics. ([06:00](https://youtu.be/5-Bb84eTTqQ?t=360), [slide 5](slides.md#slide-5))

**Grübler's formula, worked.** Count the ground as a link. A planar arm with two revolute joints: $m = 3$ freedoms per planar body, $N = 3$ bodies, two joints with $c = 2$ constraints each, so $3(3-1) - 4 = 2$ DoF, as expected. The slide's form is the same as the textbook's $m(N - 1 - J) + \sum f_i$, since each joint's constraints are $c_i = m - f_i$. ([12:01](https://youtu.be/5-Bb84eTTqQ?t=721), [slide 10](slides.md#slide-10))

**The SO-101 has 5 arm joints, not 6.** Shoulder pan, shoulder lift, elbow flex, wrist flex and wrist roll, plus the gripper (that is the 6 in $q \in \mathbb{R}^6$ on slide 22). Five joints cannot reach every 6D pose: inside its workspace the SO-101 can put the gripper at a point, but only in some orientations. Expect IK targets that have no exact solution. ([slide 7](slides.md#slide-7))

**Topology matters for learning.** The 2R arm's configuration space is a torus because angles wrap: $359°$ and $1°$ are neighbours, but as raw numbers they are far apart. A network that regresses raw angles sees a jump where there is none. Common fixes: predict $(\sin\theta, \cos\theta)$, and for 3D rotations use a continuous representation instead of Euler angles (the "6D" representation of Zhou et al. is the usual choice in robot learning). ([13:01](https://youtu.be/5-Bb84eTTqQ?t=781), [slide 11](slides.md#slide-11))

**Redundancy is useful, not a nuisance.** A 7-joint arm doing a 6D task has one spare dimension: the elbow can swing while the hand stays still (the "null space"). Controllers use it to avoid obstacles or joint limits without disturbing the task. ([19:01](https://youtu.be/5-Bb84eTTqQ?t=1141), [slide 16](slides.md#slide-16))

**Jacobian-transpose IK is plain gradient descent, and it has relatives.** The update on slide 20 is gradient descent on $\frac{1}{2}\|f(\theta) - x_{\text{target}}\|^2$. It is simple but slow near the target. Two standard upgrades: the pseudo-inverse step $\Delta\theta = -J^{+} e$ (a Gauss-Newton step) and damped least squares $\Delta\theta = -J^T (J J^T + \lambda^2 I)^{-1} e$, which stays well-behaved at singularities, the configurations (such as a fully stretched arm) where $J$ loses rank and some directions become unreachable. ([22:00](https://youtu.be/5-Bb84eTTqQ?t=1320), [slide 20](slides.md#slide-20))

**Check the quintic numbers yourself.** With $f(s) = 10s^3 - 15s^4 + 6s^5$: $f'(s) = 30s^2 - 60s^3 + 30s^4$, so the peak speed at $s = 0.5$ is $1.875$ times the average speed, the slide's 1.9 m/s for a 1 m move in 1 s. $f''$ peaks at $s \approx 0.21$ with about $5.8$, the slide's acceleration peak. The cubic $3s^2 - 2s^3$ also starts and ends at rest, but its acceleration jumps at both ends; the two extra coefficients of the quintic buy zero acceleration there too. The $\pm 150$ spikes on the LERP plot are an artefact of the plot's time step: in theory the acceleration is a Dirac delta, infinite for an instant. ([25:00](https://youtu.be/5-Bb84eTTqQ?t=1500), [slides 23](slides.md#slide-23) and [25](slides.md#slide-25))

**PID: what each term really fixes.** P alone leaves a steady offset under a constant load: to push against gravity it needs a nonzero error. I accumulates that leftover error until it is gone, which is why the slide calls it "anti-gravity"; it works on any constant disturbance, but it can "wind up" (keep growing while the motor is saturated) and then overshoot. D reacts to how fast the error changes, so it brakes; it also amplifies sensor noise, so real controllers filter it. Industrial arms usually add a gravity-compensation term computed from a model, so I has little left to do. ([26:01](https://youtu.be/5-Bb84eTTqQ?t=1561), [slide 27](slides.md#slide-27))

**Where learning enters the stack.** Everything up to slide 28 is the low-level layer. A learned policy usually sits on top: it outputs a target pose or joint position a few to 50 times a second, and IK, interpolation and PID turn that into motor commands at hundreds or thousands of hertz. When a policy "jitters", the cause is often in this layer, not in the network. ([28:00](https://youtu.be/5-Bb84eTTqQ?t=1680), [slide 28](slides.md#slide-28))

**State vs observation, and why it matters later.** The state is everything needed to predict the future; the observation is what the sensors give you, a lossy function of the state. A policy that sees only $o_t$ is in general not Markov: one camera frame does not show velocities or what is hidden behind the gripper. The standard fixes are to stack several past frames or to use a sequence model, which is part of why transformers show up in week 7. ([34:00](https://youtu.be/5-Bb84eTTqQ?t=2040), [slides 33](slides.md#slide-33) and [34](slides.md#slide-34))

**Reward design has a safe version.** Dense rewards speed learning but can change what the optimal behaviour is (the robot learns to hover near the object instead of grasping it). Potential-based shaping, $r' = r + \gamma \Phi(s') - \Phi(s)$ for any function $\Phi$ of the state, provably leaves the optimal policy unchanged (Ng, Harada and Russell, 1999). The slide's question "how to detect if reward conditions are met?" is a real problem on hardware: in simulation the reward is a line of code, in the real world you need a success detector. ([41:01](https://youtu.be/5-Bb84eTTqQ?t=2461), [slide 40](slides.md#slide-40))

**The trajectory probability is the seed of policy gradients.** Take the log of slide 41: $\log p_\pi(\tau) = \log\rho(s_0) + \sum_t \log\pi(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)$. Only the middle sum depends on the policy's parameters, so its gradient needs no model of the dynamics. That one line is why model-free RL (weeks 4 and 5) can work at all. ([42:01](https://youtu.be/5-Bb84eTTqQ?t=2521), [slide 41](slides.md#slide-41))

**Effective horizon.** With discount $\gamma$, rewards about $1/(1-\gamma)$ steps away still count noticeably. $\gamma = 0.99$ means roughly 100 steps: 10 seconds of control at 10 Hz, only 2 seconds at 50 Hz. A policy that acts faster needs a $\gamma$ closer to 1 to plan equally far ahead. ([43:00](https://youtu.be/5-Bb84eTTqQ?t=2580), [slide 42](slides.md#slide-42))

### Going further

- Lynch and Park, [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) (free PDF and videos), the source of slides 9 to 11. Chapters 2 to 4 and 6 cover this lecture's first half.
- Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (free), chapter 3 for MDPs, returns and discounting.
- Zhou et al., [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035), for why Euler angles and quaternions are awkward network outputs.

## Lecture quiz

### Q1

A car's lidar sees a pedestrian. You know $T^{\text{Base}}_{\text{Sensor}}$ and $T^{\text{Sensor}}_{\text{Ped}}$. Which product gives the pedestrian's pose in the car's base frame?

- A) $T^{\text{Sensor}}_{\text{Ped}} \, T^{\text{Base}}_{\text{Sensor}}$
- B) $(T^{\text{Base}}_{\text{Sensor}})^{-1} \, T^{\text{Sensor}}_{\text{Ped}}$
- C) $T^{\text{Base}}_{\text{Sensor}} \, T^{\text{Sensor}}_{\text{Ped}}$
- D) $T^{\text{Base}}_{\text{Sensor}} + T^{\text{Sensor}}_{\text{Ped}}$

<details><summary>Answer</summary>

**C.** Neighbouring indices cancel: Base-Sensor times Sensor-Ped gives Base-Ped. Matrix products do not commute, so A is a different (and meaningless) transform. B inverts the first factor into $T^{\text{Sensor}}_{\text{Base}}$, whose indices no longer touch those of $T^{\text{Sensor}}_{\text{Ped}}$. ([slide 5](slides.md#slide-5), [06:00](https://youtu.be/5-Bb84eTTqQ?t=360))

</details>

### Q2

Using Grübler's formula, how many degrees of freedom does a planar arm with two revolute joints have? (Planar bodies have $m = 3$ freedoms; a revolute joint imposes $c = 2$ constraints in the plane.)

- A) 2
- B) 3
- C) 4
- D) 6

<details><summary>Answer</summary>

**A.** Count the ground as a link: $N = 3$. Then $m(N-1) - \sum c_i = 3 \cdot 2 - (2 + 2) = 2$. Forgetting the ground gives $3 \cdot 1 - 4 = -1$, a sign you miscounted. ([slide 10](slides.md#slide-10), [12:01](https://youtu.be/5-Bb84eTTqQ?t=721))

</details>

### Q3

A 7-joint arm controls the 6D pose of its gripper. Which statement is true?

- A) Its workspace is a 7-dimensional space.
- B) Its configuration space is $SE(3)$.
- C) Inverse kinematics has exactly one solution for every reachable pose.
- D) It is redundant: it can move its elbow while the gripper stays still (null-space motion).

<details><summary>Answer</summary>

**D.** Task space (6D) is smaller than the DoF count (7), which is the slide's definition of redundancy. The configuration space is the 7 joint angles (so B mixes up C-space and task space), the workspace is the set of reachable points in 3D space (A), and with a spare dimension IK has infinitely many solutions, not one (C). ([slide 16](slides.md#slide-16), [19:01](https://youtu.be/5-Bb84eTTqQ?t=1141))

</details>

### Q4

What is the update $\theta \leftarrow \theta - \alpha J(\theta)^T (f(\theta) - x_{\text{target}})$ doing?

- A) Solving inverse kinematics exactly in one step.
- B) Gradient descent on $\frac{1}{2}\|f(\theta) - x_{\text{target}}\|^2$.
- C) Computing forward kinematics.
- D) Inverting the Jacobian.

<details><summary>Answer</summary>

**B.** By the chain rule the gradient of the squared error is $J^T e$, so this is ordinary gradient descent with step size $\alpha$, repeated until the error is small. It never inverts $J$ (D), which is its advantage near singularities and its weakness in speed; the pseudo-inverse and damped least-squares methods are the faster relatives. ([slide 20](slides.md#slide-20), [22:00](https://youtu.be/5-Bb84eTTqQ?t=1320))

</details>

### Q5

A joint moves 1 rad in 1 s using quintic time scaling $f(s) = 10s^3 - 15s^4 + 6s^5$. What is its peak velocity?

- A) 1.875 rad/s, halfway through the move
- B) 1 rad/s, constant throughout
- C) 3 rad/s, at the start
- D) 0.5 rad/s, at the end

<details><summary>Answer</summary>

**A.** $f'(s) = 30s^2 - 60s^3 + 30s^4$ and $f'(0.5) = 7.5 - 7.5 + 1.875 = 1.875$. The velocity is zero at both ends (that is the point), so C and D are wrong, and a constant 1 rad/s (B) is LERP, which jumps from 0 to full speed instantly. Smoothness costs a higher peak speed than LERP's. ([slide 24](slides.md#slide-24), [25:00](https://youtu.be/5-Bb84eTTqQ?t=1500))

</details>

### Q6

An arm holding still sags a little below its target because of gravity, and the sag does not go away. Which PID term removes it?

- A) Proportional
- B) Derivative
- C) Integral
- D) None: raise the control frequency instead.

<details><summary>Answer</summary>

**C.** With P alone the motor only pushes when there is an error, so it settles where the push from $K_p e$ balances gravity, at a nonzero error. The integral keeps growing while any error remains, until the offset is gone. D only reacts to changes in error, and a constant sag has none. ([slide 27](slides.md#slide-27), [27:00](https://youtu.be/5-Bb84eTTqQ?t=1620))

</details>

### Q7

Which statement about states and observations is right?

- A) An observation always contains the full state.
- B) A policy that uses only the current camera image is always Markov.
- C) Driving at night in snow changes the car's state.
- D) The car's state is the same in daylight or snow; only the observation changes, and one observation may not determine the state.

<details><summary>Answer</summary>

**D.** That is slide 33's point: the mapping from state to observation is lossy. A single image hides velocities and occluded objects, so B is false in general, and A is the special fully observed case. The weather changes what you see (the observation), not where the car is (C). ([slide 33](slides.md#slide-33), [34:00](https://youtu.be/5-Bb84eTTqQ?t=2040))

</details>

### Q8

An agent gets reward 1 at every step, forever, with discount $\gamma = 0.99$. What is its return?

- A) 99
- B) 100
- C) Infinite
- D) 0.99

<details><summary>Answer</summary>

**B.** $\sum_{t \ge 0} \gamma^t = \frac{1}{1 - \gamma} = \frac{1}{0.01} = 100$. Without discounting the sum would be infinite (C), which is why infinite-horizon problems need $\gamma < 1$. The same number, $1/(1-\gamma)$, is a handy estimate of how many steps ahead the agent effectively cares about. ([slide 43](slides.md#slide-43), [45:02](https://youtu.be/5-Bb84eTTqQ?t=2702))

</details>

### Q9

In $p_\pi(\tau) = \rho(s_0) \prod_t \pi(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t)$, which factors does changing the policy's parameters affect?

- A) Only the policy terms $\pi(a_t \mid s_t)$.
- B) Only the transition terms $P$.
- C) All three factors.
- D) Only the initial-state distribution $\rho(s_0)$.

<details><summary>Answer</summary>

**A.** The initial state and the physics do not depend on the policy's weights. (The policy does change which states get visited, but that happens through the $\pi$ factors.) So $\nabla_\theta \log p_\pi(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$, with no dynamics model needed: the starting point of policy gradients in weeks 4 and 5. ([slide 41](slides.md#slide-41), [42:01](https://youtu.be/5-Bb84eTTqQ?t=2521))

</details>

## Guest talk

### Context

Abhishek Gupta leads the WEIRD lab (Washington Embodied Intelligence and Robotics Development) at the University of Washington. His running joke, "I'm a lazy potato", is the thesis of the talk: simulation is cheap data in principle, but the human hours hidden around it are not, and each part of the talk removes one of them. It pairs with this week's lecture because simulation gives you exactly what the lecture says the real world hides: the true state $s_t$, resets, and rewards you can compute.

### Extra comments

**"Off-domain data is wrong data, but useful."** Simulation, generated video and human video all differ from robot data in some way. The question is never whether the data is exact but whether it helps more than it hurts, and how little real data fixes the rest. The same argument returns with world models (week 8) and generalist policies (week 9). ([05:01](https://youtu.be/aG8NPTPhwkE?t=301))

**The three hidden costs.** Building the scene (hours in Blender), producing behaviour in it (teleoperation or reward engineering, see the many weighted reward terms he shows), and closing the sim-to-real gap (tuning and system identification). Each part of the talk automates one. ([07:00](https://youtu.be/aG8NPTPhwkE?t=420))

**Real-to-sim in plain terms.** Walk through a kitchen filming with a phone. COLMAP (structure from motion) works out where the camera was for each frame; Gaussian splatting turns the images into a 3D scene that renders realistically from new viewpoints and gives geometry; segmentation models cut out cabinets and drawers; a language model guesses which are hinged or sliding, and joints are added so the simulator can open them. He says only the camera images are used, not the phone's depth sensor. ([10:02](https://youtu.be/aG8NPTPhwkE?t=602), [32:01](https://youtu.be/aG8NPTPhwkE?t=1921))

**Simulation as an evaluation tool.** For comparing policies, what matters is that simulation ranks them in the same order as the real world, not that the success rates match. The two numbers on his plot: Pearson correlation between sim and real success (high is good) and MMRV, mean maximum rank violation (low is good), a metric introduced by the SIMPLER paper. He notes that a little simulation data had to be mixed into training for the real-world policies to behave sensibly in sim. ([14:01](https://youtu.be/aG8NPTPhwkE?t=841))

**Why resets make exploration easy.** With a sparse reward, random exploration from the start rarely reaches the goal, and the chance shrinks quickly with distance. Start right next to the goal and success is easy; then start a little further back each time. This is the old "reverse curriculum" idea (Florensa et al., 2017), and simulation makes it free because you can teleport the world into any state. For assembly he generates the start states by disassembling the finished object: "assembly by disassembly". ([18:01](https://youtu.be/aG8NPTPhwkE?t=1081), [29:00](https://youtu.be/aG8NPTPhwkE?t=1740))

**Sim for the plan, real data for the details.** Simulation tends to get the global strategy of a task right (pick up the leg, align, insert, twist) and the contact details slightly wrong. His approach keeps the long-horizon part learned in simulation and lets the robot fix the short-horizon dynamics from its own real-world experience, about 15 minutes of autonomous data in his examples. ([23:01](https://youtu.be/aG8NPTPhwkE?t=1381), [25:01](https://youtu.be/aG8NPTPhwkE?t=1501))

**Randomise rather than measure.** Asked whether to measure friction and softness of real objects, he says choosing a sensible range of physics parameters and training a policy that works across all of them has usually worked better. That is domain randomization (Tobin et al., 2017), the standard alternative to precise system identification. ([30:00](https://youtu.be/aG8NPTPhwkE?t=1800))

**A trick he does not mention but you will meet.** Because simulation exposes the true state, a common recipe is to train a "teacher" policy on privileged state with RL, then train a "student" policy that sees only camera images to imitate the teacher. Much of the legged-robot work from ETH that he refers to uses this pattern. ([06:00](https://youtu.be/aG8NPTPhwkE?t=360))

### Going further

- Florensa et al., [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/abs/1707.05300), the clean version of "start near the goal".
- Tobin et al., [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907).
- Li et al., [Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941) (SIMPLER), where MMRV comes from.
- From Gupta's group, in the direction of the talk: Torne et al., [Reconciling Reality through Simulation](https://arxiv.org/abs/2403.03949) (real-to-sim-to-real) and [Robot Learning with Super-Linear Scaling](https://arxiv.org/abs/2412.01770) (scenes from phone scans).
- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), if you want the reconstruction method itself.

## Guest talk quiz

### Q1

What does Gupta mean by the hidden cost of simulation?

- A) The price of the GPUs that run it.
- B) The human effort around it: building scenes, producing behaviour (demos or reward design), and tuning the simulator to match reality.
- C) Licence fees for physics engines.
- D) Simulation runs slower than real time.

<details><summary>Answer</summary>

**B.** Compute is the cheap part; simulation can run much faster than real time (so D is backwards). The expensive part is people: artists or PhD students building environments, engineers writing reward functions, and system identification to close the sim-to-real gap. ([07:00](https://youtu.be/aG8NPTPhwkE?t=420))

</details>

### Q2

Why do broad resets let RL learn with only a sparse reward?

- A) Resets add extra dense terms to the reward.
- B) Resets make the simulator physically more accurate.
- C) Resets remove the need for a policy network.
- D) Starting near the goal turns a needle-in-a-haystack search into short, easy problems, and starting further back step by step builds a curriculum.

<details><summary>Answer</summary>

**D.** The reward stays sparse (so not A); what changes is where episodes start. Being able to teleport the world into partly solved states is a privilege the real world does not give you. ([18:01](https://youtu.be/aG8NPTPhwkE?t=1081))

</details>

### Q3

When simulation is used to evaluate real-world policies, what matters most according to the talk?

- A) That simulation ranks policies in the same order as the real world: high correlation, low rank violation.
- B) That success rates in sim and real are identical.
- C) Nothing visual: photorealism does not matter.
- D) That the policies were trained in simulation.

<details><summary>Answer</summary>

**A.** "Things that are better in simulation would also be better in the real world." Exact rates (B) are not needed for choosing between policies. Photorealism did matter (C is wrong): less realistic scenes correlated much worse. The policies he evaluates were trained mostly on real data (D). ([14:01](https://youtu.be/aG8NPTPhwkE?t=841))

</details>

### Q4

In the real-world adaptation part, what comes from simulation and what from real data?

- A) Everything is relearned from real data; simulation only initialises the weights.
- B) Simulation provides only the visuals.
- C) The long-horizon structure of the task comes from simulation; short-horizon dynamics details are corrected with a little autonomous real-world data.
- D) Real data is used to retune the simulator's parameters by hand.

<details><summary>Answer</summary>

**C.** Simulation is right about the plan and wrong about contact details. Freezing the long-horizon part and adapting the short-horizon predictions needs little data, about 15 minutes in his examples. D is exactly the manual work he wants to avoid. ([23:01](https://youtu.be/aG8NPTPhwkE?t=1381))

</details>

### Q5

Asked whether to measure the real objects' friction and softness for the simulator, what did he say usually works better?

- A) Measuring them precisely with a force sensor.
- B) Picking a sensible range for those parameters and training a policy that copes with all of them.
- C) Ignoring physics and using only visuals.
- D) Reading them from the phone's lidar.

<details><summary>Answer</summary>

**B.** That is domain randomization. He calls precise system identification of objects an open question that has not made a big difference on his tasks. ([30:00](https://youtu.be/aG8NPTPhwkE?t=1800))

</details>
