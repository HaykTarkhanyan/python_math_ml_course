# Lecture 2: Robot Control & Markov Decision Processes - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 23.02.2026
- Source: `slides.pdf` in this folder (52 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week02_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://www.youtube.com/watch?v=5-Bb84eTTqQ (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_abhishek_gupta.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 2: Robot Control & Markov Decision Processes

Oier Mees - ETH zurich, Microsoft - 23.02.2026

## Slide 1: Feedback -> Actions (frame 2)

- Extended deadline for 1st homework to March 5th
- Created a moodle forum for QA
- Linked Course Website in MyStudies
- Made slides & recordings available
  - login: (blanked out in the recording)
  - pwd: (blanked out in the recording)
- Instructions for Paper Discussion

[Note: slide 2 does not appear in the captured frames; the numbering jumps from 1 to 3.]

## Slide 3: Robot Motion (frame 3)

- Can be represented as **rigid-body motion**

| Planar Motion | Spatial Motion |
|---|---|
| [Figure: a car seen from above at position $(x, y)$ with heading angle $\theta$] | [Figure: a quadcopter with axes $x, y, z$ and rotations roll (about $x$), pitch (about $y$), yaw (about $z$)] |
| SE(2): The special Euclidean group of rigid body motion in 2D | SE(3): The special Euclidean group of rigid body motion in 3D |

## Slide 4: Mathematical Representation (frame 4)

**SE(2)** - 3 "Degrees of Freedom"

$$T_{SE(2)} = \begin{bmatrix} \cos\theta & -\sin\theta & x \\ \sin\theta & \cos\theta & y \\ 0 & 0 & 1 \end{bmatrix}$$

**SE(3)** - 6 "Degrees of Freedom"

$$T_{SE(3)} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & x \\ r_{21} & r_{22} & r_{23} & y \\ r_{31} & r_{32} & r_{33} & z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$SO(n) = \{R \in \mathbb{R}^{m \times m} \mid R^T R = I, \det(R) = 1\}$$

[Note: the dimension should match the group, $R \in \mathbb{R}^{n \times n}$; the slide writes $m \times m$.]

## Slide 5: Chaining Transforms (frame 5)

[Figure: a Waymo self-driving car with three coordinate frames drawn in: Base Frame (on the car body), Sensor Frame (lidar on the roof) and Pedestrian Frame (on a pedestrian in a lidar point cloud, boxed "Pedestrian detected"). Dotted arrows label the transforms $T^{\text{Base}}_{\text{Sensor}}$, $T^{\text{Sensor}}_{\text{Ped}}$ and $T^{\text{Base}}_{\text{Ped}}$.]

$$T^{\text{Base}}_{\text{Ped}} = T^{\text{Base}}_{\text{Sensor}} \times T^{\text{Sensor}}_{\text{Ped}}$$

## Slide 6: Articulated Robot Bodies (frame 6)

- Robots can be defined as articulated rigid bodies, consisting of:
  - **Link**: a single rigid body
  - **Joint**: connection between links
  - **End-Effector**: a device attached to a specific link, like end of arms

[Figure: kinematic-chain diagram of a humanoid next to a photo of the robot, captioned "LOLA Humanoid Robot @ TUM".]

## Slide 7: Robot Arm Articulations (frame 7)

- **Franka Emika** [Figure: the arm with links 0-7 coloured, and the same arm with joints 1-7 marked, annotated "spherical shoulder", "elbow offsets", "joint axes 5 & 6 intercept", "non-spherical wrist"]
- **SO-101** [Figure: the arm in simulation with a coordinate frame and joint angle $\theta_i$ at each joint; a photo of the arm with joints labelled shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper]

## Slide 8: Robot End Effectors (frame 8)

[Figure: three photos:] Suction Gripper - Parallel Gripper - Dexterous Hand

## Slide 9: Robot Joints (frame 9)

- A joint provides constraints on the possible motions of the two rigid bodies it connects.

[Figure: line drawings of six joint types: Revolute (R), Prismatic (P), Helical (H), Cylindrical (C), Universal (U), Spherical (S).]

Source: Modern Robotics by K. Lynch 2017

## Slide 10: Degrees of Freedom (frame 10)

- Number of independent parameters required to completely specify the configuration or state of the robot

| Joint type | dof $f$ | Constraints $c$ between two planar rigid bodies | Constraints $c$ between two spatial rigid bodies |
|---|---|---|---|
| Revolute (R) | 1 | 2 | 5 |
| Prismatic (P) | 1 | 2 | 5 |
| Helical (H) | 1 | N/A | 5 |
| Cylindrical (C) | 2 | N/A | 4 |
| Universal (U) | 2 | N/A | 4 |
| Spherical (S) | 3 | N/A | 3 |

Chebychev-Grübler-Kutzbach Formula:

$$\text{dof} = \underbrace{m(N - 1)}_{\text{rigid body freedoms}} - \underbrace{\sum_{i=1}^{J} c_i}_{\text{joint constraints}}$$

Source: Modern Robotics by K. Lynch 2017

## Slide 11: Configuration Space (frame 11)

- Set of all possible robot configurations

| system | topology | sample representation |
|---|---|---|
| point on a plane | $\mathbb{E}^2$ | $\mathbb{R}^2$ |
| spherical pendulum | $S^2$ | $[-180°, 180°) \times [-90°, 90°]$ (longitude, latitude) |
| 2R robot arm | $T^2 = S^1 \times S^1$ | $[0, 2\pi) \times [0, 2\pi)$ |
| rotating sliding knob | $\mathbb{E}^1 \times S^1$ | $\mathbb{R}^1 \times [0, 2\pi)$ |

[Figure: each row also shows a sketch of the system, its topology (plane, sphere, torus, cylinder) and the flat chart with its edges identified.]

Source: Modern Robotics by K. Lynch 2017

## Slide 12: Workspace (frame 12)

- All points in space reachable by the robot's end-effector

[Figure: side and top views of an industrial arm's working range, shaded: side-view dimensions 580, 580, 411, 112 and 982 (mm); top view with $\pm 165°$ base rotation, "R121 Minimum turning radius axis 1" and R169.4.]

Source: ABB, Technical Data for The IRB 120 Industrial Robot

## Slide 13: Obstacles in Configuration Space (frame 13)

[Figure: left, "An obstacle in the robot's workspace": a two-link arm with joint angles $\alpha$ (base) and $\beta$ (elbow), a circular obstacle, the arm tip at A and a target point B. Right, "Configuration Space representation of this obstacle": a plot of $\alpha$ (0-180) against $\beta$ (0-360) in which the obstacle becomes a large irregular blue region, with the configurations $q_A$ and $q_B$ marked as points.]

Source: Robot Motion Planning by JC Latombe 1991

## Slide 14: Motion Planning in C-Space (frame 14)

- C-Space used for robot motion planning
- Decouples robot geometry from path planning

[Figure: a free region $\mathcal{C}_{free}$ bounded by obstacle regions $\mathcal{C}_{obs}$, with a curved path from start $q_I$ to goal $q_G$ that avoids them.]

Source: Planning Algorithms by Steven M. LaValle 2006

## Slide 15: Task Space (frame 15)

- The manifold in which the robot's task is naturally defined, independent of its embodiment
- If Task Space dimension < Robot's DoF = Redundancy

[Figure: robot arms on mobile bases cleaning a long whiteboard wall; a robot vacuum on a floor.]

- Task: whiteboard cleaning - Task Space: whiteboard surface $\mathbb{R}^2$
- Task: clean room floor - Task Space: vacuum robot's position on the floor $\mathbb{R}^2$

## Slide 16: Question Time! (frames 16-17)

[Figure: a 7-joint robot arm, captioned "7 DoF".]

| Configuration Space? | Workspace? | Task Space? |
|---|---|---|
| 7D Manifold $(\theta_1 \dots \theta_7)$ | Reachable physical space $(x, y, z)$ | 6D Pose of end-effector SE(3) |

[Second build (frame 17) replaces the arm photo:]

- Redundancy -> Null Space Motion

[Figure: side and top views of the 7-DoF arm's working range in millimetres: reach 855 to each side and 1190 up, 360 below the base; top view a full circle of radius 855.]

## Slide 17: Forward Kinematics (frame 18)

- Given robot joints, what is the end-effector pose?
- Mapping from C to Task Space, $f: \mathcal{C} \to \mathcal{X}$
- $x = f(q)$, $q \in \mathbb{R}^n$, $x \in SE(3)$ deterministic, but not bijective

[Figure: a planar three-link arm with joint angles $\theta_1, \theta_2, \theta_3$ and an unknown end-effector position $\vec{r}_e = ?$]

## Slide 18: Inverse Kinematics (frame 19)

- Given end-effector pose, what are the joint configurations?
- $q = f^{-1}(x)$, mapping $f^{-1}: \mathcal{X} \to \mathcal{P}(\mathcal{C})$

[Figure: the same arm with the end-effector position $\vec{r}_e$ given and $\theta_1 = ?$, $\theta_2 = ?$, $\theta_3 = ?$ unknown.]

## Slide 19: Inverse Kinematics [non-uniqueness] (frame 20)

- Often **non-unique or has no analytical solution**
- Optimization-based IKs

[Figure: two different joint configurations of the three-link arm that reach the same end-effector point $\vec{r}_e$, joined by a double arrow with a question mark.]

## Slide 20: Optimization-Based IK (frame 21)

- **Objective:** Minimize the distance between the current end-effector and the target
- **Loss:** $L(\theta) = \frac{1}{2}\|\underbrace{f(\theta) - x_{\text{target}}}_{e}\|^2$
- **Update:** $\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla_\theta L(\theta)$
- **Gradient of the Loss:** $\nabla_\theta L(\theta) = \left(\frac{\partial f}{\partial \theta}\right)^T e$ (arrow at $\frac{\partial f}{\partial \theta}$: Jacobian)
- **Jacobian Method:** $\theta_{\text{new}} = \theta_{\text{old}} - \alpha J(\theta)^T (f(\theta) - x_{\text{target}})$

## Slide 21: Converting IK to Motor Commands (frame 22)

- IK gives the target joints, but robot can't "jump" there
- Time for movement?

[Figure: the SO-101 arm with an arrow "Move 1cm in Y (0,1,0)".]

## Slide 22: Trajectory Waypoint Generation: LERP (frame 23)

- Inputs: $q_{\text{start}}, q_{\text{target}} \in \mathbb{R}^6$, time T, control frequency f
- Number of Waypoints $N = \lceil T \cdot f \rceil$
- Normalized Time Progress $s_i = \frac{i}{N}$, $s_i \in [0, 1]$
- Linear Interpolation $q(s_i) = q_{\text{start}} + s_i \cdot (q_{\text{target}} - q_{\text{start}})$

[Figure: small SO-101 arm with the arrow "Move 1cm in Y (0,1,0)".]

## Slide 23: Issues with LERP (frame 24)

[Figure: "LERP Trajectory Profiles (1s Move)", three stacked plots against Time (s) from -0.2 to 1.2. Position $q(t)$ (m) ramps linearly from 0 at t=0 to 1 at t=1. Velocity $\dot{q}(t)$ (m/s) jumps from 0 to 1 at t=0 and back to 0 at t=1. Acceleration $\ddot{q}(t)$ (m/s$^2$) is zero except a spike of about +150 at t=0 and about -150 at t=1.]

- [arrow at the velocity step] **Instant Max Speed!** Real motors can't reach full speed in zero time
- [arrow at the acceleration spike] **High Jerk, Dirac Deltas** Stress on actuators, vibration

## Slide 24: Trajectory Waypoint Generation: Quintic Splines (frame 25)

- $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$
- 6x Boundary Conditions for Smoothness:
  - Start & End with no velocity and accel. while keeping position
- $q(s_i) = q_{\text{start}} + (q_{\text{target}} - q_{\text{start}}) \cdot f(s_i)$, $s_i \in [0, 1]$
- Quintic Time Scaling $f(s_i) = 10 s_i^3 - 15 s_i^4 + 6 s_i^5$

## Slide 25: Trajectory Waypoint Generation: Quintic Splines [profiles] (frame 26)

- Standard for industry robots like Frankas

[Figure: "Quintic Spline Trajectory Profiles (1s Move)", three stacked plots against Time (s) from -0.2 to 1.2. Position $q(t)$: smooth S-curve from 0 to 1 between t=0 and t=1. Velocity $\dot{q}(t)$: smooth bell, zero at both ends, peak about 1.9 m/s at t=0.5. Acceleration $\ddot{q}(t)$: smooth, zero at both ends, about +5.8 m/s$^2$ near t=0.2 and about -5.8 near t=0.8. Compare with the LERP spikes on slide 23.]

## Slide 26: PID Control (frame 27)

- Given trajectory waypoints $q(s_i)$ and current measurement, continuously calculate corrective motor commands $(u_k)$
- Tracking error $e(t) = q_{desired}(t) - q_{measured}(t)$
- Continuous form $u(t) = K_p e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d \frac{de(t)}{dt}$
- Discrete form

$$u_k = \underbrace{K_p e_k}_{\text{Proportional}} + \underbrace{K_i \sum_{j=0}^{k} (e_j \cdot \Delta t)}_{\text{Integral}} + \underbrace{K_d \frac{e_k - e_{k-1}}{\Delta t}}_{\text{Derivative}}$$

## Slide 27: PID Control [intuition] (frame 28)

$$u_k = K_p e_k + K_i \sum_{j=0}^{k} (e_j \cdot \Delta t) + K_d \frac{e_k - e_{k-1}}{\Delta t}, \qquad \Delta t = 1/f$$

- **Proportional** - "Spring": The bigger the tracking error, the harder the robot pulls
- **Integral** - "Anti-Gravity": Helps compensate if gravity is too much
- **Derivative** - "Damper": Stops the arm from from overshooting past a waypoint
- $\Delta t = 1/f$: If frequency too low, "Damper" might not be able to react fast enough

[Figure: a robot arm tracking a desired path (green) through waypoints $q(s_0), q(s_1), q(s_2), \dots, q(s_N)$; the actual path (red) wobbles around it. Purple arrows from red to green are labelled "Proportional pull (P)"; a sag below the path is labelled "Gravity (needs Integral)"; the gap between the curves is "Error $e_k$"; a bump past the path is "Overshoot (needs Derivative)".]

## Slide 28: [untitled: transition question] (frame 29)

**So we know how to plan & execute robot motion are we done?**

Not quite...

## Slide 29: Robot Learning (frame 30)

Solving **Robotics** via **Machine Learning**

- Robotics: Perception, Control, ...
- Machine Learning: Imitation Learning, Reinforcement Learning, Dynamics Learning, Representation Learning, ...

## Slide 30: Sequential Decision Making (frames 31-36)

- What is the best sequence of actions to cut the sushi?

[Figure: embedded video playing across six frames: a two-armed robot workstation with a bowl, a wooden box with a red drawer, a red cup holding utensils, a tray, a cutting board and a toy sushi roll. The arms move the roll onto the board, take a red toy knife from the cup, hand it between grippers and slice the roll into pieces.]

Reference: *The Ingredients for Robotic Diffusion Transformers*, Dasari, **Mees** et. al., ICRA 2024

[Note: the paper appeared at ICRA 2025, as week 6 cites it, not ICRA 2024.]

## Slide 31: From Pixels to Decisions: Representing a Policy (frame 37)

[Figure: two parallel pipelines through the same convolutional network diagram.]

- Image classification: input $x$ (photo of a kitten in a hat) -> $f(y \mid x)$ -> output $y$: Cat 0.8, Tiger 0.0, Bear 0.0, Dog 0.2
- Policy: observation $o$ (view through a car windscreen of a snowy road) -> $\pi_\theta(a \mid o)$ -> action $a$: Left 0.0, Right 0.0, Straight 0.8, Backward 0.2

## Slide 32: Mapping Observations to Actions (frame 38)

[Figure: observation $o_t$ (snowy road through the windscreen) -> convolutional network -> action distribution $a_t$ (Left 0.0, Right 0.0, Straight 0.8, Backward 0.2), with an arrow from $a_t$ back to $o_t$ closing the loop.]

- $\pi_\theta(a_t \mid o_t)$ - Policy - Partially Observable
- $\pi_\theta(a_t \mid s_t)$ - Policy - Fully Observable
- $a_t$ - action; $o_t$ - observation; $s_t$ - state

## Slide 33: State vs Observation (frames 39-40)

[Figure: a car icon labelled "$s_t$ - state", an arrow to a road photo labelled "$o_t$ - observation", and a reverse arrow crossed out in red. The second build swaps the clear-day road for a dark, snowy night view.]

No matter the weather or visibility, the **state does not change**!

## Slide 34: Markov Property (frame 41)

- If you know $S_2$ then $S_1$ not necessary to determine $S_3$

[Figure: graphical model. States $s_1 \to s_2 \to s_3$ along the bottom, the arrows labelled **Transition Function** $p(s_{t+1} \mid s_t, a_t)$. Each state emits an observation ($s_1 \to o_1$ labelled **Lossy Mapping**, $s_2 \to o_2$); each observation maps to an action through $\pi_\theta$ ($o_1 \to a_1$, $o_2 \to a_2$); $a_1$ feeds into $s_2$. Portrait captioned "Andrey Markov".]

## Slide 35: Alternative Notations (frame 42)

| RL notation (Richard Bellman) | Control notation (Lev Pontryagin) |
|---|---|
| $a_t$ - action | $u_t$ - action |
| $s_t$ - state | $x_t$ - state |
| $r(s, a)$ - reward | $c(x, u)$ - cost function |

$$r(s, a) = -c(x, u)$$

[Figure: portraits of Richard Bellman and Lev Pontryagin.]

## Slide 36: Markov Decision Process (frame 43)

- Define by $M = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$

[Figure: loop between the environment (globe) and the robot (arm): the environment sends "state $s_t$, reward $r_t$" to the robot, the robot sends "action $a_t$" back.]

- $\mathcal{S}$: State space, $s_t \in \mathcal{S}$
- $\mathcal{A}$: Action space, $a_t \in \mathcal{A}$
- $\mathcal{P}$: Transition probability, $s_{t+1} \sim \mathcal{P}(\cdot|, s_t, a_t)$
- $\mathcal{R}$: Reward function, $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$, $r_t = R(s_t, a_t, s_{t+1})$

[Note: the signature $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ and the form $R(s_t, a_t, s_{t+1})$ disagree; both conventions are common, the slide mixes them.]

## Slide 37: MDP: State Space (frame 44)

- Discrete [Figure: a roulette wheel; a chess board]
- Continuous [Figure: the SO-101 robot arm; a quadcopter drone]

## Slide 38: MDP: Action Space (frame 45)

- Discrete [Figure: a hand of playing cards; keyboard arrow keys]
- Continuous [Figure: a joystick; hands on a steering wheel]

## Slide 39: MDP: Transition Model (frame 46)

- Uncertainty caused by sensor noise, motor overshooting...

| Deterministic | Stochastic |
|---|---|
| $s_{t+1} = f(s_t, a_t)$ | $s_{t+1} \sim \mathcal{P}(\cdot\vert, s_t, a_t)$ |
| Action: turn steering wheel 5° | Action: turn steering wheel 5° |
| [Figure: a car on a dry road with one curved arrow showing where it goes] | [Figure: the same car on an icy patch with many arrows fanning out] |

## Slide 40: MDP: Reward Function (frames 47-48)

- How to detect if reward conditions are met? [added in the second build]

| Sparse Rewards | Dense Rewards |
|---|---|
| e.g. 1 if grasped, 0 otherwise | e.g. distance to object |
| [Figure: video of a robot arm grasping objects on a cork table] | [Figure: video of the same arm reaching for objects, with two camera insets labelled "Selected affordance region" and "Detected affordance region center"] |

Reference: *Affordance Learning from Play for Sample-Efficient Policy Learning*, Borja, **Mees** et. al., ICRA 2022

## Slide 41: Trajectory Probability (frame 49)

- Probability of a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ under policy $\pi$:
- $p_\pi(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$
  - [braces under the three factors] **Initial State Distribution** ($\rho(s_0)$), **Policy** ($\pi$), **Transition Function** ($P$)

[Figure: several trajectories fanning out from a start state $s_0$ to end states $s_{T_1}, \dots, s_{T_4}$; one highlighted as a solid path.]

## Slide 42: Finite-Horizon vs Infinite-Horizon MDPs (frame 50)

- **Finite-Horizon**: the task has a strict time limit/steps $H$
- **Infinite-Horizon**: interaction continues forever $(T = \infty)$or until a "terminal state" is reached
  - **Discount Factor** $\gamma$: Rewards closer in time are higher weighted

[Figure: a clock face that spirals into smaller and smaller copies of itself.]

## Slide 43: The Learning Objective (frame 51)

- Accumulated reward an agent receives: $G = \sum_{t=0}^{T-1} r_t$ or $G = \sum_{t \ge 0} \gamma^t r_t$
- Expected Return: $J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t=0}^{T-1} r_t\right]$ or $J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t \ge 0} \gamma^t r_t\right]$
- Optimal Policy: $\pi^* = \arg\max_\pi J(\pi)$

[Figure: the same trajectory fan from $s_0$.]

## End (frame 52)

Thank you for your attention
