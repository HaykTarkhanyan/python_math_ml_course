# A cross-chapter callback must be checked against the deck, never against the plan

**Symptom.** L40 (JEPA world models) was drafted with three frames resting on the claim that
chapter 11 had already taught latent world models - Ha and Schmidhuber, Dreamer, TD-MPC. Frame
22 was billed in the chapter plan as *"one sentence that retroactively explains the whole of
L39"*. An outline review flagged the callback as false.

**Cause.** The callback was written from memory of my own chapter plan, which listed ch11 as a
prerequisite for "model-based RL, planning, imagination". Nobody opened `ml/ch11_rl/`.

The check takes one command:

```
$ grep -rniE "dreamer|td-mpc|schmidhuber|cross-entropy method|world model|\bMPC\b|\bCEM\b" ml/ch11_rl/*.tex
   (no output)
```

Zero hits - not just for the model names, but for the words **world model**, **MPC** and **CEM**
anywhere in the chapter. ch11 covers MDPs, value iteration, Q-learning, DQN, policy gradients,
actor-critic, PPO and RLHF. Its single model-based frame is titled **"What we just did was
cheating"** and defines model-based as *"You know `P(s'|s,a)` and `R`. Plan by computing"* - a
**given** transition table. No latent state, no learned dynamics, no reconstruction-free
prediction.

So the payoff frame ("in model-based RL the latent cannot collapse because it must also predict
reward and value") was comparing JEPA against something the room had never seen, and the frame
that was supposed to retroactively explain the previous lecture landed on nothing.

**Consequences.**

- The material was rewritten as **explicitly new**, with a bridge from what ch11 does teach:
  "chapter 11 gave you `P(s'|s,a)`; this line of work asks what happens when you must learn it,
  and learn it in a latent space." MPC and the cross-entropy method are now introduced rather
  than recalled, and frames were budgeted for that.
- The same review pass caught a second instance: L39's section transition read *"You have
  already built two of them"*, but `ml/ch12_vlm/L33_vlm_seeing.tex` carries the line
  `NO MODEL IS TRAINED in this chapter (instructor decision)`. Changed to "seen".

**This is the third consecutive session with an error of the same shape** - characterising a file
by its title or its plan entry instead of opening it. The 2026-08-07 session log records two
others: a chapter thesis aimed at a deck that never makes the claim being attacked, and a frame
"defending" ch06 against an accusation ch06 had already made itself with numbers.

**The rule that follows:** a callback is a factual claim about another file. Before writing
"as you saw in chN", grep chN for the term. If the grep is empty, it is not a callback - it is
new material, and it needs the frames new material needs.
