"""Tabular Q-learning that ACTUALLY LEARNS, on the deck's own gridworld.

Everything else in this chapter is exact planning: value iteration needs P(s'|s,a), which the
lecture spends a whole section explaining you do not have. This script closes that loop - an
agent that has never been told the rules learns Q from sampled experience alone, and we check
it against the V* that value iteration computed.

The agent may ONLY call step_env(), which samples one transition. It never sees the transition
probabilities. That restriction is the whole point, and it is enforced by construction here.

Generates into ml/ch11_rl/fig/:
  qlearning_convergence.pdf -- error to V* vs episodes, plus the policy it ends up with

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/q_learning_demo.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import (ACTIONS, ARROW, GAMMA, GOAL, PIT, STEP_COST, WALL, cell_xy, draw_grid,
                       greedy_policy, states, transitions, value_iteration)

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

EPISODES = 8000
MAX_STEPS = 60          # a wandering episode is cut off; the grid is 4x4
EPS_START, EPS_END = 1.0, 0.05
ACTION_LIST = list(ACTIONS)

# Learning rate DECAYS per (s,a) with its visit count: alpha_n = 1/(1+n)^0.6.
# A constant alpha does not converge here - it keeps chasing the most recent sample, and a
# first run with alpha=0.2 matched pi* at episode 491 and then drifted back off (6 of 13
# squares wrong at the end). This schedule is what the Q-learning slide already promises:
# sum(alpha) = inf, sum(alpha^2) < inf.
ALPHA_EXP = 0.6


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_qlearning_demo.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def step_env(rng, state, action):
    """Sample ONE transition. This is the only thing the agent is allowed to see."""
    outcomes, probs = zip(*transitions(state, action))
    nxt = outcomes[rng.choice(len(outcomes), p=probs)]
    if nxt == GOAL:
        return nxt, 1.0, True
    if nxt == PIT:
        return nxt, -1.0, True
    return nxt, STEP_COST, False


def q_learning(rng, v_star, pi_star):
    Q = {(s, a): 0.0 for s in states() for a in ACTION_LIST}
    visits = {(s, a): 0 for s in states() for a in ACTION_LIST}
    non_terminal = states()

    errors, mismatches = [], []
    first_match = None

    for ep in range(EPISODES):
        eps = EPS_END + (EPS_START - EPS_END) * (1 - ep / EPISODES)
        # exploring starts: begin anywhere. Without this the far corners are visited so
        # rarely that convergence there is dominated by luck rather than by the algorithm.
        state = non_terminal[rng.choice(len(non_terminal))]

        for _ in range(MAX_STEPS):
            if rng.random() < eps:
                action = ACTION_LIST[rng.choice(len(ACTION_LIST))]
            else:
                action = max(ACTION_LIST, key=lambda a: Q[(state, a)])

            nxt, reward, done = step_env(rng, state, action)
            future = 0.0 if done else max(Q[(nxt, a)] for a in ACTION_LIST)
            visits[(state, action)] += 1
            alpha = 1.0 / (visits[(state, action)] ** ALPHA_EXP)
            Q[(state, action)] += alpha * (reward + GAMMA * future - Q[(state, action)])

            if done:
                break
            state = nxt

        # how far is the agent's value estimate from the exact answer?
        v_hat = {s: max(Q[(s, a)] for a in ACTION_LIST) for s in non_terminal}
        errors.append(max(abs(v_hat[s] - v_star[s]) for s in non_terminal))

        pi_hat = {s: max(ACTION_LIST, key=lambda a: Q[(s, a)]) for s in non_terminal}
        wrong = sum(pi_hat[s] != pi_star[s] for s in non_terminal)
        mismatches.append(wrong)
        if wrong == 0 and first_match is None:
            first_match = ep + 1

    return Q, np.array(errors), np.array(mismatches), first_match


def fig_convergence(errors, mismatches, Q, pi_star, first_match, log):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    ep = np.arange(1, len(errors) + 1)
    ax.plot(ep, errors, color=BLUE, lw=1.3, label="max $|V_Q(s) - V^*(s)|$")
    ax.set_xscale("log")
    ax.set_xlabel("episodes of experience (log scale)")
    ax.set_ylabel("worst-state error")
    ax.grid(alpha=0.3)
    if first_match:
        ax.axvline(first_match, color=ORANGE, ls="--", lw=1.6)
        ax.text(first_match * 1.15, ax.get_ylim()[1] * 0.80,
                f"policy first matches\n$\\pi^*$ everywhere\n(episode {first_match})",
                color=ORANGE, fontsize=8.5, fontweight="bold")

    # The curve flattens well above zero. That floor is the story, so label it.
    floor = float(np.mean(errors[-500:]))
    ax.axhline(floor, color=RED, ls=":", lw=1.6)
    ax.text(1.3, floor + 0.035, f"it stops improving at ~{floor:.2f}, not 0",
            color=RED, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("It never saw $P(s'|s,a)$ - only sampled transitions", fontsize=10.5)

    ax = axes[1]
    draw_grid(ax)
    for state, color in [(GOAL, "#2e7d32"), (PIT, RED)]:
        x, y = cell_xy(state)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.75, zorder=2))
    x, y = cell_xy(WALL)
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.35", zorder=2))

    n_wrong = 0
    for s in states():
        learned = max(ACTION_LIST, key=lambda a: Q[(s, a)])
        col = BLUE if learned == pi_star[s] else RED
        n_wrong += learned != pi_star[s]
        x, y = cell_xy(s)
        dx, dy = ARROW[learned]
        ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.16, head_length=0.13,
                 fc=col, ec=col, lw=2.2, zorder=3, length_includes_head=True)
    ax.set_title(f"The policy it learned\n({len(states()) - n_wrong}/{len(states())} squares "
                 f"match the exact answer)", fontsize=10.5)

    fig.tight_layout()
    out = FIG / "qlearning_convergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"final policy disagreements with pi*: {n_wrong}")
    return out


def main():
    log = build_logger()
    rng = np.random.default_rng(SEED)
    FIG.mkdir(exist_ok=True)

    v_star, sweeps = value_iteration()
    pi_star = greedy_policy(v_star)
    log.info(f"reference: value iteration converged in {sweeps} sweeps")

    Q, errors, mismatches, first_match = q_learning(rng, v_star, pi_star)

    log.info(f"episodes run: {EPISODES}, alpha=1/n^{ALPHA_EXP}, eps {EPS_START} -> {EPS_END}")
    log.info(f"worst-state error: start {errors[0]:.3f} -> end {errors[-1]:.4f}")
    log.info(f"policy first matched pi* everywhere at episode {first_match}")
    log.info(f"squares still disagreeing at the end: {mismatches[-1]} of {len(states())}")

    # The residual error is NOT noise - it is maximization bias, and it is one-sided.
    # max_a Q(s',a) over noisy estimates systematically overshoots, which is exactly why
    # Double Q-learning (van Hasselt, 2010) exists. Report it rather than tune it away.
    v_hat = {s: max(Q[(s, a)] for a in ACTION_LIST) for s in states()}
    signed = [v_hat[s] - v_star[s] for s in states()]
    n_over = sum(d > 0 for d in signed)
    log.info(f"signed error: {n_over}/{len(signed)} states OVERestimated, "
             f"mean {np.mean(signed):+.4f}, max {max(signed):+.4f}  <- maximization bias")

    # What must hold for the slide's claims to be true.
    if mismatches[-1] > 1:
        raise RuntimeError(
            f"{mismatches[-1]} squares disagree with pi*; the slide claims at most one.")
    pit_neighbour = (2, 3)
    learned_there = max(ACTION_LIST, key=lambda a: Q[(pit_neighbour, a)])
    if learned_there == "up":
        raise RuntimeError(
            "the agent learned to walk straight past the pit from (2,3). The predict-first "
            "frame's whole point is that this is the wrong move - do not ship this run.")
    if n_over < len(signed) * 0.7:
        raise RuntimeError(
            f"only {n_over}/{len(signed)} states overestimated; the slide attributes the "
            "residual error to one-sided maximization bias, which this run does not show.")

    out = fig_convergence(errors, mismatches, Q, pi_star, first_match, log)
    log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
