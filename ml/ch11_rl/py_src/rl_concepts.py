"""Concept figures for L32 (Reinforcement Learning).

Everything here is exact arithmetic or sampling from a *specified* distribution.
NO AGENT IS TRAINED anywhere in this chapter (instructor decision, RL_CHAPTER_PLAN.md).

Generates into ml/ch11_rl/fig/:
  discount_horizon.pdf   -- gamma^k vs k: how far ahead each discount lets you see
  gamma_policy.pdf       -- the SAME MDP solved at three discounts, three different policies
  epsilon_cost.pdf       -- exact value of an eps-greedy policy vs eps (policy evaluation)
  reinforce_variance.pdf -- spread of the REINFORCE gradient estimate, with and without a baseline
  ppo_clip.pdf           -- the PPO clipped objective as a function of the probability ratio

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/rl_concepts.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import (ACTIONS, ARROW, GAMMA, GOAL, N, PIT, STEP_COST, WALL, cell_xy,
                       draw_grid, states, transitions)

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_concepts.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


# ---- shared solvers (exact, no learning) ----------------------------------------------
def value_iteration(gamma, tol=1e-12, max_sweeps=100_000):
    V = {s: 0.0 for s in states()}
    V[GOAL], V[PIT] = 1.0, -1.0
    for sweep in range(max_sweeps):
        delta = 0.0
        for s in states():
            best = max(sum(p * V[s2] for s2, p in transitions(s, a)) for a in ACTIONS)
            new = STEP_COST + gamma * best
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < tol:
            return V
    raise RuntimeError(f"value iteration did not converge at gamma={gamma}")


def greedy_policy(V, gamma):
    return {s: max(ACTIONS, key=lambda a: sum(p * V[s2] for s2, p in transitions(s, a)))
            for s in states()}


def evaluate_epsilon_greedy(policy, eps, gamma, tol=1e-12, max_sweeps=100_000):
    """Exact value of the policy that follows `policy` w.p. 1-eps and acts uniformly w.p. eps.

    This is policy EVALUATION, not learning: the policy is fixed and known, so the value is
    the solution of a linear system, reached here by repeated sweeps.
    """
    V = {s: 0.0 for s in states()}
    V[GOAL], V[PIT] = 1.0, -1.0
    n_actions = len(ACTIONS)
    for _ in range(max_sweeps):
        delta = 0.0
        for s in states():
            total = 0.0
            for a in ACTIONS:
                prob = (1 - eps) * (1.0 if a == policy[s] else 0.0) + eps / n_actions
                if prob == 0.0:
                    continue
                total += prob * sum(p * V[s2] for s2, p in transitions(s, a))
            new = STEP_COST + gamma * total
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < tol:
            return V
    raise RuntimeError(f"policy evaluation did not converge at eps={eps}")


# ---- figures ---------------------------------------------------------------------------
def fig_discount_horizon():
    k = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for gamma, color in [(0.5, RED), (0.9, BLUE), (0.99, ORANGE)]:
        ax.plot(k, gamma ** k, color=color, lw=2.2, label=f"$\\gamma={gamma}$")
        half = np.log(0.5) / np.log(gamma)
        ax.axvline(half, color=color, ls=":", lw=1.2, alpha=0.8)
        ax.text(half + 0.9, 0.94, f"half-life\n{half:.0f} steps", color=color, fontsize=10,
                va="top", fontweight="bold")
    ax.set_xlabel("steps into the future, $k$")
    ax.set_ylabel("weight $\\gamma^k$")
    ax.set_title("How far ahead the agent can see is a choice, not a fact")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "discount_horizon.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_gamma_policy(log):
    """A purpose-built corridor MDP where the discount decides the destination.

    The 4x4 gridworld is the wrong world for this: its policy barely changes with gamma
    (measured: 1-3 of 12 squares, and non-monotonically), because there is only one good
    outcome to walk towards. Myopia needs a CHOICE between a near small reward and a far
    large one, so this figure uses a corridor built for exactly that.
    """
    # Geometry chosen so the discount actually flips the decision. The crossover is where
    # gamma^(big_steps-1) * big_r == small_r, i.e. gamma = (small_r/big_r)^(1/(big_steps-1)).
    length, start = 12, 4
    small_pos, small_r = 3, 1.0        # one step to the left
    big_pos, big_r = 11, 10.0          # seven steps to the right

    def corridor_values(gamma):
        """Exact: at each cell, compare walking left to the small reward vs right to the big one."""
        best_action, values = {}, {}
        for pos in range(length):
            if pos in (small_pos, big_pos):
                continue
            left_steps, right_steps = abs(pos - small_pos), abs(big_pos - pos)
            left_val = gamma ** (left_steps - 1) * small_r if pos > small_pos else -np.inf
            right_val = gamma ** (right_steps - 1) * big_r if pos < big_pos else -np.inf
            values[pos] = max(left_val, right_val)
            best_action[pos] = "left" if left_val >= right_val else "right"
        return values, best_action

    gammas = [0.5, 0.8, 0.95]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 4.6))
    for ax, gamma in zip(axes, gammas):
        values, action = corridor_values(gamma)
        for pos in range(length):
            face = ("#2e7d32" if pos == big_pos else ORANGE if pos == small_pos else "white")
            ax.add_patch(plt.Rectangle((pos, 0), 1, 1, facecolor=face, edgecolor="0.6",
                                       alpha=0.9 if pos in (small_pos, big_pos) else 1.0))
        ax.text(small_pos + 0.5, 0.5, f"+{small_r:.0f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(big_pos + 0.5, 0.5, f"+{big_r:.0f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(start + 0.5, 1.28, "start", ha="center", fontsize=8, color=BLUE)
        for pos, a in action.items():
            dx = 0.3 if a == "right" else -0.3
            ax.arrow(pos + 0.5 - dx / 2, 0.5, dx, 0, head_width=0.16, head_length=0.14,
                     fc=BLUE, ec=BLUE, lw=1.8, length_includes_head=True)
        ax.set_xlim(-0.2, length + 0.2)
        ax.set_ylim(-0.1, 1.6)
        ax.axis("off")
        goes = action[start]
        ax.text(-0.1, 0.5, f"$\\gamma={gamma}$", ha="right", va="center", fontsize=11)
        ax.text(length + 0.35, 0.5, f"from start: go {goes}", ha="left", va="center",
                fontsize=9, color="#2e7d32" if goes == "right" else ORANGE)
        log.info(f"corridor gamma={gamma}: from start go {goes} "
                 f"(left worth {gamma ** (start - small_pos - 1) * small_r:.3f}, "
                 f"right worth {gamma ** (big_pos - start - 1) * big_r:.3f})")

    crossover = (small_r / big_r) ** (1 / (big_pos - start - 1))
    fig.suptitle(f"Same world, same rewards - the discount decides which one is worth walking to\n"
                 f"(the decision flips at $\\gamma \\approx {crossover:.2f}$)", fontsize=11.5)
    fig.tight_layout()
    out = FIG / "gamma_policy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_epsilon_cost(log):
    V_star = value_iteration(GAMMA)
    policy = greedy_policy(V_star, GAMMA)
    eps_grid = np.linspace(0, 1, 41)
    values = [evaluate_epsilon_greedy(policy, e, GAMMA)[(3, 0)] for e in eps_grid]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(eps_grid, values, color=BLUE, lw=2.4)
    ax.axhline(V_star[(3, 0)], color="0.5", ls=":", lw=1.2)
    ax.annotate(f"$\\epsilon=0$: {values[0]:.3f}\n(pure greedy)", xy=(0, values[0]),
                xytext=(0.09, values[0] - 0.16), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate(f"$\\epsilon=1$: {values[-1]:.3f}\n(pure random)", xy=(1, values[-1]),
                xytext=(0.6, values[-1] + 0.22), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("$\\epsilon$  (probability of acting at random)")
    ax.set_ylabel("value of the start state")
    ax.set_title("What exploration COSTS, once you already know the best policy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "epsilon_cost.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"eps-greedy value: eps=0 -> {values[0]:.4f}, eps=0.1 -> "
             f"{values[4]:.4f}, eps=1 -> {values[-1]:.4f}")
    return out


def fig_reinforce_variance(log):
    """Spread of the REINFORCE gradient estimate on a SPECIFIED return distribution.

    Not a training run: returns are drawn from a distribution we write down, to show what the
    estimator does to them. The only thing being illustrated is variance.
    """
    rng = np.random.default_rng(SEED)
    n_episodes, n_repeats = 20, 4000

    # One state, two actions, pi(a1) = p. With a sigmoid policy the score function is
    # grad log pi(a1) = 1 - p  and  grad log pi(a2) = -p. It DEPENDS ON THE ACTION - which is
    # exactly why a baseline helps. (A fixed score function would make the baseline useless:
    # subtracting a constant would shift the mean and leave the variance untouched.)
    p = 0.5
    score = {0: 1 - p, 1: -p}
    means = {0: 11.0, 1: 9.0}                # BOTH returns positive: action 0 is the better one
    noise = 1.0
    baseline = 10.0                          # the average return

    chosen = rng.random(size=(n_repeats, n_episodes)) > p          # False -> a1, True -> a2
    returns = np.where(chosen, means[1], means[0]) + rng.normal(0, noise, chosen.shape)
    scores = np.where(chosen, score[1], score[0])

    plain = (scores * returns).mean(axis=1)
    with_baseline = (scores * (returns - baseline)).mean(axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bins = np.linspace(min(plain.min(), with_baseline.min()),
                       max(plain.max(), with_baseline.max()), 60)
    ax.hist(plain, bins=bins, color=RED, alpha=0.65, label=f"REINFORCE  (sd {plain.std():.2f})")
    ax.hist(with_baseline, bins=bins, color=BLUE, alpha=0.75,
            label=f"with baseline  (sd {with_baseline.std():.2f})")
    ax.axvline(0, color="0.35", lw=1, ls=":")
    ax.set_xlabel("gradient estimate from a batch of 20 episodes")
    ax.set_ylabel("count")
    ax.set_title("Subtracting a baseline does not move the average - it shrinks the scatter")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG / "reinforce_variance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    wrong_plain = (plain < 0).mean() * 100
    wrong_base = (with_baseline < 0).mean() * 100
    log.info(f"REINFORCE spread: plain sd={plain.std():.3f}, with baseline sd={with_baseline.std():.3f}, "
             f"means {plain.mean():.3f} vs {with_baseline.mean():.3f}")
    log.info(f"gradient points the WRONG WAY: {wrong_plain:.1f}% without baseline, "
             f"{wrong_base:.1f}% with  <- the number the slide quotes")
    return out


def fig_ppo_clip():
    ratio = np.linspace(0, 2.2, 400)
    eps = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=False)
    for ax, adv, color, title in [(axes[0], +1.0, "#2e7d32", "advantage $\\hat{A} > 0$  (good action)"),
                                  (axes[1], -1.0, RED, "advantage $\\hat{A} < 0$  (bad action)")]:
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
        objective = np.minimum(unclipped, clipped)
        ax.plot(ratio, unclipped, color="0.65", lw=1.4, ls="--", label="unclipped $r\\hat{A}$")
        ax.plot(ratio, objective, color=color, lw=2.6, label="PPO objective")
        ax.axvspan(1 - eps, 1 + eps, color=ORANGE, alpha=0.16)
        ax.axvline(1.0, color="0.4", lw=1, ls=":")
        ax.set_xlabel("probability ratio $r_t(\\theta)$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("objective")
    fig.suptitle(f"The clip ($\\epsilon={eps}$) removes the incentive to move far from the old policy",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "ppo_clip.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    outs = [fig_discount_horizon(), fig_gamma_policy(log), fig_epsilon_cost(log),
            fig_reinforce_variance(log), fig_ppo_clip()]
    for out in outs:
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
