"""Dynamic-programming figures for L32 (deck 1: The RL problem).

Policy iteration and value iteration are the same algorithm with one knob: how many
evaluation sweeps you run before re-greedifying. This script measures that on the
gridworld from gridworld.py, so the slide can quote sweep counts instead of asserting
"policy iteration needs fewer iterations".

Generates into ml/ch11_rl/fig/:
  gpi_sweeps.pdf   -- error vs sweeps for 3 settings of the knob, + sweeps-to-optimal bars
  gpi_diagram.pdf  -- the generalised policy iteration picture

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/dynamic_programming.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import (ACTIONS, GAMMA, GOAL, PIT, STEP_COST, greedy_policy, states,
                       transitions, value_iteration)

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

TOL = 1e-10
MAX_SWEEPS = 400


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_dynamic_programming.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def fresh_values():
    V = {s: 0.0 for s in states()}
    V[GOAL], V[PIT] = 1.0, -1.0
    return V


def evaluation_sweep(V, policy):
    """One in-place sweep of iterative policy evaluation. Returns the largest change."""
    delta = 0.0
    for s in states():
        new = STEP_COST + GAMMA * sum(p * V[s2] for s2, p in transitions(s, policy[s]))
        delta = max(delta, abs(new - V[s]))
        V[s] = new
    return delta


def improve(V):
    return {s: max(ACTIONS, key=lambda a: sum(p * V[s2] for s2, p in transitions(s, a)))
            for s in states()}


def gpi(eval_sweeps, V_star, pi_star):
    """Generalised policy iteration with a fixed number of evaluation sweeps per round.

    eval_sweeps=1        -> value iteration
    eval_sweeps=3        -> modified policy iteration
    eval_sweeps=math.inf -> classic policy iteration (evaluate to convergence)

    Returns per-sweep history of (max error vs V*, whether greedy(V) == pi*).
    """
    V = fresh_values()
    policy = {s: "up" for s in states()}          # deliberately bad start
    errors, optimal_at = [], None

    total = 0
    while total < MAX_SWEEPS:
        for _ in range(eval_sweeps):
            delta = evaluation_sweep(V, policy)
            total += 1
            err = max(abs(V[s] - V_star[s]) for s in states())
            errors.append(err)
            if optimal_at is None and greedy_policy(V) == pi_star:
                optimal_at = total
            if delta < TOL:
                break
            if total >= MAX_SWEEPS:
                break

        new_policy = improve(V)
        stable = new_policy == policy
        policy = new_policy
        if stable and delta < TOL:
            break

    if optimal_at is None:
        raise RuntimeError(
            f"GPI with eval_sweeps={eval_sweeps} never reached the optimal policy in "
            f"{MAX_SWEEPS} sweeps")
    return errors, optimal_at, total


def fig_sweeps(runs, log):
    """Left: the policy is optimal long before the values are. Right: the knob."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.6, 4.2),
                                     gridspec_kw={"width_ratios": [1.35, 1]})

    colors = {"value iteration\n(1 sweep)": BLUE,
              "modified PI\n(3 sweeps)": ORANGE,
              "policy iteration\n(evaluate fully)": RED}

    vi_label = "value iteration\n(1 sweep)"
    errors, optimal_at, total = runs[vi_label]
    ax_l.semilogy(np.arange(1, len(errors) + 1), errors, color=BLUE, lw=2.4)
    ax_l.axvspan(optimal_at, total, color=BLUE, alpha=0.10)
    ax_l.axvline(optimal_at, color=RED, ls="--", lw=1.6)

    ax_l.annotate(f"greedy policy is already $\\pi^*$\nhere, at sweep {optimal_at}",
                  xy=(optimal_at, errors[optimal_at - 1]),
                  xytext=(optimal_at + 1.6, errors[0] * 0.55),
                  fontsize=9, color=RED,
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax_l.text((optimal_at + total) / 2, errors[-1] * 60,
              f"{total - optimal_at} more sweeps\nspent polishing numbers\n"
              f"that change no decision",
              fontsize=8.5, color=BLUE, ha="center")

    ax_l.set_xlabel("evaluation sweeps")
    ax_l.set_ylabel("$\\max_s |V(s) - V^*(s)|$")
    ax_l.set_title("Value iteration on the gridworld", fontsize=11)
    ax_l.spines[["top", "right"]].set_visible(False)
    ax_l.grid(alpha=0.25, which="both")

    names = list(runs.keys())
    y = np.arange(len(names))
    to_optimal = [runs[n][1] for n in names]
    to_converged = [runs[n][2] for n in names]

    b1 = ax_r.barh(y + 0.19, to_converged, height=0.36, color=GREY, alpha=0.55,
                   label="until the values converge")
    b2 = ax_r.barh(y - 0.19, to_optimal, height=0.36,
                   color=[colors[n] for n in names], alpha=0.95,
                   label="until the policy is optimal")
    ax_r.bar_label(b1, fmt="%d", padding=3, fontsize=9, color="0.3")
    ax_r.bar_label(b2, fmt="%d", padding=3, fontsize=9)

    ax_r.set_yticks(y)
    ax_r.set_yticklabels(names, fontsize=8.5)
    ax_r.set_xlim(0, max(to_converged) * 1.2)
    ax_r.set_xlabel("evaluation sweeps")
    ax_r.set_title("One knob: how long you evaluate\nbefore re-greedifying", fontsize=11)
    ax_r.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_r.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIG / "gpi_sweeps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_diagram():
    """The GPI picture: evaluation pushes onto one line, improvement onto the other."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.plot([0.05, 0.99], [0.18, 0.63], color=BLUE, lw=2.0)
    ax.text(1.02, 0.60, "$V = V^{\\pi}$", color=BLUE, fontsize=13, ha="left", va="top")
    ax.text(1.02, 0.545, "values honest\nabout the policy", color=BLUE, fontsize=8,
            ha="left", va="top")
    ax.text(0.50, 0.30, "evaluation", color=BLUE, fontsize=9.5, ha="center", rotation=24)

    ax.plot([0.05, 0.99], [0.92, 0.67], color=RED, lw=2.0)
    ax.text(1.02, 0.76, "$\\pi = \\mathrm{greedy}(V)$", color=RED, fontsize=13,
            ha="left", va="bottom")
    ax.text(1.02, 0.755, "policy greedy\nabout the values", color=RED, fontsize=8,
            ha="left", va="top")
    ax.text(0.42, 0.855, "improvement", color=RED, fontsize=9.5, ha="center", rotation=-12)

    # the zig-zag between the two lines, converging on their intersection
    pts = [(0.10, 0.88), (0.12, 0.21), (0.38, 0.81), (0.42, 0.35),
           (0.62, 0.76), (0.66, 0.47), (0.78, 0.72), (0.82, 0.56), (0.88, 0.685)]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=GREY, lw=1.4, ls="-", marker="o", ms=3.4, alpha=0.85, zorder=4)

    ax.annotate("", xy=(0.945, 0.648), xytext=(0.88, 0.685),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    ax.plot([0.962], [0.641], "*", color="black", ms=15, zorder=6)
    ax.text(0.955, 0.575, "$\\pi^*, V^*$", fontsize=12, ha="center", va="top")

    ax.text(0.03, 0.06,
            "Every method in this chapter bounces between these two lines.\n"
            "They differ only in how far they go each time.",
            fontsize=9, color="0.25")

    ax.set_xlim(0, 1.42)
    ax.set_ylim(0, 1.02)
    ax.axis("off")
    ax.set_title("Generalised policy iteration", fontsize=12, color=BLUE, loc="left")

    fig.tight_layout()
    out = FIG / "gpi_diagram.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    np.random.seed(SEED)
    FIG.mkdir(exist_ok=True)

    V_star, vi_sweeps = value_iteration()
    pi_star = greedy_policy(V_star)
    log.info(f"reference: value iteration converged in {vi_sweeps} sweeps")

    runs = {}
    for label, k in [("value iteration\n(1 sweep)", 1),
                     ("modified PI\n(3 sweeps)", 3),
                     ("policy iteration\n(evaluate fully)", MAX_SWEEPS)]:
        errors, optimal_at, total = gpi(k, V_star, pi_star)
        runs[label] = (errors, optimal_at, total)
        log.info(f"{label.replace(chr(10), ' '):>34s}: optimal policy at sweep "
                 f"{optimal_at:3d}, converged after {total} sweeps")

    # The frame's claim: the knob trades sweeps-per-round against rounds, and every
    # setting lands on the same policy. If that ever stops being true, fail loudly.
    finals = [runs[k][1] for k in runs]
    if len(set(finals)) == 1:
        log.warning("all three settings reached the optimal policy on the same sweep; "
                    "the comparison panel will look flat")

    for out in (fig_sweeps(runs, log), fig_diagram()):
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
