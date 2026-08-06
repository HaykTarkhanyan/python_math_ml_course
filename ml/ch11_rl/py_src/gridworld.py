"""Gridworld figures for L32 (Reinforcement Learning).

The running example for the whole deck: a 4x4 stochastic gridworld, solved by value
iteration. NOTHING IS TRAINED - value iteration on a fully specified MDP is deterministic
arithmetic, so every number the slides quote is exact and reproducible.

Generates into ml/ch11_rl/fig/:
  gridworld_layout.pdf  -- the grid, rewards, wall, and the slip model
  gridworld_values.pdf  -- V*(s) heatmap from value iteration
  gridworld_policy.pdf  -- the greedy policy arrows implied by V*

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/gridworld.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

# ---- the MDP, fully specified ---------------------------------------------------------
N = 4
GOAL = (0, 3)          # +1, terminal
PIT = (1, 3)           # -1, terminal
WALL = (1, 1)          # impassable
STEP_COST = -0.04      # every non-terminal move
GAMMA = 0.9
SLIP = 0.1             # probability of veering to each side of the intended direction

ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ARROW = {"up": (0, 0.3), "down": (0, -0.3), "left": (-0.3, 0), "right": (0.3, 0)}


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_gridworld.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def states():
    return [(r, c) for r in range(N) for c in range(N)
            if (r, c) != WALL and (r, c) not in (GOAL, PIT)]


def step(state, drow, dcol):
    """Where you land trying to move by (drow, dcol). Bumping a wall or edge keeps you put."""
    nr, nc = state[0] + drow, state[1] + dcol
    if not (0 <= nr < N and 0 <= nc < N) or (nr, nc) == WALL:
        return state
    return (nr, nc)


def transitions(state, action):
    """(next_state, probability) for the slip model: intended with 1-2*SLIP, sides with SLIP."""
    drow, dcol = ACTIONS[action]
    perpendicular = [(-dcol, -drow), (dcol, drow)]        # the two 90-degree turns
    out = [(step(state, drow, dcol), 1 - 2 * SLIP)]
    out += [(step(state, pr, pc), SLIP) for pr, pc in perpendicular]
    return out


def value_iteration(tol=1e-10, max_sweeps=10_000):
    V = {s: 0.0 for s in states()}
    V[GOAL], V[PIT] = 1.0, -1.0
    for sweep in range(max_sweeps):
        delta = 0.0
        for s in states():
            best = max(
                sum(p * V[s2] for s2, p in transitions(s, a)) for a in ACTIONS
            )
            new = STEP_COST + GAMMA * best
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < tol:
            return V, sweep + 1
    raise RuntimeError(f"value iteration did not converge in {max_sweeps} sweeps (delta={delta})")


def greedy_policy(V):
    policy = {}
    for s in states():
        policy[s] = max(ACTIONS, key=lambda a: sum(p * V[s2] for s2, p in transitions(s, a)))
    return policy


# ---- drawing ---------------------------------------------------------------------------
def draw_grid(ax):
    for r in range(N + 1):
        ax.plot([0, N], [r, r], color="0.7", lw=1, zorder=1)
        ax.plot([r, r], [0, N], color="0.7", lw=1, zorder=1)
    ax.set_xlim(-0.05, N + 0.05)
    ax.set_ylim(-0.05, N + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


def cell_xy(state):
    """Centre of a cell in plot coordinates (row 0 drawn at the top)."""
    r, c = state
    return c + 0.5, (N - 1 - r) + 0.5


def fig_layout():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    draw_grid(ax)

    for state, color, label in [(GOAL, "#2e7d32", "+1"), (PIT, RED, "-1")]:
        x, y = cell_xy(state)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.75, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=19, color="white",
                fontweight="bold", zorder=3)

    x, y = cell_xy(WALL)
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.35", zorder=2))
    ax.text(x, y, "wall", ha="center", va="center", fontsize=10, color="white", zorder=3)

    x, y = cell_xy((3, 0))
    ax.text(x, y, "start", ha="center", va="center", fontsize=11, color=BLUE, fontweight="bold")

    ax.set_title(f"Every move costs {STEP_COST}. Reach +1, avoid -1.\n"
                 f"Actions slip sideways with probability {SLIP} each.", fontsize=11)
    fig.tight_layout()
    out = FIG / "gridworld_layout.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_values(V):
    cmap = LinearSegmentedColormap.from_list("rl", [RED, "white", "#2e7d32"])
    vals = np.array([[V.get((r, c), np.nan) for c in range(N)] for r in range(N)])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    finite = vals[~np.isnan(vals)]
    limit = max(abs(finite.min()), abs(finite.max()))
    ax.imshow(vals, cmap=cmap, vmin=-limit, vmax=limit, extent=(0, N, 0, N), zorder=1)
    draw_grid(ax)

    for r in range(N):
        for c in range(N):
            if (r, c) == WALL:
                x, y = cell_xy((r, c))
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.35", zorder=2))
                continue
            x, y = cell_xy((r, c))
            ax.text(x, y, f"{V[(r, c)]:.2f}", ha="center", va="center", fontsize=13, zorder=3)

    ax.set_title("$V^*(s)$ after value iteration\n(green = worth being here, red = not)", fontsize=11)
    fig.tight_layout()
    out = FIG / "gridworld_values.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_policy(V, policy):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    draw_grid(ax)

    for state, color, label in [(GOAL, "#2e7d32", "+1"), (PIT, RED, "-1")]:
        x, y = cell_xy(state)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, alpha=0.75, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=17, color="white",
                fontweight="bold", zorder=3)
    x, y = cell_xy(WALL)
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.35", zorder=2))

    for state, action in policy.items():
        x, y = cell_xy(state)
        dx, dy = ARROW[action]
        ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.16, head_length=0.13,
                 fc=BLUE, ec=BLUE, lw=2.2, zorder=3, length_includes_head=True)

    ax.set_title("The greedy policy $\\pi^*(s) = \\arg\\max_a Q^*(s,a)$\n"
                 "read straight off the values", fontsize=11)
    fig.tight_layout()
    out = FIG / "gridworld_policy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    log = build_logger()
    np.random.seed(SEED)
    FIG.mkdir(exist_ok=True)

    V, sweeps = value_iteration()
    policy = greedy_policy(V)
    log.info(f"value iteration converged in {sweeps} sweeps (gamma={GAMMA}, slip={SLIP})")

    # numbers the slides quote - printed so the deck never has to guess
    start_v = V[(3, 0)]
    near_pit = V[(0, 2)]        # directly beside the goal, but also one slip from the pit
    log.info(f"V*(start=(3,0))     = {start_v:.4f}")
    log.info(f"V*((0,2)) beside +1 = {near_pit:.4f}")
    log.info(f"V*((2,3)) below pit = {V[(2, 3)]:.4f}")
    log.info(f"policy at (2,3)     = {policy[(2, 3)]}  <- detours AWAY from the pit")

    if policy[(2, 3)] == "up":
        raise RuntimeError(
            "policy at (2,3) points 'up' straight into the pit; the predict-first frame "
            "depends on it detouring. Check SLIP / STEP_COST.")

    for out in (fig_layout(), fig_values(V), fig_policy(V, policy)):
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
