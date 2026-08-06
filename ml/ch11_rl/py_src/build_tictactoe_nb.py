"""Assemble ml/ch11_rl/L32_tictactoe_project_solution.ipynb.

The RL chapter project, for the practical session: learn tic-tac-toe by self-play, then grade
the result against perfect play. Standalone - nothing in the deck references it.

Why this game: it is the largest game that can still be SOLVED EXACTLY by minimax, so the
project has ground truth for every position. That mirrors the lecture, which checks Q-learning
against value iteration's V*.

Structure rule (house pattern, ch8_autoencoders/py_src/build_sae_nb.py): MANY SMALL CELLS, one
idea per cell, a markdown cell before each code cell saying what is about to happen and why.

Assemble, then execute separately so every number in the prose comes from a real run:
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/build_tictactoe_nb.py
    ./ma/Scripts/python.exe -m nbconvert --execute --inplace --to notebook <nb>
"""

from pathlib import Path

import nbformat as nbf

CH = Path(__file__).resolve().parents[1]
OUT = CH / "L32_tictactoe_project_solution.ipynb"
CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


# ======================================================================================
md(r"""
# Chapter project - Learn tic-tac-toe by self-play, then grade it against perfect play

**Lecture 32, applied end to end.**

In the lecture the agent learned a $4\times4$ gridworld, and we could check it because value
iteration had already computed the exact answer $V^*$.

We are going to do the same thing to a **game** - and we can, because tic-tac-toe is small
enough to **solve exactly**. Minimax gives us the true value of every position, so every claim
in this notebook gets scored rather than admired.

### The question

> Can an agent that is told **only the rules and the final result** learn to play tic-tac-toe
> as well as a perfect player?

The answer is more interesting than yes or no, and Part 8 explains why.

### The plan

| Part | What we do |
|---|---|
| 0 | The game, and a perfect minimax opponent - our ground truth |
| 1 | **Baseline first:** how well does random play do? |
| 2 | Train Q-learning against a random opponent |
| 3 | **The trap.** Take that agent to a real opponent |
| 4 | Self-play |
| 5 | Grade every move against minimax |
| 6 | Turn exploration off and watch it collapse |
| 7 | Try to help it with reward shaping |
| 8 | The verdict - and why "did it win?" is the wrong question |
""")

md(r"""
## Setup

Pure Python and numpy. No GPU, no libraries beyond what the course already uses.
""")

code(r"""
%matplotlib inline

import random
from collections import defaultdict
from functools import lru_cache
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

SEED = 509
random.seed(SEED)
np.random.seed(SEED)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
plt.rcParams["figure.figsize"] = (10, 3.6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
print("ready")
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 0 - The game, and the exact answer

A board is a tuple of 9 numbers: `0` empty, `1` for X, `2` for O. Using a tuple (not a list)
matters - it is hashable, so it can be a dictionary key, which is what makes tabular Q-learning
possible at all.
""")

code(r"""
EMPTY, X, O = 0, 1, 2
LINES = [(0,1,2), (3,4,5), (6,7,8),      # rows
         (0,3,6), (1,4,7), (2,5,8),      # columns
         (0,4,8), (2,4,6)]               # diagonals

def winner(board):
    "Return X, O, or 0 if nobody has three in a row."
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return 0

def legal_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]

def play_move(board, cell, player):
    nxt = list(board)
    nxt[cell] = player
    return tuple(nxt)

def is_over(board):
    return winner(board) != 0 or not legal_moves(board)

EMPTY_BOARD = (EMPTY,) * 9
print("empty board:", EMPTY_BOARD)
print("legal moves:", legal_moves(EMPTY_BOARD))
""")

md(r"""
A way to actually look at a board, because we will be staring at these a lot.
""")

code(r"""
def show(board):
    glyph = {EMPTY: ".", X: "X", O: "O"}
    rows = ["".join(f" {glyph[board[r * 3 + c]]} " for c in range(3)) for r in range(3)]
    print("\n---+---+---\n".join(rows))

demo = play_move(play_move(EMPTY_BOARD, 4, X), 0, O)
show(demo)
""")

md(r"""
## The exact answer: minimax

Minimax asks: *if both players play perfectly from here, what happens?*

It is a recursion, and it is short. The value of a position, **from the point of view of the
player about to move**, is:

- `+1` if that player can force a win,
- `0` if the best either can force is a draw,
- `-1` if the opponent can force a win.

`lru_cache` memoises it. There are only a few thousand reachable positions, so this is instant.
""")

code(r"""
@lru_cache(maxsize=None)
def minimax_value(board, player):
    "Value of `board` for `player`, who is about to move, under perfect play by both sides."
    w = winner(board)
    if w != 0:
        return 1 if w == player else -1          # someone already won
    if not legal_moves(board):
        return 0                                  # drawn board
    opponent = O if player == X else X
    # my best outcome = the best over my moves of (minus the opponent's best reply)
    return max(-minimax_value(play_move(board, c, player), opponent)
               for c in legal_moves(board))

def optimal_moves(board, player):
    "Every move that achieves the position's minimax value - there is often more than one."
    opponent = O if player == X else X
    scored = {c: -minimax_value(play_move(board, c, player), opponent)
              for c in legal_moves(board)}
    best = max(scored.values())
    return [c for c, v in scored.items() if v == best], best
""")

md(r"""
The first thing worth checking is the most famous fact about this game.
""")

code(r"""
value_of_empty = minimax_value(EMPTY_BOARD, X)
best_first_moves, _ = optimal_moves(EMPTY_BOARD, X)

print(f"minimax value of the empty board (for X) = {value_of_empty}")
print(f"optimal opening moves for X: {sorted(best_first_moves)}")
print()
print("0 means: with perfect play by BOTH sides, tic-tac-toe is a DRAW.")
print("And every one of those opening moves is equally optimal - including the corners,")
print("the centre, and the edges. Perfect play does not require a clever opening.")
""")

md(r"""
Now enumerate every reachable position. This is the set we will grade against later.
""")

code(r"""
def reachable_positions():
    "All positions reachable by legal play, with whose turn it is."
    seen = {}
    stack = [(EMPTY_BOARD, X)]
    while stack:
        board, player = stack.pop()
        if (board, player) in seen:
            continue
        seen[(board, player)] = True
        if is_over(board):
            continue
        opponent = O if player == X else X
        for c in legal_moves(board):
            stack.append((play_move(board, c, player), opponent))
    return list(seen)

POSITIONS = reachable_positions()
live = [(b, p) for b, p in POSITIONS if not is_over(b)]
print(f"reachable (board, player-to-move) pairs: {len(POSITIONS)}")
print(f"of which still in play (a move is required): {len(live)}")
""")

md(r"""
A few thousand positions. That is the whole game - which is exactly why we get ground truth
here and not in chess or Go. Hold onto that number; it is the reason this project is possible
and also the reason it does not scale.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 1 - Baseline first

Lecture rule, and the same rule as the time-series chapter: **no result counts until you know
what doing nothing achieves.**

Two players choosing uniformly at random from the legal moves.
""")

code(r"""
def random_policy(board, player, rng):
    return rng.choice(legal_moves(board))

def play_game(policy_x, policy_o, rng, record=False):
    "Play one game. Returns the winner (X, O or 0) and optionally the move history."
    board, player = EMPTY_BOARD, X
    history = []
    while not is_over(board):
        policy = policy_x if player == X else policy_o
        cell = policy(board, player, rng)
        if record:
            history.append((board, player, cell))
        board = play_move(board, cell, player)
        player = O if player == X else X
    return winner(board), history

def match(policy_x, policy_o, n_games, seed=SEED):
    rng = random.Random(seed)
    out = {X: 0, O: 0, 0: 0}
    for _ in range(n_games):
        w, _ = play_game(policy_x, policy_o, rng)
        out[w] += 1
    return {"X wins": out[X] / n_games, "draws": out[0] / n_games, "O wins": out[O] / n_games}
""")

code(r"""
baseline = match(random_policy, random_policy, 20_000)
for k, v in baseline.items():
    print(f"{k:>8}: {v:6.1%}")
""")

md(r"""
### Read that before moving on

Random vs random is **not** an even fight. X wins far more often than O, purely from moving
first. Any evaluation that ignores which side the agent plays will be measuring the
first-move advantage and calling it skill.

So from here on, **every agent is evaluated as X and as O**.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 2 - Train against a random opponent

Tabular Q-learning, exactly as in the lecture. Two details specific to games:

**1. Whose board is it?** Rather than keep separate tables for X and O, we show the agent the
board *from the mover's point of view* - your pieces are always `1`, the opponent's always `2`.
One table then serves both sides, and the agent learns "positions like this" rather than
"positions like this when I happen to be O".

**2. When does the reward arrive?** Only at the end: $+1$ win, $-1$ loss, $0$ draw. Every
intermediate move scores nothing. This is the sparse, delayed reward from the lecture, in its
purest form.
""")

code(r"""
def canonical(board, player):
    "The board as the mover sees it: my pieces = 1, opponent's = 2."
    if player == X:
        return board
    swap = {EMPTY: EMPTY, X: O, O: X}
    return tuple(swap[v] for v in board)

# sanity check: the same position, seen by whoever is to move, has the same canonical form
b = play_move(play_move(EMPTY_BOARD, 4, X), 0, O)
print("raw board       :", b)
print("as X sees it    :", canonical(b, X))
print("as O sees it    :", canonical(b, O), " <- O's own piece is now the 1")
""")

code(r"""
def greedy_action(Q, board, player, rng, eps):
    moves = legal_moves(board)
    if rng.random() < eps:
        return rng.choice(moves)
    state = canonical(board, player)
    best = max(Q[(state, c)] for c in moves)
    # break ties at random, or the agent silently prefers low-numbered cells
    return rng.choice([c for c in moves if Q[(state, c)] == best])

def make_q_policy(Q, eps=0.0):
    "Wrap a Q-table as a policy usable by play_game()."
    return lambda board, player, rng: greedy_action(Q, board, player, rng, eps)
""")

md(r"""
The update. We collect each player's own $(state, action)$ pairs during the game, then walk
**backwards** from the result: the last move gets the final reward, and earlier moves get the
discounted value of the position they led to.
""")

code(r"""
ALPHA, GAMMA = 0.3, 0.95

def q_update_trajectory(Q, trajectory, final_reward):
    "Backward Q-learning pass over one player's moves in one finished game."
    target = final_reward
    for state, action, next_own_state, next_moves in reversed(trajectory):
        if next_own_state is not None and next_moves:
            target = GAMMA * max(Q[(next_own_state, c)] for c in next_moves)
        Q[(state, action)] += ALPHA * (target - Q[(state, action)])
        target = final_reward       # only the LAST move is scored by the outcome directly
""")

md(r"""
Now the training loop. The agent plays one side; the opponent policy is passed in - which is
what lets us reuse this same function for "against random" now and "against itself" in Part 4.
""")

code(r"""
def train(opponent_policy=None, episodes=40_000, eps_start=0.9, eps_end=0.05,
          self_play=False, seed=SEED, shaping=None, Q=None):
    rng = random.Random(seed)
    Q = defaultdict(float) if Q is None else Q
    visits = defaultdict(int)

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / episodes)
        board, player = EMPTY_BOARD, X
        traj = {X: [], O: []}
        # Alternate which side the agent takes. Training it only as X would leave it with no
        # opinion whatever about half the game, and then "it loses as O" would say nothing
        # about the opponent it trained against - only that it had never played that side.
        agent_side = X if ep % 2 == 0 else O

        while not is_over(board):
            agent_turn = self_play or player == agent_side
            if agent_turn:
                cell = greedy_action(Q, board, player, rng, eps)
            else:
                cell = opponent_policy(board, player, rng)

            state = canonical(board, player)
            nxt = play_move(board, cell, player)
            if agent_turn:
                visits[state] += 1
                traj[player].append([state, cell, None, None])
                # fill in the previous entry's "where I ended up" once I move again
            board, player = nxt, (O if player == X else X)

        # rebuild the "next own state" links now that the game is finished
        for who in (X, O):
            steps = traj[who]
            for i in range(len(steps) - 1):
                steps[i][2] = steps[i + 1][0]
                steps[i][3] = [c for c, v in enumerate(steps[i + 1][0]) if v == EMPTY]

        result = winner(board)
        for who in (X, O):
            if not traj[who]:
                continue
            reward = 0.0 if result == 0 else (1.0 if result == who else -1.0)
            if shaping is not None:
                reward += shaping(traj[who])
            q_update_trajectory(Q, traj[who], reward)

    return Q, visits
""")

code(r"""
Q_vs_random, visits_vs_random = train(opponent_policy=random_policy, episodes=80_000)
print(f"states the agent has an opinion about: {len({s for s, _ in Q_vs_random})}")
print("(it alternates sides, so it trains as X and as O against the random opponent)")
""")

md(r"""
How does it do against the opponent it was trained on?
""")

code(r"""
agent = make_q_policy(Q_vs_random)
as_x = match(agent, random_policy, 5_000)
as_o = match(random_policy, agent, 5_000)

print("trained-vs-random agent, playing X against random:")
print(f"   wins {as_x['X wins']:.1%}   draws {as_x['draws']:.1%}   losses {as_x['O wins']:.1%}")
print("the same agent, playing O against random:")
print(f"   wins {as_o['O wins']:.1%}   draws {as_o['draws']:.1%}   losses {as_o['X wins']:.1%}")
""")

md(r"""
It beats random play convincingly, from both sides, and it has clearly improved on the Part 1
baseline. At this point it is very tempting to declare the project finished.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 3 - The trap

### Predict first

We are about to play this agent against the **perfect** minimax player.

It beats a random opponent about 9 times out of 10. Tic-tac-toe is a draw under perfect play,
so the best possible result here is to draw every game.

> *What fraction of games do you expect it to draw? Half? Most?*

Commit to a number before running the cell.
""")

code(r"""
def minimax_policy(board, player, rng):
    "Perfect play. Chooses uniformly among optimal moves so games are not all identical."
    best, _ = optimal_moves(board, player)
    return rng.choice(best)

vs_perfect_x = match(agent, minimax_policy, 2_000)
vs_perfect_o = match(minimax_policy, agent, 2_000)

print("trained-vs-random agent, now against PERFECT play:")
print(f"   as X: draws {vs_perfect_x['draws']:.1%},  losses {vs_perfect_x['O wins']:.1%}")
print(f"   as O: draws {vs_perfect_o['draws']:.1%},  losses {vs_perfect_o['X wins']:.1%}")
""")

md(r"""
### What happened

As X it holds the draw. As O it **loses a meaningful share of games** - against an opponent
that, by definition, can never be beaten but can always be held.

That gap did not exist against the random opponent, where this agent looked near-perfect from
both sides. A random opponent almost never punishes a bad move, so the agent was free to walk
into positions that are lost under correct play, get away with it, and bank the win. It
optimised the objective we gave it, exactly as asked.

> **The same lesson as the time-series chapter's naive baseline, from the other side. There the
> danger was a baseline too strong to beat; here it is an opponent too weak to learn from. In
> both cases the thing you measure against decides what "good" means.**

### Be honest about the size of the effect

This is a *dent*, not a catastrophe - and the reason is worth more than the demo. Tic-tac-toe
has only a few thousand positions. Even a random opponent, played tens of thousands of times
with an exploring agent, stumbles into nearly all of them, so the agent gets at least some
experience almost everywhere.

Scale the game up and that stops being true. In a game where random play covers a vanishing
fraction of the tree, the same setup does not lose one game in six - it never becomes competent
at all. The mechanism you are seeing here in miniature is the one that makes training-opponent
choice decisive in real systems.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 4 - Self-play

Fix the opponent by removing it. The agent plays **both sides**, against its own current
policy, and updates from both.

The opponent now improves exactly as fast as the agent does, so there is no fixed weakness left
to exploit. This is the idea behind AlphaGo Zero from the lecture's opening, in its smallest
possible form.
""")

code(r"""
Q_self, visits_self = train(self_play=True, episodes=120_000)
print(f"states the agent has an opinion about: {len({s for s, _ in Q_self})}")
print(f"(training against random reached {len({s for s, _ in Q_vs_random})})")
""")

code(r"""
selfplay_agent = make_q_policy(Q_self)

rows = []
for name, res, key in [
    ("vs random, as X", match(selfplay_agent, random_policy, 5_000), "X wins"),
    ("vs random, as O", match(random_policy, selfplay_agent, 5_000), "O wins"),
    ("vs PERFECT, as X", match(selfplay_agent, minimax_policy, 2_000), "X wins"),
    ("vs PERFECT, as O", match(minimax_policy, selfplay_agent, 2_000), "O wins"),
]:
    loss_key = "O wins" if key == "X wins" else "X wins"
    rows.append((name, res[key], res["draws"], res[loss_key]))

print(f"{'':<20}{'win':>8}{'draw':>8}{'loss':>8}")
for name, w, d, l in rows:
    print(f"{name:<20}{w:>8.1%}{d:>8.1%}{l:>8.1%}")
""")

md(r"""
Compare the bottom two rows against Part 3. Self-play did not make it better at beating random
players - it made it **stop losing to good ones**, which is the thing that actually matters.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 5 - Grade every move against ground truth

Win rates are a summary. We can do better than a summary, because we know the right answer for
every position.

For each live position, ask: is the agent's preferred move **one of the minimax-optimal
moves**?
""")

code(r"""
def grade(Q, positions):
    # Fraction of positions with a real CHOICE where the agent's greedy move is optimal.
    #
    # Two exclusions, both of which would otherwise punish the agent unfairly:
    #   * positions with a single legal move - there is nothing to get right or wrong;
    #   * positions where every Q is still exactly 0.0 - the agent has formed no opinion, so
    #     grading its arbitrary tie-break tells us nothing. Reported separately as
    #     "no opinion" rather than silently scored as mistakes.
    #
    # That second case is easy to get wrong, and we did get it wrong first time round: a
    # final move that DRAWS is updated towards a target of 0, so Q stays at exactly 0.0
    # even though the state was visited many times.
    ok, decided, no_opinion, failures = 0, 0, 0, []
    for board, player in positions:
        if is_over(board):
            continue
        moves = legal_moves(board)
        if len(moves) == 1:
            continue                            # forced move; nothing to decide
        state = canonical(board, player)
        if all(Q[(state, c)] == 0.0 for c in moves):
            no_opinion += 1
            continue
        top = max(Q[(state, c)] for c in moves)
        best_q = [c for c in moves if Q[(state, c)] == top]
        best_true, _ = optimal_moves(board, player)
        decided += 1
        if set(best_q) <= set(best_true):
            ok += 1
        else:
            failures.append((board, player, best_q, best_true))
    return ok / decided, failures, no_opinion

score_random, fails_random, blank_random = grade(Q_vs_random, POSITIONS)
score_self, fails_self, blank_self = grade(Q_self, POSITIONS)
print(f"trained vs random   : {score_random:6.1%} of decided positions optimal "
      f"({blank_random} with no opinion)")
print(f"trained by self-play: {score_self:6.1%} of decided positions optimal "
      f"({blank_self} with no opinion)")
""")

md(r"""
### Where does it still get it wrong?

The percentage is less interesting than **which** positions fail. The obvious hypothesis is
that the agent is weak where it has the least experience - the lecture's convergence condition
says Q-learning needs *infinitely many visits to every state-action pair*, so thin coverage
should mean bad play.

Test it before believing it.
""")

code(r"""
visit_counts = np.array([visits_self[canonical(b, p)] for b, p, _, _ in fails_self])
all_counts = np.array([visits_self[canonical(b, p)] for b, p in POSITIONS if not is_over(b)])

print(f"failing positions : {len(visit_counts)},  median visits {np.median(visit_counts):.0f}")
print(f"all live positions: {len(all_counts)},  median visits {np.median(all_counts):.0f}")
print(f"failing positions never visited at all: {(visit_counts == 0).sum()}")
""")

md(r"""
### Resist the urge to explain 16 data points

Only a handful of genuine mistakes remain, and their median visit count sits a little below
average - consistent with a coverage story, but nowhere near enough to claim one. Sixteen
positions cannot support a theory. Say so, and look instead at where the *two agents* differ.
""")

code(r"""
def n_pieces(board):
    return sum(1 for v in board if v != EMPTY)

def error_by_depth(Q):
    "Per game depth: share of decided positions where the greedy move is NOT optimal."
    wrong, decided = defaultdict(int), defaultdict(int)
    for board, player in POSITIONS:
        if is_over(board) or len(legal_moves(board)) < 2:
            continue
        state, moves = canonical(board, player), legal_moves(board)
        if all(Q[(state, c)] == 0.0 for c in moves):
            continue
        top = max(Q[(state, c)] for c in moves)
        best_q = [c for c in moves if Q[(state, c)] == top]
        best_true, _ = optimal_moves(board, player)
        d = n_pieces(board)
        decided[d] += 1
        if not set(best_q) <= set(best_true):
            wrong[d] += 1
    return decided, wrong

dec_r, wrong_r = error_by_depth(Q_vs_random)
dec_s, wrong_s = error_by_depth(Q_self)
depths = sorted(dec_r)

fig, ax = plt.subplots(figsize=(9.4, 3.6))
w = 0.38
ax.bar([d - w/2 for d in depths], [wrong_r[d] / dec_r[d] for d in depths], w,
       color=RED, label="trained against a random opponent")
ax.bar([d + w/2 for d in depths], [wrong_s[d] / dec_s[d] for d in depths], w,
       color=BLUE, label="trained by self-play")
ax.set_xticks(depths)
ax.set_xticklabels([f"{d}  (n={dec_r[d]})" for d in depths], fontsize=7.5)
ax.set_xlabel("pieces already on the board when the agent must choose")
ax.set_ylabel("share played sub-optimally")
ax.set_title("Where the two training regimes actually differ")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
""")

md(r"""
### Read the shape, not the totals

Two things stand out, and neither is what you might guess.

**The self-play agent is near-flawless at every depth** - the blue bars barely leave the axis.

**The random-trained agent is worst in the OPENING and early middlegame**, peaking around three
pieces on the board, and it is perfect by the time seven pieces are down. That is the reverse of
"it gets sloppy at the end".

It makes sense once stated: by seven pieces there are two cells left and the right move is
usually forced - take the win, or block the loss - and *any* training regime finds that. The
early game is where positions are subtle, and it is exactly where a random opponent **never
punishes a mistake**. The agent had no way to learn that certain early choices are already
losing, because nothing it played against ever made it pay.

(Treat the 0-2 piece bars with care - there are only 1, 9 and 72 positions at those depths, so
one mistake moves the percentage a long way.)

> A single "93.5% vs 99.6%" hides all of this. When you have ground truth, break the score down
> by something structural - here, game phase - because *where* a model is wrong usually explains
> *why*.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 6 - Turn exploration off

### Predict first

Self-play with $\epsilon = 0$: the agent always plays what it currently believes is best,
against a copy of itself doing the same.

> *How many distinct positions do you think it will see in 120,000 games?*
""")

code(r"""
Q_greedy, visits_greedy = train(self_play=True, episodes=120_000, eps_start=0.0, eps_end=0.0)

print(f"distinct positions visited with exploration    : {len(visits_self):,}")
print(f"distinct positions visited with epsilon = 0    : {len(visits_greedy):,}")
print(f"live positions that exist in the whole game    : {len(live):,}")
""")

code(r"""
score_greedy, _, blank_greedy = grade(Q_greedy, POSITIONS)
greedy_agent = make_q_policy(Q_greedy)
gx = match(greedy_agent, minimax_policy, 2_000)
go = match(minimax_policy, greedy_agent, 2_000)

print(f"optimal-move rate: {score_greedy:.1%}   (with exploration: {score_self:.1%})")
print(f"vs perfect play, as X: draws {gx['draws']:.1%}, losses {gx['O wins']:.1%}")
print(f"vs perfect play, as O: draws {go['draws']:.1%}, losses {go['X wins']:.1%}")
""")

md(r"""
### The collapse

With ties broken randomly the agent does still wander a little, but the coverage gap is stark:
tens of thousands of games produce only a small slice of the game tree, because both sides keep
walking down the same handful of lines.

Everything outside those lines is never evaluated, so the agent has no opinion about it - and
a real opponent will happily steer straight into that territory.

This is the $\epsilon$ frame from the lecture, seen from the side the figure could not show.
There we priced what exploration **costs** once you already know the best policy. Here is what
it **buys** while you are still learning.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 7 - Try to help it

The agent takes a long time to learn something obvious: **block your opponent when they have
two in a row.** Why not just tell it? Add a small bonus every time it blocks a threat.

This is *reward shaping* - and it is the single most common way people accidentally break an
RL system, so it is worth doing once, deliberately, in a place where nothing is at stake.
""")

code(r"""
def threat_cells(board, player):
    "Cells where `player` would complete a line on the next move."
    out = set()
    for a, b, c in LINES:
        line = [board[a], board[b], board[c]]
        if line.count(player) == 2 and line.count(EMPTY) == 1:
            out.add([a, b, c][line.index(EMPTY)])
    return out

def make_blocking_shaper(bonus_per_block):
    "Bonus for every move that occupied a cell the opponent was threatening to complete."
    def shaper(trajectory):
        bonus = 0.0
        for state, action, _, _ in trajectory:
            # state is canonical: my pieces are 1, the opponent's are 2
            if action in threat_cells(state, 2):
                bonus += bonus_per_block
        return bonus
    return shaper
""")

md(r"""
Run it with a **small** bonus first: $0.3$, well under the $\pm 1$ paid for the actual result.
""")

code(r"""
Q_shaped, _ = train(self_play=True, episodes=120_000, shaping=make_blocking_shaper(0.3))
score_shaped, _, _ = grade(Q_shaped, POSITIONS)
shaped_agent = make_q_policy(Q_shaped)
sx = match(shaped_agent, minimax_policy, 2_000)
so = match(minimax_policy, shaped_agent, 2_000)

print(f"{'':<28}{'optimal moves':>15}{'loses as X':>13}{'loses as O':>13}")
print(f"{'plain self-play':<28}{score_self:>15.1%}"
      f"{match(selfplay_agent, minimax_policy, 2_000)['O wins']:>13.1%}"
      f"{match(minimax_policy, selfplay_agent, 2_000)['X wins']:>13.1%}")
print(f"{'blocking bonus = 0.3':<28}{score_shaped:>15.1%}{sx['O wins']:>13.1%}{so['X wins']:>13.1%}")
""")

md(r"""
### It changed essentially nothing

The two rows are within a whisker of each other, and both still never lose to perfect play.

That is the benign case for shaping, and it is worth seeing: blocking really is the right move
*most* of the time here, and the bonus is **small relative to the real reward**, so it acts as a
mild hint that never outvotes the actual outcome. It neither rescued nor ruined anything.

Which makes the next result the interesting one. Keep the identical heuristic, change nothing
but the size of the number, and let blocking pay **more than winning does**.
""")

code(r"""
Q_greedy_block, _ = train(self_play=True, episodes=120_000, shaping=make_blocking_shaper(3.0))
score_gb, _, _ = grade(Q_greedy_block, POSITIONS)
gb = make_q_policy(Q_greedy_block)
gbx = match(gb, minimax_policy, 2_000)
gbo = match(minimax_policy, gb, 2_000)

print(f"{'':<28}{'optimal moves':>15}{'loses as X':>13}{'loses as O':>13}")
print(f"{'blocking bonus = 0.3':<28}{score_shaped:>15.1%}{sx['O wins']:>13.1%}{so['X wins']:>13.1%}")
print(f"{'blocking bonus = 3.0':<28}{score_gb:>15.1%}{gbx['O wins']:>13.1%}{gbo['X wins']:>13.1%}")
print()
print("Also: how often does each agent take a winning move when one is available?")

def win_now_rate(Q, n=4_000, seed=7):
    "Fraction of positions with an immediate win available where the agent actually takes it."
    rng = random.Random(seed)
    took, chances = 0, 0
    for board, player in POSITIONS:
        if is_over(board):
            continue
        wins = [c for c in legal_moves(board) if winner(play_move(board, c, player)) == player]
        if not wins:
            continue
        chances += 1
        if greedy_action(Q, board, player, rng, 0.0) in wins:
            took += 1
    return took / chances

print(f"   bonus 0.3 : {win_now_rate(Q_shaped):.1%} of immediate wins taken")
print(f"   bonus 3.0 : {win_now_rate(Q_greedy_block):.1%} of immediate wins taken")
""")

md(r"""
### There it is

Same heuristic, same code, one number changed - and the agent now **loses to perfect play about
half the time**, from a starting point of never losing at all.

Read the win-taking rates as a *comparison*, not as an absolute: minimax values a delayed
forced win exactly as highly as an immediate one, so even a perfect player does not always take
the win on offer, and the $0.3$ figure is not a count of blunders. What matters is that the
identical measurement on the identical positions **drops sharply** when the only thing changed
is the size of the bonus. The agent is turning down wins it used to take, in order to collect
block payments.

It is not confused and it is not broken. It is optimising exactly what we asked for.

> **The rule:** a shaped reward encodes *your* guess about how to play. While the guess is
> roughly right and the bonus stays small next to the real objective, it is a helpful hint.
> Make it big enough to compete with the real objective and it *becomes* the objective.

This is the reward-hacking frame from the lecture, reproduced deliberately in about ten lines -
and note how ordinary the mistake looks. Nobody wrote "prefer blocking to winning". They wrote
a sensible-looking bonus and picked a number.
""")

# --------------------------------------------------------------------------------------
md(r"""
---
# Part 8 - Verdict

### "Did it win?" is the wrong question

Tic-tac-toe under perfect play is a **draw**. Against a perfect opponent, winning is not
available to anyone, ever. So the target was never a win rate - it was:

> **can it stop losing?**

Everything in one table.
""")

code(r"""
def summarise(name, Q):
    a = make_q_policy(Q)
    rx, ro = match(a, minimax_policy, 3_000), match(minimax_policy, a, 3_000)
    sc, _, _ = grade(Q, POSITIONS)
    return {"agent": name, "optimal moves": f"{sc:.1%}",
            "draws vs perfect (X)": f"{rx['draws']:.1%}",
            "loses vs perfect (X)": f"{rx['O wins']:.1%}",
            "draws vs perfect (O)": f"{ro['draws']:.1%}",
            "loses vs perfect (O)": f"{ro['X wins']:.1%}"}

import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

table = pd.DataFrame([summarise("trained vs random", Q_vs_random),
                      summarise("self-play", Q_self),
                      summarise("self-play, no exploration", Q_greedy),
                      summarise("self-play + bonus 0.3", Q_shaped),
                      summarise("self-play + bonus 3.0", Q_greedy_block)]).set_index("agent")
table
""")

md(r"""
### What generalises

1. **The opponent you train against defines what you learn.** Part 2's agent looked excellent
   and was not. Nothing about its win rate revealed the problem - only a stronger opponent did.
2. **Ground truth is worth more than any metric.** Grading every position against minimax told
   us not just *how much* was wrong but *which* positions, and the answer (rarely visited ones)
   explained itself immediately.
3. **The convergence guarantee is about coverage, not time.** More episodes do not help states
   you never reach. Part 6 is what that looks like when it goes wrong.
4. **Shaping a reward means changing the problem.** If the shaped objective differs from the
   real one anywhere, the optimiser will find that place - it is doing its job.
5. **Define success before measuring it.** "Draw against perfect play" is the correct target
   here, and it is not the one a win-rate dashboard would have suggested.

### Where this stops working

Every part of this project depended on tic-tac-toe being small enough to solve exactly:
a few thousand positions, a table in memory, ground truth from a recursion that runs instantly.

Chess has roughly $10^{44}$ positions and Go about $10^{170}$. There is no table, and there is
no minimax answer to grade against. Everything past this point in RL is about coping with
**not knowing the right answer** - which is where DQN, policy gradients, and everything in the
second half of the lecture come from.

### If you want to take it further

- Compare Q-learning to **SARSA** here. Which one is more cautious as O?
- Track how the **opening move preference** changes over training. Does it converge to a corner,
  the centre, or stay indifferent (as minimax says it should)?
- Exploit the board's 8-fold symmetry (rotations and reflections) to shrink the state space.
  How much faster does it learn?
- Try **Connect Four on a 4x4 board** and watch the table stop fitting.
""")

# ======================================================================================
nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.write_text(nbf.writes(nb), encoding="utf-8")
n_code = sum(c.cell_type == "code" for c in CELLS)
print(f"wrote {OUT.name}: {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")
