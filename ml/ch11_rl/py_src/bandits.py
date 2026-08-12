"""Multi-armed bandit figures for L32 (deck 1: The RL problem).

The bandit is RL with the state removed: one situation, k actions, and the only
difficulty left is exploration vs exploitation. Everything here is measured, not
asserted - the slides quote the numbers this script logs.

Generates into ml/ch11_rl/fig/:
  bandit_regret.pdf   -- cumulative regret of 4 strategies + final regret bars
  bandit_estimates.pdf -- what greedy sees after 10 pulls vs the truth

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch11_rl/py_src/bandits.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg, fail loud.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1] / "fig"

# ---- the testbed, fully specified -------------------------------------------------------
# Ten ad variants; the number is the true click-through rate. Arm 6 is best at 0.72.
# Fixed (not redrawn per run) so the "pulls per arm" story stays readable.
TRUE_P = np.array([0.31, 0.55, 0.12, 0.68, 0.44, 0.21, 0.72, 0.38, 0.60, 0.05])
BEST_ARM = int(np.argmax(TRUE_P))
BEST_P = float(TRUE_P[BEST_ARM])

N_ARMS = len(TRUE_P)
N_RUNS = 200
# Long horizon on purpose. At 1500 pulls eps-greedy still beats UCB1 (measured: 88 vs 166)
# because UCB's guarantee is asymptotic. The crossover is the point of the figure, so the
# run has to be long enough to contain it.
N_STEPS = 20_000
EPS = 0.1
UCB_C = 2.0


def build_logger():
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "rl_bandits.log", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


def _argmax_random_ties(values, rng):
    """Row-wise argmax over (n_runs, n_arms), breaking ties uniformly at random."""
    noise = rng.random(values.shape) * 1e-9
    return np.argmax(values + noise, axis=1)


def run_strategy(name, rng):
    """Vectorised across runs: step every run forward together.

    Returns pulls (n_runs, n_arms) and per-step chosen arms (n_runs, n_steps).
    """
    counts = np.zeros((N_RUNS, N_ARMS))
    totals = np.zeros((N_RUNS, N_ARMS))          # summed rewards, for the mean estimate
    alpha = np.ones((N_RUNS, N_ARMS))            # Thompson: Beta(1,1) prior
    beta = np.ones((N_RUNS, N_ARMS))
    chosen = np.zeros((N_RUNS, N_STEPS), dtype=int)

    for t in range(N_STEPS):
        q = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)

        if name == "greedy":
            arms = _argmax_random_ties(q, rng)
        elif name == "eps-greedy":
            arms = _argmax_random_ties(q, rng)
            explore = rng.random(N_RUNS) < EPS
            arms[explore] = rng.integers(0, N_ARMS, size=explore.sum())
        elif name == "UCB1":
            # An arm never pulled gets infinite priority, so the first k steps sweep all arms.
            bonus = np.where(
                counts > 0,
                np.sqrt(UCB_C * np.log(t + 1) / np.maximum(counts, 1)),
                np.inf,
            )
            arms = _argmax_random_ties(q + bonus, rng)
        elif name == "Thompson":
            arms = np.argmax(rng.beta(alpha, beta), axis=1)
        else:
            raise ValueError(f"unknown strategy {name!r}")

        rewards = (rng.random(N_RUNS) < TRUE_P[arms]).astype(float)
        rows = np.arange(N_RUNS)
        counts[rows, arms] += 1
        totals[rows, arms] += rewards
        alpha[rows, arms] += rewards
        beta[rows, arms] += 1 - rewards
        chosen[:, t] = arms

    return counts, chosen


def regret_curve(chosen):
    """Cumulative regret: what you lost by not always pulling the best arm."""
    per_step = BEST_P - TRUE_P[chosen]
    return np.cumsum(per_step, axis=1).mean(axis=0)


def fig_regret(results, log):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                                     gridspec_kw={"width_ratios": [1.55, 1]})

    order = ["greedy", "eps-greedy", "UCB1", "Thompson"]
    colors = {"greedy": GREY, "eps-greedy": ORANGE, "UCB1": BLUE, "Thompson": RED}
    labels = {"greedy": "greedy ($\\epsilon = 0$)", "eps-greedy": f"$\\epsilon$-greedy ({EPS})",
              "UCB1": "UCB1", "Thompson": "Thompson sampling"}

    finals, curves = {}, {}
    for name in order:
        curve = regret_curve(results[name][1])
        curves[name] = curve
        finals[name] = curve[-1]
        ax_l.plot(np.arange(1, N_STEPS + 1), curve, color=colors[name], lw=2.2,
                  label=labels[name])

    # Where does UCB1 finally overtake eps-greedy? Measured, not guessed.
    ahead = np.where(curves["UCB1"] < curves["eps-greedy"])[0]
    crossover = int(ahead[0]) + 1 if len(ahead) else None
    if crossover is not None:
        ax_l.axvline(crossover, color=GREY, ls=":", lw=1.3)
        ax_l.annotate(f"UCB1 overtakes\n$\\epsilon$-greedy here\n(pull {crossover:,})",
                      xy=(crossover, curves["UCB1"][crossover - 1]),
                      xytext=(crossover * 1.5, curves["eps-greedy"][-1] * 0.45),
                      fontsize=8.5, color=GREY,
                      arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1))

    ax_l.set_xscale("log")
    ax_l.set_xlabel("pulls (log scale)")
    ax_l.set_ylabel("cumulative regret\n(clicks lost vs always playing the best ad)")
    ax_l.set_title(f"Averaged over {N_RUNS} runs of {N_STEPS:,} pulls", fontsize=11)
    ax_l.legend(frameon=False, fontsize=9, loc="upper left")
    ax_l.spines[["top", "right"]].set_visible(False)
    ax_l.grid(alpha=0.25)

    names = list(reversed(order))
    values = [finals[n] for n in names]
    bars = ax_r.barh(names, values, color=[colors[n] for n in names], alpha=0.9)
    ax_r.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    ax_r.set_xlim(0, max(values) * 1.22)
    ax_r.set_yticks(range(len(names)))
    ax_r.set_yticklabels([labels[n] for n in names], fontsize=9)
    ax_r.set_xlabel(f"regret after {N_STEPS:,} pulls")
    ax_r.set_title("Lower is better", fontsize=11)
    ax_r.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIG / "bandit_regret.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    for name in order:
        counts = results[name][0]
        share = counts[:, BEST_ARM].mean() / N_STEPS
        at_1500 = curves[name][1499]
        log.info(f"{name:>12s}: regret@1500 {at_1500:7.1f}  "
                 f"regret@{N_STEPS} {finals[name]:8.1f}  best-arm share {share:5.1%}")
    log.info(f"UCB1 overtakes eps-greedy at pull {crossover}")
    return out, finals, crossover


PULLS_EACH = 10
N_REPLICATIONS = 20_000


def fig_estimates(rng, log):
    """Why pure greedy fails: a short exploration phase points at the wrong arm often.

    The headline number is measured over N_REPLICATIONS, not read off one lucky draw;
    the bars then show one representative draw that misleads.
    """
    draws = rng.binomial(PULLS_EACH, TRUE_P, size=(N_REPLICATIONS, N_ARMS)) / PULLS_EACH
    picks = np.argmax(draws, axis=1)          # ties -> lowest index, which is greedy's own rule
    miss_rate = float((picks != BEST_ARM).mean())

    misleading = np.where(picks != BEST_ARM)[0]
    if not len(misleading):
        raise RuntimeError(f"{PULLS_EACH} pulls per arm never misled in "
                           f"{N_REPLICATIONS} replications; the frame has no point")
    est = draws[misleading[0]]
    greedy_pick = int(picks[misleading[0]])

    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    x = np.arange(N_ARMS)
    ax.bar(x - 0.2, TRUE_P, width=0.4, color=BLUE, alpha=0.85, label="true click rate")
    bars = ax.bar(x + 0.2, est, width=0.4, color=ORANGE, alpha=0.95,
                  label=f"estimate after {PULLS_EACH} pulls each")

    ax.axvline(BEST_ARM, color=RED, ls="--", lw=1.4, alpha=0.8)
    ax.text(BEST_ARM + 0.12, 1.12, "truly best", color=RED, fontsize=9, va="top")

    ax.annotate("greedy commits here,\nand never pulls another arm again",
                xy=(greedy_pick + 0.2, est[greedy_pick] + 0.02),
                xytext=(max(greedy_pick - 3.4, -0.4), 1.16), fontsize=9, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels([f"ad {i}" for i in x], fontsize=8)
    ax.set_ylabel("click rate")
    ax.set_ylim(0, 1.32)
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{PULLS_EACH} pulls per arm still points at the wrong ad "
                 f"{miss_rate:.0%} of the time", fontsize=11)

    fig.tight_layout()
    out = FIG / "bandit_estimates.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    log.info(f"after {PULLS_EACH} pulls per arm, the empirical best is not the true best "
             f"{miss_rate:.1%} of the time ({N_REPLICATIONS} replications)")
    return out, greedy_pick, miss_rate


def main():
    log = build_logger()
    FIG.mkdir(exist_ok=True)
    log.info(f"testbed: {N_ARMS} arms, best is arm {BEST_ARM} at p={BEST_P}")

    rng = np.random.default_rng(SEED)
    results = {name: run_strategy(name, rng)
               for name in ["greedy", "eps-greedy", "UCB1", "Thompson"]}

    out_regret, finals, crossover = fig_regret(results, log)

    # The frame's three claims. If a reseed ever breaks one, fail loudly rather than
    # shipping a figure that contradicts the slide next to it.
    if not finals["greedy"] > finals["eps-greedy"]:
        raise RuntimeError(f"greedy should be worst, got {finals}")
    if not finals["Thompson"] < finals["UCB1"] < finals["eps-greedy"]:
        raise RuntimeError(
            f"expected Thompson < UCB1 < eps-greedy at the long horizon, got {finals}")
    if crossover is None or crossover < 1500:
        raise RuntimeError(
            f"the frame says eps-greedy leads early and loses late; crossover={crossover}")

    rng_demo = np.random.default_rng(SEED + 1)
    out_est, greedy_pick, miss_rate = fig_estimates(rng_demo, log)
    if miss_rate < 0.2:
        raise RuntimeError(
            f"only {miss_rate:.1%} of short exploration phases mislead; the frame claims "
            f"this is common. Raise PULLS_EACH or rethink the frame.")
    log.info(f"illustrated draw: greedy commits to arm {greedy_pick} "
             f"(p={TRUE_P[greedy_pick]}) instead of arm {BEST_ARM} (p={BEST_P})")

    for out in (out_regret, out_est):
        log.info(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
