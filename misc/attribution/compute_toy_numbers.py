"""Compute attribution credit under every method from one toy dataset.
All slide numbers come from here. Stdlib only. Seed fixed for reproducibility."""
import logging
import os
from itertools import combinations
from math import factorial

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/compute_toy_numbers.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
SEED = 509  # global convention; no RNG used here but recorded for traceability

CHANNELS = ["B", "P", "D"]  # Blog, Pricing, Demo (funnel order)
N = len(CHANNELS)


def canon(S):
    """Order a channel set by funnel order (B, P, D), not alphabetically."""
    return tuple(sorted(S, key=CHANNELS.index))

# Dataset: distinct-channel-set -> (converted, total). Canonical sequence = sorted by B,P,D.
DATA = {
    ("B",): (10, 100),
    ("P",): (20, 100),
    ("D",): (30, 100),
    ("B", "P"): (40, 100),
    ("B", "D"): (45, 100),
    ("P", "D"): (55, 100),
    ("B", "P", "D"): (70, 100),
}


def v(S):
    """Coalition value: conversion rate of journeys whose distinct set is exactly S."""
    S = canon(S)
    if not S:
        return 0.0
    conv, tot = DATA[S]
    return conv / tot


def shapley():
    phi = {c: 0.0 for c in CHANNELS}
    for c in CHANNELS:
        others = [p for p in CHANNELS if p != c]
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[c] += w * (v(set(S) | {c}) - v(set(S)))
    return phi


def first_last_touch():
    ft = {c: 0 for c in CHANNELS}
    lt = {c: 0 for c in CHANNELS}
    for S, (conv, tot) in DATA.items():
        seq = canon(S)  # canonical order
        ft[seq[0]] += tot
        lt[seq[-1]] += tot
    return ft, lt


def u_shaped():
    """40% first, 40% last, 20% split across middles; single-touch gets 100%."""
    credit = {c: 0.0 for c in CHANNELS}
    for S, (conv, tot) in DATA.items():
        seq = canon(S)
        if len(seq) == 1:
            credit[seq[0]] += tot * 1.0
        elif len(seq) == 2:
            credit[seq[0]] += tot * 0.5
            credit[seq[-1]] += tot * 0.5
        else:
            credit[seq[0]] += tot * 0.4
            credit[seq[-1]] += tot * 0.4
            mids = seq[1:-1]
            for m in mids:
                credit[m] += tot * (0.2 / len(mids))
    return credit


def markov_removal_effect():
    """Absorbing first-order chain over states START, B, P, D, CONV, NULL.
    Transition counts come from canonical sequences weighted by total journeys;
    a journey converts (-> CONV) with prob = its set's conversion rate, else -> NULL.
    P(convert) and removal effects computed by iterating reach-probabilities."""
    states = ["START", "B", "P", "D", "CONV", "NULL"]
    idx = {s: i for i, s in enumerate(states)}
    T = [[0.0] * len(states) for _ in states]  # transition COUNT matrix first

    def add(a, b, w):
        T[idx[a]][idx[b]] += w

    for S, (conv, tot) in DATA.items():
        seq = canon(S)
        rate = conv / tot
        add("START", seq[0], tot)
        for u, w in zip(seq, seq[1:]):
            add(u, w, tot)
        add(seq[-1], "CONV", tot * rate)
        add(seq[-1], "NULL", tot * (1 - rate))

    def normalize(counts):
        P = []
        for row in counts:
            s = sum(row)
            P.append([x / s if s else 0.0 for x in row])
        return P

    def p_convert(P, drop=None):
        """Probability of reaching CONV from START. If drop is a channel,
        redirect all its incoming mass to NULL (channel removed)."""
        import copy
        Q = copy.deepcopy(P)
        if drop is not None:
            di = idx[drop]
            for r in range(len(states)):
                if Q[r][di] > 0:
                    Q[r][idx["NULL"]] += Q[r][di]
                    Q[r][di] = 0.0
        reach = [0.0] * len(states)
        reach[idx["CONV"]] = 1.0
        for _ in range(1000):
            new = reach[:]
            for s in range(len(states)):
                if s in (idx["CONV"], idx["NULL"]):
                    continue
                new[s] = sum(Q[s][t] * reach[t] for t in range(len(states)))
            if max(abs(new[i] - reach[i]) for i in range(len(states))) < 1e-12:
                reach = new
                break
            reach = new
        return reach[idx["START"]]

    P = normalize(T)
    base = p_convert(P)
    effects = {}
    for c in CHANNELS:
        effects[c] = (base - p_convert(P, drop=c)) / base if base else 0.0
    return base, effects


def shares(d):
    tot = sum(d.values())
    return {k: round(100 * x / tot, 1) for k, x in d.items()} if tot else d


def main():
    phi = shapley()
    ft, lt = first_last_touch()
    ush = u_shaped()
    base, rem = markov_removal_effect()

    log.info("=== Coalition values v(S) ===")
    for S in [(), ("B",), ("P",), ("D",), ("B", "P"), ("B", "D"), ("P", "D"), ("B", "P", "D")]:
        log.info("v(%s) = %.3f", "".join(S) or "0", v(set(S)))
    log.info("Shapley raw: %s  (sum=%.3f, == v(grand)=0.70)", {k: round(x, 4) for k, x in phi.items()}, sum(phi.values()))
    log.info("Shapley shares %%: %s", shares(phi))
    log.info("First-touch shares %%: %s", shares(ft))
    log.info("Last-touch  shares %%: %s", shares(lt))
    log.info("U-shaped shares %%: %s", shares(ush))
    log.info("Markov P(convert) base = %.4f", base)
    log.info("Markov removal effects raw: %s", {k: round(x, 4) for k, x in rem.items()})
    log.info("Markov shares %%: %s", shares(rem))


if __name__ == "__main__":
    main()
