"""Gate G1 + figures for the intro deck: a network small enough to read completely.

Task: is a point (x, y) inside the diamond |x| + |y| <= 1? A 2 -> 4 -> 1 ReLU network learns it.
The claim the intro deck makes, measured here rather than asserted:

  1. each hidden unit is one EDGE of the diamond (a half-plane detector we can name by reading its
     two weights and bias);
  2. the output unit combines them - the circuit is "inside = not outside any edge";
  3. ablating one unit removes exactly that edge (intervention confirms the reading);
  4. retraining from other seeds finds the same four edges, in a different order - universality in
     miniature.

Training all eight seeds takes about four minutes on one CPU core. Results go to results/intro_toy_mlp.json first; the figures
are drawn from that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/intro_toy_mlp.py            # run + plot
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/intro_toy_mlp.py --plot-only
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import BLUE, FIG_DIR, GREY, ORANGE, RED, SEED, load_results, save_results, setup_logging

N_HIDDEN = 4
SEEDS = [SEED, 1, 2, 3, 4, 5, 6, 7]
GRID_N = 161
LIM = 2.0

# The diamond's four edges, as outward unit normals n and offset d: the edge is n . x = d.
EDGES = {
    "top-right": (np.array([1.0, 1.0]) / math.sqrt(2), 1 / math.sqrt(2)),
    "top-left": (np.array([-1.0, 1.0]) / math.sqrt(2), 1 / math.sqrt(2)),
    "bottom-left": (np.array([-1.0, -1.0]) / math.sqrt(2), 1 / math.sqrt(2)),
    "bottom-right": (np.array([1.0, -1.0]) / math.sqrt(2), 1 / math.sqrt(2)),
}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def make_data(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.rand(n, 2, generator=gen) * 2 - 1) * LIM
    y = (x.abs().sum(1) <= 1.0).float()
    return x, y


def train(seed: int, log) -> nn.Sequential:
    # One thread: a 2-4-1 net is all overhead at the default thread count (60 s/seed -> seconds).
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x, y = make_data(6000, gen)
    net = nn.Sequential(nn.Linear(2, N_HIDDEN), nn.ReLU(), nn.Linear(N_HIDDEN, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(4000):
        opt.zero_grad()
        loss = loss_fn(net(x).squeeze(1), y)
        loss.backward()
        opt.step()
    log.info(f"seed {seed}: final train loss {loss.item():.4f}")
    return net


def grid() -> tuple[np.ndarray, torch.Tensor]:
    g = np.linspace(-LIM, LIM, GRID_N)
    xx, yy = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
    return g, pts


def name_units(net: nn.Sequential, pts: torch.Tensor) -> list[dict]:
    """Read each hidden unit's line off its weights and match it to a diamond edge.

    A unit computes ReLU(w . x + b): it is zero on one side of the line w . x + b = 0 and grows
    linearly on the other. Written as n . x = d with n = w / |w|, it fires where n . x > d. A unit
    is matched to the edge whose line it sits on (normal within a few degrees, offset close to
    1/sqrt 2), and we record which side it fires on. A unit that never fires anywhere in the
    plane is DEAD - a dead ReLU, the failure from the initialization lecture.
    """
    w = net[0].weight.detach().numpy()          # (4, 2)
    b = net[0].bias.detach().numpy()            # (4,)
    a = net[2].weight.detach().numpy()[0]       # (4,) output weights
    h = torch.relu(net[0](pts)).detach().numpy()
    out = []
    for i in range(N_HIDDEN):
        norm = float(np.linalg.norm(w[i]))
        n = w[i] / norm
        d = float(-b[i] / norm)
        best = None
        for name, (nk, dk) in EDGES.items():
            for sign in (+1, -1):
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(sign * n @ nk)))))
                off = abs(sign * d - dk)
                score = ang / 10 + off / 0.1
                if best is None or score < best[0]:
                    best = (score, name, sign, ang, off)
        _, edge, sign, ang, off = best
        dead = bool(h[:, i].max() == 0.0)
        named = (not dead) and ang < 5 and off < 0.15
        out.append({
            "unit": i, "w": w[i].tolist(), "b": float(b[i]), "out_weight": float(a[i]),
            "normal": n.tolist(), "offset": d, "edge": edge if named else None,
            "fires": ("outside" if sign > 0 else "inside") if named else None,
            "dead": dead, "angle_err_deg": ang, "offset_err": off,
            "max_activation": float(h[:, i].max()),
        })
    return out


def predict_region(net: nn.Sequential, pts: torch.Tensor, ablate: int | None = None) -> np.ndarray:
    with torch.no_grad():
        h = torch.relu(net[0](pts))
        if ablate is not None:
            h[:, ablate] = 0.0
        return (net[2](h).squeeze(1) > 0).numpy()


def run(log) -> dict:
    _, pts = grid()
    truth = (pts.abs().sum(1) <= 1.0).numpy()
    seeds = []
    for seed in SEEDS:
        net = train(seed, log)
        pred = predict_region(net, pts)
        units = name_units(net, pts)
        edges_found = sorted(u["edge"] for u in units if u["edge"])
        clean = edges_found == sorted(EDGES) and all(u["fires"] == "outside" for u in units)
        wrong = ~(pred == truth)
        log.info(f"seed {seed}: grid accuracy {(pred == truth).mean():.4f}, named edges "
                 f"{edges_found}, dead units {sum(u['dead'] for u in units)}, clean={clean}")
        # The trained weights ARE the result: the figures re-evaluate them on a grid instead of
        # storing 161x161 grids here (that made this file 7 MB).
        entry = {"seed": seed, "grid_accuracy": float((pred == truth).mean()), "units": units,
                 "out_bias": float(net[2].bias.item()),
                 "all_four_edges": bool(clean), "n_dead": int(sum(u["dead"] for u in units)),
                 "frac_grid_wrong": float(wrong.mean()),
                 "frac_grid_predicted_inside": float(pred.mean())}
        seeds.append((entry, net, pred))

    display = next(e for e, _, _ in seeds if e["all_four_edges"])
    log.info(f"display seed: {display['seed']} (first seed in {SEEDS} that learned all four edges)")
    net, pred_before = next((n, p) for e, n, p in seeds if e is display)
    # Renumber the hidden units so unit 1..4 = top-right, top-left, bottom-left, bottom-right.
    # A permutation of hidden units changes nothing the network computes; it only makes the
    # slide labels read in order.
    edge_order = ["top-right", "top-left", "bottom-left", "bottom-right"]
    perm = sorted(range(N_HIDDEN), key=lambda i: edge_order.index(display["units"][i]["edge"]))
    with torch.no_grad():
        net[0].weight.copy_(net[0].weight[perm]); net[0].bias.copy_(net[0].bias[perm])
        net[2].weight.copy_(net[2].weight[:, perm])
    display["units"] = units = name_units(net, pts)
    assert [u["edge"] for u in units] == edge_order, "renumbering failed"
    assert (predict_region(net, pts) == pred_before).all(), "permutation changed the function"
    ablations = []
    for i in range(N_HIDDEN):
        p = predict_region(net, pts, ablate=i)
        added, lost = p & ~truth, ~p & truth
        ablations.append({"unit": i, "edge": units[i]["edge"],
                          "frac_grid_added": float(added.mean()), "frac_grid_lost": float(lost.mean())})
        log.info(f"ablate unit {i} ({units[i]['edge']}): +{added.mean():.4f} of grid newly "
                 f"'inside', -{lost.mean():.5f} lost")
    display["ablations"] = ablations
    worked = {"out_bias": float(net[2].bias.item()), "points": []}
    for xy in ([0.5, 0.25], [1.0, 0.5]):
        x = torch.tensor([xy])
        pre = net[0](x)[0]
        hid = torch.relu(pre)
        z = net[2](hid.unsqueeze(0))
        worked["points"].append({"xy": xy, "pre": pre.tolist(), "hidden": hid.tolist(),
                                 "logit": float(z), "inside": bool(abs(xy[0]) + abs(xy[1]) <= 1)})
    display["worked"] = worked

    n_clean = sum(e["all_four_edges"] for e, _, _ in seeds)
    log.info(f"GATE G1: {n_clean}/{len(seeds)} seeds learn the four edges")
    return {"n_hidden": N_HIDDEN, "grid_n": GRID_N, "lim": LIM,
            "display_seed": display["seed"], "seeds": [e for e, _, _ in seeds], "n_clean_seeds": n_clean}


def evaluate(entry: dict, ablate: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Re-run a saved network on the plotting grid from its stored weights (no training).

    Returns the hidden activations (GRID_N*GRID_N, 4) and the inside/outside prediction grid.
    """
    _, pts = grid()
    x = pts.numpy()
    w = np.array([u["w"] for u in entry["units"]])
    b = np.array([u["b"] for u in entry["units"]])
    a = np.array([u["out_weight"] for u in entry["units"]])
    h = np.maximum(0.0, x @ w.T + b)
    if ablate is not None:
        h[:, ablate] = 0.0
    pred = (h @ a + entry["out_bias"] > 0).reshape(GRID_N, GRID_N)
    return h, pred


TRUTH = (grid()[1].abs().sum(1) <= 1.0).numpy().reshape(GRID_N, GRID_N)


def diamond(ax, **kw):
    xs = [1, 0, -1, 0, 1]
    ys = [0, 1, 0, -1, 0]
    ax.plot(xs, ys, **kw)


def draw_lines(ax, units, color, t=np.linspace(-4, 4, 60)):
    for u in units:
        if u["dead"]:
            continue
        n = np.array(u["normal"]); d = u["offset"]
        line = (n * d)[None, :] + t[:, None] * np.array([-n[1], n[0]])[None, :]
        ax.plot(line[:, 0], line[:, 1], color=color, lw=1.6)


def plot(res: dict, log) -> None:
    lim = res["lim"]
    ext = [-lim, lim, -lim, lim]
    main = next(s for s in res["seeds"] if s["seed"] == res["display_seed"])
    units = main["units"]

    def style(ax, ticks=False):
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        if ticks:
            ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
        else:
            ax.set_xticks([]); ax.set_yticks([])

    # --- the task and what the network learned
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    rng = np.random.default_rng(SEED)
    pts = rng.uniform(-lim, lim, size=(700, 2))
    inside = np.abs(pts).sum(1) <= 1
    ax = axes[0]
    ax.scatter(*pts[~inside].T, s=6, color=GREY, alpha=0.5, label="outside")
    ax.scatter(*pts[inside].T, s=8, color=RED, label="inside")
    diamond(ax, color="black", lw=1.2)
    ax.set_title("the task: is the point inside?")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax = axes[1]
    main_h, main_pred = evaluate(main)
    assert abs((main_pred == TRUTH).mean() - main["grid_accuracy"]) < 2e-4, "weights do not reproduce"
    ax.imshow(main_pred.astype(int), origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.6)
    diamond(ax, color="black", lw=1.2, ls="--")
    ax.set_title(f"what the trained network says ({main['grid_accuracy']:.1%} right)")
    for a in axes:
        style(a, ticks=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_toy_task.pdf", bbox_inches="tight"); plt.close(fig)

    # --- each hidden unit over the plane
    edge_order = ["top-right", "top-left", "bottom-left", "bottom-right"]
    order = sorted(range(N_HIDDEN), key=lambda i: edge_order.index(units[i]["edge"]))
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6))
    for ax, i in zip(axes, order):
        act = main_h[:, i].reshape(GRID_N, GRID_N)
        ax.imshow(act, origin="lower", extent=ext, cmap="Blues")
        diamond(ax, color=RED, lw=1.4)
        ax.set_title(f"unit {i + 1}: fires outside\nthe {units[i]['edge']} edge", fontsize=11)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_toy_units.pdf", bbox_inches="tight"); plt.close(fig)

    # --- ablate each unit
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6))
    for ax, i in zip(axes, order):
        _, abl_pred = evaluate(main, ablate=i)
        ax.imshow(abl_pred.astype(int), origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.6)
        diamond(ax, color="black", lw=1.0, ls="--")
        ax.set_title(f"unit {i + 1} deleted", fontsize=11)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_toy_ablation.pdf", bbox_inches="tight"); plt.close(fig)

    # --- every seed: the learned lines
    seeds = res["seeds"]
    fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.3))
    for ax, s in zip(axes.ravel(), seeds):
        diamond(ax, color="black", lw=1.0, ls="--")
        draw_lines(ax, s["units"], BLUE if s["all_four_edges"] else ORANGE)
        if s["all_four_edges"]:
            ax.set_title(f"seed {s['seed']}: four edges", fontsize=10)
        elif s["n_dead"]:
            ax.set_title(f"seed {s['seed']}: {s['n_dead']} dead unit{'s' if s['n_dead'] != 1 else ''},"
                         f" {s['grid_accuracy']:.0%} right", fontsize=10, color=RED)
        else:
            ax.set_title(f"seed {s['seed']}: messier, {s['grid_accuracy']:.0%} right",
                         fontsize=10, color=RED)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_toy_seeds.pdf", bbox_inches="tight"); plt.close(fig)

    # --- the stuck network: where its mistakes are
    stuck = next(s for s in seeds if not s["all_four_edges"])
    _, stuck_pred = evaluate(stuck)
    wrong = (stuck_pred != TRUTH).astype(float)
    assert abs(wrong.mean() - stuck["frac_grid_wrong"]) < 2e-4, "weights do not reproduce"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    diamond(ax, color="black", lw=1.0, ls="--")
    draw_lines(ax, stuck["units"], ORANGE)
    ax.set_title(f"seed {stuck['seed']}: {N_HIDDEN - stuck['n_dead']} of {N_HIDDEN} units ever fire")
    style(ax, ticks=True)
    ax = axes[1]
    ax.imshow(wrong, origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.3)
    diamond(ax, color="black", lw=1.0, ls="--")
    ax.set_title(f"where it is wrong ({stuck['frac_grid_wrong']:.1%} of the plane)")
    style(ax, ticks=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intro_toy_stuck.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("wrote intro_toy_task / _units / _ablation / _seeds / _stuck")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("intro_toy_mlp")
    FIG_DIR.mkdir(exist_ok=True)
    if not args.plot_only:
        save_results("intro_toy_mlp", run(log), log)
    plot(load_results("intro_toy_mlp"), log)


if __name__ == "__main__":
    main()
