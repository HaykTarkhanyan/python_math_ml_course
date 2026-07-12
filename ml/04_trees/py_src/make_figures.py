"""Generate the real-data figures for the [17] Decision Trees deck.

Produces PDFs into ``ml/04_trees/fig/`` from the Titanic dataset plus small
synthetic demos:
  1. titanic_tree.pdf       -- a shallow (depth-2) decision tree, plot_tree.
  2. titanic_overfit.pdf    -- train vs CV-test accuracy across tree depth.
  3. titanic_pruning.pdf    -- cost-complexity (ccp_alpha) pruning path.
  4. tree_staircase.pdf     -- a diagonal boundary approximated by axis-aligned
                               steps at depth 3 / 6 / 10 (synthetic 2D).
  5. tree_extrapolation.pdf -- tree goes flat outside the training range while a
                               line keeps trending (synthetic 1D rent-vs-area).
  6. tree_greedy_xor.pdf    -- XOR: one split cannot help, two splits solve it.
  7. split_anim_1..7.pdf    -- step-by-step: sweep candidate thresholds on one
                               numeric feature, score each by Gini gain, pick the
                               winner (Beamer \only overlay animation).
  8. akinator_ig.pdf        -- information gain (bits) of candidate yes/no questions
                               on an 8-character pool; a balanced question gains most.

Run with the project venv (repo CLAUDE.md -- do NOT spin up an ephemeral env):
    ./ma/Scripts/python.exe ml/04_trees/py_src/make_figures.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours for multi-line plots.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

SEED = 509

# Armenian flag palette (per CLAUDE.md) -- blue=train, red=test/CV, orange=marker.
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]               # ml/04_trees
REPO_ROOT = HERE.parents[3]            # repo root
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l09_figures")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "make_figures.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def load_titanic(logger: logging.Logger):
    """Return (X, y, feature_names) with a tiny, explicit preprocessing."""
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    df = df[features + ["survived"]].copy()
    df["sex"] = (df["sex"] == "female").astype(int)        # female=1, male=0
    df["pclass"] = df["pclass"].astype(float)
    df["age"] = df["age"].fillna(df["age"].median())        # impute (median)
    df["fare"] = df["fare"].fillna(df["fare"].median())
    X = df[features].astype(float).to_numpy()
    y = df["survived"].astype(int).to_numpy()
    logger.info(f"Titanic: {X.shape[0]} rows, {X.shape[1]} features, "
                f"survived rate = {y.mean():.3f}")
    return X, y, features


def fig_tree(X, y, feature_names, logger):
    """A readable depth-2 tree for the slide (fewer, bigger boxes than depth-3,
    which was cramped on a projector)."""
    clf = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    clf.fit(X, y)
    display_names = ["gender" if f == "sex" else f for f in feature_names]   # relabel sex -> gender
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plot_tree(
        clf, feature_names=display_names, class_names=["died", "survived"],
        filled=True, rounded=True, impurity=False, proportion=True,
        fontsize=12, ax=ax,
    )
    fig.tight_layout()
    out = FIG_DIR / "titanic_tree.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name} (depth-2 tree, train acc={clf.score(X,y):.3f})")


def fig_staircase(logger):
    """A slanted (non-axis-aligned) true boundary and the tree's staircase
    approximation at increasing depth: axis-aligned splits can only stair-step it."""
    rng = np.random.default_rng(SEED)
    Xg = rng.uniform(0, 1, size=(400, 2))
    yg = (Xg[:, 1] > Xg[:, 0]).astype(int)            # slanted boundary x1 = x0
    xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True, sharey=True)
    for ax, d in zip(axes, [3, 6, 10]):
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xg, yg)
        zz = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=[ARM_ORANGE, ARM_BLUE], alpha=0.25)
        ax.plot([0, 1], [0, 1], color=ARM_RED, lw=2, label="true boundary")
        ax.scatter(Xg[yg == 1, 0], Xg[yg == 1, 1], s=6, color=ARM_BLUE, alpha=0.6)
        ax.scatter(Xg[yg == 0, 0], Xg[yg == 0, 1], s=6, color=ARM_ORANGE, alpha=0.6)
        ax.set_title(f"depth {d}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("A diagonal boundary, approximated by axis-aligned steps", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "tree_staircase.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_extrapolation(logger):
    """Trees predict a constant outside the training range; a line keeps trending.
    Shared demo for [17] / [18] / [20]. Synthetic rent-vs-area (k AMD)."""
    rng = np.random.default_rng(SEED)
    area = np.linspace(40, 100, 60)
    rent = 6.0 + 1.8 * area + rng.normal(0, 6, size=area.size)
    Xtr = area.reshape(-1, 1)
    lin = LinearRegression().fit(Xtr, rent)
    tree = DecisionTreeRegressor(max_depth=4, random_state=SEED).fit(Xtr, rent)
    grid = np.linspace(40, 130, 400).reshape(-1, 1)          # extend ~30% past 100
    flat = float(tree.predict(np.array([[120.0]]))[0])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(100, 130, color=ARM_ORANGE, alpha=0.10)
    ax.scatter(area, rent, s=12, color="gray", alpha=0.6, label="training data (area 40-100)")
    ax.plot(grid, lin.predict(grid), color=ARM_BLUE, lw=2, label="linear fit")
    ax.plot(grid, tree.predict(grid), color=ARM_RED, lw=2, label="tree fit")
    ax.axvline(100, color="gray", ls=":", lw=1.2)
    ax.annotate("beyond the training range:\ntree stays flat, line keeps trending",
                xy=(122, flat), xytext=(52, rent.max() * 0.78), fontsize=8,
                color=ARM_ORANGE, arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel(r"apartment area (m$^2$)")
    ax.set_ylabel("rent (k AMD)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "tree_extrapolation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_greedy_xor(logger):
    """XOR: a single axis-aligned split gains nothing (both sides stay ~50/50), so a
    greedy 'stop when no split helps' rule would quit. The first split is a tie, so
    greedy places it badly and needs extra depth to separate XOR -- a real caveat.
    Background tint = the tree's PREDICTED class; dots = the TRUE class."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rng = np.random.default_rng(SEED)
    Xg = rng.uniform(-1, 1, size=(300, 2))
    yg = ((Xg[:, 0] > 0) ^ (Xg[:, 1] > 0)).astype(int)       # XOR / checkerboard
    xx, yy = np.meshgrid(np.linspace(-1, 1, 300), np.linspace(-1, 1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3), sharex=True, sharey=True)
    titles = ["depth 1: one split (no gain)", "depth 3: grown deeper"]
    for ax, d, title in zip(axes, [1, 3], titles):
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xg, yg)
        zz = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=[ARM_ORANGE, ARM_BLUE], alpha=0.20)
        ax.scatter(Xg[yg == 0, 0], Xg[yg == 0, 1], s=13, color=ARM_ORANGE,
                   edgecolors="white", linewidths=0.4)
        ax.scatter(Xg[yg == 1, 0], Xg[yg == 1, 1], s=13, color=ARM_BLUE,
                   edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(f"{title}   (train acc {clf.score(Xg, yg):.2f})", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("feature 1", fontsize=8)
    axes[0].set_ylabel("feature 2", fontsize=8)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_ORANGE,
               markersize=8, label="data: class A"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_BLUE,
               markersize=8, label="data: class B"),
        Patch(facecolor=ARM_ORANGE, alpha=0.20, label="tree predicts A"),
        Patch(facecolor=ARM_BLUE, alpha=0.20, label="tree predicts B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIG_DIR / "tree_greedy_xor.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def _gini(labels):
    """Gini impurity of a set of 0/1 labels (0 for the empty set)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    p = labels.mean()                       # fraction of class 1 (red)
    return 1.0 - p ** 2 - (1.0 - p) ** 2


def fig_split_animation(logger):
    r"""One very good step-by-step split animation for a Beamer \only overlay.

    A single numeric feature (hours studied) with two classes (red = failed,
    blue = passed). We sweep candidate thresholds left -> right, score each by
    Gini gain, and reveal the winner. Emits split_anim_1.pdf .. split_anim_7.pdf;
    frame [17] swaps them with \only<1>..\only<7>. The class key lives in the
    Beamer caption, not here, so every overlay shares one canvas size.
    """
    hours = np.arange(1, 11, dtype=float)              # 1..10 hours studied
    # 1 = red (failed), 0 = blue (passed); one noisy failure at 9 hours.
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0])
    RED, BLUE = 1, 0
    parent = _gini(y)

    thresholds = [2.5, 4.5, 5.5, 6.5, 8.5]             # candidate midpoints, L -> R
    rows = []
    for t in thresholds:
        left, right = y[hours <= t], y[hours > t]
        gl, gr = _gini(left), _gini(right)
        w = (left.size / y.size) * gl + (right.size / y.size) * gr
        rows.append(dict(
            t=t, gl=gl, gr=gr, w=w, gain=parent - w,
            lr=(int((left == RED).sum()), int((left == BLUE).sum())),
            rr=(int((right == RED).sum()), int((right == BLUE).sum())),
        ))
    best = int(np.argmax([r["gain"] for r in rows]))
    logger.info("split-anim: parent Gini=%.3f; gains: %s; best hours<=%g",
                parent,
                ", ".join(f"{r['t']:g}->{r['gain']:.3f}" for r in rows),
                rows[best]["t"])

    from matplotlib.patches import FancyBboxPatch
    RED_TINT, BLUE_TINT, GRAY_TINT = "#F8D3D7", "#D2DCEF", "#ECECEC"

    def _leaf_style(counts):
        """Tint + edge for a child, coloured by its majority class (gray on a tie)."""
        nR, nB = counts
        if nR > nB:
            return RED_TINT, ARM_RED
        if nB > nR:
            return BLUE_TINT, ARM_BLUE
        return GRAY_TINT, "0.5"

    def draw(n):
        """n = 1 parent only; 2..6 reveal one more threshold; 7 final winner."""
        final = (n == 7)
        k = 0 if n == 1 else (len(rows) if final else n - 1)   # scorecard rows shown
        cur = None if (n == 1 or final) else k - 1             # current threshold row
        show = None if n == 1 else (rows[best] if final else rows[cur])

        fig, (axL, axT, axS) = plt.subplots(
            3, 1, figsize=(9.2, 4.8), gridspec_kw=dict(height_ratios=[0.7, 1.35, 1.0]))
        fig.subplots_adjust(hspace=0.5, top=0.93, bottom=0.06)

        # --- row 1: the number line with the current cut ---
        axL.set_xlim(0.3, 10.7)
        axL.set_ylim(-0.5, 0.6)
        axL.axhline(0, color="0.85", lw=1, zorder=0)
        axL.scatter(hours, np.zeros_like(hours),
                    c=np.where(y == RED, ARM_RED, ARM_BLUE),
                    s=210, edgecolors="white", linewidths=1.2, zorder=3)
        axL.set_yticks([])
        for s in ("left", "right", "top"):
            axL.spines[s].set_visible(False)
        axL.set_xlabel("hours studied", fontsize=10)
        axL.set_xticks(np.arange(1, 11))
        axL.tick_params(labelsize=8)
        if show is None:
            axL.set_title("Before the split: one mixed node", fontsize=11, color=ARM_BLUE)
        else:
            t = show["t"]
            axL.axvspan(0.3, t, color=ARM_RED, alpha=0.06)
            axL.axvspan(t, 10.7, color=ARM_BLUE, alpha=0.06)
            axL.axvline(t, color=(ARM_ORANGE if final else "0.35"),
                        lw=(3 if final else 2), ls=("solid" if final else "--"), zorder=4)
            if final:
                axL.set_title(rf"Best split:  hours $\leq$ {t:g}   (gain {show['gain']:.2f})",
                              fontsize=11, color=ARM_ORANGE)
            else:
                axL.set_title(rf"Try split:  hours $\leq$ {t:g}", fontsize=11, color="0.2")

        # --- row 2: the split as a little tree (parent -> two leaves) ---
        axT.axis("off")
        axT.set_xlim(0, 1)
        axT.set_ylim(0, 1)

        def _box(cx, cy, w, h, face, edge, lw=1.6):
            axT.add_patch(FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))

        def _nodebox(cx, cy, w, h, counts, gini, title, edge, lw=1.6, tail=""):
            """A node box showing its class mix as coloured dots (not text)."""
            _box(cx, cy, w, h, _leaf_style(counts)[0], edge, lw=lw)
            axT.text(cx, cy + h / 2 - 0.055, title, ha="center", va="top",
                     fontsize=9.5, fontweight="bold", zorder=3)
            nR, nB = counts
            m = nR + nB
            dx = min(0.05, (w - 0.10) / max(m - 1, 1))
            xs = cx + (np.arange(m) - (m - 1) / 2) * dx
            axT.scatter(xs, np.full(m, cy - 0.01), c=[ARM_RED] * nR + [ARM_BLUE] * nB,
                        s=62, edgecolors="white", linewidths=0.6, zorder=4)
            axT.text(cx, cy - h / 2 + 0.05, f"Gini = {gini:.2f}{tail}",
                     ha="center", va="bottom", fontsize=9, zorder=3)

        if show is None:
            _nodebox(0.5, 0.5, 0.52, 0.58,
                     (int((y == RED).sum()), int((y == BLUE).sum())),
                     parent, "parent node", ARM_BLUE, lw=1.8, tail="   (mixed)")
        else:
            t = show["t"]
            _box(0.5, 0.87, 0.30, 0.18, GRAY_TINT, "0.5")
            axT.text(0.5, 0.87, f"parent  (Gini {parent:.2f})", ha="center", va="center",
                     fontsize=8.5, zorder=3)
            axT.plot([0.5, 0.27], [0.76, 0.54], color="0.5", lw=1.3, zorder=1)
            axT.plot([0.5, 0.73], [0.76, 0.54], color="0.5", lw=1.3, zorder=1)
            axT.text(0.35, 0.68, rf"$\leq$ {t:g}", fontsize=8.5, ha="center", color="0.3")
            axT.text(0.65, 0.68, rf"$>$ {t:g}", fontsize=8.5, ha="center", color="0.3")
            ledge = ARM_ORANGE if final else _leaf_style(show["lr"])[1]
            redge = ARM_ORANGE if final else _leaf_style(show["rr"])[1]
            lw = 2.2 if final else 1.6
            _nodebox(0.27, 0.29, 0.42, 0.46, show["lr"], show["gl"], "LEFT leaf", ledge, lw=lw)
            _nodebox(0.73, 0.29, 0.42, 0.46, show["rr"], show["gr"], "RIGHT leaf", redge, lw=lw)

        # --- row 3: scorecard (accumulating gains across thresholds) ---
        axS.axis("off")
        axS.set_xlim(0, 1)
        axS.set_ylim(0, 1)
        axS.text(0.0, 0.98, "Scorecard   (gain = parent Gini - weighted leaves)",
                 fontsize=10, fontweight="bold", va="top")
        axS.text(0.0, 0.82, f"parent Gini = {parent:.2f}", fontsize=9.5, va="top", color="0.25")
        for j in range(k):
            r = rows[j]
            is_win = final and j == best
            is_cur = cur is not None and j == cur
            color = ARM_ORANGE if is_win else ("black" if is_cur else "0.55")
            weight = "bold" if (is_win or is_cur) else "normal"
            txt = rf"hours $\leq$ {r['t']:g} :   weighted Gini {r['w']:.2f}      gain = {r['gain']:.2f}"
            if is_win:
                txt += "      <- best"
            axS.text(0.03, 0.63 - j * 0.13, txt, fontsize=9.5, va="top",
                     color=color, fontweight=weight)

        # NB: no bbox_inches="tight" -- a fixed canvas keeps all 7 overlays the
        # same size so the Beamer animation does not jump between clicks.
        fig.savefig(FIG_DIR / f"split_anim_{n}.pdf")
        plt.close(fig)

    for n in range(1, 8):
        draw(n)
    logger.info("wrote split_anim_1..7.pdf")


def fig_akinator_ig(logger):
    """Akinator maximizes information gain: a balanced yes/no question removes ~1 bit,
    a question almost nobody matches barely helps. Pool of 8 illustrative characters."""
    import math

    total = 8
    Hp = math.log2(total)                          # 3 bits for a uniform pool of 8

    def ig(n_yes):
        n_no = total - n_yes
        h = lambda m: 0.0 if m == 0 else math.log2(m)   # entropy of a uniform group
        return Hp - ((n_yes / total) * h(n_yes) + (n_no / total) * h(n_no))

    # (question, yes-count) on the 8-character pool
    qs = [("Musician?  (4 / 4)", 4),
          ("Born in the 1900s?  (5 / 3)", 5),
          ("Plays chess?  (2 / 6)", 2),
          ("Is it Aivazovsky?  (1 / 7)", 1)]
    labels = [q for q, _ in qs]
    vals = [ig(n) for _, n in qs]
    order = np.argsort(vals)                       # ascending -> best ends on top
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    top = max(vals)
    colors = [ARM_ORANGE if v == top else ARM_BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.barh(labels, vals, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("information gain (bits)")
    ax.set_xlim(0, 1.2)
    ax.set_title("A balanced question gains the most (~1 bit = halve the pool)",
                 fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "akinator_ig.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (IG bits: %s)", out.name,
                ", ".join(f"{n}/{total-n}={ig(n):.2f}" for _, n in qs))


def fig_impurity_pair(logger):
    """Two leaves side by side -- one PURE (Gini 0), one MIXED (Gini 0.5) -- to make
    'impurity' concrete before we put a formula on it. Class mix shown as dots."""
    from matplotlib.patches import FancyBboxPatch
    RED_TINT, BLUE_TINT, GRAY_TINT = "#F8D3D7", "#D2DCEF", "#ECECEC"

    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def leaf(cx, cols, face, edge, title, sub):
        w, h, cy = 0.42, 0.66, 0.5
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.03", lw=2,
            edgecolor=edge, facecolor=face, zorder=1))
        ax.text(cx, cy + h / 2 - 0.05, title, ha="center", va="top",
                fontsize=11.5, fontweight="bold", zorder=3)
        per_row = 4
        for i, c in enumerate(cols):
            r, cc = divmod(i, per_row)
            x = cx + (cc - (per_row - 1) / 2) * 0.078
            yy = cy + 0.06 - r * 0.11
            ax.scatter([x], [yy], c=c, s=110, edgecolors="white", linewidths=0.8, zorder=2)
        ax.text(cx, cy - h / 2 + 0.05, sub, ha="center", va="bottom", fontsize=10, zorder=3)

    leaf(0.27, [ARM_BLUE] * 8, BLUE_TINT, ARM_BLUE, "pure leaf",
         "all one class\nGini $= 0$")
    leaf(0.73, [ARM_BLUE] * 4 + [ARM_RED] * 4, GRAY_TINT, "0.45", "mixed leaf",
         "50 / 50 split\nGini $= 0.5$ (worst)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "impurity_pair.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote impurity_pair.pdf")


def fig_ig_split(logger):
    """Full-slide information-gain illustration on the tennis 'Humidity' split:
    parent entropy -> two leaves' entropy -> the gain (the drop). Numbers match the
    worked-split table (Humidity IG = 0.15 bits)."""
    import math
    from matplotlib.patches import FancyBboxPatch

    def H(pos, tot):
        if pos == 0 or pos == tot:
            return 0.0
        p = pos / tot
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    Hp = H(9, 14)                              # parent 9 Yes / 5 No
    Hh, Hn = H(3, 7), H(6, 7)                   # High 3/4 ; Normal 6/1
    Hw = (7 / 14) * Hh + (7 / 14) * Hn          # weighted children
    ig = Hp - Hw

    fig = plt.figure(figsize=(9.4, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.5)
    axT = fig.add_subplot(gs[0])
    axT.axis("off")
    axT.set_xlim(0, 1)
    axT.set_ylim(0, 1)
    axB = fig.add_subplot(gs[1])

    GREEN_T, RED_T, GRAY_T = "#D9EAD3", "#F8D3D7", "#ECECEC"

    def node(cx, cy, w, h, face, edge, text, fs=10):
        axT.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.02", lw=1.8,
            edgecolor=edge, facecolor=face, zorder=2))
        axT.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)

    node(0.5, 0.82, 0.46, 0.30, GRAY_T, "0.4",
         "parent: 9 Yes, 5 No\n$H = 0.94$ bits", fs=10.5)
    axT.plot([0.5, 0.27], [0.66, 0.42], color="0.5", lw=1.3, zorder=1)
    axT.plot([0.5, 0.73], [0.66, 0.42], color="0.5", lw=1.3, zorder=1)
    axT.text(0.35, 0.55, "Humidity\n$=$ High", fontsize=8.5, ha="center", color="0.3")
    axT.text(0.65, 0.55, "Humidity\n$=$ Normal", fontsize=8.5, ha="center", color="0.3")
    node(0.27, 0.24, 0.44, 0.30, RED_T, ARM_RED,
         "High: 3 Yes, 4 No\n$H = 0.99$ (mixed)", fs=10)
    node(0.73, 0.24, 0.44, 0.30, GREEN_T, "#3F8F29",
         "Normal: 6 Yes, 1 No\n$H = 0.59$ (purer)", fs=10)

    axB.barh([1, 0], [Hp, Hw], color=["0.6", ARM_BLUE], height=0.5)
    axB.set_yticks([1, 0])
    axB.set_yticklabels(["before split\n(parent)", "after split\n(weighted leaves)"], fontsize=9)
    axB.text(Hp + 0.01, 1, f"{Hp:.2f}", va="center", fontsize=10)
    axB.text(Hw + 0.01, 0, f"{Hw:.2f}", va="center", fontsize=10)
    axB.annotate("", xy=(Hp, 0.5), xytext=(Hw, 0.5),
                 arrowprops=dict(arrowstyle="<->", color=ARM_ORANGE, lw=2))
    axB.text((Hw + Hp) / 2, 0.5, f"IG $= {ig:.2f}$ bits", ha="center", va="center",
             fontsize=10.5, color=ARM_ORANGE, fontweight="bold",
             bbox=dict(fc="white", ec="none", pad=1))
    axB.set_xlim(0, 1.1)
    axB.set_ylim(-0.5, 1.5)
    axB.set_xlabel("entropy / uncertainty (bits)", fontsize=9)
    axB.spines[["top", "right"]].set_visible(False)
    axB.tick_params(labelsize=9)
    fig.savefig(FIG_DIR / "ig_split.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote ig_split.pdf (Hp=%.3f Hw=%.3f IG=%.3f)", Hp, Hw, ig)


def fig_regression_split(logger):
    """Regression tree = reduce SPREAD. One leaf predicts the overall mean (big spread);
    one split -> two leaves, each predicts its own mean, and the spread drops."""
    rng = np.random.default_rng(SEED)
    area = np.linspace(40, 100, 40)
    rent = 6 + 1.8 * area + rng.normal(0, 12, area.size)      # increasing + noise
    x0 = 70.0                                                 # a split threshold
    left, right = area <= x0, area > x0
    m_all, m_l, m_r = rent.mean(), rent[left].mean(), rent[right].mean()
    sd_all = rent.std()
    sd_after = np.sqrt((left.sum() * rent[left].var() + right.sum() * rent[right].var()) / rent.size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax in (ax1, ax2):
        ax.scatter(area, rent, s=16, color="0.6", zorder=2)
        ax.set_xlabel(r"area (m$^2$)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    ax1.set_ylabel("rent (k AMD)", fontsize=9)

    ax1.axhline(m_all, color=ARM_ORANGE, lw=2.2)
    for a, r in zip(area, rent):
        ax1.plot([a, a], [m_all, r], color=ARM_ORANGE, lw=0.6, alpha=0.4, zorder=1)
    ax1.set_title(f"1 leaf: predict the mean\nspread (std) $= {sd_all:.0f}$", fontsize=9.5)

    ax2.axvline(x0, color="0.4", ls="--", lw=1.5)
    ax2.plot([area[left].min(), x0], [m_l, m_l], color=ARM_BLUE, lw=2.6)
    ax2.plot([x0, area[right].max()], [m_r, m_r], color=ARM_BLUE, lw=2.6)
    for a, r, is_l in zip(area, rent, left):
        ax2.plot([a, a], [m_l if is_l else m_r, r], color=ARM_BLUE, lw=0.6, alpha=0.4, zorder=1)
    ax2.set_title(f"2 leaves: each predicts its mean\nspread (std) $= {sd_after:.0f}$  (reduced)",
                  fontsize=9.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_split.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote regression_split.pdf (std %.1f -> %.1f)", sd_all, sd_after)


def fig_predict_leaf(X, y, feature_names, logger):
    """Classification prediction = the proportion of 1s in the leaf. Fit a depth-2
    Titanic tree, take a representative leaf, and show its mix as a 10x10 waffle."""
    from matplotlib.lines import Line2D
    clf = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X, y)
    leaves = clf.apply(X)
    best_id, best_p = None, None
    for lid in np.unique(leaves):
        mask = leaves == lid
        n, p = int(mask.sum()), float(y[mask].mean())
        if n >= 30 and (best_p is None or abs(p - 0.75) < abs(best_p - 0.75)):
            best_id, best_p = lid, p
    mask = leaves == best_id
    n_tot, n1 = int(mask.sum()), int(y[mask].sum())
    p = n1 / n_tot
    k = int(round(100 * p))                       # blue (survived) cells out of 100

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(-2.4, 11.4)
    ax.axis("off")
    for i in range(100):
        r, c = divmod(i, 10)
        ax.scatter([c], [9 - r], s=125,
                   color=ARM_BLUE if i < k else ARM_RED,
                   edgecolors="white", linewidths=0.8)
    ax.text(4.5, 10.7, "One leaf of the tree", ha="center", fontsize=12.5, fontweight="bold")
    ax.text(4.5, -1.3,
            f"survived: {n1} of {n_tot}   $\\Rightarrow$   prediction $=$ proportion of 1s "
            f"$= {p:.2f}$", ha="center", fontsize=10.5)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_BLUE, markersize=9,
               label="survived (1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_RED, markersize=9,
               label="died (0)"),
    ], loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predict_leaf.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote predict_leaf.pdf (leaf %d: %d/%d survived, p=%.2f, waffle=%d/100)",
                best_id, n1, n_tot, p, k)


def fig_overfit(X, y, feature_names, logger):
    """Train accuracy vs CV-test accuracy across depth -> overfitting."""
    # NB: shuffle the CV folds -- OpenML Titanic is ordered by pclass, so the default
    # non-shuffled StratifiedKFold badly underestimates test accuracy.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    depths = list(range(1, 17))
    train_acc, cv_acc = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        clf.fit(X, y)
        train_acc.append(clf.score(X, y))
        cv_acc.append(cross_val_score(clf, X, y, cv=cv).mean())

    # the unrestricted tree (no depth limit)
    full = DecisionTreeClassifier(random_state=SEED).fit(X, y)
    full_train = full.score(X, y)
    full_cv = cross_val_score(DecisionTreeClassifier(random_state=SEED), X, y, cv=cv).mean()
    best_d = depths[int(np.argmax(cv_acc))]
    logger.info(f"unrestricted tree: depth={full.get_depth()}, "
                f"leaves={full.get_n_leaves()}, train acc={full_train:.3f}, "
                f"CV acc={full_cv:.3f}")
    logger.info(f"depth-1 stump CV acc={cv_acc[0]:.3f}")
    logger.info(f"best CV depth={best_d} (CV acc={max(cv_acc):.3f}); "
                f"train acc there={train_acc[best_d-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(depths, train_acc, "-o", color=ARM_BLUE, lw=2, ms=3, label="train accuracy")
    ax.plot(depths, cv_acc, "-o", color=ARM_RED, lw=2, ms=3, label="test accuracy (5-fold CV)")
    ax.axvline(best_d, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate("sweet spot", xy=(best_d, max(cv_acc)),
                xytext=(best_d + 1.5, max(cv_acc) - 0.08),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel("max tree depth (complexity)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.6, 1.02)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "titanic_overfit.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(full_depth=full.get_depth(), full_leaves=full.get_n_leaves(),
                full_train=full_train, full_cv=full_cv,
                best_d=best_d, best_cv=max(cv_acc))


def fig_pruning(X, y, feature_names, logger):
    """Cost-complexity pruning path: accuracy vs ccp_alpha."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_tr, y_tr)
    alphas = path.ccp_alphas[:-1]            # drop the trivial root-only tree
    train_acc, test_acc = [], []
    for a in alphas:
        clf = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a).fit(X_tr, y_tr)
        train_acc.append(clf.score(X_tr, y_tr))
        test_acc.append(clf.score(X_te, y_te))
    best_i = int(np.argmax(test_acc))
    best_alpha = alphas[best_i]
    logger.info(f"best ccp_alpha={best_alpha:.5f} "
                f"(test acc={test_acc[best_i]:.3f}, train acc={train_acc[best_i]:.3f})")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(alphas, train_acc, "-o", color=ARM_BLUE, lw=2, ms=3, label="train accuracy")
    ax.plot(alphas, test_acc, "-o", color=ARM_RED, lw=2, ms=3, label="test accuracy")
    ax.axvline(best_alpha, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate("best pruned tree", xy=(best_alpha, test_acc[best_i]),
                xytext=(best_alpha + 0.004, test_acc[best_i] - 0.12),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel(r"ccp_alpha (cost per leaf)")
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "titanic_pruning.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(best_alpha=best_alpha, best_test=test_acc[best_i])


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    X, y, feats = load_titanic(logger)
    fig_tree(X, y, feats, logger)
    over = fig_overfit(X, y, feats, logger)
    prune = fig_pruning(X, y, feats, logger)
    fig_staircase(logger)
    fig_extrapolation(logger)
    fig_greedy_xor(logger)
    fig_split_animation(logger)
    fig_akinator_ig(logger)
    fig_impurity_pair(logger)
    fig_ig_split(logger)
    fig_regression_split(logger)
    fig_predict_leaf(X, y, feats, logger)
    logger.info("=== SUMMARY for the slides ===")
    logger.info(f"unrestricted: depth={over['full_depth']}, leaves={over['full_leaves']}, "
                f"train={over['full_train']:.2f}, CV={over['full_cv']:.2f}")
    logger.info(f"best depth={over['best_d']} (CV={over['best_cv']:.2f}); "
                f"best ccp_alpha={prune['best_alpha']:.4f} (test={prune['best_test']:.2f})")
    logger.info("done.")


if __name__ == "__main__":
    main()
