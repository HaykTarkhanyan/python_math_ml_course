"""Generate the figures for the [19] Boosting deck (incl. the headline animation).

Outputs into ``ml/04_trees/fig/``:
  1. boost_anim_1..6.pdf  -- the 6-round gradient-boosting ANIMATION (Beamer overlays).
                             2-panel per round: LEFT = data + cumulative ensemble fit F_k
                             (+ the previous fit, faint, so the new stump's effect shows);
                             RIGHT = residuals r_k = y - F_k collapsing toward zero.
  2. boost_anim_50.pdf     -- the same 2-panel at round 50: fit barely moves, residuals
                             already flat -> diminishing returns (an extra animation overlay).
  3. boost_overfit.pdf     -- a DELIBERATELY over-powered config (deep trees, eta=1) so
                             train MSE -> ~0 and test MSE makes a clear U (early-stopping).
  4. boost_learning_rate.pdf -- test error vs #trees for two learning rates (general trend:
                             smaller eta needs more trees).
  5. adaboost_rounds.pdf   -- AdaBoost, 3 rounds: point size = sample weight (misclassified
                             points grow), plus each round's stump boundary.
  6. boost_gb_classification.pdf -- classification GB: LEFT = F accumulating in log-odds
                             space; RIGHT = sigmoid(F) tightening toward the labels.

Also LOGS round-1 details (F_0 = mean(y), the first stump's split threshold + the two leaf
values + eta) so the deck's by-hand frame uses REAL numbers that MATCH animation round 1.

Run with the project venv (repo CLAUDE.md):
    ./ma/Scripts/python.exe ml/04_trees/py_src/make_boost_figures.py

Conventions: logging to console + logs/, seed 509, f-strings, Armenian-flag palette.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier

SEED = 509

# Armenian flag palette.
ARM_BLUE = "#0033A0"     # data
ARM_RED = "#D90012"      # current ensemble fit / test
ARM_ORANGE = "#F2A800"   # highlight / previous fit

# Animation learning rate -- pinned so all 6 depth-1 rounds are visibly distinct.
# (Reviewer S1: too small -> panels look identical; too large -> jagged. Eyeball output.)
ETA_ANIM = 0.4
N_ROUNDS = 6

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l11_figures")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "make_boost_figures.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def make_data(n=50, noise=22.0, wiggly=False):
    """Synthetic Yerevan rent (kAMD) vs area (m^2): nonlinear + noise.

    wiggly=True adds a higher-frequency term so the signal takes many shallow trees to
    fit -> a long, visible descent before the overfitting rise (for the U-curve figs)."""
    rng = np.random.default_rng(SEED)
    x = np.sort(rng.uniform(25, 115, n))
    y_true = 150 + 2.3 * x + 70 * np.sin(x / 18.0)
    if wiggly:
        y_true = y_true + 40 * np.sin(x / 5.5)
    y = y_true + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y


def fig_animation(X, y, logger):
    """6-round GB animation on stumps; writes boost_anim_1..6.pdf."""
    gbr = GradientBoostingRegressor(
        n_estimators=N_ROUNDS, max_depth=1, learning_rate=ETA_ANIM, random_state=SEED)
    gbr.fit(X, y)

    x = X.ravel()
    grid = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    init = float(y.mean())
    # per-tree contributions on the fine grid (leaf values are pre-shrinkage -> multiply by eta)
    contribs = np.array([est[0].predict(grid) for est in gbr.estimators_])
    contribs_X = np.array([est[0].predict(X) for est in gbr.estimators_])
    F_grid = [np.full(grid.shape[0], init)]
    F_X = [np.full(y.shape[0], init)]
    for k in range(N_ROUNDS):
        F_grid.append(F_grid[-1] + ETA_ANIM * contribs[k])
        F_X.append(F_X[-1] + ETA_ANIM * contribs_X[k])

    # log round-1 details for the by-hand slide
    t0 = gbr.estimators_[0, 0].tree_
    thr = float(t0.threshold[0])
    left_val = float(t0.value[t0.children_left[0]][0, 0])
    right_val = float(t0.value[t0.children_right[0]][0, 0])
    logger.info(f"BY-HAND round 1: F_0 = mean(y) = {init:.1f} kAMD; eta = {ETA_ANIM}")
    logger.info(f"BY-HAND round 1: first stump splits at area = {thr:.1f} m^2; "
                f"residual-mean leaf values: left(area<={thr:.1f}) = {left_val:+.1f}, "
                f"right = {right_val:+.1f}")
    logger.info(f"BY-HAND round 1: F_1 = F_0 + eta*leaf -> left {init + ETA_ANIM*left_val:.1f}, "
                f"right {init + ETA_ANIM*right_val:.1f}")

    r0 = y - init
    rmax = np.abs(r0).max() * 1.1
    ylim_fit = (y.min() - 25, y.max() + 25)

    for k in range(1, N_ROUNDS + 1):
        resid = y - F_X[k]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.7))
        # LEFT: data + previous fit (faint) + current fit
        axL.scatter(x, y, s=16, color=ARM_BLUE, alpha=0.7, zorder=3, label="data")
        if k > 1:
            axL.plot(grid.ravel(), F_grid[k - 1], color=ARM_ORANGE, lw=1.4, alpha=0.55,
                     label=f"$F_{{{k-1}}}$ (before)")
        axL.plot(grid.ravel(), F_grid[k], color=ARM_RED, lw=2.4, label=f"$F_{{{k}}}$ (now)")
        axL.set_title(f"round {k}: ensemble of {k} stump(s)", fontsize=10)
        axL.set_xlabel("area (m$^2$)"); axL.set_ylabel("rent (kAMD)")
        axL.set_ylim(*ylim_fit)
        axL.legend(loc="upper left", fontsize=8, frameon=False)
        axL.spines[["top", "right"]].set_visible(False)
        # RIGHT: residuals collapsing
        axR.axhline(0, color="gray", lw=1)
        axR.vlines(x, 0, resid, color="gray", alpha=0.5, lw=1)
        axR.scatter(x, resid, s=16, color=ARM_BLUE, alpha=0.75, zorder=3)
        axR.set_title(f"what's left: residuals $r_{{{k}}} = y - F_{{{k}}}$", fontsize=10)
        axR.set_xlabel("area (m$^2$)"); axR.set_ylabel("residual (kAMD)")
        axR.set_ylim(-rmax, rmax)
        axR.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out = FIG_DIR / f"boost_anim_{k}.pdf"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        logger.info(f"wrote {out.name}: train MSE after {k} rounds = "
                    f"{mean_squared_error(y, F_X[k]):.0f}")


def fig_overfit(logger):
    """Over-powered config -> train->0, test U-curve. Own larger, noisier dataset so the
    descend-then-rise U is clear (depth-4 + eta=1 on 30 pts overfits instantly = no U)."""
    X, y = make_data(n=160, noise=34.0, wiggly=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=SEED)
    n = 250
    gbr = GradientBoostingRegressor(n_estimators=n, max_depth=3, learning_rate=0.1,
                                    random_state=SEED)
    gbr.fit(Xtr, ytr)
    tr = [mean_squared_error(ytr, p) for p in gbr.staged_predict(Xtr)]
    te = [mean_squared_error(yte, p) for p in gbr.staged_predict(Xte)]
    best = int(np.argmin(te)) + 1
    logger.info(f"overfit: test MSE min at {best} trees ({min(te):.0f}); "
                f"train MSE at end = {tr[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rounds = np.arange(1, n + 1)
    ax.plot(rounds, tr, color=ARM_BLUE, lw=2, label="train MSE")
    ax.plot(rounds, te, color=ARM_RED, lw=2, label="test MSE")
    ax.axvline(best, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate("early-stop here", xy=(best, min(te)),
                xytext=(best + 40, min(te) + 0.4 * (max(te) - min(te))),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel("number of trees"); ax.set_ylabel("MSE")
    ax.set_title("enough trees: train -> 0, but test error U-turns", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "boost_overfit.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_learning_rate(logger):
    """Test MSE vs #trees for two learning rates (general trend)."""
    X, y = make_data(n=160, noise=34.0, wiggly=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=SEED)
    n = 150
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for eta, color, ls in [(0.5, ARM_RED, "-"), (0.1, ARM_BLUE, "-")]:
        gbr = GradientBoostingRegressor(n_estimators=n, max_depth=3, learning_rate=eta,
                                        random_state=SEED).fit(Xtr, ytr)
        te = [mean_squared_error(yte, p) for p in gbr.staged_predict(Xte)]
        ax.plot(np.arange(1, n + 1), te, color=color, ls=ls, lw=2,
                label=f"eta = {eta}")
        logger.info(f"learning_rate eta={eta}: test MSE min at "
                    f"{int(np.argmin(te))+1} trees ({min(te):.0f})")
    ax.set_xlabel("number of trees"); ax.set_ylabel("test MSE")
    ax.set_title("smaller eta needs more trees", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "boost_learning_rate.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_anim_late(X, y, logger):
    """Round 50 of the SAME animation: the fit barely moves and residuals are already
    flat -> diminishing returns. Same 2-panel style and residual scale as rounds 1-6."""
    K = 50
    gbr = GradientBoostingRegressor(
        n_estimators=K, max_depth=1, learning_rate=ETA_ANIM, random_state=SEED).fit(X, y)
    x = X.ravel()
    grid = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    staged_grid = list(gbr.staged_predict(grid))
    F_grid, F_grid_prev = staged_grid[-1], staged_grid[-7]     # 6 rounds earlier = "before"
    F_X = list(gbr.staged_predict(X))[-1]
    resid = y - F_X
    rmax = np.abs(y - y.mean()).max() * 1.1                    # same scale as rounds 1-6
    ylim_fit = (y.min() - 25, y.max() + 25)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axL.scatter(x, y, s=16, color=ARM_BLUE, alpha=0.7, zorder=3, label="data")
    axL.plot(grid.ravel(), F_grid_prev, color=ARM_ORANGE, lw=1.4, alpha=0.55,
             label="$F_{44}$ (before)")
    axL.plot(grid.ravel(), F_grid, color=ARM_RED, lw=2.4, label="$F_{50}$ (now)")
    axL.set_title("round 50: the fit barely moves anymore", fontsize=10)
    axL.set_xlabel("area (m$^2$)"); axL.set_ylabel("rent (kAMD)")
    axL.set_ylim(*ylim_fit)
    axL.legend(loc="upper left", fontsize=8, frameon=False)
    axL.spines[["top", "right"]].set_visible(False)
    axR.axhline(0, color="gray", lw=1)
    axR.vlines(x, 0, resid, color="gray", alpha=0.5, lw=1)
    axR.scatter(x, resid, s=16, color=ARM_BLUE, alpha=0.75, zorder=3)
    axR.set_title("residuals $r_{50}$: already flat", fontsize=10)
    axR.set_xlabel("area (m$^2$)"); axR.set_ylabel("residual (kAMD)")
    axR.set_ylim(-rmax, rmax)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "boost_anim_50.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name}: train MSE after 50 rounds = "
                f"{mean_squared_error(y, F_X):.0f}")


def fig_adaboost(logger):
    """AdaBoost reweighting, 3 rounds (manual SAMME so we can show the weights). Point
    size = sample weight; a RED ring marks the points THIS round's stump got wrong -- they
    grow next round. Each title shows the weighted error and the stump weight alpha."""
    from matplotlib.lines import Line2D
    Xs, ys = make_moons(n_samples=120, noise=0.28, random_state=SEED)
    ypm = np.where(ys == 1, 1.0, -1.0)
    n = len(ys)
    w = np.full(n, 1.0 / n)
    xx, yy = np.meshgrid(
        np.linspace(Xs[:, 0].min() - 0.3, Xs[:, 0].max() + 0.3, 300),
        np.linspace(Xs[:, 1].min() - 0.3, Xs[:, 1].max() + 0.3, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), sharex=True, sharey=True)
    for r in range(3):
        stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
        stump.fit(Xs, ys, sample_weight=w)
        pred = stump.predict(Xs)
        miss = pred != ys
        zz = stump.predict(grid).reshape(xx.shape)
        ax = axes[r]
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=[ARM_ORANGE, ARM_BLUE], alpha=0.15)
        sizes = 12 + w / w.max() * 150
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=sizes[ys == 0], color=ARM_ORANGE,
                   edgecolor="white", linewidths=0.4, alpha=0.85, zorder=3)
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=sizes[ys == 1], color=ARM_BLUE,
                   edgecolor="white", linewidths=0.4, alpha=0.85, zorder=3)
        ax.scatter(Xs[miss, 0], Xs[miss, 1], s=sizes[miss], facecolors="none",
                   edgecolors=ARM_RED, linewidths=1.6, zorder=4)
        err = min(max(w[miss].sum() / w.sum(), 1e-10), 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)
        ax.set_title(f"round {r + 1}:   err = {err:.2f},   " + r"$\alpha$ = " + f"{alpha:.2f}",
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        w = w * np.exp(-alpha * ypm * np.where(pred == 1, 1.0, -1.0))
        w = w / w.sum()
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markersize=10,
               label="dot size = sample weight"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor=ARM_RED, markerfacecolor="none",
               markeredgewidth=1.6, markersize=10, label="wrong this round -> up-weighted next"),
    ], loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIG_DIR / "adaboost_rounds.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_adaboost_combined(logger):
    """Weak -> strong, shown: one stump (a single straight cut) vs an AdaBoost of 50 stumps
    (a curved boundary that follows the moons). Reports both accuracies."""
    from sklearn.ensemble import AdaBoostClassifier
    Xs, ys = make_moons(n_samples=300, noise=0.30, random_state=SEED)
    xx, yy = np.meshgrid(
        np.linspace(Xs[:, 0].min() - 0.3, Xs[:, 0].max() + 0.3, 300),
        np.linspace(Xs[:, 1].min() - 0.3, Xs[:, 1].max() + 0.3, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xs, ys)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                             n_estimators=50, random_state=SEED).fit(Xs, ys)
    a_stump, a_ada = stump.score(Xs, ys), ada.score(Xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharex=True, sharey=True)
    for ax, clf, title in [
            (axes[0], stump, f"1 stump (weak): acc {a_stump:.2f}"),
            (axes[1], ada, f"50 weighted stumps (strong): acc {a_ada:.2f}")]:
        zz = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=[ARM_ORANGE, ARM_BLUE], alpha=0.20)
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=10, color=ARM_ORANGE,
                   edgecolor="white", linewidths=0.3)
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=10, color=ARM_BLUE,
                   edgecolor="white", linewidths=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = FIG_DIR / "adaboost_combined.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name}: stump acc={a_stump:.3f}, ada50 acc={a_ada:.3f}")


def fig_gb_classification(logger):
    """Classification GB: LEFT = the model F accumulating in log-odds space for a few
    points; RIGHT = sigmoid(F) tightening toward 0 / 1. Sigmoid is applied at the end.
    Each line = ONE training point's trajectory (3 per class), made explicit via legend."""
    from matplotlib.lines import Line2D
    Xs, ys = make_moons(n_samples=300, noise=0.30, random_state=SEED)
    gbc = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.3, random_state=SEED).fit(Xs, ys)
    F = np.array([f.ravel() for f in gbc.staged_decision_function(Xs)])   # (rounds, samples)
    rng = np.random.default_rng(SEED)
    idx0 = rng.choice(np.where(ys == 0)[0], 3, replace=False)
    idx1 = rng.choice(np.where(ys == 1)[0], 3, replace=False)
    rounds = np.arange(1, F.shape[0] + 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for i in idx1:
        axL.plot(rounds, F[:, i], color=ARM_BLUE, lw=2.0, alpha=0.9)
        axR.plot(rounds, 1 / (1 + np.exp(-F[:, i])), color=ARM_BLUE, lw=2.0, alpha=0.9)
    for i in idx0:
        axL.plot(rounds, F[:, i], color=ARM_ORANGE, lw=2.0, alpha=0.9)
        axR.plot(rounds, 1 / (1 + np.exp(-F[:, i])), color=ARM_ORANGE, lw=2.0, alpha=0.9)
    axL.axhline(0, color="gray", lw=1, ls=":")
    axL.set_title(r"each line = one point's score $F$ (log-odds)", fontsize=10)
    axL.set_xlabel("round"); axL.set_ylabel("$F$ (log-odds)")
    axL.spines[["top", "right"]].set_visible(False)
    axR.axhline(0.5, color="gray", lw=1, ls=":")
    axR.set_title(r"$\sigma(F)$: that same point's probability", fontsize=10)
    axR.set_xlabel("round"); axR.set_ylabel("predicted probability")
    axR.set_ylim(-0.02, 1.02)
    axR.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[
        Line2D([0], [0], color=ARM_BLUE, lw=2.2, label="a class-1 training point"),
        Line2D([0], [0], color=ARM_ORANGE, lw=2.2, label="a class-0 training point"),
    ], loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIG_DIR / "boost_gb_classification.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"wrote {out.name} (blue = a class-1 point, orange = a class-0 point)")


def fig_boost_meme(logger):
    """Outline gag for boosting: three guys taking turns, 'first me, then you, then me' --
    the sequential fix-the-last idea. Armenian caption baked in (pdflatex has no Armenian);
    the Outline frame hyperlinks the whole image to the video."""
    import matplotlib.image as mpimg
    img = mpimg.imread(str(FIG_DIR / "boost_meme_raw.png"))
    h, w = img.shape[:2]
    caption = "▶ սկզբից ես, հետո դու, հետո ես"   # 'first me, then you, then me'
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * h / w + 0.7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=19, color=ARM_BLUE, fontweight="bold", pad=8)
    fig.tight_layout()
    out = FIG_DIR / "boost_meme.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    fig_boost_meme(logger)
    X, y = make_data()
    logger.info(f"data: {X.shape[0]} points, rent mean = {y.mean():.1f} kAMD, "
                f"animation eta = {ETA_ANIM}")
    fig_animation(X, y, logger)
    fig_anim_late(X, y, logger)
    fig_overfit(logger)
    fig_learning_rate(logger)
    fig_adaboost(logger)
    fig_adaboost_combined(logger)
    fig_gb_classification(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
