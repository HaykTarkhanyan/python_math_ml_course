"""Figures for Lecture 2 -- ML for time series (31_ml_time_series).

Generates PDFs into ml/08_time_series/fig/:
  time_aware_split.pdf   -- random split (leakage) vs forward-chaining split.
  supervised_reframe.pdf -- a series redrawn as a lag-feature design matrix.
  ts_cv.pdf              -- TimeSeriesSplit: expanding vs rolling vs gapped.
  gbm_forecast.pdf       -- gradient boosting on lag+calendar features vs actual.
  feature_importance.pdf -- permutation importance on TEST of the engineered features.
  model_comparison.pdf   -- seasonal-naive vs SARIMA vs GBM, all at h=18 (MAE + MASE).
  deep_ts_timeline.pdf   -- landscape of deep and foundation TS models.

Review pass 2026-07-31 fixed three methodology defects that contradicted the slides:
the seasonal-naive baseline read 6 of its 18 values out of the test window; permutation
importance was scored on train (lecture 23 says test); and the GBM was scored 1 step
ahead while SARIMA forecast 18, with the horizon stated nowhere.

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/08_time_series/py_src/ml_figs.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg, fail loud (no silent except).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEED = 509
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"
GREEN = "#008C46"
GREY = "#5a5a5a"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l08_ml_ts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "ml_figs.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def save(fig, name: str, log: logging.Logger) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.relative_to(REPO_ROOT)}")


def synthetic_monthly(n_years: int = 8, start: str = "2016-01") -> pd.Series:
    """Same generator as classical_figs.py (kept self-contained per repo style)."""
    rng = np.random.default_rng(SEED)
    n = n_years * 12
    idx = pd.period_range(start=start, periods=n, freq="M").to_timestamp()
    t = np.arange(n)
    trend = 100 + 1.4 * t
    season = 18 * np.sin(2 * np.pi * (t % 12) / 12) + 8 * np.cos(2 * np.pi * (t % 12) / 6)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 6)
    return pd.Series(trend + season + noise, index=idx, name="sales")


def make_supervised(series: pd.Series, lags=(1, 2, 3, 12)) -> pd.DataFrame:
    """Predict the raw LEVEL from lagged levels + calendar (used for the pitfall demo)."""
    df = pd.DataFrame({"y": series})
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    df["roll_mean_3"] = series.shift(1).rolling(3).mean()
    df["month"] = series.index.month
    return df.dropna()


def make_supervised_diff(series: pd.Series, lags=(1, 2, 3, 12)) -> pd.DataFrame:
    """Predict the 1st DIFFERENCE (stationary target) so trees never extrapolate.

    Target d(t) = y(t) - y(t-1). Features are lagged differences + calendar.
    Reconstruct the level as y_hat(t) = y(t-1) + d_hat(t).
    """
    d = series.diff()
    df = pd.DataFrame({"d": d, "y_prev": series.shift(1)})
    for L in lags:
        df[f"dlag_{L}"] = d.shift(L)
    df["month"] = series.index.month
    return df.dropna()


def seasonal_naive_from_origin(series: pd.Series, h: int, m: int = 12) -> np.ndarray:
    """Seasonal naive for an h-step forecast issued ONCE at the end of the training set.

    yhat(T+k) = y(T + k - m*ceil(k/m)), which is always an OBSERVED training value.

    The tempting one-liner `series.shift(m)` is wrong here: for k > m it reaches back
    only m steps and lands inside the test window, so the "baseline" quietly reads the
    future it is supposed to be predicting. On this series that is 6 of the 18 points.
    """
    n_train = len(series) - h
    out = []
    for k in range(1, h + 1):
        pos = n_train + k - 1 - m * int(np.ceil(k / m))
        if pos >= n_train:
            raise ValueError(f"seasonal naive step {k} would read position {pos} "
                             f"from the test window (train ends at {n_train - 1})")
        out.append(series.values[pos])
    return np.array(out)


def recursive_forecast(model, feats, series: pd.Series, h: int, lags=(1, 2, 3, 12)) -> np.ndarray:
    """Genuine h-step forecast: feed each PREDICTED difference back in as the next lag.

    This is what SARIMA's get_forecast(steps=h) does, so it is the only version that
    can be put on the same bar chart as SARIMA.
    """
    d_all = series.diff().dropna()
    d_hist = list(d_all.values[:len(d_all) - h])
    level = series.values[len(series) - h - 1]
    out = []
    for k in range(h):
        row = {f"dlag_{L}": d_hist[-L] for L in lags}
        row["month"] = series.index[len(series) - h + k].month
        dhat = model.predict(pd.DataFrame([row])[feats])[0]
        level = level + dhat
        out.append(level)
        d_hist.append(dhat)
    return np.array(out)


# ----------------------------------------------------------------------------
def fig_time_aware_split(series: pd.Series, log: logging.Logger) -> None:
    n = len(series)
    rng = np.random.default_rng(SEED)
    test_mask_random = np.zeros(n, dtype=bool)
    test_mask_random[rng.choice(n, size=n // 4, replace=False)] = True
    cut = int(n * 0.75)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True)
    # random split (wrong)
    ax = axes[0]
    ax.plot(series.index, series.values, color=GREY, lw=0.8, zorder=1)
    ax.scatter(series.index[~test_mask_random], series.values[~test_mask_random],
               s=10, color=ARM_BLUE, label="train")
    ax.scatter(series.index[test_mask_random], series.values[test_mask_random],
               s=14, color=ARM_RED, label="test")
    ax.set_title("Random split  ->  test points sit BEFORE train points = leakage",
                 fontsize=11, loc="left", color=ARM_RED)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    # temporal split (right)
    ax = axes[1]
    ax.plot(series.index, series.values, color=GREY, lw=0.8, zorder=1)
    ax.scatter(series.index[:cut], series.values[:cut], s=10, color=ARM_BLUE, label="train (past)")
    ax.scatter(series.index[cut:], series.values[cut:], s=14, color=GREEN, label="test (future)")
    ax.axvline(series.index[cut], color="k", ls="--", lw=1)
    ax.set_title("Forward split  ->  train on the past, test on the future = honest",
                 fontsize=11, loc="left", color=GREEN)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "time_aware_split.pdf", log)


def fig_supervised_reframe(series: pd.Series, log: logging.Logger) -> None:
    df = make_supervised(series, lags=(1, 2, 3)).head(6)
    show = df[["lag_3", "lag_2", "lag_1", "y"]].round(0).astype(int)
    show.columns = ["y(t-3)", "y(t-2)", "y(t-1)", "y(t) = target"]

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.7)
    ncol = show.shape[1]
    for j in range(ncol):
        tbl[0, j].set_facecolor(ARM_BLUE if j < ncol - 1 else ARM_RED)
        tbl[0, j].set_text_props(color="white", weight="bold")
        for i in range(1, show.shape[0] + 1):
            tbl[i, j].set_facecolor("#eef2fb" if j < ncol - 1 else "#fdeceb")
    ax.set_title("Forecasting IS supervised learning: sliding a window makes (X, y) rows",
                 fontsize=12, pad=14)
    fig.tight_layout()
    save(fig, "supervised_reframe.pdf", log)


def fig_ts_cv(log: logging.Logger) -> None:
    """Three flavours of forward-chaining CV: expanding, rolling, and gapped."""
    n = 40
    variants = [
        ("expanding window\n(the default)", dict(n_splits=5, test_size=5)),
        ("rolling window\n(max_train_size=15)", dict(n_splits=5, test_size=5, max_train_size=15)),
        ("expanding + gap=3\n(kills lag leakage)", dict(n_splits=5, test_size=5, gap=3)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), sharey=True)
    for ax, (title, kw) in zip(axes, variants):
        tscv = TimeSeriesSplit(**kw)
        for k, (tr, te) in enumerate(tscv.split(np.arange(n))):
            y = 5 - k
            ax.scatter(tr, [y] * len(tr), marker="s", s=30, color=ARM_BLUE)
            ax.scatter(te, [y] * len(te), marker="s", s=30, color=ARM_ORANGE)
            dropped = sorted(set(range(tr[-1] + 1, te[0])))
            if dropped:
                ax.scatter(dropped, [y] * len(dropped), marker="x", s=34, color=ARM_RED)
        ax.set_yticks(range(1, 6))
        ax.set_yticklabels([f"fold {6 - k}" for k in range(1, 6)])
        ax.set_xlabel("time index ->", fontsize=10)
        ax.set_ylim(0.4, 5.6)
        ax.grid(False)
        ax.set_title(title, fontsize=11, loc="left")
        ax.tick_params(labelsize=10)
    handles = [mpatches.Patch(color=ARM_BLUE, label="train"),
               mpatches.Patch(color=ARM_ORANGE, label="validation"),
               mpatches.Patch(color=ARM_RED, label="discarded (gap)")]
    fig.legend(handles=handles, fontsize=10, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.suptitle("Forward-chaining cross-validation: the validation block is always in the future",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "ts_cv.pdf", log)


def _fit_gbm_level(train_df):
    feats = [c for c in train_df.columns if c != "y"]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_depth=3, random_state=SEED)
    model.fit(train_df[feats], train_df["y"])
    return model, feats


def _fit_gbm_diff(train_df):
    feats = [c for c in train_df.columns if c not in ("d", "y_prev")]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_depth=3, random_state=SEED)
    model.fit(train_df[feats], train_df["d"])
    return model, feats


def fig_gbm_pitfall(series: pd.Series, log: logging.Logger):
    """Trees can't extrapolate: raw-level GBM under-forecasts the trend; the
    differenced GBM tracks it. Returns the differenced-model pieces for reuse."""
    h = 18
    # (a) raw levels -- the pitfall
    lvl = make_supervised(series)
    lvl_tr, lvl_te = lvl.iloc[:-h], lvl.iloc[-h:]
    m_lvl, f_lvl = _fit_gbm_level(lvl_tr)
    pred_lvl = m_lvl.predict(lvl_te[f_lvl])
    mae_lvl = mean_absolute_error(lvl_te["y"], pred_lvl)

    # (b) differenced target -- the fix
    dff = make_supervised_diff(series)
    d_tr, d_te = dff.iloc[:-h], dff.iloc[-h:]
    m_diff, f_diff = _fit_gbm_diff(d_tr)
    pred_d = m_diff.predict(d_te[f_diff])
    pred_diff = d_te["y_prev"].values + pred_d          # reconstruct the level
    actual = (d_te["y_prev"].values + d_te["d"].values)
    mae_diff = mean_absolute_error(actual, pred_diff)
    log.info(f"GBM raw-level MAE={mae_lvl:.2f}   GBM differenced MAE={mae_diff:.2f}")

    idx = d_te.index
    train_max = series.iloc[:-h].max()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, pred, title, c in [
        (axes[0], pred_lvl,
         f"Predict the raw level: stuck under the ceiling\n(MAE = {mae_lvl:.1f})", ARM_RED),
        (axes[1], pred_diff,
         f"Predict the difference, then add back\n(MAE = {mae_diff:.1f})", GREEN),
    ]:
        ax.plot(series.index[-40:-h], series.values[-40:-h], color=ARM_BLUE, lw=1.3, label="train")
        ax.plot(idx, actual, color="k", lw=1.9, label="actual")
        ax.plot(idx, pred, color=c, lw=2.2, marker="o", ms=3.5, label="GBM")
        ax.axhline(train_max, color=GREY, ls=":", lw=1.2)
        ax.axvline(series.index[-h - 1], color=GREY, ls="--", lw=1)
        ax.set_title(title, fontsize=11.5, loc="left", color=c)
        # lower right is the only quadrant both panels leave empty; upper left is
        # needed for the "largest y seen in training" annotation
        ax.legend(fontsize=9.5, loc="lower right")
        # one tick per year, else the monthly labels collide into an unreadable smear
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=9.5)
    axes[0].text(series.index[-h - 2], train_max, "largest y seen in training ",
                 ha="right", va="bottom", fontsize=9, color=GREY)
    fig.suptitle("Gradient boosting on a trending series: the extrapolation trap and its fix",
                 fontsize=12.5)
    fig.text(0.5, -0.03,
             "Both panels are 1-step-ahead: the model is handed the TRUE previous value each month. "
             "Even so, the raw-level tree cannot climb.",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "gbm_forecast.pdf", log)
    log.info(f"training max = {train_max:.1f}; raw-level GBM predicts "
             f"{pred_lvl.min():.1f}-{pred_lvl.max():.1f} (it plateaus BELOW the ceiling, "
             f"at its top leaf's mean)")
    return dff, d_tr, d_te, m_diff, f_diff, pred_diff, actual


def fig_feature_importance(test_df, model, feats, log: logging.Logger) -> None:
    """Permutation importance on the TEST split -- lecture 23's rule, applied here too.

    Scoring on train would report how much the model *leans* on each feature; on test it
    reports how much each feature actually buys on unseen data. The two disagree here:
    on test, `month` overtakes `dlag_3` and `dlag_1` goes negative.
    """
    r = permutation_importance(model, test_df[feats], test_df["d"],
                               n_repeats=20, random_state=SEED)
    order = np.argsort(r.importances_mean)
    names = np.array(feats)[order]
    vals = r.importances_mean[order]
    colors = [ARM_RED if v < 0 else ARM_BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.barh(names, vals, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=9.5, padding=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("permutation importance on test (drop in $R^2$)", fontsize=10.5)
    ax.set_title("Which engineered feature actually buys accuracy?", fontsize=12.5, loc="left")
    ax.tick_params(labelsize=10.5)
    # rcParams turns BOTH axes on, so switch everything off before re-enabling the one
    # we want. Gridlines have to run ACROSS the bars to be readable, never along them.
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.margins(x=0.16)         # room for the end-aligned labels
    fig.tight_layout()
    save(fig, "feature_importance.pdf", log)
    for n, v in zip(names[::-1], vals[::-1]):
        log.info(f"  PFI(test) {n:10s} = {v:+.3f}")


def fig_model_comparison(series: pd.Series, model, feats, gbm_1step, gbm_actual,
                         log: logging.Logger) -> None:
    """All contenders on ONE horizon (18 steps from a fixed origin), plus a deliberate
    apples-to-oranges bar so students see that the horizon has to be stated."""
    h = 18
    actual = gbm_actual
    train_series = series.iloc[:-h]

    snaive = seasonal_naive_from_origin(series, h)
    # airline model -- orders read off the correlogram in lecture 30, not guessed.
    # Keep in sync with classical_figs.ORDER / SEASONAL_ORDER.
    sar = SARIMAX(train_series, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sar.get_forecast(steps=h).predicted_mean.values
    gbm_rec = recursive_forecast(model, feats, series, h)

    # MASE denominator: the IN-SAMPLE one-step seasonal-naive MAE on the training set
    # (Hyndman's definition). That is why the held-out seasonal naive need not land on 1.0.
    naive_err = np.abs(train_series.values[12:] - train_series.values[:-12]).mean()
    log.info(f"MASE denominator (in-sample seasonal-naive MAE on train) = {naive_err:.2f}")

    rows = [("Seasonal naive\n(h=18)", snaive, ARM_ORANGE),
            ("SARIMA\n(h=18)", sarima_pred, ARM_BLUE),
            ("GBM recursive\n(h=18)", gbm_rec, GREEN),
            ("GBM\n(h=1)", gbm_1step, ARM_RED)]
    labels, maes, mases, colors = [], [], [], []
    for name, pred, c in rows:
        labels.append(name)
        maes.append(mean_absolute_error(actual, pred))
        mases.append(maes[-1] / naive_err)
        colors.append(c)
        log.info(f"{name.replace(chr(10), ' '):26s} MAE={maes[-1]:6.2f} MASE={mases[-1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, vals, title, fmt in [(axes[0], maes, "MAE (lower is better)", "%.2f"),
                                 (axes[1], mases, "MASE (< 1 beats the naive benchmark)", "%.2f")]:
        bars = ax.bar(labels, vals, color=colors)
        ax.bar_label(bars, fmt=fmt, fontsize=10, padding=2)
        ax.set_title(title, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.grid(False)         # rcParams enables both axes; clear then re-enable one
        ax.grid(True, axis="y")
        ax.margins(y=0.18)
        # the h=1 bar is a different task -- fence it off
        ax.axvline(2.5, color=GREY, ls="--", lw=1.2)
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    fig.suptitle("Same series, same test window -- but only the first three answer the same question",
                 fontsize=12)
    fig.text(0.5, -0.04,
             "Right of the dashed line: the identical GBM scored 1 step ahead instead of 18. "
             "Not better or worse - a different question.",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "model_comparison.pdf", log)


def fig_deep_timeline(log: logging.Logger) -> None:
    milestones = [
        (2017, "Transformer", ARM_BLUE, 1),
        (2018, "DeepAR", ARM_BLUE, -1),
        (2019, "N-BEATS", ARM_BLUE, 1),
        (2020, "TFT", ARM_BLUE, -1),
        (2023, "PatchTST", ARM_ORANGE, 1),
        (2023, "TimeGPT", ARM_RED, -1.6),
        (2024, "TimesFM\nChronos\nMoirai", ARM_RED, 1.2),
        (2025, "Chronos-2\nMoirai-2\nTime-MoE", ARM_RED, -1.2),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.axhline(0, color=GREY, lw=2)
    for year, name, c, side in milestones:
        ax.plot(year, 0, "o", color=c, ms=9, zorder=3)
        ax.annotate(name, xy=(year, 0), xytext=(year, side * 0.55),
                    ha="center", va="center", fontsize=8.8, color=c, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1))
    ax.annotate("classical DL for TS", xy=(2018.5, 1.6), fontsize=9, color=ARM_BLUE, ha="center")
    ax.annotate("foundation models\n(zero-shot forecasting)", xy=(2024.3, 1.9),
                fontsize=9, color=ARM_RED, ha="center")
    ax.set_xlim(2016.3, 2025.8)
    ax.set_ylim(-2.2, 2.4)
    # axis("off") would hide the YEARS -- a timeline without dates. Strip everything
    # except the x tick labels instead.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(range(2017, 2026))
    ax.tick_params(axis="x", labelsize=9.5, length=0, colors=GREY)
    ax.grid(False)
    ax.set_title("Deep and foundation models for time series", fontsize=12)
    fig.tight_layout()
    save(fig, "deep_ts_timeline.pdf", log)


def main() -> None:
    log = setup_logging()
    np.random.seed(SEED)
    log.info(f"seed={SEED} fig_dir={FIG_DIR}")
    series = synthetic_monthly()
    fig_time_aware_split(series, log)
    fig_supervised_reframe(series, log)
    fig_ts_cv(log)
    dff, d_tr, d_te, m_diff, f_diff, gbm_pred, gbm_actual = fig_gbm_pitfall(series, log)
    fig_feature_importance(d_te, m_diff, f_diff, log)
    fig_model_comparison(series, m_diff, f_diff, gbm_pred, gbm_actual, log)
    fig_deep_timeline(log)
    log.info("ml figures done")


if __name__ == "__main__":
    main()
