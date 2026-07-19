"""Figures for Lecture 2 -- ML for time series (08_ml_time_series).

Generates PDFs into ml/07_time_series/fig/:
  time_aware_split.pdf   -- random split (leakage) vs forward-chaining split.
  supervised_reframe.pdf -- a series redrawn as a lag-feature design matrix.
  ts_cv.pdf              -- TimeSeriesSplit: expanding-window cross-validation.
  gbm_forecast.pdf       -- gradient boosting on lag+calendar features vs actual.
  feature_importance.pdf -- permutation importance of the engineered features.
  model_comparison.pdf   -- seasonal-naive vs SARIMA vs GBM (MAE + MASE) on test.
  deep_ts_timeline.pdf   -- landscape of deep and foundation TS models.

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/07_time_series/py_src/ml_figs.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg, fail loud (no silent except).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    n = 40
    tscv = TimeSeriesSplit(n_splits=5, test_size=5)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for k, (tr, te) in enumerate(tscv.split(np.arange(n))):
        y = 5 - k
        ax.scatter(tr, [y] * len(tr), marker="s", s=42, color=ARM_BLUE)
        ax.scatter(te, [y] * len(te), marker="s", s=42, color=ARM_ORANGE)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"fold {6 - k}" for k in range(1, 6)])
    ax.set_xlabel("time index ->")
    ax.set_ylim(0.4, 5.6)
    ax.grid(False)
    train_p = mpatches.Patch(color=ARM_BLUE, label="train (expanding past)")
    test_p = mpatches.Patch(color=ARM_ORANGE, label="validation (next block)")
    ax.legend(handles=[train_p, test_p], fontsize=9, loc="lower right")
    ax.set_title("TimeSeriesSplit: the train window only ever grows into the past",
                 fontsize=12, loc="left")
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
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    for ax, pred, title, c, mae in [
        (axes[0], pred_lvl, f"Predict raw level: trees cap at the training max\n(MAE = {mae_lvl:.1f})", ARM_RED, mae_lvl),
        (axes[1], pred_diff, f"Predict the difference, then add back\n(MAE = {mae_diff:.1f})", GREEN, mae_diff),
    ]:
        ax.plot(series.index[-40:-h], series.values[-40:-h], color=ARM_BLUE, lw=1.1, label="train")
        ax.plot(idx, actual, color="k", lw=1.7, label="actual")
        ax.plot(idx, pred, color=c, lw=2, marker="o", ms=3, label="GBM")
        ax.axhline(train_max, color=GREY, ls=":", lw=1)
        ax.axvline(series.index[-h - 1], color=GREY, ls="--", lw=1)
        ax.set_title(title, fontsize=10.5, loc="left", color=c)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].text(series.index[-h - 2], train_max, "training max ", ha="right", va="bottom",
                 fontsize=8, color=GREY)
    fig.suptitle("Gradient boosting on a trending series: the extrapolation trap and its fix",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "gbm_forecast.pdf", log)
    return dff, d_tr, d_te, m_diff, f_diff, pred_diff, actual


def fig_feature_importance(train_df, model, feats, log: logging.Logger) -> None:
    r = permutation_importance(model, train_df[feats], train_df["d"],
                               n_repeats=20, random_state=SEED)
    order = np.argsort(r.importances_mean)
    names = np.array(feats)[order]
    vals = r.importances_mean[order]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.barh(names, vals, color=ARM_BLUE)
    ax.set_xlabel("permutation importance (drop in R2)")
    ax.set_title("Which engineered feature carries the signal?", fontsize=12, loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "feature_importance.pdf", log)


def fig_model_comparison(series: pd.Series, gbm_pred, gbm_actual,
                         log: logging.Logger) -> None:
    h = 18
    test_idx = series.index[-h:]
    actual = gbm_actual

    # seasonal-naive: y_hat(t) = y(t-12)
    snaive = series.shift(12).reindex(test_idx).values
    # SARIMA on the raw series
    train_series = series.iloc[:-h]
    sar = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sar.get_forecast(steps=h).predicted_mean.values

    # MASE denominator: in-sample seasonal-naive MAE on the training series
    naive_err = np.abs(train_series.values[12:] - train_series.values[:-12]).mean()

    def mase(a, p):
        return mean_absolute_error(a, p) / naive_err

    rows = [("Seasonal naive", snaive, ARM_ORANGE),
            ("SARIMA", sarima_pred, ARM_BLUE),
            ("GBM (differenced)", gbm_pred, ARM_RED)]
    labels, maes, mases, colors = [], [], [], []
    for name, pred, c in rows:
        labels.append(name)
        maes.append(mean_absolute_error(actual, pred))
        mases.append(mase(actual, pred))
        colors.append(c)
        log.info(f"{name}: MAE={maes[-1]:.2f} MASE={mases[-1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, vals, title in [(axes[0], maes, "MAE (lower is better)"),
                            (axes[1], mases, "MASE (vs seasonal naive = 1.0)")]:
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=8.5, rotation=12)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.grid(axis="x")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    fig.suptitle("Always benchmark against the naive baseline", fontsize=12)
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
    ax.axis("off")
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
    fig_feature_importance(d_tr, m_diff, f_diff, log)
    fig_model_comparison(series, gbm_pred, gbm_actual, log)
    fig_deep_timeline(log)
    log.info("ml figures done")


if __name__ == "__main__":
    main()
