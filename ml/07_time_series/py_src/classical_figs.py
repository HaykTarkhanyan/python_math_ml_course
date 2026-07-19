"""Figures for Lecture 1 -- Classical time series (07_classical_time_series).

Generates PDFs into ml/07_time_series/fig/:
  ts_anatomy.pdf       -- one series annotated: trend + seasonality + noise.
  stl_decomposition.pdf-- STL: observed / trend / seasonal / residual.
  stationarity.pdf     -- stationary vs 3 kinds of non-stationary.
  differencing.pdf     -- non-stationary series -> differenced, with ADF p-values.
  acf_pacf.pdf         -- ACF & PACF of an AR(2) process (how to read the orders).
  arima_forecast.pdf   -- SARIMA forecast on a held-out tail, with 95% interval.
  exp_smoothing.pdf    -- naive / Holt / Holt-Winters forecasts compared.

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/07_time_series/py_src/classical_figs.py

Conventions (repo CLAUDE.md): console + logs/ logging, fixed seed, f-strings,
Armenian-flag colours, matplotlib Agg (no display), fail loud (no silent except).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

SEED = 509
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"
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
    logger = logging.getLogger("l07_classical_ts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "classical_figs.log")
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
    """Monthly series: linear trend + yearly seasonality + AR(1)-ish noise.

    Deterministic given SEED. Roughly airline-passengers flavoured but additive.
    """
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


# ----------------------------------------------------------------------------
def fig_anatomy(series: pd.Series, log: logging.Logger) -> None:
    """One series with the three additive parts drawn separately below it."""
    t = np.arange(len(series))
    trend = 100 + 1.4 * t
    season = 18 * np.sin(2 * np.pi * (t % 12) / 12) + 8 * np.cos(2 * np.pi * (t % 12) / 6)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.2), sharex=True)
    axes[0].plot(series.index, series.values, color=ARM_BLUE, lw=1.6)
    axes[0].set_ylabel("observed")
    axes[0].set_title("A time series = trend + seasonality + noise", fontsize=12, loc="left")

    axes[1].plot(series.index, trend, color=ARM_RED, lw=2)
    axes[1].set_ylabel("trend")

    axes[2].plot(series.index, season, color=ARM_ORANGE, lw=1.4)
    axes[2].set_ylabel("seasonality")

    axes[3].plot(series.index, series.values - trend - season, color=GREY, lw=1)
    axes[3].axhline(0, color="k", lw=0.6)
    axes[3].set_ylabel("noise")
    fig.align_ylabels(axes)
    save(fig, "ts_anatomy.pdf", log)


def fig_stl(series: pd.Series, log: logging.Logger) -> None:
    res = STL(series, period=12, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.2), sharex=True)
    for ax, data, name, c in [
        (axes[0], res.observed, "observed", ARM_BLUE),
        (axes[1], res.trend, "trend", ARM_RED),
        (axes[2], res.seasonal, "seasonal", ARM_ORANGE),
        (axes[3], res.resid, "resid", GREY),
    ]:
        ax.plot(series.index, data, color=c, lw=1.3)
        ax.set_ylabel(name)
    axes[0].set_title("STL decomposition (statsmodels)", fontsize=12, loc="left")
    fig.align_ylabels(axes)
    save(fig, "stl_decomposition.pdf", log)


def fig_stationarity(log: logging.Logger) -> None:
    rng = np.random.default_rng(SEED)
    n = 200
    t = np.arange(n)
    stationary = rng.normal(0, 1, n)
    trending = 0.05 * t + rng.normal(0, 1, n)
    var_growing = rng.normal(0, 1, n) * (1 + 0.02 * t)
    seasonal = 3 * np.sin(2 * np.pi * t / 25) + rng.normal(0, 0.6, n)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    panels = [
        (axes[0, 0], stationary, "Stationary", ARM_BLUE, True),
        (axes[0, 1], trending, "Non-stationary: trend in mean", ARM_RED, False),
        (axes[1, 0], var_growing, "Non-stationary: growing variance", ARM_RED, False),
        (axes[1, 1], seasonal, "Non-stationary: seasonality", ARM_ORANGE, False),
    ]
    for ax, data, title, c, ok in panels:
        ax.plot(t, data, color=c, lw=1.1)
        ax.set_title(title, fontsize=10.5,
                     color=(ARM_BLUE if ok else ARM_RED))
    fig.suptitle("Stationary vs non-stationary", fontsize=12)
    fig.tight_layout()
    save(fig, "stationarity.pdf", log)


def fig_differencing(series: pd.Series, log: logging.Logger) -> None:
    d1 = series.diff().dropna()
    p_raw = adfuller(series)[1]
    p_diff = adfuller(d1)[1]
    log.info(f"ADF p-value raw={p_raw:.3f}  differenced={p_diff:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.8), sharex=False)
    axes[0].plot(series.index, series.values, color=ARM_RED, lw=1.4)
    axes[0].set_title(f"Original: trending, ADF p = {p_raw:.2f}  "
                      f"({'non-stationary' if p_raw > 0.05 else 'stationary'})",
                      fontsize=11, loc="left", color=ARM_RED)
    axes[1].plot(d1.index, d1.values, color=ARM_BLUE, lw=1.1)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title(f"After 1st differencing $y_t - y_{{t-1}}$: ADF p = {p_diff:.3f}  "
                      f"({'stationary' if p_diff < 0.05 else 'non-stationary'})",
                      fontsize=11, loc="left", color=ARM_BLUE)
    fig.tight_layout()
    save(fig, "differencing.pdf", log)


def fig_acf_pacf(log: logging.Logger) -> None:
    # AR(2) process: PACF cuts off at lag 2, ACF decays.
    ar = np.array([1, -0.6, -0.25])   # 1 - 0.6 L - 0.25 L^2
    ma = np.array([1])
    proc = ArmaProcess(ar, ma)
    y = proc.generate_sample(nsample=400, distrvs=np.random.default_rng(SEED).standard_normal)
    nlags = 20
    a = acf(y, nlags=nlags)
    p = pacf(y, nlags=nlags)
    ci = 1.96 / np.sqrt(len(y))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, vals, title, c in [
        (axes[0], a, "ACF (autocorrelation)", ARM_BLUE),
        (axes[1], p, "PACF (partial autocorrelation)", ARM_RED),
    ]:
        lags = np.arange(len(vals))
        ax.vlines(lags, 0, vals, color=c, lw=2)
        ax.plot(lags, vals, "o", color=c, ms=4)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-ci, ci, color="grey", alpha=0.15)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lag")
    axes[1].annotate("cuts off after lag 2\n=> AR(2)", xy=(2, p[2]),
                     xytext=(6, 0.55), fontsize=9, color=ARM_RED,
                     arrowprops=dict(arrowstyle="->", color=ARM_RED))
    fig.suptitle("Reading model orders off ACF / PACF (AR(2) sample)", fontsize=12)
    fig.tight_layout()
    save(fig, "acf_pacf.pdf", log)


def fig_arima_forecast(series: pd.Series, log: logging.Logger) -> None:
    h = 18
    train, test = series.iloc[:-h], series.iloc[-h:]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=h)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(train.index, train.values, color=ARM_BLUE, lw=1.3, label="train")
    ax.plot(test.index, test.values, color="k", lw=1.6, label="actual (held out)")
    ax.plot(mean.index, mean.values, color=ARM_RED, lw=2, label="SARIMA forecast")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color=ARM_RED, alpha=0.15, label="95% interval")
    ax.axvline(train.index[-1], color=GREY, ls="--", lw=1)
    ax.set_title("SARIMA(1,1,1)(1,1,0)$_{12}$ forecast", fontsize=12, loc="left")
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "arima_forecast.pdf", log)


def fig_exp_smoothing(series: pd.Series, log: logging.Logger) -> None:
    h = 18
    train, test = series.iloc[:-h], series.iloc[-h:]
    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    holt = Holt(train, initialization_method="estimated").fit()
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                              initialization_method="estimated").fit()

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(train.index[-36:], train.values[-36:], color=ARM_BLUE, lw=1.2, label="train")
    ax.plot(test.index, test.values, color="k", lw=1.6, label="actual")
    ax.plot(test.index, ses.forecast(h).values, color=GREY, lw=1.6, ls=":",
            label="simple (level only)")
    ax.plot(test.index, holt.forecast(h).values, color=ARM_ORANGE, lw=1.6, ls="--",
            label="Holt (level+trend)")
    ax.plot(test.index, hw.forecast(h).values, color=ARM_RED, lw=2,
            label="Holt-Winters (+season)")
    ax.axvline(train.index[-1], color=GREY, ls="--", lw=1)
    ax.set_title("Exponential smoothing: only Holt-Winters captures the season",
                 fontsize=11.5, loc="left")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    save(fig, "exp_smoothing.pdf", log)


def fig_ar_vs_ma(log: logging.Logger) -> None:
    """Sample paths of AR(1) vs MA(1) so students see 'persistence' vs 'echo'."""
    n = 140
    ar1 = ArmaProcess(ar=[1, -0.8], ma=[1]).generate_sample(
        n, distrvs=np.random.default_rng(SEED).standard_normal)
    ma1 = ArmaProcess(ar=[1], ma=[1, 0.8]).generate_sample(
        n, distrvs=np.random.default_rng(SEED + 1).standard_normal)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    axes[0].plot(ar1, color=ARM_BLUE, lw=1.4)
    axes[0].axhline(0, color=GREY, lw=1, ls=":")
    axes[0].set_title(r"AR(1), $\phi=0.8$: today $\approx 0.8\cdot$yesterday $+$ shock",
                      fontsize=11, loc="left")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("value")
    axes[0].text(0.03, 0.05, "persistent - wanders, slow to revert",
                 transform=axes[0].transAxes, fontsize=9.5, color=ARM_BLUE, style="italic")
    axes[1].plot(ma1, color=ARM_RED, lw=1.4)
    axes[1].axhline(0, color=GREY, lw=1, ls=":")
    axes[1].set_title(r"MA(1), $\theta=0.8$: today $=$ shock $+\,0.8\cdot$last shock",
                      fontsize=11, loc="left")
    axes[1].set_xlabel("time")
    axes[1].text(0.03, 0.05, "brief echo - reverts almost at once",
                 transform=axes[1].transAxes, fontsize=9.5, color=ARM_RED, style="italic")
    fig.tight_layout()
    save(fig, "ar_vs_ma.pdf", log)


def main() -> None:
    log = setup_logging()
    np.random.seed(SEED)
    log.info(f"seed={SEED} fig_dir={FIG_DIR}")
    series = synthetic_monthly()
    series.to_csv(CH_DIR / "data" / "synthetic_monthly_sales.csv", header=True)
    fig_anatomy(series, log)
    fig_stl(series, log)
    fig_stationarity(log)
    fig_differencing(series, log)
    fig_acf_pacf(log)
    fig_ar_vs_ma(log)
    fig_arima_forecast(series, log)
    fig_exp_smoothing(series, log)
    log.info("classical figures done")


if __name__ == "__main__":
    main()
