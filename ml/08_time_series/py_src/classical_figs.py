"""Figures for Lecture 1 -- Classical time series (29_classical_time_series).

Generates PDFs into ml/08_time_series/fig/:
  ts_anatomy.pdf       -- one series annotated: trend + seasonality + noise.
  season_vs_cycle.pdf  -- fixed-period season vs variable-period cycle, side by side.
  stl_decomposition.pdf-- STL: observed / trend / seasonal / residual.
  stationarity.pdf     -- stationary vs 3 kinds of non-stationary.
  differencing.pdf     -- non-stationary series -> differenced, ADF *and* KPSS p-values.
  over_differencing.pdf-- what differencing one time too many does to the variance.
  rolling_mean.pdf     -- the ORDINARY moving average: sliding window, then smoothing.
  acf_pacf.pdf         -- ACF & PACF of an AR(2) process (how to read the orders).
  lag_scatter.pdf      -- where autocorrelation comes from: y_t vs y_{t-k} scatters.
  acf_pacf_series.pdf  -- ACF & PACF of OUR OWN series, raw and after (1-B)(1-B^12).
  ar_vs_ma.pdf         -- AR(1) persistence vs MA(1) echo, sample paths.
  arima_forecast.pdf   -- SARIMA forecast on a held-out tail, with 95% interval.
  exp_smoothing.pdf    -- SES / Holt / Holt-Winters forecasts compared.
  ses_weights.pdf      -- geometrically decaying weights for three values of alpha.
  alpha_search.pdf     -- how alpha is actually chosen: SSE(alpha) curve with a minimum.
  ma_invertibility.pdf -- recovering the MA shocks from a wrong seed, and why |theta|<1.

NOTE (2026-08-14): over_differencing.pdf is still generated and still logs the std
values the deck quotes in prose, but its frame was cut, so no slide embeds it now.

Run with the project venv (repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/08_time_series/py_src/classical_figs.py

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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

SEED = 509

# SARIMA orders, chosen from the correlogram in fig_acf_pacf_series rather than guessed:
# after (1-B)(1-B^12) the ACF has a single negative spike at lag 12, the textbook
# signature of a SEASONAL MA term. This is Box-Jenkins' "airline model". It beats the
# seasonal-AR guess this deck used until 2026-07-31 on both AIC (343.8 vs 356.4) and
# held-out MAE (6.07 vs 7.24). Keep in sync with ml_figs.py.
ORDER, SEASONAL_ORDER = (0, 1, 1), (0, 1, 1, 12)

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


def _kpss_p(x) -> str:
    """KPSS p-value, honestly reported when statsmodels clips it to its lookup table.

    KPSS's null is the OPPOSITE of ADF's: H0 = stationary. statsmodels only tabulates
    p in [0.01, 0.10] and warns when the statistic falls outside, so returning the
    clipped number as if it were exact would be a quiet lie.
    """
    import warnings
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = kpss(x, regression="c", nlags="auto")[1]
        clipped = any(issubclass(w.category, InterpolationWarning) for w in caught)
    if clipped:
        return f"< {p:.2f}" if p <= 0.01 else f"> {p:.2f}"
    return f"= {p:.3f}"


def fig_differencing(series: pd.Series, log: logging.Logger) -> None:
    d1 = series.diff().dropna()
    p_raw, p_diff = adfuller(series)[1], adfuller(d1)[1]
    k_raw, k_diff = _kpss_p(series), _kpss_p(d1)
    log.info(f"ADF  p raw={p_raw:.3f}  differenced={p_diff:.3f}")
    log.info(f"KPSS p raw {k_raw}      differenced {k_diff}")

    # sharex + point markers + one worked subtraction, so a student can trace how a
    # value in the lower panel is built from two neighbouring values in the upper one
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    axes[0].plot(series.index, series.values, color=ARM_RED, lw=1.4,
                 marker="o", ms=2.6)
    axes[0].set_title(f"Original: trending.   ADF p = {p_raw:.2f} (cannot reject "
                      f"non-stationary)   KPSS p {k_raw} (rejects stationary)",
                      fontsize=10.5, loc="left", color=ARM_RED)
    axes[1].plot(d1.index, d1.values, color=ARM_BLUE, lw=1.1, marker="o", ms=2.6)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title(f"After 1st differencing $y_t - y_{{t-1}}$.   ADF p = {p_diff:.3f} "
                      f"(stationary)   KPSS p {k_diff} (agrees)",
                      fontsize=10.5, loc="left", color=ARM_BLUE)

    j = 40                                   # the point we walk through explicitly
    t_now, t_prev = series.index[j], series.index[j - 1]
    y_now, y_prev = series.iloc[j], series.iloc[j - 1]
    for ax in axes:
        ax.axvline(t_now, color=GREY, ls=":", lw=1.2)
        ax.axvline(t_prev, color=GREY, ls=":", lw=1.2)
        ax.grid(axis="x", alpha=0.35)
        ax.tick_params(labelsize=9.5)
    axes[0].plot([t_prev, t_now], [y_prev, y_now], "o", color="k", ms=6, zorder=5)
    axes[0].annotate(f"{y_prev:.0f} $\\to$ {y_now:.0f}", xy=(t_now, y_now),
                     xytext=(6, -14), textcoords="offset points",
                     fontsize=9.5, color="k")
    axes[1].plot([t_now], [y_now - y_prev], "o", color="k", ms=6, zorder=5)
    axes[1].annotate(f"{y_now:.0f} $-$ {y_prev:.0f} $=$ {y_now - y_prev:+.0f}",
                     xy=(t_now, y_now - y_prev), xytext=(6, 6),
                     textcoords="offset points", fontsize=9.5, color="k")
    log.info(f"differencing walkthrough: {y_now:.1f} - {y_prev:.1f} = {y_now-y_prev:+.1f}")
    fig.text(0.5, -0.02, "Two tests, opposite nulls. Agreement is the evidence you want; "
             "disagreement means look again.", ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "differencing.pdf", log)


def fig_over_differencing(series: pd.Series, log: logging.Logger) -> None:
    """The trap the deck states in words: differencing once more inflates the variance."""
    levels = [("raw $y_t$", series, ARM_RED),
              ("$d=1$: $y_t - y_{t-1}$", series.diff().dropna(), ARM_BLUE),
              ("$d=2$: differenced again", series.diff().diff().dropna(), ARM_ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (name, data, c) in zip(axes, levels):
        ax.plot(data.index, data.values, color=c, lw=1.2)
        sd = data.std()
        ax.set_title(f"{name}\nstd = {sd:.1f}", fontsize=11, loc="left", color=c)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        log.info(f"over-differencing: {name:26s} std={sd:.2f}  "
                 f"ADF p={adfuller(data)[1]:.4f}")
        if name.startswith("$d=1$"):
            ax.axhline(0, color="k", lw=0.6)
        if name.startswith("$d=2$"):
            ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("Difference the MINIMUM amount that passes the test", fontsize=12.5)
    fig.text(0.5, -0.04, "One difference already passes ADF. The second one does not remove "
             "more trend - there is none left - it just amplifies the noise.",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "over_differencing.pdf", log)


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


def fig_acf_pacf_series(series: pd.Series, log: logging.Logger) -> None:
    """ACF/PACF of OUR series -- raw, then after (1-B)(1-B^12) -- so the SARIMA orders
    on the next frame are read off a picture instead of appearing from nowhere."""
    stationary = series.diff().diff(12).dropna()
    nlags = 26
    panels = [("Raw series: ACF decays slowly = still trending", series, "acf", ARM_RED),
              (r"After $(1-B)(1-B^{12})$: ACF", stationary, "acf", ARM_BLUE),
              (r"After $(1-B)(1-B^{12})$: PACF", stationary, "pacf", ARM_ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    for ax, (title, data, kind, c) in zip(axes, panels):
        vals = acf(data, nlags=nlags) if kind == "acf" else pacf(data, nlags=nlags)
        ci = 1.96 / np.sqrt(len(data))
        lags = np.arange(len(vals))
        ax.vlines(lags, 0, vals, color=c, lw=1.8)
        ax.plot(lags, vals, "o", color=c, ms=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-ci, ci, color="grey", alpha=0.15)
        ax.set_title(title, fontsize=10.5, loc="left", color=c)
        ax.set_xlabel("lag", fontsize=10)
        ax.tick_params(labelsize=9.5)
        if data is stationary:
            ax.axvline(12, color=GREY, ls=":", lw=1.2)
            ax.annotate("lag 12", xy=(12, ax.get_ylim()[1] * 0.82), fontsize=9,
                        color=GREY, ha="center")
        log.info(f"{title[:44]:46s} lag1={vals[1]:+.2f} lag12={vals[12]:+.2f}")
    fig.suptitle("Reading the orders off OUR monthly series", fontsize=12.5)
    fig.tight_layout()
    save(fig, "acf_pacf_series.pdf", log)


def fig_arima_forecast(series: pd.Series, log: logging.Logger) -> None:
    h = 18
    train, test = series.iloc[:-h], series.iloc[-h:]
    model = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=h)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    log.info(f"SARIMA{ORDER}{SEASONAL_ORDER} AIC={fit.aic:.1f} "
             f"test MAE={np.abs(test.values - mean.values).mean():.2f}")

    # sized for a ~55% slide column, so every font is set well above matplotlib's
    # defaults - at deck scale the old 8.5pt legend was unreadable
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(train.index, train.values, color=ARM_BLUE, lw=1.5, label="train")
    ax.plot(test.index, test.values, color="k", lw=2.0, label="actual (held out)")
    ax.plot(mean.index, mean.values, color=ARM_RED, lw=2.4, label="SARIMA forecast")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color=ARM_RED, alpha=0.18, label="95% interval")
    ax.axvline(train.index[-1], color=GREY, ls="--", lw=1.2)
    ax.set_title("SARIMA(0,1,1)(0,1,1)$_{12}$ forecast -- the 'airline model'",
                 fontsize=13.5, loc="left")
    ax.legend(fontsize=11, ncol=2, loc="upper left")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    save(fig, "arima_forecast.pdf", log)


def fig_exp_smoothing(series: pd.Series, log: logging.Logger) -> None:
    h = 18
    train, test = series.iloc[:-h], series.iloc[-h:]
    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    holt = Holt(train, initialization_method="estimated").fit()
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                              initialization_method="estimated").fit()

    # same sizing note as fig_arima_forecast: this lands in a ~55% column
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(train.index[-36:], train.values[-36:], color=ARM_BLUE, lw=1.4, label="train")
    ax.plot(test.index, test.values, color="k", lw=2.0, label="actual")
    ax.plot(test.index, ses.forecast(h).values, color=GREY, lw=2.0, ls=":",
            label="simple (level only)")
    ax.plot(test.index, holt.forecast(h).values, color=ARM_ORANGE, lw=2.0, ls="--",
            label="Holt (level+trend)")
    ax.plot(test.index, hw.forecast(h).values, color=ARM_RED, lw=2.4,
            label="Holt-Winters (+season)")
    ax.axvline(train.index[-1], color=GREY, ls="--", lw=1.2)
    ax.set_title("Exponential smoothing: only Holt-Winters captures the season",
                 fontsize=13, loc="left")
    ax.legend(fontsize=10.5, ncol=2, loc="upper left")
    ax.tick_params(labelsize=11)
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


def fig_season_vs_cycle(log: logging.Logger) -> None:
    """Seasonality repeats on a FIXED period; a cycle does not.

    Exists because the four-components frame names a cycle that ts_anatomy.pdf does
    not contain (the running series has no cycle in it). Rather than fake one into
    the running example, the distinction gets its own picture.
    """
    rng = np.random.default_rng(SEED)
    n = 240
    t = np.arange(n)

    seasonal = 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, n)
    # incommensurate periods -> swings that never line up on a fixed grid
    cyclic = (10 * np.sin(2 * np.pi * t / 47)
              + 6 * np.sin(2 * np.pi * t / 83 + 1.1)
              + rng.normal(0, 1.0, n))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)

    axes[0].plot(t, seasonal, color=ARM_ORANGE, lw=1.3)
    for k in range(0, n, 12):
        axes[0].axvline(k, color=GREY, ls=":", lw=0.9)
    axes[0].set_title("Seasonality: period is FIXED (every 12 steps)",
                      fontsize=11, loc="left", color=ARM_ORANGE)
    axes[0].annotate("", xy=(12, -14.5), xytext=(24, -14.5),
                     arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    axes[0].text(18, -13.6, "12", ha="center", va="bottom", fontsize=9, color=GREY)

    axes[1].plot(t, cyclic, color=ARM_RED, lw=1.3)
    # mark the actual peaks and label the (unequal) gaps between them
    peaks = []
    for i in range(3, n - 3):
        window = cyclic[i - 3:i + 4]
        if cyclic[i] == window.max() and cyclic[i] > 6:
            if not peaks or i - peaks[-1] > 15:
                peaks.append(i)
    for pk in peaks:
        axes[1].axvline(pk, color=GREY, ls=":", lw=0.9)
    for a, b in zip(peaks, peaks[1:]):
        axes[1].annotate("", xy=(a, -14), xytext=(b, -14),
                         arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
        axes[1].text((a + b) / 2, -14, f"{b - a}", ha="center", va="center",
                     fontsize=9, color=GREY,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axes[1].set_title("Cycle: swings with NO fixed period",
                      fontsize=11, loc="left", color=ARM_RED)
    log.info(f"season_vs_cycle: cycle peak gaps = "
             f"{[b - a for a, b in zip(peaks, peaks[1:])]}")

    for ax in axes:
        ax.set_xlabel("month")
        ax.tick_params(labelsize=9.5)
    fig.text(0.5, -0.04, "You can put a season on a calendar. You cannot put a cycle on one - "
             "which is why only the season gets modelled explicitly.",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    save(fig, "season_vs_cycle.pdf", log)


def fig_rolling_mean(series: pd.Series, log: logging.Logger) -> None:
    """The ORDINARY moving average, shown before ARIMA's MA(q) hijacks the name."""
    roll3 = series.rolling(3).mean()
    roll12 = series.rolling(12).mean()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))

    axes[0].plot(series.index, series.values, color=GREY, lw=1.0, alpha=0.6, label="raw $y_t$")
    axes[0].plot(roll3.index, roll3.values, color=ARM_ORANGE, lw=1.6, label="rolling(3)")
    axes[0].plot(roll12.index, roll12.values, color=ARM_RED, lw=2.2, label="rolling(12)")
    axes[0].set_title("A wider window smooths harder", fontsize=11, loc="left")
    axes[0].legend(fontsize=9.5, loc="upper left")
    axes[0].tick_params(labelsize=9)
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(4))

    # right panel: the arithmetic, on 3 consecutive windows of width 3.
    # The windows are drawn as brackets BELOW the curve rather than full-height
    # shading - three overlapping axvspans just made a muddy stripe.
    lo = 24
    seg = series.iloc[lo:lo + 8]
    x = np.arange(len(seg))
    axes[1].plot(x, seg.values, "o-", color=ARM_BLUE, lw=1.6, ms=6)
    for i, v in enumerate(seg.values):
        axes[1].annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8.5, color=ARM_BLUE)

    base, drop = seg.values.min() - 3, 2.6
    for j, c in zip((0, 1, 2), (ARM_ORANGE, ARM_RED, GREY)):
        m = seg.values[j:j + 3].mean()
        yb = base - j * drop
        axes[1].plot([j, j + 2], [yb, yb], color=c, lw=3, solid_capstyle="butt")
        axes[1].plot([j + 2], [yb], ">", color=c, ms=7)
        axes[1].text(j + 2.25, yb, f"mean = {m:.1f}", va="center",
                     fontsize=9, color=c)
        log.info(f"rolling demo window {j}: mean of "
                 f"{np.round(seg.values[j:j + 3], 1)} = {m:.2f}")
    axes[1].set_ylim(base - 3 * drop - 1, seg.values.max() + 4)
    axes[1].set_xlim(-0.6, 9.2)
    axes[1].set_title("the window of 3 slides one step at a time",
                      fontsize=11, loc="left")
    axes[1].set_xlabel("step")
    axes[1].tick_params(labelsize=9)

    fig.suptitle("The moving (rolling) average: a sliding mean of the VALUES", fontsize=12.5)
    fig.tight_layout()
    save(fig, "rolling_mean.pdf", log)


def fig_lag_scatter(series: pd.Series, log: logging.Logger) -> None:
    """Where the ACF comes from: one scatter per lag, one correlation per scatter."""
    lags = (1, 6, 12)
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.2))

    for ax, k, c in zip(axes[:3], lags, (ARM_BLUE, ARM_ORANGE, ARM_RED)):
        x, y = series.shift(k).values, series.values
        ok = ~np.isnan(x)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        ax.plot(x[ok], y[ok], "o", color=c, ms=3.5, alpha=0.75)
        ax.set_title(f"lag {k}:  r = {r:+.2f}", fontsize=11, loc="left", color=c)
        ax.set_xlabel(f"$y_{{t-{k}}}$", fontsize=10)
        ax.tick_params(labelsize=9)
        log.info(f"lag_scatter: lag {k:2d} r = {r:+.3f}")
    axes[0].set_ylabel("$y_t$", fontsize=10)

    vals = acf(series, nlags=24)
    ci = 1.96 / np.sqrt(len(series))
    ax = axes[3]
    ax.vlines(np.arange(len(vals)), 0, vals, color=GREY, lw=1.8)
    ax.plot(np.arange(len(vals)), vals, "o", color=GREY, ms=3)
    for k, c in zip(lags, (ARM_BLUE, ARM_ORANGE, ARM_RED)):
        ax.vlines(k, 0, vals[k], color=c, lw=3.2)
        ax.plot([k], [vals[k]], "o", color=c, ms=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-ci, ci, color="grey", alpha=0.15)
    ax.set_title("all lags at once = the ACF", fontsize=11, loc="left")
    ax.set_xlabel("lag", fontsize=10)
    ax.tick_params(labelsize=9)

    fig.suptitle("One scatter per lag; collect the correlations and you have drawn the ACF",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "lag_scatter.pdf", log)


def fig_ses_weights(log: logging.Logger) -> None:
    """Why it is called EXPONENTIAL smoothing: the weights decay geometrically."""
    k = np.arange(0, 16)
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for alpha, c in [(0.2, ARM_BLUE), (0.5, ARM_ORANGE), (0.8, ARM_RED)]:
        w = alpha * (1 - alpha) ** k
        ax.plot(k, w, "o-", color=c, lw=1.8, ms=5,
                label=rf"$\alpha={alpha}$  (weight on $y_{{t-1}}$ = {w[0]:.2f})")
        log.info(f"ses weights alpha={alpha}: first four = {np.round(w[:4], 3)}, "
                 f"sum over 16 lags = {w.sum():.3f}")
    ax.set_xlabel("how many steps back ($k$)")
    ax.set_ylabel(r"weight $\alpha(1-\alpha)^k$")
    ax.set_title(r"Every past value counts, but geometrically less. Weights sum to 1.",
                 fontsize=12, loc="left")
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "ses_weights.pdf", log)


def _ses_sse(y: np.ndarray, alpha: float) -> float:
    """One-step-ahead sum of squared errors for SES, level initialised at y[0]."""
    level, sse = y[0], 0.0
    for t in range(1, len(y)):
        sse += (y[t] - level) ** 2
        level = alpha * y[t] + (1 - alpha) * level
    return sse


TOY = np.array([10.0, 14.0, 12.0, 18.0, 16.0])


def fig_alpha_search(log: logging.Logger) -> None:
    """How alpha is ACTUALLY chosen: score every candidate, keep the best.

    Uses the same 5-point toy series the deck works by hand, so the two numbers on
    the slide are literally two points on this curve.
    """
    grid = np.linspace(0.01, 1.0, 400)
    sse = np.array([_ses_sse(TOY, a) for a in grid])
    best_a, best_sse = grid[sse.argmin()], sse.min()
    log.info(f"alpha_search: optimum alpha = {best_a:.3f} (SSE {best_sse:.2f}); "
             f"SSE(0.2) = {_ses_sse(TOY, 0.2):.2f}, SSE(0.8) = {_ses_sse(TOY, 0.8):.2f}, "
             f"SSE(1.0) = {_ses_sse(TOY, 1.0):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))

    steps = np.arange(1, len(TOY) + 1)
    axes[0].plot(steps, TOY, "o-", color="k", lw=2.0, ms=6, label="actual $y_t$")
    for alpha, c in [(0.2, ARM_BLUE), (0.8, ARM_RED)]:
        level, preds = TOY[0], [np.nan]
        for t in range(1, len(TOY)):
            preds.append(level)
            level = alpha * TOY[t] + (1 - alpha) * level
        axes[0].plot(steps, preds, "s--", color=c, lw=1.6, ms=5,
                     label=rf"$\alpha={alpha}$: SSE {_ses_sse(TOY, alpha):.1f}")
    axes[0].set_title("Two candidate $\\alpha$ on a 5-point series", fontsize=11, loc="left")
    axes[0].set_xlabel("t")
    axes[0].set_xticks(steps)
    axes[0].legend(fontsize=9.5, loc="upper left")
    axes[0].tick_params(labelsize=9.5)

    axes[1].plot(grid, sse, color=ARM_ORANGE, lw=2.2)
    for alpha, c in [(0.2, ARM_BLUE), (0.8, ARM_RED)]:
        axes[1].plot([alpha], [_ses_sse(TOY, alpha)], "s", color=c, ms=9, zorder=5)
    axes[1].plot([best_a], [best_sse], "*", color=ARM_BLUE, ms=18, zorder=6)
    axes[1].annotate(rf"best $\alpha$ = {best_a:.2f}", xy=(best_a, best_sse),
                     xytext=(best_a - 0.42, best_sse + 12), fontsize=10, color=ARM_BLUE,
                     arrowprops=dict(arrowstyle="->", color=ARM_BLUE))
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("SSE of one-step errors")
    axes[1].set_title(r"The optimiser just walks down this curve", fontsize=11, loc="left")
    axes[1].tick_params(labelsize=9.5)

    fig.suptitle("Fitting a smoothing parameter: no formula, a search", fontsize=12.5)
    fig.tight_layout()
    save(fig, "alpha_search.pdf", log)


def fig_ma_invertibility(log: logging.Logger) -> None:
    """Can the MA shocks actually be recovered, and what |theta| < 1 is really for.

    Builds MA(1) data from a shock sequence we keep, then re-derives the shocks with
    the recursion the deck teaches, deliberately seeded with the WRONG value eps_0=0.
    The seed error is multiplied by theta at every step, so it dies out when
    |theta| < 1 and explodes when it is not. That is the whole content of the
    invertibility condition, and it is measured here rather than asserted.
    """
    rng = np.random.default_rng(SEED)
    n = 60
    c_true = 50.0
    eps_true = rng.normal(0, 1.0, n + 1)          # eps_true[0] is the pre-sample shock

    def build(theta: float) -> np.ndarray:
        return np.array([c_true + eps_true[t + 1] + theta * eps_true[t]
                         for t in range(n)])

    def recover(y: np.ndarray, theta: float, eps0: float = 0.0) -> np.ndarray:
        eps = np.empty(len(y))
        prev = eps0
        for t in range(len(y)):
            eps[t] = y[t] - c_true - theta * prev
            prev = eps[t]
        return eps

    FLOOR = 1e-17          # so exact zeros survive a log axis
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))

    for theta, col in [(0.3, ARM_BLUE), (0.7, ARM_ORANGE), (0.9, ARM_RED)]:
        err = np.abs(recover(build(theta), theta) - eps_true[1:])
        axes[0].semilogy(np.arange(n), np.maximum(err, FLOOR), color=col, lw=1.9,
                         label=rf"$\theta={theta}$")
        log.info(f"invertible theta={theta}: recovery error t=0 {err[0]:.2e}, "
                 f"t=20 {err[20]:.2e}, t=40 {err[40]:.2e}")
    axes[0].set_title(r"$|\theta| < 1$: the wrong seed is forgotten",
                      fontsize=11, loc="left", color=ARM_BLUE)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$|\hat\varepsilon_t - \varepsilon_t|$")
    axes[0].legend(fontsize=9.5)
    axes[0].tick_params(labelsize=9)

    theta_bad = 1.3
    err_bad = np.abs(recover(build(theta_bad), theta_bad) - eps_true[1:])
    axes[1].semilogy(np.arange(n), np.maximum(err_bad, FLOOR), color=ARM_RED, lw=2.2)
    axes[1].set_title(r"$\theta = 1.3$, and the $\theta$ is CORRECT: it explodes",
                      fontsize=11, loc="left", color=ARM_RED)
    axes[1].set_xlabel("t")
    axes[1].tick_params(labelsize=9)
    log.info(f"non-invertible theta={theta_bad}: error t=20 {err_bad[20]:.2e}, "
             f"t=40 {err_bad[40]:.2e}, t=59 {err_bad[59]:.2e}")

    # No suptitle: the slide title already frames this, and the frame needs the room.
    fig.tight_layout()
    save(fig, "ma_invertibility.pdf", log)

    # the recovered shocks ARE the one-step-ahead forecast errors - assert it
    theta = 0.7
    y = build(theta)
    eps_hat = recover(y, theta)
    yhat = c_true + theta * np.concatenate([[0.0], eps_hat[:-1]])
    gap = float(np.abs((y - yhat) - eps_hat).max())
    if gap > 1e-10:
        raise AssertionError(f"recovered shocks are not the one-step errors: {gap}")
    log.info(f"recovered eps == one-step forecast errors to {gap:.1e}")


def log_ar_by_ols(series: pd.Series, log: logging.Logger) -> None:
    """No figure - just the numbers the 'AR is only least squares' frame quotes.

    Fitting AR(1) two ways: plain OLS on a lag column (what the slide claims you can
    do) and statsmodels' exact maximum likelihood (what the library actually runs).
    The slide quotes both, so both get measured here rather than asserted.
    """
    y = series.values
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    c_ols, phi_ols = np.linalg.lstsq(X, y[1:], rcond=None)[0]

    from statsmodels.tsa.arima.model import ARIMA
    phi_mle = ARIMA(series, order=(1, 0, 0)).fit().params["ar.L1"]

    log.info(f"AR(1) by OLS on a lag column: c = {c_ols:.3f}, phi = {phi_ols:.4f}")
    log.info(f"AR(1) by statsmodels exact MLE: phi = {phi_mle:.4f} "
             f"(gap = {abs(phi_mle - phi_ols):.4f})")


def main() -> None:
    log = setup_logging()
    np.random.seed(SEED)
    log.info(f"seed={SEED} fig_dir={FIG_DIR}")
    series = synthetic_monthly()
    series.to_csv(CH_DIR / "data" / "synthetic_monthly_sales.csv", header=True)
    fig_anatomy(series, log)
    fig_season_vs_cycle(log)
    fig_stl(series, log)
    fig_stationarity(log)
    fig_differencing(series, log)
    fig_over_differencing(series, log)
    fig_rolling_mean(series, log)
    fig_acf_pacf(log)
    fig_lag_scatter(series, log)
    fig_acf_pacf_series(series, log)
    fig_ar_vs_ma(log)
    fig_arima_forecast(series, log)
    fig_exp_smoothing(series, log)
    fig_ses_weights(log)
    fig_alpha_search(log)
    fig_ma_invertibility(log)
    log_ar_by_ols(series, log)
    log.info("classical figures done")


if __name__ == "__main__":
    main()
