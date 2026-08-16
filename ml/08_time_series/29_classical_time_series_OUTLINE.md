# 29 Classical Time Series — outline

Lecture 1 of the time-series chapter. **41 frames / 41 PDF pages** (no `\pause`
frames).

> **Restructured 2026-08-16 (ToDo.md items 5-6).** The standalone "Where the
> coefficients come from" section was dissolved and its frames moved beside the
> models they explain: AR's least-squares frame now follows AR($p$); MA's
> optimiser frame and the invertibility frame follow MA($q$); the by-hand $\alpha$
> and "same recipe" frames close the exponential-smoothing section. The section's
> `[plain]` transition frame was dropped, hence 42 -> 41 pages. Clean PDF only (not delivered yet, so no `_notes.pdf`). Bridges from all
prior i.i.d. supervised lectures.

> Renumbered from **07** on 2026-07-31: that number belongs to the delivered
> [07] Regularization. 28–29 are left free for `ml/07_classic_methods`, which
> comes first in course order.

## Sections / frames

Frame order below matches the `.tex`. (Before 2026-08-14 this file listed
Autocorrelation *before* the ARIMA family; the deck has always had it after, and
depends on that — the ACF frame opens with "now that AR($p$) and MA($q$) are on the
table".)

1. **Cold open** — "every model so far assumed rows were interchangeable"; time
   series breaks it. Two-lecture roadmap.
2. **What makes time series different** — vocabulary (horizon, uni/multivariate,
   **exogenous vs endogenous**, and the catch that an exogenous driver needs its own
   future value); the four components; **seasonality vs cycle** on its own frame,
   because the anatomy figure only shows three of the four; STL decomposition.
   *(figs: ts_anatomy, season_vs_cycle, stl_decomposition)*
3. **Stationarity and differencing** — definition, with the autocovariance condition
   stated in plain words (Jan-to-Feb behaves like Jul-to-Aug) and the reason it
   matters; stationary vs 3 non-stationary panels; differencing with **ADF and KPSS**
   (opposite nulls); then a frame deriving **what $d$ actually counts** — for
   $y=a+bt$ one difference leaves the constant $b$; for $a+bt+ct^2$ it leaves
   $b+c(2t-1)$, still linear, so $d=2$ — closing with the minimum-differencing rule
   (std $40.9 \to 10.8 \to 11.7$). *(figs: stationarity, differencing)*
4. **ARIMA family** — AR($p$) on its own frame with every symbol named ($\phi$ = the
   weight, $\eps$ = the shock) and a two-step hand calculation; a **rolling-mean
   interlude** so the ordinary moving average is understood *before* MA($q$) borrows
   the name; MA($q$) with the shock recursion unrolled over four steps, showing that
   its memory is exactly $q$ long; the **MA(q)-is-not-a-rolling-mean** frame; then
   ARMA / ARIMA together, and SARIMA with the one-season-only limit.
   *(figs: rolling_mean, ar_vs_ma)*
5. **Autocorrelation** — a from-scratch frame first (shift a copy, scatter, take
   Pearson's $r$; lag 1 $=+0.96$, lag 6 $=+0.77$, lag 12 $=+0.97$, and collecting
   every lag *is* the ACF), then ACF vs PACF motivated by the **double-counting**
   problem (today correlates with two days ago for free, borrowed through yesterday).
   Then orders read off **our own series**: after $(1-B)(1-B^{12})$ a single negative
   lag-12 spike ⇒ seasonal MA ⇒ the **airline model** (0,1,1)(0,1,1)₁₂, confirmed by
   AIC (343.8 vs 356.4) and test MAE (6.07 vs 7.24).
   *(figs: lag_scatter, acf_pacf, acf_pacf_series, arima_forecast)*
6. **Exponential smoothing** — why weighted averages at all (mean-of-everything is
   sluggish, last-$k$ is a cliff, geometric decay is neither); SES in both recursive
   and error-correction form, plus the substitution that produces the geometric
   weights; then ETS = Error/Trend/Seasonality, SES → Holt → Holt-Winters with
   $\beta$ and $\gamma$ described as the same dial applied to slope and season.
   *(figs: ses_weights, exp_smoothing)*
7. **Where the coefficients come from** — the section that answers "who picks these
   numbers?". AR really is **ordinary least squares** on a lag table (measured: OLS
   $\phi = 0.9686$ vs exact MLE $0.9749$, and *why* the gap exists), with Yule-Walker
   noted as a second closed form straight off the correlogram. MA breaks the trick
   because the shocks are unobserved — escaped by fixing $\theta$, reconstructing the
   residuals recursively and scoring SSE($\theta$), which is conditional maximum
   likelihood. Then a frame answering the question that recursion provokes: **are the
   shocks actually recoverable, and are they parameters?** No — three numbers are
   fitted ($c, \theta, \sigma^2$) however long the series, the $\eps$ are residuals,
   and they are exactly the one-step-ahead forecast errors. The wrong seed
   $\eps_0 = 0$ is forgiven geometrically ($2.2\cdot10^{-2} \to 1.4\cdot10^{-8}$ by
   $t=40$ at $\theta=0.7$), which is **where the invertibility condition comes from**:
   at $\theta = 1.3$ the same recursion diverges past $10^5$ even with the correct
   $\theta$. Then **$\alpha$ chosen by hand** on a 5-point series (SSE 78.61 at
   $\alpha=0.2$ vs 51.34 at $\alpha=0.8$), and the unifying recipe with the SSE curve
   bottoming at $\alpha = 0.67$. Includes the fitting-vs-AIC pre-empt: fitting picks
   coefficients for one $(p,d,q)$, AIC picks between different $(p,d,q)$.
   *(figs: ma_invertibility, alpha_search)*
8. **Baselines and evaluation** — naive / seasonal-naive / drift;
   MAE/RMSE/MAPE/MASE spelled out; **MASE worked by hand** (6/18 = 0.33), which is
   also where the in-sample denominator gets explained; time-ordered holdout;
   **even the baseline has to respect the arrow** — `shift(12)` past one season
   reads inside the test window, and fixing it moves seasonal naive from MASE 0.90
   to 1.22. *(fig: time_aware_split)*
9. **Recap** + green "Next: ML for time series" box.

## Notes

- All figures Python-generated by `py_src/classical_figs.py` (seed 509) on a
  deterministic synthetic monthly series. No TikZ, no external images.
- Hands off to `30_ml_time_series.tex` exactly at "time-aware validation".
- **No predict-first frame.** The deck's only one (over-differencing) was cut on
  instructor request 2026-08-14; its lesson survives as prose on the $d$ frame.
  `over_differencing.pdf` is still generated and still logs the std values the
  slide quotes, but nothing embeds it now.
- **Still open:** everything runs on one synthetic series with an exactly linear
  trend. No real data appears; the dram exchange rate is named in the cold open
  and never shown.
