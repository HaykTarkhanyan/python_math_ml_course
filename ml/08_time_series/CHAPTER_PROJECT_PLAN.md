# Chapter project plan — Forecasting Armenian electricity production

**Status:** BUILT 2026-08-05, then expanded the same day after a pedagogical review.
`32_electricity_forecast_solution.ipynb`, **104 cells (46 code, 58 markdown), 13 figures**,
executes end to end in ~45 s on CPU with 0 errors and 0 empty cells. Assembled by
`py_src/build_forecast_nb.py`, so it can be regenerated rather than hand-patched.

## Expansion pass (instructor asked for more steps and more explanation)

Ten weaknesses found and fixed; the notebook went 62 -> 104 cells.

- `make_features` built 11 features in one cell, against the house "one idea per cell" rule. Now
  built family by family (lags -> rolling -> calendar), each with the table shown and read.
- "MASE by hand" was a one-line `np.abs(...).mean()`. Now four explicit steps - the 12 errors, the
  mean, the in-sample scale, the division - with an `assert` tying the hand calc to the helper.
- Differencing was never **plotted**; students read p-values without seeing a stationary series.
  Added a 3-panel raw / seasonal / double-differenced figure.
- Additive-vs-multiplicative was never addressed, although the chapter's own assignment asks it.
  Added Part 2b.
- No residual diagnostics after ARIMA. Added Ljung-Box (p = 0.19 / 0.12 / 0.33, all pass) plus a
  residual time-plot, ACF and histogram.
- SARIMA coefficients were printed and never read. Now each term is explained in words.
- Recursive vs direct multi-step (a deck [31] topic) was one clause. Now a table and a rationale.
- No per-month error breakdown. Added signed-error table, grouped bar chart, and bias reading.
- Why Holt-Winters won was never explained. Now derived from four independent pieces of evidence.
- Only one predict-first moment. Added one before the baseline.

**Two errors in my own new material, caught by executing it:**

1. The lag-availability demo used 2025-07 as the target and printed `lag_11 -> available`, directly
   contradicting the rule it was meant to teach. The binding case is the *furthest* horizon month;
   rewritten to show 2025-07 and 2025-12 side by side, where the disagreement is the lesson.
2. Part 2b measured `corr(level, %swing) = +0.16` against `corr(level, abs swing) = +0.83`, i.e.
   **multiplicative** seasonality - while Part 5 used `seasonal="add"` and the prose claimed the
   check justified it. Now both variants are fitted.

**That fix improved the result and created the notebook's best teaching moment.** Holt-Winters
`mul` wins outright (MASE 0.578 vs `add` 0.609; AIC 172.2 vs 195.3), so the notebook now contains
**two diagnostics with opposite fates** - Part 2b's was confirmed, Part 3's was refuted. The lesson
sharpened from "diagnostics are not verdicts" to "a diagnostic is a hypothesis; checking costs one
extra fit, so check."

The multiplicative fit set `beta = 0.000` **and** `gamma = 0.000`, leaving a slowly-drifting level
times a fixed seasonal profile - one active parameter, and the best model in the notebook.

## What the build changed about this plan

**The plan's Part 3 was wrong, and the notebook now teaches the correction.**

The plan assumed SARIMA orders would come straight off one correlogram. In practice the
stationarity tests said `(1-B^12)` **alone** is already stationary (ADF 0.041, KPSS 0.073), and the
regular difference reduced variance by only 1% (7.11 -> 7.03), so `d=0` looked like the disciplined
choice. But the two differencing choices give *different, equally textbook* readings:

- after `(1-B^12)`: ACF decays, PACF cuts off at lag 1 -> **AR(1)**, i.e. `ARIMA(1,0,0)(0,1,1)12`
- after `(1-B^12)(1-B)`: single negative ACF spike at lag 1 -> **MA(1)**, the airline model

So the notebook fits **both** and lets the held-out year decide. It does, decisively:

| model | MASE | vs naive |
|---|---|---|
| Holt-Winters (mul) | **0.578** | -50.1% |
| Holt-Winters (add) | 0.609 | -47.4% |
| GBM on differences | 0.806 | -30.4% |
| SARIMA(0,1,1)(0,1,1)12 | 0.889 | -23.3% |
| LightGBM (lag>=12) | 1.059 | -8.6% |
| seasonal naive | 1.159 | 0.0% |
| GBM on (t, month) - TRAP | 1.227 | +5.9% |
| ARIMA(1,0,0)(0,1,1)12 | 1.322 | +14.1% |

**The "disciplined" `d=0` model finished last** - worse than doing nothing - while the extra,
"unnecessary" difference produced one of the best. This became the notebook's most valuable lesson
(Part 5 and verdict point 3): *in-sample diagnostics describe the training data; only held-out
evaluation speaks about forecasting.* It was not planned; it was measured.

Other build notes:

- Holt-Winters chose `beta = 0.000` on its own - it declined to model a trend at all, independently
  confirming the differencing finding. Kept and called out.
- The extrapolation trap works as intended and is stark: the trap model's highest possible
  prediction is 35.6 bn dram against a training max of 38.0 and a 2025 actual max of **42.8**.
- The chapter page already carried a 7-step assignment brief with no worked example. The notebook
  maps onto those steps, so it is published as the **walkthrough on a different series**, explicitly
  "an example, not the answer" - students still pick their own series.

---

**Original plan below, as approved.**

**Shape:** one solution-only walkthrough notebook (instructor's choice), following the
`05_interpretability/25_startup_success_solution.ipynb` precedent — a single real project in
sequential parts, not a list of exercises.
**Covers:** deck [30] (classical) and deck [31] (ML), in that order.

---

## The data

`ml/08_time_series/data/armstat_electricity_monthly.csv`, fetched by
`py_src/fetch_armstat_electricity.py` (re-runnable; logs to `logs/`).

- **Source:** ArmStatBank table `IC-ind-m-01.px`, activity 31 = "35. Electricity, gas, steam and
  air conditioning supply". Volume of industrial production, **current prices**, thousand drams.
- **Fetched:** 198 months, 2010-01 .. 2026-06, no gaps.

### Why this series and not tourism

The instructor first chose monthly tourism arrivals. **Armenia does not publish tourism monthly.**
Verified three ways: armstat's tourism page is quarterly and PDF-only; ArmStatBank's entire Tourism
folder is one *annual* table (hotels by marz); air passenger transport is also annual. Quarterly
would give ~40 points — unusable for deck [31]'s lag/rolling/`TimeSeriesSplit` material.

Electricity was selected from a ranking of all 36 sectors in the same table by seasonal amplitude
and 2020-vs-2019 change. It is the largest and cleanest strongly-seasonal monthly series available.
**Accepted cost:** no COVID structural break (electricity is essential; 2020 was +1.5%). The
"your model dies at the shock" lesson is not available here.

### Measured properties (computed, not assumed)

| Property | Value |
|---|---|
| Seasonal peak | December or January in **every** one of 16 years |
| Seasonal amplitude (2014+) | 1.55x (Jan/Dec ~30.6B vs May ~19.8B drams) |
| Usable segment | 2014-01 onward, **150 months** |
| Train / test split | train 2014-01..2024-12 (132), test 2025 (12) |
| Seasonal-naive baseline | **MASE 1.159**, MAPE 8.1%, MAE 2.63B drams |

The baseline MASE being **above 1** matters: 2025 was harder than the training history, so there is
genuine room to improve, and a student who beats it has done something real.

### The data-quality trap (deliberate, and the reason Part 1 exists)

Yearly totals grow **+43.8% (2011), +28.7% (2012), +18.3% (2013)**, then settle to low single
digits from 2014. National electricity output does not grow 44% in a year. In a **nominal** series
this is a tariff and/or coverage change, not demand. Students who model 2010-2013 as "trend" learn
the wrong lesson.

This mirrors the interpretability project, whose Part 1 is a leakage audit — the most important
step is the one before any modelling.

---

## Notebook outline

**Part 0 — The question and the data.** Forecast monthly electricity production 12 months ahead.
Load, plot, state the horizon out loud (deck [31] frame "Say your horizon out loud").

**Part 1 — Data-quality audit.** Plot yearly totals and YoY growth; find the 2011-2013 ramp; decide
to cut to 2014+ and *justify it*. Name the nominal-price issue: "trend" here mixes volume and price,
so any trend claim is about drams, not kilowatt-hours.

**Part 2 — Anatomy ([30]).** STL decomposition into trend / seasonal / residual. Confirm the
Dec-Jan peak and that the seasonal shape is stable across years.

**Part 3 — Stationarity ([30]).** ADF **and** KPSS (the deck insists on both), then
`(1-B)(1-B^12)` differencing. ACF/PACF on the differenced series to *read* the orders rather than
grid-searching them. Include the deck's over-differencing warning.

**Part 4 — Baseline first ([30]).** Seasonal naive, then **MASE computed by hand** before using any
library. Establishes 1.159 as the number every later model must beat. Deck rule: never report a
model without its naive baseline.

**Part 5 — Classical models ([30]).** SARIMA with the orders read in Part 3; Holt-Winters. Evaluate
on the *same* held-out 2025, never a random split.

**Part 6 — The ML reframe ([31]).** Forecasting as supervised learning: lag, rolling, and calendar
features. Forward split, and `TimeSeriesSplit` with `gap` — showing why a random split leaks.

**Part 7 — The extrapolation trap ([31]).** Point gradient boosting at the raw trending series and
watch it flat-line beyond the training range. Then difference first and re-run. This is a
predict-first moment: ask before showing.

**Part 8 — Honest comparison ([31]).** All models on one table, same horizon, same test window,
all against the baseline. Report MASE, not just MAPE.

**Part 9 — Intervals ([31]).** A conformal prediction interval for the best ML model, since the
deck explicitly says "we still owe you an interval".

**Part 10 — Verdict.** Which won, by how much, and whether a 12-point test is enough to call it.
Expected honest answer: classical and ML land close, and the interesting result is *how hard the
naive baseline is to beat* — which is the chapter's real lesson.

---

## Build rules

- Solution notebook is **executed end to end**; every number in the prose comes from a real cell,
  never from this plan. If a result contradicts the plan, the plan is wrong — record it and move on
  (precedent: the diffusion CFG slide and the GAN collapse that did not happen).
- CPU-only, thread-capped, must run in ~1 minute. No GPU, no Colab.
- Seed 509. Armenian flag colours where 3+ series share a chart. `logging`, not `print`.
- Fail loud: no bare excepts, no silent fallback if a model does not converge.
- Register in `08_time_series.qmd` under `# 🏡 Տնային`, replacing that section's `TBD`.

## Instructor decisions (2026-08-05)

1. **Prose language: English.** Unlike the CNN homeworks, this notebook narrates in English
   throughout. The chapter page stays Armenian.
2. **Split confirmed:** train 2014-01..2024-12 (132 months), test 2025 (12 months). 2026 H1 is left
   out of both — it is the most current data but only 6 points.
3. **Part 9 (conformal intervals) is CUT.** Deck [31] gives it a single frame and it was the
   thinnest-taught piece. The notebook ends at the honest comparison plus verdict, renumbered:
   Parts 0-8, with the verdict as Part 9.
