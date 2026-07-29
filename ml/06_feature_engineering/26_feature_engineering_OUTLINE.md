# [26] Feature Engineering — Outline (v3)

Design doc for the **first deck of `ml/06_feature_engineering/`**. House style + `ml/SLIDE_STYLE.md`.
Replaces `L01g_feature_engineering.tex` (written 2026-06-05, before chapters 4 and 5 existed).

> **v3** = self-review fixes applied (see *Changelog*). **v2** = instructor interview, 4 rounds.
> **Status: draft for approval.** Nothing built yet.

---

## Scope / thesis

**Features are not given to you — somebody made them.** The bike dataset proves it on slide 1: of
the 16 columns students have used for three straight lectures, exactly **one** (`dteday`) is raw.

The chapter's operating thesis, which frame 7 states outright and every later section obeys:
**generate cheaply, select ruthlessly.** Mechanical feature engineering buys less than it did in
2015 because gradient boosting derives ratios and interactions on its own — so the payoff has moved
to the things a model *cannot* reach (external joins, domain knowledge, small data, leakage-free
time features), and to being disciplined about pruning whatever you do generate. Deck [27] is the
pruning half.

**Scope note: this chapter is regression-only** (instructor decision). Classification-specific
feature engineering — class-conditional aggregates, weight-of-evidence encoding — is deliberately
out of scope, not overlooked.

## Locked decisions (from the interview)

| Decision | Choice |
|---|---|
| Dataset | **Bike sharing only** (`ml/05_interpretability/data/bike-day.csv`, 731 rows) |
| Lags / rolling / `TimeSeriesSplit` | **Handed forward** to the time-series chapter — one pointer frame, no code |
| Chapter shape | **2 decks, no practical.** [26], [27], then classic methods at [28] |
| Depth | **Same as ch5** — mechanics, not just names |
| Framing | **Nuanced 2026 view**, not "good features beat a better algorithm" |
| Deck chaining | **Chained.** [26] generates ~200 candidates; [27] prunes exactly those |
| Candidate pool | **Systematic expansion** — mechanical, reproducible, ~30 lines |
| Null results | **Go on the slide** as measured |
| Leakage | **Stays distributed** as warnings on the frames where it bites. NOT a section |
| Learned features | **One honest contrast frame** (now frame 8), forward-pointer to the NN chapter |
| Geospatial | **Kept, as an openly standalone illustration** — no continuity claim |
| Text | **Kept, as a labelled illustration** |
| Automation | **One awareness frame** — the measured head-to-head is cut (see Changelog) |
| Armenian examples | **Not forced.** Chapter stays on bike |
| Project-time survey | **Kept, reframed as folklore** |
| Classification FE | **Out of scope**, stated deliberately |
| Length | **~34 frames is fine**, do not pre-emptively trim |
| Homework | **Out of scope for now** |

## Changelog — v2 → v3 (self-review)

| # | Issue | Fix |
|---|---|---|
| 1 | `atemp − temp` is **mathematically meaningless** — the two columns use different normalizations (`(T+8)/47` vs `(T+16)/66`), so their difference is an arbitrary affine mix, not a feels-like gap | Frame 11 now reconstructs Celsius first. Turns a bug into the deck's best callback to [22] |
| 2 | Systematic expansion **divides by zero**: `hum` has exactly one 0 (2011-03-10, verified) — a missing-value sentinel, since 0% humidity in DC in March is impossible | Frame 28 now surfaces this **on the slide** as a data-quality discovery rather than guarding it silently |
| 3 | Geospatial rationale was oversold — `bike-day.csv` is daily **system-wide** totals with no station dimension, so station coordinates join to nothing | Frame 24 is now openly standalone. No continuity claim |
| 4 | featuretools head-to-head was **rigged** — on a single flat table DFS can only emit what systematic expansion already produces | Measured comparison **cut**. Automation demoted to one awareness frame (29) |
| 5 | Frame 7's thesis contradicted frames 27-29 with no reconciliation | "Generate cheaply, select ruthlessly" is now stated on frame 7 and is the chapter thesis |
| 6 | Learned-features frame was **misfiled** under "From outside the data" — it is not a source | Moved to frame 8, paired with the 2026 thesis frame where it belongs |
| 7 | No frame on getting a hand-built feature **into a Pipeline** — so frames 18-19 teach a rule students cannot obey | New frame 20 |
| 8 | Polynomial features blow up numerically without scaling first | Added to frame 12 |
| 9 | No acronym-check section (deck introduces TF-IDF, DFS, POI, H3, CTR, OHE, GBM) | Added below |
| 10 | Frame count and "three sources = the ToC" both wrong | Corrected |

## What is dropped from `L01g` and why

| Old frame | Disposition |
|---|---|
| Datetime features (components, cyclic sin/cos) | **Cut to a callback** — [03] teaches this including sin/cos, and is delivered |
| Time-series lags and rolling windows | **Handed forward** (frame 21) |
| Numerical transformations, log/power half | **Cut to a callback** — [03] has it. Binning survives, that part is new |
| "After engineering: select (L01h preview)" | **Cut.** A teaser for the very next lecture is filler |
| Pitfall #6, "don't engineer what the model discovers" | **Promoted** to the punchline of frame 13 |
| HW01g frame | **Removed** |
| Yerevan rent running example | **Cut entirely** — the dataset does not exist in the repo |

## Callbacks

- **[03] preprocessing** — fit-on-train-only, datetime components, log transforms, target/frequency
  encoding, `Pipeline`/`ColumnTransformer`. This deck extends that chapter rather than repeating it.
- **[22]** — main effect vs interaction; "trees discover interactions via nested splits"; correlated
  features split the credit; **the `temp` normalization `(T+8)/47`, which frame 11 now depends on.**
- **[23]** — PFI on bike; ICE/PDP curves showing `temp`'s non-monotonic effect (frame 14 leans on
  this directly); overall H_temp = 0.21.
- **[07]** — Lasso, for "did the model keep it?"  **[02]** — polynomial regression as OLS on `φ(x)`.
- **math 30** — curse of dimensionality.

## Research needed before building

- **Frame 7 is the thesis frame and I do not yet have evidence for it.** The claim is that
  well-tuned gradient boosting plus light FE now matches heavy FE on many tabular problems.
  Candidates: Grinsztajn et al. 2022 (trees vs deep nets — adjacent, not the same claim), AutoML
  benchmark papers, Kaggle post-mortems. **Search first; if the evidence is thin, narrow the frame
  to a measured claim about bike rather than a general one.**
- Citations to verify: Wide & Deep (Cheng et al., year?), Deep Feature Synthesis (Kanter &
  Veeramachaneni, 2015), tsfresh (Christ et al., 2018), bike dataset (Fanaee-T & Gama, 2014 — reuse
  the form already used in [22]).
- 60-80% project-time figures: CrowdFlower 2016, Anaconda 2020. **Vendor surveys.** Verify they say
  what the old deck claims, then present as folklore, not measurement.
- **Verify every package is alive on PyPI before listing it.** The `pyartemis` incident in [23] sent
  students to an unrelated package for a full lecture.

## Acronym discipline

Deck introduces `TF-IDF`, `DFS`, `POI`, `H3`, `CTR`, `OHE`, `GBM`, `MAE`. Plain phrase first,
acronym in parentheses; never letter-highlighting inside words (it breaks both grep and PDF search).
Before calling the deck done:

```bash
sed 's/%.*//' 26_feature_engineering.tex | grep -oE '\b[A-Z]{2,6}\b' | sort -u
```

---

## Measured results (2026-07-29, `py_src/fe_measure.py`, full log `logs/fe_measure.log`)

Every `[MEASURE]` frame now has a real number. **Baseline reproduces chapter 5 exactly:** forest
test MAE **455.1**, R² **0.879**; Ridge test MAE **655.2**.

| # | Measurement | Result | Verdict |
|---|---|---|---|
| M1 | `casual + registered = cnt` | true on all 731 rows. Forest with them: **R² 0.9956, MAE 81.7** vs honest 0.879 / 455.1 | ✅ frame 3 lands |
| M2 | Ratio features | naive gap **663.4**, feels-gap **664.0**, discomfort **654.1** vs baseline 655.2. `corr(feels_gap, naive_gap) = −0.205` | ⚠️ **NULL** — see frame 11 rewrite |
| M3 | `temp × workingday` | Ridge **655.2 → 648.3** (+6.9); forest **455.1 → 456.1** (−1.1) | ✅ both signs correct |
| M4 | Binning `temp` | Ridge **655.2 → 596.2** (+59.0) | ✅ the deck's biggest single FE win |
| M5 | Group-by leak | scales with group cardinality: +1.5 (4 groups) → **+118.0 / 17.4%** (168 groups) | ✅ better than planned |
| M6 | Candidate pool | **230 candidates** (227 mechanical + 3 domain). 14,120 non-finite cells across 50 ratio columns | ✅ |
| M8 | Expansion + selection | Ridge **655.2 → 573.7** (expansion) **→ 505.5** (selection) | ✅ |
| M14 | **The thesis, model held fixed** | Ridge **655.2 → 573.7** (+81.5). Forest **455.1 → 459.4** (**−4.3**) | ✅ **frame 7, no confound** |

**The single most important result in the chapter (M14).** The first version of frame 7 compared
Ridge+engineering against a forest, which confounds feature engineering with model class. Holding
the model fixed instead: the same 219 engineered features are worth **+81.5 MAE to a linear model
and −4.3 to a forest**. Feature engineering transforms the model that cannot derive features and
does *nothing* for the model that can. That is the 2026 argument, cleanly, and it needed no
literature search.

**`hum` zero confirmed:** exactly one row, `2011-03-10`. But the wider result is better — `yr`,
`holiday`, `weekday` and `workingday` all contain zeros too, so mechanical ratio expansion produced
**14,120 non-finite cells across 50 columns**. Division by zero is pervasive, not a curiosity.

---

## Frame-by-frame (34 frames: 33 numbered + Outline)

### Cold open (before the Outline)

1. **"You have used these 11 features for three lectures. Which ones are real?"**
   The raw `bike-day.csv` header and one data row, exactly as on disk. Predict-first: *which columns
   came off a sensor or a calendar, and which did a human build?* Counter-intuitive enough for a `\pause`.

2. **The reveal: one raw column.**
   Provenance diagram *(TikZ, boxes-and-arrows, no data — permitted)*:
   - `dteday` → `yr`, `mnth`, `weekday`, `season`  *(same-row transformations)*
   - holiday calendar → `holiday`  *(external join)*
   - weather station → `weathersit`, `temp`, `atemp`, `hum`, `windspeed`  *(external join, then
     min-max normalized — which is why `temp` = 0.34 and not 12°C, callback to [22])*
   - `weekday` + `holiday` → `workingday`  *(a combination of two engineered features)*
   **Punchline:** the design matrix they have been interpreting for three lectures *is* a worked
   feature-engineering example, and it already contains every category this deck teaches.
   Figure: `fig/bike_provenance.pdf`

3. **And one that should never have been there.** **[MEASURE]**
   `casual + registered = cnt`, exactly, on **all 731 rows** (verified). Two shipped columns sum to
   the target. Fit on them and you get a near-perfect model that has learned nothing. Report that R²
   beside the honest forest R² of 0.879 from [22].
   Figure: `fig/fe_leak_casual_registered.pdf`

### Outline

### Section 1 — What and why

4. **`[plain]` transition — "Three places a feature can come from."**

5. **Preprocess, engineer, select — three different verbs.**
   Fix the data ([03]) / create columns (this deck) / choose columns ([27]). One line and one code
   line each. Drop two of the old frame's three code blocks.

6. **Where does an ML project's time actually go?**
   Predict-first, then the bar chart. **Numbers labelled as widely-quoted industry surveys, not
   measurement** — say plainly nobody has measured this rigorously. The value is that students
   reliably guess the reverse, and that survives the honesty.
   Figure: `fig/fe_project_time.pdf`

7. **Does feature engineering still matter in 2026?** *(the thesis frame)* **[MEASURED]**
   **No longer blocked on research — the chapter answers it with its own data.** Show the 2×2, which
   holds the model fixed and varies only the features, so there is **no model-class confound**:

   | | 11 raw features | 230 engineered | feature engineering bought |
   |---|---|---|---|
   | **Ridge** | 655.2 | 573.7 | **+81.5** |
   | **Random forest** | **455.1** | 459.4 | **−4.3 (slightly worse)** |

   **The same 219 engineered features that transform a linear model do nothing for a forest** —
   because the forest was already deriving them internally, which is precisely what [22] taught with
   nested splits. That is the whole 2026 argument in one table, measured on their data.
   *(And with [27]'s selection pass Ridge still only reaches 505.5 — see [27] frame 8.)*
   Feature engineering still wins where the model **cannot get there alone**: external joins (no
   model invents the weather — frame 2 proves it), domain knowledge, small data, and leakage-free
   time features.
   **Closing line, and the chapter's thesis: *generate cheaply, select ruthlessly.*** Mechanical
   expansion is fine as a *candidate* step precisely because [27] exists to prune it.
   *(Still worth a quick literature check to see whether the general claim is supported, but the
   frame no longer depends on it.)*

8. **When the model does this for you.** *(moved here from Section 4 — same argument, other direction)*
   Hand-engineering is what you do when the model cannot get there itself. Convolutional networks
   learn pixel features, transformers learn token features; for **tabular** the picture is genuinely
   mixed. Forward-pointer to the NN chapter, no mechanics.

9. **The three sources.**
   Same row / many rows / outside. Each illustrated with a **bike** column from frame 2, so the
   taxonomy is grounded in something they have already interpreted. *(Maps onto sections 2-4; the
   deck has six sections in total.)*

### Section 2 — From the same row

10. **`[plain]` transition — "One row in, one feature out."**

11. **Ratios and differences — and why you must know your units.** **[MEASURED — it is a null, and
    that is now the frame]**
    The category linear models cannot construct. BMI, loan-to-income, debt-to-equity as canonical.
    **Bike:** the naive `atemp − temp` is **meaningless** — the columns use different normalizations
    (`(T+8)/47` vs `(T+16)/66`), so the difference is an arbitrary affine mix. Measured
    `corr(feels_gap, naive_gap) = −0.205`: not merely noisier, **anti-correlated**. Reconstruct
    Celsius first (`47·temp − 8`, `66·atemp − 16`), *then* difference — the real gap spans
    −26.0 to +8.2 °C, the naive one spans −0.481 to +0.042 of nothing.
    **This is the deck's sharpest callback to [22]:** the normalization taught there is a
    constraint, not trivia.

    **Then the honest result.** Added to the 11 raw features, Ridge test MAE:
    baseline 655.2 → naive gap **663.4** → correct feels-gap **664.0** → discomfort **654.1**.
    *The correct domain feature made the model slightly worse.*

    **The twist, and why the frame is better for it** *(forward-pointer to [27])*: in the
    230-candidate pool, RFE-CV **keeps** `feels_gap_C` — one of only 2 hand-designed features among
    45 survivors. A feature that is useless bolted onto a small model earns its place once the model
    can use it in combination. This is [27]'s independence trap, two decks early, and it is a far
    better lesson than "ratios help".
    Figure: `fig/fe_ratio_gain.pdf` (the MAE ladder **and** the units comparison)

12. **Interactions: what a linear model cannot see.**
    Callback to [22]'s main-effect/interaction definition. `φ(x) = (1, x₁, x₂, x₁², x₂², x₁x₂)`.
    **Practical gotcha:** polynomial features blow up numerically if you do not scale first —
    `x` in the hundreds means `x³` in the millions, and the penalty in Ridge/Lasso then falls almost
    entirely on the low-order terms. Scale, then expand.

13. **Predict-first: do you need `temp × workingday` for (a) Ridge, (b) Random Forest?** **[MEASURE]**
    Yes / no. Trees find it via nested splits. **Polynomial features are a linear-model crutch, not
    a universal good** — old pitfall #6, promoted to a punchline. Bike has a real commuter-vs-leisure
    interaction and [23] measured overall H_temp = 0.21.
    Figure: `fig/fe_interaction_lin_vs_tree.pdf`

14. **Binning: when a straight line is the wrong shape.** **[MEASURE]**
    Equal-width / equal-frequency / supervised. Bin when the effect is **non-monotonic** and the
    model is linear. **Bike is the ideal case** — [23]'s ICE/PDP curves show `temp` rising then
    *falling* at the top (too hot to cycle). A linear model cannot express that; five quantile bins
    can. Ridge on raw `temp` vs binned. One line: trees bin implicitly and never need this.
    Figure: `fig/fe_binning_temp.pdf`

15. **Callback, not a lesson: log transforms and datetime components.**
    Single frame pointing at [03] for `np.log1p`, `PowerTransformer`, `dt.month`/`dt.dayofweek`,
    cyclic sin/cos. No code. Exists so students know it is *deliberately* not repeated.

### Section 3 — From many rows

16. **`[plain]` transition — "A group of rows in, one feature out."**

17. **Group-by aggregations.**
    Attach the group's profile to each row: mean, median, std, count, percentile rank, ratio to group
    mean. **Bike:** mean `cnt` per `(season, workingday)` cell; each day's rank within its month.

18. **The leak hiding in the previous frame — and what controls its size.** **[MEASURED]**
    Group mean on the full dataset then split → the target leaks into a feature. Train-fold only →
    honest. **Replaces the old deck's fabricated "32 vs 24 kAMD, ~25%"** — and the measurement found
    something better than a single number. Sweeping the grouping key's cardinality (CV MAE):

    | grouping key | groups | rows/group | honest | leaky | the lie |
    |---|---|---|---|---|---|
    | `season` | 4 | 182.8 | 648.9 | 647.4 | +1.5 (0.2%) |
    | `season × workingday` | 8 | 91.4 | 646.2 | 644.2 | +2.0 (0.3%) |
    | `mnth` | 12 | 60.9 | 622.0 | 617.7 | +4.2 (0.7%) |
    | `mnth × weekday` | 84 | 8.7 | 660.5 | 607.4 | **+53.0 (8.0%)** |
    | `yr × mnth × weekday` | 168 | 4.4 | 678.1 | 560.1 | **+118.0 (17.4%)** |

    **The rule, which no fixed percentage could have taught:** leakage severity scales with how
    precisely the group identifies the row. At 183 rows per group the mean is a coarse summary; at
    4.4 rows per group it is nearly the row's own target.
    **The trap, in the bottom row:** the leaky version looks like the *best* feature in the table
    (560.1, baseline 660.5) while the honest version is the *worst* (678.1 — actively harmful).
    The more attractive the number, the more the feature is lying to you.
    Figure: `fig/fe_groupby_leakage.pdf` (honest vs leaky vs baseline against rows-per-group)

19. **Out-of-fold encoding: how to do it right.**
    The mechanics frame 18 implies but does not show: per fold, compute the aggregate on the *other*
    folds; at inference use the full-train statistic. Smoothing toward the global mean for small
    groups. `category_encoders` CV-aware encoders. *(Per the interview, leakage stays attached to its
    technique rather than becoming its own section.)*

20. **Getting your feature into the Pipeline.** *(new — closes the gap frames 18-19 open)*
    Frames 18-19 give a rule students cannot obey with pandas alone: engineer in a notebook,
    cross-validate, leak. The fix is to make the feature a **transformer**, so `fit` sees only the
    training fold. `FunctionTransformer` for stateless features (ratios, differences — no fitted
    state, safe anywhere); a small `BaseEstimator + TransformerMixin` class for anything that
    **learns a statistic** (group means, bin edges, encoders). Callback to [03]'s
    `Pipeline`/`ColumnTransformer` frame — same idea, one level up.
    **This is the frame students need within ten minutes of opening a notebook.**

21. **Handed forward: lags, rolling windows, and the forward split.**
    One frame, no code. Everything above assumed rows are exchangeable. When they are ordered in
    time, aggregation gets a second rule — *only look backwards* — and that discipline (trailing
    windows, `shift` before `rolling`, `TimeSeriesSplit`) is the spine of the time-series chapter.
    `paramgreen` forward-pointer box.

### Section 4 — From outside the data

22. **`[plain]` transition — "The information is not in your table."**

23. **Joins: the highest-leverage features you can add.**
    Back to frame 2 — `holiday` and the five weather columns *are* joins. Somebody decided weather
    mattered and went and got it. Types: calendar, weather, demographic, reference data.
    **Point-in-time correctness:** join the value *as known on that date*, not today's revision.
    Ties directly to frame 7's first bullet.

24. **Geospatial.** *(openly standalone — say so plainly on the slide)*
    `bike-day.csv` is daily **system-wide** totals with no station dimension, so there is no
    geospatial feature to build on our running data. Separate worked illustration instead: haversine
    distance to a city centre, station/POI density, cluster IDs from k-means on coordinates. Show
    that raw lat/lon are weak linear features while distances and clusters are not. Mention geohash
    and H3 briefly. **No pretence of continuity** — the slide says "different dataset, same idea."
    Figure: `fig/fe_geo_illustration.pdf`

25. **High-cardinality: crosses.**
    Target and frequency encoding were [03] — callback, do not re-teach. New content: the **cross**
    (`district × rooms` → `Kentron_3br`), the combinatorial explosion, the embedding alternative
    (pointer to frame 8). Cite Wide & Deep.

26. **Text, in one frame.** *(labelled illustration)*
    Three layers: engineered scalars (length, caps ratio, punctuation) → bag-of-words / TF-IDF →
    embeddings. Explicit pointer, embeddings deferred to NLP. One TF-IDF snippet.

### Section 5 — Generating candidates at scale

27. **`[plain]` transition — "Now do it 200 times."**

28. **Systematic expansion: the candidate pool.** *(the frame that feeds deck [27])* **[MEASURED]*
    Mechanically expand the 11 features with ~30 lines: all pairwise products (55), differences (55)
    and ratios (110), plus the originals. **231 raw → 230 final** after dropping 3 constant and 1
    duplicated column, of which **227 are mechanical and 3 hand-designed** (`feels_gap_C`, `temp_C`,
    `temp_bin` — the last three are *not reachable* by mechanical expansion, because they require
    knowing the normalization constants from the dataset documentation. That is what domain
    knowledge buys, concretely).
    Most of the 230 is junk — which is the point, and [27]'s opening problem. Frame 7's *generate
    cheaply, select ruthlessly* is the licence.

    **The data-quality discovery, on the slide.** Ratio expansion produced **14,120 non-finite cells
    across 50 columns**. Five of the 11 features contain zeros: `holiday` (710 rows), `yr` (365),
    `workingday` (231), `weekday` (105) — and `hum` (**1** row, `2011-03-10`). The first four are
    legitimate binary/indicator zeros. **The last one is not:** 0% humidity in Washington DC in March
    is physically impossible. It is an unflagged missing value that four lectures of students never
    saw, and mechanical expansion is what surfaced it.
    Show the `inf`, then the guard, then the lesson: **expansion is also a data-quality probe.**
    Figure: `fig/fe_candidate_pool.pdf`

29. **Automated feature engineering: the libraries.** *(awareness only — the measured comparison is cut)*
    featuretools / Deep Feature Synthesis, tsfresh, autofeat. What DFS emits on a **relational**
    schema, which is what it is actually for. Honest caveat: bike-day is a single flat table, the
    case DFS is weakest on — which is why we are not benchmarking it here. Use for candidate
    generation and exploration, not production.

### Section 6 — Pitfalls and production

30. **`[plain]` transition — "Five ways to shoot yourself."**

31. **Five pitfalls.**
    Target leakage (frames 3, 18, 19) / look-ahead bias (pointer to time series) / train-serve skew /
    hard-coded statistics (frame 20 is the fix) / over-engineering. Old #6 is now frame 13.

32. **Feature stores.**
    Genuinely unique material and the chapter's only production-facing frame. Definition, batch vs
    online, Feast / Tecton / Vertex / SageMaker, train-serve consistency. Close with: for coursework
    a `Pipeline` (frame 20) is enough. *(Diagram is TikZ, no data — permitted.)*

### Wrap-up

33. **Recap** + `paramgreen` **"Next: [27] — you now have ~200 features. Which do you keep?"**

---

## Figures (`py_src/fe_figs.py` → `fig/`)

| File | Frame | Measured? |
|---|---|---|
| `bike_provenance.pdf` | 2 | TikZ, no data |
| `fe_leak_casual_registered.pdf` | 3 | **yes** |
| `fe_project_time.pdf` | 6 | survey numbers, labelled folklore |
| `fe_ratio_gain.pdf` | 11 | **yes** |
| `fe_interaction_lin_vs_tree.pdf` | 13 | **yes** |
| `fe_binning_temp.pdf` | 14 | **yes** |
| `fe_groupby_leakage.pdf` | 18 | **yes — replaces the fabricated 25%** |
| `fe_geo_illustration.pdf` | 24 | separate data, labelled |
| `fe_candidate_pool.pdf` | 28 | **yes — feeds deck [27]** |

*(`fe_auto_vs_manual.pdf` cut with the head-to-head.)*

All: `ma` venv, seed 509, **same 70/30 split as [22]-[24]** so every number is comparable to slides
students have already seen. Armenian flag colours for 3+ series, `ax.bar_label()` on bars, logging
to `logs/`.

**The candidate pool is a build artefact, not just a figure.** `py_src/fe_figs.py` must write the
~200-column matrix to disk (`data/bike_candidates.parquet` or similar) so `py_src/fs_figs.py` in
deck [27] consumes exactly the same pool. Without that the chaining is decorative.

**Compute:** all small (731 rows). Run sequentially, one script, no parallel agents.

## Deliverables beyond the deck

- `ml/06_feature_engineering/06_feature_engineering.qmd` — does not exist; `_quarto.yml` jumps 05 → 08.
- `ml/00_random_image/06_*.jpg` — missing, so the qmd would have no 🎲 Random section.
- Housekeeping: `figures/` holds 24 unused LMU evaluation/tuning PDFs; `L03_OUTLINE.md` is the
  *regularization* deck's outline in the wrong folder; `py_files/`, `py_src/`, `fig/` are empty.
  `git rm` `L01g_feature_engineering.{tex,pdf}` once this compiles.

## Open questions

1. **Frame 7 is blocked on research.** If the "gradient boosting has eaten mechanical FE" claim does
   not survive a literature check, it narrows to a measured claim about bike. I will bring you what
   I find before writing it.
2. **Cut candidates if it runs long**, in order: 26 (text), 24 (geo), 29 (automation). All three are
   self-contained with no downstream dependency. **Do not cut 20** — it is load-bearing for 18-19.
