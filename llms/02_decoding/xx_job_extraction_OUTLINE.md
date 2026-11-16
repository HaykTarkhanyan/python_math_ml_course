# xx - From a job ad to checked JSON (practical after the API notebook)

**Built 2026-11-11** (outline approved by self-review at the instructor's request). Notebook `xx_job_extraction_solution.ipynb`,
generated and executed by `py_src/build_job_extraction_nb.py` (like `build_api_nb.py`), HTML export next to it.
`xx` becomes the YouTube number once it is taught.

## Instructor decisions (interview, 2026-11-09 and 2026-11-11)

- Industry-style task on the decoding folder: structured extraction (document AI), "don't make it too big".
- **An in-class practical, run by the instructor; students do not work on it.** Worked demo only (no predict
  prompts). Time is not a constraint.
- Placement: its own practical, after the API notebook (it uses OpenRouter, structured outputs and logprobs from there).
- Data: the 148 staff.am postings already in the repo, as they are.
- Fields: all five - seniority, minimum years of experience, skills, salary, work mode.
- Models: GPT-4o mini, DeepSeek V4.1 Flash, Gemma 4 26B (A4B), Llama 3.1 8B.
- Kept: a baseline, prompt-for-JSON vs strict schema, a logprob review queue, Armenian vs English, temperature 0
  run twice. Dropped: reading the cases where model and employer disagree, counting invented salaries.
- The earlier student-project brief (`job_extraction_hy.md`) was deleted.

## Data

`_knowledge/ml_job_market/data/staffam_2026-10-28.json`, key `jobs`: 148 postings (120 English, 26 Armenian, 2
Russian; median 2,237 characters). Contact details were removed when it was collected. **Correction 2026-11-11:**
those counts are the scraper's `lang` field (which language box was filled in); by letter count the texts are 116
English, 31 Armenian, 1 Russian, and the notebook now uses the letter count.

- **Input to the model:** `title`, `description`, `responsibilities`, `required_qualifications`,
  `additional_information`. Nothing else.
- **Answer key** (filled in on the site by the employer, never sent to the model):

| Field | Key | Filled | Use |
|---|---|---|---|
| seniority | `level` | 132 of 148 (16 "Not defined") | the main accuracy number |
| salary | `salary_from` / `salary_to` / `salary_currency` | 15 | exact-match check on those 15 |
| work mode | `remote` | 6 true, 142 false | a weak check, shown with that caveat |
| minimum years, skills | none | - | agreement between the four models, plus a few postings read |

Level mapping (instructor, 2026-11-11): Junior and Student -> junior, Mid level -> mid, Senior -> senior, C level ->
lead, Not defined -> left out of accuracy. That gives 132 scored postings: junior 26, mid 57, senior 44, lead 5.

## Schema (Pydantic, sent as a strict JSON schema)

`seniority`: junior / mid / senior / lead / not_stated · `min_years_experience`: integer or null · `skills`: list of up
to 10 strings · `salary`: {min, max, currency} or null · `work_mode`: on_site / hybrid / remote / not_stated.

## Structure

0. **Setup.** The B0 client and a cost ledger. Each open model is pinned to one provider that supports structured
   outputs, logprobs and temperature (`require_parameters`, no fallbacks), picked from the endpoints table in code.
   Thinking switched off for DeepSeek and Gemma, so the tokens and the logprobs belong to the answer. One live call
   per model on one posting; everything else is read from the saved run (`results/job_extraction_calls.json`), so the
   notebook re-executes in seconds in class.
1. **The task and the data.** One English and one Armenian posting, the five fields, the answer key and its gaps.
2. **Baseline.** The text-only rules of `_knowledge/ml_job_market/py_src/analyze.py` (title words, years of
   experience), without its `level` input, which is the answer. Seniority accuracy on the 132.
3. **Prompt for JSON vs strict schema.** All four models, both ways, on all 148: output that does not parse, output
   that breaks the schema (an unknown seniority, a string where a number belongs), seniority accuracy, salary on the
   15, tokens, cost, time.
4. **Four models side by side** (strict mode). Accuracy, cost per 1,000 postings and seconds per posting in one
   table; how often the models agree on years and skills; three failures read aloud.
5. **How sure was it?** The probability of the seniority value, from its token's logprob. Accuracy by confidence
   band, and the review-queue curve: share of postings sent to a person vs errors left.
6. **Armenian vs English.** Accuracy and tokens per posting by language, per model: LLM-1's tax in dollars per
   1,000 postings.
7. **Temperature 0, twice.** A second strict pass; how many postings got a different answer, in which fields, per
   model.
8. **Would you ship it?** One summary table (accuracy, invalid output, cost, speed, review load) and the
   notebook's total cost from the ledger.

## Predictions, written before building (the notebook reports what actually happened)

| Prediction | Check in |
|---|---|
| Strict schema: 0 invalid outputs for every model. Prompt only: some for Llama 3.1 8B (5-15%), about 0 for GPT-4o mini | 3 |
| Baseline 50-60% on seniority; the LLMs 70-85% | 2, 4 |
| The employer key caps everyone below 100% (titles like "Senior ... " filed under Mid level) | 4 |
| Low confidence is more often wrong: sending the least sure 20% to a person removes about half the errors | 5 |
| Armenian: lower accuracy and 2-4x the tokens on GPT-4o mini and Llama, less extra on DeepSeek and Gemma | 6 |
| Temperature 0: a few percent of answers change between the two runs | 7 |

## Cost (list prices on OpenRouter, 2026-11-11; about 1,000 input and 150 output tokens per call)

| Model | Per pass over 148 |
|---|---|
| GPT-4o mini | ~$0.036 |
| DeepSeek V4.1 Flash | ~$0.020 |
| Gemma 4 26B | ~$0.014 |
| Llama 3.1 8B | ~$0.004 |

All four: ~$0.075 a pass. The notebook makes three passes (prompt only, strict, strict again): **~$0.22 per full
run.** Building it takes about three runs plus small probes on 10 postings: **~$0.7 in total**. Armenian postings
use more tokens, so the ledger may come out higher; real numbers get reported.

## Self-review, 2026-11-11 (instructor: "you self review pls")

Checked with the providers table (free) and a probe: 2 postings (one English Senior, one Armenian Junior) x 4 models,
strict schema, logprobs, temperature 0. Cost $0.0013.

| Risk in the outline | Found | Change |
|---|---|---|
| One provider per model with strict schema **and** logprobs? | GPT-4o mini: OpenAI. DeepSeek: 15 of 36 endpoints (Morph cheapest, fp8). Gemma: 9 (NextBit, bf16). **Llama 3.1 8B: only CoreWeave, $0.22 / $0.22 per 1M, 10x the cheapest Llama provider** | Pick in code: cheapest endpoint with both, quantization bf16 or fp8 (4-bit endpoints skipped) |
| Strict mode works everywhere | 8 of 8 valid JSON | keep |
| Logprobs usable inside JSON | Entries for every model; CoreWeave's Llama now returns 60-76 (the one-entry reply of 11-09 did not come back). DeepSeek on Morph: 47 entries for 72 output tokens | Locate the seniority value by character offset in the output, not by token text; report entries vs output tokens per model; leave a model out of section 5, saying so, if its entries do not line up |
| Logprobs show the allowed values | GPT-4o mini on the Armenian posting: "jun" 99.1%, "mid" 0.9%; Gemma "mid" 99.6%, "junior" 0.4% | keep: confidence = probability of the value's first token |
| The models agree on easy postings | Armenian Junior posting: GPT-4o mini and Llama junior, Gemma mid, DeepSeek not_stated | good material for sections 4 and 6 |
| Armenian token tax | Armenian posting: GPT-4o mini 634 prompt tokens, Gemma 821, DeepSeek 1,124, **Llama 2,556** | Llama is the dearest model per pass here despite being the smallest: say so in sections 6 and 8 |
| 4 models x 3 passes x 148 = 1,776 calls | ~1-2 s each: 45-90 minutes in a row | The builder runs the calls with a small thread pool (network-bound, light on the laptop) and saves every call; the notebook makes one live call per model in class and reads the saved run for the rest, so a re-execution takes seconds and costs nothing (`RERUN = True` repeats the calls) |
| Salary "exact match on the 15" | The employer's salary field need not appear in the text (pay words in only ~7 texts) | Score salary only where the text states pay; count the rest separately |
| An honest "not_stated" counts as wrong against the key | it does | Seniority accuracy two ways: over all 132, and over the postings where the model committed; plus the share of not_stated |
| The baseline outputs entry / mid / senior only | it cannot say lead or not_stated | Map entry -> junior; say what it cannot express |
| Small groups (26 Armenian, 5 lead) | | Report counts next to every percentage |
| Truncated JSON | max 400 output tokens; 66-136 used in the probe | Record finish reasons; count "length" as invalid |

**Cost, measured per call in the probe:** GPT-4o mini ~$0.00013, DeepSeek ~$0.00005, Gemma ~$0.0001, Llama
$0.00012 (English) to $0.00058 (Armenian). Per pass over 148: about $0.019 + $0.008 + $0.014 + $0.029 = **$0.07**;
three passes **~$0.21** per full run, **~$0.5** to build.

## Files (when built)

- `py_src/build_job_extraction_nb.py`: writes, executes and exports the notebook.
- `results/job_extraction_calls.json`: every call (model, mode, posting id, raw output, parsed JSON, seniority
  logprob, tokens, cost, seconds). The artifact of record: the notebook's tables are computed from it.
- `results/job_extraction_endpoints.json`: the endpoint test and the chosen endpoint per model, reused on re-execution.

The calls file doubles as the cost ledger (cost per call). Measured 2026-11-11: 1,776 calls, 0 failed, **$0.389**;
the predictions above mostly failed, and the notebook ends with a predicted-vs-measured table.

- `results/job_extraction_logprob_mismatch.json`: what the four "do not line up" DeepSeek endpoints sent, from
  `py_src/non_essential/probe_logprob_mismatch.py` (section 4 reads it).

## Added after the first run (instructor, 2026-11-11)

- **A flagship for comparison:** GPT-6.1 Sol, strict pass only, low-effort thinking, no logprobs or temperature on
  offer. 148 calls, $0.361; seniority 30.3% overall, 81.6% where it named a level, 83 of 132 `not_stated`. Saved run:
  1,924 calls, $0.750.
- **More shown in the notebook:** sample rows of the run, the raw failures (Gemma's fences, Llama's schema break),
  real skills examples, a crosstab per model, the least-sure postings, one answer that changed on rerun; in section 1
  the data itself (a table, counts, one full record, staff.am's own skill tags) and how it was scraped.
- **A bug in the endpoint test:** `END_MARKS` lacked DeepSeek's `<｜end▁of▁sentence｜>`, so two of the four "do not
  line up" failures (Cohere, Makora) were ours. Fixed; the run keeps `parasail/fast`, and section 4 says so.
