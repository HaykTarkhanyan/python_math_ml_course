# Website check, 2026-11-12 (Claude, while the instructor was away)

How it was checked: `non_essential/check_links.py` (source and `--live`: 102 pages, 194 s), a scan of every YouTube
link's text against the video's real title (oEmbed, 394 videos), the Metric channel's public video list against
the pages, a scan for stale "no video yet" banners, and for lists that Pandoc renders as plain text. Live pages
were checked with Playwright after each deploy.

## Fixed and pushed (live-checked)

- `e3fa3ee` the API notebook renamed `llms/02_decoding/04_llm_api.ipynb`, ledger `results/04_api_ledger.json`.
- `8358158`:
  - video links (exact titles) for every public lesson no page linked: [22]-[28], [38]-[47], [53], [54],
    LLM [01]-[03];
  - the stale "videos not yet recorded / coming soon" banners on those pages;
  - [52]'s link text, which had no title;
  - the API practical on the decoding page (download + GitHub);
  - a blank line before 22 math-page lists that rendered as one paragraph (math/01, 03, 08, 20, 22, 23).
- `6a6e1a5`:
  - 16 video links on ML 01-04 now carry the exact YouTube title (missing suffix, titles renamed since);
  - a blank line before 62 lists in 21 Python / Python-libs notebooks.

## Decided with the instructor (2026-11-12, ~01:30)

- Q9 the 161 generic labels (item 3): **leave them for now**, no CONVENTIONS change.
- Q10 the attention page (item 4): **moved under 🧠 LLMs**, after decoding (`_quarto.yml`).
- Q11 the dead songs and shorts (item 2): **the instructor replaces them**.
- Q12 libgen / Anna's Archive links (item 2, last bullet): **keep as they are**.

## For you (decisions or content only you can supply)

1. **[04] LLM APIs on the decoding page** shows "📺 TBD": I did not link the unlisted video. When you make it
   public, the line is `[📺 [04] LLM APIs | Large Language Models](https://youtu.be/xdiq1N3KBPg)`.
2. **Dead links, need your replacement (or removal):**
   - `python/02_conditions.ipynb`: song "ժողովուրդ" `0UOa7qtqkIw` (removed / private)
   - `python/03_str_range_list_some_funcs.ipynb`: "Eric Clapton" `pKwQlm-wldA` (removed / private)
   - `python/05_lst_str_methods_one_line_if_for.ipynb`: "Jah Cure" `38A1NZEHpxY` (removed / private)
   - `python_libs/13_pydantic.ipynb`: "5am (Ուր էլ գնաս)" `XFUrPRTI2e6` (YouTube says the ID is invalid: typo?)
   - `python_libs/06_data_viz.ipynb`: "MonkeyExplains" `dHy07B-UHkE` (oEmbed 401: private or embedding off; check)
   - `python_libs/12_sql.ipynb`: a short, `rH7r85UwJSg` (private / login required)
   - `math/04_calc_lim_continuity_deriv.qmd`: "Random link" short `2sphKXkgzqE` (removed)
   - `math/05_calc_extrema_convexity_taylor.qmd`: "Random link" short `Z_Gnf2PA56w` (removed) and
     "Տիգրան Մանսուրյան (Հին Օրերի Երգը)" `f2UE7tJLsns` (removed)
   - `python_libs/18_clean_code_architecture.ipynb`: libgen.is, libgen.rs, annas-archive.org do not resolve
     from here (dead or blocked domains): keep, swap or drop?
3. **The exact-title rule on the older pages.** 161 links to our own videos use generic labels ("Տեսադասը",
   "Գործնականը", "Տեսագրությունը (2023)"), all in `python/`, `python_libs/` and `math/` pages (most: python/01 and
   02 with 8 each, python_libs/02_numpy, math/09, 17, 19 with 6 each). CONVENTIONS says link text = the exact
   title. Rewrite them all (mechanical, I can do it), or exempt the older pages and say so in CONVENTIONS?
4. **`llms/04_attention/attention.qmd` sits in the ML part of the menu** (between GANs and Diffusion), not under
   "🧠 LLMs". Move it? (`llms/README.md` calls it the older page covering LLM-4 to LLM-6.)
5. **"Not finished / drafted by Claude" warnings** on `ml/07_classic_methods` (still says "not taught yet"; [28]
   was taught) and `ml/13_rnns` (both lectures taught): remove once you have reviewed those pages.
6. **Unwritten practical sections:** "🛠️ Գործնական ToDo" headings on math/03, 04, 05, and a bare "ToDo" line above
   math/08's practical video list.
7. **`python_libs/10_scraping__parallelization.ipynb`**: 8 lists still render as plain text. Its JSON has a lone
   surrogate escape, so the byte-safe edit skipped it; insert a blank line before each list in Jupyter.

## Other sessions' work in progress (not touched; heads-up for when they are committed)

- `ml/ch19_mech_interp/`: 16 renames `xx_*` -> `0N_*` are staged and `mech_interp.qmd` is edited locally. Commit
  them together: the live page links the `xx_*` PDFs, which GitHub still has today.
- `ml/ch14_tabular_fm/`: the local `tabular_fm.qmd` links `L37b_tabpfn_followups.pdf` and
  `L37c_tabpfn_explanations.pdf`, which are untracked: add them in the same commit, or the links 404.
- `llms/02_decoding/xx_job_extraction_*` (untracked): the job-extraction practical.

## Not checked by hand

89 external links the checker could not decide (Unsplash, Stack Overflow, Facebook block scripts; a few TLS-chain
warnings). Full list: `logs/check_links_live.json`.
