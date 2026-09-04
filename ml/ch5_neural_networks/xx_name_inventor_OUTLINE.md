# The Armenian Name Inventor - practical outline (for approval)

Drafted 2026-09-05 (Claude), from the interview decisions below. Cell-by-cell plan for the
chapter 5 (neural networks) flagship practical: a character-level MLP language model that
invents Armenian surnames. See `REVIEW.md` sections 4-8 for the ideation trail.

## Locked decisions (instructor interview, 2026-09-05)

- **Task:** char-level MLP language model on Armenian surnames; the net generates new ones.
- **Data:** the Armenian Wikipedia category "Հայկական ազգանուններ" - measured: **705 pages**,
  669 ending in -յան, native script, a few non-Armenian-pattern strays (e.g. Աբալակով).
- **Ladder:** straight to the MLP (no count-table or single-neuron warm-up stages).
- **PyTorch:** first contact for students; introduced **inline, as we go** - every new torch
  concept gets an explanatory markdown moment at first use.
- **W&B:** first contact with experiment tracking; **runs + samples table + a sweep**.
  Students need a free wandb account (note in the notebook).
- **Prose:** English. **Format:** instructor-led walkthrough (narrated solution notebook run
  in class; no TODO gaps).
- **Bonuses:** all four - temperature, memorization check, vowel-PCA of embeddings,
  conditional generation.
- Fashion-MNIST practical: left as is for now; its fate decided later.

## Supporting files to build

- `py_src/fetch_surnames.py` - prep script: pulls the category members from the hy.wikipedia
  API, filters to the Armenian unicode block, lowercases, dedupes, writes
  `data/surnames_hy.txt` (pinned in the repo so the notebook has no network dependency).
  Logging to `logs/`, fail-loud, seed 509 where relevant.
- `xx_name_inventor_solution.ipynb` - the walkthrough notebook (plan below).

---

## Cell plan

Numbers are cells; (md) = markdown, (code) = code. Every quantitative claim in prose must be
printed by a cell in the same notebook (2026-08-13 rule); claims marked **[pilot]** below are
written only after measuring.

### Section 0 - Framing

1. (md) **Title + pitch.** "Every ChatGPT ancestor tree ends at this model": Bengio et al.
   2003's neural language model was an embedding layer plus an MLP - exactly the parts from
   L14/L15. Today: train it on 705 surnames from Armenian Wikipedia and have it invent
   surnames that do not exist. Tooling preview: PyTorch (new - explained as we go) and
   Weights & Biases (new - "a lab notebook that writes itself"; today involves ~20 training
   runs, and nobody can hold 20 loss curves in their head).
2. (code) Imports, `SEED = 509`, `torch.manual_seed`, print torch version. Install note in a
   comment (`uv pip install torch wandb`; CPU is fine).

### Section 1 - Data

3. (md) Provenance: the wiki category, the prep script pointer, what cleaning was done and
   why (Armenian-block filter, lowercase so the vocab does not double, dedupe; the Russian
   strays like Աբալակով stay or go - **[pilot]** decide after seeing their effect on samples).
4. (code) Load `data/surnames_hy.txt`; print count, 10 samples, min/max/mean length, and the
   fraction ending in -յան (measured: 669/705 on the raw category - reprint after cleaning).
5. (md predict-first) "How many *distinct characters* do you think 705 surnames use?" then
   (code) build the vocabulary from the data, print it and its size. The alphabet is
   discovered from data, not hardcoded - and the special `.` start/end token is added here,
   with the explanation of what it is for (the net must learn how names *end*).
6. (code) Train/val split **by whole names** (80/20, seeded), print sizes. (md) one line on
   why we split names, not windows - windows from one name are near-duplicates (leakage
   callback, ch2/[06]).

### Section 2 - From characters to tensors (torch contact #1)

7. (md) The encoding story: why integer codes lie (magnitude/order that does not exist -
   categorical-encoding callback to the feature-engineering chapter), one-hots, and the
   context window. The sliding-window diagram on one worked name:
   `... -> Պ`, `..Պ -> ո`, `.Պո -> ղ`, ..., `յան -> .` (context K = 3 to start).
8. (md) Two-minute torch orientation, placed exactly where it is first needed: "a tensor is a
   numpy array that remembers how it was computed (for backprop, L15) and can live on a GPU;
   dtype and shape are the two things you check first."
   (code) `stoi`/`itos`; `build_dataset(names, K)` returning integer-code tensors `X (N, K)`
   and `Y (N,)`; print shapes and 5 rows decoded back to letters so the sliding window is
   visible in the tensors.

### Section 3 - The model (torch contact #2)

9. (md predict-first) Architecture diagram in text: chars -> embedding -> concat -> hidden
   ReLU -> logits -> softmax. The embedding taught as nothing new: "a one-hot times a matrix
   just selects a row; `nn.Embedding` skips the multiply and indexes the row - it is the
   first linear layer, computed lazily." Then: "count the parameters by hand before we ask
   torch" (emb V x d, layer 1 (K·d) x h + h, layer 2 h x V + V; students guess, next cell
   settles it).
10. (code) `NameMLP(nn.Module)` - `nn.Embedding`, two `nn.Linear`, ReLU; every line carries a
    short comment tying it to the L14 picture (affine + activation). Instantiate, print the
    module, `sum(p.numel() ...)` vs the hand count. Starting config **[pilot]**: K=3, d=8,
    h=128 - final numbers set after measuring.
11. (md + code) Forward pass **before any training**: feed the context "պող", show the
    softmax over next characters (top-5 print or small bar chart). Predict-first in the md:
    "what should an *untrained* net believe?" - roughly uniform, and the cell shows it. This
    is L14's worked forward pass, now in code.
12. (md + code) **Define `sample()` and run it on the untrained net.** The sampling loop is
    explained here (seed context with `...`, `torch.multinomial` on the softmax, slide, stop
    at `.`), and `torch.no_grad()` gets its explanatory moment ("we are predicting, not
    learning - no need to remember for backprop"). Generate 20 names from the untrained net:
    gibberish, printed - the "before" picture the trained net will be judged against.
    (Defining `sample()` here also lets the training function log sample tables without a
    forward reference.) Display convention: samples are title-cased for printing (training
    data is lowercased).

### Section 4 - Training (torch contact #3, W&B contact #1)

13. (md) The L15 training loop, quoted verbatim from the slide, then mapped line-by-line to
    what we are about to run: `zero_grad` / `backward` (this is the backprop from L15 -
    autograd runs the delta recursion for us) / `step`. Cross-entropy is the same log-loss
    from L11; one honest sentence on why torch's `F.cross_entropy` takes raw logits (it
    applies log-softmax internally - numerics, not new math). **Print the clueless
    baseline** here: uniform guessing over V characters costs `-log(1/V)` (about 3.7 nats at
    V ~ 40) - the reference line every loss curve is read against.
14. (md) W&B orientation: what a *run* is, what goes in `config` (every knob we might turn),
    `wandb.init` / `wandb.log` / `wandb.finish`; login instructions and the free-account
    note. **Class logistics fallback:** anyone without an account keeps moving via
    `WANDB_MODE=offline` (or `anonymous="allow"`) - nobody is blocked mid-walkthrough.
    Framing: "the discipline is that every experiment is recorded with its exact settings -
    by hand this dies at run 3."
15. (code) `train(config)` function: mini-batches, epochs; logs train/val loss per epoch
    (plus the uniform baseline as a constant) and, every few epochs, a **`wandb.Table` of 10
    sampled names** via the cell-12 `sample()` - so the dashboard shows the net learning to
    speak Armenian over time. Run it (~1-2 min CPU **[pilot]** exact time).
16. (md + code) **The payoff cell.** Same `sample()` as cell 12, now on the trained net: 30
    names, printed next to the untrained gibberish. Discussion prompts in md: which feel
    real, which *are* real (foreshadows the memorization bonus), did -յան emerge?
17. (md) Read the dashboard together: the loss curves against the 3.7 baseline (overfitting
    visibility at 705 names **[pilot]**), the samples table over epochs, and the config
    panel. The L15 read-your-curves frame, now on their own run.

### Section 5 - Twenty experiments without losing your mind (W&B sweep, [08] callback)

18. (md) What a sweep is; the search space: context K in {2, 3, 5}, embedding d in {4, 8,
    16}, hidden h in {32, 128}, lr in a small log-range; **optimized metric: val loss**,
    declared in the sweep config. Random search callback to [08] - same machinery, new
    model. Each run is seconds here, so a 12-16 run sweep is minutes **[pilot]** total time.
19. (code) Sweep config dict, `wandb.sweep`, `wandb.agent(..., count=12)` (the agent function
    also appends `(config, val_loss)` to a local list so the best config prints without the
    wandb API).
20. (md + code) Harvest: read the parallel-coordinates plot in the UI (screenshot-guided md),
    print the best config + val loss from the local list, retrain it as the final model and
    sample from it - **with the honest caveat that val loss is a proxy: eyeball the winner's
    samples before crowning it.** Which knob mattered most is written **only after the
    pilot** - no pre-claimed findings.

### Section 6 - Bonus acts

21. **Temperature.** (md) divide logits by T before softmax; (code) same net sampled at
    T = 0.5 / 1.0 / 1.5 side by side. Low T: conservative, near-real names; high T: wild
    inventions. The quality/diversity knob every generative product ships.
22. **Memorization check.** (md predict-first) "train 10x longer - better names?" (code)
    train long; every N epochs log val loss AND the % of generated names that are verbatim
    training names; plot both. **Nuance to state up front:** even a good model emits some
    real names by chance (short, high-probability ones like Աբելյան), so the baseline is not
    0% - what indicts memorization is the *trajectory* (verbatim % climbing while val loss
    worsens), measured against the early-training level. Shape **[pilot]**; with 705 names
    it should be vivid. One md paragraph ties it to LLM privacy (models reciting training
    data) and to L15 early stopping as the cure.
23. **Vowel-PCA.** (code) PCA (sklearn, ch10 callback) of the learned embedding rows to 2D,
    scatter with Armenian letter labels. (md) written as an investigation, not a promise:
    in the Bengio/makemore setup vowels separate from consonants - does it reproduce at our
    data size? **[pilot]** decides the prose.
24. **Conditional generation.** (code) steer by prefix: seed the context with chosen letters
    (e.g. the student's initials) and let the net finish - everyone generates their own
    personalized fake surnames. (md) one paragraph naming this "conditioning" and pointing
    at the grown-up versions (prompting is conditioning).
25. (md) **Closing.** What was built = Bengio 2003, the direct ancestor; what separates it
    from ChatGPT (much more data, much longer context, attention instead of a fixed window -
    named, not taught); the W&B habit carries to every later chapter. Forward pointer per
    the course Next-box convention.

---

## Pilot checklist (before writing any prose - 2026-08-13 rule)

1. Sample quality at 705 names with the starting config - are generated names charming or
   garbage? If garbage: try smaller model / more epochs / augment data (first-names category,
   translit-back of the big Latin lists) - in that order.
2. Overfit visibility on the val curve at this data size (drives cell 16's prose).
3. Memorization-curve shape (cell 21) - verbatim % vs val loss over long training.
4. Vowel/consonant separation in the embedding PCA (cell 22) - reproduce or reframe.
5. Wall-clock: single run and full sweep on the instructor laptop (CPU).
6. W&B free-tier UX check: samples table renders per-epoch, parallel-coords plot on the sweep,
   login flow from a fresh account.
7. Cleaning decision: keep or drop the non--յան / Russian-pattern strays (measure their
   effect on samples).

## What the pilot found (2026-09-05, measured; design assumptions corrected)

Pilot script + raw JSON: session scratchpad (`pilot_name_inventor.py`, `pilot_results.json`).
Data: 689 usable surnames (705 category pages minus disambiguations/hyphenated), 95.8% -յան,
vocab 38 letters + '.', uniform baseline loss 3.664.

1. **Sample quality: works.** Charming inventions by epoch 50 (Ավանիբեկյան, Զոհրաբյան,
   Գյուրգուլյան), 100% -յան endings from epoch 50 on. The flagship premise holds.
2. **Design wrong: "150-300 epochs of training" - the val minimum is at ~epoch 25-50**
   (val 1.70), and val loss climbs relentlessly after (3.0 by ep 300, 4.5 by ep 2000 -
   **past the uniform baseline**). Correction: `train()` keeps best-val weights (early
   stopping built in), and the "worse than guessing on new names while reciting old ones"
   moment became a core teaching beat of the memorization bonus.
3. **Design wrong: sweep compared by final val loss.** At a fixed epoch budget that punishes
   big models for overshooting their early val minimum, not for being bad (pilot grid: small
   d/h "won" final-val by huge margins). Correction: sweep metric = best val loss during the
   run.
4. **Memorization trajectory confirmed and vivid:** verbatim share of samples 3% (ep 100)
   -> 30% (ep 500) -> 47% (ep 2000) while val loss degrades. Also: temperature 0.5 on an
   overtrained model recites (50% verbatim) - low temperature amplifies memorization.
5. **Vowel-PCA: partial, not clean.** Separation ratio 1.17; ե/է/ի/օ drift together, ա sits
   among consonants. Prose written as an investigation ("suggestive, not clean; sharpens
   with data"), not as the makemore poster result.
6. **Wall-clock: mini-batch 256 costs ~0.5 s/epoch on this CPU** (per-step overhead
   dominates at this size). Corrections: batch 1024 throughout, sweep runs at 100 epochs,
   memorization bonus at 1200 epochs.

## Build incident (2026-09-05): wandb runs went to a stranger's account

First execution pushed runs to entity `hr-davtyan` - a stale `wandb login` left in this
machine's `~/_netrc` by someone else. Caught by the instructor ("I don't see the project").
Fixes: the key now comes from repo-root `.env` (`WANDB_API_KEY=`, gitignored; env var beats
netrc), the notebook's setup cell loads it and a guard cell prints
`runs will go to wandb entity: ...` before any training, and `wandb/` run dirs are
gitignored. Lesson: a stored login proves *a* login exists, not *whose* - check
`wandb.Api().default_entity` before pushing.

## Conventions

- Notebook prose English; seed 509; every number in prose printed by a cell; figures inline;
  fail-loud (no silent try/except); `ma` venv (`torch` 2.9.0+cpu and `wandb` - install wandb
  into `ma` if missing).
- Register on `neural_networks.qmd` once built as an **instructor-led practical** (like
  genes-geography on the ch10 page): summary + download/GitHub links. No graded cheese
  tasks - it is a walkthrough, not homework.
- Seed the sampling too (`torch.manual_seed` before each sampling cell) so reruns print the
  same names as the delivered session.
