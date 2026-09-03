# 34 — The semantic tree of Armenian words (project 4, design doc)

**Status:** **BUILT** 2026-08-21. Dataset `py_src/armenian_words.py`, embeddings
`data/armenian_words.npz` (563 KB), solution `xx_semantic_tree_solution.ipynb` (41 cells,
17 code, executed clean in 19s, validates, zero errors), task text live in
`09_clustering.qmd`. Every number below is measured from the shipped notebook.

**Model:** `Metric-AI/armenian-text-embeddings-2-large` — already in the local HF cache. An
XLM-R finetune of `intfloat/multilingual-e5-large`, and it cites arXiv **2603.22290**, which
is paper 08 in `ml/text_embedding/papers/`. The course's own reading list, the instructor's
own model, and the chapter's own practical close a loop.

---

## Why a fourth project

The three existing practicals all answer the same question - *which representation wins?* -
and answer it with pixels. This one asks a question none of them can:

> Fix the representation. Does the **metric**, the **linkage**, and the **tokenizer** still
> decide your clusters?

It is also the only project where the clusters are **readable**. You cannot look at a pixel
cluster and say whether it is right; you can look at `մայր, հայր, քույր, եղբայր` and know
instantly. That makes the dendrogram - not a score - the deliverable.

---

## The dataset

`py_src/armenian_words.py`, **108 leaves**: 99 words across **12 semantic families**, plus 9
synonym-partner words. All Armenian.

Families: ուտելիք, կենդանիներ, ընտանիք, գույներ, քաղաքներ, մասնագիտություններ, եղանակ,
տրանսպորտ, երաժշտություն, դպրոց, մարմին, բնություն.

Four kinds of planted probe:

| Probe | What it is | Purpose |
|---|---|---|
| `ORTHO_PROBES` | words sharing an opening, in different families | the central experiment |
| `PHRASE_FORMS` | 23 of those words also as a short natural phrase | controlled words-vs-phrases test |
| `SYNONYM_PAIRS` | 6 pairs that *should* merge first | the control |
| `POLYSEMY` | Արա, սեր, մարտ, բաց - one form, several senses, in full sentences | the hardest test |

**Every probe word also belongs to a real family.** This was a bug in the first draft: ձեռք
was probe-only, had no semantic home, drifted to its orthographic neighbours, and the result
was uninterpretable. A probe word with nowhere else to go proves nothing.

---

## Verified findings — all measured on the drafted list

### The headline: collisions track TOKENS, not letters

| Pair | Leading tokens | Merge height |
|---|---|---|
| ձու + ձի | both `▁ձ` | **0.505** |
| ձու + ձուկ | both `▁ձ` | 0.611 |
| ձու + ձյուն | both `▁ձ` | 0.748 |
| գարուն + գարեջուր | both contain `գար` | 0.703 |
| **ձեռք + ձի** | `▁ձեռք` vs `▁ձ` | **1.399** |
| արջ + արև | `▁ար`,`ջ` vs `▁արեւ` | 1.355 |
| քար + քամի | `▁քար` vs `▁`,`քա`,`մի` | 1.128 |
| մայր + մածուն | `▁մայր` vs `▁մ`,`ած`,`ուն` | 1.584 |

Median synonym pair: **0.605**. Median orthographic pair: **1.355**. Tightest pair in the
whole tree: **ձու + ձի at 0.505** — egg and horse, closer than any real synonym.

**ձեռք is the control that makes it a discovery rather than an assertion.** It begins with
the same letter as ձի and sits **1.6× further away**, because it tokenizes as one token. So
"the model groups by spelling" is refutable *within the dataset*, and the student has to open
the tokenizer to find the real rule.

### How much does it cost?

| Setup | ARI | AMI |
|---|---|---|
| all 99 words, bare | 0.287 | 0.415 |
| minus the 6 token-colliding words | **0.350** | 0.487 |
| phrase form used where available (23 of 99) | 0.321 | 0.433 |

**6 words out of 99 - 6% of the data - and removing them lifts the ARI by 22%.** Phrases recover
about half of that damage; they are a mitigation, not a fix.

### Linkage matters more than the metric

| Linkage / metric | ARI | AMI |
|---|---|---|
| ward / euclidean | **0.287** | 0.415 |
| complete / cosine | 0.251 | 0.394 |
| average / cosine | 0.117 | 0.313 |

### Euclidean vs cosine: mostly a no-op, and scipy will not warn you

On L2-normalized vectors `‖a−b‖² = 2 − 2·cos(a,b)`, so the two are monotonically related.
Measured: rank correlation of the two distance matrices **1.000000**.

- **single** and **complete** linkage → **identical** at every `k` tested (2, 3, 5, 8, 12, 16,
  20, 30). They depend only on the ordering of distances, so this is provable, not observed.
- **average** linkage → the merge **order** genuinely differs, but the effect is tiny: the
  partitions coincide at every `k` except 16 (ARI 0.9877). Worth knowing before anyone spends
  an afternoon agonising over the choice.
- **ward** is only defined for Euclidean. `linkage(X, method="ward", metric="cosine")` raises
  `ValueError: method=ward requires the distance metric to be Euclidean`, but
  `linkage(pdist(X, metric="cosine"), method="ward")` **runs silently**. The guardrail exists
  and is bypassed by the more advanced-looking calling convention.

### Linkage decides far more than the metric

| linkage | ARI | AMI |
|---|---|---|
| ward | **0.287** | 0.415 |
| complete | 0.251 | 0.394 |
| average | 0.126 | 0.326 |
| single | **0.001** | 0.001 |

Single linkage collapses completely — textbook chaining, on real data.

### Polysemy: the model picks the dominant sense

Senses matched to their own anchor sentence: **6/9**. Bare tokens and short phrases separate
none of them; only full sentences get this far, which is the argument for phrases over words.

Every failure is the **rarer** sense losing to the **dominant** one, and every failure is
narrow:

| Form | sense | result | margin |
|---|---|---|---|
| Արա | թագավոր | ✓ | +0.299 |
| Արա | դիմելաձև | ✗ | lost by 0.005 |
| Արա | հրամայական | ✓ | +0.132 |
| սեր | զգացմունք | ✓ | +0.326 |
| սեր | կաթնամթերք | ✗ | lost by 0.057 |
| մարտ | ամիս | ✓ | +0.270 |
| մարտ | ճակատամարտ | ✗ | lost by 0.084 |
| բաց | բացված / երանգ | ✓✓ | +0.140 / +0.100 |

`մարտ` is the month, not the battle. `սեր` is love, not cream. The model holds one prototype
per surface form and context is not enough to pull the minority reading out of it.

### Anisotropy is real, and is NOT what breaks polysemy

‖mean vector‖ = **0.647**; mean pairwise cosine **+0.385** with everything crammed into
0.249…0.661 and nothing negative. Centering genuinely opens that range up. It does **not**
repair the polysemy result:

| centring basis | polysemy | cosine range after |
|---|---|---|
| none | 6/9 | +0.249…+0.661 |
| the 18 polysemy texts | **7/9** | −0.235…+0.391 |
| the 108 word vectors | 6/9 | +0.023…+0.562 |
| everything | 6/9 | −0.040…+0.525 |

> **Correction to an earlier draft of this doc,** which claimed centering gives 9/9. That
> number came from centering *within each surface form's own 4–6 sentences* — a mean estimated
> from almost nothing, and one that includes the anchors being compared, so it flattered
> itself. Centered on any honest sample it is 6/9 or 7/9. The conclusion changes with it: the
> obstacle is dominant-sense bias, not anisotropy.

### Practicalities confirmed

- Armenian renders in matplotlib's default DejaVu Sans — **no missing glyphs**.
- The model returns **unit-norm vectors** through `sentence-transformers` regardless of the
  `normalize_embeddings` flag (std 0.0000), so the normalization trap does *not* exist by that
  route. The anisotropy/centering task replaces it, and is better.
- Model needs the **`query: ` prefix** on every input (E5 convention).
- Windows console is cp1252 and **cannot print Armenian** — `sys.stdout.reconfigure(encoding="utf-8")`.

---

## Proposed tasks (🧀🧀🧀)

1. **Build the tree.** Load the pre-computed embeddings, cluster with Ward, draw the
   dendrogram with Armenian labels. Read it: which families came out clean?
2. **Predict first.** Before looking: which two words in this list do you think merge first?
   Then find the actual answer (ձու + ձի) and say why that is alarming.
3. **Form your hypothesis and break it.** "It groups by spelling" is the obvious explanation.
   Find ձեռք, which begins with the same letter and sits 1.6× further away. Revise.
4. **Open the tokenizer.** Print `tokenizer.tokenize(w)` for every probe word. State the rule
   that predicts which pairs collide, and verify it against all six probe groups.
5. **Quantify the damage.** Score ARI/AMI against the families; drop the colliding words and
   re-score. How much does 6% of the vocabulary cost?
6. **Does context repair it?** Re-embed using the phrase forms and compare pair distances and
   the overall ARI. Does it fix the problem or only reduce it?
7. **Metric.** Prove `‖a−b‖² = 2 − 2cos` for your vectors, then test which linkages actually
   change when you switch. Explain why single and complete cannot change but average can.
8. **Linkage.** Compare ward / complete / average on the same data. Report ARI and say which
   assumption each makes.
9. **Polysemy.** Take Արա's three senses and their anchors. Which sense wins, and does it
   match? Repeat for սեր / մարտ / բաց.
10. **Name every cluster automatically** from the words it contains, and find one cluster that
    is a tokenizer artifact rather than a meaning.

**Bonus:** anisotropy — measure the mean pairwise cosine, then centre on four different bases
and see how little the polysemy score moves (6/9 → 7/9 at best, and only for the least honest
basis); try `armenian-text-embeddings-2-base` and see whether the smaller model collides more;
add your own words and try to break it further.

---

## Build plan

| Artifact | Status |
|---|---|
| `py_src/armenian_words.py` | **built** — self-reviewed, instructor review still welcome |
| `py_src/embed_armenian_words.py` → `data/armenian_words.npz` | **built**, 563 KB |
| `xx_semantic_tree_solution.ipynb` | **built**, 41 cells, executes clean |
| qmd task text in `09_clustering.qmd` | **built**, 10 tasks + bonus |

`embed_armenian_words.py` **asserts** the probe design rather than trusting it: it recomputes
the shared-token test for all six groups and raises if any group stops behaving as the doc
claims. If the model or tokenizer is ever swapped, the build fails loudly instead of quietly
invalidating the task text.

### An unplanned finding worth keeping

The shipped tree contains a collision nobody planted: `հայր, մայր, մազ, քար, թառ, պար, անտառ,
ծառ, լեռ` cluster together — family, nature and music words sharing the **-ար / -այր / -առ**
ending. So it is not only shared *prefixes*; shared **suffix** tokens do it too. Good material
for whoever wants to extend task 4.

Students get the `.npz` and never download the model — same pattern as the CLIP practical.
Embedding 108 words + 23 phrases + 18 sentences takes about a minute on CPU.

---

## What needs the instructor, not me

1. **The Armenian itself** — spelling, and whether each word sits in the right family.
2. **The probe groups specifically.** The whole project rests on ձու/ձի/ձուկ/ձյուն reading as
   genuinely unrelated words to a native speaker, and on ձեռք reading as obviously the same
   kind of thing. If any of these feel forced, the lesson feels forced.
3. **The polysemy sentences** — especially «Արա, այ տղա, ինչո՞ւ ես ուշացել։», which has to
   sound like something a person would actually say.
4. **Family count.** 12 families over 99 words is ambitious; ARI 0.287 is modest partly
   because of that. Merging մարմին and բնություն into the others would raise the score but
   lose two probe homes.

## Risks

- **ARI 0.287 is low enough to read as "it does not work."** Mitigation: the tasks are written
  so the *tree* is the artifact and the score is diagnostic. Worth watching in review whether
  students come away thinking embeddings are bad.
- Model version drift: the local copy was built with sentence-transformers 5.2.3 and the `ma`
  venv has 5.1.2, which prints a warning. Results were consistent across runs, but the `.npz`
  should be generated once and committed so students are never exposed to it.
