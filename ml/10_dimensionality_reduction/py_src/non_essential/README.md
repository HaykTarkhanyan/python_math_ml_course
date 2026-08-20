# Non-essential scripts — ch10 (dimensionality reduction)

Maintained and expected to work, but not needed to build the decks or the chapter page.

| Script | What it does | When to re-run |
|---|---|---|
| `validate_lfw_project.py` | Answer key for **Project 2** in `10_dimensionality_reduction.qmd`. Reproduces the whole arc the assignment asks students to discover: the Olivetti→LFW accuracy collapse, the majority-class baseline, the correlation between PC1 and image brightness, and the three attempted fixes with their measured accuracies. Numbers as of 2026-08-16 are in the module docstring. | Before editing Project 2, or after an sklearn upgrade, to confirm the assignment's premises still hold. Downloads ~200 MB of LFW on first run (cached in `~/scikit_learn_data/`). |

Added 2026-08-16: written because Project 2 assigns a *diagnosis* rather than a recipe, so
the diagnosis had to be verified to be real before the project was set. It was — PC1
correlates with image brightness at **+0.998**.
