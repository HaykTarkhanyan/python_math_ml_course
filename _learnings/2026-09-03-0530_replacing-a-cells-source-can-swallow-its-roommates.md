# Replacing a notebook cell's whole source can swallow content that shares the cell

**Symptom.** The genes-geography practical lost its entire Wrap-up section (the trust/don't-trust
table, the gradients-not-categories note, the references) and nobody noticed for two edit
rounds. It surfaced only because an adversarial reviewer's report discussed every part of the
notebook EXCEPT the wrap-up - the silence was the tell. Grep confirmed: 0 hits for "Wrap-up".

**Cause.** The builder had packed two logical sections ("Act 4 discussion" + "## Wrap-up") into
one markdown cell. A later fix patched that cell by matching a substring ("The misses are
almost all") and replacing the cell's **whole source** with the rewritten discussion - the
wrap-up half was collateral. Cell count stayed constant (the check that WAS run), so the loss
was invisible to the verification in place.

**Consequences / rules.**

- When patching a notebook cell, replace the matched **span**, not `cell.source` wholesale -
  or first assert the cell contains nothing beyond what the replacement covers
  (`assert cell.source.strip().endswith(...)`).
- One logical section per markdown cell in builders. Packed cells are what make whole-source
  replacement destructive (and they also hide section headers from Jupyter's outline - the
  reviewer flagged that independently as a pedagogy defect).
- Cell-count and error-count checks do not detect content loss. After any notebook patch, grep
  for the section headers that are supposed to exist.
- An external reviewer's *omissions* are evidence too: if a thorough review never mentions a
  section, check that the section is still there.

Caught 2026-09-03 during the second (subagent) review of
`ml/10_dimensionality_reduction/xx_genes_geography_solution.ipynb`; restored same day.
