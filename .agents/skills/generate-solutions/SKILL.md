---
name: generate-solutions
description: >
  Generate detailed homework solution files (LaTeX PDF + Jupyter notebook)
  for a given homework .qmd file. Matches the established format from
  hw_05_06, hw_07_08, hw_09 solutions. Use when a new homework needs
  solutions or when exercises have been updated.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Glob
  - Grep
  - Agent
user-invocable: true
---

# Generate Homework Solutions

Given a homework `.qmd` file, generate a complete solutions package: a
detailed LaTeX PDF and a Jupyter notebook for Python exercises.

## Arguments

The user provides either:
- A homework qmd filename (e.g., `25_stat_hypothesis_testing.qmd`)
- A lecture number range (e.g., `09` or `07_08`)

## Step 1: Gather context

1. **Read the homework .qmd file** to get all exercises, their types
   (pencil/Python/both), difficulty levels, and exact problem statements.

2. **Read the most recent existing solution** to match format exactly:
   ```
   math/Lectures/stat/hw_*_solutions.tex   (LaTeX format reference)
   math/Lectures/stat/hw_*_solutions.ipynb  (notebook format reference)
   ```
   Use the newest one as the template. Key conventions to preserve:
   - Same `\documentclass`, packages, `\lstdefinestyle{python}` block
   - `\section*{Problem NN · Title}` with `\texorpdfstring`
   - `\subsection*{Exercise}` restating the problem, then `\subsection*{Solution}`
   - `\paragraph{(a)}` for subparts
   - `\boxed{}` for final numerical answers
   - `\textbf{Common mistake:}` callouts where students typically go wrong
   - Intuitive explanations after computations (italicized)
   - Python code in `\begin{lstlisting}[style=python]` blocks

3. **Read relevant lecture slides** (if they exist) to ensure solutions
   align with the notation and approach used in class.

## Step 2: Write the LaTeX solutions

Create `math/Lectures/stat/hw_NN_solutions.tex` with:

### For each problem:

1. **Exercise block**: restate the problem verbatim from the .qmd
2. **Solution block** with detailed, step-by-step work:
   - Show every intermediate computation (students are learning)
   - Explain *why* each step works, not just *what* to compute
   - Use `\boxed{}` for final answers
   - Add `\textbf{Common mistake:}` where students typically err
   - Add `\textit{Intuition/interpretation}` after key results
   - For Python problems: include code in `lstlisting` blocks AND
     say "See the accompanying Jupyter notebook for all code and plots"

### Quality checklist:
- [ ] Every numerical answer has been verified (run Python to check)
- [ ] SE formulas use correct variable (p0 for tests, p_hat for CIs, etc.)
- [ ] One-sided vs two-sided is explicitly addressed
- [ ] Effect sizes / practical significance mentioned where relevant
- [ ] Forward/backward references between related problems

## Step 3: Write the Jupyter notebook

Create `math/Lectures/stat/hw_NN_solutions.ipynb` with:

1. **Title cell** (markdown): "Homework Solutions: Lecture NN - Topic"
   + note pointing to PDF for analytical solutions
2. **Setup cell**: imports + Armenian flag colors
   ```python
   import numpy as np
   from scipy import stats
   import matplotlib.pyplot as plt

   RED = '#D90012'
   BLUE = '#0033A0'
   ORANGE = '#F2A800'
   ```
3. **For each Python problem**:
   - Markdown cell restating the exercise
   - One code cell per subpart (not one giant cell)
   - Use `np.random.seed(509)` (course default seed)
   - Plots with Armenian flag colors, clear labels, appropriate titles
   - Print statements explaining results (students read the output)
   - When implementing algorithms (BH, bootstrap, etc.), implement
     manually FIRST, then verify with library functions

### Notebook conventions:
- `figsize=(10, 5)` for single plots
- `figsize=(14, 5)` for side-by-side plots
- Histogram: `density=True, alpha=0.4, edgecolor='white'`
- Reference lines: `linestyle='--', linewidth=2`
- Legend: `fontsize=9` or `fontsize=10`

## Step 4: Verify everything

1. **Compile the LaTeX**: `pdflatex -interaction=nonstopmode FILE.tex`
   - Must compile with zero errors
   - Check page count is reasonable (8-12 pages typical)

2. **Verify all math with Python**: run key computations to confirm
   numerical answers match what's written in the LaTeX

3. **Validate notebook syntax**: parse all cells with `ast.parse()`

## Step 5: Link from homework .qmd

Add solution links to the homework page's lecture section:
```markdown
- [📄 Solutions PDF](Lectures/stat/hw_NN_solutions.pdf), [🐍 Jupyter notebook](Lectures/stat/hw_NN_solutions.ipynb)
```

## Step 6: Report

Tell the user:
- Number of problems solved (pencil vs Python vs both)
- PDF page count
- Any problems that were tricky or where assumptions were needed
- Ask if they want to commit
