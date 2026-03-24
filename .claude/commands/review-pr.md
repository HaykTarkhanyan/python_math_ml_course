# Review Pull Request

Review a pull request on this repository.

## Input
- $ARGUMENTS: PR number to review (e.g., "42")

## Instructions

1. **Fetch the PR details** using the GitHub MCP tools:
   - Get the PR title, description, and changed files
   - Check CI status and any existing review comments

2. **Analyze the changes** by reading each modified file:
   - For `.tex` files: Check LaTeX syntax, consistent use of color macros, TikZ correctness, and adherence to the Beamer template conventions
   - For `.ipynb` files: Check cell structure, code correctness, markdown formatting, and that the notebook follows the standard cell sequence
   - For `.qmd` files: Check problem numbering, difficulty tags, solution blocks, LaTeX math rendering, and callout formatting
   - For `.py` files: Check code quality, type hints, and Python 3.10+ idioms
   - For `_quarto.yml`: Check that new entries are in the correct section and properly formatted

3. **Check for common issues**:
   - Missing Colab badge links or broken URLs
   - Inconsistent problem numbering in homework files
   - Missing solution blocks for homework problems
   - LaTeX compilation issues (unmatched braces, missing packages)
   - TikZ diagrams that reference undefined colors or styles
   - Notebook cells missing outputs or with stale outputs
   - Files not added to `_quarto.yml`

4. **Post a review** with:
   - A summary of what the PR does
   - Specific line comments for issues found
   - Suggestions for improvements
   - Overall recommendation: approve, request changes, or comment
