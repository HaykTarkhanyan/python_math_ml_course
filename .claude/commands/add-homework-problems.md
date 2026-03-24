# Add Homework Problems

Add new problems to an existing math homework `.qmd` file.

## Input
- $ARGUMENTS: The module file to add problems to, topic area, number of problems, and any specific focus areas

## Instructions

1. **Read the existing file** to understand what problems already exist, the current numbering, and subtopic sections.

2. **Design new problems** following these rules:
   - Number problems sequentially continuing from the last existing problem
   - Use `###` headers with the pattern: `### NN Problem Title {data-difficulty="N"}`
   - Difficulty levels: `1` = 🧀 easy, `2` = 🧀🧀 medium, `3` = 🧀🧀🧀 hard
   - For bonus problems add `.bonus-problem`: `### NN Problem Title {.bonus-problem data-difficulty="1"}`
   - Aim for 40% easy, 40% medium, 20% hard

3. **Problem design philosophy** (intuition-first, real-world grounding):
   - Frame problems as real-world scenarios (ML pipelines, medical studies, polling, A/B tests, sports analytics, engineering, finance)
   - Use sub-questions (a/b/c/d) that build on each other
   - Add context boxes for ML/data-science motivation:
     ```
     ::: {.callout-tip collapse="true" appearance="minimal"}
     #### Context
     [Brief real-world context explaining why this matters]
     :::
     ```
   - Include LaTeX math notation: `$inline$` and `$$display$$`
   - Reference course notation (vectors as $\mathbf{v}$, matrices as capital letters, etc.)

4. **Add solutions** wrapped in profile-conditional blocks:
   ```
   ::: {.content-visible when-profile="solution"}
   ### Solution {.solution-header}

   **Part a)**
   [Step-by-step solution with math]

   **Part b)**
   [Solution continues...]
   :::
   ```

5. **Group problems by subtopic** if the file uses subtopic sections (e.g., "## Basics", "## Conditional Probability").

6. **Verify** that the file renders correctly with proper LaTeX and that problem numbering is consistent.
