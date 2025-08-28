# Homework File Structure Guide

## Required YAML Header

Every homework file must start with a simplified YAML header:

```yaml
---
title: "Homework X: Topic Name"
---
```

**Note**: Most configuration options (TOC, formatting, CSS, execution settings) are now handled by the main `_quarto.yml` file, so individual homework files only need the title.

### YAML Header Options Explained:

- **`title`**: The homework title following the pattern "Homework X: Topic Name"

**Previously required options now handled globally:**
- Table of contents settings
- HTML formatting options  
- CSS styling
- Code execution settings

## Required JavaScript Section

After the YAML header, include the JavaScript functionality:

```html
<!-- Include homework JavaScript functionality -->
<script src="homework-scripts.js"></script>
```

**Note**: The `homework-scripts.js` file contains the necessary JavaScript functionality for:
- Cheese emoji difficulty indicators for problem headers

## Problem Structure

Each problem should follow this structure:

```markdown
## Problem X: Problem Title

[Problem statement here, using LaTeX math notation when needed]

::: {.content-visible when-profile="solution"}

### Solution {.solution-header}

[Solution content here]

:::
```

### Cheese Emoji Feature

You can add cheese emojis (🧀) as a difficulty indicator prefix to problem headers by using the `data-difficulty` attribute:

```markdown
## Problem 1: Easy Problem {data-difficulty="1"}
## Problem 2: Medium Problem {data-difficulty="2"}  
## Problem 3: Hard Problem {data-difficulty="3"}
```

- **Usage**: Add `data-difficulty="X"` to any problem header where X is the difficulty level (1-3)
- **Result**: The specified number of 🧀 emojis will be automatically added as a prefix
- **Example**: `data-difficulty="3"` will add "🧀🧀🧀 " before the problem title
- **Maximum**: Limited to 3 cheese emojis maximum
- **Compatibility**: Works with other classes like `.bonus-problem`
- **Auto-detection**: The JavaScript automatically processes all h2 headers on page load


### Special Sections

#### Video Solutions
```markdown
::: {.video-solution}
[📹 Video Solution in Armenian](https://youtu.be/example)
:::
```

#### Bonus Problems
```markdown
## Problem Title {.bonus-problem}
```

#### Additional Problems
Mark additional/optional problems:
```markdown
## Problem X: Title (Additional)
```

## CSS Classes and Styling

The homework uses custom CSS classes:

- **`.solution-header`**: Styles solution headers
- **`.content-visible`**: Controls conditional content display
- **`.bonus-problem`**: Styles bonus problem headers
- **`.video-solution`**: Styles video solution links


### Naming Convention
- Files: `XX_topic_description.qmd` (where XX is zero-padded number)
- Example: `00_intro_sets_comb_funcs.qmd`

### Main Configuration
The homework files are included in the main `_quarto.yml` under the Math section:

```yaml
- part: Math
  chapters:
  - file: math/Homeworks/00_intro_sets_comb_funcs.qmd
  - file: math/Homeworks/01_another_homework.qmd
```

This centralized configuration handles:
- Global formatting settings
- CSS and theme management  
- TOC and navigation settings
- Code execution preferences

## JavaScript File Structure

The homework functionality is contained in `homework-scripts.js` which should be placed in the same directory as the homework files. This file includes:

### Functions Available:
- **`addCheeseEmojis()`**: Automatically adds cheese emojis to headers with `data-difficulty` attributes

### Automatic Initialization:
The script automatically runs when the page loads and:
1. Processes all problem headers for difficulty indicators
2. Adds appropriate cheese emojis

### File Location:
```
math/
  Homeworks/
    homework-scripts.js     # JavaScript functionality
    homework-styles.css     # CSS styles
    XX_topic_name.qmd      # Homework files
```

## Best Practices

1. **Consistent numbering**: Start problems from 1, increment sequentially
2. **Clear problem statements**: Include all necessary information
3. **Step-by-step solutions**: Break complex solutions into numbered steps
4. **Verification**: Include answer verification when possible
5. **Visual aids**: Use diagrams, plots, or figures when helpful
6. **Cross-references**: Link related problems or concepts
7. **Alternative methods**: Show multiple solution approaches when valuable

## Example Problem Template

```markdown
## Problem 1: Sample Problem Title {data-difficulty="2"}

Given vectors $\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$, find:

a) $\mathbf{a} + \mathbf{b}$
b) $\mathbf{a} \cdot \mathbf{b}$

::: {.content-visible when-profile="solution"}

### Solutions {.solution-header}

**Part a) Vector Addition**
$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \\ 9 \end{bmatrix}$$

**Part b) Dot Product**
$$\mathbf{a} \cdot \mathbf{b} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$$

:::

## Problem 2: Bonus Problem {.bonus-problem data-difficulty="1"}

This is a bonus problem with difficulty level 1.

## Problem 3: Regular Problem

This problem has no difficulty indicator.
```

## Rendering

To render homework files:
```bash
quarto render homework_file.qmd --profile solution
```

This generates an HTML file with solutions visible. Omit `--profile solution` to render without solutions.
