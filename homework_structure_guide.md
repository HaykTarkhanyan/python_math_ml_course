# Homework File Structure Guide

## Required YAML Header

Every homework file must start with a YAML header that includes:

```yaml
---
title: "Homework X: Topic Name"
toc: true
toc-location: right
toc-title: "Contents"
format:
  html:
    embed-resources: true
    css: homework-styles.css

execute:
  echo: false
  warning: false
  cache: true
---
```

### YAML Header Options Explained:

- **`title`**: The homework title following the pattern "Homework X: Topic Name"
- **`toc: true`**: Enables table of contents
- **`toc-location: right`**: Places TOC on the right side
- **`toc-title: "Contents"`**: Sets the TOC title
- **`format.html.embed-resources: true`**: Creates self-contained HTML files
- **`format.html.css: homework-styles.css`**: Links to the standard homework stylesheet
- **`execute.echo: false`**: Hides code chunks in output (for Python/R code)
- **`execute.warning: false`**: Suppresses warnings in output
- **`execute.cache: true`**: Enables caching for faster rendering

## Required JavaScript Section

After the YAML header, include the dark mode toggle functionality:

```html
<!-- Dark Mode Toggle Button -->
<button class="dark-mode-toggle" onclick="toggleDarkMode()" title="Toggle Dark Mode">
  <span id="theme-icon">🌙</span>
</button>

<script>
// Dark mode functionality
function toggleDarkMode() {
  const html = document.documentElement;
  const themeIcon = document.getElementById('theme-icon');
  
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme');
    themeIcon.textContent = '🌙';
    localStorage.setItem('theme', 'light');
  } else {
    html.setAttribute('data-theme', 'dark');
    themeIcon.textContent = '☀️';
    localStorage.setItem('theme', 'dark');
  }
}

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function() {
  const savedTheme = localStorage.getItem('theme');
  const themeIcon = document.getElementById('theme-icon');
  
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeIcon.textContent = '☀️';
  } else {
    themeIcon.textContent = '🌙';
  }
});
</script>
```

## Problem Structure

Each problem should follow this structure:

```markdown
## X Problem Title

[Problem statement here, using LaTeX math notation when needed]

::: {.content-visible when-profile="solution"}

### Solution {.solution-header}

[Solution content here]

:::
```


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
- **`.dark-mode-toggle`**: Styles the dark mode button
- **`.bonus-problem`**: Styles bonus problem headers
- **`.video-solution`**: Styles video solution links


### Naming Convention
- Files: `XX_topic_description.qmd` (where XX is zero-padded number)
- Example: `00_intro_sets_comb_funcs.qmd`

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
## Problem 1: Sample Problem Title

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
```

## Rendering

To render homework files:
```bash
quarto render homework_file.qmd --profile solution
```

This generates an HTML file with solutions visible. Omit `--profile solution` to render without solutions.
