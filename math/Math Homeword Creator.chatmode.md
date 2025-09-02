---
description: 'Helps create "math for ML" homeworks with Quarto'
tools: ['codebase', 'usages', 'vscodeAPI', 'think', 'problems', 'changes', 'testFailure', 'terminalSelection', 'terminalLastCommand', 'openSimpleBrowser', 'fetch', 'findTestFiles', 'searchResults', 'githubRepo', 'extensions', 'runTests', 'editFiles', 'runNotebooks', 'search', 'new', 'runCommands', 'runTasks', 'copilotCodingAgent', 'activePullRequest', 'mssql_show_schema', 'mssql_connect', 'mssql_disconnect', 'mssql_list_servers', 'mssql_list_databases', 'mssql_get_connection_details', 'mssql_change_database', 'mssql_list_tables', 'mssql_list_schemas', 'mssql_list_views', 'mssql_list_functions', 'mssql_run_query', 'getPythonEnvironmentInfo', 'getPythonExecutableCommand', 'installPythonPackage', 'configurePythonEnvironment', 'configureNotebook', 'listNotebookPackages', 'installNotebookPackages']
---

# General
1. Always use Quarto (.qmd) files for homework content.
2. Include the JavaScript file `homework-scripts.js` in the header of each homework file to enable cheese emoji functionality.
3. Include `homework-styles` in the YAML header. Also, provide a title.
4. Structure problems with heading-2 (`##`) and zero-padded numbering (01, 02, etc.).
5. Whenever asked to create solutions for each exercise add them like this:
```markdown
::: {.content-visible when-profile="solution"}

### Solution {.solution-header}
...
:::
```

6. Organize problems into sections using heading-1 (`#`).
7. Whenever using lists (either numbered or bulleted) use a blank line before and after the list. Also, when numerating always use digits and not letters.


### Header Structure
```yaml
---
title: "Homework Title"
---

<script src="homework-scripts.js"></script>
```

### Problem Format
- Use heading-2 (`##`) for each problem
- Include zero-padded numbering (01, 02, 03, etc.)
- Add `data-difficulty` attribute with values 1-3
- Use descriptive problem titles

Example:
```markdown
## 01 Problem Title {data-difficulty="2"}
Problem content with $mathematical$ notation...
```

### Mathematical Notation
- Use `$` delimiters for inline math: `$f(x) = x^2$`
- Use `$$` for display math (if needed)
- Avoid `\(` and `\)` delimiters for consistency

## Difficulty System

### Cheese Emoji Indicators
- **Level 1** 🧀: Basic concepts, direct application
- **Level 2** 🧀🧀: Intermediate difficulty, requires reasoning
- **Level 3** 🧀🧀🧀: Advanced problems, complex analysis
### _quarto.yml Configuration
Ensure homework files are included in the book structure:

```yaml
book:
  parts:
    - title: "Math"
      chapters:
        - math/Homeworks/00_intro_sets_comb_funcs.qmd
```

## Best Practices

### Problem Design
- Start with concrete, relatable examples
- Progress from simple to complex within each section
- Include variety in problem types and contexts
- Provide hints for difficult problems
- Design problems to be at least somehow related to Machine Learning whenever possible. But when using not common knowledge concepts add a "Context" collapsable note explaining it.

## Example Template

```markdown
---
title: "New Homework Set"
---

<script src="homework-scripts.js"></script>

# Section Name

## 01: Problem Title {data-difficulty="1"}
Problem statement with $mathematical$ notation...

1. First part
2. Second part

## 02: Another Problem {data-difficulty="2"}
More complex problem requiring reasoning...
```