# Copilot Instructions: Mathematical Homework System

## Overview
This document provides comprehensive instructions for working with the mathematical homework system built using Quarto, JavaScript, and structured markdown files. The system features visual difficulty indicators, organized problem sets, and standardized formatting.

## System Architecture

### Core Components
1. **Quarto (.qmd) Files**: Main homework content with YAML headers and mathematical notation
2. **JavaScript (homework-scripts.js)**: Automated cheese emoji difficulty indicators
3. **YAML Configuration (_quarto.yml)**: Main book configuration integrating homework files
4. **HTML Data Attributes**: `data-difficulty` system for problem classification

### File Structure
```
math/
├── Homeworks/
│   ├── 00_intro_sets_comb_funcs.qmd    # Main homework file
│   ├── homework-scripts.js              # Cheese emoji functionality
│   └── COPILOT_INSTRUCTIONS.md         # This file
└── _quarto.yml                          # Main configuration
```

## Homework File Standards

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
## 01: Problem Title {data-difficulty="2"}
Problem content with $mathematical$ notation...
```

### Section Organization
Organize problems into logical sections using heading-1 (`#`):
1. **Sets** - Set theory, inclusion-exclusion, intersections
2. **Combinatorics** - Counting, permutations, combinations
3. **Functions** - Function properties, mappings, bijections

### Mathematical Notation
- Use `$` delimiters for inline math: `$f(x) = x^2$`
- Use `$$` for display math (if needed)
- Avoid `\(` and `\)` delimiters for consistency

## Difficulty System

### Cheese Emoji Indicators
- **Level 1** 🧀: Basic concepts, direct application
- **Level 2** 🧀🧀: Intermediate difficulty, requires reasoning
- **Level 3** 🧀🧀🧀: Advanced problems, complex analysis

### JavaScript Implementation
The `homework-scripts.js` automatically adds cheese emojis based on `data-difficulty` attributes:

```javascript
function addCheeseEmojis() {
    const problems = document.querySelectorAll('h2[data-difficulty]');
    problems.forEach(problem => {
        const difficulty = parseInt(problem.getAttribute('data-difficulty'));
        const cheeseCount = Math.min(Math.max(difficulty, 1), 3);
        const cheeseEmojis = '🧀'.repeat(cheeseCount);
        
        if (!problem.textContent.includes('🧀')) {
            problem.innerHTML = cheeseEmojis + ' ' + problem.innerHTML;
        }
    });
}
```

## Content Guidelines

### Problem Types by Section

#### Sets Section
- Train/test split validation
- Inclusion-exclusion principle
- Set operations and counting
- Multi-license intersections
- Venn diagram applications

#### Combinatorics Section
- Pipeline counting (multiplication principle)
- Hyperparameter grid searches
- Permutations with repetition
- Real-world counting problems
- Armenian car plates (format: DD LL DDD)

#### Functions Section
- Function classification (injective, surjective, bijective)
- Real-world examples
- Small function counting
- Even/odd function properties
- Domain/range analysis

### Mathematical Rigor
- Always show work for complex problems
- Use proper mathematical notation
- Include concrete examples where appropriate
- State assumptions clearly (e.g., for car plate formats)

## Development Workflow

### Adding New Problems
1. Determine appropriate section (Sets, Combinatorics, Functions)
2. Assign sequential zero-padded number
3. Set appropriate difficulty level (1-3)
4. Use standardized problem format
5. Test cheese emoji functionality

### Modifying Existing Problems
1. Maintain existing numbering unless reorganizing
2. Preserve `data-difficulty` attributes
3. Keep mathematical notation consistent with `$` delimiters
4. Update section headers if moving problems

### File Organization
- Keep problems in logical topic order within sections
- Use descriptive problem titles
- Maintain consistent formatting throughout
- Include context for real-world problems

## Integration with Quarto

### _quarto.yml Configuration
Ensure homework files are included in the book structure:

```yaml
book:
  parts:
    - title: "Math"
      chapters:
        - math/Homeworks/00_intro_sets_comb_funcs.qmd
```

### Rendering
- Problems automatically get cheese emoji indicators
- Mathematical notation renders properly with Quarto
- HTML data attributes are preserved in output

## Best Practices

### Problem Design
- Start with concrete, relatable examples
- Progress from simple to complex within each section
- Include variety in problem types and contexts
- Provide hints for difficult problems

### Code Quality
- Keep JavaScript simple and focused
- Use semantic HTML attributes
- Maintain clean, readable markdown
- Follow consistent naming conventions

### Maintenance
- Regularly test emoji functionality
- Validate mathematical notation rendering
- Check for proper section organization
- Update instructions as system evolves

## Troubleshooting

### Common Issues
1. **Emojis not appearing**: Check `data-difficulty` attribute syntax
2. **Math not rendering**: Verify `$` delimiter usage
3. **Problems not organizing**: Confirm heading levels (h1 for sections, h2 for problems)
4. **Numbering conflicts**: Ensure zero-padded sequential numbering

### Debug Steps
1. Check browser console for JavaScript errors
2. Validate YAML headers in .qmd files
3. Confirm script inclusion in file headers
4. Test with simple problems first

## Future Enhancements

### Potential Features
- Difficulty-based filtering
- Progress tracking
- Solution toggling
- Interactive problem elements
- Category-based organization

### Scalability Considerations
- Modular problem organization
- Automated numbering systems
- Cross-reference capabilities
- Export to different formats

## Example Template

```markdown
---
title: "New Homework Set"
---

<script src="homework-scripts.js"></script>

# Section Name

## 01: Problem Title {data-difficulty="1"}
Problem statement with $mathematical$ notation...

a) First part

b) Second part

## 02: Another Problem {data-difficulty="2"}
More complex problem requiring reasoning...
```

This system provides a robust foundation for creating and maintaining mathematical homework sets with visual difficulty indicators and organized content structure.
