# Create Python Lesson Notebook

Create a new Jupyter notebook (.ipynb) for the Python or Python Libraries course section.

## Input
- $ARGUMENTS: Module number, topic name, and which section it belongs to ("python" or "python_libs")

## Instructions

1. **Determine filename and location**:
   - Python fundamentals: `python/NN_TopicName.ipynb`
   - Python libraries: `python_libs/NN_topic_description.ipynb`

2. **Use the standard notebook structure** (as Jupyter .ipynb JSON). Every notebook must follow this pattern of markdown and code cells:

### Cell sequence:

**Cell 1 (Raw/YAML):**
```yaml
---
title: "NN Topic Title"
lightbox: true
code-fold: false
---
```

**Cell 2 (Markdown) — Header with photo + attribution:**
```markdown
![image.png](../background_photos/PLACEHOLDER.jpg)
[Photo credit](UNSPLASH_LINK), Author: [Name](AUTHOR_LINK)
```

**Cell 3 (Markdown) — Colab badge:**
```markdown
<a target="_blank" href="https://colab.research.google.com/github/HaykTarkhanyan/python_math_ml_course/blob/main/SECTION/NN_file.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
```

**Cell 4 (Markdown) — Song/quote:**
```markdown
> 🎵 [Song or quote reference]
```

**Cell 5 (Markdown) — Video links:**
```markdown
# 📺 Տեdelays

## Տեsakan (Theory)
- [📺 2025](YOUTUBE_LINK)
- [📺 2023](YOUTUBE_LINK)

## Gortsnakan (Practical)
- [📺 Gortsnakan](YOUTUBE_LINK)

## Lusumner (Solutions)
- [📺 Tanyin lusumner](YOUTUBE_LINK)
```

**Cell 6+ (Markdown + Code) — Lesson content:**
```markdown
# 📚 Nyuthy (Material)
```
Then alternate markdown explanations with code cells demonstrating concepts.

**Practical section:**
```markdown
# 🛠️ Gortsnakan (Practical Problems)
```

**Homework section:**
```markdown
# 🏡 Tanyin (Homework)
- [Profound Academy link](https://profound.academy/...)
```

**Fun section:**
```markdown
# 🎲 NN
- ▶️ [Video]()
- 🔗 [Link]()
- 🇦🇲🎶 [Armenian song]()
- 🌐🎶 [International song]()
- 🤌 [Kargin]()
```

**Footer cell:**
```html
<a href="http://s01.flagcounter.com/more/1oO"><img src="https://s01.flagcounter.com/count2/1oO/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter"></a>
```

3. **Content guidelines**:
   - Intuition-first teaching — explain the "why" before the "how"
   - Use real-world examples grounded in data science / ML contexts
   - Include code cells with working Python 3.10+ examples
   - Use type hints and modern Python features (match/case, dataclasses, walrus operator where appropriate)
   - Keep code cells focused — one concept per cell
   - Add markdown explanations between code cells

4. **After creating the file**, remind the user to:
   - Add the file to `_quarto.yml` under the appropriate part
   - Add a background photo from Unsplash (Armenian-themed preferred)
   - Fill in YouTube links and Profound Academy homework links
