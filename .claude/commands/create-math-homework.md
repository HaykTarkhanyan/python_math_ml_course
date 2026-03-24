# Create Math Homework Module

Create a new math homework `.qmd` file for this Quarto-based course.

## Input
- $ARGUMENTS: Topic name, module number, and any specific subtopics or problem areas to cover

## Instructions

1. **Determine the module number and filename**: Follow the pattern `math/NN_topic_description.qmd` (zero-padded two-digit number, snake_case description).

2. **Use this exact file structure**:

```qmd
---
title: "NN Topic Title"
format:
  html:
    css: homework-styles.css
---

<script src="homework-scripts.js"></script>

![image.png](../background_photos/math_NN_PLACEHOLDER.jpg)
[լուսdelays](UNSPLASH_LINK), Հեղinak ART AUTHOR

# 📚 Նyouth

::: {.callout-tip collapse="true"}
## ⚠️ Note
YouTube links in this section were auto-extracted. If you spot a mistake, please let me know!
:::

## Դasakhosyutyun (Lecture)
- [📺 Dasakhosyutyun — TOPIC](YOUTUBE_LINK)
- [🎞️ Slidener](Lectures/LECTURE_PDF)

## Gortsnakan (Practical)
- [📺 Gortsnakan — TOPIC](YOUTUBE_LINK)
- [🛠️🗂️ Gortsnakani PDF-y](Homeworks/hw_NN_topic.pdf)

# 🏡 Tanyin

::: {.callout-note collapse="false"}
1. ❗❗❗ DON'T CHECK THE SOLUTIONS BEFORE TRYING TO DO THE HOMEWORK BY YOURSELF❗❗❗
2. Please don't hesitate to ask questions, never forget about the 🍊karalyok🍊 principle!
3. The harder the problem is, the more 🧀cheeses🧀 it has.
4. Problems with 🎁 are just extra bonuses. It would be good to try to solve them, but also it's not the highest priority task.
5. If the problem involve many boring calculations, feel free to skip them - important part is understanding the concepts.
6. Submit your solutions [here](https://forms.gle/CFEvNqFiTSsDLiFc6) (even if it's unfinished)
:::


## SUBTOPIC SECTION HEADER

### 01 Problem Title {data-difficulty="1"}
[Problem statement with LaTeX math notation]

### 02 Problem Title {data-difficulty="2"}
[Problem statement]

...more problems...


# 🎲 NN (XX) TODO
- ▶️[ToDo]()
- 🔗[Random link]()
- 🇦🇲🎶[ToDo]()
- 🌐🎶[ToDo]()
- 🤌[Կارگin]()


<a href="http://s01.flagcounter.com/more/1oO"><img src="https://s01.flagcounter.com/count2/1oO/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter"></a>
```

3. **Problem design guidelines**:
   - Use `data-difficulty="1"` (🧀 easy), `"2"` (🧀🧀 medium), `"3"` (🧀🧀🧀 hard) on `###` headers
   - Use `.bonus-problem` class for extra credit: `### Problem Title {.bonus-problem}`
   - Ground problems in real-world contexts (ML, medicine, finance, sports, engineering, polling)
   - Structure as scenarios with sub-questions (a/b/c), not abstract exercises
   - Use LaTeX math notation: `$inline$` and `$$display$$`
   - Include ML-flavored context boxes where appropriate:
     ```
     ::: {.callout-tip collapse="true" appearance="minimal"}
     #### Context
     [Real-world ML/data science context for the problem]
     :::
     ```
   - Wrap solutions in profile-conditional blocks:
     ```
     ::: {.content-visible when-profile="solution"}
     ### Solution {.solution-header}
     [Step-by-step solution]
     :::
     ```

4. **Mix of difficulty levels**: Aim for roughly 40% easy, 40% medium, 20% hard problems. Include 1-2 bonus problems.

5. **After creating the file**, remind the user to:
   - Add the file to `_quarto.yml` under the appropriate part/chapter
   - Add a background photo
   - Fill in YouTube and PDF links when available
