# Create Beamer Lecture Slides

Create a new LaTeX Beamer presentation for this course's lecture series.

## Input
- $ARGUMENTS: Lecture number, topic, and subtopics to cover (e.g., "07 Sampling Distributions and CLT")

## Instructions

1. **Use the standard Beamer template** that matches the existing stat/optim lecture style:

```latex
\documentclass[aspectratio=169]{beamer}

% Minimal theme
\usetheme{default}
\usecolortheme{dove}

% Remove navigation symbols
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{%
  \hfill{\large\insertframenumber\,/\,\inserttotalframenumber}\hspace{0.8em}\vspace{0.5em}%
}

% Colors
\definecolor{popblue}{RGB}{52, 101, 164}
\definecolor{sampred}{RGB}{204, 0, 0}
\definecolor{paramgreen}{RGB}{0, 140, 70}
\definecolor{lightbg}{RGB}{245, 245, 250}
\definecolor{warnred}{RGB}{180, 40, 40}
\definecolor{orange1}{RGB}{220, 120, 0}
\definecolor{violet1}{RGB}{120, 50, 160}

\setbeamercolor{frametitle}{fg=popblue}
\setbeamercolor{title}{fg=popblue}

% Packages
\usepackage{pgfplots}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning, calc, decorations.pathreplacing, patterns}
\pgfplotsset{compat=1.18}
\usepackage{amsmath, amssymb}
\usepackage{fontenc}
\usepackage{colortbl}

\title{Lecture N: Title}
\subtitle{Concept A $\cdot$ Concept B $\cdot$ Concept C}
\date{}

\begin{document}
...
\end{document}
```

2. **Color coding conventions** (semantic colors for stat lectures):
   - `popblue` — theory, definitions, frameworks
   - `sampred` — data, samples, observations
   - `paramgreen` — parameters, population quantities
   - `warnred` — warnings, common mistakes
   - `orange1` — highlights, key results
   - `violet1` — special concepts, advanced ideas

3. **Lecture structure**:
   - Title slide
   - "Previously, on..." recap frame (5 styled boxes summarizing prior lecture)
   - Motivating question/scenario frame
   - Content sections with `\section{}` separators
   - All diagrams must be TikZ/pgfplots code (no external images)
   - Practical application slide near the end
   - "Questions?" closing frame

4. **Frame design guidelines**:
   - Use centered TikZ diagrams for visual explanations
   - Use `itemize` with colored text for key definitions
   - Use `\textcolor{popblue}{\textbf{term}}` for defined terms
   - Use `\textcolor{paramgreen}{\theta}` for parameters, `\textcolor{sampred}{\hat{\theta}}` for estimates
   - Subtitles use centered dots ($\cdot$) to join key concepts
   - Keep frames focused — one concept per frame
   - Aim for 25-45 frames per lecture

5. **Notation conventions**:
   - θ for parameters, θ̂ for estimates
   - ℓ(θ) for log-likelihood, s(θ) for score function
   - I(θ) for Fisher information
   - X₁, ..., Xₙ for random samples
   - x₁, ..., xₙ for observed values

6. **File placement**:
   - Stat lectures: `math/Lectures/stat/NN_stat.tex`
   - Optim lectures: `math/Lectures/optim/NN_topic.tex`
   - Other lectures: `math/Lectures/LNN_Topic_Name.tex` (with corresponding PDF)

7. **After creating**, remind the user to compile with `pdflatex` (TeX Live 2025).
