# Python, Math & Machine Learning Course


**[Course Website](https://hayktarkhanyan.github.io/python_math_ml_course/)** | **[YouTube](https://www.youtube.com/@MetricAcademy)** | **[Telegram](https://t.me/metric_academy)**

---

## Course Structure

The course is organized into four major blocks, delivered as an interactive [Quarto website](https://quarto.org/docs/websites/).

### 1. Python (18 modules)

Core Python from first principles through advanced OOP and practical projects.

| # | Topic |
|---|-------|
| 01 | Introduction to Python |
| 02 | Conditions |
| 03 | Strings, Range, Lists |
| 04 | Loops |
| 05 | List/String Methods, Comprehensions |
| 06 | Tuples, Sets, Dictionaries |
| 07 | Functions I |
| 08 | Functions II |
| 09 | Files, Packages, Terminal |
| 10 | Git, Conda, PEP 8 |
| 11 | Exception Handling |
| 12 | Streamlit & Recursion |
| 13 | Decorators |
| 14 | Classes |
| 15 | Inheritance & Polymorphism |
| 16 | Encapsulation & Abstraction |
| 17 | Dataclasses, Iterators, Generators, Context Managers |
| 18 | Project: YouTube Translator |

### 2. Python Libraries (18 modules)

Real-world tools, data engineering, APIs, and software practices.

| # | Topic |
|---|-------|
| 01 | OpenAI API & Timestamps |
| 02 | NumPy |
| 03 | Pandas I |
| 04 | Pandas II |
| 05 | Data Analysis Project (Noble People) |
| 06 | Data Visualization |
| 07 | Project: Kargin |
| 08 | Logging & CLIs |
| 09 | Testing & Debugging |
| 10 | Web Scraping & Parallelization |
| 11 | Project: YSU Scraping |
| 12 | SQL |
| 13 | Pydantic |
| 14 | Miscellaneous Libraries |
| 15 | FastAPI |
| 16 | Databases & Supabase |
| 17 | Vibe Coding |
| 18 | Clean Code & Architecture |

### 3. Mathematics (26 modules)

A rigorous yet intuition-first math curriculum, progressing from linear algebra through statistics. Each module is problem-centric with graded difficulty levels.

| # | Area | Topic |
|---|------|-------|
| 00 | Foundations | Sets, Combinatorics, Functions |
| 01 | Linear Algebra | Vectors, Norms, KNN |
| 02 | Linear Algebra | Matrices, Transformations |
| 03 | Linear Algebra | Linear Systems, Eigenvalues, Regression |
| 04 | Calculus | Limits, Continuity, Derivatives |
| 05 | Calculus | Extrema, Convexity, Taylor Series |
| 06 | Calculus | Integrals |
| 07 | Calculus | Multivariate Calculus, Gradient Descent |
| 08 | Optimization | Univariate (Golden Section, Brent's) |
| 09 | Optimization | Prerequisites & Gradient Descent |
| 10 | Optimization | Momentum & First-Order Methods |
| 11 | Optimization | Second-Order Methods |
| 12 | Optimization | Derivative-Free Methods |
| 13 | Optimization | Evolutionary Algorithms |
| 14 | Optimization | Bayesian Optimization |
| 15 | Optimization | Multi-Criteria Optimization |
| 16 | Probability | Basics, Bayes' Rule, Monty Hall |
| 17 | Probability | Expectation, Variance, Inequalities |
| 18 | Probability | Covariance & Correlation |
| 19 | Probability | Distributions (Discrete & Continuous) |
| 20 | Probability | Convergence, LLN, CLT |
| 21 | Statistics | Fundamentals |
| 22 | Statistics | Estimators |
| 23 | Statistics | MLE & MAP |
| 24 | Statistics | Confidence Intervals |
| 25 | Statistics | Hypothesis Testing |

Accompanying **Beamer slide decks** (compiled with LaTeX/TikZ) are available in `math/Lectures/` for both the optimization and statistics series.

### 4. Machine Learning (6 chapters)

Not started yet

---

## Repository Layout

```
.
├── python/              # Python modules (Jupyter notebooks)
├── python_libs/         # Libraries & tools modules (Jupyter notebooks)
├── math/                # Math modules (Quarto .qmd files)
│   ├── Lectures/        # Beamer slide decks (.tex → .pdf)
│   │   ├── stat/        # Statistics lecture series
│   │   └── optim/       # Optimization lecture series
│   ├── Homeworks/       # Homework assignments (.pdf)
│   └── assets/          # Images, data files, helper notebooks
├── ml/                  # Machine Learning chapters
│   ├── Chapter 1–6/     # Lecture notes, code, homeworks per chapter
│   └── Datasets/        # Shared ML datasets
├── misc/                # Miscellaneous guides (Google Colab, etc.)
├── docs/                # Rendered Quarto book (GitHub Pages)
├── _quarto.yml          # Quarto book configuration
└── index.qmd            # Course landing page
```

---

## Getting Started

### Viewing the course

The easiest way is to visit the **[course website](https://hayktarkhanyan.github.io/python_math_ml_course/)**, which renders all notebooks and Quarto documents as a searchable book with dark mode support.

### Running locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/HaykTarkhanyan/python_math_ml_course.git
   cd python_math_ml_course
   ```

2. **Install dependencies**
   - [Python 3.10+](https://www.python.org/)
   - [Quarto](https://quarto.org/docs/get-started/) (for rendering `.qmd` files)
   - Jupyter / JupyterLab (for `.ipynb` notebooks)
   - TeX Live (for compiling Beamer slides)

3. **Render the Quarto book**
   ```bash
   quarto render
   ```
   The rendered site will be in the `docs/` directory.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Quarto](https://quarto.org/) | Book framework — renders `.qmd` and `.ipynb` into a unified website |
| Jupyter Notebooks | Python and Libraries modules |
| LaTeX / Beamer | Lecture slide decks with TikZ diagrams |
| GitHub Pages | Hosting the course website |

---

*Last updated: March 2026*
