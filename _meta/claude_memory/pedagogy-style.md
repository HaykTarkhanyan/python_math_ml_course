---
name: Pedagogical Style & Conventions
description: Teaching philosophy, notation conventions, and content patterns used consistently across the course
type: project
---

## Teaching Philosophy
- **Intuition-first, computation-secondary** — no heavy proofs, build mental models
- **Problem-centric** — scenarios with sub-questions (a/b/c), not abstract exercises
- **Real-world grounding** — examples from medicine, finance, ML, sports, science, engineering, polling
- **Bilingual** — Armenian (Հայերեն) + English throughout all materials
- **Armenian cultural references** — locations, songs, history, "karalyok principle" (feedback loops)
- **Multi-modal** — notebooks, videos, slides, problem platforms, mini-projects

## Difficulty System (.qmd homework files)
- `data-difficulty="1"` → 🧀 (Easy)
- `data-difficulty="2"` → 🧀🧀 (Medium)
- `data-difficulty="3"` → 🧀🧀🧀 (Hard)
- `.bonus-problem` → 🎁 (Extra credit)
- Implemented via homework-scripts.js + homework-styles.css

## Stat Lecture Conventions
- Beamer color coding: popblue=theory, sampred=data/samples, paramgreen=parameters/population, warnred=warnings, orange1=highlights, violet1=special concepts
- θ for parameters, θ̂ for estimates, ℓ(θ) for log-likelihood, s(θ) for score, I(θ) for Fisher info
- Every lecture opens with "Previously, on..." recap (5 styled boxes)
- Every lecture ends with Practical slide + "Questions?" closing
- Subtitles use centered dots (·) to join key concepts

## Stat Lecture Arc
1. L1-L2: "What does data look like?" (foundations, visualization)
2. L3-L8: "How do we estimate and measure uncertainty?" (theory + practice)
3. L9-L11: "Is an effect real or noise?" (testing + experimentation)
4. L12-L13: "How do variables relate?" (regression, GLMs)
5. L14: "Did X cause Y?" (causal inference)
6. L15: Recap of all 14 lectures
7. L16: "How to Lie with Statistics" — critical thinking

## Homework Platform
Profound.academy — bilingual Armenian/English problem sets, used across Python and Math courses
