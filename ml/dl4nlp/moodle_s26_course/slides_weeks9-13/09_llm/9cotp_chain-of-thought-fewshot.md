# 09 LLMs — Chain-of-thought few-shot prompting

> Auto-extracted from `9cotp_chain-of-thought-fewshot.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet); diagrams are NOT captured — see the PDF for figures.

Deep Learning for NLP
Large Language Models (LLMs) Chain-of-thought few-shot prompting
Learning goals illustrate chain-of-thought few-shot prompting and point out the benefits it brings to LMs illustrate tree-of-thought and point out the benefits it brings to LMs

Chain-of-thought few-shot prompting MOTIVATION
How to boost the reasoning capabilities of LMs? Wei et al., 2021 Use formal approaches, e.g., logic, symbolic reasoning Example: BeliefBank Difficult to train and deploy, not widely used Standard few-shot learning via prompting works for many tasks Still, it works poorly for many tasks that require reasoning COT-FS-P A form of few-shot prompting Each "training example" has the form: <input, chain of thought, output> chain of thought: series of reasoning steps that lead to the final answer application: reasoning tasks: complex, commonsense, symbolic etc


## Neurosymbolic Approach (Currently Infrequently Used)


## Lms Not Good At Reasoning Tasks (This Is Why We Need Cot-Fs-P.)

Question: What is the problem here?

Chain-of-thought few-shot prompting PARADIGM

Source: Wei et al., 2022

chain-of-thought few-shot prompting PARADIGM

BENEFITS OF chain-of-thought few-shot prompting
Decompose multi-step problems and thus allocate more compute to problems requiring more reasoning steps. This helps LMs solve problems they otherwise would not be able to solve. By describing the reasoning, interpretability is increased. It provides the possibility to observe where reasoning went wrong COT-FS-P is closer to how humans solve tasks using language

chain-of-thought few-shot prompting
Following slides: Examples of <input, chain of thought, output> triples for commonsense and symbolic reasoning
Source: Wei et al., 2022
The few shots (training examples) are omitted from these examples to save space, but this is chain-of-thought few-shot prompting, that is, the model is prompted with examples of chain of thought reasoning.


## Examples


## Examples


## Examples


## Examples


## Examples


## Examples


## Cot-Fs-P Improves Arithmetic


## Cot-Fs-P Improves Arithmetic
SVAMP: math word problems with varying structures; MAWPS: repository unifying math problems from different sources;
Source: Wei et al., 2022


## Cot-Fs-P Improves Commonsense

Source: Wei et al., 2022


## Tree-Of-Thought: Motivation
The token-level and left-to-right decisions of the autoregressive mechanism pose a limitation for:
Tasks where initial decisions play a pivotal role Tasks requiring exploration Strategy to solve those: Maintain and explore diverse alternatives instead of just picking one Evaluate current status and look ahead or backtrack to make global decisions


## Tree-Of-Thought: Prompting Paradigm
Rectangle box = thought = a coherent language sequence serving as an intermediate step in problem solving.

Yao et al., 2023


## Tree-Of-Thought For Creative Writing
A step of deliberate search in a Creative Writing task. Given the input, the LM samples five different plans, and then votes five times to decide which plan is best.

Yao et al., 2023


## Tree-Of-Thought For Creative Writing (2)

Chain-of-thought few-shot prompting: SUMMARY
Decompose complex problems into a sequence of reasoning steps By describing the reasoning, interpretability is increased. It provides the possibility to observe where reasoning went wrong It is closer to how humans solve tasks using language Language models, if given a well designed chain-of-thought prompt, can solve problems they otherwise would not be able to solve. Question: What could go wrong?

Chain-of-thought few-shot prompting: ERROR BREAKDOWN
8% calculator error 16% symbol mapping error 22% one missing step error rest: semantic issues, incoherent COT-FS-P Source: Stanford CS25: Beyond LLMs: Agents, Emergent Abilities, Intermediate-Guided Reasoning

Chain-of-thought few-shot prompting
Question: Do top-of-the-line LLMs use chain-of-thought few-shot prompting?


## Chain-Of-Thought: Terminology
Shot = "training example" few-shot prompting = few-shot learning The prompt "think step by step" by itself (without shots) is not chain-of-thought few-shot prompting. Chain-of-thought few-shot prompting is defined as including shots. Chain-of-thought is currently used as a general term to refer to the idea of LMs using explicit reasoning steps to arrive at an answer. So the current usage of chain-of-thought is more general than chain-of-thought few-shot prompting.


## Generator-Verifier Gap (Noam Brown)
For many important problems, it is much easier to verify a solution than generating one. Chain-of-thought is expected to help for such problems with a generator-verifier gap. Problems with generator-verifier gap: Sudoku, doing math, programming Problems with less of a generator-verifier gap: knowledge questions (what is the capital of bhutan?), simple pattern matching (which language is this?)


## Slido
1313837 https://app.sli.do/event/dinLdZRBHw2fXo5R31C3Nt 1435969 https://app.sli.do/event/kWQDLpHa14256yiwxyCDr5 4039244 https://app.sli.do/event/ef5nQS8XmbhWAQDVk9CYQs 42248917 https://app.sli.do/event/5YwPZfoEFAibQ4DbFTzfj2
