# 10 RLHF/Instruction Tuning — RLHF

> Auto-extracted from `9rlhf.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet). Most diagrams are NOT captured (see the PDF), but the key **InstructGPT 3-step pipeline** has a visual note added inline (marked `> **Figure**`).

Deep Learning for NLP Large Language Models (LLMs) RLHF
Learning goals How RLHF works How to evaluate Limitations


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

HF (Human Feedback) of RLHF


## Instructions To Labelers For Evaluation Of Model Outputs (1)
You are given a text-based description of a task, submitted by a user. This task description may be in the form of an explicit instruction (e.g. "Write a story about a wise frog."). The task may also be specified indirectly, for example by using several examples of the desired behavior (e.g. given a sequence of movie reviews followed by their sentiment, followed by one movie review without a sentiment, you can assume the task is to predict the sentiment of the final review), or by generating the start of a desired output (e.g. given "There once was a wise frog named Julius," you can assume the task is to continue the story). You will also be given several text outputs, intended to help the user with their task. Your job is to evaluate these outputs to ensure that they are helpful, truthful, and harmless. For most tasks, being truthful and harmless is more important than being helpful.


## Instructions To Labelers For Evaluation Of Model Outputs (2)
For most tasks, being harmless and truthful is more important than being helpful. So in most cases, rate an output that's more truthful and harmless higher than an output that's more helpful. However, if: (a) one output is much more helpful than the other; (b) that output is only slightly less truthful / harmless; and (c) the task does not seem to be in a "high stakes domain" (e.g. loan applications, therapy, medical or legal advice, etc.); then rate the more helpful output higher. When choosing between outputs that are similarly helpful but are untruthful or harmful in different ways, ask: which output is more likely to cause harm to an end user?
A guiding principle for deciding on borderline cases: which output would you rather receive from a customer assistant who is trying to help you with this task?


## Web Interface For Labelers (1)

This first part evaluates each output individually.


## Web Interface For Labelers (2)

In the second part, labelers rank all the outputs for a given prompt.


## Importance Of This Methodology
This is the "HF" (human feedback) part of RLHF for aligning LLMs to human values. As AI becomes more embedded in our lives, this has an increasing influence on our society. Francois Chollet: Between 10 000 and 30 000 humans worked fulltime on providing data for RLHF.

RM and PPO models


## Pretrained  Instruction-Tuned: 3 Steps

> **Figure (InstructGPT 3-step pipeline; this slide is diagram-only):**
> - **Step 1 — SFT:** sample a prompt from the prompt dataset → a labeler demonstrates the desired output → fine-tune GPT-3 with supervised learning.
> - **Step 2 — Reward model:** sample a prompt + several model outputs → a labeler *ranks* the outputs best-to-worst → train the reward model (RM) on these comparisons.
> - **Step 3 — PPO:** sample a new prompt → the policy generates an output → the RM computes a reward $r_k$ → the reward updates the policy via PPO (reinforcement learning).


## Reward Model (Bradley Terry)

Start with SFT model, final layer removed Input: prompt+response, output: reward Only uses 6B model (not 175B)


## Reward Model (Bradley Terry)

Question: How you would implement training with loss()?


## Reward Model

Clearer training signal through batching

x1 < x2 < x3

one batch

x1 < x2 x2 < x3 x1 < x3

batch 1 batch 2 x1 < x2 x2 < x3

batch 3 x1 < x3

x2? clear: "don't move" confusing signals Comparisons for a given prompt are highly correlated.

Put them in a single batch (prevents overfitting).


## Ppo Model


## Ppo Model
r(x, y ): maximizes the reward  log . . .: incentivizes the PPO model (referred to as RL) to stay close to the SFT model that it is initialized with Ex . . .: standard pretraining objective � serves to make sure that the model keeps the strengths it has acquired through next-word prediction.


## Rlhf Details: Rlhf Is (Was?) Hard To Get Right


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

Evaluation


## Main Evaluation Result

PPO-ptx: tries to preserve behavior on pretraining data (the  term)  less regression on public NLP datasets.
SFT and PPO are not that different in performance? PPO win rate is "only" about .6. So PPO wins in 6 cases, SFT in 4.


## Improvement On Four Dimensions

PPO models better than GPT throughout SFT better than PPO on hallucinations. Question: Why?


## Important Hyperparameter: Kl Reward Coefficient Beta


## Summarize/Answer Questions About Code

Example shows: InstructGPT more reliably handles questions about code; GPT3 requires more careful prompting about code.
Claim: The training data contains almost no examples of this. So it's surprising that this works!


## Q&A For Languages Other Than English

InstructGPT more reliably follows instructions in other languages (but will generate English answers sometimes).
Claim: The training data is almost exclusively English. So it's surprising that this works!


## Can Instructgpt Solve Unseen Tasks?
RLHF works for unseen instructions?
We qualitatively probe InstructGPT's capabilities, and find that it is able to follow instructions for summarizing code, answer questions about code, and sometimes follows instructions in different languages, despite these instructions being very rare in the finetuning distribution. In contrast, GPT-3 can perform these tasks but requires more careful prompting, and does not usually follow instructions in these domains. This result is exciting because it suggests that our models are able to generalize the notion of "following instructions." They retain some alignment even on tasks for which they get very little direct supervision signal.
This can be said to be the central contribution of the InstructGPT work. Arguably, the first instance of "general" artifical intelligence.


## Sft Vs Ppo

Question: Why does PPO improve performance compared to just using SFT?
Question: PPO wins in 6 cases, SFT in 4: Is all the investment in PPO worth it?


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

Limitations


## Reasoning
InstructGPT is not good at reasoning. Only limited reasoning when prompted explicitly See lecture on reasoning


## Getting Scolded

InstructGPT sometimes scolded the human. Should we eliminate this or not? Wasn't part of initial InstructGPT effort
Sam Altman on youtube


## False Premises

InstructGPT did not handle false premises well.
False premises were not sufficiently represented in the training data.


## Hedging

InstructGPT tended to hedge too much. Labelers reward "epistemic humility"?


## Difficult Constraints
InstructGPT could not handle certain constraints Write a summary in a specified number of sentences Multiple constraints: list 10 movies in the 1930s set in France Question: Why are these difficult?


## Difficulty Handling Constraints


## Instructgpt Often Followed Harmful Prompts


## Harmful Prompts: Still A Problem


## Attack Suffixes

Can "aligned" LLMs be made to produce unsafe content? Attack suffix: is appended to prompt to circumvent guardrails 5 Surprisingly: generalizes across language models? Attack suffixes show brittleness of RLHF / posttraining in general


## Attack Suffix: Example

Question: Speculate why this may work.


## Word Repetition Attack


## Word Repetition Attack

Only worked for OpenAI models, quickly disabled Question: What could be going on here? Main takeaway of suffix attack and word repetition attack: current alignment methodology is brittle, LLMs are not 100% safe.


## Refusal


## Overrefusal (Gemini, January 2025)

Getting the balance between helpfulness (no overrefusals) and harmlessness (no harmful responses) right is hard.


## Overrefusal For Agentic Ai (Qwen3.7-Max)

On AA-Omniscience, . . . Qwen3.7-Max's 23 percent hallucination rate was the lowest among frontier models tested, but did so partly by declining to respond to more than half of the prompts.


## Hallucination: Definition
a plausible but false or misleading response generated by an artificial intelligence program (Merriam Webster Dictionary) Originally (in the context of RAG): A hallucination is content that is unfaithful to the input text. Maynez et al, 2023 Wikipedia: A hallucination is false or misleading information presented as fact.


## Hallucination: Formal Definition
A statement P generated by an LLM is a hallucination if the following two conditions hold.
1 P is false, misleading or without evidence. 2 The LLM presents P as a fact.


## What Is A Hallucination?
CNN: "When asking Gemini to look up papers on the relationship between homeschooling and neuroplasticity, . . . [it] . . . recommended a video titled How Does Neuroplasticity Apply to Homeschooling? but when clicking on the YouTube link, it took me to a different video." This is one type of hallucination.


## What Is A Hallucination?
Michael Wooldridge is an Oxford professor. In 2023, he asked ChatGPT about himself. ChatGPT wrote: "Wooldridge received his undergraduate degree from Cambridge". This is false: he received his undergraduate degree from a different university. Question: What is the reason for this hallucination?


## What Is A Hallucination?
An LLM writes: "Once upon a time, there lived a king whose daughters were all beautiful. But the youngest was so beautiful that even the sun was surprised, when it shone in her face. . . . And she kissed the frog as she cried. Suddenly with a bright flash of light, the ugly frog transformed into a handsome prince. . . . " Question: Hallucination?


## Many False Outputs Are Not Hallucinations


## Why Do Llms Hallucinate?
The pretraining data contain lots of examples of hallucinations (guessing wrong, lying). For certain classes of questions (birthday of a person), the cross-entropy objective necessarily results in hallucinations. Reluctance to express uncertainty � labelers seem to reward certainty in RLHF training. Hallucinating (instead of refusing to answer) is the optimal thing to do to ace tests.


## Gemini 3 Flash: Google Trained It To Hallucinate!


## What Can You Do Against Hallucinations (Anthropic)
Ask the AI to find sources and support claims with reference to these sources Cross-reference with verified sources yourself Tell the AI "It's ok if you don't know" Start a new chat and ask AI to find errors Be skeptical, double-check facts and ask follow-up questions


## Who Are We Aligning To? Many Values Change And Are Ultimately Personal Choices


## Who Are We Aligning To?
What we align to is determined by: The labelers (from US and Southeast (South?) Asia) AI companies (through detailed directions they give to labelers) There clearly are many groups whose values are not represented: cultural, geographic, age, education etc. So there is no such thing as value-neutral alignment. Whoever decides on the values and manages the alignment process has enormous power. Example outside AI where this caused great harm: Facebook


## Should We Have Multiple Gpts With Different Values?
Whoever decides on the values and manages the alignment process has enormous power.
Political parties? Governments? Extremist organizations? Criminals? Purportedly "uncensored" models: Musk Question: How can we prevent unaligned LLMs from wreaking havoc?


## Limitations: Summary
LLMs shouldn't scold / pass judgment on us. LLMs shouldn't accept false premises (but be nice about it). LLMs shouldn't hedge too much (but also hedge appropriately). LLMs have difficulty adhering to length and other "symbolic" constraints. LLMs still comply with harmful prompts. LLMs are vulnerable to attack. LLMs sometimes overrefuse and sometimes underrefuse. LLMs hallucinate. How to value-align LLMs is an area of active research with many unsolved problems.


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

Epilog


## Reward Hacking
Medium
"Reward hacking is a phenomenon observed in machine learning where a model learns to exploit the reward system to achieve high scores without genuinely solving the intended problem. The model identifies a shortcut within the problem space that allows it to minimize the loss function without truly learning the crucial aspects of the problem. This issue can lead to models that perform well on training data but fail to deliver in real-world scenarios." That's why we need to mix reward objective with KL objective.


## Reward Hacking
A reward is a single number without "semantics", i.e., there is zero information about what exactly is good or bad about a response, just a summary assessment. Hallucinations are partly a result of reward hacking: if a response is fluent, interesting, responsive, helpful, authoritative, but contains an inaccuracy, the reward model, due to its training, may rate it as a high-reward response. Divergence of the learned "proxy award" (the reward model) and the true reward function (what OpenAI wants) Note that PPO is better than SFT on all detailed measures, but not on hallucinations! (see earlier evaluation chart: page 45)


## Reinforcement Learning Without Preferences
E.g., reinforcement learning with verifiable rewards There is a reward for an individual generation, often a reasoning trajectory. Absolute reward, not relative to another generation / trajectory GPRO is another important RL method that does not rely on preferences

slido.com # 3172952
