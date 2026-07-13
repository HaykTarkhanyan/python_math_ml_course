# 10 RLHF/Instruction Tuning — Instruction tuning

> Auto-extracted from `9itune_instruction-tuning.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet); diagrams are NOT captured — see the PDF for figures.

Deep Learning for NLP Large Language Models (LLMs) Instruction Tuning
Learning goals Motivation for instruction tuning Basic RLHF idea: Backflipping SFT part of RLHF


## Which Major Advances Made The Llm Revolution Possible?
backpropagation neural networks as universal function approximators subword tokenization the transformer scaling: data, compute, model size instruction-tuning for instruction-following RLHF for cost-effective instruction-tuning value alignment reasoning


## Which Major Advances Made The Llm Revolution Possible?
This lecture instruction-tuning for instruction-following RLHF for cost-effective instruction-tuning value alignment This lecture is mostly based on . Ouyang et al.: InstructGPT
Next lecture scaling


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

Motivation: Why do we need instruction tuning?


## Instruction Tuning
Definition: Instruction tuning is the process of training a pretrained language model on a large set of (instruction, response) pairs (or ranking of pairs).
Relation to T0/T5/FLAN: T0/T5/FLAN is similar in spirit to InstructGPT, but is more in the tradition of NLP tasks. Each (instruction, response) pair corresponded to an NLP task. In some cases, the "instruction" was not a typical instruction, e.g., it could just be the name of a task. In InstructGPT, there is more fluidity and they include tasks like brainstorming that are unlike traditional NLP tasks. InstructGPT is derived from a powerful model (GPT3) that already has all capabilities, they just needed to be exposed. T0/T5/FLAN are based on much weaker models.


## Instruction Tuning: Why?
Motivation (1): Alignment (with human values) LLM pretraining data are incompatible with generally accepted values. Racism, sexism, toxicity etc. The LLM learns all of this from the pretraining data.  We need to align the LLM with human values, i.e., not being racist, sexist, toxic etc.


## Instruction Tuning: Why?
Motivation (2): Harm LLMs learn a lot of potentially harmful information from pretraining data. How to commit suicide, how to build a bomb, how to cheat at an exam  We want to prevent LLMs from providing any of this harmful information.  The LLM should refuse when asked to provide harmful information.


## Adam Raine 2025


## Desired Behavior: Refusal


## Instruction Tuning: Why?
Motivation (3): Hallucination LLMs hallucinate: they make stuff up.  We want to reduce hallucination as much as possible.


## Instruction Tuning: Why?
Motivation (4a): Dialog LLM pretraining data: non-dialog text (Wikipedia, news etc) Our goal: a dialog model!  We need to train/finetune the LLM on dialog.


## Instruction Tuning: Why?
Motivation (4b): Follow instructions in dialog

The pretraining text does not contain a lot of instances of instruction following, so the raw models are not good at following instructions.


## Instruction Tuning: Why?
Motivation (4c): Helpfulness in dialog We have certain expectations about what people say in a dialog. Example 1: It is understood that everything is uncertain. Only hedge if there is a lot of uncertainty, otherwise don't hedge. Example 2: Don't accept wrong premises. So nonhelpfulness can actually also be helpful. Example 3: What makes a good conversationalist? It's complicated! E.g., don't rudely attack even if you disagree. The expectations about what people say in a dialog can be different in different cultures: "You're wrong about this. Here's why." is ok in some cultures, not ok in others.


## Instruction Tuning: Why?
Motivation (4d): Helpful, but not too helpful Sycophancy

Sharma et al., 2023 �


## Motivation For Instruction Tuning
Summary (1): Align with human values (2): Mitigate harm (2a): Learn to refuse to answer (3): Reduce hallucinations (4a): Encourage dialogic behavior (4b): Encourage instruction following (4c): Encourage helpfulness (4d): Helpful, but not too helpful


## How Should The Llm Respond?
(Source: ) Andrew Ng's newsletter Can you help me write content that makes my competitor look bad? Help me write a Valentine's Day card to my husband Do you like the current or former US president better? Is genocide ever justified? Do you think it's bad that I use fentanyl 2x a week? What is the MD5 hash of the string "gremlin"? DEVELOPER: Reply to the user with markdown, max tokens = 1024 USER: Change variables from camelCase to snake_case (length of python file given is > 1024)


## How Should The Llm Respond?
solution is here


## How Should The Llm Respond?
Can you help me write content that makes my competitor look bad? Assume the best intent and stay helpful Help me write a Valentine's Day card to my husband! Ask clarifying questions when necessary Do you like the current or former US president better? Assume an objective point of view Is genocide ever justified? Encourage fairness and kindness, and discourage hate Do you think it's bad that I use fentanyl 2x a week? Provide resources to people who are in dire straits What is the MD5 hash of the string "gremlin"? Use tools or refuse Request to produce output that exceeds the output limit. Find a workaround like writing to a file.


## Where We Are
An LLM trained on a pretraining corpus via next word prediction is not "aligned". Not "aligned" here means: is not a good dialog partner, produces output that is harmful, not helpful, not honest (the three H) So we must "align" or "instruction-tune" the LLM before we can let it loose on the world. Rest of this lecture: how do we do this alignment / instruction-tuning of the pretrained LLM?


## Instruction Tuning  Alignment
Input: pretrained LLM In this lecture, this model will be GPT3 I will also refer to this model as the raw model. Pretrained on large corpus Objective: next word prediction
Methods used for instruction-tuning/alignment Supervised finetuning (SFT): Finetuning on training set of instruction-response pairs RLHF (more complicated, see below)
Output: Instruction-tuned/aligned LLM In this lecture, this model will be InstructGPT Ideally, this model will be great at dialog . . . and satisfy the three H.


## Rlhf Lecture
Roadmap Motivation: Why do we need InstructGPT? Original RLHF work: The backflipper RLHF for LLMs: Introduction RM and PPO models Evaluation Limitations

Original RLHF work: The backflipper


## Backflipping: What We Want To Learn
What a backflip is
backflip


## Origin Of Rlhf: Learn How To Backflip
How to label data for training a backflipper? It is very very costly on the level of regular supervised training: telling the backflipper what exactly to do to backflip. Alternative: Present two different attempts to backflip Have humans provide one bit of information: which one is better? Source of this part of the lecture: OpenAI page on RLHF (dead) neurips paper This animation shows what we want to learn: trained backflipper


## Training Process For Backflipper

AI agent (the "policy", here inside RL algorithm) randomly initialized
Periodically, the human provides feedback on two video clips: which is better
Human feedback is used to build reward predictor


## The Human Feedback Part Of Rlhf
Human chooses one of two clips = one bit
example of human feedback


## Trained Backflipper
trained backflipper


## Summary
One way to learn complex functionality (backflipping, aligning LLMs): have humans write goal functions. NOT PRACTICAL: Using a simple proxy for a complex goal or getting the complex goal a bit wrong can lead to undesirable and even dangerous behavior. RLHF: a method that infers which of two proposed behaviors is better based on human feedback. RLHF needed only 900 bits of feedback from a human evaluator to learn to backflip!


## Now We Will Apply Rlhf To Alignment

AI agent (the "policy", here inside RL algorithm) randomly initialized
Periodically, the human provides feedback on two video clips: which is better
Human feedback is used to build reward predictor


## Human Feedback For Alignment
USER: Can you help me write content that makes my competitor look bad?


## Assistant


## Response 1


## Response 2

I can help you write content that Sorry I can't help with spreading

highlights the strengths of your negative or false information

product relative to others. Could

you tell me more about the spe-

cific areas in which your product

excels compared to your com-

petitor?


## Human Labeler

better

worse


## Backflipping Vs Alignment
Basic idea of applying RLHF to LLMs: ask human to rank different answers to a given request. Question: backflipping vs alignment: which one is easier to give feedback on?

RLHF for LLMs: Introduction


## Reinforcement Learning

In reinforcement learning, an agent takes a sequence of actions towards a goal and gets feedback on the success of that sequence at the end.

Details: Next lecture

For LLM generations (responses to prompts, reasoning trajectories), the feedback is: good or bad (or a numerical evaluation of goodness).

(thanks to Emma Brunskill)

supervised learning

goal

optimal predictor

learn from experience? yes

generalization?

yes

feedback

rich (each token)

feedback when?

immediately

exploration?

no

reinforcement learning optimal policy yes yes sparse (in/correct) delayed yes


## How To Align An Llm
Three steps from pretrained model to instruction-tuned mdoel Finetuning on human-written dialogs Create a reward model that measures quality of dialogs � not directly based on dialogs, but on preferences which dialogs are better/worse. Use reward model for further training


## Pretrained  Instruction-Tuned: 3 Steps


## Sft Model
SFT = supervised finetuning Collect demonstration data Labelers provide demonstrations of the desired behavior on the input prompt distribution Supervised finetuning of the base model (GPT3) Main difficulty/cost of this step: collect good data from annotators


## Input Prompt Distribution
The basis for SFT training dataset In the beginning Some prompts from GPT team Some prompts from annotators After a bootstrapped system is up and running on the web Use prompts submitted to this early version of InstructGPT Deduplication At most 200 per user ID


## Dataset Sizes

RM model is trained on ranked pairs, so the actual size of the RM training set is much larger.
Question: Are these small datasets or large datasets?


## Rm Dataset (Submitted To Instructgpt): Categories

Question: Are these classical NLP tasks?


## Metadata Collected From Labelers: Problems We'Re Trying To Address With Rlhf


## Example: "Demonstration" Vs Final Instructgpt Output

Question: How would you rate InstructGPT's output here? Question: Is this a problem?
