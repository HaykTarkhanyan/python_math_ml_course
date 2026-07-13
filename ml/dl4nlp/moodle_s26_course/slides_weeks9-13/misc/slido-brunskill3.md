# RLHF review quiz (slido) — with instructor's handwritten answers

> Visually transcribed from `slido-brunskill3.pdf` (2 pages, image-only — no extractable text).
> A True/False slido poll on RLHF; the answers shown are **handwritten by the instructor**
> (Schütze) on the exported slides, so they are authoritative. `DEPENDS` = written in as a
> correction/nuance. Poll bars all show 0% (blank export).

## Page 1

| # | Claim | Answer |
|---|---|---|
| 1 | After effective RLHF training, an LLM should not give information about how to cheat on an exam. | **TRUE** |
| 2 | After effective RLHF training, an LLM should be able to follow constraints like "give an answer in 15 words". | **FALSE** |
| 3 | After effective RLHF training, an LLM will in general precisely state what it assumes when giving an answer. | FALSE → **DEPENDS** |
| 4 | After effective RLHF training, an LLM will educate the user if their question is based on an incorrect assumption. | FALSE → **DEPENDS** |
| 5 | The improvement of RLHF over SFT is about as large as the improvement of SFT over the raw model. | **TRUE** |
| 6 | After effective RLHF training, an LLM will understand python code better. | **FALSE** |
| 7 | After effective RLHF training, an LLM will hallucinate less compared to the SFT model. | **DEPENDS** |

## Page 2

| # | Claim | Answer |
|---|---|---|
| 1 | The reward model takes a prompt as input and returns a number that tells us how well written the prompt is. (*"prompt" circled*) | **FALSE** |
| 2 | In RLHF, we train the LLM on input-output pairs. | **TRUE** |
| 3 | Supervised instruction finetuning is expensive because labelers demonstrate desired output behavior. | **TRUE** |
| 4 | A ranked list of 5 outputs gives rise to 10 output pairs for training the reward model. | **TRUE** |
| 5 | The training objective for the PPO model has three terms: the reward term, one KL-divergence term that ensures we do not diverge too much from the pretraining objective, and one log-likelihood term for fitting the instruction-tuning training set. | **FALSE** |
| 6 | The SFT model, the reward model and the PPO model are all initialized with a "raw" language model trained with the next-word-prediction objective on the pretraining corpus. | **TRUE** |
| 7 | In RLHF training, an LLM learns to give answers in several different languages (not just English) because these languages occur in the training set. | **FALSE** |
