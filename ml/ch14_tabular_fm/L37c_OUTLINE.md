# L37c — Explaining TabPFN

Companion to L37, centered on Rundel, Kobialka, von Crailsheim, Feurer, Nagler & Rügamer (2024), *Interpretable Machine Learning for TabPFN* (arXiv:2403.10923). The paper studies the original, small-context TabPFN. Its method ideas need revalidation on later checkpoints.

1. Cold open: explain a prediction, a feature's usefulness, or the training rows; these are different questions.
2. Outline and a running loan-risk table.
3. Feature effects: ICE, partial dependence and ALE; batch the altered queries in one forward pass.
4. Feature importance: leave-one-covariate-out on a held-out set, with worked numbers.
5. Local Shapley values: feature coalitions and a two-feature calculation; omit a feature from both training context and query to evaluate the restricted learner exactly for that coalition.
6. Explain the paper's "exact retraining" term: exact *subset evaluation*, while sampled coalitions still approximate the Shapley sum.
7. Data valuation: leave-one-out and Data Shapley; selecting a fixed-size context from a larger pool, with paper results and computational cost.
8. Other toolbox methods: global Shapley importance, sensitivity gradients, counterfactuals, uncertainty and decision analysis.
9. Limits: predictive, not causal; correlated features; held-out data; paper's original 1,024-row context.
10. Recap and pointers to the original paper and code.

No model execution. Worked numbers are illustrative and labeled as such.
