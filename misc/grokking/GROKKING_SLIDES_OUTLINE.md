# Grokking and Mechanistic Interpretability: Slide Plan

Status: outline only. Draft and approve before building the Beamer deck.

## Deck goal

Show how a small transformer can first memorize modular addition, later generalize, and expose a readable sine/cosine circuit. Then connect this toy setting to mechanistic interpretability in modern language models.

## Proposed flow

1. **Cold open: generalization that arrives late**
   - Predict-first question: if training accuracy is already 100%, can test accuracy still change dramatically?
   - One-line definition of grokking.

2. **The problem: modular addition from incomplete tables**
   - By-hand examples of `a + b mod P`.
   - A partially hidden addition table and its train/test split.
   - Why tokens are intentionally structureless to the model.

3. **The tiny transformer**
   - Architecture diagram: `a b =` -> embeddings and positions -> one attention layer -> ReLU MLP -> logits for `c`.
   - What is trained, what is predicted, and what is held out.

4. **Training dynamics: memorize, then grok**
   - Train/test accuracy and loss curves.
   - Grokking diagram with the memorization, circuit-formation, and cleanup phases.
   - Clarify full-batch epoch versus minibatch optimizer step.

5. **What do we inspect inside the model?**
   - Embeddings, attention patterns, MLP-neuron activations, neuron-to-logit map, and output logits.
   - Fourier views and targeted ablations as the evidence standard.

6. **Sine and cosine features**
   - Fourier spectrum of number embeddings.
   - Individual embedding and neuron activation surfaces with sine/cosine fits.
   - 2D Fourier spectrum over all `(a, b)` inputs.

7. **The clock picture of modular arithmetic**
   - Numbers as points on a clock/circle.
   - Adding angles corresponds to adding residues modulo `P`.
   - A worked small-`P` clock example.

8. **The trigonometric mechanism**
   - Step-by-step derivation of `cos(x+y) = cos(x)cos(y) - sin(x)sin(y)`.
   - Explain how attention carries periodic features and the MLP forms products.
   - Readout rotates by candidate output `c` to score `cos(x+y-c)`.

9. **Mechanistic-interpretability evidence**
   - Fourier-logit and neuron-logit plots.
   - Restricted and excluded loss.
   - Attention/MLP and key-frequency ablations.
   - What counts as a circuit-level explanation, and what does not.

10. **From toy circuits to Claude Haiku**
    - Anthropic's line-break prediction example.
    - A concise comparison: same interpretability questions, much larger model.

11. **Sparse autoencoders (SAEs)**
    - Diagram: dense model activations -> encoder -> sparse latent features -> decoder/reconstruction.
    - Why SAEs are used to find interpretable features.

12. **Superposition: many responsibilities in high-dimensional space**
   - Diagram of many features represented by fewer neuron dimensions.
   - A neuron/dimension can participate in several features rather than owning one clean concept.
   - High-dimensional geometry makes this possible: for a fixed tolerance around 90 degrees, the number of directions with small pairwise overlap grows exponentially with dimension. In practice, many feature directions can sit around 89 to 91 degrees apart, so their interference is often small rather than zero.
   - The cost is polysemanticity: different contexts activate overlapping directions, making individual neurons hard to interpret.
   - SAEs learn an overcomplete sparse dictionary intended to separate these mixed responsibilities into feature directions.

13. **Recap and forward pointer**
    - Grokking can hide continuous circuit formation behind abrupt accuracy changes.
    - Fourier circuits make modular addition unusually legible.
    - Next: applying these tools to larger, less clean models.

## Source material and assets

- Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability*.
- Welch Labs grokking video and local frames in `frames/`.
- Local experiment outputs and paper-style figures under `runs/`.
- Anthropic line-break material recorded in `papers/anthropic_2025_when_models_manipulate_manifolds.md`.

## Build notes

- Use a standard 16:9 Beamer deck with section transition slides, a cold open, outline, recap, and next-step frame.
- Recreate essential technical figures with the local experiment and matplotlib; use video frames only with attribution.
- Keep the Claude and SAE portions conceptual unless separate source figures are cleared for use.
