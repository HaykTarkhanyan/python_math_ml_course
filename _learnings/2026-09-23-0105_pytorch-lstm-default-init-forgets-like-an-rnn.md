# A PyTorch LSTM at default init forgets as fast as a vanilla RNN

**Symptom.** In July, L21's "Does it work?" frame ran a quick LSTM-vs-RNN comparison on a
recall task and found **no** LSTM advantage (early-step gradient ratio LSTM/vanilla = 0.02x). The
frame reported it honestly, so the slide asked "does the LSTM work?" and then showed that it did
not. On 2026-09-23 a first-token-recall probe at T=40 reproduced it: vanilla RNN 100%, LSTM 24%
(chance 25%) after 1500 steps.

**Cause.** `nn.LSTM` / `nn.LSTMCell` initialise every bias uniformly around 0, so the forget gate
starts at sigmoid(0) = 0.5. Along the cell highway the backward factor is the forget gate itself
(d c_t / d c_{t-1} = diag(f_t)), so a half-shut gate halves the memory every step - the same
exponential decay as a vanilla RNN with |lambda| < 1. The LSTM *can* learn to open the gate, but
the gradient that would teach it is exactly the one that has vanished.

Measured with `ml/ch7_rnn/py_src/memory_highway.py` (sensitivity of the final state to the input
k steps back, at initialisation, hidden 32, T=100, seed 509):

| model | k=20 | k=40 | k=99 |
|---|---|---|---|
| vanilla RNN | 1.3e-05 | 7.6e-11 | 0 |
| LSTM, forget bias 0 (~PyTorch default) | 1.3e-05 | 2.7e-09 | 1.2e-19 |
| LSTM, forget bias 1 (Keras default) | 4.5e-03 | 3.0e-04 | 4.8e-07 |
| LSTM, forget bias 3 | 5.5e-02 | 3.7e-02 | 1.4e-01 |

And after training (first-token recall, <=1500 Adam steps, 2 seeds): at T=40 the bias-3 LSTM
solved it in **100 steps** on both seeds; the bias-0 LSTM stayed at chance on both.

**Consequences.**

- Keras sets `unit_forget_bias=True` by default (adds 1 to the forget-gate bias, following
  Jozefowicz et al. 2015); PyTorch does not. Any "LSTM beats RNN" demo written in PyTorch needs
  `lstm.bias_ih_l0.data[H:2*H].fill_(...)` (gate order i, f, g, o) or it may show nothing.
- The July null result was not a bad seed or a short budget - it had a cause, and the cause is
  the better lesson. L21 now teaches it ("The price, and one gotcha"), and the RNN practical's
  Part 3 makes students set the bias themselves.
- Even with the gate open, T=80 was a coin flip (75% / 24% by seed) at this budget: addition
  makes long memory *learnable*, not free. Say so on the slide rather than cherry-picking T.
