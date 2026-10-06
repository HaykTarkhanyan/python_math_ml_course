# NumPy's default integer is 32-bit on Windows here, and it overflows without a word

**Symptom.** LLM-12's KV-cache chart (`ml/14_llms/py_src/quant_figs.py`) showed Llama 3 8B's KV cache
as "0 GiB" at 32K and 128K tokens. The by-hand checks in the same script, written with plain Python
integers, said 4 and 16 GiB. No error, no warning.

**Cause.** The `ma` venv has NumPy 1.26.4. Before NumPy 2.0, the default integer on Windows is
int32, so an array built from Python ints is int32 and products past 2^31 wrap around:

```
>>> np.__version__, np.array([1]).dtype, (np.array([131072]) * np.array([32768])).dtype
('1.26.4', dtype('int32'), dtype('int32'))
```

131,072 bytes per token x 32,768 tokens = 4.3e9 > 2^31 - 1 = 2.1e9.

**Consequences.**
- Byte counts, token counts x sizes, FLOP counts: build the array with `dtype=np.float64` (or
  `np.int64`) whenever a product can pass 2.1e9.
- A check written with Python ints does not protect a plot computed with NumPy ints - check the
  values the figure actually draws (the script now asserts the plotted GiB values too).
- Upgrading the venv to NumPy 2.x would change the default to int64 on Windows; that is a separate,
  deliberate decision (pins, other notebooks).
