# An endpoint test failed providers for our own bug: DeepSeek's end marker was missing

**Symptom.** In the job-extraction practical (`llms/02_decoding/`), the endpoint test rejected four DeepSeek V4.1
Flash endpoints with "logprobs do not line up with the reply" and picked `parasail/fast` ($0.45 / $1.80 per 1M
tokens) over cheaper ones such as `cohere/fp8` ($0.20 / $0.66). The notebook then blamed the providers.

**Cause.** The test glues the logprob pieces together, strips trailing end-of-text markers (`END_MARKS`) and requires
the result to equal the reply. The list had Gemma's `<eos>` and Llama's `<|eot_id|>`, but not DeepSeek's
`<｜end▁of▁sentence｜>` (fullwidth bars, U+2581 underscores). A re-probe on 2026-11-11
(`py_src/non_essential/probe_logprob_mismatch.py`, $0.0003) showed:

```
morph/fp8   57 tokens, 19 entries   -> broken (holes)
cohere/fp8  70 tokens, 70 entries   -> identical to the reply + <｜end▁of▁sentence｜>
novita/fp8  127 tokens, 2 entries   -> broken ('{"remote', not even in the reply)
makora/fp8  70 tokens, 70 entries   -> identical to the reply + <｜end▁of▁sentence｜>
```

With the marker added, Cohere and Makora line up. (A quicker probe with a simpler schema once saw Cohere's reply
differ in whitespace too, `"salary": \tnull` vs `"salary":null`, so Cohere is not perfectly clean either.)

**Consequences.**

- When a check rejects an option and the rejection costs money (here 2-3x per token), **save and read what the
  rejected option actually sent** before accepting the verdict. A verdict string alone hid the bug.
- End markers differ per model family; when adding a model, look at the last logprob entry of one reply.
- The saved run kept `parasail/fast` (redoing DeepSeek's 444 calls was not worth it); the notebook says so.
