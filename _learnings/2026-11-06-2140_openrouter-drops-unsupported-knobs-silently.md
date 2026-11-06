# OpenRouter drops a knob the provider does not support, without any error

**Symptom.** `anthropic/claude-haiku-5.5` called with `temperature=2.0` through OpenRouter returned a normal,
fluent sentence and HTTP 200. The same temperature on Llama 3.1 8B produced word salad
("The chilled onionatois-modris Medicallicensed socialism Dent allowNull ..."). Same for `openai/gpt-5-nano`
with `temperature=0.3`: accepted, no warning.

**Cause.** OpenRouter's docs (provider routing, read 2026-11-06): providers that do not support a parameter
"can still receive the request, but will ignore unknown parameters". The model's `supported_parameters`
in `GET /api/v1/models` shows what it takes: for Haiku 5.5 and GPT-5 nano there is no `temperature` at all.
Its migration guides say Claude Opus 4.7 and Sonnet 5 silently ignore `temperature`, `top_p` and `top_k`.

**The fix: ask for the failure.** `extra_body={"provider": {"require_parameters": True}}` routes only to
providers that support every parameter in the request; if none does, the call fails:

```
openai.NotFoundError: Error code: 404 - No endpoints found that can handle the requested parameters.
```

**Consequences.**
- "The request succeeded" does not mean "my settings were used". For any experiment with sampling knobs, set
  `require_parameters`, or check `supported_parameters` first.
- The same applies to `top_k`, `min_p`, `logprobs`, `seed`: open models support them only on some providers
  (see `2026-11-06-2140_same-model-different-providers-on-openrouter.md`).
- Evidence: `llms/06_decoding/xx_llm_api_solution.ipynb`, section 8a, and its ledger
  `llms/06_decoding/results/xx_api_ledger.json`.
