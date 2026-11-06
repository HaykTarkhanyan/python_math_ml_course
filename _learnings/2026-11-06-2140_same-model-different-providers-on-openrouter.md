# One open model on OpenRouter is several different deployments: pin the provider before comparing answers

**Symptom.** Three calls to `meta-llama/llama-3.1-8b-instruct` at temperature 0 with the same prompt
(exploration, 2026-11-06): two identical answers from Cloudflare, a different one from DeepInfra. The answers
split exactly along the `provider` field of the reply.

**Cause.** OpenRouter spreads requests for an open model over every company that serves it (mostly by price,
with fallbacks). `GET /api/v1/models/<id>/endpoints` (public, no key) shows how different they are, measured
2026-11-06 for Llama 3.1 8B:

```
DeepInfra   fp8   $0.020 / $0.040 per 1M   no logprobs
Groq        ?     $0.050 / $0.080          no top_k, no min_p, no logprobs
Cloudflare  fp8   $0.152 / $0.287          no logprobs
CoreWeave   bf16  $0.220 / $0.220          no min_p
```

Different weight formats, different supported knobs, prices 11x apart.

**Consequences.**
- For any comparison (temperature 0, seeds, logprobs, knob sweeps), pin one provider:
  `extra_body={"provider": {"order": ["deepinfra"], "allow_fallbacks": False}}`. Pick it from the endpoints
  table in code (the API notebook's `cheapest(need)`), not by hard-coding a name that may disappear.
- Pinned is not deterministic either: 5 calls pinned to DeepInfra at temperature 0 still gave 2 different
  answers (the batching effect from LLM-6). Without the `provider` column you cannot tell the two effects apart.
- On the final notebook run all 8 routed calls happened to land on DeepInfra, so "routed" and "pinned" looked
  the same; the prose says so instead of claiming a provider split.
- Prompt caching only pays if the next request reaches the provider holding the cache; OpenRouter's sticky
  routing does that for 10 minutes, but not when `provider.order` is set.
- Idea from the built-in "you should know" plugin, confirmed by the measurement above.
