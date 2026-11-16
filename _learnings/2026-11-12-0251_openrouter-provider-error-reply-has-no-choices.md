# OpenRouter can return HTTP 200 with no `choices` when the provider rejects a tool call

**Symptom.** Building the tools practical (`llms/tools_agents/`), the agent loop crashed with
`TypeError: 'NoneType' object is not subscriptable` at `r.choices[0]`, only for Llama 3.1 8B pinned to Groq.

**Cause.** The upstream provider rejected the model's tool call, and OpenRouter passed that on as a normal reply
whose `choices` is `None` and whose body carries an `error` object; the `openai` SDK does not raise. Seen on
2026-11-12, printing only the error body:

```
"Upstream error from Groq: tool call validation failed: attempted to call tool 'math_calculator' which was not in request.tools", code 502
"Upstream error from Groq: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details."
```

The model had invented a tool, or produced a malformed call, and Groq validates tool calls on its side.

**Consequences.**

- Check `r.choices` before indexing, and read `r.model_dump()["error"]["message"]` when it is empty. In an agent
  loop, report it as a failed run (printed, counted as a failure); elsewhere, raise.
- When replaying a saved run, a dump without `choices` does not pass `ChatCompletion.model_validate`; use
  `ChatCompletion.model_construct(**d)` for those.
- It is teaching material: a small model can call a tool that does not exist, not only pass bad arguments.
