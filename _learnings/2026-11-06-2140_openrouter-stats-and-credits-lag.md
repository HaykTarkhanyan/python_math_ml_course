# OpenRouter's per-call stats and the account total lag behind the calls

**Symptom 1.** The first run of the API notebook crashed in its generation-stats cell:

```
HTTPError: 404 Client Error: Not Found for url: https://openrouter.ai/api/v1/generation?id=gen-...
```

after 6 tries 2 s apart (12 s). A separate probe the same evening got 404 at 0.9 s and 5.0 s and 200 at 9.4 s.

**Cause 1.** `GET /api/v1/generation?id=...` only knows a generation some seconds after the call returns, and
that delay varies (9 s, and more than 12 s, in two tries on 2026-11-06).

**Fix 1.** A bounded wait, then fail loudly: 20 tries x 3 s, `raise_for_status()` after the last one
(`llms/06_decoding/py_src/build_api_nb.py`, `generation_stats`). Do not silently skip the stats.

**Symptom 2.** After 66 calls, the notebook's ledger (summed `usage.cost` from every reply) said $0.00294;
`GET /api/v1/credits` before and after, with a 10 s wait, moved by only $0.00117.

**Cause 2.** The account total trails the newest calls; the 10 s wait used in practical 02 was not enough here.

**Consequences.** The per-reply `usage.cost` ledger is the number to trust and report. Use `/credits` only as a
rough cross-check, long after a batch, and never as the only cost measure.
