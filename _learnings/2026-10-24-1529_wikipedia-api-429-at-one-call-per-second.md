# The Armenian Wikipedia API answers HTTP 429 about once a minute even at one call per second

**Symptom.** `ml/13_rnns/py_src/build_seq2seq_data.py` fetches article text through the MediaWiki
API (`prop=extracts`, `list=random`, `prop=info`). The first run, 0.15 s between calls, died with
`urllib.error.HTTPError: HTTP Error 429: Too Many Requests` after about 15 calls.

**What worked.** One call per second, plus a retry on 429/503 that waits for the server's
`Retry-After` header, logs a warning, and raises after 5 failed retries. The successful run still got a 429 roughly every
minute, each with `Retry-After` of 43-46 s:

```
WARNING HTTP 429, retry 1/5 in 45 s     (x19, about one per minute)
INFO wrote hy_sentences.tsv: 4509 sentences (3705 train, 804 test) from 187 articles, 943,956 bytes, in 1181 s
```

So a fetch of a few hundred articles takes about 20 minutes, mostly spent waiting. A User-Agent
with a contact URL was set throughout. The 429s are a rate limit, not a block.

**Consequences.**
- Budget ~20 min for a few hundred articles. Run it in the background with progress output, and
  ship the result as a data file so nobody has to fetch again.
- The same run also showed that redirected titles must be resolved through the API's `normalized`
  and `redirects` maps. Matching the returned page titles against the requested ones reported 3
  redirects as "missing".
- Of the 187 articles fetched, only 178 contribute sentences after cleaning. Quote the count from the
  data, not from the fetch log.
