# Decode BPE merge pairs as concatenated bytes - each half alone is invalid UTF-8

**Symptom.** Scanning `tokenizer.json` merge lists for the first Armenian-producing merge
reported Qwen 2.5 as having "no Armenian merges at all" - while the vocab census of the same
tokenizer counted 79 Armenian tokens. Contradiction.

**Cause.** The scan decoded each merge *part* separately and concatenated the strings:

```python
merge_part_to_text(a) + merge_part_to_text(b)   # WRONG
```

An Armenian letter is 2 UTF-8 bytes. The merge that creates it joins two single-byte tokens
('Õ' + '¡' in the GPT-2 byte alphabet for ա). Each half alone is an invalid UTF-8 sequence, so
`bytes(...).decode("utf-8", errors="ignore")` turned BOTH halves into empty strings and the merge
became invisible. Every merge whose boundary splits a multi-byte character is silently dropped
this way - which for a non-Latin script is *all the early, most interesting ones*.

**Fix.** Concatenate the bytes of both parts first, decode once:

```python
raw = bytes(BYTE_DEC[ch] for ch in a + b)
raw.decode("utf-8", errors="ignore")
```

Measured effect: Qwen 2.5's first Armenian merge appeared at rank 144,669 (95.6% through the
vocab) instead of "none"; Llama 3's moved from 267,891 to the true 263,556 (94.1%). The
tiktoken-ranks path never had the bug because ranks give whole-token bytes already merged.

Where it lives: `ml/claude_projects/armenian_glitch_token_hunt/armenian_token_hunt.ipynb`
(`merged_surface`).
