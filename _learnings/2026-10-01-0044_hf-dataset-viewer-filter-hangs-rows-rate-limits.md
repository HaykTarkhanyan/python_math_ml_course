# Hugging Face dataset viewer: `/filter` can hang, `/rows` rate-limits fast - estimate offsets instead

**Symptom.** Pulling a few `like` / `dislike` / `no_gesture` images from
`cj-mills/hagrid-classification-512p-no-gesture-150k` (153,735 rows) without downloading its
3.8 GB of parquet:

```
/filter?...&where="label"=4      curl: http=000 after 90 s;  urllib: RemoteDisconnected
/rows?offset=0&length=3          http=200 in 1.1 s
binary search, 1-row /rows calls HTTP Error 429: Too Many Requests  (inside the first bisection, <= 18 calls)
3 calls, minutes later           HTTP Error 429 on the first call
3 calls tens of minutes later, 4 s apart   all 200
```

**What worked.** The rows of that dataset are sorted by label (offset 0 -> label 0, offset 50,000
-> label 6), and its Kaggle twin (`innominate817/hagrid-classification-512p-no-gesture-150k`) lists
the counts: 125,912 gesture images over 18 alphabetical classes, about 7,000 each, plus 27,823
`no_gesture`. So estimate each class's offset, fetch **one** `length=30` page per class, and
assert every returned label before saving anything. Three calls total; 90 images, 692 KB.

**Consequences.**

- Treat the dataset-viewer API as a light, rate-limited convenience: a few spaced calls, never a
  loop. Never build anything students run on it.
- Look for sortedness before searching: an index estimate plus one verifying call beats a
  bisection that burns the rate limit.
- On a 429, stop and switch approach; retrying in a loop only extends the block.
