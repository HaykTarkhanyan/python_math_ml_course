# Checking YouTube links: oEmbed first, the watch page only when oEmbed is ambiguous

**Symptom 1 - oEmbed alone over-reports.** Checking 395 YouTube links through
`https://www.youtube.com/oembed?url=...` flagged 10 (404 / 401 / 403 / 400). A plain GET of the
watch URL returns 200 for all of them, deleted videos included. The watch page embeds
`"playabilityStatus":{"status":"...","reason":"..."}`, which settles it:

```
rH7r85UwJSg ('ERROR', "This video has been removed for violating YouTube's Terms of Service")  <- oEmbed 403
0UOa7qtqkIw ('ERROR', 'This video is unavailable')                                             <- oEmbed 404
dHy07B-UHkE ('OK', None) | The Art Market Scam Explained Using Bananas                         <- oEmbed 401
```

oEmbed 401 means "embedding disabled", not dead. Every oEmbed 404 / 400 (8 of 8) was dead.

**Symptom 2 - the watch page alone gets you blocked, and a naive check calls that "fine".** The
first `check_links.py --live` fetched ~390 watch pages at 8 threads. Google then answered every
watch URL from this IP with its rate-limit page, and Shorts URLs with the EU consent wall:

```
https://www.youtube.com/watch?v=7rIw7ocwMP4 -> 302 https://www.google.com/sorry/index?continue=...
https://youtube.com/shorts/2sphKXkgzqE      -> https://consent.youtube.com/m?continue=...
```

Neither page contains `playabilityStatus`, and the checker treated "no status found" as OK, so
five known-dead videos passed. oEmbed from the same IP kept answering normally.

**What the checker does now** (`non_essential/check_links.py`, verified with a known-answer test:
10 of 12 cases correct, the other 2 honestly "unverified" while the IP was rate-limited):

1. Extract the id from any URL form (watch, youtu.be, shorts, embed) and use `watch?v=ID`.
2. oEmbed 200 -> ok; 404 / 400 -> broken.
3. Anything else -> watch page, at most 2 in parallel: `ERROR` -> broken; `UNPLAYABLE` /
   `LOGIN_REQUIRED` -> unverified (region / age block: two videos were "UNPLAYABLE" from here
   yet exist); rate-limit page, consent page or no status -> unverified, never ok.

**Also.** A video id whose 11th character is outside `[AEIMQUYcgkosw048]` is almost certainly a
typo: ids encode 64 bits in 11 base64 characters, so the last one carries only 4 bits. Every real
id seen in the audit fit this; `XFUrPRTI2e6` did not, and oEmbed answered 400, not 404.
