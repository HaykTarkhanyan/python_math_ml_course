# A 200 after a redirect to the site's homepage is a dead link, not a working one

**Symptom.** The flag counter's details link (`http://s01.flagcounter.com/more/1oO`) timed out.
I "fixed" it to `https://info.flagcounter.com/1oO` because curl returned 200 - and shipped that on
the home page in round one. It pointed at the Flag Counter marketing homepage, not our stats:

```
https://info.flagcounter.com/1oO   -> HTTP:302 FINAL:http://flagcounter.com/ | title: (none) | mentions 1oO: 0
https://s01.flagcounter.com/more/1oO/ -> HTTP:200 | title: Flag Counter » Overview | mentions 1oO: 4
```

The real problem with the original link was only plain `http` (port 80 times out); https works.

**Cause.** `curl -L` follows redirects and reports the final status. Many sites answer an
unknown path with a redirect to `/`, so "200" only proves the domain is alive.

**Consequences.**

- Verify a replacement URL by content (page title, the id appearing in the body), not by status.
- `check_links.py --live` flags "redirects to a homepage" (final path `/`, requested path not `/`,
  different host) as broken.
- Disproven theory: "the /more/ page is gone". It exists; only the scheme was wrong.
