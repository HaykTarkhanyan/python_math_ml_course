# git prints Armenian filenames octal-escaped unless you pass `-c core.quotepath=off`

**Symptom.** The first link checker (2026-09-28 audit) reported the home page's link to
`math/Lectures/Կարճ մաթեմ մեքենայական ուսուցման համար.pdf` as "path not in repo HEAD". The file
was in git and the link worked:

```
$ git -c core.quotepath=off ls-files math/Lectures | grep Կարճ
math/Lectures/Կարճ մաթեմ մեքենայական ուսուցման համար.pdf
$ curl -s -o /dev/null -w "%{http_code}" "https://github.com/.../math/Lectures/%D4%BF%D5%A1..."
200
```

**Cause.** With the default `core.quotepath=true`, `git ls-files` / `git ls-tree` print every
non-ASCII path quoted and octal-escaped (`"math/Lectures/\324\277\325\241..."`). A script that
builds a set of tracked paths from that output never matches a real Armenian filename.

**Consequences.**

- Any script that reads paths from git in this repo passes `-c core.quotepath=off`
  (`non_essential/check_links.py` does, in `git_lines()`).
- The bug only produces false "missing" reports, never hides a real one, so it shows up as noise
  on exactly the Armenian files. If a checker flags only non-ASCII paths, suspect this first.
