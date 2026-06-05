# Playwright runbook: build the form in the browser

Exact, ordered steps to paste a generated `build_form.gs` into Apps Script and run it.
Tool names below are the Playwright MCP tools (`browser_*`). The user does the login and
the authorization click; everything else is automated.

## 0. Prereqs
- A generated `build_form.gs` on disk (see `scripts/generate_form_gs.py`).
- Playwright MCP available. Load the tools you need: `browser_navigate`, `browser_snapshot`,
  `browser_take_screenshot`, `browser_click`, `browser_press_key`, `browser_evaluate`,
  `browser_wait_for`, `browser_tabs`.

## 1. Open a project and have the user log in
- `browser_navigate` to `https://forms.google.com` first if you want to surface the Google
  sign-in page, OR go straight to a new script project: `browser_navigate` to
  `https://script.new`.
- If the user isn't logged in, the page is the Google sign-in. Tell the user to log in, then
  continue. Confirm login by checking the page URL/title is the Apps Script editor
  (`...Project Editor - Apps Script`).
- `script.new` lands on `.../home/projects/<ID>/edit`. **Note this project URL** — reuse it
  for every additional form to avoid re-authorizing (see step 7).

## 2. Put the script on the OS clipboard
The browser shares the OS clipboard, so this is how the code gets in reliably.
- macOS: `pbcopy < build_form.gs`
- Linux: `xclip -selection clipboard < build_form.gs`
- Windows PowerShell: `Set-Clipboard -Value (Get-Content -Raw -Encoding UTF8 build_form.gs)`
- **Non-ASCII content**: always read as UTF-8 (the `-Encoding UTF8` above). Verify it survived:
  print `(Get-Clipboard -Raw).Length` and check a known non-ASCII substring is present.

## 3. Confirm the editor is ready, then paste
- Wait ~3s for Monaco to load (`browser_wait_for` time:3).
- Probe Monaco is reachable (it is, in the Apps Script IDE):
  `browser_evaluate`: `() => ({ has: typeof window.monaco, models: monaco.editor.getModels().length })`
- Focus + select-all + paste:
  - `browser_click` target `.monaco-editor`
  - `browser_press_key` `Control+a`  (use `Meta+a` on macOS)
  - `browser_press_key` `Control+v`  (use `Meta+v` on macOS)
- **Do NOT** use `browser_type` to enter the code — Monaco auto-indents and auto-closes
  brackets on typed input and will corrupt it. Paste fires the paste handler and inserts
  verbatim.

## 4. Verify the paste landed
`browser_evaluate`:
```js
() => {
  const v = window.monaco.editor.getModels()[0].getValue();
  return { length: v.length, hasBuildForm: v.includes('function buildForm'),
           rowCount: (v.match(/"correctIndex"/g) || []).length, head: v.slice(0,60) };
}
```
Check `length` matches the file, `hasBuildForm` is true, and `rowCount` equals your question
count. If replacing earlier code (step 7), also confirm the OLD title string is gone.

## 5. Save
- `browser_press_key` `Control+s` (`Meta+s` on macOS).
- `browser_wait_for` textGone:`Saving project...`

## 6. Run, and let the user authorize (first time only)
- Click Run via its stable label: `browser_click` target `[aria-label="Run the selected function"]`.
  (The single embedded function `buildForm` is preselected in the run dropdown.)
- The **first** run on a new project opens an authorization tab requesting the **Forms** scope
  (`.../auth/forms`). **Hand this to the user**: ask them to review and approve (they'll pass
  the "unverified app" interstitial: Advanced → Go to project → Allow). Do not click consent
  for them.
- After they approve, the run continues automatically. `browser_tabs` select index 0 to return
  to the editor.

## 7. Building more forms in the same project = no re-auth
Authorization is per project per scope. Same Forms scope means no second prompt. To build the
next form: copy its `.gs` to the clipboard (step 2), `browser_navigate` back to the **project
URL** from step 1, then repeat paste → verify → save → run (steps 3-6). The run executes
without an auth prompt.

## 8. Read the result from the execution log
Building 40-50 questions takes ~60-90s. Poll, don't assume.
- `browser_wait_for` time:15, then `browser_evaluate`:
```js
() => {
  const t = document.body.innerText || '';
  const lines = t.split('\n').map(l=>l.trim())
    .filter(l => /Built|Edit URL|Published URL|Execution (started|completed)|Error|Exception/i.test(l));
  const urls = t.match(/https?:\/\/(?:docs\.google\.com\/forms|forms\.gle)\/[^\s"')]+/g) || [];
  return { lines, urls: Array.from(new Set(urls)) };
}
```
- Repeat the wait+read until you see `Built N questions`, the **Edit URL**, the **Published
  URL**, and `Execution completed`. The URL lines stream in just after `Built N`.

## 9. Verify the form (don't just trust the log)
- `browser_navigate` to the Edit URL.
- Confirm via DOM, e.g. count rendered question titles and read `Total points`:
```js
() => {
  const t = document.body.innerText || '';
  const numbered = t.split('\n').map(l=>l.trim()).filter(l=>/^\d+\.\s/.test(l));
  return { numbered: numbered.length, points: (t.match(/Total points:\s*\d+/)||[])[0],
           first: numbered.slice(0,2), last: numbered.slice(-1) };
}
```
  (The Forms editor scrolls an inner container, not `window`, so reading `innerText` is more
  reliable than scrolling+screenshotting to inspect questions.)

## Common failure modes
- **Pasted code looks mangled / indented weirdly** → you typed instead of pasted, or the
  editor wasn't focused. Re-focus `.monaco-editor`, `Control+a`, `Control+v`.
- **Mojibake in non-Latin text** → clipboard was read as ANSI. Re-copy with UTF-8.
- **Auth prompt on a reused project** → the new code needs a scope the project wasn't granted
  (e.g. you added DriveApp). Keep the builder Forms-only, or have the user approve the new scope.
- **`Built N` but URLs missing** → still streaming; wait a few seconds and re-read the log.
