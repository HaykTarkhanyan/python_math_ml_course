"use strict";
/* Course browser for learning/robot-learning. Data comes from data.js (scripts/build_browser.py). */

const C = window.COURSE;
if (!C) {
  document.body.innerHTML = "<p style='padding:2em'>data.js is missing. Run <code>scripts/build_browser.py</code>.</p>";
  throw new Error("COURSE data missing - run scripts/build_browser.py");
}

const HTTP = location.protocol.startsWith("http");
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pad = (n) => String(n).padStart(2, "0");
const fmt = (t) => {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
};
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); return true; } catch (e) { return false; } },
};
const weekByKey = Object.fromEntries(C.weeks.map((w) => [w.key, w]));
const hwByKey = Object.fromEntries(C.homework.map((h) => [h.key, h]));
const main = $("#main");

/* ---------- theme ---------- */
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}
applyTheme(store.get("rl:theme"));
function toggleTheme() {
  const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
  const next = dark ? "light" : "dark";
  applyTheme(next);
  store.set("rl:theme", next);
}

/* ---------- sidebar ---------- */
function renderSidebar() {
  const weeks = [...C.weeks].sort((a, b) => a.key.localeCompare(b.key));
  $("#sidebar").innerHTML = `
    <div class="brand"><a href="#">Robot Learning<span>ETH Zurich &middot; Spring 2026 &middot; Oier Mees</span></a></div>
    <form class="search" id="searchForm" role="search">
      <input id="q" type="search" placeholder="Search transcripts &amp; slides  ( / )" autocomplete="off">
    </form>
    <nav>
      <div class="navhead">Weeks</div>
      ${weeks.map((w) => `<a href="#w${w.key}" data-nav="w${w.key}" class="${w.placeholder ? "dim" : ""}">
          <span class="num">${w.key}</span><span>${esc(w.topic)}</span></a>`).join("")}
      <div class="navhead">Homework</div>
      ${C.homework.map((h) => `<a href="#hw/${h.key}" data-nav="hw/${h.key}">${esc(h.label)}</a>`).join("")}
      <div class="navhead">More</div>
      <a href="#about" data-nav="about">About these files</a>
    </nav>
    <div class="sidefoot">
      <button id="themeBtn" type="button" title="Toggle light / dark">Theme</button>
      <button id="exportBtn" type="button" title="Notes live only in this browser. Download them all as Markdown to keep a copy.">Export notes</button>
    </div>`;
  $("#searchForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#q").value.trim();
    if (q) location.hash = "#search/" + encodeURIComponent(q);
  });
  $("#themeBtn").addEventListener("click", toggleTheme);
  $("#exportBtn").addEventListener("click", exportNotes);
}

function markNav(id) {
  $$("#sidebar nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === id));
}

/* ---------- YouTube players ---------- */
let ytReady = null;
const players = {}; // slot -> {player, ready, vid}
let followTimer = null;

function loadYT() {
  if (!ytReady) {
    ytReady = new Promise((resolve, reject) => {
      window.onYouTubeIframeAPIReady = resolve;
      const s = document.createElement("script");
      s.src = "https://www.youtube.com/iframe_api";
      s.onerror = () => reject(new Error("YouTube iframe API failed to load (offline?)"));
      document.head.appendChild(s);
    });
  }
  return ytReady;
}

function destroyPlayers() {
  clearInterval(followTimer);
  followTimer = null;
  for (const k of Object.keys(players)) {
    try { players[k].player.destroy(); } catch (e) { console.warn("player destroy failed", e); }
    delete players[k];
  }
}

function mountPlayers() {
  const mounts = $$(".ytmount");
  if (!mounts.length) return;
  loadYT().then(() => {
    for (const m of mounts) {
      const slot = m.dataset.slot, vid = m.dataset.vid;
      const entry = { vid, ready: false, last: -1 };
      entry.player = new YT.Player(m.id, {
        videoId: vid,
        playerVars: { rel: 0, modestbranding: 1, origin: location.origin },
        events: { onReady: () => { entry.ready = true; } },
      });
      players[slot] = entry;
    }
    followTimer = setInterval(follow, 700);
  }).catch((err) => {
    console.error(err);
    mounts.forEach((m) => { m.outerHTML = posterHtml({ id: m.dataset.vid }, "The YouTube player could not load (are you offline?)."); });
  });
}

function follow() {
  for (const [slot, P] of Object.entries(players)) {
    if (!P.ready || P.player.getPlayerState() !== 1) continue;
    const box = $(`#vid-${slot}`);
    if (!box || !$(".followbox", box).checked) continue;
    const t = Math.floor(P.player.getCurrentTime());
    if (Math.floor(t / 15) === Math.floor(P.last / 15)) continue;
    P.last = t;
    highlight(slot, t, true);
  }
}

function seek(slot, t) {
  const box = $(`#vid-${slot}`);
  highlight(slot, t, true);
  const P = players[slot];
  if (P && P.ready) {
    P.player.seekTo(t, true);
    P.player.playVideo();
    const r = $(".player", box).getBoundingClientRect();
    if (r.top < 0 || r.bottom > innerHeight) $(".player", box).scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    window.open(`https://www.youtube.com/watch?v=${box.dataset.vid}&t=${t}s`, "_blank", "noopener");
  }
}

function highlight(slot, t, scroll) {
  const box = $(`#vid-${slot}`);
  if (!box) return;
  const pick = (els) => {
    let cur = null;
    for (const el of els) { if (+el.dataset.t <= t) cur = el; else break; }
    return cur;
  };
  const paras = $$(".transcript p", box), items = $$(".tocitem", box);
  paras.forEach((p) => p.classList.remove("cur"));
  items.forEach((p) => p.classList.remove("cur"));
  const p = pick(paras), it = pick(items);
  if (it) it.classList.add("cur");
  if (p) {
    p.classList.add("cur");
    if (scroll) {
      const tr = $(".transcript", box);
      tr.scrollTo({ top: p.offsetTop - 40, behavior: "smooth" });
    }
  }
  if (it && scroll) {
    it.closest("ol").scrollTo({ top: it.offsetTop - 60, behavior: "smooth" });
  }
}

/* ---------- builders ---------- */
function card(id, title, body, sub = "", open = true) {
  return `<details class="card" id="sec-${id}" ${open ? "open" : ""}>
    <summary><h2>${title}</h2>${sub ? `<span class="sub">${sub}</span>` : ""}</summary>
    <div class="cardbody">${body}</div></details>`;
}

function posterHtml(v, msg) {
  const why = msg || `Embedded playback needs a local server (YouTube refuses <code>file://</code> pages):<br>
    <code>python -m http.server 8765 -d learning/robot-learning/browser</code> then open <code>http://localhost:8765</code>.<br>
    Until then, every timestamp opens YouTube in a new tab at that moment.`;
  return `<div class="poster"><img src="https://i.ytimg.com/vi/${v.id}/hqdefault.jpg" alt="">
    <div><a class="play" href="https://www.youtube.com/watch?v=${v.id}" target="_blank" rel="noopener">&#9654; Watch on YouTube</a>
    <small>${why}</small></div></div>`;
}

function linkify(text) {
  return esc(text).replace(/https?:\/\/[^\s<]+/g, (u) => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
}

function videoHtml(v, slot) {
  return `<div class="vid" id="vid-${slot}" data-vid="${v.id}" data-slot="${slot}">
    <div class="vtop">
      <div class="player">${HTTP ? `<div class="ytmount" id="yt-${slot}" data-slot="${slot}" data-vid="${v.id}"></div>` : posterHtml(v)}</div>
      <div class="toc">
        <div class="tochead"><span>Contents</span><span class="src">${esc(v.toc_source)}</span></div>
        <ol>${v.toc.map((c) => `<li><button type="button" class="tocitem" data-t="${c.t}">
          <span class="tt">${fmt(c.t)}</span><span>${esc(c.title)}</span></button></li>`).join("")}</ol>
      </div>
    </div>
    <div class="vmeta"><span>${esc(v.title)}</span><span>${esc(v.duration)}</span>
      <a href="https://www.youtube.com/watch?v=${v.id}" target="_blank" rel="noopener">YouTube &#8599;</a>
      <label title="Highlight and scroll the transcript while the video plays"><input type="checkbox" class="followbox" checked> follow video</label></div>
    ${v.description ? `<details class="desc"><summary>Video description</summary><pre>${linkify(v.description)}</pre></details>` : ""}
    <div class="transcript">${v.paragraphs.map((p) => `<p data-t="${p.t}"><button type="button" class="ts" data-t="${p.t}">${fmt(p.t)}</button>${esc(p.text)}</p>`).join("")}</div>
    <div class="caveat">Transcript = YouTube auto-captions, cleaned. Expect misheard names and jargon.</div>
  </div>`;
}

function frameSrc(w, n) { return `frames/${w.dir}/p-${String(n - 1).padStart(3, "0")}.jpg`; }

function slidesHtml(w) {
  const s = w.slides;
  const timed = !!w.lecture;
  return `<div class="slideabout">${s.about_html}</div>
    <div class="slidetoc">${s.sections.map((x) => `<a class="chip" href="#w${w.key}/slide/${x.id}" title="${esc(x.title)}">${x.num ?? (x.kind === "title" ? "T" : x.kind === "end" ? "E" : "&middot;")}</a>`).join("")}</div>
    ${s.sections.map((x) => {
      const [a, b] = x.frames;
      return `<article class="slide" id="slide-${x.id}">
        <button type="button" class="thumb" data-a="${a}" data-b="${b}" title="View the original frame${a === b ? "" : "s"}">
          <img loading="lazy" src="${frameSrc(w, b)}" alt="slide frame ${b}"><span>${a === b ? `frame ${a}` : `frames ${a}-${b}`}</span></button>
        <div class="sbody"><h3>${x.title_html}</h3>
          <div class="smeta">${timed && x.t != null ? `<button type="button" class="slidetime" data-t="${x.t}" title="Jump to roughly this point in the lecture">&#9654; &asymp;${fmt(x.t)}</button>` : ""}</div>
          ${x.html}</div></article>`;
    }).join("")}`;
}

function overviewHtml(w) {
  const hw = w.homework.map((k) => hwByKey[k]).filter(Boolean);
  const guests = w.guests || [];
  return `<div class="ov">
    <div>
      <h3>Summary <span class="comment-tag">(Claude's comments)</span></h3><p>${esc(w.summary)}</p>
      ${w.key_ideas.length ? `<h3>Key ideas</h3><ul>${w.key_ideas.map((k) => `<li>${esc(k)}</li>`).join("")}</ul>` : ""}
    </div>
    <div>
      ${guests.length ? `<h3>Guest talk</h3><ul>${guests.map((g) => `<li><b>${esc(g.speaker)}</b>${g.affiliation ? `, ${esc(g.affiliation)}` : ""}<br><span class="who">${esc(g.title.replace(/\s*\[ETHZ Robot Learning 2026\]$/, "").replace(/^[^:]+:\s*/, ""))}</span></li>`).join("")}</ul>` : ""}
      ${w.papers.length ? `<h3>Paper discussion</h3><ul class="papers">${w.papers.map((p) => `<li><a href="${p.url}" target="_blank" rel="noopener">${esc(p.title)}</a><br><span class="who">${esc(p.who)}</span></li>`).join("")}</ul>` : ""}
      ${hw.length ? `<h3>Homework</h3><ul>${hw.map((h) => `<li><a href="#hw/${h.key}">${esc(h.label)}</a>${h.due ? ` <span class="who">due ${esc(h.due)}</span>` : ""}</li>`).join("")}</ul>` : ""}
    </div></div>`;
}

function notesHtml(w) {
  return `<p class="empty">Remarks I added while transcribing: slide errors, inconsistencies, missing slide numbers.</p>
    <ul class="notes">${w.notes.map((n) => `<li><a class="where" href="#w${w.key}/slide/${n.slide}">${esc(n.label)}</a>${n.html}</li>`).join("")}</ul>`;
}

function myNotesHtml(w) {
  const saved = store.get(`rl:notes:${w.key}`) ?? "";
  return `<p class="warnbox"><b>Your notes are not saved anywhere permanent.</b> They are kept only in this browser's local storage
    on this computer: not in the repo, not online, not synced to other browsers or devices. Clearing browser data or a private window
    loses them. Use <b>Export notes</b> in the sidebar to download a Markdown copy.</p>
    <textarea class="mynotes" data-key="${w.key}" placeholder="Your own notes for week ${w.key} (kept in this browser only - export to keep them)">${esc(saved)}</textarea>
    <div class="savestate"></div>`;
}

/* ---------- views ---------- */
let currentWeek = null;

function renderWeek(w, rest) {
  if (currentWeek !== w.key) {
    destroyPlayers();
    currentWeek = w.key;
    const hasVideo = !!w.lecture;
    const sections = [["overview", "Overview"]];
    if (hasVideo) sections.push(["lecture", "Lecture"]);
    (w.guests || []).forEach((g, i) => sections.push([`guest${i}`, "Guest talk"]));
    if (w.slides) sections.push(["slides", "Slides"]);
    if (w.notes.length) sections.push(["notes", "Transcriber notes"]);
    sections.push(["mynotes", "My notes"]);
    main.innerHTML = `<div class="wrap">
      <div class="kicker">Week ${w.key} &middot; ${esc(w.date)}</div>
      <h1>${esc(w.topic)}</h1>
      <div class="onpage">${sections.map(([id, label]) => `<button type="button" data-goto="sec-${id}">${label}</button>`).join("")}</div>
      ${card("overview", "Overview", overviewHtml(w))}
      ${hasVideo ? card("lecture", `Lecture &middot; ${esc(w.lecture.speaker)}`, videoHtml(w.lecture, "lecture"), `${esc(w.lecture.duration)} &middot; ${w.lecture.toc.length} entries`) : ""}
      ${(w.guests || []).map((g, i) => card(`guest${i}`, `Guest talk &middot; ${esc(g.speaker)}`, videoHtml(g, `guest${i}`), esc(g.duration))).join("")}
      ${w.slides ? card("slides", "Slides", slidesHtml(w), `${w.slides.sections.length} sections`) : ""}
      ${w.notes.length ? card("notes", "Transcriber notes", notesHtml(w), `${w.notes.length}`) : ""}
      ${card("mynotes", "My notes", myNotesHtml(w))}
    </div>`;
    main.scrollTop = 0;
    mountPlayers();
  }
  applyFocus(w, rest);
}

function applyFocus(w, rest) {
  const [kind, arg] = rest;
  if (kind === "slide" && arg) {
    const el = $(`#slide-${CSS.escape(arg)}`);
    if (!el) return;
    $("#sec-slides").open = true;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  } else if ((kind === "lecture" || /^guest\d+$/.test(kind || "")) && arg != null) {
    const sec = $(`#sec-${kind}`);
    if (!sec) return;
    sec.open = true;
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
    highlight(kind, +arg, true);
  }
}

function renderHome() {
  destroyPlayers(); currentWeek = null;
  const weeks = [...C.weeks].sort((a, b) => a.key.localeCompare(b.key));
  const words = (w) => w.lecture ? [w.lecture, ...w.guests].reduce((n, v) => n + v.paragraphs.reduce((m, p) => m + p.text.split(" ").length, 0), 0) : 0;
  main.innerHTML = `<div class="wrap">
    <div class="kicker">ETH Zurich 263-5911-00L &middot; Spring 2026 &middot; Oier Mees</div>
    <h1>Robot Learning: From Fundamentals to Foundation Models</h1>
    <p class="empty">Lectures, guest talks, transcripts, slide text and homework in one place. Search with <b>/</b>. Built ${esc(C.built)}.</p>
    <div class="grid">${weeks.map((w) => `<a class="weekcard" href="#w${w.key}">
      <div class="k">Week ${w.key} &middot; ${esc(w.date)}</div><h3>${esc(w.topic)}</h3><p>${esc(w.summary)}</p>
      <div class="stats">${w.placeholder ? "no recording" : `${1 + w.guests.length} video${w.guests.length ? "s" : ""} &middot; ${w.slides.sections.length} slide sections &middot; ${Math.round(words(w) / 1000)}k transcript words`}</div></a>`).join("")}</div>
  </div>`;
}

function renderDoc(title, kicker, html, extra = "") {
  destroyPlayers(); currentWeek = null;
  main.innerHTML = `<div class="wrap"><div class="kicker">${kicker}</div><h1>${title}</h1>${extra}<div class="doc">${html}</div></div>`;
  main.scrollTop = 0;
}

/* ---------- search ---------- */
let index = null;
function buildIndex() {
  index = [];
  for (const w of C.weeks) {
    if (!w.lecture) continue;
    [["lecture", w.lecture], ...w.guests.map((g, i) => [`guest${i}`, g])].forEach(([slot, v]) => {
      for (const p of v.paragraphs) index.push({ w, kind: "video", slot, who: v.speaker, t: p.t, text: p.text, low: p.text.toLowerCase() });
    });
    for (const s of w.slides.sections) index.push({ w, kind: "slide", id: s.id, title: s.title, text: s.text, low: (s.title + " " + s.text).toLowerCase() });
  }
}

function snippet(text, terms) {
  const low = text.toLowerCase();
  const at = Math.max(0, low.indexOf(terms[0]) - 90);
  let s = (at ? "&hellip;" : "") + esc(text.slice(at, at + 260)) + (at + 260 < text.length ? "&hellip;" : "");
  for (const t of terms) {
    const safe = esc(t).replace(/[.*+?^$(){}|[\]\\]/g, "\\$&");
    s = s.replace(new RegExp("(" + safe + ")", "gi"), "<mark>$1</mark>");
  }
  return s;
}

function renderSearch(q) {
  destroyPlayers(); currentWeek = null;
  if (!index) buildIndex();
  $("#q").value = q;
  const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
  const hits = terms.length ? index.filter((r) => terms.every((t) => r.low.includes(t))) : [];
  const shown = hits.slice(0, 300);
  main.innerHTML = `<div class="wrap"><div class="kicker">Search</div><h1>&ldquo;${esc(q)}&rdquo;</h1>
    <p class="empty">${hits.length} match${hits.length === 1 ? "" : "es"} in transcripts and slides${hits.length > shown.length ? ` (first ${shown.length} shown)` : ""}. All words must appear in the same minute of transcript or the same slide.</p>
    <ul class="results">${shown.map((r) => r.kind === "video"
      ? `<li><a href="#w${r.w.key}/${r.slot}/${r.t}"><div class="rh">Week ${r.w.key} &middot; ${esc(r.who)} &middot; ${fmt(r.t)}</div>${snippet(r.text, terms)}</a></li>`
      : `<li><a href="#w${r.w.key}/slide/${r.id}"><div class="rh">Week ${r.w.key} &middot; slides &middot; ${esc(r.title)}</div>${snippet(r.text, terms)}</a></li>`).join("")}</ul></div>`;
  main.scrollTop = 0;
}

/* ---------- notes export ---------- */
function exportNotes() {
  const parts = [];
  for (const w of [...C.weeks].sort((a, b) => a.key.localeCompare(b.key))) {
    const t = (store.get(`rl:notes:${w.key}`) || "").trim();
    if (t) parts.push(`## Week ${w.key}: ${w.topic}\n\n${t}\n`);
  }
  if (!parts.length) { alert("No notes saved yet (or this browser blocks local storage)."); return; }
  const blob = new Blob([`# My notes - Robot Learning 2026\n\n${parts.join("\n")}`], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "robot-learning-notes.md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ---------- lightbox ---------- */
const lb = { w: null, n: 0, max: 0 };
function openLightbox(w, n) {
  lb.w = w; lb.n = n; lb.max = Math.max(...w.slides.sections.map((s) => s.frames[1]));
  showFrame();
  $("#lightbox").hidden = false;
}
function showFrame() {
  const sec = lb.w.slides.sections.find((s) => lb.n >= s.frames[0] && lb.n <= s.frames[1]);
  $("#lightbox img").src = frameSrc(lb.w, lb.n);
  $("#lightbox figcaption").textContent = `Week ${lb.w.key} · frame ${lb.n} of ${lb.max}${sec ? ` · ${sec.title}` : ""}  (← → to step, Esc to close)`;
}
function stepFrame(d) { lb.n = Math.min(lb.max, Math.max(1, lb.n + d)); showFrame(); }

/* ---------- events ---------- */
document.addEventListener("click", (e) => {
  const t = e.target.closest("button, a");
  if (!t) return;
  if (t.matches(".tocitem, .ts")) { seek(t.closest(".vid").dataset.slot, +t.dataset.t); }
  else if (t.matches(".slidetime")) {
    const sec = $("#sec-lecture");
    if (sec) sec.open = true;
    seek("lecture", +t.dataset.t);
  }
  else if (t.matches(".thumb")) { openLightbox(weekByKey[currentWeek], +t.dataset.a); }
  else if (t.matches("[data-goto]")) {
    const el = $(`#${t.dataset.goto}`);
    el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  else if (t.matches(".lb-close")) { $("#lightbox").hidden = true; }
  else if (t.matches(".lb-prev")) { stepFrame(-1); }
  else if (t.matches(".lb-next")) { stepFrame(1); }
  if (t.closest("#sidebar nav")) document.body.classList.remove("menu-open");
});

let saveTimer = null;
document.addEventListener("input", (e) => {
  if (!e.target.matches("textarea.mynotes")) return;
  const ta = e.target;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const ok = store.set(`rl:notes:${ta.dataset.key}`, ta.value);
    ta.nextElementSibling.textContent = ok
      ? `Kept in this browser at ${new Date().toLocaleTimeString()} (not saved to the repo or online - export to keep a copy)`
      : "NOT kept: this browser blocks local storage, so this text will be lost on reload. Copy it somewhere safe.";
  }, 400);
});

document.addEventListener("keydown", (e) => {
  if (!$("#lightbox").hidden) {
    if (e.key === "Escape") $("#lightbox").hidden = true;
    if (e.key === "ArrowLeft") stepFrame(-1);
    if (e.key === "ArrowRight") stepFrame(1);
    return;
  }
  if (e.key === "/" && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
    e.preventDefault();
    $("#q").focus();
  }
});

$("#menuBtn").addEventListener("click", () => document.body.classList.toggle("menu-open"));

/* ---------- router ---------- */
function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  const parts = h.split("/");
  const head = parts[0];
  if (/^w\d\d$/.test(head) && weekByKey[head.slice(1)]) {
    renderWeek(weekByKey[head.slice(1)], parts.slice(1));
    markNav(head);
  } else if (head === "hw" && hwByKey[parts[1]]) {
    const hw = hwByKey[parts[1]];
    renderDoc(esc(hw.label), `Homework${hw.due ? ` &middot; due ${esc(hw.due)}` : ""}`, hw.html,
      `<p class="empty">From the official course repo: <a href="${hw.source}" target="_blank" rel="noopener">${esc(hw.source)}</a>. Images and links point there.</p>`);
    markNav(`hw/${hw.key}`);
  } else if (head === "about") {
    renderDoc("About these files", "learning/robot-learning", C.about_html);
    markNav("about");
  } else if (head === "search") {
    renderSearch(parts.slice(1).join("/"));
    markNav("");
  } else {
    renderHome();
    markNav("");
  }
}

renderSidebar();
window.addEventListener("hashchange", route);
route();
