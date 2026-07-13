# Deep Learning for NLP [s26] — LMU Munich Moodle archive

Snapshot of LMU Moodle course **s26_dl4nlp** (id=44041), Summer Semester 2026.
Source: <https://moodle.lmu.de/course/view.php?id=44041>
Seen: 2026-07-12

> Mirrors the layout of `ml/deep_learning/moodle_s26_course/`. **Structural difference from
> the DL course:** the lecture slides are not per-week Moodle PDFs — they live on GitHub
> (`slds-lmu/lecture_dl4nlp`) and as web chapters. Moodle only hosts downloadable folders
> (Slides, Assignments), assignment submissions, exam info pages, and forums.

## People

- **[Lecture]** Hinrich Schütze — <http://cis.lmu.de/~hs/teach/26s/dl4nlp/>
- **[Lecture]** Christian Heumann — chris@stat.uni-muenchen.de
- **[Lecture, Orga Stats]** Matthias Aßenmacher — matthias@stat.uni-muenchen.de
- **[Exercise, Orga CL]** Yihong Liu — yihong@cis.lmu.de
- **[Tutor]** Adrian Mülthaler — Adrian.Muelthaler@campus.lmu.de

Communication: email **both** Yihong and Matthias for general orga; for specific questions,
CL → Yihong, Stats → Matthias.

## Time & location (Oettingenstr. 67)

| | When | Room |
|---|---|---|
| **Lecture** | Wednesday, 12:15–13:45 | 151 |
| **Exercise** | Tuesday, 16:15–17:45 | 057 |
| **Consultation (w/ Tutor)** | Tuesday, 14:15–15:45 | C 105 |

Course language: English. Target audience: advanced Master students from Computational
Linguistics, Statistics & Data Science, ESG Data Science (others welcome).

## Exam

- **Date:** 29.07.2026, 10:00–12:00
- **Place:** Oettingenstr. 67, B 001
- **Format:** written, in-person, 90 minutes — **6 ECTS**
- **Registration:** Stats & DS / all others 01.06.26–29.06.26; CL 15.06.26–29.06.26
- **Inspection:** 06.08.2026, 10:30 am
- **Retake:** none this year (already the second DL4NLP exam this year); next regular exam with retake in summer 2027
- **Past exams:** SLDS exam archive — <https://moodle.lmu.de/course/view.php?id=32068> (without solutions, from WT 23/24 on)

## Lecture material (weekly links)

Slide repo (source): <https://github.com/slds-lmu/lecture_dl4nlp/tree/main/slides>
→ already cloned locally at `../../../_reference/lecture_dl4nlp/` (repo root `_reference/`, gitignored).

Local slide copies in this archive:
- `slides_tex/` — `.tex` sources for **weeks 1–8** (chapters 01–08), text-readable.
- `slides_weeks1-8/` — **36 compiled slide PDFs** for weeks 1–8 (viewable form of the above).
- `slides_weeks9-13/` — **9 PDFs + readable `.md`** for weeks 9–13 (Schütze's decks).

| Week | Link |
|---|---|
| 1 | <https://slds-lmu.github.io/dl4nlp/chapters/01_introduction/> |
| 2 | <https://slds-lmu.github.io/dl4nlp/chapters/02_dl_basics/> |
| 3 | <https://slds-lmu.github.io/dl4nlp/chapters/03_transformer/> |
| 4 | <https://slds-lmu.github.io/dl4nlp/chapters/04_bert/> |
| 5 | <https://slds-lmu.github.io/dl4nlp/chapters/05_bert_based/> |
| 6 | <https://slds-lmu.github.io/dl4nlp/chapters/06_post_bert_t5/> |
| 7 | <https://slds-lmu.github.io/dl4nlp/chapters/07_gpt/> |
| 8 | <https://slds-lmu.github.io/dl4nlp/chapters/08_decoding/> |
| 9–13 | <http://cis.lmu.de/~hs/teach/26s/dl4nlp/> (Schütze's page) — decks downloaded to `slides_weeks9-13/` (9 PDFs) |

## Moodle activities (downloadable / to archive)

- **Slides Folder** — <https://moodle.lmu.de/mod/folder/view.php?id=2366834>
- **Assignments Folder** — <https://moodle.lmu.de/mod/folder/view.php?id=2366835>
- **Assignment submissions 1–5** — `mod/assign` ids 2366836–2366840 (due dates: 04.05, 18.05, 01.06, 15.06, 29.06.2026)
- **Detail pages:** Course Format (`mod/page` 2366831) · Exam (2366832) · Timetable (2366833)
- **Forums:** Ankündigungen (`mod/forum` 2366828) · Forum (2366829)

## Related local resources (do not duplicate)

- `../../../_reference/lecture_dl4nlp/` — cloned slide-source repo (84 `.tex` in `slides/`), the source-of-truth for slide content.
- `misc/dl4nlp/` — the instructor's **own** 18 DL4NLP decks (`01_pre_transformer` … `18_reinforcement_learning`), separate from this LMU course.

## Plan for "similar stuff" (TODO — awaiting confirmation)

1. Download **Slides Folder** + **Assignments Folder** as ZIPs → `slides_moodle/`, `assignments/`.
2. Grab **past exams** from the SLDS exam archive (and the already-downloaded `Exam-Questions-DL-Summer-2025-.pdf`) → `exams/`, then transcribe to readable `.md` with visual checks (as done for the DL mock exams).
3. Point `slides_tex/` at the existing `_reference/lecture_dl4nlp/slides` sources (or copy the relevant `.tex`, mirroring `ml/deep_learning/moodle_s26_course/slides_tex/`).
4. Save the 3 detail pages (Course Format / Exam / Timetable) text to `.md`.
