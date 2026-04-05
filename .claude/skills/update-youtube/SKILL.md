---
name: update-youtube
description: >
  Fetch recent YouTube videos from the Metric Academy channel, update the
  video registry (_meta/youtube_channel.md), add video + notes links to
  math/*.qmd homework pages, and optionally commit.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Glob
  - Grep
  - WebFetch
user-invocable: true
---

# Update YouTube Videos & Course Links

Fetches recent videos from the Metric Academy YouTube channel and updates
all relevant files in the repo.

## Key files

| File | Purpose |
|------|---------|
| `_meta/youtube_channel.md` | Master registry of all videos (178+ entries) |
| `math/2X_stat_*.qmd` | Homework pages with video/slide/notes links |
| `math/Lectures/stat/*_notes.pdf` | Lecture notes PDFs (add links where they exist) |

## Channel info

- **Channel:** @MetricAcademy
- **Channel ID:** `UC5BavugdnUkti3Es7-deZDg`
- **RSS feed:** `https://www.youtube.com/feeds/videos.xml?channel_id=UC5BavugdnUkti3Es7-deZDg`

## Step 1: Fetch recent videos via RSS

Use WebFetch on the RSS feed URL:
```
https://www.youtube.com/feeds/videos.xml?channel_id=UC5BavugdnUkti3Es7-deZDg
```

Extract from each entry: **title**, **video ID** (from `yt:videoId`), **published date**, **URL**.

## Step 2: Compare with existing registry

Read `_meta/youtube_channel.md` and find the `Last scraped` date. Identify
videos published **after** that date. These are the new ones to add.

Also check for videos that exist in the feed but are missing from the
registry (may have been uploaded just before the last scrape).

Categorize each new video:
- Title contains "Lecture" or "Դ" (Armenian for lesson) → **Math/ML Lectures** table
- Title contains "Practical" or "Գordelays" → **Math/ML Practicals** table
- Title contains "Python" → **Python** tables
- Otherwise → ask the user

## Step 3: Update the registry

In `_meta/youtube_channel.md`:
1. Update the `Last scraped` date to today
2. Update `Total Videos` count
3. Add new rows to the appropriate table(s)
4. For new entries where views/duration are unknown, use `—` as placeholder
5. Update section headers if counts changed (e.g., "Math/ML Lectures (47 videos)")

## Step 4: Update homework .qmd files

The math homework files follow this pattern for resources:

```markdown
## Դasakhosuration (Armenian for "Lecture")
- [📺 Դ - Topic](https://youtu.be/VIDEO_ID)
- [🎞️ Ս - NN stat](Lectures/stat/NN_stat.pdf), [📝 Notes](Lectures/stat/NN_stat_notes.pdf)

## Գdelays (Armenian for "Practical")
- [🛠️📺 Գ NN - Topic](https://youtu.be/VIDEO_ID)
```

**IMPORTANT:** Armenian UTF-8 characters get corrupted in the Edit tool.
Always use Python (via Bash) for editing .qmd files:

```python
python -c "
import pathlib
f = pathlib.Path('math/FILE.qmd')
c = f.read_text(encoding='utf-8')
# ... do replacements ...
f.write_text(c, encoding='utf-8')
"
```

### Mapping: which video goes to which .qmd

| Slide deck | Homework .qmd |
|------------|---------------|
| 01-02 stat | `21_stat_fundamentals.qmd` |
| 03-04 stat | `22_stat_estimators.qmd` |
| 05-06 stat | `23_stat_mle_map.qmd` |
| 07-08 stat | `24_stat_confidence_intervals.qmd` |
| 09-10 stat | `25_stat_hypothesis_testing.qmd` |

### Check for notes PDFs

```bash
ls math/Lectures/stat/*_notes.pdf
```

For each slide PDF linked in a .qmd, if a corresponding `_notes.pdf`
exists and isn't already linked, add it inline:

```markdown
- [🎞️ Slides - NN](Lectures/stat/NN_stat.pdf), [📝 Notes](Lectures/stat/NN_stat_notes.pdf)
```

## Step 5: Report & confirm

Show the user:
1. How many new videos were found
2. Which files were updated
3. Ask if they want to commit and push

## Notes

- The RSS feed returns the 15 most recent videos. For older videos, use
  the `pytubefix` library via `python/mini_projects/youtube_playlist/youtube_info.py`
- Video titles are bilingual (Armenian + English). The Armenian prefix
  "Դ" means "Lesson", "Գ" means "Practical"
- Lecture numbering in YouTube (e.g., Lecture 46) does NOT match slide
  deck numbering (e.g., 08_stat). The mapping is in the table above.
