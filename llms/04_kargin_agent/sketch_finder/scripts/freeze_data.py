"""Freeze the sketch data for video 07, so the notebook does not depend on the Kargin repo changing.

Reads kargin_eng.csv and data/youtube_metadata.csv from ~/OneDrive/Desktop/kargin_project and writes
llms/04_kargin_agent/data/kargin_sketches.csv. All 702 rows stay, duplicates and messy names included: cleaning them is
part of the lesson.

Run from the project folder (about 1 s):
    uv run python scripts/freeze_data.py
"""
import csv
import logging
import pathlib
import time

KARGIN_REPO = pathlib.Path.home() / "OneDrive" / "Desktop" / "kargin_project"
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "data" / "kargin_sketches.csv"
COLUMNS = [
    "sketch_id",
    "title",
    "url",
    "location",
    "language",
    "actors_raw",
    "duration_sec",
    "upload_date",
    "view_count",
    "duplicate_of",
    "dialogue",
]

(PROJECT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "logs" / "freeze_data.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def whole_number(value):
    """YouTube numbers arrive as '1234.0'. Empty stays empty."""
    return int(float(value)) if value else ""


def main():
    started = time.time()
    sketches = read_csv(KARGIN_REPO / "kargin_eng.csv")
    youtube_by_id = {row["video_id"]: row for row in read_csv(KARGIN_REPO / "data" / "youtube_metadata.csv")}

    missing = [row["video_id"] for row in sketches if row["video_id"] not in youtube_by_id]
    if missing:
        raise ValueError(f"{len(missing)} sketches have no YouTube metadata, first: {missing[:3]}")

    rows = []
    for sketch in sketches:
        youtube = youtube_by_id[sketch["video_id"]]
        rows.append({
            "sketch_id": int(sketch["id"]),
            "title": sketch["titles"],
            "url": f"https://youtu.be/{sketch['video_id']}",
            "location": sketch["location"].strip(),
            "language": sketch["languages"].strip(),
            "actors_raw": sketch["main_actors"].strip(),
            "duration_sec": whole_number(youtube["duration_sec"]),
            "upload_date": youtube["upload_date"],
            "view_count": whole_number(youtube["view_count"]),
            "duplicate_of": sketch["duplicate_of"].strip(),
            "dialogue": sketch["text"].strip(),
        })

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    duplicates = sum(1 for row in rows if row["duplicate_of"])
    with_dialogue = sum(1 for row in rows if row["dialogue"] and not row["duplicate_of"])
    log.info(f"wrote {OUTPUT_PATH.name}: {len(rows)} rows, {duplicates} duplicates, "
             f"{len(rows) - duplicates} unique, {with_dialogue} unique with dialogue, "
             f"{OUTPUT_PATH.stat().st_size / 1e6:.2f} MB, {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
