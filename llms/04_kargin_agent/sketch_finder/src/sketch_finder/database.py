"""The SQLite database: built from the frozen CSV, opened read-only by every tool."""
import csv
import re
import sqlite3
from collections import Counter

from sketch_finder import config


def split_actor_names(raw_names, known_names):
    """Split 'Հայկո , Մկո Հասմիկ' into three names. A space only splits when every word is a known name."""
    names = []
    for part in re.split(r"[,;|/]", raw_names):
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) > 1 and all(word in known_names for word in words):
            names.extend(words)
        else:
            names.append(part)
    return list(dict.fromkeys(names))


def find_known_names(sketches):
    """A one-word name seen in at least 5 sketches counts as a real actor name."""
    counts = Counter()
    for sketch in sketches:
        for part in re.split(r"[,;|/]", sketch["actors_raw"]):
            part = part.strip()
            if part and " " not in part:
                counts[part] += 1
    return {name for name, count in counts.items() if count >= 5}


def empty_to_none(value):
    return value if value != "" else None


def build_database(csv_path=None, db_path=None):
    """Build the SQLite file: one row per unique sketch, plus the actors in their own tables."""
    csv_path = csv_path or config.CSV_PATH
    db_path = db_path or config.DB_PATH

    # 1. read the sketches and drop the duplicates
    with open(csv_path, encoding="utf-8", newline="") as file:
        sketches = [row for row in csv.DictReader(file) if not row["duplicate_of"]]
    known_names = find_known_names(sketches)

    # 2. create the three tables
    connection = sqlite3.connect(db_path)
    connection.executescript("""
        DROP TABLE IF EXISTS sketch_actors;
        DROP TABLE IF EXISTS actors;
        DROP TABLE IF EXISTS sketches;
        CREATE TABLE sketches (
            sketch_id INTEGER PRIMARY KEY, title TEXT, url TEXT, location TEXT, language TEXT,
            duration_sec INTEGER, upload_date TEXT, view_count INTEGER, dialogue TEXT
        );
        CREATE TABLE actors (actor_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE sketch_actors (sketch_id INTEGER, actor_id INTEGER, PRIMARY KEY (sketch_id, actor_id));
    """)

    # 3. insert every sketch and link its actors
    for sketch in sketches:
        connection.execute(
            "INSERT INTO sketches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(sketch["sketch_id"]),
                sketch["title"],
                sketch["url"],
                empty_to_none(sketch["location"]),
                empty_to_none(sketch["language"]),
                int(sketch["duration_sec"]),
                sketch["upload_date"],
                int(sketch["view_count"]),
                empty_to_none(sketch["dialogue"]),
            ),
        )
        for name in split_actor_names(sketch["actors_raw"], known_names):
            connection.execute("INSERT OR IGNORE INTO actors (name) VALUES (?)", (name,))
            actor_id = connection.execute("SELECT actor_id FROM actors WHERE name = ?", (name,)).fetchone()[0]
            connection.execute("INSERT OR IGNORE INTO sketch_actors VALUES (?, ?)", (int(sketch["sketch_id"]), actor_id))
    connection.commit()

    # 4. report what is in it
    counts = {
        "sketches": connection.execute("SELECT count(*) FROM sketches").fetchone()[0],
        "with_dialogue": connection.execute("SELECT count(*) FROM sketches WHERE dialogue IS NOT NULL").fetchone()[0],
        "actors": connection.execute("SELECT count(*) FROM actors").fetchone()[0],
    }
    connection.close()
    return counts


def open_read_only():
    """Every tool connects like this, so even a bad query from the model cannot change the data."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"{config.DB_PATH} does not exist: run `sketch-finder build-db` first")
    return sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
