"""A tiny database for the tests: no model calls, no network, a few milliseconds."""
import csv

import pytest

from sketch_finder import config
from sketch_finder.database import build_database

COLUMNS = ["sketch_id", "title", "url", "location", "language", "actors_raw", "duration_sec", "upload_date",
           "view_count", "duplicate_of", "dialogue"]


def sketch(sketch_id, actors, dialogue, location="Տուն", views=100, duplicate_of=""):
    return {
        "sketch_id": sketch_id, "title": f"Kargin Haghordum sketch {sketch_id}", "url": f"https://youtu.be/id{sketch_id}",
        "location": location, "language": "հայերեն", "actors_raw": actors, "duration_sec": 200,
        "upload_date": "20121228", "view_count": views, "duplicate_of": duplicate_of, "dialogue": dialogue,
    }


SKETCHES = [
    sketch(1, "Հայկո, Մկո", "բարև ձեզ, դպրոցի տնօրենն եմ", views=500),
    sketch(2, "Հայկո, Մկո", "դպրոցում ուսուցիչը չեկավ", location="Հիվանդանոց"),
    sketch(3, "Հայկո, Մկո", "մառշուտկեն լիքն ա, տեղ չկա նստելու"),
    sketch(4, "Հայկո, Մկո", "հղի եմ, տեղ տվեք նստեմ"),
    sketch(5, "Հայկո, Մկո", "", location=""),
    sketch(6, "Հայկո Մկո Հասմիկ", "ձուկ չի բռնվում"),
    sketch(7, "Հայկո, Մկո", "կրկնօրինակ", duplicate_of="1"),
]


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    csv_path = tmp_path / "sketches.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(SKETCHES)
    monkeypatch.setattr(config, "CSV_PATH", csv_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "logs" / "calls.jsonl")
    return build_database()
