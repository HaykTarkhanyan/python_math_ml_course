import sqlite3

import pytest

from sketch_finder import config
from sketch_finder.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, read_sketch, run_sql, search_dialogue


def test_sql_returns_columns_and_rows(tiny_db):
    result = run_sql("SELECT count(*) AS n FROM sketches WHERE location = 'Հիվանդանոց'")
    assert result == {"columns": ["n"], "rows": [["1"]]}


def test_the_database_refuses_to_change(tiny_db):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        run_sql("DROP TABLE sketches")


def test_sql_returns_at_most_the_row_limit(tiny_db, monkeypatch):
    monkeypatch.setattr(config, "MAX_SQL_ROWS", 2)
    assert len(run_sql("SELECT sketch_id FROM sketches")["rows"]) == 2


def test_a_stem_finds_every_ending(tiny_db):
    # դպրոցի and դպրոցում are two forms of one word
    assert search_dialogue(["դպրոց"])["total_matches"] == 2


def test_all_patterns_must_match(tiny_db):
    found = search_dialogue(["նստ", "հղի"])
    assert [match["sketch_id"] for match in found["results"]] == [4]


def test_a_comma_is_just_a_comma(tiny_db):
    assert search_dialogue(["հղի, նստ"])["total_matches"] == 0


def test_skip_pages_through_results(tiny_db, monkeypatch):
    monkeypatch.setattr(config, "MAX_SEARCH_RESULTS", 1)
    first = search_dialogue(["դպրոց"])["results"]
    second = search_dialogue(["դպրոց"], skip=1)["results"]
    assert first[0]["sketch_id"] != second[0]["sketch_id"]


def test_an_overlong_pattern_is_refused(tiny_db):
    with pytest.raises(ValueError, match="shorter"):
        search_dialogue(["ա" * (config.MAX_PATTERN_CHARS + 1)])


def test_read_sketch(tiny_db):
    sketch = read_sketch(4)
    assert sketch["url"] == "https://youtu.be/id4"
    assert sorted(sketch["actors"]) == ["Հայկո", "Մկո"]


def test_a_missing_sketch_fails_loudly(tiny_db):
    with pytest.raises(ValueError, match="no sketch"):
        read_sketch(999)


def test_every_tool_has_a_matching_description():
    assert set(TOOL_SCHEMAS) == set(TOOL_FUNCTIONS)
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["function"]["name"] == name
        assert schema["function"]["description"]
