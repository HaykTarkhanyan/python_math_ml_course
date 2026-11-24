from sketch_finder.database import split_actor_names
from sketch_finder.tools import run_sql


def test_split_on_commas_and_known_names():
    known = {"Հայկո", "Մկո", "Հասմիկ"}
    assert split_actor_names("Հայկո , Մկո Հասմիկ", known) == ["Հայկո", "Մկո", "Հասմիկ"]


def test_unknown_words_stay_together():
    # "փոքր երեխա" (a small child) is a role, not two actors
    assert split_actor_names("Մկո, փոքր երեխա", {"Մկո"}) == ["Մկո", "փոքր երեխա"]


def test_repeated_names_count_once():
    assert split_actor_names("Մկո, Հայկո, Մկո", {"Մկո", "Հայկո"}) == ["Մկո", "Հայկո"]


def test_build_drops_the_duplicate_and_the_empty_dialogue(tiny_db):
    # 7 rows, one is a duplicate; sketch 5 has no dialogue
    assert tiny_db["sketches"] == 6
    assert tiny_db["with_dialogue"] == 5


def test_a_space_splits_only_when_every_word_is_known(tiny_db):
    # Հայկո and Մկո appear alone in 5 sketches, Հասմիկ never does, so "Հայկո Մկո Հասմիկ" is kept whole
    rows = run_sql("SELECT name FROM actors JOIN sketch_actors USING (actor_id) WHERE sketch_id = 6")["rows"]
    assert rows == [["Հայկո Մկո Հասմիկ"]]
    assert tiny_db["actors"] == 3
