"""Write the known-answer question set for video 07. The RAG practical reruns the same file.

SQL answers and exact quotes are computed from data/kargin_sketches.csv. The hazy descriptions are written by hand for
sketches NOT used while tuning the agent's prompt (the 12 in archive/2026-11-24_regex_premise), so they are a fair test.

Writes data/questions.json and 07_kargin_questions_REVIEW.md (for the instructor's check).
Run from the project folder (about 1 s; needs pandas from the maintainer group):
    uv run --group maintainer python scripts/build_questions.py
"""
import collections
import json
import logging
import pathlib
import random
import re

import pandas as pd

SEED = 509
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "kargin_sketches.csv"
QUESTIONS_PATH = PROJECT_DIR / "data" / "questions.json"
REVIEW_PATH = PROJECT_DIR.parent / "07_kargin_questions_REVIEW.md"   # the lesson folder, for the instructor
QUOTE_SKETCHES = [408, 155, 223, 469, 73]
QUOTE_WORDS = 5
HAZY = {
    673: "A guy with a tiny salary somehow owns a very fancy car, and the police want to know how.",
    383: "A driver gets stuck on the road and the one who helps him is the last person you would expect.",
    367: "A teacher asks a clever science question and the student answers with a joke about the economy.",
    522: "Someone is offered a choice of what he wants most in life and regrets what he picks.",
    146: "A professor wants to fail everybody but is told from above to pass them quickly.",
    616: "A father and son go fishing and catch nothing, and it turns out to be because of the food.",
    395: "An Armenian living abroad pays a restaurant to treat him badly so that he feels at home.",
    595: "An old man keeps getting woken up by someone calling him.",
    451: "An army officer tries to show a smarter way to punish soldiers and it backfires.",
    544: "A secretary on the phone thinks she is getting fired because of some furniture.",
}

(PROJECT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "logs" / "build_questions.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def split_actor_names(raw, known_names):
    """Split 'Հայկո , Մկո Հասմիկ' into three names. A space only splits when every word is a known name."""
    names = []
    for part in re.split(r"[,;|/]", raw):
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) > 1 and all(word in known_names for word in words):
            names.extend(words)
        else:
            names.append(part)
    return list(dict.fromkeys(names))


def count_actor(sketches, actor):
    single_names = collections.Counter(
        part.strip()
        for raw in sketches["actors_raw"].dropna()
        for part in re.split(r"[,;|/]", raw)
        if part.strip() and " " not in part.strip()
    )
    known_names = {name for name, count in single_names.items() if count >= 5}
    return sum(actor in split_actor_names(raw, known_names) for raw in sketches["actors_raw"].dropna())


def sql_questions(sketches):
    locations = sketches["location"].value_counts()
    most_viewed = int(sketches.loc[sketches["view_count"].idxmax(), "sketch_id"])
    return [
        {"kind": "sql", "question": "How many sketches take place in a hospital?",
         "answer": str(int((sketches["location"] == "Հիվանդանոց").sum())), "answer_type": "number"},
        {"kind": "sql", "question": "Which sketch has the most YouTube views?",
         "answer": str(most_viewed), "answer_type": "sketch_id"},
        {"kind": "sql", "question": "How many sketches does the actor Ashot (Աշոտ) appear in?",
         "answer": str(count_actor(sketches, "Աշոտ")), "answer_type": "number"},
        {"kind": "sql", "question": "What is the most common location of a sketch? Answer with the value in the table.",
         "answer": locations.index[0], "answer_type": "text"},
        {"kind": "sql", "question": "How many sketches are longer than five minutes?",
         "answer": str(int((sketches["duration_sec"] > 300).sum())), "answer_type": "number"},
    ]


def quote_questions(sketches):
    random_generator = random.Random(SEED)
    with_dialogue = sketches.dropna(subset=["dialogue"])
    questions = []
    for sketch_id in QUOTE_SKETCHES:
        dialogue = with_dialogue.loc[with_dialogue["sketch_id"] == sketch_id, "dialogue"].iloc[0]
        words = re.findall(r"[Ա-և]+", dialogue)
        positions = list(range(5, len(words) - QUOTE_WORDS))
        random_generator.shuffle(positions)
        for start in positions:
            phrase = words[start:start + QUOTE_WORDS]
            pattern = r"\W+".join(phrase)
            if with_dialogue["dialogue"].str.contains(pattern, regex=True).sum() == 1:
                questions.append({"kind": "quote", "question": f'Which sketch has the line "{" ".join(phrase)}"?',
                                  "answer": str(sketch_id), "answer_type": "sketch_id"})
                break
        else:
            raise ValueError(f"no unique phrase found in sketch {sketch_id}")
    return questions


def hazy_questions():
    return [
        {"kind": "hazy", "question": f"I remember a sketch but not its name: {memory} Which sketch is it?",
         "answer": str(sketch_id), "answer_type": "sketch_id"}
        for sketch_id, memory in HAZY.items()
    ]


def write_review(questions, sketches):
    urls = dict(zip(sketches["sketch_id"].astype(str), sketches["url"]))
    labels = {"sql": "Database", "quote": "Exact quote", "hazy": "Hazy memory"}
    lines = [
        "# Video 07: the 20 test questions, for your check",
        "",
        "These are the known-answer questions the agent runs in class. The RAG practical reruns the same file.",
        "Select any text and press C to comment: a wording no fan would use, a sketch you would not show in class, a wrong answer.",
        "",
    ]
    for kind, label in labels.items():
        lines += [f"## {label}", "", "| # | Question | Right answer | Video |", "|---|---|---|---|"]
        for question in questions:
            if question["kind"] != kind:
                continue
            text = question["question"].replace("I remember a sketch but not its name: ", "").replace(" Which sketch is it?", "")
            link = f"[watch]({urls[question['answer']]})" if question["answer_type"] == "sketch_id" else ""
            lines.append(f"| {question['id']} | {text} | {question['answer']} | {link} |")
        lines.append("")
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    table = pd.read_csv(DATA_PATH, dtype={"upload_date": str})
    sketches = table[table["duplicate_of"].isna()].copy()
    missing = [sketch_id for sketch_id in HAZY if sketch_id not in set(sketches.dropna(subset=["dialogue"])["sketch_id"])]
    if missing:
        raise ValueError(f"hazy sketches without dialogue or duplicated: {missing}")

    questions = sql_questions(sketches) + quote_questions(sketches) + hazy_questions()
    for number, question in enumerate(questions, 1):
        question["id"] = f"q{number:02d}"
    QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    write_review(questions, sketches)
    log.info(f"wrote {len(questions)} questions: {dict(collections.Counter(q['kind'] for q in questions))}")


if __name__ == "__main__":
    main()
