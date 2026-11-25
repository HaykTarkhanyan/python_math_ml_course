"""Throwaway prototype of the Kargin agent: does SQL work, do exact quotes work, does regex fail on vague memories?

Run from the repo root with the ma venv. Expected: ~3 min, about $0.05 on gpt-4o-mini, hard stop at $0.20.
"""
import csv
import json
import os
import pathlib
import random
import re
import sqlite3
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
SEED = 509
BUDGET_USD = 0.20
MAX_STEPS = 8
MODEL = "openai/gpt-4o-mini"
SCRATCH = pathlib.Path(__file__).parent
KARGIN = pathlib.Path.home() / "OneDrive" / "Desktop" / "kargin_project"
DB_PATH = SCRATCH / "kargin_proto.db"
RESULTS_PATH = SCRATCH / "kargin_proto_results.json"


def clean_actor_names(raw, known_names):
    names = []
    for part in re.split(r"[,;|/]", raw or ""):
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) > 1 and all(word in known_names for word in words):
            names.extend(words)
        else:
            names.append(part)
    return list(dict.fromkeys(names))


def build_database():
    with (KARGIN / "kargin_eng.csv").open(encoding="utf-8", newline="") as f:
        table = [r for r in csv.DictReader(f) if not (r["duplicate_of"] or "").strip()]
    with (KARGIN / "data" / "youtube_metadata.csv").open(encoding="utf-8", newline="") as f:
        youtube = {r["video_id"]: r for r in csv.DictReader(f)}

    single_words = {}
    for row in table:
        for part in re.split(r"[,;|/]", row["main_actors"] or ""):
            part = part.strip()
            if part and " " not in part:
                single_words[part] = single_words.get(part, 0) + 1
    known_names = {name for name, count in single_words.items() if count >= 5}

    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        DROP TABLE IF EXISTS sketch_actors; DROP TABLE IF EXISTS actors; DROP TABLE IF EXISTS sketches;
        CREATE TABLE sketches (sketch_id INTEGER PRIMARY KEY, title TEXT, url TEXT, location TEXT, language TEXT,
                               duration_sec INTEGER, upload_date TEXT, view_count INTEGER, dialogue TEXT);
        CREATE TABLE actors (actor_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE sketch_actors (sketch_id INTEGER, actor_id INTEGER, PRIMARY KEY (sketch_id, actor_id));
    """)
    for row in table:
        meta = youtube.get(row["video_id"], {})
        db.execute("INSERT INTO sketches VALUES (?,?,?,?,?,?,?,?,?)", (
            int(row["id"]), row["titles"], f"https://youtu.be/{row['video_id']}", row["location"] or None,
            row["languages"] or None, int(float(meta["duration_sec"])) if meta.get("duration_sec") else None,
            meta.get("upload_date") or None, int(float(meta["view_count"])) if meta.get("view_count") else None,
            row["text"] or None,
        ))
        for name in clean_actor_names(row["main_actors"], known_names):
            db.execute("INSERT OR IGNORE INTO actors (name) VALUES (?)", (name,))
            actor_id = db.execute("SELECT actor_id FROM actors WHERE name = ?", (name,)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO sketch_actors VALUES (?,?)", (int(row["id"]), actor_id))
    db.commit()
    print("db:", db.execute("SELECT count(*) FROM sketches").fetchone()[0], "sketches,",
          db.execute("SELECT count(*) FROM actors").fetchone()[0], "actor names,",
          db.execute("SELECT count(*) FROM sketches WHERE dialogue IS NOT NULL").fetchone()[0], "with dialogue", flush=True)
    return db, table


def run_sql(query):
    connection = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(30)
        return {"columns": columns, "rows": [[str(v)[:120] for v in row] for row in rows]}
    finally:
        connection.close()


def search_dialogue(pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    connection = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    rows = connection.execute("SELECT sketch_id, title, dialogue FROM sketches").fetchall()
    connection.close()
    matches = []
    for sketch_id, title, dialogue in rows:
        text = f"{title}\n{dialogue or ''}"
        found = regex.search(text)
        if found:
            matches.append({"sketch_id": sketch_id, "title": title[:50],
                            "snippet": text[max(0, found.start() - 50):found.end() + 50].replace("\n", " ")})
    return {"total_matches": len(matches), "first_matches": matches[:8]}


def read_sketch(sketch_id):
    connection = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    row = connection.execute("SELECT title, dialogue FROM sketches WHERE sketch_id = ?", (sketch_id,)).fetchone()
    connection.close()
    if row is None:
        return {"error": f"no sketch {sketch_id}"}
    return {"title": row[0], "dialogue": (row[1] or "")[:1500]}


TOOLS = [
    {"type": "function", "function": {"name": "run_sql", "description": "Run one read-only SQLite query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_dialogue",
        "description": "Search sketch titles and Armenian dialogue with a Python regex. Returns count and up to 8 snippets.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_sketch", "description": "Read one sketch's title and dialogue.",
        "parameters": {"type": "object", "properties": {"sketch_id": {"type": "integer"}}, "required": ["sketch_id"]}}},
]
TOOL_FUNCTIONS = {"run_sql": run_sql, "search_dialogue": search_dialogue, "read_sketch": read_sketch}
SYSTEM = """You answer questions about Kargin Haghordum, an Armenian sketch comedy show, using the tools.
Tables: sketches(sketch_id, title, url, location, language, duration_sec, upload_date, view_count, dialogue),
actors(actor_id, name), sketch_actors(sketch_id, actor_id). Names, locations and dialogue are in Armenian
(locations include Տուն, Հիվանդանոց, Գրասենյակ; actors include Հայկո, Մկո, Աշոտ, Հասմիկ). Dialogue is colloquial and
words change endings, so search stems. Try several searches when one fails. Check a candidate with read_sketch.
End with a final line exactly like: ANSWER: <value>. For "which sketch" the value is the sketch_id."""


def run_agent(client, question, spent):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls, cost = [], 0.0
    for step in range(MAX_STEPS):
        if spent + cost > BUDGET_USD:
            raise RuntimeError(f"budget of ${BUDGET_USD} reached")
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0, max_tokens=500,
            extra_body={"provider": {"order": ["openai"], "allow_fallbacks": False}, "usage": {"include": True}},
        )
        call_cost = getattr(response.usage, "cost", None)
        if call_cost is None:
            raise RuntimeError("OpenRouter returned no cost")
        cost += call_cost
        message = response.choices[0].message
        if not message.tool_calls:
            found = re.findall(r"ANSWER:\s*(.+)", message.content or "")
            return {"answer": found[-1].strip() if found else None, "steps": step + 1, "calls": calls, "cost": cost}
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            calls.append({"tool": tool_call.function.name, "arguments": arguments})
            try:
                result = TOOL_FUNCTIONS[tool_call.function.name](**arguments)
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)})
    return {"answer": None, "steps": MAX_STEPS, "calls": calls, "cost": cost, "hit_step_cap": True}


def make_questions(db, table):
    rng = random.Random(SEED)
    questions = []
    one = lambda sql: db.execute(sql).fetchone()[0]
    questions += [
        ("sql", "How many sketches are set in a hospital (location Հիվանդանոց)?",
         str(one("SELECT count(*) FROM sketches WHERE location = 'Հիվանդանոց'"))),
        ("sql", "Which sketch has the most YouTube views? Give its sketch_id.",
         str(one("SELECT sketch_id FROM sketches ORDER BY view_count DESC LIMIT 1"))),
        ("sql", "In how many sketches does Աշոտ appear?",
         str(one("SELECT count(*) FROM sketch_actors JOIN actors USING (actor_id) WHERE name = 'Աշոտ'"))),
        ("sql", "Which location is the most common, not counting empty values?",
         one("SELECT location FROM sketches WHERE location IS NOT NULL GROUP BY location ORDER BY count(*) DESC LIMIT 1")),
    ]
    with_dialogue = [r for r in table if (r["text"] or "").strip()]
    quotes = 0
    for row in rng.sample(with_dialogue, len(with_dialogue)):
        words = re.findall(r"[Ա-և]+", row["text"])
        if len(words) < 30:
            continue
        start = rng.randrange(5, len(words) - 8)
        phrase = r"\W+".join(words[start:start + 5])
        hits = [r for r in with_dialogue if re.search(phrase, r["text"])]
        if len(hits) == 1:
            questions.append(("quote", f'Which sketch contains the line "{" ".join(words[start:start + 5])}"?', row["id"]))
            quotes += 1
        if quotes == 4:
            break
    annotations = {}
    for path in (KARGIN / "data" / "text_annotations").glob("*.json"):
        note = json.loads(path.read_text(encoding="utf-8"))
        annotations[note["video_id"]] = note["annotation"]["summary_en"]
    for row in rng.sample([r for r in with_dialogue if r["video_id"] in annotations], 12):
        questions.append(("vague", f"I remember a sketch but not its name: {annotations[row['video_id']]} Which sketch is it?", row["id"]))
    return questions


def main():
    load_dotenv(".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    db, table = build_database()
    questions = make_questions(db, table)
    print(f"{len(questions)} questions", flush=True)
    results, spent, started = [], 0.0, time.time()
    for index, (kind, question, expected) in enumerate(questions, 1):
        outcome = run_agent(client, question, spent)
        spent += outcome["cost"]
        answer = outcome["answer"] or ""
        correct = str(expected) == re.sub(r"[^\wԱ-և]", "", answer) or (kind == "sql" and str(expected) in answer)
        results.append({"kind": kind, "question": question, "expected": expected, "correct": correct, **outcome})
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        elapsed = time.time() - started
        print(f"{index}/{len(questions)} {kind}: {'OK ' if correct else 'NO '} expected={str(expected)[:15]} got={answer[:15]} "
              f"steps={outcome['steps']} tools={len(outcome['calls'])} ${spent:.4f} {elapsed:.0f}s, "
              f"~{elapsed / index * (len(questions) - index):.0f}s left", flush=True)
    for kind in ("sql", "quote", "vague"):
        group = [r for r in results if r["kind"] == kind]
        print(f"{kind}: {sum(r['correct'] for r in group)}/{len(group)} correct, mean tool calls {sum(len(r['calls']) for r in group) / len(group):.1f}")
    print(f"total ${spent:.4f}, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
