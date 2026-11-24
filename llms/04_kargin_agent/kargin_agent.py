"""The Kargin agent of video 07: the database, the tools, and the agent loop.

The notebook shows this code cell by cell (its builder copies the functions from here, so the two cannot drift).
kargin_server.py and py_src/run_model_comparison.py import it.

Needs OPENROUTER_API_KEY in the repository's .env, and TAVILY_API_KEY for web_search. Keys are read, never printed.
"""
import csv
import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

LESSON_DIR = Path(__file__).resolve().parent
REPO_ROOT = LESSON_DIR.parents[1]
CSV_PATH = LESSON_DIR / "data" / "kargin_sketches.csv"
DB_PATH = LESSON_DIR / "data" / "kargin.db"

MAX_SQL_ROWS = 30
MAX_CELL_CHARS = 200
MAX_SEARCH_RESULTS = 10
MAX_PATTERN_CHARS = 200
MAX_DIALOGUE_CHARS = 2000

DEFAULT_MODEL = "openai/gpt-6.1-terra"
MODELS = {
    "openai/gpt-6.1-terra": {"provider": "openai", "temperature": False, "reasoning": True},
    "openai/gpt-6.1-sol": {"provider": "openai", "temperature": False, "reasoning": True},
    "deepseek/deepseek-v4.1-flash": {"provider": "together", "temperature": True, "reasoning": True},
    "google/gemini-3-flash-preview": {"provider": "google-ai-studio", "temperature": True, "reasoning": True},
    "openai/gpt-4o-mini": {"provider": "openai", "temperature": True, "reasoning": False},
}


# ------------------------------------------------------------------------------------------ the database
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


def build_database(csv_path=CSV_PATH, db_path=DB_PATH):
    """Build the SQLite file: one row per unique sketch, plus the actors in their own tables."""
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
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)


# ------------------------------------------------------------------------------------------ the tools
def run_sql(query: str) -> dict:
    """Run one read-only SQL query on the Kargin database. Returns the columns and at most 30 rows."""
    connection = open_read_only()
    try:
        cursor = connection.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(MAX_SQL_ROWS)
    finally:
        connection.close()
    short_rows = [[str(value)[:MAX_CELL_CHARS] for value in row] for row in rows]
    return {"columns": columns, "rows": short_rows}


def search_dialogue(all_patterns: list[str], skip: int = 0) -> dict:
    """Find sketches whose title or Armenian dialogue match EVERY regex in all_patterns (1 to 3 short patterns).
    Returns how many matched and 10 snippets. Use skip to see the next 10."""
    # 1. compile the patterns
    for pattern in all_patterns:
        if len(pattern) > MAX_PATTERN_CHARS:
            raise ValueError(f"pattern longer than {MAX_PATTERN_CHARS} characters: use a shorter one")
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in all_patterns]

    # 2. read every sketch
    connection = open_read_only()
    sketches = connection.execute("SELECT sketch_id, title, dialogue FROM sketches").fetchall()
    connection.close()

    # 3. keep the sketches where every pattern matches
    matches = []
    for sketch_id, title, dialogue in sketches:
        text = f"{title}\n{dialogue or ''}"
        found = [regex.search(text) for regex in regexes]
        if all(found):
            first = found[0]
            snippet = text[max(0, first.start() - 50):first.end() + 50].replace("\n", " ")
            matches.append({"sketch_id": sketch_id, "title": title[:50], "snippet": snippet})
    return {"total_matches": len(matches), "results": matches[skip:skip + MAX_SEARCH_RESULTS]}


def read_sketch(sketch_id: int) -> dict:
    """Read one sketch: its title, YouTube link, location, actors and dialogue."""
    connection = open_read_only()
    sketch = connection.execute(
        "SELECT title, url, location, dialogue FROM sketches WHERE sketch_id = ?",
        (sketch_id,),
    ).fetchone()
    actors = connection.execute(
        "SELECT name FROM actors JOIN sketch_actors USING (actor_id) WHERE sketch_id = ?",
        (sketch_id,),
    ).fetchall()
    connection.close()
    if sketch is None:
        raise ValueError(f"there is no sketch {sketch_id}")
    title, url, location, dialogue = sketch
    return {
        "title": title,
        "url": url,
        "location": location,
        "actors": [actor[0] for actor in actors],
        "dialogue": (dialogue or "")[:MAX_DIALOGUE_CHARS],
    }


def web_search(query: str) -> list[dict]:
    """Search the web for facts that are not in the database. Returns the top 3 pages, each with a short text."""
    from tavily import TavilyClient
    response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=3)
    return [
        {"title": page["title"], "url": page["url"], "text": page["content"][:500]}
        for page in response["results"]
    ]


TOOL_FUNCTIONS = {
    "run_sql": run_sql,
    "search_dialogue": search_dialogue,
    "read_sketch": read_sketch,
    "web_search": web_search,
}
ALL_TOOL_NAMES = tuple(TOOL_FUNCTIONS)
REGEX_TOOL_NAMES = ("search_dialogue", "read_sketch")


def tool_schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = {
    "run_sql": tool_schema(
        "run_sql",
        run_sql.__doc__,
        {"query": {"type": "string", "description": "One SQLite SELECT query"}},
        ["query"],
    ),
    "search_dialogue": tool_schema(
        "search_dialogue",
        search_dialogue.__doc__,
        {
            "all_patterns": {"type": "array", "items": {"type": "string"}, "description": "1 to 3 regexes, all must match"},
            "skip": {"type": "integer", "description": "How many results to skip, for paging"},
        },
        ["all_patterns"],
    ),
    "read_sketch": tool_schema(
        "read_sketch",
        read_sketch.__doc__,
        {"sketch_id": {"type": "integer"}},
        ["sketch_id"],
    ),
    "web_search": tool_schema(
        "web_search",
        web_search.__doc__,
        {"query": {"type": "string"}},
        ["query"],
    ),
}

SYSTEM_PROMPT = """You answer questions about Kargin Haghordum, an Armenian sketch comedy show, using the tools.

The database has three tables:
  sketches(sketch_id, title, url, location, language, duration_sec, upload_date, view_count, dialogue)
  actors(actor_id, name)
  sketch_actors(sketch_id, actor_id)
Names, locations and dialogue are in Armenian. Locations include Տուն (home), Հիվանդանոց (hospital), Գրասենյակ (office).
Actors include Հայկո, Մկո, Աշոտ, Հասմիկ.

Use SQL for counts, rankings and anything about actors, locations and views.

To find a sketch from what happens in it, search the dialogue. It is spoken, colloquial Armenian: people do not describe
the plot, they just talk, and they often use slang and Russian loanwords.
- Patterns are Python regexes over Armenian text. Never put English words or commas in a pattern.
- Search one idea at a time, as a short word stem, because endings change: հիվանդ matches հիվանդանոց, հիվանդի.
- Use | for synonyms and spelling variants: (կնիք|պեչատ).
- Think about what a character would actually SAY in the scene, not how you would summarize it.
- Use all_patterns to require two or three stems in the same sketch when one stem matches too many sketches.
- Use skip to page through long result lists. Check a candidate with read_sketch before answering.

End with a final line exactly like: ANSWER: <value>. For "which sketch" the value is the sketch_id.
If nothing fits after your searches, write ANSWER: none."""


# ------------------------------------------------------------------------------------------ the agent loop
def make_client():
    load_dotenv(REPO_ROOT / ".env")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=3,
        timeout=120,
    )


def call_model(client, model, messages, tools):
    """One model call, pinned to one provider so runs stay comparable. Returns the reply and its cost in dollars."""
    settings = MODELS[model]
    extra_body = {
        "provider": {"order": [settings["provider"]], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }
    if settings["reasoning"]:
        extra_body["reasoning"] = {"effort": "low"}
    request = {"model": model, "messages": messages, "tools": tools, "max_tokens": 2500, "extra_body": extra_body}
    if settings["temperature"]:
        request["temperature"] = 0

    response = client.chat.completions.create(**request)
    if not response.choices:
        raise RuntimeError(f"{model}: the provider returned an error: {getattr(response, 'error', None)}")
    cost = getattr(response.usage, "cost", None)
    if cost is None:
        raise RuntimeError(f"{model}: OpenRouter returned no cost")
    return response.choices[0].message, cost


def run_tool(name, arguments):
    """Run one tool. A failure goes back to the model as text, so it can fix its own call."""
    if arguments is None:
        return {"error": "Your tool call was too long and got cut off. Use a much shorter pattern."}
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def read_tool_arguments(reply):
    """Parse each call's JSON. A reply cut off by max_tokens leaves broken JSON: mark it and keep the history valid."""
    arguments_by_id = {}
    for tool_call in reply.tool_calls:
        try:
            arguments_by_id[tool_call.id] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments_by_id[tool_call.id] = None
            tool_call.function.arguments = "{}"
    return arguments_by_id


def read_final_answer(text):
    found = re.findall(r"ANSWER:\s*(.+)", text or "")
    return found[-1].strip() if found else None


def run_agent(client, question, model=DEFAULT_MODEL, tool_names=ALL_TOOL_NAMES, max_steps=10, verbose=False):
    """The loop: ask the model, run the tools it asks for, hand back the results, stop when it answers."""
    # 1. start the conversation and offer the tools
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    tools = [TOOL_SCHEMAS[name] for name in tool_names]
    calls = []
    total_cost = 0.0
    started = time.time()

    for step in range(1, max_steps + 1):
        # 2. ask the model what to do next
        reply, cost = call_model(client, model, messages, tools)
        total_cost += cost

        # 3. no tool requested: this is the final answer
        if not reply.tool_calls:
            if verbose:
                print(f"step {step}: final answer\n{reply.content}")
            return {
                "answer": read_final_answer(reply.content),
                "text": reply.content,
                "steps": step,
                "calls": calls,
                "cost": total_cost,
                "seconds": time.time() - started,
                "hit_step_cap": False,
            }

        # 4. run every tool the model asked for and hand back the results
        arguments_by_id = read_tool_arguments(reply)
        messages.append(reply.model_dump(exclude_none=True))
        for tool_call in reply.tool_calls:
            name = tool_call.function.name
            arguments = arguments_by_id[tool_call.id]
            result = run_tool(name, arguments)
            calls.append({"tool": name, "arguments": arguments, "broken_json": arguments is None})
            result_text = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
            if verbose:
                print(f"step {step}: {name}({json.dumps(arguments, ensure_ascii=False)})")
                print(f"   -> {result_text[:220]}")

    # 5. out of steps: stop instead of spending forever
    if verbose:
        print(f"stopped: no answer after {max_steps} steps")
    return {
        "answer": None,
        "text": None,
        "steps": max_steps,
        "calls": calls,
        "cost": total_cost,
        "seconds": time.time() - started,
        "hit_step_cap": True,
    }


def is_correct(answer, question):
    """Compare the agent's answer with the known one. Numbers and sketch ids compare as numbers: '074' equals '74'."""
    if not answer:
        return False
    if question["answer_type"] == "text":
        return question["answer"].casefold() in answer.casefold()
    numbers = re.findall(r"\d+", answer)
    return bool(numbers) and int(numbers[0]) == int(question["answer"])
