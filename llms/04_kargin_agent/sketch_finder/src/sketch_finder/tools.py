"""The four tools the model may ask for, and the descriptions it sees."""
import os
import re

from sketch_finder import config
from sketch_finder.database import open_read_only


def run_sql(query: str) -> dict:
    """Run one read-only SQL query on the Kargin database. Returns the columns and at most 30 rows."""
    connection = open_read_only()
    try:
        cursor = connection.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(config.MAX_SQL_ROWS)
    finally:
        connection.close()
    short_rows = [[str(value)[:config.MAX_CELL_CHARS] for value in row] for row in rows]
    return {"columns": columns, "rows": short_rows}


def search_dialogue(all_patterns: list[str], skip: int = 0) -> dict:
    """Find sketches whose title or Armenian dialogue match EVERY regex in all_patterns (1 to 3 short patterns).
    Returns how many matched and 10 snippets. Use skip to see the next 10."""
    # 1. compile the patterns
    for pattern in all_patterns:
        if len(pattern) > config.MAX_PATTERN_CHARS:
            raise ValueError(f"pattern longer than {config.MAX_PATTERN_CHARS} characters: use a shorter one")
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
    return {"total_matches": len(matches), "results": matches[skip:skip + config.MAX_SEARCH_RESULTS]}


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
        "dialogue": (dialogue or "")[:config.MAX_DIALOGUE_CHARS],
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
SEARCH_TOOL_NAMES = ("run_sql", "search_dialogue", "read_sketch")
REGEX_TOOL_NAMES = ("search_dialogue", "read_sketch")
TOOL_SETS = {"all": ALL_TOOL_NAMES, "search": SEARCH_TOOL_NAMES, "regex": REGEX_TOOL_NAMES, "sql": ("run_sql",)}


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
