"""The four tools as an MCP server, so any MCP client (Claude Code, Cursor, ...) can use them.

Start it with `sketch-finder serve`. To register it in Claude Code, from the project folder in PowerShell:
    claude mcp add kargin -- uv run --directory "$PWD" sketch-finder serve
"""
from mcp.server import MCPServer

from sketch_finder import config, tools
from sketch_finder.database import build_database

if not config.DB_PATH.exists():
    build_database()

mcp = MCPServer("kargin")


@mcp.tool()
def run_sql(query: str) -> dict:
    """Run one read-only SQL query on the Kargin database. Tables: sketches(sketch_id, title, url, location, language,
    duration_sec, upload_date, view_count, dialogue), actors(actor_id, name), sketch_actors(sketch_id, actor_id).
    Returns the columns and at most 30 rows."""
    return tools.run_sql(query)


@mcp.tool()
def search_dialogue(all_patterns: list[str], skip: int = 0) -> dict:
    """Find sketches whose title or Armenian dialogue match EVERY regex in all_patterns (1 to 3 short patterns).
    Returns how many matched and 10 snippets. Use skip to see the next 10."""
    return tools.search_dialogue(all_patterns, skip)


@mcp.tool()
def read_sketch(sketch_id: int) -> dict:
    """Read one sketch: its title, YouTube link, location, actors and dialogue."""
    return tools.read_sketch(sketch_id)


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web for facts that are not in the database. Returns the top 3 pages, each with a short text."""
    return tools.web_search(query)
