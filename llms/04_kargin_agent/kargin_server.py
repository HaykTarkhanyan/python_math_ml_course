"""kargin: the four tools of video 07 as an MCP server.

Any MCP client can start it over stdio. In Claude Code, from the repository root in PowerShell:
    claude mcp add kargin -- "$PWD/ma/Scripts/python.exe" "$PWD/llms/04_kargin_agent/kargin_server.py"
The notebook connects to the same server object in memory (Client(mcp)), with no process at all.
"""
from dotenv import load_dotenv
from mcp.server import MCPServer

import kargin_agent

load_dotenv(kargin_agent.REPO_ROOT / ".env")
if not kargin_agent.DB_PATH.exists():
    kargin_agent.build_database()

mcp = MCPServer("kargin")


@mcp.tool()
def run_sql(query: str) -> dict:
    """Run one read-only SQL query on the Kargin database. Tables: sketches(sketch_id, title, url, location, language,
    duration_sec, upload_date, view_count, dialogue), actors(actor_id, name), sketch_actors(sketch_id, actor_id).
    Returns the columns and at most 30 rows."""
    return kargin_agent.run_sql(query)


@mcp.tool()
def search_dialogue(all_patterns: list[str], skip: int = 0) -> dict:
    """Find sketches whose title or Armenian dialogue match EVERY regex in all_patterns (1 to 3 short patterns).
    Returns how many matched and 10 snippets. Use skip to see the next 10."""
    return kargin_agent.search_dialogue(all_patterns, skip)


@mcp.tool()
def read_sketch(sketch_id: int) -> dict:
    """Read one sketch: its title, YouTube link, location, actors and dialogue."""
    return kargin_agent.read_sketch(sketch_id)


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web for facts that are not in the database. Returns the top 3 pages, each with a short text."""
    return kargin_agent.web_search(query)


if __name__ == "__main__":
    mcp.run()
