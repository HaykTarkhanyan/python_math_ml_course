"""yerevan-tools: session 1's two tools as an MCP server (tools practical, session 2).

Any MCP client can start it over stdio. For example, in Claude Code (run from this folder):
    claude mcp add yerevan-tools -- python server.py
Session 2's notebook connects to the same server object in memory (Client(mcp)), with no process at all.
"""
from mcp.server import MCPServer

import tools_lib

mcp = MCPServer("yerevan-tools")


@mcp.tool()
def get_weather(city: str, date: str | None = None, hour: int = 9) -> dict:
    """Weather in a city: now, or the forecast for a date (YYYY-MM-DD) at an hour (local time, 0-23).
    Includes low cloud, visibility and PM2.5, which decide whether distant mountains are visible."""
    return tools_lib.get_weather(city, date, hour)


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web for current facts, news or schedules. Returns the top 3 pages, each with a short text."""
    return tools_lib.web_search(query)


if __name__ == "__main__":
    mcp.run()
