# MCP Python SDK v2 in Jupyter: in-memory client, one cell at a time; stdio only outside

**Context.** Session 2 of the tools practical runs an MCP server and client inside the course's Jupyter kernel
(`ma`, Windows, `mcp==2.3.0`, ipykernel 6.30.1).

**What works.** `async with Client(server):` inside **one** cell (top-level `await` is fine): `list_tools` and
`call_tool` go through the real protocol, no process, no port (probe notebook executed in 7.8 s on 2026-11-12).

**What does not.**

- Entering the client in one cell and closing it in a later cell throws an anyio `ExceptionGroup` (found by the
  outline review's probe). So every MCP cell opens and closes its own client; a bridge function does the same per call.
- `Client(StdioServerParameters(...))` fails inside Jupyter with `UnsupportedOperation: fileno` (the SDK hands
  Jupyter's stderr to the subprocess). The same `server.py` over stdio works from a plain Python script, which is how
  it was tested; in class the real-client demo happens outside the notebook.

**Also.**

- v2 names differ from most tutorials: `MCPServer` (was `FastMCP`), `tool.input_schema` in Python (`inputSchema`
  on the wire), `Client(...)` instead of `ClientSession` plus a transport.
- A `dict` return comes back as JSON text (`structured_content` is `None`); a `list[dict]` return comes back as one
  text block per item **and** `structured_content == {"result": [...]}`. Prefer `structured_content` when present.
- `import mcp` turns on an INFO log line per HTTP request (httpx) in notebook output:
  `logging.getLogger("httpx").setLevel(logging.WARNING)`.
