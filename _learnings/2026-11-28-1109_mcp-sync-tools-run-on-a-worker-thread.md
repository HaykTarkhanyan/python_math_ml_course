# MCP servers run a plain `def` tool on a worker thread, so signal-based timeouts do nothing there

**Symptom.** A tool that limits its own run time with `signal.SIGALRM` / `signal.setitimer` (the usual Unix
"stop after N seconds" trick) works when the tool function is called directly, from a script, a notebook or a pytest
test, but has no effect when the same function is called through the MCP server. A runaway computation (sympy on a
hard equation, a big loop) then runs for as long as it takes. Found while reviewing a student's MCP project
(2026-11-28): its 20 s math limit checked `threading.current_thread() is threading.main_thread()` and skipped the
limit otherwise, so inside the server the limit was silently off, on macOS as well as Windows. Its tests passed,
because they called the tools directly.

**Cause.** `mcp` 2.3.0 (the version this course pins in `ma/pyproject.toml`, `sketch_finder/pyproject.toml` and
`llms/03_tools_agents/requirements.txt`) calls a sync tool with `anyio.to_thread.run_sync`. In
`mcp/server/mcpserver/utilities/func_metadata.py`, `call_fn`: "A sync function runs on a worker thread." Python allows
signal handlers only in the main thread. An `async def` tool runs on the event loop's thread (the main thread), but
blocking work inside it freezes the whole server, so making the tool async is not the fix.

**Evidence.** A minimal server with three tools, called through `MCPServer.call_tool` (mcp 2.3.0, Python 3.12):

```
sync_tool -> {   "main_thread": false,   "thread": "AnyIO worker thread" }
async_tool -> {   "main_thread": true,   "thread": "MainThread" }
sync_tool_sets_signal -> ValueError: signal only works in main thread of the main interpreter
```

**Consequences.**

- A time limit for a tool that can run away has to live outside the thread: run the work in a child process
  (`multiprocessing.Process` + `join(timeout)` + `terminate()`). That works on every OS and in any thread, and it
  actually stops the work. A Python thread can't be killed, so a thread-based timeout only stops the waiting, not
  the work.
- SIGALRM is also missing on Windows, so a signal-based limit is off there even in a plain script.
- Test a tool the way the server runs it (through `call_tool` or an in-memory MCP client), not only by calling the
  function, or this class of bug stays invisible.
- The course's own servers (`llms/03_tools_agents/server.py`, `sketch_finder`) have no signal-based limits today
  (only HTTP `timeout=` arguments, which work in any thread), so nothing is broken. This rule is for the next tool
  that needs one.
