"""Record and replay model calls and tool results for the tools practical's notebooks (llms/03_tools_agents).

In class a notebook replays the run saved in results/tools_N_run.json, so its outputs match its text: weather and
web search change every hour, so the tool results are saved too, not only the model's replies. Each demo has its
own recording: demo = rec.demo("name"), then demo.model(...) and demo.tool(...) play back what it saved, in order.
A demo opened with live=True calls for real and is never saved. With rerun=True every demo records again and
replaces its saved copy.

Replay is checked: each saved model reply remembers a hash of the request that produced it, and a cell that now sends
something else (because it was edited) raises instead of quietly replaying an old answer.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

from openai.types.chat import ChatCompletion

_MISSING = "missing-key"


def request_hash(kw):
    keep = {k: kw.get(k) for k in ("model", "messages", "tools", "tool_choice", "response_format")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


class Recorder:
    def __init__(self, path, client, rerun=False):
        self.path, self.client, self.rerun = Path(path), client, rerun
        if not self.path.exists() and not rerun:
            raise FileNotFoundError(f"{self.path} not found. Run the notebook from its own folder "
                                    "(llms/03_tools_agents), or set RERUN = True to record a new run.")
        self.saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.saved.setdefault("_dates", {})

    def demo(self, name, live=False):
        return Demo(self, name, live)

    def day_of(self, name):
        """The day a saved demo was recorded (today for a demo that is not saved yet)."""
        d = self.saved["_dates"].get(name) or self.saved.get("_date") or date.today().isoformat()
        return date.fromisoformat(d)

    def demos(self):
        return {k: v for k, v in self.saved.items() if not k.startswith("_")}

    def write(self):
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.saved, ensure_ascii=False, indent=1), encoding="utf-8")


class Demo:
    def __init__(self, rec, name, live=False):
        self.rec, self.name = rec, name
        self.record = rec.rerun or (name not in rec.saved and not live)     # goes into the saved run
        self.live = live or self.record                                     # calls for real
        self.today = date.today() if self.live else rec.day_of(name)       # the day the model is told it is
        self.items, self.i, self.dirty = ([] if self.live else rec.saved[name]), 0, False

    def _next(self, kind):
        if self.i >= len(self.items):
            raise RuntimeError(f"demo '{self.name}': the saved run has no more {kind}s. This cell asks for more than "
                               "was recorded (was it edited, or run twice?). Use live=True, or set RERUN = True.")
        item = self.items[self.i]
        self.i += 1
        return item

    def model(self, **kw):
        h = request_hash(kw)
        if self.live:
            if self.rec.client.api_key == _MISSING:
                raise RuntimeError("OPENROUTER_API_KEY is missing from .env: replaying works without it, live calls do not.")
            r = self.rec.client.chat.completions.create(**kw)
            self.items.append({**r.model_dump(), "request_hash": h})
            return r
        d = self._next("model reply")
        if "tool_result" in d:
            raise RuntimeError(f"demo '{self.name}' is out of step: expected a model reply, found a tool result.")
        if d.get("request_hash") is None:            # the first replay of an older recording: remember the request
            d["request_hash"], self.dirty = h, True
        elif d["request_hash"] != h:
            raise RuntimeError(f"demo '{self.name}': this cell sends a different request than the saved run (it was "
                               "edited). Use live=True, or set RERUN = True.")
        return ChatCompletion.model_validate(d) if d.get("choices") else ChatCompletion.model_construct(**d)

    def _replay_tool(self, fn, args):
        d = self._next("tool result")
        if "tool_result" not in d:
            raise RuntimeError(f"demo '{self.name}' is out of step: expected a tool result, found a model reply.")
        return d["tool_result"]

    def tool(self, fn, args):
        if self.live:
            try:
                out = fn(**args)
            except Exception as e:                   # a failing tool is reported to the model, and recorded
                out = {"error": f"{type(e).__name__}: {e}"}
            self.items.append({"tool_result": out})
            return out
        out = self._replay_tool(fn, args)
        if getattr(fn, "local", False):             # local side effects (a fake outbox) happen on replay too
            fn(**args)
        return out

    async def atool(self, fn, *args):
        if self.live:
            try:
                out = await fn(*args)
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            self.items.append({"tool_result": out})
            return out
        return self._replay_tool(fn, args)

    def cost(self):
        return sum((x.get("usage") or {}).get("cost") or 0 for x in self.items if "choices" in x)

    def save(self):                                  # LIVE demos are never saved
        if self.record:
            self.rec.saved[self.name] = self.items
            self.rec.saved["_dates"][self.name] = self.rec.saved["_date"] = self.today.isoformat()
        elif self.dirty:
            self.rec.saved[self.name] = self.items
        if self.record or self.dirty:
            self.rec.write()
