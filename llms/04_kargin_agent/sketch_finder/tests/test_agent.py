"""The loop with a fake model: no API key, no network, no cost."""
import json
from types import SimpleNamespace

from sketch_finder.agent import read_final_answer, read_tool_arguments, run_agent, run_tool
from sketch_finder.evaluate import is_correct, said_nothing, summarize
from sketch_finder.ledger import read_ledger


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    """Replays a fixed list of replies and remembers what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        usage = SimpleNamespace(cost=0.001, prompt_tokens=100, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))], usage=usage)


def test_the_loop_runs_a_tool_and_stops_at_the_answer(tiny_db):
    client = FakeClient([
        FakeMessage(tool_calls=[tool_call("c1", "run_sql", json.dumps({"query": "SELECT count(*) FROM sketches"}))]),
        FakeMessage(content="There are 6.\nANSWER: 6"),
    ])
    outcome = run_agent(client, "How many sketches?", model="openai/gpt-4o-mini", tool_names=("run_sql",))

    assert outcome["answer"] == "6"
    assert outcome["steps"] == 2
    assert outcome["calls"][0]["tool"] == "run_sql"
    assert (outcome["calls"][0]["step"], outcome["calls"][0]["result"]) == (1, "1 rows")
    assert abs(outcome["cost"] - 0.002) < 1e-9
    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and '"6"' in tool_message["content"]
    assert len(read_ledger()) == 2


def test_the_step_cap_stops_a_loop_that_never_answers(tiny_db):
    endless = [FakeMessage(tool_calls=[tool_call(f"c{i}", "search_dialogue", '{"all_patterns": ["xyz"]}')])
               for i in range(3)]
    outcome = run_agent(FakeClient(endless), "?", model="openai/gpt-4o-mini", max_steps=3)
    assert outcome["hit_step_cap"] and outcome["answer"] is None and outcome["steps"] == 3


def test_a_cut_off_call_is_marked_and_explained():
    reply = SimpleNamespace(tool_calls=[tool_call("c1", "search_dialogue", '{"all_patterns": ["(կնիք|պեչ')])
    arguments = read_tool_arguments(reply)
    assert arguments == {"c1": None}
    assert reply.tool_calls[0].function.arguments == "{}"
    assert "cut off" in run_tool("search_dialogue", None)["error"]


def test_a_failing_tool_reports_instead_of_crashing(tiny_db):
    assert run_tool("read_sketch", {"sketch_id": 999})["error"].startswith("ValueError")


def test_the_final_answer_is_the_last_answer_line():
    assert read_final_answer("ANSWER: 1\nno wait\nANSWER: 198") == "198"
    assert read_final_answer("I could not find it.") is None


def test_answers_compare_as_numbers_or_text():
    sketch_question = {"answer": "74", "answer_type": "sketch_id"}
    assert is_correct("074", sketch_question)
    assert is_correct("Sketch 74", sketch_question)
    assert not is_correct("740", sketch_question)
    assert not is_correct(None, sketch_question)
    assert is_correct("Տուն (home)", {"answer": "Տուն", "answer_type": "text"})


def test_summary_separates_wrong_answers_from_no_answer():
    rows = [
        {"model": "m", "correct": True, "answer": "1", "cost": 0.01, "calls": [1, 2]},
        {"model": "m", "correct": False, "answer": "5", "cost": 0.01, "calls": [1]},
        {"model": "m", "correct": False, "answer": "none.", "cost": 0.01, "calls": []},
    ]
    line = summarize(rows)[0]
    assert (line["right"], line["wrong_sketch"], line["no_answer"]) == (1, 1, 1)
    assert said_nothing(None) and said_nothing(" None. ")
