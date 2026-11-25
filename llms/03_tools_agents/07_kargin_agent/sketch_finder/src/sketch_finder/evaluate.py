"""Our own benchmark: run the known-answer questions, save every run as JSON, summarize per model."""
import json
import logging
import re
import time
from collections import defaultdict

from sketch_finder import config
from sketch_finder.agent import run_agent

log = logging.getLogger(__name__)


def load_questions(kinds=None):
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [question for question in questions if kinds is None or question["kind"] in kinds]


def is_correct(answer, question):
    """Compare the agent's answer with the known one. Numbers and sketch ids compare as numbers: '074' equals '74'."""
    if not answer:
        return False
    if question["answer_type"] == "text":
        return question["answer"].casefold() in answer.casefold()
    numbers = re.findall(r"\d+", answer)
    return bool(numbers) and int(numbers[0]) == int(question["answer"])


def said_nothing(answer):
    return answer is None or answer.strip(" .").lower() == "none"


def run_eval(client, questions, models, tool_names, output_path, budget_usd):
    """Every model on every question. Saves after each run and resumes from the file if it already exists."""
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
    finished = {(row["model"], row["id"]) for row in results}
    spent = sum(row["cost"] for row in results)
    total = len(models) * len(questions)
    started = time.time()
    run_now = 0

    for model in models:
        for question in questions:
            if (model, question["id"]) in finished:
                continue
            if spent > budget_usd:
                raise RuntimeError(f"budget of ${budget_usd} reached after ${spent:.4f}")
            outcome = run_agent(client, question["question"], model=model, tool_names=tool_names, label="eval")
            spent += outcome["cost"]
            run_now += 1
            correct = is_correct(outcome["answer"], question)
            results.append({"model": model, "id": question["id"], "expected": question["answer"],
                            "correct": correct, **outcome})
            output_path.parent.mkdir(exist_ok=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            elapsed = time.time() - started
            left = total - len(results)
            log.info(f"{len(results)}/{total} {model.split('/')[1]} {question['id']}: {'right' if correct else 'WRONG'} "
                     f"got={outcome['answer']} tools={len(outcome['calls'])} ${spent:.4f}, "
                     f"{elapsed:.0f}s elapsed, ~{elapsed / run_now * left:.0f}s left")
    return results


def summarize(results):
    """One line per model: right, confidently wrong, gave up, cost and tool calls per question."""
    by_model = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)
    summary = []
    for model, rows in by_model.items():
        summary.append({
            "model": model,
            "right": sum(row["correct"] for row in rows),
            "wrong_sketch": sum(not row["correct"] and not said_nothing(row["answer"]) for row in rows),
            "no_answer": sum(not row["correct"] and said_nothing(row["answer"]) for row in rows),
            "questions": len(rows),
            "cost_per_question": sum(row["cost"] for row in rows) / len(rows),
            "tool_calls_per_question": sum(len(row["calls"]) for row in rows) / len(rows),
        })
    return sorted(summary, key=lambda line: (-line["right"], line["cost_per_question"]))
