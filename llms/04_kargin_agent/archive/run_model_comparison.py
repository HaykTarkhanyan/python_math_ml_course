"""Our own benchmark for video 07: five models on the 10 fresh hazy questions, regex tools only.

The notebook reads the saved result instead of paying for this again in class. Resumes if interrupted.
Writes results/model_comparison.json after every question. Approved budget about $0.41 (Q62), hard stop at $0.55.

Run from the repo root (about 8 min):
    ./ma/Scripts/python.exe -u llms/04_kargin_agent/py_src/run_model_comparison.py
"""
import json
import logging
import pathlib
import sys
import time

LESSON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON_DIR))
import kargin_agent  # noqa: E402

BUDGET_USD = 0.55
QUESTIONS_PATH = LESSON_DIR / "data" / "questions.json"
RESULTS_PATH = LESSON_DIR / "results" / "model_comparison.json"
MODELS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-6.1-terra",
    "google/gemini-3-flash-preview",
    "openai/gpt-6.1-sol",
]

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run_model_comparison.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    questions = [q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if q["kind"] == "hazy"]
    kargin_agent.build_database()
    client = kargin_agent.make_client()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    finished = {(row["model"], row["id"]) for row in results}
    spent = sum(row["cost"] for row in results)
    total = len(MODELS) * len(questions)
    started = time.time()
    run_now = 0

    for model in MODELS:
        for question in questions:
            if (model, question["id"]) in finished:
                continue
            if spent > BUDGET_USD:
                raise RuntimeError(f"budget of ${BUDGET_USD} reached after ${spent:.4f}")
            outcome = kargin_agent.run_agent(client, question["question"], model=model,
                                             tool_names=kargin_agent.REGEX_TOOL_NAMES)
            spent += outcome["cost"]
            run_now += 1
            correct = kargin_agent.is_correct(outcome["answer"], question)
            results.append({"model": model, "id": question["id"], "expected": question["answer"],
                            "correct": correct, **outcome})
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            left = total - len(results)
            elapsed = time.time() - started
            log.info(f"{len(results)}/{total} {model.split('/')[1]} {question['id']}: {'OK' if correct else 'NO'} "
                     f"got={outcome['answer']} tools={len(outcome['calls'])} ${spent:.4f}, "
                     f"{elapsed:.0f}s elapsed, ~{elapsed / run_now * left:.0f}s left")

    for model in MODELS:
        rows = [row for row in results if row["model"] == model]
        log.info(f"{model}: {sum(row['correct'] for row in rows)}/{len(rows)} right, "
                 f"${sum(row['cost'] for row in rows):.4f}, "
                 f"{sum(len(row['calls']) for row in rows) / len(rows):.1f} tool calls per question")
    log.info(f"total ${spent:.4f}, {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
