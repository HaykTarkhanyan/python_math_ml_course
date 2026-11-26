"""Rebuild the ArmBench-LLM leaderboard from its published result files, for part 8 of the video 07 notebook.

Scoring as the Space's README describes it: BLEU / 100, exam scores / 20, the mean per category, then the unweighted mean
of the five categories. Adds each entry's origin (a full evaluation log, or rounded scores supplied by hand) and its live
OpenRouter price. Writes results/armbench_2026-11-24.json.

Run from the repo root (about 30 s):
    ./ma/.venv/Scripts/python.exe llms/03_tools_agents/07_kargin_agent/py_src/build_armbench_table.py
"""
import json
import logging
import pathlib
import time
import urllib.parse

import requests

LESSON_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PATH = LESSON_DIR / "results" / "armbench_2026-11-24.json"
SPACE_FILES = "https://huggingface.co/spaces/Metric-AI/ArmBench-LLM/resolve/main/results/"
SPACE_TREE = "https://huggingface.co/api/spaces/Metric-AI/ArmBench-LLM/tree/main/results"
CATEGORIES = {
    "Knowledge": ["exam_history", "exam_literature", "exam_math", "include", "hartak", "arak", "mmlu_pro"],
    "Reading": ["belebele", "syndarin", "scientific", "dream", "squad", "ms_marco"],
    "Analysis": ["topic-14class", "sentiment", "pioner", "finer", "pos"],
    "Generation": ["email", "conversation", "paraphrase", "short_sentences_translation"],
    "Correction": ["punctuation", "space_fix"],
}
PLAIN_METRICS = {"exact_match_mcqa", "armenian_mmlu_pro_score", "ner_accuracy", "ud_pos_regex_acc",
                 "punctuation_accuracy", "space_accuracy"}
NAME_ALIASES = {"deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash"}

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/build_armbench_table.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def get_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def task_score(metrics):
    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if name == "bleu":
            return value / 100
        if name == "armenian_exam_score":
            return value / 20
        if name in PLAIN_METRICS:
            return value
    return None


def score_entry(data):
    category_scores = {}
    for category, tasks in CATEGORIES.items():
        values = []
        for task in tasks:
            key = f"armenian:{task}|0"
            if key in data["results"]:
                value = task_score(data["results"][key])
                if value is None:
                    raise ValueError(f"no known metric for {key}: {list(data['results'][key])[:5]}")
                values.append(value)
        if values:
            category_scores[category] = sum(values) / len(values)
    return category_scores


def main():
    started = time.time()
    file_names = [item["path"].split("/", 1)[1] for item in get_json(SPACE_TREE) if item["path"].endswith(".json")]
    prices = {model["id"].lower(): model for model in get_json("https://openrouter.ai/api/v1/models")["data"]}

    entries = {}
    for file_name in file_names:
        data = get_json(SPACE_FILES + urllib.parse.quote(file_name))
        general = data.get("config_general", {})
        name = (general.get("model_config", {}).get("model_name") or general.get("model_name") or file_name).split(",")[0]
        category_scores = score_entry(data)
        short_name = name.lower().removeprefix("openrouter/")
        short_name = NAME_ALIASES.get(short_name, short_name)
        priced = prices.get(short_name)
        entries[short_name] = {
            "model": short_name,
            "average": sum(category_scores.values()) / len(category_scores),
            **category_scores,
            "origin": "supplied rounded scores" if data.get("source") else "full evaluation log",
            "price_in_per_million": float(priced["pricing"]["prompt"]) * 1e6 if priced else None,
            "price_out_per_million": float(priced["pricing"]["completion"]) * 1e6 if priced else None,
        }

    table = sorted(entries.values(), key=lambda entry: -entry["average"])
    for rank, entry in enumerate(table, 1):
        entry["rank"] = rank
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {OUTPUT_PATH.name}: {len(table)} models from {len(file_names)} files in {time.time() - started:.0f} s")
    for entry in table[:8]:
        log.info(f"#{entry['rank']} {entry['model']} {entry['average']:.4f} ({entry['origin']})")


if __name__ == "__main__":
    main()
