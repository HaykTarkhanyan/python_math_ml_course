"""Rebuild the ArmBench-LLM leaderboard from its published result files and join it with live OpenRouter prices.

Scoring per the Space's README: BLEU / 100, exams / 20, mean per category, unweighted mean of categories. About 30 s.
"""
import json
import pathlib
import sys
import urllib.parse

import requests

sys.stdout.reconfigure(encoding="utf-8")
SCRATCH = pathlib.Path(__file__).parent
FOLDER = SCRATCH / "armbench_results"
FOLDER.mkdir(exist_ok=True)
SPACE = "https://huggingface.co/spaces/Metric-AI/ArmBench-LLM/resolve/main/results/"

CATEGORIES = {
    "Knowledge": ["exam_history", "exam_literature", "exam_math", "include", "hartak", "arak", "mmlu_pro"],
    "Reading": ["belebele", "syndarin", "scientific", "dream", "squad", "ms_marco"],
    "Analysis": ["topic-14class", "sentiment", "pioner", "finer", "pos"],
    "Generation": ["email", "conversation", "paraphrase", "short_sentences_translation"],
    "Correction": ["punctuation", "space_fix"],
}

tree = requests.get("https://huggingface.co/api/spaces/Metric-AI/ArmBench-LLM/tree/main/results", timeout=30)
tree.raise_for_status()
names = [item["path"].split("/", 1)[1] for item in tree.json() if item["path"].endswith(".json")]
for name in names:
    path = FOLDER / name
    if not path.exists():
        response = requests.get(SPACE + urllib.parse.quote(name), timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)


def score_value(metrics):
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if key == "bleu":
            return value / 100
        if key == "armenian_exam_score":
            return value / 20
        if key in ("exact_match_mcqa", "armenian_mmlu_pro_score", "ner_accuracy", "ud_pos_regex_acc",
                   "punctuation_accuracy", "space_accuracy"):
            return value
    return None


rows = {}
for name in names:
    data = json.loads((FOLDER / name).read_text(encoding="utf-8"))
    general = data.get("config_general", {})
    model = general.get("model_config", {}).get("model_name") or general.get("model_name") or name
    model = model.split(",")[0]
    results = data["results"]
    categories = {}
    tasks_found = 0
    for category, tasks in CATEGORIES.items():
        values = []
        for task in tasks:
            key = f"armenian:{task}|0"
            if key in results:
                value = score_value(results[key])
                if value is not None:
                    values.append(value)
        tasks_found += len(values)
        if values:
            categories[category] = sum(values) / len(values)
    if categories:
        rows[model] = {"model": model, "tasks": tasks_found, **categories,
                       "average": sum(categories.values()) / len(categories), "file": name}

prices = {}
for model in json.loads((SCRATCH / "or_models.json").read_text(encoding="utf-8"))["data"]:
    prices[model["id"]] = (float(model["pricing"]["prompt"]) * 1e6, float(model["pricing"]["completion"]) * 1e6,
                           "tools" in (model.get("supported_parameters") or []))

table = sorted(rows.values(), key=lambda row: -row["average"])
(SCRATCH / "armbench_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(names)} files, {len(table)} models")
print(f"{'#':>2} {'model':42} {'tasks':>5} {'avg':>6} {'know':>6} {'read':>6} {'anal':>6} {'gen':>6} {'corr':>6}  $in/$out tools")
for index, row in enumerate(table, 1):
    price = prices.get(row["model"])
    price_text = f"{price[0]:.2f}/{price[1]:.2f} {'yes' if price[2] else 'NO'}" if price else "not on OpenRouter under this id"
    print(f"{index:>2} {row['model'][:42]:42} {row['tasks']:>5} {row['average']:.4f} " +
          " ".join(f"{row.get(c, float('nan')):.4f}" for c in CATEGORIES) + f"  {price_text}")
