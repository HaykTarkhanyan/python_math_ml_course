"""Paths, limits and model settings. Everything else reads these at call time, so tests can point them elsewhere."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "kargin_sketches.csv"
DB_PATH = DATA_DIR / "kargin.db"
QUESTIONS_PATH = DATA_DIR / "questions.json"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"
LEDGER_PATH = LOGS_DIR / "calls.jsonl"

# Keys: the project's own .env first, then the course repository's (the one used in class)
ENV_FILES = (PROJECT_DIR / ".env", PROJECT_DIR.parents[3] / ".env")

MAX_SQL_ROWS = 30
MAX_CELL_CHARS = 200
MAX_SEARCH_RESULTS = 10
MAX_PATTERN_CHARS = 200
MAX_DIALOGUE_CHARS = 2000
MAX_STEPS = 10

DEFAULT_MODEL = "openai/gpt-6.1-sol"
MODELS = {
    "openai/gpt-6.1-sol": {"provider": "openai", "temperature": False, "reasoning": True},
    "openai/gpt-6.1-terra": {"provider": "openai", "temperature": False, "reasoning": True},
    "deepseek/deepseek-v4.1-flash": {"provider": "together", "temperature": True, "reasoning": True},
    "google/gemini-3-flash-preview": {"provider": "google-ai-studio", "temperature": True, "reasoning": True},
    "openai/gpt-4o-mini": {"provider": "openai", "temperature": True, "reasoning": False},
}
