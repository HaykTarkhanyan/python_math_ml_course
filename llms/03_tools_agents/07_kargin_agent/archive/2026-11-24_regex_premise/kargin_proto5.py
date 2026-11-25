"""ArmBench's #1 (GPT-6.1 Sol) on the same 12 hazy questions, regex tools only.

Run from the repo root with the ma venv. Expected: ~3 min, about $0.18, hard stop at $0.24.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kargin_proto2 as proto2
import kargin_proto4 as proto4

proto2.NO_TEMPERATURE.add("openai/gpt-6.1-sol")
proto4.BUDGET_USD = 0.24
proto4.MODELS = [("openai/gpt-6.1-sol", "openai")]
proto4.RESULTS_PATH = pathlib.Path(__file__).parent / "kargin_proto5_results.json"

if __name__ == "__main__":
    proto4.main()
