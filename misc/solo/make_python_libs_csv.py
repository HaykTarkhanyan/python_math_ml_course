"""Generate python_libs_topics.csv: study tracker for the libraries notebooks (01-18).

Matches the structure of optimization.csv and python_topics.csv: same headers,
utf-8-sig encoding so Google Sheets / Excel render correctly. Re-runnable.

Run: uv run python misc/solo/make_python_libs_csv.py
"""

import csv
import logging
from pathlib import Path


Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/make_python_libs_csv.log"),
    ],
)
log = logging.getLogger(__name__)


OUT = Path("misc/solo/python_libs_topics.csv")

HEADERS = ["Section", "Item", "Link", "Note", "Done", "Question / comment"]

BASE = "https://hayktarkhanyan.github.io/python_math_ml_course/python_libs/"

SEC_DATA = "Data stack and early projects"
SEC_TOOL = "Tooling: scraping, storage, web"
SEC_META = "AI tooling and clean code"

# (section, item, link, note)
ROWS = [
    (SEC_DATA, "01 OpenAI API & timestamp generator", BASE + "01_openai_api_timestamp_generator.html",
     "Nice project, good to watch. Once you've seen it, we can plan another project together."),
    (SEC_DATA, "02 NumPy", BASE + "02_numpy.html",
     "You may already know most of this - just check the notebook."),
    (SEC_DATA, "03 Pandas (part 1)", BASE + "03_pandas_1.html",
     "Pandas is very important. No need to memorize function names - focus on capabilities and nuances."),
    (SEC_DATA, "04 Pandas (part 2)", BASE + "04_pandas_2.html",
     "Continuation. Same approach as 03: capabilities and nuances over memorization."),
    (SEC_DATA, "05 Noble people analysis (project)", BASE + "05_noble_people_analysis.html",
     "Just a project. We can pick a different one if you prefer."),
    (SEC_DATA, "06 Data visualization", BASE + "06_data_viz.html",
     "Quite important."),
    (SEC_DATA, "07 Kargin project", BASE + "07_kargin_project.html",
     "Also just a project. But since you like Kargin, maybe worth a watch."),
    (SEC_DATA, "08 Logging & CLIs", BASE + "08_logging__clis.html",
     "Will be used quite a lot."),
    (SEC_DATA, "09 Testing & debugging", BASE + "09_testing__debugging.html",
     "Used in every professional scenario. Quite needed."),

    (SEC_TOOL, "10 Scraping & parallelization", BASE + "10_scraping__parallelization.html",
     "Bonus for now."),
    (SEC_TOOL, "11 YSU scraping (project)", BASE + "11_ysu_scraping.html",
     "Scraping project. We can change the target website, but it's a nice experience."),
    (SEC_TOOL, "12 SQL", BASE + "12_sql.html",
     "Not needed for pure ML for now - can be deferred."),
    (SEC_TOOL, "13 Pydantic", BASE + "13_pydantic.html",
     "Important topics, but not super urgent."),
    (SEC_TOOL, "14 Misc libraries", BASE + "14_misc_libraries.html",
     "Just check the notebook. The lecture video is split into multiple parts."),
    (SEC_TOOL, "15 FastAPI", BASE + "15_fast_api.html",
     "Will be important in the future, not urgent right now."),
    (SEC_TOOL, "16 Databases (Supabase)", BASE + "16_dbs_supabase.html",
     "Same as 15 - important later, not urgent now."),

    (SEC_META, "17 Vibe coding", BASE + "17_vibe_coding.html",
     "Quick demo of AI-agent coding, already quite outdated. GitHub Copilot is significantly worse than current tools like Codex / Claude Code."),
    (SEC_META, "18 Clean code & architecture", BASE + "18_clean_code_architecture.html",
     "Just the video is worth checking."),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for section, item, link, note in ROWS:
            w.writerow([section, item, link, note, "", ""])
    log.info(f"Wrote {OUT.resolve()} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
