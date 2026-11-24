"""The command line: `sketch-finder build-db | ask | eval | serve | cost`.

    uv run sketch-finder build-db                                     # about 1 s
    uv run sketch-finder ask "the one where a lady lies to get a seat on the minibus"   # about 10 s, about $0.02
    uv run sketch-finder eval --kinds hazy --tools regex --models openai/gpt-6.1-sol   # about 3 min, about $0.17
    uv run sketch-finder serve                                        # the MCP server, over stdio
    uv run sketch-finder cost                                         # what the ledger says you spent
"""
import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from sketch_finder import config

log = logging.getLogger("sketch_finder")


def setup_logging(to_console=True):
    config.LOGS_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [logging.FileHandler(config.LOGS_DIR / "sketch_finder.log", encoding="utf-8")]
    if to_console:
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def command_build_db(arguments):
    from sketch_finder.database import build_database
    log.info(f"built {config.DB_PATH.name}: {build_database()}")


def command_ask(arguments):
    from sketch_finder.agent import make_client, run_agent
    from sketch_finder.tools import TOOL_SETS
    outcome = run_agent(make_client(), arguments.question, model=arguments.model,
                        tool_names=TOOL_SETS[arguments.tools], verbose=arguments.verbose, label="ask")
    log.info(f"answer: {outcome['answer']} | {outcome['steps']} steps, {len(outcome['calls'])} tool calls, "
             f"${outcome['cost']:.4f}, {outcome['seconds']:.0f} s")
    log.info(outcome["text"] or "(no answer: the agent ran out of steps)")


def command_eval(arguments):
    from sketch_finder.agent import make_client
    from sketch_finder.evaluate import load_questions, run_eval, summarize
    from sketch_finder.tools import TOOL_SETS
    questions = load_questions(arguments.kinds)
    output_path = Path(arguments.output) if arguments.output else config.RESULTS_DIR / f"eval_{time.strftime('%Y-%m-%d_%H%M')}.json"
    log.info(f"{len(questions)} questions x {len(arguments.models)} models, tools: {arguments.tools}, saving to {output_path}")
    results = run_eval(make_client(), questions, arguments.models, TOOL_SETS[arguments.tools], output_path, arguments.budget)
    for line in summarize(results):
        log.info(f"{line['model']}: {line['right']}/{line['questions']} right, {line['wrong_sketch']} wrong, "
                 f"{line['no_answer']} no answer, ${line['cost_per_question']:.4f} and "
                 f"{line['tool_calls_per_question']:.1f} tool calls per question")


def command_serve(arguments):
    from sketch_finder.agent import make_client
    from sketch_finder.server import mcp
    make_client()   # loads the keys from .env, for web_search
    mcp.run()


def command_cost(arguments):
    from sketch_finder.ledger import read_ledger
    entries = read_ledger()
    by_label = defaultdict(float)
    for entry in entries:
        by_label[entry["label"]] += entry["cost"]
    log.info(f"{len(entries)} model calls, ${sum(by_label.values()):.4f} in total: "
             + ", ".join(f"{label} ${cost:.4f}" for label, cost in sorted(by_label.items())))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sketch-finder", description="Find Kargin Haghordum sketches with an agent.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("build-db", help="build data/kargin.db from the CSV").set_defaults(run=command_build_db)

    ask = commands.add_parser("ask", help="ask the agent one question")
    ask.add_argument("question")
    ask.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ask.add_argument("--tools", default="all", choices=["all", "search", "regex", "sql"])
    ask.add_argument("--verbose", action="store_true", help="print every step")
    ask.set_defaults(run=command_ask)

    evaluation = commands.add_parser("eval", help="run the known-answer questions and save the results")
    evaluation.add_argument("--models", nargs="+", default=[config.DEFAULT_MODEL], choices=list(config.MODELS))
    evaluation.add_argument("--kinds", nargs="+", choices=["sql", "quote", "hazy"], help="default: all")
    evaluation.add_argument("--tools", default="search", choices=["all", "search", "regex", "sql"])
    evaluation.add_argument("--output", help="results file (resumes if it exists)")
    evaluation.add_argument("--budget", type=float, default=0.50, help="stop above this many dollars")
    evaluation.set_defaults(run=command_eval)

    commands.add_parser("serve", help="run the MCP server over stdio").set_defaults(run=command_serve)
    commands.add_parser("cost", help="sum the cost ledger").set_defaults(run=command_cost)

    arguments = parser.parse_args(argv)
    setup_logging(to_console=arguments.command != "serve")   # stdio belongs to MCP while serving
    arguments.run(arguments)


if __name__ == "__main__":
    main()
