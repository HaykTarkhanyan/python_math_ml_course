# ============================================================
#  CLI libraries — terminal commands
#  Run these from:  python_libs/clis/
# ============================================================

# ================= 1. argparse (stdlib) =====================
# pip install — nothing, it's built-in

python 08_02_argparse.py --help                          # auto-generated help
python 08_02_argparse.py --version                       # prints "08_02_argparse.py 1.0"
python 08_02_argparse.py assets --logs -c option1        # positional dir + flag + choice
python 08_02_argparse.py assets -c option2               # without --logs
python 08_02_argparse.py assets                          # error: --choice is required
python 08_02_argparse.py                                 # error: directory is required

# ================= 2. click ==================================
pip install click

python 08_02_click.py --help                             # shows group-level help (lists commands)
python 08_02_click.py count-files --help                 # help for one command (note: underscore → hyphen!)
python 08_02_click.py count-files assets --logs          # count files with logging
python 08_02_click.py count-files assets                 # count files, no logging
python 08_02_click.py another-function                   # default cheese = Ամասիա
python 08_02_click.py another-function --cheese="Gouda"  # custom cheese

# ================= 3. fire (Google) ==========================
pip install fire

python 08_02_fire.py assets                              # positional arg (fire.Fire(count_files))
python 08_02_fire.py assets --logs=True                  # with flag
python 08_02_fire.py -- --help                           # auto-generated help (note the --)

# To expose ALL functions: change fire.Fire(count_files) → fire.Fire()
# python 08_02_fire.py count_files assets --logs=True
# python 08_02_fire.py another_function --cheese="Gouda"

# ================= 4. typer (FastAPI team) ====================
pip install typer

python 08_02_typer.py --help                             # auto-generated help
python 08_02_typer.py World                              # prints "Hello World"
python 08_02_typer.py "Բարև Աշխարh"                    # works with Armenian too

# alternative runner (no if __name__ block needed):
typer 08_02_typer.py run --help
typer 08_02_typer.py run World

# ================= Comparison cheat-sheet ====================
#  Library   | Install       | Decorators? | Auto help? | Type hints? | Subcommands?
#  ----------|---------------|-------------|------------|-------------|-------------
#  argparse  | stdlib        | No          | Yes        | manual      | Yes (subparsers)
#  click     | pip install   | Yes         | Yes        | No*         | Yes (@group)
#  fire      | pip install   | No          | Yes        | No          | Yes (fire.Fire())
#  typer     | pip install   | Optional    | Yes        | Yes!        | Yes (typer.Typer())
#
# * click uses its own type system (click.Path, click.INT, etc.)
