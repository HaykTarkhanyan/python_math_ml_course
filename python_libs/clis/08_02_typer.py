"""
pip install typer

https://github.com/fastapi/typer

Run:
    typer 08_02_typer.py run
"""
import typer

def main(name: str):
    print(f"Hello {name}")


if __name__ == "__main__":
    typer.run(main)