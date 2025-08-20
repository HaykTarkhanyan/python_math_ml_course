#!/usr/bin/env python3
"""
CLI - Command Line Interface using **Click**

Install:
    pip install click

Run:
    python 08_02_click.py count_files assets --logs True
    python 08_02_click.py another_function --cheese="Gouda"
"""
import os
import click


@click.group()
def cli():
    """Entry‑point for all commands. Some text here"""
    pass


@cli.command()
@click.argument("directory", type=click.Path())
@click.option("--logs", default=False, is_flag=True, help="Enable logging")
def count_files(directory: str, logs: bool):
    """Count files in a directory."""
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    if logs:
        print(f"Counting files in directory: {directory}")

    num_files = len(os.listdir(directory))

    if logs:
        print(f"Number of files: {num_files}")

    print(num_files) 
    return num_files


@cli.command()
@click.option("--cheese", "-c", default="Ամասիա", help="Cheese string")
def another_function(cheese: str):
    """Insert cheese emoji between each character."""
    cheese_emoji = " 🧀 "
    print(cheese_emoji.join(cheese))


if __name__ == "__main__":
    cli()
