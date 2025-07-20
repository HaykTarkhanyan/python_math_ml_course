"""
CLI - Command Line Interface using Python Fire

https://github.com/google/python-fire

pip install fire

Run the script like this:
python 08_02_fire.py count_files assets --logs=True
"""

import os
import fire

def count_files(directory, logs=False):
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    if logs:
        print(f"Counting files in directory: {directory}")

    num_files = len(os.listdir(directory))
    if logs:
        print(f"Number of files: {num_files}")
        
    return num_files

def another_function(cheese: str = "Ամասիա"):
    # insert cheese emoji between each character
    cheese_emoji = " 🧀 "
    print(cheese_emoji.join(cheese))
    
if __name__ == "__main__":
    # fire.Fire(count_files)
    fire.Fire()

