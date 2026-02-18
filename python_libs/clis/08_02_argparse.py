"""
CLI - Command Line Interface

https://docs.python.org/3/library/argparse.html

python 08_02_argparse.py assets --logs
"""
import os
import argparse # առանց r

def count_files(directory, logs=False):
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    if logs:
        print(f"Counting files in directory: {directory}")

    num_files = len(os.listdir(directory))
    if logs:
        print(f"Number of files: {num_files}")
        
    return num_files

if __name__ == "__main__":
    # print(count_files("argparser"))
    
    parser = argparse.ArgumentParser(description="Count files in a directory")
    
    parser.add_argument("directory", type=str, help="Directory path")
    parser.add_argument("--logs", "-l", action="store_true", help="Enable logging")
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 1.0", help="Show program version")
    parser.add_argument("--choice", "-c", choices=["option1", "option2"], 
                        help="Choose an option", required=True)#, default="option1")

    args = parser.parse_args()
    print(args)
    directory = args.directory
    logs = args.logs
    
    # directory = "test_dir"
    # logs = False
    
    num_files = count_files(directory, logs)
    print(f"Number of files in '{directory}': {num_files}")
    
