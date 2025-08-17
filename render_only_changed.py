#!/usr/bin/env python3
"""
Python translation of render_only_changed.r
Finds changed Jupyter notebook files and renders them with Quarto
Based on: https://github.com/quarto-dev/quarto-cli/discussions/3674
"""

import subprocess
import sys
import os

def run_git_command(command):
    """Run a git command and return the output as a list of lines."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        return []

def main():
    print("Starting")
    
    # Get list of changed files (equivalent to git diff command in R)
    git_command = "git diff main origin/main --name-only -- . :^docs"
    all_files = run_git_command(git_command)
    
    # Filter for .ipynb files (equivalent to str_detect(files, 'ipynb$'))
    ipynb_files = [f for f in all_files if f.endswith('.ipynb')]
    
    print(f"Found {len(ipynb_files)} changed files.")
    
    if len(ipynb_files) > 0:
        print("Rendering uncommitted *ipynb files:")
        for file in ipynb_files:
            print(f"\t{file}")
        print()
        
        for file in ipynb_files:
            print(file)
            cmd = f"quarto render {file}"
            print(cmd)
            
            # Run the quarto render command
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error rendering {file}: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()
