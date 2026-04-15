#!/bin/bash
# Thin wrapper: pipes hook JSON to the Python validator.
python "$(dirname "$0")/validate_qmd.py"
