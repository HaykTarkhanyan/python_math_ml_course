"""Validate homework .qmd files for common issues.

Called by the PostToolUse hook after Edit/Write on math/*.qmd files.
Reads hook input JSON from stdin, extracts file path, runs checks.

Exit 0 always (we warn, never block). Warnings go to stdout.
"""
import json
import os
import re
import sys


def get_file_path():
    """Extract file path from hook input JSON on stdin."""
    try:
        data = json.load(sys.stdin)
        fp = data.get("tool_input", {}).get("file_path", "")
        if not fp:
            fp = data.get("input", {}).get("file_path", "")
        return fp
    except Exception:
        return ""


def is_homework_qmd(path):
    """Check if this is a homework .qmd we should validate."""
    if not path or not path.endswith(".qmd"):
        return False
    basename = os.path.basename(path)
    # Match math/NN_*.qmd or math/*_stat_*.qmd
    if "math" in path and (
        re.match(r"^\d+_", basename) or "stat" in basename
    ):
        return True
    return False


def check_latex_outside_math(lines):
    """Find LaTeX commands used outside $...$ math mode."""
    warnings = []
    in_code_block = False
    in_frontmatter = False
    frontmatter_count = 0

    latex_cmds = [
        "quad", "qquad", "text", "textbf", "textit", "frac", "dfrac",
        "sqrt", "sum", "prod", "int", "alpha", "beta", "gamma", "delta",
        "mu", "sigma", "lambda", "theta", "epsilon", "omega", "pi",
        "hat", "bar", "tilde", "vec", "dot",
        "mathbb", "mathcal", "mathrm", "mathbf",
        "displaystyle", "left", "right", "big", "Big", "bigg",
        "leq", "geq", "neq", "approx", "infty", "partial",
        "cdot", "times", "div", "pm", "mp",
        "overset", "underset", "stackrel",
    ]
    pattern = re.compile(r"\\(" + "|".join(latex_cmds) + r")\b")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track YAML frontmatter
        if stripped == "---":
            frontmatter_count += 1
            if frontmatter_count <= 2:
                in_frontmatter = not in_frontmatter
                continue
        if in_frontmatter:
            continue

        # Track fenced code blocks
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Skip HTML, callout directives, script tags
        if stripped.startswith("<") or stripped.startswith(":::"):
            continue

        # Remove inline math $...$ and display math $$...$$
        no_math = re.sub(r"\$\$[^$]+\$\$", "", line)
        no_math = re.sub(r"\$[^$]+\$", "", no_math)

        matches = pattern.findall(no_math)
        if matches:
            warnings.append(
                f"LaTeX outside math mode at line {i}: \\{matches[0]}"
            )

    return warnings


def check_missing_blank_before_list(lines):
    """Find list items that aren't preceded by a blank line (Quarto gotcha)."""
    warnings = []

    for i in range(1, len(lines)):
        curr = lines[i].strip()
        prev = lines[i - 1].strip()

        # Is current line a list start?
        is_list = (
            curr.startswith("- ")
            or curr.startswith("* ")
            or (len(curr) > 2 and curr[0].isdigit() and curr[1] in ".)")
        )
        if not is_list:
            continue

        # Previous line should be blank or another structural element
        safe_prev = (
            not prev  # blank line
            or prev.startswith("-")
            or prev.startswith("*")
            or prev.startswith("#")
            or prev.startswith("<")
            or prev.startswith("---")
            or prev.startswith("|")
            or prev.startswith(":::")
            or prev.startswith("```")
            or prev.startswith("~~~")
            or (len(prev) > 1 and prev[0].isdigit() and prev[1] in ".)")
        )

        if not safe_prev:
            warnings.append(
                f"Missing blank line before list at line {i + 1} "
                f"(after: {prev[:50]})"
            )

    return warnings


def check_missing_difficulty(lines):
    """Find exercise headers (### NN ...) missing {data-difficulty="N"}."""
    warnings = []
    for i, line in enumerate(lines, 1):
        if re.match(r"^###\s+\d+", line.strip()):
            if "data-difficulty" not in line:
                warnings.append(
                    f'Missing {{data-difficulty="N"}} at line {i}: '
                    f"{line.strip()[:60]}"
                )
    return warnings


def main():
    filepath = get_file_path()
    if not is_homework_qmd(filepath):
        return

    if not os.path.isfile(filepath):
        return

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    all_warnings = []
    all_warnings.extend(check_latex_outside_math(lines))
    all_warnings.extend(check_missing_blank_before_list(lines))
    all_warnings.extend(check_missing_difficulty(lines))

    if all_warnings:
        basename = os.path.basename(filepath)
        print(f"qmd-validator: {len(all_warnings)} issue(s) in {basename}:")
        for w in all_warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
