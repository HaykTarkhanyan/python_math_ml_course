"""Derive gesture_snake/gesture_snake.ipynb (student task version) from the solution notebook.

Keeps every markdown cell, every assert and all the given machinery (loading, training loop,
plots, export). Blanks only the functions that ARE the lesson - the burst split, the label-
changing mirror, the CNN, Grad-CAM, and the frozen-ResNet features. Outputs are cleared.

Fails loudly if a target is missing or if an answer survives - the silent failure mode is
shipping a task notebook that still contains the solutions. Stub logic follows
ml/ch7_rnn/py_src/build_rnn_practical_tasks.py.

One source of truth: edit build_gesture_snake_nb.py, rebuild it, then re-run this (~1 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/build_gesture_snake_tasks.py
"""
import io
import logging
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
LOGS_DIR = REPO_ROOT / "logs"
SRC = CH / "gesture_snake" / "gesture_snake_solution.ipynb"
DST = CH / "gesture_snake" / "gesture_snake.ipynb"

FN_STUBS = [
    ("burst_split", "for every user and every class, list its bursts (sorted); raise ValueError if "
                    "there are fewer than 2; the LAST burst goes to validation. Return "
                    "(train indices, validation indices)"),
    ("mirror_batch", "flip the images left-right (dimension 3 of (B, 3, H, W)) and map every "
                     "label through FLIP"),
    ("small_cnn", "nn.AvgPool2d(2), then three blocks conv 3x3 (padding 1) -> BatchNorm -> ReLU -> "
                  "MaxPool2d(2) with 3 -> 16 -> 32 -> 64 channels. head='flatten': Flatten + "
                  "Linear(64*8*8, 5); head='gap': AdaptiveAvgPool2d(1) + Flatten + Linear(64, 5)"),
    ("gradcam", "hook `layer` to catch its output A; forward the crop; k = argmax; "
                "torch.autograd.grad of logits[0, k] w.r.t. A; weight each map by its gradient "
                "averaged over height and width; ReLU of the weighted sum over maps; upsample to "
                "the crop size (F.interpolate, bilinear); divide by the max; return (numpy map, k)"),
    ("features", "for each batch of crops: bgr_to_tensor, flip(3) if flip, normalize with MEAN "
                 "and STD, run the trunk under torch.no_grad(); concatenate, return numpy"),
]

LEAKS = [
    "val_mask |= bursts == bs[-1]",
    "return xb.flip(3), FLIP[yb]",
    "nn.Linear(64 * 8 * 8, len(CLASSES))",
    "grad.mean(dim=(2, 3), keepdim=True) * A",
    "trunk(((xb.flip(3) if flip else xb) - MEAN) / STD)",
]

log = logging.getLogger("build_gesture_snake_tasks")


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(LOGS_DIR / "build_gesture_snake_tasks.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def stub_function(src, name, hint):
    """Keep `def name(...)` and its docstring; replace the rest of the body with a stub."""
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if re.match(rf"^(\s*)def {re.escape(name)}\s*\(", ln):
            indent = len(ln) - len(ln.lstrip())
            body = " " * (indent + 4)
            j = i
            while not lines[j].rstrip().endswith(":"):
                j += 1
            j += 1
            if j < len(lines) and lines[j].lstrip().startswith(('"""', "'''")):
                q = lines[j].lstrip()[:3]
                if lines[j].strip() != q and lines[j].rstrip().endswith(q) and len(lines[j].strip()) > 5:
                    j += 1
                else:
                    j += 1
                    while j < len(lines) and not lines[j].rstrip().endswith(q):
                        j += 1
                    j += 1
            k = j
            while k < len(lines):
                s = lines[k]
                if s.strip() and (len(s) - len(s.lstrip())) <= indent:
                    break
                k += 1
            new = [f"{body}# YOUR CODE HERE - {hint}", f'{body}raise NotImplementedError("{name}")', ""]
            return "\n".join(lines[:j] + new + lines[k:])
    raise SystemExit(f"stub_function: {name!r} not found")


def main():
    setup_logging()
    nb = nbf.read(str(SRC), as_version=4)
    done = set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        src = cell.source
        for name, hint in FN_STUBS:
            if re.search(rf"^\s*def {re.escape(name)}\s*\(", src, re.M):
                src = stub_function(src, name, hint)
                done.add(name)
        cell.source = src
    missing = [n for n, _ in FN_STUBS if n not in done]
    if missing:
        raise SystemExit(f"never stubbed: {missing}")

    old = "> This is the **solution** notebook; the task version is `gesture_snake.ipynb`."
    if old not in nb.cells[0].source:
        raise SystemExit("header line not found in cell 0")
    nb.cells[0].source = nb.cells[0].source.replace(
        old, "> This is the **task** notebook. Fill in every `# YOUR CODE HERE`; the asserts check "
             "you as you go.\n> Solutions: `gesture_snake_solution.ipynb`.")

    with io.open(DST, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    body = DST.read_text(encoding="utf-8")
    for leak in LEAKS:
        if leak in body:
            raise SystemExit(f"LEAK: {leak!r} still present in {DST.name}")
    log.info(f"wrote {DST.name}: {len(nb.cells)} cells, {len(FN_STUBS)} functions stubbed, "
             f"leak check clean ({len(LEAKS)} patterns)")


if __name__ == "__main__":
    main()
