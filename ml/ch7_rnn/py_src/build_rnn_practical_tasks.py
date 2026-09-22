"""Derive xx_rnn_memory.ipynb (student task version) from the executed solution notebook.

Keeps every markdown cell, every assert and all the given machinery (training loops, plots).
Blanks only the functions that ARE the lesson - the RNN step, the unrolled forward pass, the
weight copy into nn.RNN, the autograd influence measurement, the forget-gate dial, the recall
task, the name encoding and the sampling loop. Outputs are cleared.

Fails loudly if a target is missing or if an answer survives - the silent failure mode is
shipping a task notebook that still contains the solutions. Stub logic follows
ml/ch8_autoencoders/py_src/build_sae_tasks.py.

One source of truth: edit build_rnn_practical_nb.py, re-execute it, then re-run this.
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/build_rnn_practical_tasks.py
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
SRC, DST = CH / "xx_rnn_memory_solution.ipynb", CH / "xx_rnn_memory.ipynb"

FN_STUBS = [
    ("rnn_step", "return tanh(W^T x + V^T z_prev + b)"),
    ("rnn_forward", "start from a zero state, apply rnn_step to one_hot(i) for every word, "
                    "collect the states; score = sigmoid(U @ z_last)"),
    ("load_toy_weights", "inside torch.no_grad(): W.T -> weight_ih_l0, V.T -> weight_hh_l0, "
                         "b -> bias_ih_l0, and zero bias_hh_l0"),
    ("influence", "A = lam * Q with Q from torch.linalg.qr of a random HxH matrix (float64); "
                  "unroll T steps of h = A @ h + noise from h0 (requires_grad=True); backprop "
                  "u @ h_T; return h0.grad.norm() / u.norm()"),
    ("set_forget_bias", "inside torch.no_grad(): the forget gate is the slice [H:2H] of BOTH "
                        "bias_ih_l0 and bias_hh_l0 - give each total / 2"),
    ("recall_batch", "key = randint(0, N_KEYS, (n,)), distractors = randint(N_KEYS, VOCAB, "
                     "(n, T-1)), concatenate, one-hot to VOCAB; return (x, key)"),
    ("encode_names", "x: 0 ('.') then the letter ids, zero-padded; y: the letter ids then 0 "
                     "('.'), padded with -100"),
    ("sample_name", "x = '.' ; loop: logits, state = model(x, state); probs = softmax(logits "
                    "at the last position / temp); sample with torch.multinomial; stop on '.'; "
                    "otherwise append the letter and feed it back"),
]

LEAKS = [
    "np.tanh(W.T @ x + V.T @ z_prev + b)",
    "rnn.weight_ih_l0.copy_(torch.tensor(W.T",
    "(u @ h).backward()",
    "bias[H:2 * H].fill_(total / 2)",
    "torch.randint(N_KEYS, VOCAB, (n, T - 1), generator=g)",
    'y[i, len(ids)] = stoi["."]',
    "torch.multinomial(probs, 1, generator=g)",
]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("build_rnn_practical_tasks")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "build_rnn_practical_tasks.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


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
                if lines[j].strip() != q and lines[j].rstrip().endswith(q) \
                        and len(lines[j].strip()) > 5:
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
            new = [f"{body}# YOUR CODE HERE - {hint}",
                   f'{body}raise NotImplementedError("{name}")', ""]
            return "\n".join(lines[:j] + new + lines[k:])
    raise SystemExit(f"stub_function: {name!r} not found")


def main():
    log = setup_logging()
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

    old = "> This is the **solution** notebook; the task version is `xx_rnn_memory.ipynb`."
    if old not in nb.cells[0].source:
        raise SystemExit("header line not found in cell 0")
    nb.cells[0].source = nb.cells[0].source.replace(
        old, "> This is the **task** notebook. Fill in every `# YOUR CODE HERE`; the asserts "
             "check you as you go.\n> Solutions: `xx_rnn_memory_solution.ipynb`.")

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
