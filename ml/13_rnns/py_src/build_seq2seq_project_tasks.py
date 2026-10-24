"""Derive seq2seq_dates_and_spaces.ipynb (student task version) from the executed solution notebook.

Keeps every markdown cell, every assert and the given machinery (generator, training loops,
plots). Blanks only the functions that ARE the lesson: padding a batch, the encoder, the
teacher-forcing shift, greedy decoding, the two metrics and the tagger's two conversions. The
birthday cell loses its hand-written answers. Outputs are cleared.

Fails loudly if a target is missing or if an answer survives - the silent failure mode is
shipping a task notebook that still contains the solutions. Stub logic follows
py_src/archive/build_rnn_practical_tasks.py.

One source of truth: edit build_seq2seq_project_nb.py, execute and merge it, then run this
(a second or two):
    ./ma/Scripts/python.exe ml/13_rnns/py_src/build_seq2seq_project_tasks.py
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
SRC, DST = CH / "seq2seq_dates_and_spaces_solution.ipynb", CH / "seq2seq_dates_and_spaces.ipynb"

FN_STUBS = [
    ("encode_batch", "map every character to vocab.stoi, append EOS if add_eos, pad every row "
                     "with PAD to the longest; return (ids on DEV, lengths on the CPU)"),
    ("encode", "embed x with self.enc_emb, pack it with pack_padded_sequence(..., lengths, "
               "batch_first=True, enforce_sorted=False), run self.encoder, return its final (z, c)"),
    ("make_decoder_input", "a column of SOS, then y without its last column (torch.cat on dim 1); "
                           "then set every EOS in the result to PAD"),
    ("greedy_decode", "state = model.encode(...); token = a column of SOS; repeat max_len times: "
                      "o, state = model.decoder(model.dec_emb(token), state); token = argmax of "
                      "model.head(o); collect the tokens; switch model.eval() / model.train()"),
    ("cer", "rapidfuzz's Levenshtein.distance(pred, truth) divided by len(truth)"),
    ("first_error", "None if equal; else the first index where the two differ, over "
                    "len(truth); if one ends first, the shorter length over len(truth)"),
    ("spaced_to_tags", "walk the characters: a letter appends a 0, a space turns the last tag into 1"),
    ("tags_to_spaced", "each letter, followed by a space if its tag is 1 and it is not the last letter"),
]

LEAKS = [
    "[EOS] if add_eos else []",
    "pack_padded_sequence(self.enc_emb(x)",
    "y_in[y_in == EOS] = PAD",
    "token = model.head(o).argmax(dim=-1)",
    "Levenshtein.distance(pred, truth) / len(truth)",
    "return i / len(truth)",
    "tags[-1] = 1",
    'ch + (" " if t == 1',
    "սեպտեմբերի հինգ, երկու հազար մեկ թ.",
]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("build_seq2seq_project_tasks")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "build_seq2seq_project_tasks.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
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
    log = setup_logging()
    nb = nbf.read(str(SRC), as_version=4)
    done, birthday = set(), False
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
        if "my_birthday =" in src:
            src, n = re.subn(r"my_forms = \[.*?\]", 'my_forms = ["...",                                  '
                             '# ... and write these by hand first\n            "...",\n            "...",\n'
                             '            "..."]', src, flags=re.S)
            birthday = n == 1
        cell.source = src
    missing = [n for n, _ in FN_STUBS if n not in done]
    if missing or not birthday:
        raise SystemExit(f"never stubbed: {missing}, birthday blanked: {birthday}")

    old = "> This is the **solution** notebook; the task version is `seq2seq_dates_and_spaces.ipynb`."
    if old not in nb.cells[0].source:
        raise SystemExit("header line not found in cell 0")
    nb.cells[0].source = nb.cells[0].source.replace(
        old, "> This is the **task** notebook. Fill in every `# YOUR CODE HERE`; the asserts check you "
             "as you go.\n> Solutions: `seq2seq_dates_and_spaces_solution.ipynb`.")

    with io.open(DST, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    body = DST.read_text(encoding="utf-8")
    for leak in LEAKS:
        if leak in body:
            raise SystemExit(f"LEAK: {leak!r} still present in {DST.name}")
    log.info(f"wrote {DST.name}: {len(nb.cells)} cells, {len(FN_STUBS)} functions stubbed, "
             f"birthday blanked, leak check clean ({len(LEAKS)} patterns)")


if __name__ == "__main__":
    main()
