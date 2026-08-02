"""Derive HW1_sae_rnn.ipynb (student task version) from the executed solution notebook.

Keeps every markdown cell, every assert, and all the infrastructure. Blanks only the pieces that
ARE the lesson: the SAE forward pass, the unit-norm trick, the sparse loss, the probe, and the
ablation intervention.

Fails loudly if any target is missing. The silent failure mode here is shipping a task notebook
that still contains the answers, and no student would ever report that.

One source of truth: edit build_sae_nb.py, re-execute the solution, then re-run this.
"""
import io
import re
from pathlib import Path

import nbformat as nbf

CH = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course\ml\ch8_autoencoders")
SRC, DST = "HW1_sae_rnn_solution.ipynb", "HW1_sae_rnn.ipynb"


def stub_function(src, name, hint):
    """Keep `def name(...)` and its docstring; replace the rest of the body with a stub."""
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if re.match(rf"^(\s*)def {re.escape(name)}\s*\(", ln):
            indent = len(ln) - len(ln.lstrip())
            body = " " * (indent + 4)

            # one-line form:  def f(self, x): return expr
            one = re.match(rf"^\s*def {re.escape(name)}\s*\(.*\)\s*:\s*(\S.*)$", ln)
            if one:
                sig_end = ln.rindex(":", 0, ln.index(one.group(1)))
                new = [ln[:sig_end + 1],
                       f"{body}# YOUR CODE HERE - {hint}",
                       f'{body}raise NotImplementedError("{hint}")']
                return "\n".join(lines[:i] + new + lines[i + 1:])

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
            new = [f"{body}# YOUR CODE HERE - {hint}",
                   f'{body}raise NotImplementedError("{hint}")', ""]
            return "\n".join(lines[:j] + new + lines[k:])
    raise SystemExit(f"stub_function: {name!r} not found")


FN_STUBS = [
    ("encode", "z = ReLU((h - b_dec) @ W_enc + b_enc)"),
    ("decode", "h_hat = z @ W_dec + b_dec"),
    ("normalize_decoder",
     "divide each decoder ROW by its norm so every dictionary direction is a unit vector"),
    ("probe_f1", "fit LogisticRegression on Xm, predict, and return the F1 score"),
]

BLOCK_EDITS = [
    (
        "            loss = ((xh-xb)**2).sum(1).mean() + l1*z.abs().sum(1).mean()",
        "            # YOUR CODE HERE - the SAE objective: squared reconstruction error per\n"
        "            # vector, plus l1 times the L1 norm of the code. Both averaged over the batch.\n"
        "            loss = None",
    ),
    (
        "    z1 = z0.clone(); z1[:, feat] = 0.0\n"
        "    abl  = sae.decode(z1)/scale + mu",
        "    # YOUR CODE HERE - ablate the feature: copy the code, set column `feat` to zero,\n"
        "    # decode it back, and undo the normalization (divide by scale, add mu).\n"
        "    abl  = None",
    ),
]


def main():
    nb = nbf.read(str(CH / SRC), as_version=4)
    done_fn, done_bl = set(), set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        src = cell.source
        if cell.cell_type == "code":
            for name, hint in FN_STUBS:
                if re.search(rf"^\s*def {re.escape(name)}\s*\(", src, re.M):
                    src = stub_function(src, name, hint)
                    done_fn.add(name)
            for needle, repl in BLOCK_EDITS:
                if needle in src:
                    src = src.replace(needle, repl)
                    done_bl.add(needle[:35])
        cell.source = src

    missing = [n for n, _ in FN_STUBS if n not in done_fn]
    if missing:
        raise SystemExit(f"never stubbed: {missing}")
    if len(done_bl) != len(BLOCK_EDITS):
        raise SystemExit(f"blocks applied {len(done_bl)}/{len(BLOCK_EDITS)}")

    nb.cells[0].source = nb.cells[0].source.replace(
        "> This is the **solution** notebook; the task version is `HW1_sae_rnn.ipynb`.",
        "> This is the **task** notebook. Solutions: `HW1_sae_rnn_solution.ipynb`.")

    with io.open(CH / DST, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"wrote {DST}: {len(nb.cells)} cells, {len(FN_STUBS)} functions stubbed, "
          f"{len(BLOCK_EDITS)} blocks blanked")

    # paranoia: the answers must not survive anywhere in the task notebook
    body = (CH / DST).read_text(encoding="utf-8")
    for leak in ("z @ self.W_dec + self.b_dec", "l1*z.abs().sum(1).mean()"):
        if leak in body:
            raise SystemExit(f"LEAK: {leak!r} still present in the task notebook")
    print("leak check: clean")


if __name__ == "__main__":
    main()
