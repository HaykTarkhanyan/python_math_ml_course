"""Generate the Indirect Object Identification (IOI) dataset used across chapter 19.

The chapter's running example is a single sentence shape:

    "Then, Mary and John went to the store. John gave a drink to" -> " Mary"

Every technique in L45-L47 lands on this task, so the dataset has to serve all of them:

* **clean prompts** for logit lens, attention patterns and direct logit attribution;
* **two corruption types** for activation patching, which answer different questions;
* **a balanced control set** for the L45 linear-probe figure, where half the prompts have a
  duplicated name and half do not.

The one correctness detail that matters more than it looks: **every name must be a single GPT-2
token when preceded by a space**. Logit difference is computed between two single-token answers;
if a name splits into two tokens the metric silently measures something else. The script filters
the name list against the real tokenizer and fails loudly if too few survive.

Output: ``ml/ch19_mech_interp/data/ioi_dataset.json`` (the artifact of record - figures read this
file, they never regenerate the data).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from transformers import AutoTokenizer

SEED = 509

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = OUT_DIR / "ioi_dataset.json"

N_IOI = 512
N_CONTROL = 512

# Filtered against the GPT-2 tokenizer at run time - this is the candidate pool, not the final list.
CANDIDATE_NAMES = [
    "Mary", "John", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Anna", "Rose",
    "Paul", "Kate", "Mark", "Alex", "Sam", "Chris", "Nick", "Emily", "Laura", "Peter",
    "Sarah", "David", "Jessica", "Robert", "Michael", "Jennifer", "William", "Linda",
    "Richard", "Susan", "Joseph", "Karen", "Thomas", "Nancy", "Charles", "Betty",
    "Daniel", "Helen", "Matthew", "Sandra", "Anthony", "Donna", "Steven", "Carol",
]

PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "station", "museum"]
OBJECTS = ["drink", "book", "kiss", "ring", "bone", "basketball", "computer", "necklace"]

# Both orders put the repeated name (S) in the "gave" clause; they differ in which name is
# mentioned first. Wang et al. call these BABA and ABBA.
TEMPLATES = {
    "BABA": "Then, {s} and {io} went to the {place}. {s} gave a {obj} to",
    "ABBA": "Then, {io} and {s} went to the {place}. {s} gave a {obj} to",
}


def setup_logging() -> logging.Logger:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "make_ioi_dataset.log", mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def single_token_names(tokenizer, candidates: list[str], log: logging.Logger) -> list[str]:
    """Keep only names that are one GPT-2 token when preceded by a space."""
    kept, dropped = [], []
    for name in candidates:
        if len(tokenizer.encode(f" {name}")) == 1:
            kept.append(name)
        else:
            dropped.append(name)

    log.info(f"single-token names: kept {len(kept)}, dropped {len(dropped)}")
    if dropped:
        log.info(f"dropped (multi-token with leading space): {', '.join(dropped)}")

    if len(kept) < 10:
        raise RuntimeError(
            f"only {len(kept)} single-token names survived filtering; need at least 10 for a "
            f"varied dataset. Add more candidates to CANDIDATE_NAMES."
        )
    return kept


def build_ioi_samples(rng: random.Random, names: list[str], n: int) -> list[dict]:
    """Clean IOI prompts plus their two corruptions.

    ``corrupt_swap``  - swap S and IO, so the correct answer flips to the other name.
                        Tests "does this component carry *which* name to copy?"
    ``corrupt_abc``   - replace S with an unrelated third name, so no name is duplicated.
                        Tests "does this component depend on the duplication signal at all?"
    """
    samples = []
    for i in range(n):
        order = "BABA" if i % 2 == 0 else "ABBA"
        io, s, c = rng.sample(names, 3)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        template = TEMPLATES[order]

        samples.append(
            {
                "order": order,
                "io": io,
                "s": s,
                "c": c,
                "place": place,
                "object": obj,
                "clean": template.format(s=s, io=io, place=place, obj=obj),
                "corrupt_swap": template.format(s=io, io=s, place=place, obj=obj),
                "corrupt_abc": template.format(s=s, io=io, place=place, obj=obj).replace(
                    f". {s} gave", f". {c} gave"
                ),
                "answer": f" {io}",
                "wrong_answer": f" {s}",
            }
        )
    return samples


def build_probe_samples(rng: random.Random, names: list[str], n: int) -> list[dict]:
    """Balanced set for the L45 probe figure: half duplicate a name, half do not.

    The probe question is "is one of these names repeated?" - the thing the duplicate-token
    heads are supposed to compute. Without the negative half the probe has nothing to separate
    and its accuracy is meaningless.
    """
    samples = []
    for i in range(n):
        has_duplicate = i % 2 == 0
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)

        if has_duplicate:
            io, s = rng.sample(names, 2)
            text = TEMPLATES["BABA"].format(s=s, io=io, place=place, obj=obj)
        else:
            # Same sentence shape, three distinct names, so the only difference the probe can
            # pick up is the duplication itself - not sentence length or structure.
            n1, n2, n3 = rng.sample(names, 3)
            text = TEMPLATES["BABA"].format(s=n1, io=n2, place=place, obj=obj)
            text = text.replace(f". {n1} gave", f". {n3} gave")

        samples.append({"text": text, "has_duplicate": has_duplicate})
    return samples


def main() -> None:
    log = setup_logging()
    rng = random.Random(SEED)

    log.info("loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    names = single_token_names(tokenizer, CANDIDATE_NAMES, log)

    ioi = build_ioi_samples(rng, names, N_IOI)
    probe = build_probe_samples(rng, names, N_CONTROL)

    orders = {o: sum(1 for s in ioi if s["order"] == o) for o in TEMPLATES}
    dupes = sum(1 for s in probe if s["has_duplicate"])
    log.info(f"IOI samples: {len(ioi)} ({orders})")
    log.info(f"probe samples: {len(probe)} ({dupes} with duplicate, {len(probe) - dupes} without)")

    if orders["BABA"] != orders["ABBA"]:
        raise RuntimeError(f"IOI template orders are unbalanced: {orders}")
    if dupes * 2 != len(probe):
        raise RuntimeError(f"probe set is unbalanced: {dupes} of {len(probe)} have a duplicate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "names": names,
        "places": PLACES,
        "objects": OBJECTS,
        "templates": TEMPLATES,
        "ioi": ioi,
        "probe": probe,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")

    log.info("example clean       : %s -> %r", ioi[0]["clean"], ioi[0]["answer"])
    log.info("example corrupt_swap: %s -> %r", ioi[0]["corrupt_swap"], ioi[0]["wrong_answer"])
    log.info("example corrupt_abc : %s", ioi[0]["corrupt_abc"])


if __name__ == "__main__":
    main()
