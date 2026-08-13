"""Shared helpers for the chapter 19 IOI experiments.

Every figure script in this chapter needs the same four things: the model, a batch of
equal-length IOI prompts, the logit-difference metric, and per-head direct logit attribution.
They live here so the scripts stay short and the metric is defined exactly once.

The metric throughout the chapter is **logit difference**:

    logit_diff = logit(" Mary") - logit(" John")   at the final position

Not accuracy, not loss. Logit difference is linear in the residual stream, which is what makes
per-component attribution possible at all - the property L45 section 1 spends a frame on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

SEED = 509

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ioi_dataset.json"
FIG_DIR = Path(__file__).resolve().parents[1] / "fig"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODEL_NAME = "gpt2-small"


def setup_logging(script_name: str) -> logging.Logger:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{script_name}.log", mode="w", encoding="utf-8"),
        ],
    )
    # TransformerLens and huggingface_hub are chatty at INFO; we want our own lines readable.
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(script_name)


def load_model(log: logging.Logger) -> HookedTransformer:
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    log.info(f"loading {MODEL_NAME} on CPU")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    log.info(f"{model.cfg.n_layers} layers x {model.cfg.n_heads} heads, d_model={model.cfg.d_model}")
    return model


def load_prompts(model: HookedTransformer, log: logging.Logger, n: int = 128) -> dict:
    """Load IOI prompts and keep the largest equal-token-length group.

    Batching requires equal lengths, and padding would put pad tokens in the residual stream at
    positions the patching figures index by number. Filtering to one length keeps every position
    index meaningful across the whole batch, which is what the L46 heatmap needs.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run make_ioi_dataset.py first. Figures read the saved "
            f"dataset, they never regenerate it."
        )

    samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))["ioi"]

    by_length: dict[int, list[dict]] = {}
    for s in samples:
        length = len(model.to_tokens(s["clean"])[0])
        by_length.setdefault(length, []).append(s)

    best_len = max(by_length, key=lambda k: len(by_length[k]))
    group = by_length[best_len]
    log.info(
        f"token-length groups: "
        f"{ {k: len(v) for k, v in sorted(by_length.items())} }; using length {best_len}"
    )

    if len(group) < n:
        raise RuntimeError(
            f"only {len(group)} prompts at length {best_len}, need {n}. Widen N_IOI in "
            f"make_ioi_dataset.py or lower n."
        )

    group = group[:n]

    # The two corruptions must tokenize to the same length as the clean prompt, or patching
    # position-by-position compares different words. Check, do not assume.
    for key in ("corrupt_swap", "corrupt_abc"):
        lengths = {len(model.to_tokens(s[key])[0]) for s in group}
        if lengths != {best_len}:
            raise RuntimeError(f"{key} prompts have lengths {lengths}, expected {{{best_len}}}")

    clean = model.to_tokens([s["clean"] for s in group])
    corrupt_swap = model.to_tokens([s["corrupt_swap"] for s in group])
    corrupt_abc = model.to_tokens([s["corrupt_abc"] for s in group])

    answer_tokens = torch.tensor(
        [[model.to_single_token(s["answer"]), model.to_single_token(s["wrong_answer"])] for s in group]
    )

    log.info(f"batch: {len(group)} prompts, {best_len} tokens each")
    return {
        "samples": group,
        "clean": clean,
        "corrupt_swap": corrupt_swap,
        "corrupt_abc": corrupt_abc,
        "answer_tokens": answer_tokens,
        "seq_len": best_len,
    }


def logit_diff(logits: torch.Tensor, answer_tokens: torch.Tensor, per_prompt: bool = False):
    """logit(correct name) - logit(wrong name) at the final position."""
    final = logits[:, -1, :]
    correct = final.gather(1, answer_tokens[:, 0].unsqueeze(1)).squeeze(1)
    wrong = final.gather(1, answer_tokens[:, 1].unsqueeze(1)).squeeze(1)
    diff = correct - wrong
    return diff if per_prompt else diff.mean()


def logit_diff_directions(model: HookedTransformer, answer_tokens: torch.Tensor) -> torch.Tensor:
    """The residual-stream direction that logit difference reads off.

    Because the unembedding is linear, "how much did component X push toward ' Mary' over
    ' John'" is just the dot product of X's output with this direction.
    """
    residual_directions = model.tokens_to_residual_directions(answer_tokens)
    return residual_directions[:, 0, :] - residual_directions[:, 1, :]


def per_head_dla(model: HookedTransformer, cache, directions: torch.Tensor) -> torch.Tensor:
    """Direct logit attribution per attention head -> tensor [n_layers, n_heads].

    Each head's output at the final position is projected onto the logit-difference direction,
    after the same LayerNorm scaling the real forward pass applies. Skipping the LayerNorm step
    is the classic way to get attributions that look right and do not sum to the truth.
    """
    stack, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)

    # [component, batch, d_model] . [batch, d_model] -> [component], averaged over prompts
    attribution = torch.einsum("cbd,bd->cb", stack, directions).mean(dim=1)

    expected = model.cfg.n_layers * model.cfg.n_heads
    if attribution.shape[0] != expected:
        raise RuntimeError(
            f"expected {expected} head components, got {attribution.shape[0]} (labels: {labels[:3]}...)"
        )
    return attribution.reshape(model.cfg.n_layers, model.cfg.n_heads)


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def save_results(name: str, payload: dict, log: logging.Logger) -> Path:
    """Raw results as JSON first - the report and every figure derive from this file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"wrote {path.relative_to(REPO_ROOT)}")
    return path
