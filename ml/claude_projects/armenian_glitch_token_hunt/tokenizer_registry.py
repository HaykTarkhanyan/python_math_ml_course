"""Unified access to the tokenizers under study.

One dict (TOKENIZERS) declares everything; load_tokenizer() wraps each entry in a
LoadedTokenizer with the same tiny surface, so every analysis in the notebook and
the Gradio app runs on any tokenizer without special cases.

Two kinds:
  - "hf"       : Hugging Face fast tokenizers (SentencePiece, WordPiece, byte-BPE)
  - "tiktoken" : OpenAI encodings (byte-level BPE with public merge ranks)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Armenian main block + the five Armenian presentation-form ligatures.
ARMENIAN_RE = re.compile(r"[԰-֏ﬓ-ﬗ]")

TOKENIZERS: dict[str, dict] = {
    "xlm-roberta": {"kind": "hf", "repo": "FacebookAI/xlm-roberta-base",
                    "label": "XLM-R (SentencePiece, 250k, 2019)"},
    "mbert":       {"kind": "hf", "repo": "google-bert/bert-base-multilingual-cased",
                    "label": "mBERT (WordPiece, 120k, 2018)"},
    "gpt2":        {"kind": "tiktoken", "encoding": "gpt2",
                    "label": "GPT-2 (byte BPE, 50k, 2019)"},
    "cl100k":      {"kind": "tiktoken", "encoding": "cl100k_base",
                    "label": "cl100k / GPT-4 (byte BPE, 100k, 2023)"},
    "o200k":       {"kind": "tiktoken", "encoding": "o200k_base",
                    "label": "o200k / GPT-4o (byte BPE, 200k, 2024)"},
    "llama3":      {"kind": "hf", "repo": "Xenova/llama3-tokenizer",
                    "label": "Llama 3 (byte BPE, 128k, 2024)"},
    "gemma2":      {"kind": "hf", "repo": "unsloth/gemma-2-9b",
                    "label": "Gemma 2 (SentencePiece, 256k, 2024)"},
    "qwen2.5":     {"kind": "hf", "repo": "Qwen/Qwen2.5-7B",
                    "label": "Qwen 2.5 (byte BPE, 152k, 2024)"},
}


@dataclass
class LoadedTokenizer:
    name: str
    label: str
    kind: str          # "hf" | "tiktoken"
    raw: object        # the underlying tokenizer object

    def encode(self, text: str) -> list[int]:
        if self.kind == "hf":
            return self.raw.encode(text, add_special_tokens=False)
        return self.raw.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self.raw.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self.kind == "hf":
            return len(self.raw)
        return self.raw.n_vocab

    def token_pieces(self, ids: list[int]) -> list[str]:
        """Per-token surface strings, in order. Partial UTF-8 bytes show as the
        replacement char - that is real information, not a bug."""
        if self.kind == "hf":
            return [self.raw.decode([i]) for i in ids]
        return [b.decode("utf-8", errors="replace")
                for b in self.raw.decode_tokens_bytes(ids)]

    def internal_forms(self, ids: list[int]) -> list[str]:
        """The tokenizer's own spelling of each token (with meta-symbols like
        the SentencePiece low line, WordPiece ##, or byte-BPE G-dot)."""
        if self.kind == "hf":
            return self.raw.convert_ids_to_tokens(ids)
        out = []
        for i in ids:
            try:
                out.append(str(self.raw.decode_single_token_bytes(i))[2:-1])
            except KeyError:  # special token
                out.append(f"<special {i}>")
        return out

    def all_token_strings(self) -> dict[int, str]:
        """id -> decoded surface string for the ENTIRE vocab (special tokens excluded
        for hf where possible). This is the basis for every vocab census."""
        if self.kind == "hf":
            ids = list(range(len(self.raw)))
            pieces = self.raw.batch_decode([[i] for i in ids],
                                           clean_up_tokenization_spaces=False)
            return dict(zip(ids, pieces))
        out = {}
        for i in range(self.raw.n_vocab):
            try:
                out[i] = self.raw.decode_single_token_bytes(i).decode(
                    "utf-8", errors="replace")
            except KeyError:  # gaps / special tokens in tiktoken tables
                continue
        return out


def load_tokenizer(name: str) -> LoadedTokenizer:
    spec = TOKENIZERS[name]
    if spec["kind"] == "hf":
        from transformers import AutoTokenizer
        raw = AutoTokenizer.from_pretrained(spec["repo"])
    else:
        import tiktoken
        raw = tiktoken.get_encoding(spec["encoding"])
    return LoadedTokenizer(name=name, label=spec["label"], kind=spec["kind"], raw=raw)


def load_all() -> dict[str, LoadedTokenizer]:
    return {name: load_tokenizer(name) for name in TOKENIZERS}
