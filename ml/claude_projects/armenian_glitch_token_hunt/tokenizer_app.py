"""Local tokenizer playground.

Type text, pick a tokenizer, see the exact token decomposition - like
tiktokenizer.vercel.app, but running locally and accepting ANY Hugging Face
tokenizer repo in addition to the built-in registry.

Run:  python tokenizer_app.py   (opens http://127.0.0.1:7860)
"""

import logging
from pathlib import Path

import gradio as gr

from tokenizer_registry import TOKENIZERS, LoadedTokenizer, load_tokenizer

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/tokenizer_app.log", encoding="utf-8")],
)
log = logging.getLogger("tokenizer_app")

CUSTOM = "custom Hugging Face repo..."
CHOICES = list(TOKENIZERS) + [CUSTOM]

# readable tints of the Armenian flag colors, cycled over adjacent tokens
SPAN_COLORS = {"0": "#f2b8bd", "1": "#b8c9ee", "2": "#f9e2ad"}

_cache: dict[str, LoadedTokenizer] = {}


def get_tok(choice: str, custom_repo: str) -> LoadedTokenizer:
    if choice == CUSTOM:
        repo = custom_repo.strip()
        if not repo:
            raise gr.Error("Enter a Hugging Face repo id (e.g. openai-community/gpt2).")
        if repo not in _cache:
            log.info(f"loading custom tokenizer {repo}")
            from transformers import AutoTokenizer
            _cache[repo] = LoadedTokenizer(
                name=repo, label=repo, kind="hf",
                raw=AutoTokenizer.from_pretrained(repo))
        return _cache[repo]
    if choice not in _cache:
        log.info(f"loading {choice}")
        _cache[choice] = load_tokenizer(choice)
    return _cache[choice]


def tokenize(text: str, choice: str, custom_repo: str):
    t = get_tok(choice, custom_repo)
    ids = t.encode(text)
    pieces = t.token_pieces(ids)
    internal = t.internal_forms(ids)
    spans = [(p, str(i % len(SPAN_COLORS))) for i, p in enumerate(pieces)]
    n_chars = len(text)
    stats = (f"**{t.label}** - **{len(ids)} tokens** for {n_chars} chars "
             f"({len(text.encode('utf-8'))} bytes) -> "
             f"{n_chars / len(ids):.2f} chars/token" if ids else "0 tokens")
    table = [[i, tid, internal[i], pieces[i]] for i, tid in enumerate(ids)]
    return stats, spans, table


def compare_all(text: str):
    rows = []
    for name in TOKENIZERS:
        t = get_tok(name, "")
        ids = t.encode(text)
        rows.append([t.label, len(ids),
                     round(len(text) / len(ids), 2) if ids else 0.0,
                     " | ".join(t.token_pieces(ids))])
    rows.sort(key=lambda r: r[1])
    return rows


DEMO_TEXT = "Բարև ձեզ, ես Երևանից եմ։ Շնորհակալություն, որ եկել եք։"

with gr.Blocks(title="Armenian Tokenizer Playground") as app:
    gr.Markdown("# Tokenizer playground\n"
                "See how any tokenizer decomposes your text. Built for the "
                "Armenian glitch token hunt.")
    with gr.Tab("Tokenize"):
        with gr.Row():
            choice = gr.Dropdown(CHOICES, value="xlm-roberta", label="Tokenizer")
            custom_repo = gr.Textbox(label="Custom HF repo id",
                                     placeholder="e.g. openai-community/gpt2")
        text = gr.Textbox(value=DEMO_TEXT, lines=4, label="Text")
        btn = gr.Button("Tokenize", variant="primary")
        stats = gr.Markdown()
        spans = gr.HighlightedText(label="Tokens", color_map=SPAN_COLORS,
                                   show_legend=False, show_inline_category=False,
                                   combine_adjacent=False)
        table = gr.Dataframe(headers=["#", "id", "internal form", "surface piece"],
                             label="Token table", interactive=False)
        inputs = [text, choice, custom_repo]
        btn.click(tokenize, inputs, [stats, spans, table])
        text.submit(tokenize, inputs, [stats, spans, table])
    with gr.Tab("Compare all"):
        text2 = gr.Textbox(value=DEMO_TEXT, lines=4, label="Text")
        btn2 = gr.Button("Compare", variant="primary")
        cmp = gr.Dataframe(headers=["tokenizer", "tokens", "chars/token", "pieces"],
                           label="All registry tokenizers (fewest tokens first)",
                           interactive=False, wrap=True)
        btn2.click(compare_all, [text2], [cmp])

if __name__ == "__main__":
    log.info("starting app")
    app.launch(inbrowser=True, show_error=True)
