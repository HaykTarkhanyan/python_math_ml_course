# Armenian Glitch Token Hunt

Token-level archaeology of eight tokenizers, asking what the world's tokenizer
infrastructure actually knows about Armenian. In the spirit of GPT-2's
` SolidGoldMagikarp`: vocab census, BPE merge-order archaeology, corpus
efficiency (the "Armenian premium" vs English), token-explosion words that cost
more tokens than letters, and a glitch-token hunt on XLM-R (corpus-null tokens
x embedding norms x a repeat-after-me MLM probe).

Built as a clean, minimal walkthrough intended for a live-coding session.

## Files

| File | What it is |
|---|---|
| `armenian_token_hunt.ipynb` | The experiment. Runs every analysis, saves JSONs to `out/`. |
| `tokenizer_registry.py` | The one shared abstraction: 8 tokenizers behind one tiny interface. Add a tokenizer = add one dict line. |
| `build_report.py` | Compiles `out/*.json` into a self-contained `out/report.html` (inline plotly). Never reruns the experiment. |
| `tokenizer_app.py` | Local Gradio playground: text in, colored token decomposition out. Accepts any HF repo id, plus a compare-all-tokenizers tab. |
| `out/` | JSON results (artifact of record), figures, report. |
| `data/` | Cached Wikipedia samples (gitignored). |

## Run

Everything uses the repo venv `ma/`:

```bash
# the experiment (~10-15 min: downloads wiki samples + XLM-R weights on first run)
../../../ma/Scripts/python.exe -m nbconvert --to notebook --execute --inplace armenian_token_hunt.ipynb

# the report (fast, JSONs only)
../../../ma/Scripts/python.exe build_report.py

# the playground
../../../ma/Scripts/python.exe tokenizer_app.py
```

Notebook knobs (env vars): `ATH_MAX_CHARS` (hy corpus chars, default 20M),
`ATH_EN_MAX_CHARS` (en baseline, default 5M), `ATH_SKIP_WEIGHTS=1` (skip the
1.1 GB XLM-R weight download and the embedding/probe analysis).

## Tokenizer lineup

XLM-R (deep dive), mBERT, GPT-2 / cl100k / o200k (tiktoken), Llama 3 (via the
ungated `Xenova/llama3-tokenizer` mirror), Gemma 2 (via `unsloth/gemma-2-9b`),
Qwen 2.5. Corpus: `wikimedia/wikipedia` `20231101.hy` sample, with a
`20231101.en` sample as the premium baseline.
