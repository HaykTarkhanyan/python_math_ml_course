"""Local BASE-model glitch probe: Qwen2.5-0.5B, raw token in, chaos out.

The original glitch tokens were found on base models with raw prompts - no chat
template, no instruction tuning to paper over a dead embedding. This is the
purest version of the experiment, and it runs free on this CPU:

  a) RAW continuation - the entire input is the single candidate token id
     (bypasses prompting and tokenization ambiguity completely), greedy decode.
  b) Few-shot echo - an 'Input: X / Output: X' pattern, then the candidate;
     we record the greedy output AND the exact probability the model assigns
     to copying the candidate token (quantitative glitchiness per token).

Model: Qwen/Qwen2.5-0.5B (base, ~1 GB) - its tokenizer's Armenian candidates
come from out/model_candidates.json. Controls: high-frequency Armenian chars.
Results -> out/local_base_probe.json.
"""

import json
import logging
import sys
from pathlib import Path

import torch

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/local_base_probe.py.log",
                                  encoding="utf-8")],
)
log = logging.getLogger("local_base_probe")

MODEL_ID = "Qwen/Qwen2.5-0.5B"
CONTROLS = ["ա", "ե", "բարև", "Երևան"]
GEN_TOKENS = 30


def main() -> None:
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"loading {MODEL_ID} (base) on CPU...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    mc = json.loads(Path("out/model_candidates.json").read_text(encoding="utf-8"))
    targets = [(c["surface"], c["count"]) for c in mc["qwen2.5"]["candidates"]
               if c["probeable"]]
    targets += [(w, None) for w in CONTROLS]
    log.info(f"probing {len(targets)} strings")

    def greedy(ids: list[int], n: int = GEN_TOKENS) -> str:
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens=n,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][len(ids):], skip_special_tokens=True)

    results = []
    for w, cnt in targets:
        wids = tok.encode(w, add_special_tokens=False)
        single = len(wids) == 1

        raw_cont = greedy(wids)

        shot = f"Echo test.\nInput: խնձոր\nOutput: խնձոր\nInput: {w}\nOutput:"
        sids = tok.encode(shot, add_special_tokens=False)
        echo_out = greedy(sids, n=max(4, len(wids) + 2))
        with torch.no_grad():
            logits = model(torch.tensor([sids])).logits[0, -1]
        probs = logits.softmax(-1)
        p_copy = float(probs[wids[0]])
        rank = int((probs > probs[wids[0]]).sum()) + 1

        entry = {"word": w, "corpus_count": cnt, "n_tokens": len(wids),
                 "single_token": single,
                 "raw_continuation": raw_cont.strip()[:200],
                 "echo_output": echo_out.strip()[:100],
                 "echo_ok": echo_out.strip().startswith(w),
                 "p_copy_first_token": round(p_copy, 5),
                 "copy_rank": rank}
        results.append(entry)
        log.info(f"{w:12s} count={cnt!s:>6} p_copy={p_copy:.4f} rank={rank:>6} "
                 f"echo={echo_out.strip()[:20]!r} raw={raw_cont.strip()[:40]!r}")

    out_path = Path("out/local_base_probe.json")
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model": MODEL_ID, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
