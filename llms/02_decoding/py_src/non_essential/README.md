# non_essential - probes behind the decoding practicals

Not needed to build or execute the notebooks; they refresh a saved result that a notebook reads.

| Script | What it does, and when to re-run it |
|---|---|
| `probe_logprob_mismatch.py` | Sends the job-extraction endpoint test to every DeepSeek endpoint that failed with "logprobs do not line up", saves reply + logprob pieces to `results/job_extraction_logprob_mismatch.json` (read by section 4 of `xx_job_extraction_solution.ipynb`). Re-run after the endpoint test is redone. ~10 s, ~$0.0002. Added 2026-11-11. |
