# Archive - superseded figure scripts

Kept as a record only. Not expected to run against the current decks, and nothing imports them.

| Script | What it did | Why it is here |
|---|---|---|
| `charlm_anim.py` | Hand-chosen ("illustrative") next-character probability bars on the word Պանիր, for L21's "The mechanics, click by click" (`fig/charlm_anim_0..5.pdf`). | Moved 2026-10-03: replaced by `../surname_gru.py`, which shows a trained GRU's real distributions on a validation surname (DECISIONS #64). |
| `charlm_generate.py` | A hand-written "typical progression" of char-LSTM output on Armenian text plus a schematic loss curve, for L21's demo and loss frames (`fig/charlm_gen_0..4.pdf`, `fig/charlm_loss.pdf`). | Moved 2026-10-03: replaced by `../surname_gru.py` (real samples at five checkpoints, real validation curve), same reason. |

Re-run either only to regenerate the old figures for comparison; their outputs are still in `../../fig/`, no longer referenced by any deck.
