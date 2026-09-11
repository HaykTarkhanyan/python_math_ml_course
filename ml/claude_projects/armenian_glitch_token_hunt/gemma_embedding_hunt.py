"""Gemma 2 embedding deep dive - without downloading the 18 GB model.

safetensors puts every tensor's byte range in a JSON header at the start of the
file, so we fetch ONLY model.embed_tokens.weight (256k x 3584 bf16, ~1.8 GB)
from the 9B repo with HTTP Range requests. Then the same analysis XLM-R got:
norms, Armenian subset, corpus counts, k-means, PCA map - closing the triangle
(corpus-null x low-norm x live-glitch) for a model that actually FAILED the
live repeat test (Վերցված, ֍, ֎, ֏, հղ).

Honest note: live probes ran on gemma-2-27b-it (9b is not served); same
tokenizer and training family, different embedding matrix.

Outputs: data/gemma_embed.bin (cached, resumable), data/gemma_all_norms.npz,
out/gemma_glitch.json, out/fig_gemma_map.png.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch

from tokenizer_registry import ARMENIAN_RE, load_tokenizer

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/gemma_embedding_hunt.py.log",
                                  encoding="utf-8")],
)
log = logging.getLogger("gemma_embedding_hunt")

REPO = "unsloth/gemma-2-9b"
SHARD = "model-00001-of-00004.safetensors"
TENSOR = "model.embed_tokens.weight"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
BIN = Path("data/gemma_embed.bin")
SEED = 509
SEP = "\n\n=====ARTICLE_BREAK=====\n\n"


def fetch_embedding() -> tuple[np.ndarray, list[int]]:
    s = requests.Session()
    head = s.get(URL, headers={"Range": "bytes=0-7"}, timeout=30)
    head.raise_for_status()
    hlen = int.from_bytes(head.content, "little")
    hjson = s.get(URL, headers={"Range": f"bytes=8-{7 + hlen}"}, timeout=60)
    meta = json.loads(hjson.content)
    ent = meta[TENSOR]
    assert ent["dtype"] == "BF16", f"expected BF16, got {ent['dtype']}"
    shape = ent["shape"]
    a, b = ent["data_offsets"]
    start, size = 8 + hlen + a, b - a
    log.info(f"{TENSOR}: shape={shape}, {size / 1e9:.2f} GB at offset {start:,}")

    done = BIN.stat().st_size if BIN.exists() else 0
    if done < size:
        log.info(f"downloading (resuming from {done / 1e9:.2f} GB)...")
        r = s.get(URL, stream=True, timeout=120,
                  headers={"Range": f"bytes={start + done}-{start + size - 1}"})
        r.raise_for_status()
        with open(BIN, "ab") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                if done % (1 << 28) < (1 << 22):
                    log.info(f"  {done / 1e9:.2f} / {size / 1e9:.2f} GB")
    if BIN.stat().st_size != size:
        raise RuntimeError(f"size mismatch: {BIN.stat().st_size} != {size}")
    log.info("embedding tensor on disk, computing norms...")

    buf = BIN.read_bytes()
    W = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(*shape)
    del buf
    norms = torch.cat([W[i:i + 16384].float().norm(dim=1)
                       for i in range(0, shape[0], 16384)]).numpy()
    return W, norms


def main() -> None:
    W, norms = fetch_embedding()
    np.savez_compressed("data/gemma_all_norms.npz", norms=norms)

    t = load_tokenizer("gemma2")
    surfaces = t.all_token_strings()
    arm = {i: s for i, s in surfaces.items() if ARMENIAN_RE.search(s)}
    del surfaces
    log.info(f"Armenian tokens in Gemma 2 vocab: {len(arm)}")

    corpus = Path("data/hy_wiki_20000000.txt")
    articles = corpus.read_text(encoding="utf-8").split(SEP)
    cnt = Counter()
    for k in range(0, len(articles), 200):
        for ids in t.raw(articles[k:k + 200], add_special_tokens=False)["input_ids"]:
            cnt.update(ids)
    del articles

    arm_ids = sorted(arm)
    forms = t.internal_forms(arm_ids)
    rows = [{"id": i, "internal": forms[k], "count": int(cnt.get(i, 0)),
             "norm": round(float(norms[i]), 4)} for k, i in enumerate(arm_ids)]
    by_norm = sorted(rows, key=lambda r: r["norm"])
    log.info("lowest-norm Armenian tokens in Gemma 2:")
    for r in by_norm[:12]:
        log.info(f"  norm={r['norm']:>8.3f} count={r['count']:>7} {r['internal']}")

    bottom_ids = np.argsort(norms)[:30]
    bottom = [{"id": int(i), "internal": t.internal_forms([int(i)])[0],
               "norm": round(float(norms[i]), 4),
               "count": int(cnt.get(int(i), 0))} for i in bottom_ids]

    # embedding map of the Armenian slice
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    E = W[arm_ids].float().numpy()
    del W
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    K = 5
    lab = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(En)
    xy = PCA(n_components=2, random_state=SEED).fit_transform(En)

    null_set = {i for i in arm_ids if cnt.get(i, 0) == 0}
    is_null = np.array([i in null_set for i in arm_ids])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xy[~is_null, 0], xy[~is_null, 1], s=14, c=BLUE, alpha=0.5,
               label="Armenian tokens")
    ax.scatter(xy[is_null, 0], xy[is_null, 1], s=90, c=RED, marker="X",
               label="corpus-null (incl. live-glitch)")
    for k, i in enumerate(arm_ids):
        if i in null_set:
            ax.annotate(forms[k], (xy[k, 0], xy[k, 1]), fontsize=8)
    ax.legend()
    ax.set_title("Gemma 2 (9B) Armenian token embeddings, PCA")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("out/fig_gemma_map.png", dpi=150)

    result = {
        "repo": REPO, "note": "live probes ran on gemma-2-27b-it (same tokenizer)",
        "n_armenian": len(arm),
        "norm_median_all": round(float(np.median(norms)), 4),
        "norm_median_armenian": round(float(np.median([r["norm"] for r in rows])), 4),
        "armenian_by_norm": by_norm[:25],
        "global_bottom30": bottom,
        "pca_map": {"x": np.round(xy[:, 0], 4).tolist(),
                    "y": np.round(xy[:, 1], 4).tolist(),
                    "token": forms,
                    "count": [int(cnt.get(i, 0)) for i in arm_ids],
                    "norm": [r["norm"] for r in rows],
                    "cluster": lab.tolist(),
                    "is_null": is_null.astype(int).tolist()},
    }
    Path("out/gemma_glitch.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote out/gemma_glitch.json + out/fig_gemma_map.png")


if __name__ == "__main__":
    sys.exit(main())
