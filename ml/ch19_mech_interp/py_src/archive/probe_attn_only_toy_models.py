import torch, time
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
torch.set_num_threads(4)
for name in ["attn-only-1l", "attn-only-2l"]:
    t = time.time()
    m = HookedTransformer.from_pretrained(name, device="cpu")
    c = m.cfg
    print(f"== {name} loaded in {time.time()-t:.0f}s: layers {c.n_layers} heads {c.n_heads} d_model {c.d_model} d_head {c.d_head} vocab {c.d_vocab} pos {c.positional_embedding_type} norm {c.normalization_type} act {c.act_fn} tokenizer {c.tokenizer_name}")
    gen = torch.Generator().manual_seed(509)
    L = 50
    rand = torch.randint(100, 20000, (20, L), generator=gen)
    bos = torch.full((20, 1), m.tokenizer.bos_token_id if m.tokenizer.bos_token_id is not None else 0)
    seq = torch.cat([bos, rand, rand], 1)
    logits, cache = m.run_with_cache(seq)
    pred = logits[:, :-1].argmax(-1); corr = (pred == seq[:, 1:]).float()
    print("  top-1 first copy", corr[:, :L].mean().item(), "second copy", corr[:, L:].mean().item())
    for layer in range(c.n_layers):
        pat = cache["pattern", layer]  # b h q k
        prev = torch.stack([pat[:, :, i, i-1] for i in range(1, seq.shape[1])], -1).mean((0, 2))
        ind = torch.stack([pat[:, :, i, i-L+1] for i in range(L+1, seq.shape[1])], -1).mean((0, 2))
        print(f"  layer {layer} prev-token score", [round(x, 2) for x in prev.tolist()])
        print(f"  layer {layer} induction score ", [round(x, 2) for x in ind.tolist()])
    if c.n_layers == 2:
        for mode in ["Q", "K", "V"]:
            cs = m.all_composition_scores(mode)
            print(f"  {mode}-composition shape {tuple(cs.shape)}; layer0->layer1 block:")
            blk = cs[0, :, 1, :]
            for h in range(c.n_heads):
                print("   ", h, [round(x, 3) for x in blk[h].tolist()])
