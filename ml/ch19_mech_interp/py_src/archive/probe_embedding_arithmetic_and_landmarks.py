import time, torch
t=time.time()
from transformer_lens import HookedTransformer
print("import", time.time()-t)
torch.set_grad_enabled(False)
m = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print("load", time.time()-t)
def top(prompt,k=5):
    lp = m(prompt)[0,-1].log_softmax(-1)
    v,i = lp.topk(k)
    return [(m.tokenizer.decode(j), round(float(x.exp()),4)) for x,j in zip(v,i)]
for p in ["The Eiffel Tower is in the city of","The Colosseum is in the city of","Big Ben is in the city of","The capital of France is","Then, Alex and William went to the restaurant. Alex gave a book to"]:
    print(p, top(p))
WE = m.W_E
def tok(s):
    t = m.to_tokens(s, prepend_bos=False)[0]
    assert len(t)==1, (s,t)
    return t.item()
En = WE / WE.norm(dim=-1, keepdim=True)
for a,b,c in [(" king"," man"," woman"),(" Paris"," France"," Italy"),(" walked"," walk"," run"),(" cats"," cat"," dog"),(" bigger"," big"," small")]:
    v = WE[tok(a)]-WE[tok(b)]+WE[tok(c)]
    sims = En @ (v/v.norm())
    ex = {tok(a),tok(b),tok(c)}
    idx = [i for i in sims.topk(10).indices.tolist() if i not in ex][:5]
    print(a,"-",b,"+",c,"->",[(m.tokenizer.decode(i), round(float(sims[i]),3)) for i in idx])
