import torch
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
m = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
def top(prompt,k=3):
    lp = m(prompt)[0,-1].softmax(-1)
    v,i = lp.topk(k)
    return [(m.tokenizer.decode(j), round(float(x),3)) for x,j in zip(v,i)]
cands = [
 ("The Space Needle is located in downtown", " Seattle"),
 ("The Space Needle is in the city of", " Seattle"),
 ("Michael Jordan plays the sport of", " basketball"),
 ("Michael Jordan is a professional", " basketball"),
 ("Miles Davis plays the", " trumpet"),
 ("The Eiffel Tower is located in", " Paris"),
 ("The Eiffel Tower, located in", " Paris"),
 ("The Statue of Liberty is in", " New"),
 ("The Golden Gate Bridge is in", " San"),
 ("The Kremlin is in", " Moscow"),
 ("The Colosseum is located in", " Rome"),
 ("The Louvre is a museum in", " Paris"),
 ("Tokyo is the capital of", " Japan"),
 ("Paris is the capital of", " France"),
 ("Berlin is the capital of", " Germany"),
 ("Rome is the capital of", " Italy"),
 ("Madrid is the capital of", " Spain"),
 ("Moscow is the capital of", " Russia"),
 ("The capital of Japan is", " Tokyo"),
 ("The official language of France is", " French"),
 ("The mother tongue of Vladimir Putin is", " Russian"),
 ("Albert Einstein was born in", " Germany"),
 ("Barack Obama was born in", " Hawaii"),
 ("The Beatles were a band from", " Liverpool"),
 ("LeBron James plays the sport of", " basketball"),
 ("Lionel Messi plays the sport of", " soccer"),
 ("Roger Federer plays the sport of", " tennis"),
 ("Tiger Woods plays the sport of", " golf"),
 ("Steve Jobs was the founder of", " Apple"),
 ("Bill Gates is the founder of", " Microsoft"),
 ("Mark Zuckerberg is the founder of", " Facebook"),
 ("The iPhone is made by", " Apple"),
 ("Windows is an operating system made by", " Microsoft"),
 ("Toyota is a car company from", " Japan"),
 ("BMW is a car company from", " Germany"),
 ("Sushi is a dish from", " Japan"),
 ("Pizza is a dish from", " Italy"),
 ("The Great Wall is in", " China"),
 ("The Taj Mahal is in", " India"),
 ("Mount Everest is in", " Nepal"),
 ("The Nile is a river in", " Egypt"),
 ("Big Ben is located in", " London"),
 ("Buckingham Palace is in", " London"),
 ("The Pyramids of Giza are in", " Egypt"),
 ("The Sydney Opera House is in", " Sydney"),
 ("Shakespeare was born in", " England"),
 ("Mozart was a famous", " composer"),
 ("Picasso was a famous", " painter"),
]
for p,t in cands:
    tt = m.to_tokens(t, prepend_bos=False)[0][0].item()
    pr = m(p)[0,-1].softmax(-1)
    rank = int((pr > pr[tt]).sum())+1
    print(f"{rank:4d} {float(pr[tt]):.3f} | {p!r} -> {t!r}   top={top(p)}")
