import torch
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
m = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
cands = [("Michael Jordan"," basketball"),("LeBron James"," basketball"),("Kobe Bryant"," basketball"),("Shaquille O'Neal"," basketball"),
 ("Tiger Woods"," golf"),("Phil Mickelson"," golf"),("Jack Nicklaus"," golf"),("Roger Federer"," tennis"),("Serena Williams"," tennis"),("Rafael Nadal"," tennis"),("Andy Murray"," tennis"),
 ("Tom Brady"," football"),("Peyton Manning"," football"),("Babe Ruth"," baseball"),("Derek Jeter"," baseball"),("Wayne Gretzky"," hockey"),("Sidney Crosby"," hockey"),
 ("Lionel Messi"," soccer"),("Cristiano Ronaldo"," soccer"),("David Beckham"," soccer"),("Usain Bolt"," sprint"),("Michael Phelps"," swimming")]
for name, ans in cands:
    p = f"{name} plays the sport of"
    t = m.to_tokens(p)
    pr = m(t)[0,-1].softmax(-1)
    a = m.to_single_token(ans)
    top = m.tokenizer.decode(int(pr.argmax()))
    print(f"{t.shape[1]:2d} {name:18s} {ans:12s} P={float(pr[a]):.3f} top={top!r} {[m.tokenizer.decode(int(x)) for x in t[0,1:-4]]}")
