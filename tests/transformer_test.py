import torch
from src.models.transformer import Transformer
from src.preprocessing.pre_training.tokenizer import Tokenizer

# Tokenizer
tokenizer = Tokenizer()
with open("/Users/rcss/Documents/master/math-reasoning/data/corpus/mathlib_corpus.txt", "r") as f:
    text = f.read()
    tokenizer.train(text, verbose=True)

model = Transformer(vocab_size=1000, emb_dim=156, context_size=128, n_heads=1, n_layers=2)
tok, loss = model.generate(torch.zeros(1, 1, dtype=torch.long), 50)

# decoding tokens
out = tokenizer.decode(tok)

print(out)