from pathlib import Path

filepath = Path("data/mathlib_corpus.txt")

with open(filepath, "r") as f:
    file = f.read()

print("Corpus size:")
print(len(file))
