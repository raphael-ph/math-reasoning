import json
from pathlib import Path
from src.models.transformer import Transformer
from src.inference.formalizer_engine import FormalizerInference

# --- CONFIGURATION ----
## Paths
VOCAB_METADATA_PATH = "./data/corpus/metadata.json"
## Vars
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)
CONTEXT_SIZE = vocab_config["context_size"]
VOCAB_SIZE = vocab_config["vocab_size"]
# ---------------------
model_path = Path("models/formalizer/best_model.pt")
vocab_size=VOCAB_SIZE
context_size=CONTEXT_SIZE
n_embeddings=912
n_heads=12
n_layer=12

model = Transformer(vocab_size=vocab_size, 
            emb_dim=n_embeddings, 
            context_size=context_size,
            n_heads=n_heads,
            n_layers=n_layer
        )

formalizer = FormalizerInference(model_path=model_path, model=model)

if __name__ == "__main__":
    input_text = "import sympy"
    print(60*"=")
    print("TEST WITH NEWLY TRAINED MODEL")
    print(60*"=")
    print(formalizer.run(input_text))