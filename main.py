import json
from pathlib import Path
from src.models.transformer import Transformer
from src.inference.formalizer_engine import FormalizerInference, GenerationConfig

# --- CONFIGURATION ----
## Paths
VOCAB_METADATA_PATH = "./data/corpus/metadata.json"
BASE_MODEL_PATH = Path("models/formalizer/best_model.pt")
SFT_MODEL_PATH_2 = Path("models/sft/formalizer_v2/final_model.pt")
SFT_MODEL_PATH_3 = Path("models/sft/formalizer_v3/final_model.pt")
## Vars
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)
CONTEXT_SIZE = vocab_config["context_size"]
VOCAB_SIZE = vocab_config["vocab_size"]
# ---------------------
vocab_size=VOCAB_SIZE
context_size=CONTEXT_SIZE
n_embeddings=912
n_heads=12
n_layer=12

def build_model() -> Transformer:
    return Transformer(vocab_size=vocab_size,
                emb_dim=n_embeddings,
                context_size=context_size,
                n_heads=n_heads,
                n_layers=n_layer
            )

generation_config = GenerationConfig(top_p=0.9, temperature=0.8)

base_formalizer = FormalizerInference(model_path=BASE_MODEL_PATH, model=build_model(), generation_config=generation_config)
sft_formalizer_2 = FormalizerInference(model_path=SFT_MODEL_PATH_2, model=build_model(), generation_config=generation_config)
sft_formalizer_3 = FormalizerInference(model_path=SFT_MODEL_PATH_3, model=build_model(), generation_config=generation_config)


# A natural-language, informal math resolution — this is exactly what the Formalizer
# is meant to translate into sympy code.
TEST_PROMPT = (
    "Natalia sold clips to 48 of her friends in April. Then she sold half as many "
    "clips in May. So she sold 48 / 2 = 24 clips in May. In total she sold "
    "48 + 24 = 72 clips."
)

if __name__ == "__main__":
    print(60*"=")
    print("BASE MODEL (pretrained, no SFT)")
    print(60*"=")
    print(f"Generation config: {base_formalizer.generation_config}")
    print(f"EOS token id (resolved from tokenizer): {base_formalizer.eos_token_id}")
    print(base_formalizer.run(TEST_PROMPT))

    print(60*"=")
    print("SFT MODEL 2")
    print(60*"=")
    print(f"Generation config: {sft_formalizer_2.generation_config}")
    print(f"EOS token id (resolved from tokenizer): {sft_formalizer_2.eos_token_id}")
    sft_prompt = f"<|bos|> <|user|> {TEST_PROMPT} <|assistant|>"
    print(sft_formalizer_2.run(sft_prompt))

    print(60*"=")
    print("SFT MODEL 3")
    print(60*"=")
    print(f"Generation config: {sft_formalizer_3.generation_config}")
    print(f"EOS token id (resolved from tokenizer): {sft_formalizer_3.eos_token_id}")
    sft_prompt = f"<|bos|> <|user|> {TEST_PROMPT} <|assistant|>"
    print(sft_formalizer_3.run(sft_prompt))