"""This is the script to train the Formalizer Model"""
import json

from tokenizers import Tokenizer
from src.models.transformer import Transformer
from src.trainer.base import BaseTrainerConfig
from src.trainer.formalizer_trainer import FormalizerDataset, FormalizerTrainer
from src.utils.logger import get_logger

# --- CONFIGURATION AND GLOBAL VARIABLES ---
# Paths
TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
VOCAB_METADATA_PATH = "data/corpus/metadata.json"
CORPUS_PATH = "data/corpus/corpus.bin"
TRAIN_INDICES = "data/corpus/train_indices.npy"
VAL_INDICES = "data/corpus/val_indices.npy"
FINAL_MODEL_PATH = "models/formalizer/final_model.pt"

# ---------------- Global Vars --------------------
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)

VOCAB_SIZE = vocab_config["vocab_size"]
CONTEXT_SIZE = vocab_config["context_size"]
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
logger = get_logger("trainer")
# -------------------------------------------------

config = BaseTrainerConfig(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    max_iters=200_000,
    eval_iters=50,
    eval_interval=1000,
    checkpoint_interval=5000,
    batch_size=32,
    n_embeddings=1280,
    n_heads=16,
    n_layer=24,
    learning_rate=3e-4,
    device="cuda",
    final_model_path=FINAL_MODEL_PATH,
)

train_ds = FormalizerDataset(
    corpus_path=CORPUS_PATH,
    chunk_indices_path=TRAIN_INDICES,
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE
)

val_ds = FormalizerDataset(
    corpus_path=CORPUS_PATH,
    chunk_indices_path=VAL_INDICES,
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE
)

model = Transformer(vocab_size=config.vocab_size, 
                    emb_dim=config.n_embeddings, 
                    context_size=config.context_size,
                    n_heads=config.n_heads,
                    n_layers=config.n_layer)

total = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total/1e6:.1f}M")

trainer = FormalizerTrainer(model=model,
                            train_dataset=train_ds,
                            val_dataset=val_ds,
                            config=config)

if __name__ == "__main__":
    trainer.train()