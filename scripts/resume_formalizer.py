"""Resume formalizer training from an MLflow artifact checkpoint."""
import argparse
import json

from tokenizers import Tokenizer
from src.models.transformer import Transformer
from src.trainer.base import BaseTrainerConfig
from src.trainer.formalizer_trainer import FormalizerDataset, FormalizerTrainer
from src.utils.logger import get_logger

# --- CONFIGURATION AND GLOBAL VARIABLES ---
TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
VOCAB_METADATA_PATH = "data/corpus/metadata.json"
CORPUS_PATH = "data/corpus/corpus.bin"
TRAIN_INDICES = "data/corpus/train_indices.npy"
VAL_INDICES = "data/corpus/val_indices.npy"
FINAL_MODEL_PATH = "models/formalizer/final_model.pt"

with open(VOCAB_METADATA_PATH, "rb") as file:
    vocab_config = json.loads(file.read())

VOCAB_SIZE = vocab_config["vocab_size"]
CONTEXT_SIZE = vocab_config["context_size"]
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
logger = get_logger("formalizer_resume")

config = BaseTrainerConfig(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    max_iters=600_000,
    eval_iters=50,
    eval_interval=1000,
    checkpoint_interval=5000,
    batch_size=16,
    n_embeddings=912,
    n_heads=12,
    n_layer=12,
    learning_rate=3e-4,
    warmup_steps=2000,
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

model = Transformer(
    vocab_size=config.vocab_size,
    emb_dim=config.n_embeddings,
    context_size=config.context_size,
    n_heads=config.n_heads,
    n_layers=config.n_layer,
)

total = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total/1e6:.1f}M")

trainer = FormalizerTrainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    config=config
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to resume from")
    parser.add_argument("--run", type=str, required=True, help="MLflow run ID to resume")
    parser.add_argument("--path", type=str, default=None, help="Explicit checkpoint path (defaults to models/formalizer/checkpoint_{step}.pt)")
    args = parser.parse_args()

    from pathlib import Path
    checkpoint_path = Path(args.path) if args.path else Path(f"models/formalizer/checkpoint_{args.step}.pt")
    logger.info(f"Resuming from {checkpoint_path} — run={args.run}, step={args.step}")
    trainer.resume_from_state_dict(checkpoint_path=checkpoint_path, run_id=args.run, step=args.step)
