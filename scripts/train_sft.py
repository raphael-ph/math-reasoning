"""This is the script to fine-tune (SFT) the Formalizer Model"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from src.models.transformer import Transformer
from src.trainer.base import BaseTrainerConfig
from src.trainer.sft import SFTFormalizerDataset, SFTTrainer
from src.utils.logger import get_logger

# --- CONFIGURATION AND GLOBAL VARIABLES ---
# Paths
TOKENIZER_PATH = "data/vocab/tokenizer_vocab.json"
VOCAB_METADATA_PATH = "data/corpus/metadata.json"
CORPUS_PATH = "data/posttraining/metamath_sympy"
FINAL_MODEL_PATH = "models/sft/formalizer_v2/final_model.pt"
DEFAULT_BASE_MODEL_PATH = "models/formalizer/best_model.pt"

# ---------------- Global Vars --------------------
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)

VOCAB_SIZE = vocab_config["vocab_size"]
CONTEXT_SIZE = vocab_config["context_size"]
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
logger = get_logger("sft_trainer")
# -------------------------------------------------

config = BaseTrainerConfig(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    # we have 6M total tokens on the train split. With 6k iters, we hit 16 epochs, same as the
    # "Training language models to follow instructions with human feedback" paper
    # link: http://arxiv.org/abs/2203.02155
    max_iters=12_000, # enhancing now to 12k steps to try to force model to overfit
    eval_iters=50,
    eval_interval=200,
    checkpoint_interval=500,
    batch_size=16,
    n_embeddings=912,
    n_heads=12,
    n_layer=12,
    learning_rate=3e-5,
    warmup_steps=100,
    device="cuda",
    final_model_path=FINAL_MODEL_PATH,
)

train_ds = SFTFormalizerDataset(
    corpus_path=CORPUS_PATH,
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    split="train",
)

val_ds = SFTFormalizerDataset(
    corpus_path=CORPUS_PATH,
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    split="val",
)


def load_base_model(checkpoint_path: Path) -> Transformer:
    """Builds a fresh Transformer and loads pretrained weights to fine-tune from."""
    model = Transformer(
        vocab_size=config.vocab_size,
        emb_dim=config.n_embeddings,
        context_size=config.context_size,
        n_heads=config.n_heads,
        n_layers=config.n_layer,
    )

    state = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Path to the pretrained checkpoint to fine-tune (defaults to the best pretrained formalizer checkpoint)",
    )
    args = parser.parse_args()

    base_model_path = Path(args.base_model)
    logger.info(f"Loading base model from {base_model_path}")
    model = load_base_model(base_model_path)

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total/1e6:.1f}M")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
    )

    trainer.train()
