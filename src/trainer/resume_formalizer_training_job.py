import argparse
from pathlib import Path

from src.models.transformer import Transformer
from src.trainer.formalizer_trainer import FormalizerTrainer, BaseTrainerConfig, FormalizerDataset
from src.utils.logger import get_logger
from tokenizers import Tokenizer as HFTokenizer

logger = get_logger("formalizer_resume", level="DEBUG")

TOKENIZER_PATH = "data/vocab/fast_tokenizer.json"
TRAIN_PATH = Path("data/corpus/train.txt")
VAL_PATH = Path("data/corpus/val.txt")

parser = argparse.ArgumentParser(description="Resume formalizer training from an MLflow artifact checkpoint.")
parser.add_argument("--step", type=int, required=True, help="Step number of the checkpoint to resume from")
parser.add_argument("--run", type=str, required=True, help="MLflow run ID to resume")
args = parser.parse_args()

logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = HFTokenizer.from_file(TOKENIZER_PATH)
pad_id = tokenizer.token_to_id("<|pad|>")
tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>")
vocab_size = tokenizer.get_vocab_size()
logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")

training_config_dict = {
    "vocab_size": vocab_size,
    "context_size": 1024,
    "max_iters": 10000,
    "eval_interval": 500,
    "eval_iters": 50,
    "checkpoint_interval": 2000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "n_embeddings": 512,
    "n_heads": 8,
    "n_layer": 8,
    "device": "cuda"
}

config = BaseTrainerConfig(**training_config_dict)
logger.info(f"Configuration: {config.model_dump()}")

logger.info("Initializing datasets...")
train_dataset = FormalizerDataset(corpus_path=TRAIN_PATH, tokenizer=tokenizer, context_size=config.context_size)
val_dataset = FormalizerDataset(corpus_path=VAL_PATH, tokenizer=tokenizer, context_size=config.context_size)

logger.info("Initializing model architecture...")
formalizer_model = Transformer(
    vocab_size=config.vocab_size,
    emb_dim=config.n_embeddings,
    context_size=config.context_size,
    n_layers=config.n_layer,
    n_heads=config.n_heads,
)

trainer = FormalizerTrainer(
    model=formalizer_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)

logger.info(f"Resuming from MLflow artifact — run={args.run}, step={args.step}")
trainer.resume_from_mlflow_artifact(run_id=args.run, step=args.step)
