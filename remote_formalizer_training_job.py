# --- Formalizer Training Entrypoint ---
# This is a remote entrypoint to launch the training job to Vast AI

from pathlib import Path

# internal imports
from src.models.transformer import Transformer
from src.trainer.formalizer_trainer import FormalizerTrainer, BaseTrainerConfig, FormalizerDataset
from src.utils.logger import get_logger
from tokenizers import Tokenizer as HFTokenizer

# setting up log
logger = get_logger("formalizer_training_remote", level="DEBUG")

# --- Constants ---
DATASET_PATH = Path("data/corpus/final_training_corpus.txt")
TOKENIZER_PATH = "data/vocab/fast_tokenizer.json"
TRAIN_PATH = Path("data/corpus/train.txt")
VAL_PATH = Path("data/corpus/val.txt")

# --- Dataset Partitioning ---
# We perform this ONCE before training starts
def partition_dataset(source_path, train_path, val_path, split_ratio=0.9):
    if train_path.exists() and val_path.exists():
        logger.info(f"Datasets already partitioned found at {train_path} and {val_path}. Skipping split.")
        return

    logger.info(f"Reading {source_path} for partitioning...")
    with open(source_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    # Simple character-based split (for a continuous corpus)
    n = len(data)
    train_data = data[:int(n*split_ratio)]
    val_data = data[int(n*split_ratio):]
    
    logger.info(f"Writing train set ({len(train_data)/1e6:.2f}M chars) to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_data)
        
    logger.info(f"Writing val set ({len(val_data)/1e6:.2f}M chars) to {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_data)

partition_dataset(DATASET_PATH, TRAIN_PATH, VAL_PATH, split_ratio=0.90)

# --- Load Tokenizer ---
logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = HFTokenizer.from_file(TOKENIZER_PATH)
pad_id = tokenizer.token_to_id("<|pad|>")
tokenizer.enable_padding(
        pad_id=pad_id,
        pad_token="<|pad|>",
)
vocab_size = tokenizer.get_vocab_size() 
logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")

# --- Configuration: Nano Formalizer (~33M Params) ---

training_config_dict = {
    # Data Params
    "vocab_size": vocab_size,      # 12257
    "context_size": 1024,
    
    # Training Duration
    # Since model is small/data-rich, we can train for 2 epochs safely.
    # 6700 steps = 1 epoch. Let's do 10,000 steps (~1.5 epochs).
    "max_iters": 10000,             
    "eval_interval": 500,
    "eval_iters": 50,
    "checkpoint_interval": 2000,
    
    # Optimization
    "batch_size": 64,             
    "learning_rate": 1e-3,         # Smaller models can handle higher Learning Rates (1e-3 is standard for Nano)
    
    # Model Architecture: "Nano"
    # ~33M Parameters
    "n_embeddings": 512,  
    "n_heads": 8,                  # 64 dim per head
    "n_layer": 8,                  # 8 Layers, Attention is all you need uses 6
    
    "device": "cuda"
}

config = BaseTrainerConfig(**training_config_dict)
logger.info(f"Configuration loaded: {config.model_dump()}")

# --- Defining Eval and Training Dataset ---
logger.info("Initializing Datasets...")
train_dataset = FormalizerDataset(corpus_path=TRAIN_PATH, tokenizer=tokenizer, context_size=config.context_size)
val_dataset = FormalizerDataset(corpus_path=VAL_PATH, tokenizer=tokenizer, context_size=config.context_size)

# --- Defining Model ---
logger.info("Initializing Transformer Model...")
# Ensure your Transformer class accepts the config object or specific args
formalizer_model = Transformer(
    vocab_size=config.vocab_size,
    emb_dim=config.n_embeddings,  
    context_size=config.context_size,
    n_layers=config.n_layer,
    n_heads=config.n_heads,
)

# --- Start Training ---
logger.info("Starting Trainer...")
trainer = FormalizerTrainer(
    model=formalizer_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)

trainer.train()