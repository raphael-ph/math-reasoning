import time
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# internal imports
from ..utils.logger import get_logger

_logger = get_logger("hf_tokenizer", level="DEBUG")

VOCAB_SIZE = 32000 # based on llama 2: 
DIRECTORY_PATH = Path("./data/pretraining")
MODEL_SAVE_PATH = Path("./data/vocab/fast_tokenizer.json")
MODEL_SAVE_PATH.parent.mkdir(exist_ok=True, parents=True) # create if does not exist

FILE_EXT = ".parquet"

def train_tokenizer():
    _logger.info(f"--- Starting HF-Tokenizer training (Target Vocab: {VOCAB_SIZE}) ---")
    start_time = time.time()

    # Initialize a BPE Tokenizer
    # We use ByteLevel BPE, which is the standard for GPT-2/3/4 models.
    # It handles all unicode characters by falling back to bytes.
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenization (How to split the text before BPE)
    # This uses the standard GPT-2 byte-level pre-tokenizer.
    # It handles the "space" logic automatically.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoding (How to go back to text)
    tokenizer.decoder = decoders.ByteLevel()

    # Define the Trainer
    # special_tokens matches what you had. 
    # The library handles the merge math in Rust (C++ speed).
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", 
                        "<|fim_prefix|>", 
                        "<|fim_middle|>", 
                        "<|fim_suffix|>", 
                        "<|pad|>", 
                        "<|bos|>",
                        # User and assistant will be used in post-training, in order to teach the model
                        # the conversation flow.
                        "<|user|>",
                        "<|assistant|>",
                        ], 
        show_progress=True,
        min_frequency=2,
    )

    _logger.info(f"Reading {DIRECTORY_PATH} and training...")
    tokenizer.train_from_iterator(_iter_all_documents(DIRECTORY_PATH), trainer)


    end_time = time.time()
    elapsed = end_time - start_time

    # Save
    tokenizer.save(str(MODEL_SAVE_PATH))

    _logger.info("\n" + "="*40)
    _logger.info(f"DONE!")
    _logger.info(f"Total Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    _logger.info(f"Vocab Size: {tokenizer.get_vocab_size()}")
    _logger.info(f"Saved to:   {MODEL_SAVE_PATH}")  
    _logger.info("="*40)

# --- Helper Functions ---
def _iter_all_documents(directory: Path):
    for item in directory.iterdir():
            if item.is_dir():
                for shard in item.glob(f"*{FILE_EXT}"):
                    pf = pq.ParquetFile(shard)
                    for batch in pf.iter_batches(columns=["text"]):
                        yield from batch.column("text").to_pylist()

if __name__ == "__main__":
    train_tokenizer()